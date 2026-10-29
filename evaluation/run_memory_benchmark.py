"""三级记忆基准：工作记忆、跨会话情景记忆、摘要压缩、用户隔离与画像。

旧版脚本直接使用 Chroma collection。运行时已迁移到 PostgreSQL/pgvector，
因此本入口被保留为明确的迁移提示：基准需要隔离的 PostgreSQL 和 Redis，
不会由脚本隐式创建或清理。
"""
import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any


from memory.conversation_memory import MemoryManager, MsgRole


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "evaluation" / "fixtures"


def _normalise_fact(text: str) -> str:
    """统一常见日期、时间与标点表达，避免将等价表述判为遗漏。"""
    value = text.lower().replace("日至", "-").replace("到", "-")
    value = re.sub(r"(\d+)点半", r"\1:30", value)
    value = re.sub(r"(\d+)月(\d+)-(\d+)月(\d+)", r"\1月\2-\4", value)
    return re.sub(r"[\s，。、：:；;（）()]", "", value)


def _facts_found(text: str, facts: list[str]) -> tuple[int, int]:
    normalised_text = _normalise_fact(text)
    return sum(_normalise_fact(fact) in normalised_text for fact in facts), len(facts)


def _load_scenarios() -> list[dict[str, Any]]:
    scenarios = json.loads((FIXTURES / "memory_scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    expansion_path = FIXTURES / "memory_scenarios_expanded.json"
    if expansion_path.exists():
        scenarios += json.loads(expansion_path.read_text(encoding="utf-8"))["scenarios"]
    return scenarios


class MemoryBenchmark:
    def __init__(self) -> None:
        self.run_id = uuid.uuid4().hex[:12]
        self.user_prefix = f"eval-memory-{self.run_id}"
        self.redis_url = os.environ["REDIS_URL"].rsplit("/", 1)[0] + "/15"
        self.chroma = chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", "8000")),
        )
        self.episodic_name = f"eval_episodic_{self.run_id}"
        self.profile_name = f"eval_profile_{self.run_id}"
        self.manager = MemoryManager(
            redis_url=self.redis_url,
            chroma_host=os.getenv("CHROMA_HOST", "localhost"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        )
        # 避免写入项目默认 episodic/user_profile collection。
        self.manager._episodic = self.chroma.get_or_create_collection(self.episodic_name)
        self.manager._profile = self.chroma.get_or_create_collection(self.profile_name)
        self.keys: set[str] = set()

    def user(self, scenario_id: str, suffix: str = "user") -> str:
        return f"{self.user_prefix}-{scenario_id}-{suffix}"

    async def add_turns(self, user_id: str, conv_id: str, turns: list[str]) -> None:
        self.keys.add(self.manager._wm_key(user_id, conv_id))
        self.keys.add(self.manager._summary_key(user_id, conv_id))
        for turn in turns:
            await self.manager.add_message(user_id, conv_id, MsgRole.USER, turn)

    def nearest_episodic_distance(self, user_id: str, query: str) -> float | None:
        result = self.manager._episodic.query(
            query_texts=[query], n_results=1, where={"user_id": user_id}, include=["distances"],
        )
        distances = result.get("distances") or [[]]
        return round(float(distances[0][0]), 4) if distances and distances[0] else None

    async def evaluate(self) -> dict[str, Any]:
        scenarios = _load_scenarios()
        totals: dict[str, list[int]] = {
            "working": [0, 0], "episodic": [0, 0], "summary": [0, 0], "profile": [0, 0],
        }
        compressions: list[float] = []
        isolation_passed = 0
        isolation_total = 0
        negative_passed = 0
        negative_total = 0
        details: list[dict[str, Any]] = []

        for scenario in scenarios:
            scenario_id = scenario["id"]
            kind = scenario["type"]
            if kind == "working_memory":
                user_id, conv_id = self.user(scenario_id), "current"
                started = time.monotonic()
                await self.add_turns(user_id, conv_id, scenario["turns"])
                context = await self.manager.get_context(user_id, conv_id, scenario["final_query"])
                latency_ms = (time.monotonic() - started) * 1000
                found, total = _facts_found(context.to_prompt_text(), scenario["must_recall"])
                totals["working"][0] += found; totals["working"][1] += total
                details.append({"id": scenario_id, "type": kind, "found": found, "total": total, "latency_ms": round(latency_ms, 2)})

            elif kind == "episodic_memory":
                user_id = self.user(scenario_id)
                transcript = "\n".join(scenario["session_1"])
                # 此处直接写入一条已归档历史，单独验证跨会话语义检索能力。
                await self.manager._store_episodic(user_id, "session-1", transcript, transcript)
                started = time.monotonic()
                context = await self.manager.get_context(user_id, "session-2", scenario["session_2_query"])
                latency_ms = (time.monotonic() - started) * 1000
                found, total = _facts_found(context.to_prompt_text(), scenario["must_recall"])
                totals["episodic"][0] += found; totals["episodic"][1] += total
                details.append({"id": scenario_id, "type": kind, "found": found, "total": total, "latency_ms": round(latency_ms, 2), "nearest_distance": self.nearest_episodic_distance(user_id, scenario["session_2_query"])})

            elif kind == "summary_compression":
                user_id, conv_id = self.user(scenario_id), "long-session"
                raw_chars = len("\n".join(scenario["turns"]))
                started = time.monotonic()
                await self.add_turns(user_id, conv_id, scenario["turns"])
                context = await self.manager.get_context(user_id, conv_id, scenario["final_query"])
                latency_ms = (time.monotonic() - started) * 1000
                found, total = _facts_found(context.to_prompt_text(), scenario["must_recall"])
                totals["summary"][0] += found; totals["summary"][1] += total
                retained_chars = len(context.summary) + sum(len(message.content) for message in context.recent_messages)
                compressions.append(1 - retained_chars / raw_chars)
                details.append({"id": scenario_id, "type": kind, "found": found, "total": total, "latency_ms": round(latency_ms, 2), "compression_ratio": round(compressions[-1], 4), "summary": context.summary})

            elif kind == "user_isolation":
                isolation_total += 1
                user_a, user_b = self.user(scenario_id, "a"), self.user(scenario_id, "b")
                await self.manager._store_episodic(user_a, "old", "\n".join(scenario["user_a"]), "\n".join(scenario["user_a"]))
                await self.manager._store_episodic(user_b, "old", "\n".join(scenario["user_b"]), "\n".join(scenario["user_b"]))
                context = await self.manager.get_context(user_b, "new", scenario["final_query"])
                text = context.to_prompt_text()
                found, total = _facts_found(text, scenario["must_recall"])
                leaked, _ = _facts_found(text, scenario["must_not_recall"])
                passed = found == total and leaked == 0
                isolation_passed += passed
                details.append({"id": scenario_id, "type": kind, "passed": passed, "found": found, "total": total, "leaked": leaked})

            elif kind == "user_profile":
                user_id, conv_id = self.user(scenario_id), "profile"
                await self.add_turns(user_id, conv_id, scenario["turns"])
                started = time.monotonic()
                await self.manager.update_profile(user_id, conv_id)
                context = await self.manager.get_context(user_id, "next", scenario["final_query"])
                latency_ms = (time.monotonic() - started) * 1000
                found, total = _facts_found(context.to_prompt_text(), scenario["must_recall"])
                totals["profile"][0] += found; totals["profile"][1] += total
                details.append({"id": scenario_id, "type": kind, "found": found, "total": total, "latency_ms": round(latency_ms, 2), "profile": context.user_profile})

            elif kind == "negative_retrieval":
                negative_total += 1
                user_id = self.user(scenario_id)
                transcript = "\n".join(scenario["history"])
                await self.manager._store_episodic(user_id, "old", transcript, transcript)
                context = await self.manager.get_context(user_id, "new", scenario["final_query"])
                leaked, _ = _facts_found(context.to_prompt_text(), scenario["must_not_recall"])
                passed = leaked == 0
                negative_passed += passed
                details.append({"id": scenario_id, "type": kind, "passed": passed, "leaked": leaked, "nearest_distance": self.nearest_episodic_distance(user_id, scenario["final_query"])})

        def rate(key: str) -> float:
            found, total = totals[key]
            return round(found / total, 4) if total else 0.0

        return {
            "scenario_count": len(scenarios),
            "working_fact_recall": rate("working"),
            "episodic_fact_recall": rate("episodic"),
            "summary_fact_recall": rate("summary"),
            "summary_avg_compression_ratio": round(sum(compressions) / len(compressions), 4),
            "profile_fact_recall": rate("profile"),
            "user_isolation_pass_rate": round(isolation_passed / isolation_total, 4) if isolation_total else 0.0,
            "negative_retrieval_pass_rate": round(negative_passed / negative_total, 4) if negative_total else 0.0,
            "user_isolation": {"passed": isolation_passed, "total": isolation_total},
            "negative_retrieval": {"passed": negative_passed, "total": negative_total},
            "details": details,
        }

    def cleanup(self) -> None:
        for key in self.keys:
            self.manager._redis.delete(key)
        self.chroma.delete_collection(self.episodic_name)
        self.chroma.delete_collection(self.profile_name)


async def main() -> None:
    raise RuntimeError(
        "Memory benchmark needs an isolated PostgreSQL/pgvector + Redis environment; "
        "the legacy Chroma benchmark is retired."
    )


if __name__ == "__main__":
    asyncio.run(main())
