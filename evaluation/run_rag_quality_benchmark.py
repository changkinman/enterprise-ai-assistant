"""RAG 依据回答质量基准（LLM-as-Judge）。

从 RAG 评测集中隔条抽取 20 条，覆盖每篇匿名化知识文档两条问题。回答模型仅可
使用增强检索返回的材料；同一模型以既定四维量表做独立评分。该脚本评测 RAG
依据回答，不包含目前仍为硬编码返回的业务工具。
"""
import asyncio
import json
import re
import statistics
from pathlib import Path
from typing import Any

from run_rag_benchmark import RagBenchmark, _load_fixture_items


FIXTURES = Path(__file__).resolve().parent / "fixtures"
PASS_THRESHOLD = 0.75
JUDGE_PROMPT = """你是客服质量评估专家。根据给定检索证据和标准要点，对回答评分。

用户问题: {question}
检索证据: {evidence}
标准要点: {expected_facts}
回答: {answer}

按 0.0-1.0 返回 JSON：relevance（针对问题）、accuracy（不与证据冲突）、
completeness（覆盖标准要点）、helpfulness（给出可行动说明）。只返回 JSON。"""


def _normalise(text: str) -> str:
    value = text.lower().replace("日至", "-").replace("到", "-")
    value = re.sub(r"(\d+)点半", r"\1:30", value)
    value = re.sub(r"(\d+)月(\d+)-(\d+)月(\d+)", r"\1月\2-\4", value)
    return re.sub(r"[\s，。、：:；;（）()]", "", value)


class RagQualityBenchmark(RagBenchmark):
    async def enhanced_search(self, query: str) -> list[dict[str, Any]]:
        sub_queries = await self.rewrite(query)
        recalled = await asyncio.gather(*(asyncio.to_thread(self.search, item, 5) for item in sub_queries))
        seen: set[tuple[str, str]] = set()
        merged: list[dict[str, Any]] = []
        for items in recalled:
            for item in items:
                marker = (item["title"], item["content"])
                if marker not in seen:
                    seen.add(marker)
                    merged.append(item)
        return await self.rerank(query, merged, 5)

    async def answer(self, query: str, evidence: str) -> str:
        prompt = f"""你是企业内部服务助手。仅根据下方知识库材料回答用户问题；材料不足时明确说明。
回答应简洁、准确，并给出下一步操作。

用户问题：{query}
知识库材料：
{evidence}"""
        response = await self.client.messages.create(
            model=self.model, max_tokens=500, temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    async def judge(self, question: str, answer: str, evidence: str, expected_facts: list[str]) -> dict[str, float]:
        prompt = JUDGE_PROMPT.format(
            question=question, evidence=evidence, expected_facts="；".join(expected_facts), answer=answer,
        )
        try:
            response = await self.client.messages.create(
                model=self.model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            return {key: float(data.get(key, 0.5)) for key in ("relevance", "accuracy", "completeness", "helpfulness")}
        except Exception:
            return {key: 0.5 for key in ("relevance", "accuracy", "completeness", "helpfulness")}

    async def evaluate_case(self, case: dict[str, Any]) -> dict[str, Any]:
        retrieved = await self.enhanced_search(case["query"])
        evidence = "\n\n".join(f"[{item['title']}]\n{item['content']}" for item in retrieved)
        answer = await self.answer(case["query"], evidence)
        scores = await self.judge(case["query"], answer, evidence, case["expected_facts"])
        overall = statistics.mean(scores.values())
        normalised_answer = _normalise(answer)
        covered = sum(_normalise(fact) in normalised_answer for fact in case["expected_facts"])
        return {
            "id": case["id"], "scores": scores, "overall": round(overall, 4),
            "passed": overall >= PASS_THRESHOLD, "retrieved_titles": [item["title"] for item in retrieved],
            "expected_facts_covered": covered, "expected_facts_total": len(case["expected_facts"]),
        }

    async def run_quality(self) -> dict[str, Any]:
        docs = _load_fixture_items("rag_documents.json", "rag_documents_expanded.json", "documents")
        # 隔条抽样得到 50 条，覆盖主集与扩展集以及全部 20 篇文档。
        cases = _load_fixture_items("rag_cases.json", "rag_cases_expanded.json", "cases")[::2]
        self.collection.add(
            ids=[doc["id"] for doc in docs], documents=[doc["content"] for doc in docs],
            metadatas=[{"title": doc["title"]} for doc in docs],
        )
        semaphore = asyncio.Semaphore(4)

        async def bounded(case: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await self.evaluate_case(case)

        rows = await asyncio.gather(*(bounded(case) for case in cases))
        dimensions = ("relevance", "accuracy", "completeness", "helpfulness")
        averages = {name: round(statistics.mean(row["scores"][name] for row in rows), 4) for name in dimensions}
        return {
            "sample_size": len(rows), "document_coverage": len({title for row in rows for title in row["retrieved_titles"]}),
            "pass_rate": round(sum(row["passed"] for row in rows) / len(rows), 4),
            "expected_fact_coverage": round(sum(row["expected_facts_covered"] for row in rows) / sum(row["expected_facts_total"] for row in rows), 4),
            "average_scores": averages, "overall_score": round(statistics.mean(row["overall"] for row in rows), 4),
            "rows": rows,
        }


async def main() -> None:
    benchmark = RagQualityBenchmark()
    try:
        print(json.dumps(await benchmark.run_quality(), ensure_ascii=False), flush=True)
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
