"""RAG 检索基准：原始向量召回 vs 查询改写、多路召回和 LLM 重排。

该脚本在含 ChromaDB Python 客户端与模型配置的环境中运行，默认读取
evaluation/fixtures/rag_documents.json 与 rag_cases.json。每次运行都使用独立的
临时 collection，结束后删除，不会写入项目正式知识库。
"""
import asyncio
import json
import math
import os
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import chromadb
from anthropic import AsyncAnthropic


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "evaluation" / "fixtures"


def _load_fixture_items(primary_name: str, expansion_name: str, key: str) -> list[dict[str, Any]]:
    """合并主测试集与人工扩展集，扩展集缺失时仍可运行主测试集。"""
    primary = json.loads((FIXTURES / primary_name).read_text(encoding="utf-8"))[key]
    expansion_path = FIXTURES / expansion_name
    if not expansion_path.exists():
        return primary
    expansion = json.loads(expansion_path.read_text(encoding="utf-8"))[key]
    return primary + expansion


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * quantile
    low, high = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


def _metrics(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    def rank(row: dict[str, Any]) -> int | None:
        relevant = set(row["expected_titles"])
        for index, title in enumerate(row[key], start=1):
            if title in relevant:
                return index
        return None

    ranks = [rank(row) for row in rows]
    return {
        "recall_at_1": round(sum(value is not None and value <= 1 for value in ranks) / len(rows), 4),
        "recall_at_3": round(sum(value is not None and value <= 3 for value in ranks) / len(rows), 4),
        "recall_at_5": round(sum(value is not None and value <= 5 for value in ranks) / len(rows), 4),
        "mrr": round(statistics.mean(1 / value if value else 0 for value in ranks), 4),
        # 当前标注每条至少有一个正确文档；单相关文档时 nDCG 等于折损命中分数。
        "ndcg_at_5": round(statistics.mean(1 / math.log2(value + 1) if value and value <= 5 else 0 for value in ranks), 4),
    }


class RagBenchmark:
    def __init__(self) -> None:
        self.client = AsyncAnthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        )
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        self.chroma = chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", "8000")),
        )
        self.collection_name = f"eval_rag_{uuid.uuid4().hex[:12]}"
        self.collection = self.chroma.get_or_create_collection(self.collection_name)

    async def rewrite(self, query: str) -> list[str]:
        prompt = f'''将以下用户查询改写为 3 个不同角度的搜索子查询，用于检索知识库。
要求：每个子查询角度不同，覆盖原始问题的不同方面。
原始查询: "{query}"
返回 JSON 数组，例如: ["子查询1", "子查询2", "子查询3"]'''
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            data = json.loads(raw[raw.find("["):raw.rfind("]") + 1])
            return list(dict.fromkeys([query] + [str(item) for item in data]))
        except Exception:
            return [query]

    async def rerank(self, query: str, items: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        if len(items) <= top_k:
            return items
        text = "\n".join(
            f"{index}. {json.dumps(item, ensure_ascii=False)[:200]}"
            for index, item in enumerate(items)
        )
        prompt = f'''根据用户查询，对以下检索结果按相关性打分（0-10），返回 JSON 数组。
用户查询: "{query}"
检索结果:
{text}

返回格式（按相关性降序排列的索引列表）: [最相关的索引, ..., 最不相关的索引]
只返回 JSON 数组，不要其他文字。'''
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            order = json.loads(raw[raw.find("["):raw.rfind("]") + 1])
            ranked = [items[index] for index in order if isinstance(index, int) and 0 <= index < len(items)]
            return ranked[:top_k] if ranked else items[:top_k]
        except Exception:
            return items[:top_k]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        result = self.collection.query(query_texts=[query], n_results=top_k)
        return [
            {"title": meta["title"], "content": doc, "score": 1.0 - distance}
            for doc, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]

    async def evaluate_case(self, case: dict[str, Any]) -> dict[str, Any]:
        query = case["query"]
        start = time.monotonic()
        baseline = await asyncio.to_thread(self.search, query, 5)
        baseline_ms = (time.monotonic() - start) * 1000

        start = time.monotonic()
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
        enhanced = await self.rerank(query, merged, 5)
        enhanced_ms = (time.monotonic() - start) * 1000
        return {
            "id": case["id"],
            "expected_titles": case["expected_titles"],
            "baseline_titles": [item["title"] for item in baseline],
            "enhanced_titles": [item["title"] for item in enhanced],
            "baseline_latency_ms": baseline_ms,
            "enhanced_latency_ms": enhanced_ms,
            "sub_query_count": len(sub_queries),
        }

    async def run(self) -> dict[str, Any]:
        documents = _load_fixture_items("rag_documents.json", "rag_documents_expanded.json", "documents")
        cases = _load_fixture_items("rag_cases.json", "rag_cases_expanded.json", "cases")
        self.collection.add(
            ids=[document["id"] for document in documents],
            documents=[document["content"] for document in documents],
            metadatas=[{"title": document["title"]} for document in documents],
        )
        semaphore = asyncio.Semaphore(6)

        async def bounded(case: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await self.evaluate_case(case)

        rows = await asyncio.gather(*(bounded(case) for case in cases))
        return {
            "total": len(rows),
            "baseline": _metrics(rows, "baseline_titles"),
            "enhanced": _metrics(rows, "enhanced_titles"),
            "latency_ms": {
                "baseline_p50": round(_percentile([row["baseline_latency_ms"] for row in rows], 0.5), 2),
                "baseline_p95": round(_percentile([row["baseline_latency_ms"] for row in rows], 0.95), 2),
                "enhanced_p50": round(_percentile([row["enhanced_latency_ms"] for row in rows], 0.5), 2),
                "enhanced_p95": round(_percentile([row["enhanced_latency_ms"] for row in rows], 0.95), 2),
            },
            "rows": rows,
        }

    def cleanup(self) -> None:
        self.chroma.delete_collection(self.collection_name)


async def main() -> None:
    benchmark = RagBenchmark()
    try:
        print(json.dumps(await benchmark.run(), ensure_ascii=False), flush=True)
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
