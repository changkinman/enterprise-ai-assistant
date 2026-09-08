"""意图识别离线基准：合并主集与人工扩展集，避免实体提取影响分类时延。"""
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "evaluation" / "fixtures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.intent_recognizer import IntentCategory, IntentRecognizer


def percentile(values: list[float], quantile: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower, upper = int(position), min(int(position) + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def load_cases() -> list[dict[str, Any]]:
    cases = json.loads((FIXTURES / "intent_cases.json").read_text(encoding="utf-8"))["cases"]
    expansion_path = FIXTURES / "intent_cases_expanded.json"
    if expansion_path.exists():
        cases += json.loads(expansion_path.read_text(encoding="utf-8"))["cases"]
    return cases


async def classify_only(recognizer: IntentRecognizer, message: str) -> tuple[str, float]:
    """复用正式三路投票，不调用实体提取，避免将另一项能力混入分类延迟。"""
    started = time.monotonic()
    llm_task = asyncio.create_task(recognizer._llm_recognize(message, None))
    emb_task = asyncio.create_task(recognizer._embedding_recognize(message)) if recognizer._embedding_enabled else None
    pattern = recognizer._pattern_recognize(message)
    if emb_task:
        llm, embedding = await asyncio.gather(llm_task, emb_task)
    else:
        llm, embedding = await llm_task, {"intent": IntentCategory.OTHER, "confidence": 0.0}
    return recognizer._vote(llm, embedding, pattern).value, (time.monotonic() - started) * 1000


async def main() -> None:
    load_dotenv(ROOT / ".env")
    recognizer = IntentRecognizer(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
    )
    cases = load_cases()
    semaphore = asyncio.Semaphore(20)

    async def run_case(case: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            predicted, latency_ms = await classify_only(recognizer, case["message"])
            return {"id": case["id"], "expected": case["expected_intent"], "predicted": predicted, "latency_ms": latency_ms}

    rows = await asyncio.gather(*(run_case(case) for case in cases))
    labels = sorted({row["expected"] for row in rows} | {row["predicted"] for row in rows})
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(row["expected"] == label and row["predicted"] == label for row in rows)
        fp = sum(row["expected"] != label and row["predicted"] == label for row in rows)
        fn = sum(row["expected"] == label and row["predicted"] != label for row in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0, "support": sum(row["expected"] == label for row in rows)}
    correct = sum(row["expected"] == row["predicted"] for row in rows)
    report = {
        "total": len(rows), "correct": correct, "accuracy": round(correct / len(rows), 4),
        "macro_f1": round(statistics.mean(value["f1"] for value in per_class.values()), 4),
        "p50_latency_ms": round(percentile([row["latency_ms"] for row in rows], 0.5), 2),
        "p95_latency_ms": round(percentile([row["latency_ms"] for row in rows], 0.95), 2),
        "distribution": dict(Counter(row["expected"] for row in rows)), "per_class": per_class,
        "errors": [row for row in rows if row["expected"] != row["predicted"]],
    }
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
