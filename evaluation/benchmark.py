"""Run evaluation benchmark suite."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean

from src.pipeline.rag_validator import RAGValidator


def _load_dataset(path: Path) -> list[dict]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid dataset JSON: {path}") from exc

        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        if isinstance(data, list):
            return data
        raise ValueError("Dataset must be a JSON list or a dict with `data` list")

    synthetic = []
    for idx in range(100):
        question = f"What does synthetic benchmark item {idx} show?"
        context = f"Synthetic benchmark item {idx} shows validated evidence {idx % 9}."
        answer = f"Synthetic benchmark item {idx} shows validated evidence {idx % 9}."
        synthetic.append({"query": question, "contexts": [context], "answer": answer})
    return synthetic


async def _run(samples: list[dict]) -> dict:
    validator = RAGValidator(llm_client=None)

    results = []
    for item in samples:
        query = item.get("query") or item.get("question") or ""
        chunks = item.get("contexts") or [item.get("context", "")]
        answer = item.get("answer") or ""

        result = await validator.validate(query=query, retrieved_chunks=chunks, answer=answer)
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round((passed / len(results)) if results else 0.0, 4),
        "avg_faithfulness": round(mean(r.faithfulness for r in results), 4) if results else 0.0,
        "avg_consistency": round(mean(r.consistency for r in results), 4) if results else 0.0,
        "avg_attribution": round(mean(r.attribution for r in results), 4) if results else 0.0,
        "avg_overall_score": round(mean(r.overall_score for r in results), 4) if results else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark suite for validator quality")
    parser.add_argument("--dataset", default="evaluation/datasets/test_500.json", help="Path to dataset JSON")
    parser.add_argument("--output", default="evaluation/results/benchmark_report.json", help="Output report JSON path")
    args = parser.parse_args()

    dataset = _load_dataset(Path(args.dataset))
    report = asyncio.run(_run(dataset))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
