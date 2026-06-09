"""Run side-by-side metric comparison for RAGAS and local validator metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean

from src.pipeline.rag_validator import RAGValidator


def _load_test_pairs(dataset_path: Path, limit: int = 500) -> list[dict]:
    if dataset_path.exists():
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if not isinstance(data, list):
            raise ValueError("Dataset should be a JSON list of test pairs")
        return data[:limit]

    generated = []
    for idx in range(limit):
        query = f"What is the key fact for synthetic sample {idx}?"
        context = f"Synthetic sample {idx} confirms fact {idx % 13} with controlled evidence."
        answer = f"The key fact for synthetic sample {idx} is fact {idx % 13}."
        generated.append(
            {
                "question": query,
                "query": query,
                "contexts": [context],
                "answer": answer,
                "ground_truth": answer,
            }
        )
    return generated


async def _run_validator_metrics(samples: list[dict]) -> dict[str, float]:
    validator = RAGValidator(llm_client=None)
    faithfulness_scores = []
    answer_relevancy_scores = []
    context_recall_scores = []

    for item in samples:
        query = item.get("query") or item.get("question") or ""
        contexts = item.get("contexts") or [item.get("context", "")]
        answer = item.get("answer") or ""
        ground_truth = item.get("ground_truth") or answer

        context_text = "\n\n".join(c for c in contexts if c)
        faithfulness = await validator._check_faithfulness(answer, context_text)
        answer_relevancy = validator._pairwise_similarity([query, answer])
        context_recall = validator._lexical_faithfulness(ground_truth, context_text)

        faithfulness_scores.append(faithfulness)
        answer_relevancy_scores.append(answer_relevancy)
        context_recall_scores.append(context_recall)

    return {
        "faithfulness": round(mean(faithfulness_scores), 4),
        "answer_relevancy": round(mean(answer_relevancy_scores), 4),
        "context_recall": round(mean(context_recall_scores), 4),
    }


def _run_ragas_metrics(samples: list[dict]) -> tuple[dict[str, float], str]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_recall, faithfulness

        ds = Dataset.from_dict(
            {
                "question": [item.get("query") or item.get("question") or "" for item in samples],
                "answer": [item.get("answer") or "" for item in samples],
                "contexts": [item.get("contexts") or [item.get("context", "")] for item in samples],
                "ground_truth": [item.get("ground_truth") or item.get("answer") or "" for item in samples],
            }
        )
        result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_recall])

        if isinstance(result, dict):
            score_dict = result
        elif hasattr(result, "to_pandas"):
            score_dict = result.to_pandas().mean().to_dict()
        else:
            raise TypeError("Unsupported RAGAS evaluation result format")

        return (
            {
                "faithfulness": round(float(score_dict["faithfulness"]), 4),
                "answer_relevancy": round(float(score_dict["answer_relevancy"]), 4),
                "context_recall": round(float(score_dict["context_recall"]), 4),
            },
            "ragas",
        )
    except Exception:
        return ({"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0}, "ragas_unavailable")


def _build_side_by_side(ragas_scores: dict[str, float], validator_scores: dict[str, float]) -> list[dict]:
    rows = []
    for metric in ("faithfulness", "answer_relevancy", "context_recall"):
        rows.append(
            {
                "metric": metric,
                "ragas": ragas_scores[metric],
                "validator": validator_scores[metric],
                "delta_validator_minus_ragas": round(validator_scores[metric] - ragas_scores[metric], 4),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RAGAS metrics with validator metrics on 500 test pairs")
    parser.add_argument("--dataset", default="evaluation/datasets/test_500.json", help="Path to 500-pair dataset JSON")
    parser.add_argument("--output", default="evaluation/results/ragas_comparison.json", help="Output JSON report")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    samples = _load_test_pairs(dataset_path, limit=500)

    validator_scores = asyncio.run(_run_validator_metrics(samples))
    ragas_scores, ragas_mode = _run_ragas_metrics(samples)
    table = _build_side_by_side(ragas_scores, validator_scores)

    report = {
        "dataset": str(dataset_path),
        "total_pairs": len(samples),
        "ragas_mode": ragas_mode,
        "rows": table,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("metric | ragas | validator | delta")
    for row in table:
        print(f"{row['metric']} | {row['ragas']:.4f} | {row['validator']:.4f} | {row['delta_validator_minus_ragas']:+.4f}")
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
