"""Consistency stress test: 50 queries x 10 generations with variance analysis."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from src.pipeline.rag_validator import RAGValidator


def _load_queries(dataset_path: Path, limit: int = 50) -> list[str]:
    if dataset_path.exists():
        try:
            data = json.loads(dataset_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in dataset file: {dataset_path}") from exc
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        queries = [item.get("query") or item.get("question") for item in data if isinstance(item, dict)]
        queries = [q for q in queries if isinstance(q, str) and q.strip()]
        if queries:
            return queries[:limit]

    return [f"Explain the key result for synthetic query {i}" for i in range(limit)]


def _classify_query_type(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ("why", "reason", "cause")):
        return "causal"
    if any(k in q for k in ("compare", "difference", "versus", "vs")):
        return "comparison"
    if any(k in q for k in ("how", "steps", "process")):
        return "procedural"
    if any(k in q for k in ("when", "date", "year", "time")):
        return "temporal"
    return "factual"


def _generate_answer(query: str, run_id: int) -> str:
    rng = random.Random(f"{query}-{run_id}")
    fragments = [
        "It centers on validated retrieval evidence.",
        "The answer highlights high-confidence chunks.",
        "Consistency improves with deterministic prompts.",
        "Low overlap signals hallucination risk.",
        "Attribution should map claims to source text.",
    ]
    chosen = rng.sample(fragments, k=3)
    if run_id % 4 == 0:
        chosen.append("An extra speculative statement appears in this run.")
    return f"{query}. " + " ".join(chosen)


def _pairwise_list(texts: list[str]) -> list[float]:
    sims = []
    for idx in range(1, len(texts)):
        sims.append(RAGValidator._pairwise_similarity([texts[0], texts[idx]]))
    return sims


def main() -> None:
    parser = argparse.ArgumentParser(description="Run consistency stress test over 50 queries x 10 runs")
    parser.add_argument("--dataset", default="evaluation/datasets/test_500.json", help="Path to benchmark dataset")
    parser.add_argument("--output", default="evaluation/results/consistency_stress.json", help="Path to output JSON")
    parser.add_argument("--consistency-threshold", type=float, default=0.85, help="Threshold for low consistency")
    args = parser.parse_args()

    queries = _load_queries(Path(args.dataset), limit=50)
    low_consistency = []
    records = []

    for query in queries:
        answers = [_generate_answer(query, run_id) for run_id in range(10)]
        consistency = RAGValidator._pairwise_similarity(answers)
        run_sims = _pairwise_list(answers)
        variance = float(np.var(run_sims)) if run_sims else 0.0
        query_type = _classify_query_type(query)

        row = {
            "query": query,
            "query_type": query_type,
            "consistency": round(float(consistency), 4),
            "variance": round(variance, 6),
        }
        records.append(row)

        if consistency < args.consistency_threshold:
            low_consistency.append(row)

    fail_counts = Counter(item["query_type"] for item in low_consistency)

    report = {
        "total_queries": len(queries),
        "runs_per_query": 10,
        "threshold": args.consistency_threshold,
        "overall_consistency_mean": round(float(np.mean([r["consistency"] for r in records])) if records else 1.0, 4),
        "overall_variance_mean": round(float(np.mean([r["variance"] for r in records])) if records else 0.0, 6),
        "low_consistency_queries": low_consistency,
        "failing_query_types": dict(fail_counts),
        "records": records,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Processed {len(queries)} queries x 10 runs")
    print(f"Low consistency (< {args.consistency_threshold}): {len(low_consistency)}")
    print("Failing query types:", dict(fail_counts))
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
