"""Validation wrapper orchestration."""

from __future__ import annotations

from dataclasses import asdict

from src.generation.explainability import sentence_level_attribution
from src.generation.consistency import evaluate_consistency
from src.pipeline.rag_validator import RAGValidator
from src.retrieval.coverage import analyze_query_coverage
from src.retrieval.validator import rank_chunks_by_relevance


async def run_validation_suite(
    query: str,
    answer: str,
    retrieved_chunks: list[str],
    regenerate_fn=None,
    llm_client=None,
) -> dict:
    """Runs retrieval and generation validators and returns structured report."""
    validator = RAGValidator(llm_client=llm_client)
    base = await validator.validate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        answer=answer,
        regenerate_fn=regenerate_fn,
    )

    retrieval_ranking = rank_chunks_by_relevance(query, retrieved_chunks)
    coverage = analyze_query_coverage(query, retrieved_chunks)
    attribution_rows = sentence_level_attribution(answer, retrieved_chunks)
    consistency_detail = evaluate_consistency([answer])

    return {
        "validation": asdict(base),
        "retrieval": {
            "ranking": [
                {"chunk_index": idx, "score": round(float(score), 4), "chunk": chunk}
                for idx, score, chunk in retrieval_ranking
            ],
            "coverage": asdict(coverage),
        },
        "generation": {
            "attribution": [asdict(row) for row in attribution_rows],
            "consistency": asdict(consistency_detail),
        },
    }
