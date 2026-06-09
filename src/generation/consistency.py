"""Multi-run consistency testing."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from src.pipeline.rag_validator import RAGValidator


@dataclass(slots=True)
class ConsistencyReport:
    score: float
    pairwise_scores: list[float]
    is_consistent: bool


def evaluate_consistency(answers: list[str], threshold: float = 0.85) -> ConsistencyReport:
    """Computes consistency using pairwise similarity from the validator utility."""
    if not answers:
        return ConsistencyReport(score=1.0, pairwise_scores=[], is_consistent=True)

    pairwise = []
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            pairwise.append(RAGValidator._pairwise_similarity([answers[i], answers[j]]))

    score = mean(pairwise) if pairwise else 1.0
    return ConsistencyReport(
        score=round(float(score), 4),
        pairwise_scores=[round(float(x), 4) for x in pairwise],
        is_consistent=score >= threshold,
    )
