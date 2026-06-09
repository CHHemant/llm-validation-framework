"""Query coverage analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass


_TOKEN_PATTERN = re.compile(r"\b\w{4,}\b")


@dataclass(slots=True)
class CoverageResult:
    score: float
    missing_terms: list[str]
    covered_terms: list[str]


def _extract_terms(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


def analyze_query_coverage(query: str, chunks: list[str]) -> CoverageResult:
    """Analyzes how well retrieved chunks cover key query terms."""
    query_terms = _extract_terms(query)
    if not query_terms:
        return CoverageResult(score=1.0, missing_terms=[], covered_terms=[])

    context_terms = _extract_terms("\n".join(chunks))
    covered = sorted(query_terms & context_terms)
    missing = sorted(query_terms - context_terms)

    score = len(covered) / len(query_terms)
    return CoverageResult(score=round(score, 4), missing_terms=missing, covered_terms=covered)
