"""Retrieval chunk relevance scoring."""

from __future__ import annotations

import math
import re
from collections import Counter


_TOKEN_PATTERN = re.compile(r"\b\w{4,}\b")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _cosine_similarity(a_tokens: list[str], b_tokens: list[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0

    a_counts = Counter(a_tokens)
    b_counts = Counter(b_tokens)

    dot = sum(a_counts[token] * b_counts[token] for token in a_counts.keys() & b_counts.keys())
    a_norm = math.sqrt(sum(v * v for v in a_counts.values()))
    b_norm = math.sqrt(sum(v * v for v in b_counts.values()))

    if a_norm == 0 or b_norm == 0:
        return 0.0
    return dot / (a_norm * b_norm)


def score_chunk_relevance(query: str, chunk: str) -> float:
    """Scores lexical relevance between a query and retrieved chunk in [0, 1]."""
    query_tokens = _tokenize(query)
    chunk_tokens = _tokenize(chunk)
    return max(0.0, min(1.0, _cosine_similarity(query_tokens, chunk_tokens)))


def rank_chunks_by_relevance(query: str, chunks: list[str]) -> list[tuple[int, float, str]]:
    """Returns chunk ranking as tuples of (original_index, score, chunk)."""
    scored = [(idx, score_chunk_relevance(query, chunk), chunk) for idx, chunk in enumerate(chunks)]
    return sorted(scored, key=lambda row: row[1], reverse=True)


def filter_relevant_chunks(query: str, chunks: list[str], threshold: float = 0.12) -> list[str]:
    """Filters chunks using relevance threshold while preserving top-ranked order."""
    ranked = rank_chunks_by_relevance(query, chunks)
    return [chunk for _, score, chunk in ranked if score >= threshold]
