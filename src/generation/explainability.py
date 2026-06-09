"""Attribution and evidence flagging."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.pipeline.rag_validator import RAGValidator


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class AttributionRecord:
    sentence: str
    best_chunk_index: int
    score: float
    attributed: bool


def sentence_level_attribution(answer: str, chunks: list[str], threshold: float = 0.35) -> list[AttributionRecord]:
    """Maps each answer sentence to the best matching source chunk."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer) if s.strip()]
    if not sentences:
        return []

    records = []
    for sentence in sentences:
        best_index = -1
        best_score = 0.0
        for idx, chunk in enumerate(chunks):
            score = RAGValidator._lexical_faithfulness(sentence, chunk)
            if score > best_score:
                best_score = score
                best_index = idx

        records.append(
            AttributionRecord(
                sentence=sentence,
                best_chunk_index=best_index,
                score=round(float(best_score), 4),
                attributed=best_score >= threshold,
            )
        )

    return records
