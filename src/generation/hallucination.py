"""Faithfulness scoring utilities with NLI-based entailment."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any


DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
ALLOWED_NLI_MODELS = {DEFAULT_NLI_MODEL}
MAX_SENTENCES = 12  # caps per-answer sentence scoring for stable latency
MAX_CHUNKS = 12  # caps chunk comparisons to bound model calls
MAX_CHUNK_CHARS = 1200  # keeps chunk inputs within reasonable token budget


class NLIFaithfulnessScorer:
    """Sentence-to-chunk entailment scoring using a HuggingFace NLI cross-encoder."""

    def __init__(self, model_name: str | None = None):
        env_model = model_name or os.getenv("NLI_MODEL", DEFAULT_NLI_MODEL)
        self.model_name = env_model if env_model in ALLOWED_NLI_MODELS else DEFAULT_NLI_MODEL
        self._pipeline = None
        self._pipeline_failed = False

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if self._pipeline_failed:
            raise RuntimeError("NLI pipeline unavailable")
        from transformers import pipeline

        try:
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                truncation=True,
                return_all_scores=True,
            )
            return self._pipeline
        except Exception:
            self._pipeline_failed = True
            raise

    @staticmethod
    def _extract_entailment_score(raw_output: Any) -> float:
        if not raw_output:
            return 0.0

        rows = raw_output[0] if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], list) else raw_output
        if isinstance(rows, dict):
            rows = [rows]

        if not isinstance(rows, list):
            return 0.0

        label_to_score = {
            str(item.get("label", "")).lower(): float(item.get("score", 0.0))
            for item in rows
            if isinstance(item, dict)
        }

        for key, value in label_to_score.items():
            if "entail" in key:
                return value

        if "label_2" in label_to_score:
            return label_to_score["label_2"]

        return max(label_to_score.values(), default=0.0)

    def entailment_probability(self, answer_sentence: str, chunk: str) -> float:
        pipe = self._load_pipeline()
        output = pipe([{"text": answer_sentence, "text_pair": chunk}])
        return max(0.0, min(1.0, self._extract_entailment_score(output)))

    def score(self, answer: str, context: str) -> float:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        chunks = [c.strip() for c in re.split(r"\n{2,}", context) if c.strip()]

        if not sentences:
            return 1.0
        if not chunks:
            return 0.0

        limited_chunks = chunks[:MAX_CHUNKS]
        sentence_scores = []
        for sentence in sentences[:MAX_SENTENCES]:
            best = 0.0
            for chunk in limited_chunks:
                best = max(best, self.entailment_probability(sentence, chunk[:MAX_CHUNK_CHARS]))
            sentence_scores.append(best)

        return sum(sentence_scores) / len(sentence_scores)


@lru_cache(maxsize=1)
def get_nli_scorer() -> NLIFaithfulnessScorer:
    return NLIFaithfulnessScorer()


def lexical_faithfulness(answer: str, context: str) -> float:
    answer_words = set(re.findall(r"\b\w{4,}\b", answer.lower()))
    context_words = set(re.findall(r"\b\w{4,}\b", context.lower()))
    if not answer_words:
        return 1.0
    overlap = answer_words & context_words
    return len(overlap) / len(answer_words)


def score_faithfulness(answer: str, context: str) -> float:
    """Returns NLI faithfulness, with lexical fallback if model/deps are unavailable."""
    try:
        return get_nli_scorer().score(answer, context)
    except Exception:
        return lexical_faithfulness(answer, context)
