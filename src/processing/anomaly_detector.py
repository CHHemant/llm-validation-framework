"""3-layer anomaly detection cascade: structural → statistical → semantic.

This module is currently staged in this repository for compatibility, but belongs
in the NLP pipeline repository as the long-term home.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean, pstdev


@dataclass(slots=True)
class AnomalyReport:
    structural: bool
    statistical: bool
    semantic: bool
    score: float
    reasons: list[str]


def detect_anomalies(text: str, historical_scores: list[float] | None = None) -> AnomalyReport:
    """Applies simple anomaly heuristics suitable for online validation."""
    reasons: list[str] = []

    structural = len(text.strip()) < 20 or text.count("\n") > 40
    if structural:
        reasons.append("Response shape is unusual")

    statistical = False
    if historical_scores:
        avg = mean(historical_scores)
        sigma = pstdev(historical_scores) if len(historical_scores) > 1 else 0.0
        current = min(1.0, len(text) / 500)
        if sigma > 0 and abs(current - avg) > 2 * sigma:
            statistical = True
            reasons.append("Response length-derived score is an outlier")

    # Two consecutive repeats are enough to flag noisy low-quality generations.
    repeated_tokens = re.findall(r"\b(\w+)\b(?:\s+\1\b){1,}", text.lower())
    semantic = len(repeated_tokens) > 0
    if semantic:
        reasons.append("Detected suspicious repeated-token patterns")

    triggered = sum((structural, statistical, semantic))
    score = triggered / 3

    return AnomalyReport(
        structural=structural,
        statistical=statistical,
        semantic=semantic,
        score=round(score, 4),
        reasons=reasons,
    )
