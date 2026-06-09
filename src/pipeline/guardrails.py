"""Hard-stop guardrail conditions."""

from __future__ import annotations

from dataclasses import dataclass

from src.pipeline.rag_validator import ValidationResult


@dataclass(slots=True)
class GuardrailDecision:
    allow_response: bool
    reason: str
    severity: str


def evaluate_guardrails(result: ValidationResult) -> GuardrailDecision:
    """Converts validation outputs into a production decision."""
    if result.passed:
        return GuardrailDecision(allow_response=True, reason="All checks passed", severity="none")

    if result.faithfulness < 0.50 or result.attribution < 0.40:
        return GuardrailDecision(
            allow_response=False,
            reason="High hallucination risk detected",
            severity="high",
        )

    if result.consistency < 0.70:
        return GuardrailDecision(
            allow_response=False,
            reason="Answer is unstable across repeated generations",
            severity="medium",
        )

    return GuardrailDecision(
        allow_response=False,
        reason="Validation thresholds not met",
        severity="medium",
    )
