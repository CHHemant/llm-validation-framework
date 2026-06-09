from src.generation.consistency import evaluate_consistency
from src.generation.explainability import sentence_level_attribution
from src.pipeline.guardrails import evaluate_guardrails
from src.pipeline.rag_validator import ValidationResult
from src.retrieval.coverage import analyze_query_coverage
from src.retrieval.validator import rank_chunks_by_relevance


def test_retrieval_ranking_and_coverage_are_reasonable():
    query = "python functions"
    chunks = [
        "SQL indexing improves query performance.",
        "Python functions support reusable logic blocks.",
    ]

    ranked = rank_chunks_by_relevance(query, chunks)
    coverage = analyze_query_coverage(query, chunks)

    assert ranked[0][2] == chunks[1]
    assert coverage.score > 0.5


def test_explainability_and_consistency_reports_work():
    answer = "Python functions support reusable logic. Indentation is syntax-significant."
    chunks = ["Python uses indentation. Functions support reusable logic."]

    attribution = sentence_level_attribution(answer, chunks)
    consistency = evaluate_consistency([answer, answer])

    assert len(attribution) == 2
    assert consistency.is_consistent is True


def test_guardrail_decision_blocks_low_faithfulness():
    result = ValidationResult(
        passed=False,
        faithfulness=0.2,
        consistency=0.8,
        attribution=0.7,
        overall_score=0.5,
        issues=["low faithfulness"],
    )

    decision = evaluate_guardrails(result)

    assert decision.allow_response is False
    assert decision.severity == "high"
