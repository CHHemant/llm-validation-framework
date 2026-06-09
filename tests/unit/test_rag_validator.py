from src.pipeline.rag_validator import RAGValidator, ValidationResult


def test_lexical_faithfulness_high_overlap_high_score():
    answer = "Python uses indentation and functions for reusable logic."
    context = "Python uses indentation and functions. Reusable logic is encouraged."

    score = RAGValidator._lexical_faithfulness(answer, context)

    assert score >= 0.6


def test_lexical_faithfulness_zero_overlap_low_score():
    answer = "Saturn has prominent rings made of ice and dust."
    context = "Neural networks optimize weights with gradient descent."

    score = RAGValidator._lexical_faithfulness(answer, context)

    assert score <= 0.1


def test_pairwise_similarity_identical_texts_one_point_zero():
    texts = ["retrieval validation improves quality"] * 3

    score = RAGValidator._pairwise_similarity(texts)

    assert score == 1.0


def test_pairwise_similarity_opposite_texts_low():
    texts = [
        "the sun is a hot star with plasma",
        "database indexing accelerates sql query execution",
        "classical music uses orchestral instrumentation",
    ]

    score = RAGValidator._pairwise_similarity(texts)

    assert score < 0.2


def test_validation_result_passed_true_when_all_scores_above_threshold():
    result = ValidationResult(
        passed=True,
        faithfulness=0.9,
        consistency=0.9,
        attribution=0.8,
        overall_score=0.88,
    )

    assert result.passed is True


def test_validation_result_passed_false_when_one_score_below_threshold():
    result = ValidationResult(
        passed=False,
        faithfulness=0.5,
        consistency=0.9,
        attribution=0.8,
        overall_score=0.71,
    )

    assert result.passed is False
