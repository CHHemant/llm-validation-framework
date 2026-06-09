"""LangChain pipeline assembly."""

from __future__ import annotations

from typing import Awaitable, Callable

from src.pipeline.guardrails import evaluate_guardrails
from src.pipeline.rag_validator import ValidationResult
from src.pipeline.validators import run_validation_suite


async def run_rag_pipeline(
    query: str,
    retrieve_fn: Callable[[str], Awaitable[list[str]]],
    generate_fn: Callable[[str, list[str]], Awaitable[str]],
    regenerate_fn: Callable[[str], Awaitable[tuple[list[str], str]]] | None = None,
    llm_client=None,
) -> dict:
    """Runs retrieval, generation, validation, and guardrail decision in one flow."""
    chunks = await retrieve_fn(query)
    answer = await generate_fn(query, chunks)

    report = await run_validation_suite(
        query=query,
        answer=answer,
        retrieved_chunks=chunks,
        regenerate_fn=regenerate_fn,
        llm_client=llm_client,
    )

    validation = report["validation"]
    validation_result = ValidationResult(
        passed=validation["passed"],
        faithfulness=validation["faithfulness"],
        consistency=validation["consistency"],
        attribution=validation["attribution"],
        overall_score=validation["overall_score"],
        issues=validation["issues"],
        latency_ms=validation["latency_ms"],
    )
    decision = evaluate_guardrails(validation_result)

    report["decision"] = {
        "allow_response": decision.allow_response,
        "reason": decision.reason,
        "severity": decision.severity,
    }
    report["query"] = query
    report["answer"] = answer
    return report
