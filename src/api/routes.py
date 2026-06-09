"""API route definitions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from src.api.middleware import InMemoryRateLimiter
from src.pipeline.rag_validator import RAGValidator


router = APIRouter(prefix="/v1", tags=["validation"])
_rate_limiter = InMemoryRateLimiter(max_requests=120, window_seconds=60)


class ValidationRequest(BaseModel):
    query: str = Field(min_length=1)
    retrieved_chunks: list[str] = Field(default_factory=list)
    answer: str = Field(min_length=1)


class ValidationResponse(BaseModel):
    passed: bool
    faithfulness: float
    consistency: float
    attribution: float
    overall_score: float
    issues: list[str]
    latency_ms: float


def rate_limit_dependency(request: Request) -> None:
    _rate_limiter.check(request)


@router.get("/health", tags=["system"])
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/validate", response_model=ValidationResponse)
async def validate_response(payload: ValidationRequest, _=Depends(rate_limit_dependency)) -> ValidationResponse:
    validator = RAGValidator(llm_client=None)
    result = await validator.validate(
        query=payload.query,
        retrieved_chunks=payload.retrieved_chunks,
        answer=payload.answer,
        regenerate_fn=None,
    )

    return ValidationResponse(
        passed=result.passed,
        faithfulness=result.faithfulness,
        consistency=result.consistency,
        attribution=result.attribution,
        overall_score=result.overall_score,
        issues=result.issues,
        latency_ms=result.latency_ms,
    )
