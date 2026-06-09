"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.middleware import APIKeyAuthMiddleware
from src.api.routes import router


app = FastAPI(
    title="LLM Validation Framework API",
    version="1.0.0",
    description="API for retrieval and generation validation in RAG pipelines.",
)

app.include_router(router)
app.add_middleware(APIKeyAuthMiddleware)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "llm-validation-framework", "status": "running"}
