"""Pinecone interface layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RetrievalMatch:
    id: str
    score: float
    text: str
    metadata: dict[str, Any]


class VectorStoreClient:
    """Small wrapper around Pinecone with safe lazy initialization."""

    def __init__(self, index_name: str, api_key: str | None = None, host: str | None = None):
        self.index_name = index_name
        self.api_key = api_key
        self.host = host
        self._index = None

    def _get_index(self):
        if self._index is not None:
            return self._index

        from pinecone import Pinecone

        if not self.api_key:
            raise ValueError("Pinecone API key is required")

        client = Pinecone(api_key=self.api_key)
        self._index = client.Index(name=self.index_name, host=self.host) if self.host else client.Index(self.index_name)
        return self._index

    def query(self, vector: list[float], top_k: int = 5, namespace: str | None = None) -> list[RetrievalMatch]:
        index = self._get_index()
        response = index.query(vector=vector, top_k=top_k, namespace=namespace, include_metadata=True)

        matches = []
        for raw in response.get("matches", []):
            metadata = raw.get("metadata") or {}
            matches.append(
                RetrievalMatch(
                    id=str(raw.get("id", "")),
                    score=float(raw.get("score", 0.0)),
                    text=str(metadata.get("text", "")),
                    metadata=metadata,
                )
            )
        return matches
