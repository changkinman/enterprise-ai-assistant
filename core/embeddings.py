"""Explicit asynchronous embeddings without importing a model at startup."""

import asyncio
from typing import Protocol

from core.database import DatabaseSettings


class EmbeddingProvider(Protocol):
    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbeddingProvider:
    """Load and encode in a worker thread; serialize access to the shared model."""

    dimension = 512

    def __init__(self, settings: DatabaseSettings | None = None) -> None:
        self.settings = settings if settings is not None else DatabaseSettings.from_env()
        if self.settings.embedding_dimension != self.dimension:
            raise ValueError("The initial PostgreSQL schema requires embedding dimension 512")
        self._model = None
        self._lock = asyncio.Lock()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._encode, list(texts))

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.settings.embedding_model)
        vectors = [[float(value) for value in row] for row in self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        )]
        if len(vectors) != len(texts) or any(len(row) != self.dimension for row in vectors):
            raise ValueError("Embedding output does not match the configured dimension or batch")
        return vectors
