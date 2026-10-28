"""Parameterized pgvector persistence for already chunked knowledge and memory."""

from typing import Any
from uuid import uuid4

from core.database import Database
from core.embeddings import EmbeddingProvider


class VectorStore:
    def __init__(self, database: Database, provider: EmbeddingProvider) -> None:
        self.database = database
        self.provider = provider

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = await self.provider.embed(texts)
        if len(vectors) != len(texts) or any(
            len(row) != self.provider.dimension for row in vectors
        ):
            raise ValueError("Embedding output does not match the provider dimension or batch")
        return vectors

    async def add_knowledge(self, chunks: list[dict[str, Any]]) -> int:
        """Persist pre-chunked documents atomically, with one embedding per chunk."""
        if not chunks:
            return 0
        vectors = await self._embed([chunk["content"] for chunk in chunks])
        async with self.database.transaction() as conn:
            for chunk, vector in zip(chunks, vectors):
                title = chunk.get("title", "")
                metadata = dict(chunk.get("metadata", {}))
                metadata["title"] = title
                metadata.setdefault("chunk_index", metadata.get("chunk", 0))
                await conn.execute(
                    """INSERT INTO knowledge_chunks (id, title, content, metadata, embedding)
                       VALUES ($1, $2, $3, $4::jsonb, $5)
                       ON CONFLICT (id) DO UPDATE SET
                           title = EXCLUDED.title,
                           content = EXCLUDED.content,
                           metadata = EXCLUDED.metadata,
                           embedding = EXCLUDED.embedding""",
                    chunk["id"], title, chunk["content"], metadata, vector,
                )
        return len(chunks)

    async def count_knowledge(self) -> int:
        async with self.database.transaction() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM knowledge_chunks")

    async def search_knowledge(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query.strip() or top_k <= 0:
            return []
        vector = (await self._embed([query]))[0]
        async with self.database.transaction() as conn:
            rows = await conn.fetch(
                """SELECT content, metadata, embedding <=> $1 AS distance
                   FROM knowledge_chunks ORDER BY embedding <=> $1 LIMIT $2""",
                vector, top_k,
            )
        return [{
            "title": row["metadata"].get("title", ""),
            "content": row["content"],
            "score": round(1.0 - row["distance"], 4),
            "chunk": row["metadata"].get("chunk_index", row["metadata"].get("chunk", 0)),
        } for row in rows]

    async def store_episodic(self, user_id: str, conv_id: str, text: str, summary: str) -> None:
        vector = (await self._embed([summary]))[0]
        async with self.database.transaction() as conn:
            await conn.execute(
                """INSERT INTO episodic_memories
                   (id, user_id, conv_id, summary, full_text, embedding)
                   VALUES ($1, $2, $3, $4, $5, $6)""",
                uuid4().hex, user_id, conv_id, summary, text, vector,
            )

    async def search_episodic(self, user_id: str, query: str, top_k: int = 3) -> list[str]:
        if not query.strip() or top_k <= 0:
            return []
        vector = (await self._embed([query]))[0]
        async with self.database.transaction() as conn:
            rows = await conn.fetch(
                """SELECT summary, embedding <=> $2 AS distance FROM episodic_memories
                   WHERE user_id = $1 ORDER BY embedding <=> $2 LIMIT $3""",
                user_id, vector, top_k,
            )
        return [row["summary"] for row in rows]

    async def upsert_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        async with self.database.transaction() as conn:
            await conn.execute(
                """INSERT INTO user_profiles (user_id, profile) VALUES ($1, $2::jsonb)
                   ON CONFLICT (user_id) DO UPDATE
                   SET profile = EXCLUDED.profile, updated_at = now()""",
                user_id, profile,
            )

    async def get_profile(self, user_id: str) -> dict[str, Any]:
        async with self.database.transaction() as conn:
            row = await conn.fetchrow(
                "SELECT profile FROM user_profiles WHERE user_id = $1", user_id,
            )
        return row["profile"] if row is not None else {}
