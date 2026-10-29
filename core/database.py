"""PostgreSQL settings and an explicitly started asynchronous connection pool."""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    import asyncpg


@dataclass(frozen=True)
class DatabaseSettings:
    database_url: str = field(
        default="postgresql://echomind:change_me@localhost:5432/echomind", repr=False
    )
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dimension: int = 512

    def __post_init__(self) -> None:
        if self.embedding_dimension <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be a positive integer")

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        try:
            dimension = int(os.environ.get("EMBEDDING_DIMENSION", "512"))
        except ValueError as exc:
            raise ValueError("EMBEDDING_DIMENSION must be a positive integer") from exc
        return cls(
            database_url=os.environ.get("DATABASE_URL", cls.database_url),
            embedding_model=os.environ.get("EMBEDDING_MODEL", cls.embedding_model),
            embedding_dimension=dimension,
        )


class Database:
    """Own a pool; connect at startup, then share transaction connections."""

    def __init__(self, settings: DatabaseSettings | None = None) -> None:
        self.settings = settings if settings is not None else DatabaseSettings.from_env()
        self.pool: "asyncpg.Pool | None" = None
        self._lifecycle_lock = asyncio.Lock()

    async def connect(self) -> None:
        import asyncpg
        from pgvector.asyncpg import register_vector

        async def initialize(connection: "asyncpg.Connection") -> None:
            await register_vector(connection)
            await connection.set_type_codec(
                "jsonb", schema="pg_catalog", encoder=json.dumps,
                decoder=json.loads, format="text",
            )

        async with self._lifecycle_lock:
            if self.pool is None:
                self.pool = await asyncpg.create_pool(
                    dsn=self.settings.database_url, min_size=1, max_size=10,
                    init=initialize,
                )

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self.pool is not None:
                await self.pool.close()
                self.pool = None

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator["asyncpg.Connection"]:
        pool = self.pool
        if pool is None:
            raise RuntimeError("Call Database.connect() before opening a transaction")
        async with pool.acquire() as connection:
            async with connection.transaction():
                yield connection
