"""Persist operation previews, idempotency responses and audit events."""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, AsyncContextManager, AsyncIterator, Protocol
from uuid import uuid4

from core.auth import Actor
from core.database import Database
from core.operations import (
    IdempotencyConflict, OperationState, OperationStateConflict,
    PreviewPayload, StoredOperation,
)


class OperationRepository(Protocol):
    def transaction(self) -> AsyncContextManager[Any]: ...

    async def create_pending(
        self, actor: Actor, action_type: str, params: dict[str, Any], preview: PreviewPayload,
        resource_version: int, snapshot_hash: str, expires_at: datetime,
    ) -> StoredOperation: ...

    async def get_for_actor(self, operation_id: str, actor: Actor) -> StoredOperation | None: ...

    async def lock_for_confirmation(
        self, operation_id: str, actor: Actor, connection: Any,
    ) -> StoredOperation | None: ...

    async def load_idempotency(
        self, actor: Actor, action_type: str, key: str, request_hash: str,
        connection: Any = None,
    ) -> dict[str, Any] | None: ...

    async def save_idempotency(
        self, actor: Actor, action_type: str, key: str, request_hash: str,
        response: dict[str, Any], connection: Any = None,
    ) -> dict[str, Any]: ...

    async def save_completed(
        self, operation_id: str, actor: Actor, connection: Any = None,
    ) -> StoredOperation: ...

    async def append_audit(
        self, operation_id: str | None, event_type: str, actor: Actor,
        detail: dict[str, Any], connection: Any = None,
    ) -> None: ...


class InMemoryOperationRepository:
    """Test-only store. Its transaction lock serializes the full confirm flow."""

    def __init__(self) -> None:
        self._operations: dict[str, StoredOperation] = {}
        self._idempotency: dict[tuple[str, str, str], tuple[str, dict[str, Any]]] = {}
        self._audit_events: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._transaction_marker = object()

    @property
    def audit_events(self) -> list[dict[str, Any]]:
        return deepcopy(self._audit_events)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[object]:
        async with self._lock:
            yield self._transaction_marker

    @asynccontextmanager
    async def _guard(self, connection: Any) -> AsyncIterator[None]:
        if connection is self._transaction_marker:
            yield
        else:
            async with self._lock:
                yield

    def _expire(self, operation: StoredOperation) -> StoredOperation:
        if operation.state is OperationState.PENDING and operation.expires_at <= datetime.now(timezone.utc):
            operation = replace(operation, state=OperationState.EXPIRED)
            self._operations[operation.operation_id] = operation
            self._audit_events.append({
                "operation_id": operation.operation_id, "event_type": "preview_expired",
                "actor_id": operation.actor_id, "detail": {},
            })
        return operation

    async def create_pending(
        self, actor: Actor, action_type: str, params: dict[str, Any], preview: PreviewPayload,
        resource_version: int, snapshot_hash: str, expires_at: datetime,
    ) -> StoredOperation:
        operation = StoredOperation(
            uuid4().hex, actor.id, action_type, deepcopy(params), deepcopy(preview),
            snapshot_hash, resource_version, OperationState.PENDING, expires_at,
        )
        async with self._lock:
            self._operations[operation.operation_id] = operation
        return deepcopy(operation)

    async def get_for_actor(self, operation_id: str, actor: Actor) -> StoredOperation | None:
        async with self._lock:
            operation = self._operations.get(operation_id)
            if operation is None or operation.actor_id != actor.id:
                return None
            return deepcopy(self._expire(operation))

    async def lock_for_confirmation(
        self, operation_id: str, actor: Actor, connection: Any = None,
    ) -> StoredOperation | None:
        async with self._guard(connection):
            operation = self._operations.get(operation_id)
            if operation is None or operation.actor_id != actor.id:
                return None
            return deepcopy(self._expire(operation))

    async def load_idempotency(
        self, actor: Actor, action_type: str, key: str, request_hash: str,
        connection: Any = None,
    ) -> dict[str, Any] | None:
        async with self._guard(connection):
            record = self._idempotency.get((actor.id, action_type, key))
            if record is None:
                return None
            if record[0] != request_hash:
                raise IdempotencyConflict("Idempotency key belongs to a different request")
            return deepcopy(record[1])

    async def save_idempotency(
        self, actor: Actor, action_type: str, key: str, request_hash: str,
        response: dict[str, Any], connection: Any = None,
    ) -> dict[str, Any]:
        async with self._guard(connection):
            index = (actor.id, action_type, key)
            record = self._idempotency.get(index)
            if record is not None:
                if record[0] != request_hash:
                    raise IdempotencyConflict("Idempotency key belongs to a different request")
                return deepcopy(record[1])
            self._idempotency[index] = (request_hash, deepcopy(response))
            return deepcopy(response)

    async def save_completed(
        self, operation_id: str, actor: Actor, connection: Any = None,
    ) -> StoredOperation:
        async with self._guard(connection):
            operation = self._operations.get(operation_id)
            if operation is None or operation.actor_id != actor.id:
                raise OperationStateConflict("Operation is unavailable")
            operation = self._expire(operation)
            if operation.state is not OperationState.PENDING:
                raise OperationStateConflict("Operation is not pending")
            completed = replace(operation, state=OperationState.COMPLETED)
            self._operations[operation_id] = completed
            return deepcopy(completed)

    async def append_audit(
        self, operation_id: str | None, event_type: str, actor: Actor,
        detail: dict[str, Any], connection: Any = None,
    ) -> None:
        async with self._guard(connection):
            self._audit_events.append({
                "operation_id": operation_id, "event_type": event_type,
                "actor_id": actor.id, "detail": deepcopy(detail),
            })


class PostgresOperationRepository:
    """Async PostgreSQL store. Confirm callers supply one transaction connection."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def transaction(self):
        return self.database.transaction()

    @staticmethod
    def _operation(row: Any) -> StoredOperation:
        data = dict(row)
        raw_preview = data["preview"]
        preview = PreviewPayload(**raw_preview) if isinstance(raw_preview, dict) else raw_preview
        return StoredOperation(
            data["operation_id"], data["actor_id"], data["action_type"],
            data["params"], preview, data["snapshot_hash"],
            data["resource_version"], OperationState(data["state"]), data["expires_at"],
        )

    @asynccontextmanager
    async def _connection(self, connection: Any) -> AsyncIterator[Any]:
        if connection is not None:
            yield connection
        else:
            async with self.database.transaction() as opened:
                yield opened

    async def create_pending(
        self, actor: Actor, action_type: str, params: dict[str, Any], preview: PreviewPayload,
        resource_version: int, snapshot_hash: str, expires_at: datetime,
    ) -> StoredOperation:
        async with self._connection(None) as connection:
            row = await connection.fetchrow(
                """INSERT INTO operations
                   (operation_id, actor_id, action_type, params, preview, snapshot_hash,
                    resource_version, state, expires_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,'pending',$8) RETURNING *""",
                uuid4().hex, actor.id, action_type, params,
                {"summary": preview.summary, "details": preview.details},
                snapshot_hash, resource_version, expires_at,
            )
            return self._operation(row)

    async def _expire_if_needed(self, operation: StoredOperation, connection: Any) -> StoredOperation:
        if operation.state is OperationState.PENDING and operation.expires_at <= datetime.now(timezone.utc):
            await connection.execute(
                "UPDATE operations SET state='expired', updated_at=now() WHERE operation_id=$1 AND state='pending'",
                operation.operation_id,
            )
            await self.append_audit(
                operation.operation_id, "preview_expired", Actor(operation.actor_id, ""), {}, connection,
            )
            return replace(operation, state=OperationState.EXPIRED)
        return operation

    async def get_for_actor(self, operation_id: str, actor: Actor) -> StoredOperation | None:
        async with self._connection(None) as connection:
            row = await connection.fetchrow(
                "SELECT * FROM operations WHERE operation_id=$1 AND actor_id=$2 FOR UPDATE",
                operation_id, actor.id,
            )
            return await self._expire_if_needed(self._operation(row), connection) if row else None

    async def lock_for_confirmation(
        self, operation_id: str, actor: Actor, connection: Any,
    ) -> StoredOperation | None:
        if connection is None:
            raise ValueError("Confirmation lock requires an active transaction connection")
        row = await connection.fetchrow(
            "SELECT * FROM operations WHERE operation_id=$1 AND actor_id=$2 FOR UPDATE",
            operation_id, actor.id,
        )
        return await self._expire_if_needed(self._operation(row), connection) if row else None

    async def load_idempotency(
        self, actor: Actor, action_type: str, key: str, request_hash: str,
        connection: Any = None,
    ) -> dict[str, Any] | None:
        async with self._connection(connection) as active:
            row = await active.fetchrow(
                """SELECT request_hash, response FROM idempotency_records
                   WHERE actor_id=$1 AND action_type=$2 AND idempotency_key=$3""",
                actor.id, action_type, key,
            )
            if row is None:
                return None
            if row["request_hash"] != request_hash:
                raise IdempotencyConflict("Idempotency key belongs to a different request")
            return dict(row["response"])

    async def save_idempotency(
        self, actor: Actor, action_type: str, key: str, request_hash: str,
        response: dict[str, Any], connection: Any = None,
    ) -> dict[str, Any]:
        async with self._connection(connection) as active:
            row = await active.fetchrow(
                """INSERT INTO idempotency_records
                   (actor_id, action_type, idempotency_key, request_hash, response)
                   VALUES ($1,$2,$3,$4,$5)
                   ON CONFLICT (actor_id, action_type, idempotency_key) DO UPDATE
                   SET request_hash=idempotency_records.request_hash
                   RETURNING request_hash, response""",
                actor.id, action_type, key, request_hash, response,
            )
            if row["request_hash"] != request_hash:
                raise IdempotencyConflict("Idempotency key belongs to a different request")
            return dict(row["response"])

    async def save_completed(
        self, operation_id: str, actor: Actor, connection: Any = None,
    ) -> StoredOperation:
        async with self._connection(connection) as active:
            row = await active.fetchrow(
                """UPDATE operations SET state='completed', updated_at=now()
                   WHERE operation_id=$1 AND actor_id=$2 AND state='pending'
                   RETURNING *""",
                operation_id, actor.id,
            )
            if row is None:
                raise OperationStateConflict("Operation is not pending or unavailable")
            return self._operation(row)

    async def append_audit(
        self, operation_id: str | None, event_type: str, actor: Actor,
        detail: dict[str, Any], connection: Any = None,
    ) -> None:
        async with self._connection(connection) as active:
            await active.execute(
                """INSERT INTO operation_audit_events
                   (operation_id, event_type, actor_id, detail) VALUES ($1,$2,$3,$4)""",
                operation_id, event_type, actor.id, detail,
            )
