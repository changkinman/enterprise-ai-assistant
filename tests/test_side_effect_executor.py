"""Offline contract tests for authenticated, persisted HITL primitives."""

import asyncio
from copy import deepcopy
import os
import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from unittest.mock import patch

import jwt
from fastapi import HTTPException

from core.auth import Actor, AuthenticationError, create_access_token, create_dev_access_token, decode_actor, get_current_actor
from core.operation_repository import InMemoryOperationRepository, PostgresOperationRepository
from core.operations import IdempotencyConflict, OperationExpired, OperationState, PreviewPayload

TEST_SECRET = "test-secret-key-with-at-least-32-bytes"


class ActorTokenTests(unittest.IsolatedAsyncioTestCase):
    def test_valid_signed_token_is_the_only_actor_source(self):
        token = create_access_token(Actor("emp-001", "employee"), secret=TEST_SECRET)
        self.assertEqual(decode_actor(token, secret=TEST_SECRET), Actor("emp-001", "employee"))
        self.assertEqual(set(jwt.decode(token, TEST_SECRET, algorithms=["HS256"])), {"sub", "role", "exp"})
        with self.assertRaises(AuthenticationError):
            decode_actor(token + "x", secret=TEST_SECRET)

    def test_actor_is_immutable_and_expired_token_is_rejected(self):
        actor = Actor("emp-001", "employee")
        with self.assertRaises(FrozenInstanceError):
            actor.id = "emp-002"
        token = create_access_token(actor, secret=TEST_SECRET, expires_in=timedelta(seconds=-1))
        with self.assertRaises(AuthenticationError):
            decode_actor(token, secret=TEST_SECRET)

    async def test_bearer_dependency_rejects_missing_and_malformed_header(self):
        for authorization in (None, "Basic abc", "Bearer", "Bearer abc extra"):
            with self.subTest(authorization=authorization), self.assertRaises(HTTPException) as caught:
                await get_current_actor(authorization)
            self.assertEqual(caught.exception.status_code, 401)

    async def test_bearer_dependency_uses_signed_claims(self):
        with patch.dict(os.environ, {"JWT_SECRET_KEY": TEST_SECRET}):
            token = create_access_token(Actor("emp-001", "employee"), secret=TEST_SECRET)
            self.assertEqual(await get_current_actor(f"Bearer {token}"), Actor("emp-001", "employee"))
            expired = create_access_token(Actor("emp-001", "employee"), secret=TEST_SECRET,
                                          expires_in=timedelta(seconds=-1))
            with self.assertRaises(HTTPException) as caught:
                await get_current_actor(f"Bearer {expired}")
            self.assertEqual(caught.exception.status_code, 401)

    def test_dev_token_helper_requires_development_environment(self):
        with patch.dict(os.environ, {"APP_ENV": "production", "JWT_SECRET_KEY": TEST_SECRET}):
            with self.assertRaises(AuthenticationError):
                create_dev_access_token(Actor("emp-001", "employee"))
        with patch.dict(os.environ, {"APP_ENV": "development", "JWT_SECRET_KEY": TEST_SECRET}):
            token = create_dev_access_token(Actor("emp-001", "employee"))
            self.assertEqual(decode_actor(token, secret=TEST_SECRET), Actor("emp-001", "employee"))


class InMemoryOperationRepositoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = InMemoryOperationRepository()
        self.employee = Actor("emp-001", "employee")
        self.future = datetime.now(timezone.utc) + timedelta(minutes=10)
        self.preview = PreviewPayload(summary="Apply for access", details={"system": "wiki"})

    async def _pending(self, expires_at=None):
        return await self.repo.create_pending(
            self.employee, "access_request", {"system": "wiki"}, self.preview,
            1, "snapshot-a", expires_at or self.future,
        )

    async def test_same_key_with_changed_request_hash_is_rejected(self):
        await self.repo.save_idempotency(self.employee, "access_request", "k1", "hash-a", {"ok": True})
        with self.assertRaises(IdempotencyConflict):
            await self.repo.save_idempotency(self.employee, "access_request", "k1", "hash-b", {"ok": True})
        self.assertEqual(
            await self.repo.load_idempotency(self.employee, "access_request", "k1", "hash-a"),
            {"ok": True},
        )

    async def test_other_actor_cannot_load_pending_operation(self):
        operation = await self._pending()
        self.assertIsNone(await self.repo.get_for_actor(operation.operation_id, Actor("emp-002", "employee")))
        self.assertEqual((await self.repo.get_for_actor(operation.operation_id, self.employee)).state, OperationState.PENDING)

    async def test_expired_preview_transitions_once_and_is_audited(self):
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        operation = await self._pending(past)
        first = await self.repo.get_for_actor(operation.operation_id, self.employee)
        second = await self.repo.get_for_actor(operation.operation_id, self.employee)
        self.assertEqual(first.state, OperationState.EXPIRED)
        self.assertEqual(second.state, OperationState.EXPIRED)
        self.assertEqual(
            [event["event_type"] for event in self.repo.audit_events],
            ["preview_expired"],
        )

    async def test_locked_confirmation_and_completed_result_are_durable(self):
        operation = await self._pending()
        async with self.repo.transaction() as connection:
            locked = await self.repo.lock_for_confirmation(operation.operation_id, self.employee, connection)
            self.assertEqual(locked.operation_id, operation.operation_id)
            await self.repo.save_completed(operation.operation_id, self.employee, connection)
            await self.repo.save_idempotency(self.employee, "access_request", "k1", "hash-a", {"ticket": "A1"}, connection)
        self.assertEqual((await self.repo.get_for_actor(operation.operation_id, self.employee)).state, OperationState.COMPLETED)
        self.assertEqual(await self.repo.load_idempotency(self.employee, "access_request", "k1", "hash-a"), {"ticket": "A1"})


class FakeConnection:
    def __init__(self, row=None):
        self.row = row
        self.statements = []

    async def fetchrow(self, sql, *params):
        self.statements.append((sql, params))
        if self.row is None:
            return None
        if "FROM operations" not in sql and "AS is_expired" in sql:
            return {"is_expired": params[0] <= datetime.now(timezone.utc)}
        result = deepcopy(self.row)
        if "AS is_expired" in sql:
            result["is_expired"] = result["expires_at"] <= datetime.now(timezone.utc)
        return result

    async def execute(self, sql, *params):
        self.statements.append((sql, params))
        return "UPDATE 1"


class FakeDatabase:
    def __init__(self, connection):
        self.connection = connection

    @asynccontextmanager
    async def transaction(self):
        yield self.connection


class TransactionalConnection:
    """Only the SQL effects under test; commit or rollback belongs to the fake DB."""

    def __init__(self, row, audit, now, after_lock=None):
        self.row = row
        self.audit = audit
        self.now = now
        self.after_lock = after_lock
        self.transaction_started_at = now()
        self.statements = []

    def sql_time(self, sql):
        return self.now() if "clock_timestamp()" in sql else self.transaction_started_at

    async def fetchrow(self, sql, *params):
        self.statements.append((sql, params))
        if "FROM operations" not in sql and "AS is_expired" in sql:
            return {"is_expired": params[0] <= self.sql_time(sql)}
        if sql.lstrip().startswith("SELECT"):
            if (self.row["operation_id"], self.row["actor_id"]) != params[:2]:
                return None
            result = deepcopy(self.row)
            if "AS is_expired" in sql:
                result["is_expired"] = self.row["expires_at"] <= self.sql_time(sql)
            if "FOR UPDATE" in sql and self.after_lock is not None:
                callback, self.after_lock = self.after_lock, None
                callback()
            return result
        if sql.lstrip().startswith("UPDATE operations SET state='completed'"):
            if (self.row["operation_id"], self.row["actor_id"]) != params[:2]:
                return None
            if self.row["state"] != "pending":
                return None
            if ("expires_at > now()" in sql or "expires_at > clock_timestamp()" in sql) and self.row["expires_at"] <= self.sql_time(sql):
                return None
            self.row["state"] = "completed"
            return deepcopy(self.row)
        raise AssertionError(f"Unexpected SQL: {sql}")

    async def execute(self, sql, *params):
        self.statements.append((sql, params))
        if sql.lstrip().startswith("UPDATE operations SET state='expired'"):
            if (self.row["operation_id"] == params[0] and self.row["state"] == "pending"
                    and self.row["expires_at"] <= self.sql_time(sql)):
                self.row["state"] = "expired"
                return "UPDATE 1"
            return "UPDATE 0"
        if sql.lstrip().startswith("INSERT INTO operation_audit_events"):
            self.audit.append({"operation_id": params[0], "event_type": params[1], "actor_id": params[2]})
            return "INSERT 1"
        raise AssertionError(f"Unexpected SQL: {sql}")


class TransactionalDatabase:
    def __init__(self, row):
        self.row = deepcopy(row)
        self.audit = []
        self.connections = []
        self.now = datetime.now(timezone.utc)
        self.after_lock = None

    @asynccontextmanager
    async def transaction(self):
        connection = TransactionalConnection(
            deepcopy(self.row), deepcopy(self.audit), lambda: self.now, self.after_lock,
        )
        self.after_lock = None
        self.connections.append(connection)
        yield connection
        self.row, self.audit = connection.row, connection.audit


class PostgresOperationRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirmation_locks_only_actor_owned_row_in_given_transaction(self):
        future = datetime.now(timezone.utc) + timedelta(minutes=10)
        row = {
            "operation_id": "op-1", "actor_id": "emp-001", "action_type": "access_request",
            "params": {"system": "wiki"}, "preview": {"summary": "Apply", "details": {}},
            "snapshot_hash": "snapshot", "resource_version": 1,
            "state": "pending", "expires_at": future,
        }
        connection = FakeConnection(row)
        repository = PostgresOperationRepository(FakeDatabase(connection))
        operation = await repository.lock_for_confirmation("op-1", Actor("emp-001", "employee"), connection)
        self.assertEqual(operation.state, OperationState.PENDING)
        self.assertIn("FOR UPDATE", connection.statements[0][0])
        self.assertEqual(connection.statements[0][1], ("op-1", "emp-001"))

    async def test_expiry_persists_after_confirmation_transaction_rolls_back(self):
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        row = {
            "operation_id": "op-1", "actor_id": "emp-001", "action_type": "access_request",
            "params": {}, "preview": {"summary": "Apply", "details": {}},
            "snapshot_hash": "snapshot", "resource_version": 1,
            "state": "pending", "expires_at": past,
        }
        database = TransactionalDatabase(row)
        repository = PostgresOperationRepository(database)
        with patch("core.operation_repository.datetime") as app_clock:
            app_clock.now.return_value = past - timedelta(minutes=10)
            with self.assertRaises(OperationExpired):
                async with repository.transaction() as connection:
                    await repository.lock_for_confirmation("op-1", Actor("emp-001", "employee"), connection)
        self.assertEqual(database.row["state"], "expired")
        self.assertEqual(database.audit, [{"operation_id": "op-1", "event_type": "preview_expired", "actor_id": "emp-001"}])
        self.assertEqual(len(database.connections), 2)  # rolled-back business transaction, committed expiry

    async def test_app_clock_ahead_does_not_falsely_expire_live_postgres_preview(self):
        database_now = datetime.now(timezone.utc)
        future = database_now + timedelta(minutes=10)
        row = {
            "operation_id": "op-1", "actor_id": "emp-001", "action_type": "access_request",
            "params": {}, "preview": {"summary": "Apply", "details": {}},
            "snapshot_hash": "snapshot", "resource_version": 1,
            "state": "pending", "expires_at": future,
        }
        database = TransactionalDatabase(row)
        database.now = database_now
        repository = PostgresOperationRepository(database)
        with patch("core.operation_repository.datetime") as app_clock:
            app_clock.now.return_value = future + timedelta(seconds=1)
            async with repository.transaction() as connection:
                locked = await repository.lock_for_confirmation("op-1", Actor("emp-001", "employee"), connection)
        self.assertEqual(locked.state, OperationState.PENDING)
        self.assertEqual(database.row["state"], "pending")
        self.assertEqual(database.audit, [])

    async def test_lock_wait_past_expiry_uses_live_database_clock(self):
        database_now = datetime.now(timezone.utc)
        future = database_now + timedelta(minutes=10)
        row = {
            "operation_id": "op-1", "actor_id": "emp-001", "action_type": "access_request",
            "params": {}, "preview": {"summary": "Apply", "details": {}},
            "snapshot_hash": "snapshot", "resource_version": 1,
            "state": "pending", "expires_at": future,
        }
        database = TransactionalDatabase(row)
        database.now = database_now
        database.after_lock = lambda: setattr(database, "now", future + timedelta(seconds=1))
        repository = PostgresOperationRepository(database)
        with self.assertRaises(OperationExpired):
            async with repository.transaction() as connection:
                await repository.lock_for_confirmation("op-1", Actor("emp-001", "employee"), connection)
        self.assertEqual(database.row["state"], "expired")
        self.assertEqual(database.audit, [{"operation_id": "op-1", "event_type": "preview_expired", "actor_id": "emp-001"}])

    async def test_completion_rejects_expiry_that_occurs_after_lock(self):
        future = datetime.now(timezone.utc) + timedelta(minutes=10)
        row = {
            "operation_id": "op-1", "actor_id": "emp-001", "action_type": "access_request",
            "params": {}, "preview": {"summary": "Apply", "details": {}},
            "snapshot_hash": "snapshot", "resource_version": 1,
            "state": "pending", "expires_at": future,
        }
        database = TransactionalDatabase(row)
        repository = PostgresOperationRepository(database)
        application_now = database.now
        with patch("core.operation_repository.datetime") as clock:
            # The database clock advances past expiry while the app clock lags.
            clock.now.side_effect = lambda tz: application_now
            with self.assertRaises(OperationExpired):
                async with repository.transaction() as connection:
                    locked = await repository.lock_for_confirmation("op-1", Actor("emp-001", "employee"), connection)
                    self.assertEqual(locked.state, OperationState.PENDING)
                    database.now = future + timedelta(seconds=1)
                    await repository.save_completed("op-1", Actor("emp-001", "employee"), connection)
        self.assertEqual(database.row["state"], "expired")
        self.assertEqual(len(database.audit), 1)
        self.assertEqual(database.audit[0]["event_type"], "preview_expired")
        completion_sql = next(sql for connection in database.connections for sql, _ in connection.statements
                              if "UPDATE operations SET state='completed'" in sql)
        self.assertIn("expires_at > clock_timestamp()", completion_sql)


if __name__ == "__main__":
    unittest.main()
