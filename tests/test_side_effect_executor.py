"""Offline contract tests for authenticated, persisted HITL primitives."""

import asyncio
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
from core.operations import IdempotencyConflict, OperationState, PreviewPayload

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
        return self.row

    async def execute(self, sql, *params):
        self.statements.append((sql, params))
        return "UPDATE 1"


class FakeDatabase:
    def __init__(self, connection):
        self.connection = connection

    @asynccontextmanager
    async def transaction(self):
        yield self.connection


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

    async def test_postgres_expiry_updates_record_and_appends_one_audit_event(self):
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        row = {
            "operation_id": "op-1", "actor_id": "emp-001", "action_type": "access_request",
            "params": {}, "preview": {"summary": "Apply", "details": {}},
            "snapshot_hash": "snapshot", "resource_version": 1,
            "state": "pending", "expires_at": past,
        }
        connection = FakeConnection(row)
        repository = PostgresOperationRepository(FakeDatabase(connection))
        operation = await repository.lock_for_confirmation("op-1", Actor("emp-001", "employee"), connection)
        self.assertEqual(operation.state, OperationState.EXPIRED)
        self.assertTrue(any("UPDATE operations SET state='expired'" in sql for sql, _ in connection.statements))
        self.assertTrue(any("INSERT INTO operation_audit_events" in sql and params[1] == "preview_expired"
                            for sql, params in connection.statements))


if __name__ == "__main__":
    unittest.main()
