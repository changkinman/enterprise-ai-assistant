"""Offline ASGI tests for JWT-scoped operation endpoints."""

import os
import sys
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import httpx

sys.modules.setdefault("anthropic", SimpleNamespace(AsyncAnthropic=lambda **_: None))

from api import main as api
from core.access_request import AccessRequestOperation, InMemoryAccessRequestStore
from core.auth import Actor, create_access_token
from core.operation_repository import InMemoryOperationRepository
from core.operations import ValidationResult
from core.side_effect_executor import SideEffectExecutor


class OperationApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.secret = "test-secret-key-with-at-least-32-bytes"
        self.env = patch.dict(os.environ, {"JWT_SECRET_KEY": self.secret, "APP_ENV": "development"})
        self.env.start()
        self.store = InMemoryAccessRequestStore()
        self.repo = InMemoryOperationRepository()
        api._side_effect_executor = SideEffectExecutor(
            self.repo, {"access_request": AccessRequestOperation(self.store)}
        )

    def tearDown(self):
        api._side_effect_executor = None
        self.env.stop()

    def token(self, actor_id="emp-001", role="employee"):
        return {"Authorization": "Bearer " + create_access_token(Actor(actor_id, role), secret=self.secret)}

    async def call(self, method, path, **kwargs):
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    async def prepare(self):
        return await self.call("POST", "/access-requests/prepare", headers=self.token(), json={
            "system_name": "vpn", "permission": "read", "reason": "remote support",
        })

    async def test_prepare_get_confirm_and_replay(self):
        prepared = await self.prepare()
        self.assertEqual(prepared.status_code, 200, prepared.text)
        operation_id = prepared.json()["operation_id"]
        self.assertEqual((await self.call("GET", f"/operations/{operation_id}",
                                         headers=self.token())).json()["state"], "pending")
        headers = {**self.token(), "Idempotency-Key": "browser-retry"}
        first = await self.call("POST", f"/operations/{operation_id}/confirm", headers=headers)
        second = await self.call("POST", f"/operations/{operation_id}/confirm", headers=headers)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.json(), first.json())
        self.assertEqual(len(await self.store.list_for_owner("emp-001")), 1)

    async def test_auth_required_and_other_actor_forbidden(self):
        self.assertEqual((await self.call("POST", "/access-requests/prepare", json={})).status_code, 401)
        self.assertEqual((await self.call("POST", "/operations/missing/confirm")).status_code, 401)
        self.assertEqual((await self.call("GET", "/operations/missing",
                                          headers={"Authorization": "Bearer bad"})).status_code, 401)
        operation_id = (await self.prepare()).json()["operation_id"]
        other = self.token("emp-002")
        self.assertEqual((await self.call("GET", f"/operations/{operation_id}", headers=other)).status_code, 403)
        self.assertEqual((await self.call("POST", f"/operations/{operation_id}/confirm",
                                          headers={**other, "Idempotency-Key": "k1"})).status_code, 403)

    async def test_missing_header_and_conflicts(self):
        operation_id = (await self.prepare()).json()["operation_id"]
        path = f"/operations/{operation_id}/confirm"
        self.assertEqual((await self.call("POST", path, headers=self.token())).status_code, 422)
        first = await self.call("POST", path, headers={**self.token(), "Idempotency-Key": "k1"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual((await self.call("POST", path, headers={**self.token(),
                          "Idempotency-Key": "k2"})).status_code, 409)
        second_id = (await self.prepare()).json()["operation_id"]
        self.assertEqual((await self.call("POST", f"/operations/{second_id}/confirm",
                          headers={**self.token(), "Idempotency-Key": "k1"})).status_code, 409)

    async def test_expired_preview_conflicts(self):
        operation_id = (await self.prepare()).json()["operation_id"]
        self.repo._operations[operation_id] = replace(
            self.repo._operations[operation_id], expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)
        )
        response = await self.call("POST", f"/operations/{operation_id}/confirm",
                                   headers={**self.token(), "Idempotency-Key": "expired"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(await self.store.list_for_owner("emp-001"), [])

    async def test_stale_preview_conflicts(self):
        operation_id = (await self.prepare()).json()["operation_id"]
        adapter = api._side_effect_executor.adapters["access_request"]
        original = adapter.precheck

        async def changed_version(actor, params):
            checked = await original(actor, params)
            return ValidationResult(checked.allowed, checked.normalized_params, 2, checked.reason)

        adapter.precheck = changed_version
        response = await self.call("POST", f"/operations/{operation_id}/confirm",
                                   headers={**self.token(), "Idempotency-Key": "stale"})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(await self.store.list_for_owner("emp-001"), [])

    async def test_unknown_operation_is_404(self):
        self.assertEqual((await self.call("GET", "/operations/missing",
                         headers=self.token())).status_code, 404)
        self.assertEqual((await self.call("POST", "/operations/missing/confirm",
                         headers={**self.token(), "Idempotency-Key": "k1"})).status_code, 404)

    async def test_validation_denial_and_dev_token_gate(self):
        invalid = await self.call("POST", "/access-requests/prepare", headers=self.token(), json={
            "system_name": "vpn", "permission": "write", "reason": "need it",
        })
        self.assertEqual(invalid.status_code, 422)
        token = await self.call("POST", "/auth/dev-token", json={"actor_id": "emp-001", "role": "employee"})
        self.assertEqual(token.status_code, 200, token.text)
        with patch.dict(os.environ, {"APP_ENV": "production"}):
            hidden = await self.call("POST", "/auth/dev-token", json={"actor_id": "emp-001", "role": "employee"})
        self.assertEqual(hidden.status_code, 404)
