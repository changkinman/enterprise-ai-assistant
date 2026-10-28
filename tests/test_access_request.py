"""Offline access-request adapter contract tests."""

import unittest

from core.auth import Actor
from core.operation_repository import InMemoryOperationRepository
from core.operations import ActionProposal, ValidationRejected
from core.side_effect_executor import SideEffectExecutor
from core.access_request import AccessRequestOperation, InMemoryAccessRequestStore


class AccessRequestTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = InMemoryAccessRequestStore()
        self.adapter = AccessRequestOperation(self.store)
        self.executor = SideEffectExecutor(
            InMemoryOperationRepository(), {"access_request": self.adapter}
        )
        self.employee = Actor("emp-001", "employee")

    async def test_prepare_then_confirm_creates_one_local_submitted_request(self):
        proposal = ActionProposal("access_request", {
            "system_name": "vpn", "permission": "read", "reason": "remote support",
        })
        preview = await self.executor.prepare(self.employee, proposal)
        self.assertEqual(preview.preview.details["system_name"], "vpn")
        self.assertEqual(await self.store.list_for_owner("emp-001"), [])
        first = await self.executor.confirm(self.employee, preview.operation_id, "access-1")
        second = await self.executor.confirm(self.employee, preview.operation_id, "access-1")
        self.assertEqual(first, second)
        self.assertEqual(first["state"], "submitted")
        self.assertEqual(first["owner_id"], "emp-001")
        self.assertEqual(len(await self.store.list_for_owner("emp-001")), 1)

    async def test_invalid_values_and_employee_write_are_denied_before_preview(self):
        for params in (
            {"system_name": "unknown", "permission": "read", "reason": "need it"},
            {"system_name": ["vpn"], "permission": "read", "reason": "need it"},
            {"system_name": "vpn", "permission": "admin", "reason": "need it"},
            {"system_name": "vpn", "permission": ["read"], "reason": "need it"},
            {"system_name": "vpn", "permission": "read", "reason": " "},
            {"system_name": "vpn", "permission": "read", "reason": "x" * 201},
            {"system_name": "vpn", "permission": "write", "reason": "need it"},
        ):
            with self.subTest(params=params), self.assertRaises(ValidationRejected):
                await self.executor.prepare(self.employee, ActionProposal("access_request", params))
        self.assertEqual(await self.store.list_for_owner("emp-001"), [])

    async def test_manager_can_prepare_write_and_reason_boundary(self):
        preview = await self.executor.prepare(Actor("mgr-001", "manager"), ActionProposal(
            "access_request", {"system_name": "gitlab", "permission": "write", "reason": "x" * 200}
        ))
        self.assertEqual(preview.preview.details["permission"], "write")
