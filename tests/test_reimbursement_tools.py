import asyncio
import math
import sys
import unittest
from types import SimpleNamespace

sys.modules.setdefault("anthropic", SimpleNamespace(AsyncAnthropic=lambda **_: None))

from core.reimbursement_models import ClaimState
from core.reimbursement_repository import InMemoryReimbursementRepository
from mcp.tool_manager import MCPToolManager, Tool
from tools.loader import discover_agent_tools


CONTEXT = {"user_id": "emp-001", "conv_id": "conv-1", "agent_type": "billing"}


class ReimbursementToolsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = InMemoryReimbursementRepository(budgets={"engineering": 3000.0})
        self.manager = MCPToolManager(api_key="test")
        discover_agent_tools(self.manager, dependencies={"reimbursement_repository": self.repo})

    async def call(self, name, params, context=None, agent="billing"):
        return await self.manager.call(name, params, context or CONTEXT, caller_agent=agent)

    async def ready_case(self, expenses=None):
        case = await self.repo.get_or_create_active_case("emp-001", "conv-1")
        case.state = ClaimState.AWAITING_CONFIRMATION
        case.facts = {"trip_city": "上海", "trip_date": "2026-10-20", "project_code": "P1"}
        case.expenses = expenses or [{"type": "hotel", "amount": 700.0}, {"type": "transport", "amount": 100.0}]
        return await self.repo.save_case(case)

    async def test_submission_requires_trusted_confirmation(self):
        case = await self.ready_case()
        result = await self.call("submit_claim", {"case_id": case.case_id, "idempotency_key": "k1"}, {**CONTEXT, "confirmed": False})
        self.assertFalse(result.success)
        self.assertIn("确认", result.error)
        self.assertEqual(await self.repo.remaining_budget("engineering"), 3000.0)

    async def test_submission_is_idempotent_and_reserves_budget_once(self):
        case = await self.ready_case()
        params = {"case_id": case.case_id, "idempotency_key": "k1"}
        first = await self.call("submit_claim", params, {**CONTEXT, "confirmed": True})
        second = await self.call("submit_claim", params, {**CONTEXT, "confirmed": True})
        self.assertTrue(first.success and second.success, (first.error, second.error))
        self.assertEqual(first.data, second.data)
        self.assertTrue(first.data["claim_no"].startswith("CLM-"))
        self.assertEqual(await self.repo.remaining_budget("engineering"), 2200.0)
        stored = await self.repo.get_case(case.case_id)
        self.assertEqual(stored.state, ClaimState.SUBMITTED)
        self.assertTrue(stored.budget_reserved)

    async def test_permission_schema_limit_and_budget_rejections(self):
        case = await self.ready_case([{"type": "hotel", "amount": 900.0}])
        denied = await self.call("get_case", {"case_id": case.case_id}, agent="technical")
        missing = await self.call("get_case", {})
        validation = await self.call("validate_claim", {"case_id": case.case_id})
        self.assertFalse(denied.success)
        self.assertFalse(missing.success)
        self.assertFalse(validation.data["valid"])
        self.assertIn("hotel_over_limit", [issue["code"] for issue in validation.data["issues"]])

        case.expenses = [{"type": "transport", "amount": 3200.0}]
        await self.repo.save_case(case)
        insufficient = await self.call("submit_claim", {"case_id": case.case_id, "idempotency_key": "k2"}, {**CONTEXT, "confirmed": True})
        self.assertFalse(insufficient.success)
        self.assertIn("预算", insufficient.error)
        self.assertEqual(await self.repo.remaining_budget("engineering"), 3000.0)

    async def test_read_tools_and_draft_workflow(self):
        policy = await self.call("search_policy", {"city": "上海"})
        self.assertEqual(policy.data["hotel_limit"], 800.0)
        draft = await self.call("create_draft", {"facts": {"trip_city": "上海", "trip_date": "2026-10-20", "project_code": "P1"}})
        self.assertTrue(draft.success, draft.error)
        case_id = draft.data["case_id"]
        added = await self.call("add_expense_items", {"case_id": case_id, "items": [{"type": "hotel", "amount": 700.0}]})
        self.assertTrue(added.success, added.error)
        validation = await self.call("validate_claim", {"case_id": case_id})
        self.assertTrue(validation.data["valid"])
        self.assertEqual(validation.data["total_amount"], 700.0)
        self.assertEqual((await self.call("get_case", {"case_id": case_id})).data["state"], "awaiting_confirmation")

    async def test_idempotent_result_cannot_be_read_by_another_user(self):
        case = await self.ready_case()
        params = {"case_id": case.case_id, "idempotency_key": "k1"}
        self.assertTrue((await self.call("submit_claim", params, {**CONTEXT, "confirmed": True})).success)
        other = await self.call("submit_claim", params, {"user_id": "other", "conv_id": "conv-1", "confirmed": True})
        self.assertFalse(other.success)

    async def test_payment_requires_finance_review(self):
        case = await self.ready_case()
        await self.call("submit_claim", {"case_id": case.case_id, "idempotency_key": "k1"}, {**CONTEXT, "confirmed": True})
        finance = {"user_id": "finance-001", "conv_id": "conv-1", "agent_type": "billing"}
        payment = await self.call("simulate_payment", {"case_id": case.case_id}, finance)
        self.assertFalse(payment.success)
        reviewed = await self.call("review_claim", {"case_id": case.case_id, "approved": True}, finance)
        self.assertTrue(reviewed.success)
        paid = await self.call("simulate_payment", {"case_id": case.case_id}, finance)
        self.assertTrue(paid.success)
        self.assertEqual(paid.data["state"], "paid")

    async def test_concurrent_distinct_keys_submit_only_once(self):
        case = await self.ready_case()
        async def submit(key):
            return await self.call("submit_claim", {"case_id": case.case_id, "idempotency_key": key},
                                   {**CONTEXT, "confirmed": True})
        first, second = await asyncio.gather(submit("a"), submit("b"))
        self.assertEqual(sum((first.success, second.success)), 1)
        self.assertEqual(await self.repo.remaining_budget("engineering"), 2200.0)

    async def test_cached_case_and_budget_do_not_cross_trusted_identity(self):
        case = await self.ready_case()
        for name in ("get_case", "check_budget"):
            first = await self.call(name, {"case_id": case.case_id})
            other_user = await self.call(name, {"case_id": case.case_id},
                                         {"user_id": "other", "conv_id": "conv-1"})
            other_conversation = await self.call(name, {"case_id": case.case_id},
                                                 {"user_id": "emp-001", "conv_id": "other"})
            self.assertTrue(first.success)
            self.assertFalse(other_user.success)
            self.assertFalse(other_conversation.success)

    async def test_rejection_refunds_once_and_new_submission_reserves_once(self):
        case = await self.ready_case()
        submit_context = {**CONTEXT, "confirmed": True}
        first = await self.call("submit_claim", {"case_id": case.case_id, "idempotency_key": "first"}, submit_context)
        self.assertTrue(first.success)
        self.assertEqual(await self.repo.remaining_budget("engineering"), 2200.0)
        reviewer = {"user_id": "manager-001", "conv_id": "conv-1"}
        rejected = await self.call("review_claim", {"case_id": case.case_id, "approved": False}, reviewer)
        duplicate = await self.call("review_claim", {"case_id": case.case_id, "approved": False}, reviewer)
        self.assertTrue(rejected.success)
        self.assertFalse(duplicate.success)
        self.assertEqual(await self.repo.remaining_budget("engineering"), 3000.0)
        self.assertFalse((await self.repo.get_case(case.case_id)).budget_reserved)
        validated = await self.call("validate_claim", {"case_id": case.case_id})
        self.assertTrue(validated.data["valid"])
        second = await self.call("submit_claim", {"case_id": case.case_id, "idempotency_key": "second"}, submit_context)
        self.assertTrue(second.success)
        self.assertEqual(await self.repo.remaining_budget("engineering"), 2200.0)
        self.assertNotEqual(first.data["claim_no"], second.data["claim_no"])

    async def test_reusing_original_key_after_rejection_reserves_again(self):
        case = await self.ready_case()
        params = {"case_id": case.case_id, "idempotency_key": "same"}
        submit_context = {**CONTEXT, "confirmed": True}
        first = await self.call("submit_claim", params, submit_context)
        self.assertTrue(first.success)
        rejected = await self.call("review_claim", {"case_id": case.case_id, "approved": False},
                                   {"user_id": "manager-001", "conv_id": "conv-1"})
        self.assertTrue(rejected.success)
        validated = await self.call("validate_claim", {"case_id": case.case_id})
        self.assertTrue(validated.data["valid"])
        second = await self.call("submit_claim", params, submit_context)
        self.assertTrue(second.success)
        self.assertNotEqual(first.data["claim_no"], second.data["claim_no"])
        self.assertEqual(await self.repo.remaining_budget("engineering"), 2200.0)
        current = await self.repo.get_case(case.case_id)
        self.assertTrue(current.budget_reserved)
        self.assertEqual(current.state, ClaimState.SUBMITTED)

    async def test_concurrent_approval_and_rejection_cannot_restore_stale_approved_case(self):
        case = await self.ready_case()
        submitted = await self.call("submit_claim", {"case_id": case.case_id, "idempotency_key": "first"},
                                    {**CONTEXT, "confirmed": True})
        self.assertTrue(submitted.success)
        original_save = self.repo.save_case
        approval_at_save = asyncio.Event()
        release_approval = asyncio.Event()

        async def delayed_save(snapshot):
            if snapshot.validation.get("review_approved"):
                approval_at_save.set()
                await release_approval.wait()
            return await original_save(snapshot)

        self.repo.save_case = delayed_save
        reviewer = {"user_id": "manager-001", "conv_id": "conv-1"}
        approval = asyncio.create_task(self.call("review_claim", {"case_id": case.case_id, "approved": True}, reviewer))
        try:
            try:
                await asyncio.wait_for(approval_at_save.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass  # A locked transition never calls save_case with an approval snapshot.
            rejected = await self.call("review_claim", {"case_id": case.case_id, "approved": False}, reviewer)
        finally:
            release_approval.set()
        await approval
        self.assertTrue(rejected.success)
        current = await self.repo.get_case(case.case_id)
        self.assertEqual(current.state, ClaimState.REWORK_REQUIRED)
        self.assertFalse(current.budget_reserved)
        self.assertFalse(current.validation.get("review_approved", False))
        self.assertEqual(await self.repo.remaining_budget("engineering"), 3000.0)

    async def test_concurrent_payment_and_rejection_cannot_mark_refunded_case_paid(self):
        case = await self.ready_case()
        await self.call("submit_claim", {"case_id": case.case_id, "idempotency_key": "first"},
                        {**CONTEXT, "confirmed": True})
        finance = {"user_id": "finance-001", "conv_id": "conv-1"}
        self.assertTrue((await self.call("review_claim", {"case_id": case.case_id, "approved": True}, finance)).success)
        original_save = self.repo.save_case
        payment_at_save = asyncio.Event()
        release_payment = asyncio.Event()

        async def delayed_save(snapshot):
            if snapshot.state == ClaimState.PAID:
                payment_at_save.set()
                await release_payment.wait()
            return await original_save(snapshot)

        self.repo.save_case = delayed_save
        payment = asyncio.create_task(self.call("simulate_payment", {"case_id": case.case_id}, finance))
        try:
            try:
                await asyncio.wait_for(payment_at_save.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass  # A locked payment transition does not call save_case.
            await self.call("review_claim", {"case_id": case.case_id, "approved": False}, finance)
        finally:
            release_payment.set()
        await payment
        current = await self.repo.get_case(case.case_id)
        remaining = await self.repo.remaining_budget("engineering")
        self.assertFalse(current.state == ClaimState.PAID and remaining == 3000.0)
        self.assertEqual(current.budget_reserved, remaining == 2200.0)

    async def test_nonfinite_expenses_cannot_change_budget(self):
        for amount in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(amount=amount):
                repo = InMemoryReimbursementRepository(budgets={"engineering": 3000.0})
                manager = MCPToolManager(api_key="test")
                discover_agent_tools(manager, dependencies={"reimbursement_repository": repo})
                case = await repo.get_or_create_active_case("emp-001", "conv-1")
                case.state = ClaimState.AWAITING_CONFIRMATION
                case.facts = {"trip_city": "上海", "trip_date": "2026-10-20", "project_code": "P1"}
                case.expenses = [{"type": "transport", "amount": amount}]
                await repo.save_case(case)
                validation = await manager.call("validate_claim", {"case_id": case.case_id}, CONTEXT, caller_agent="billing")
                self.assertFalse(validation.data["valid"])
                self.assertIn("invalid_amount", [issue["code"] for issue in validation.data["issues"]])
                submission = await manager.call("submit_claim", {"case_id": case.case_id, "idempotency_key": "x"},
                                                {**CONTEXT, "confirmed": True}, caller_agent="billing")
                self.assertFalse(submission.success)
                self.assertTrue(math.isfinite(await repo.remaining_budget("engineering")))
                self.assertEqual(await repo.remaining_budget("engineering"), 3000.0)

    async def test_repository_rejects_nonfinite_validator_amount(self):
        case = await self.ready_case()
        for amount in (float("nan"), float("inf")):
            with self.subTest(amount=amount):
                with self.assertRaises(ValueError):
                    await self.repo.submit_claim(case.case_id, str(amount), "emp-001", "conv-1",
                                                 lambda _: {"valid": True, "required_budget": amount})
                self.assertEqual(await self.repo.remaining_budget("engineering"), 3000.0)


class CacheContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_identity_scopes_cache_and_absent_context_still_hits(self):
        manager = MCPToolManager(api_key="test")
        calls = 0

        async def handler(params, context):
            nonlocal calls
            calls += 1
            return {"caller": (context or {}).get("user_id"), "call": calls}

        manager.register(Tool("identity_probe", "identity cache probe", handler,
                              {"type": "object", "properties": {}, "required": []},
                              cache_ttl=60, allowed_agents=frozenset({"billing"})))
        first = await manager.call("identity_probe", {}, {"user_id": "a", "conv_id": "1"}, caller_agent="billing")
        same = await manager.call("identity_probe", {}, {"user_id": "a", "conv_id": "1"}, caller_agent="billing")
        other_user = await manager.call("identity_probe", {}, {"user_id": "b", "conv_id": "1"}, caller_agent="billing")
        other_conv = await manager.call("identity_probe", {}, {"user_id": "a", "conv_id": "2"}, caller_agent="billing")
        absent = await manager.call("identity_probe", {}, caller_agent="billing")
        absent_again = await manager.call("identity_probe", {}, caller_agent="billing")
        self.assertFalse(first.cached)
        self.assertTrue(same.cached)
        self.assertFalse(other_user.cached)
        self.assertFalse(other_conv.cached)
        self.assertFalse(absent.cached)
        self.assertTrue(absent_again.cached)
        self.assertEqual(calls, 4)


if __name__ == "__main__":
    unittest.main()
