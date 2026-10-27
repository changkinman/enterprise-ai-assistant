import asyncio
import sys
import unittest
from types import SimpleNamespace

sys.modules.setdefault("anthropic", SimpleNamespace(AsyncAnthropic=lambda **_: None))

from core.reimbursement_models import ClaimState
from core.reimbursement_repository import InMemoryReimbursementRepository
from mcp.tool_manager import MCPToolManager
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


if __name__ == "__main__":
    unittest.main()
