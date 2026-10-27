import json
import sys
import unittest
from types import SimpleNamespace

sys.modules.setdefault("anthropic", SimpleNamespace(AsyncAnthropic=lambda **_: None))

from fastapi import HTTPException

from api import main as api
from core.reimbursement_harness import ReimbursementCaseHarness
from core.reimbursement_models import ClaimState, ReimbursementCase
from core.reimbursement_repository import InMemoryReimbursementRepository
from mcp.tool_manager import MCPToolManager
from tools.loader import discover_agent_tools


class ReimbursementApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repository = InMemoryReimbursementRepository()
        manager = MCPToolManager(api_key="test")
        discover_agent_tools(manager, dependencies={"reimbursement_repository": self.repository})
        self.harness = ReimbursementCaseHarness(self.repository, manager)
        api._reimbursement_harness = self.harness

    def tearDown(self):
        api._reimbursement_harness = None

    async def make_case(self, state=ClaimState.COLLECTING_INFO, approved=False):
        case = ReimbursementCase("CASE-1", "emp-001", "conv-1", state=state)
        case.budget_reserved = state == ClaimState.SUBMITTED
        case.validation = {"review_approved": approved}
        case.claim_no = "CLM-1" if state == ClaimState.SUBMITTED else None
        await self.repository.save_case(case)

    async def test_get_case_returns_serializable_view(self):
        await self.make_case()
        payload = await api.get_reimbursement_case("CASE-1")
        self.assertEqual(payload["case_id"], "CASE-1")
        self.assertEqual(payload["state"], "collecting_info")
        json.dumps(payload)

    async def test_get_missing_case_returns_404(self):
        with self.assertRaises(HTTPException) as caught:
            await api.get_reimbursement_case("missing")
        self.assertEqual(caught.exception.status_code, 404)

    async def test_review_requires_manager_role(self):
        await self.make_case(ClaimState.SUBMITTED)
        with self.assertRaises(HTTPException) as caught:
            await api.review_reimbursement_claim(
                "CASE-1", api.ReviewInput(reviewer_id="emp-001", approved=True)
            )
        self.assertEqual(caught.exception.status_code, 403)

    async def test_review_approval_uses_harness(self):
        await self.make_case(ClaimState.SUBMITTED)
        payload = await api.review_reimbursement_claim(
            "CASE-1", api.ReviewInput(reviewer_id="manager-001", approved=True)
        )
        self.assertEqual(payload["case_id"], "CASE-1")
        self.assertTrue((await self.repository.get_case("CASE-1")).validation["review_approved"])

    async def test_review_invalid_state_returns_409(self):
        await self.make_case()
        with self.assertRaises(HTTPException) as caught:
            await api.review_reimbursement_claim(
                "CASE-1", api.ReviewInput(reviewer_id="manager-001", approved=True)
            )
        self.assertEqual(caught.exception.status_code, 409)

    async def test_review_missing_case_returns_404(self):
        with self.assertRaises(HTTPException) as caught:
            await api.review_reimbursement_claim(
                "missing", api.ReviewInput(reviewer_id="manager-001", approved=True)
            )
        self.assertEqual(caught.exception.status_code, 404)

    async def test_payment_requires_finance_role(self):
        await self.make_case(ClaimState.SUBMITTED, approved=True)
        with self.assertRaises(HTTPException) as caught:
            await api.pay_reimbursement_claim(
                "CASE-1", api.PaymentInput(finance_user_id="manager-001")
            )
        self.assertEqual(caught.exception.status_code, 403)

    async def test_payment_requires_approved_submission(self):
        await self.make_case(ClaimState.SUBMITTED)
        with self.assertRaises(HTTPException) as caught:
            await api.pay_reimbursement_claim(
                "CASE-1", api.PaymentInput(finance_user_id="finance-001")
            )
        self.assertEqual(caught.exception.status_code, 409)

    async def test_payment_uses_harness(self):
        await self.make_case(ClaimState.SUBMITTED, approved=True)
        payload = await api.pay_reimbursement_claim(
            "CASE-1", api.PaymentInput(finance_user_id="finance-001")
        )
        self.assertEqual(payload["state"], "paid")
        self.assertEqual((await self.repository.get_case("CASE-1")).state, ClaimState.PAID)

    async def test_payment_missing_case_returns_404(self):
        with self.assertRaises(HTTPException) as caught:
            await api.pay_reimbursement_claim(
                "missing", api.PaymentInput(finance_user_id="finance-001")
            )
        self.assertEqual(caught.exception.status_code, 404)

    def test_chat_response_serializes_ordinary_tool_call(self):
        payload = api.ChatResponse(
            conv_id="conv-1", response="已校验", intent="billing", agent_type="billing",
            escalated=False, latency_ms=1.0,
            tool_calls=[api.ToolCallResponse(
                round=1, tool_call_id="call-1", tool_name="validate_claim",
                arguments={"case_id": "CASE-1"}, success=True, result={"valid": True}
            )],
        ).model_dump()
        self.assertEqual(payload["tool_calls"][0]["tool_name"], "validate_claim")


if __name__ == "__main__":
    unittest.main()
