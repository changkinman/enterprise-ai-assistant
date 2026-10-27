import asyncio
import sys
import unittest
from types import SimpleNamespace

sys.modules.setdefault("anthropic", SimpleNamespace(AsyncAnthropic=lambda **_: None))

from core.reimbursement_models import ClaimState, ReimbursementAction, ReimbursementDecision
from core.reimbursement_repository import InMemoryReimbursementRepository
from core.reimbursement_harness import ReimbursementCaseHarness
from mcp.tool_manager import MCPToolManager, Tool
from tools.loader import discover_agent_tools


FACTS = {"trip_city": "上海", "trip_date": "2026-10-20", "project_code": "P1",
         "expenses": [{"type": "hotel", "amount": 600, "receipt_present": True}]}


class DecisionProvider:
    def __init__(self, *decisions):
        self.decisions = list(decisions)

    async def decide_reimbursement(self, _req, _case):
        return self.decisions.pop(0)


def request(message="报销"):
    return SimpleNamespace(message=message, user_id="emp-001", conv_id="conv-1", request_id="req-1")


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repo = InMemoryReimbursementRepository(budgets={"engineering": 3000})
        self.manager = MCPToolManager(api_key="test")
        discover_agent_tools(self.manager, dependencies={"reimbursement_repository": self.repo})
        self.harness = ReimbursementCaseHarness(self.repo, self.manager)

    async def decide(self, facts, action=ReimbursementAction.COLLECT_FACTS, *, confirmed=False, message="报销"):
        return await self.harness.handle(request(message), DecisionProvider(
            ReimbursementDecision(facts, action, explicit_confirmation=confirmed)))

    async def ready(self, facts=None):
        result = await self.decide(facts or FACTS, ReimbursementAction.VALIDATE)
        self.assertEqual(result.case.state, ClaimState.AWAITING_CONFIRMATION, result.content)
        return result

    async def submitted(self):
        await self.ready()
        result = await self.decide({}, ReimbursementAction.CONFIRM_SUBMIT, confirmed=True, message="确认提交")
        self.assertEqual(result.case.state, ClaimState.SUBMITTED, result.content)
        return result

    async def test_missing_project_code_keeps_case_and_asks_one_question(self):
        result = await self.decide({"trip_city": "上海", "trip_date": "2026-10-20",
                                    "expenses": [{"type": "hotel", "amount": 600, "receipt_present": True}]})
        self.assertIn("项目", result.content)
        self.assertEqual(result.content.count("？"), 1)
        self.assertEqual(result.case.state, ClaimState.COLLECTING_INFO)
        self.assertEqual(result.case.facts["trip_city"], "上海")
        self.assertEqual(result.case.expenses[0]["amount"], 600)
        self.assertEqual(self.harness.stats()["clarifications"], 1)
        self.assertFalse(result.tool_calls)

    async def test_valid_flow_requires_confirmation_and_traces_governed_tools(self):
        ready = await self.ready()
        self.assertEqual([t.tool_name for t in ready.tool_calls],
                         ["create_draft", "add_expense_items", "search_policy", "validate_claim",
                          "check_budget"])
        self.assertTrue(all(t.success and t.tool_call_id for t in ready.tool_calls))
        ambiguous = await self.decide({}, ReimbursementAction.CONFIRM_SUBMIT, message="好的")
        self.assertEqual(ambiguous.case.state, ClaimState.AWAITING_CONFIRMATION)
        self.assertFalse(any(t.tool_name == "submit_claim" for t in ambiguous.tool_calls))
        submitted = await self.decide({}, ReimbursementAction.CONFIRM_SUBMIT, confirmed=True, message="确认提交")
        self.assertEqual(submitted.case.state, ClaimState.SUBMITTED)
        self.assertEqual([t.tool_name for t in submitted.tool_calls], ["submit_claim"])
        self.assertEqual(self.harness.stats()["confirmations"], 1)
        self.assertEqual(self.harness.stats()["completed"], 1)
        self.assertEqual(await self.repo.remaining_budget("engineering"), 2400)
        self.assertTrue(any(e.event_type == "state_transition" and e.detail["to"] == "submitted"
                            for e in submitted.case.events))
        reached = {e.detail["to"] for e in submitted.case.events if e.event_type == "state_transition"}
        self.assertTrue({"draft_created", "validating", "awaiting_confirmation", "submitted"} <= reached)

    async def test_changed_expense_in_confirmation_requires_new_validation_and_confirmation(self):
        await self.ready()
        changed = await self.decide({"expenses": [{"type": "hotel", "amount": 700, "receipt_present": True}]},
                                    ReimbursementAction.CONFIRM_SUBMIT, confirmed=True,
                                    message="酒店改成 700，确认提交")
        self.assertEqual(changed.case.state, ClaimState.AWAITING_CONFIRMATION)
        self.assertEqual(changed.case.expenses, [{"type": "hotel", "amount": 700, "receipt_present": True}])
        self.assertIn("700", changed.content)
        self.assertFalse(any(trace.tool_name == "submit_claim" for trace in changed.tool_calls))
        self.assertEqual(await self.repo.remaining_budget("engineering"), 3000)
        submitted = await self.decide({}, ReimbursementAction.CONFIRM_SUBMIT, confirmed=True,
                                      message="确认提交这笔报销")
        self.assertEqual(submitted.case.state, ClaimState.SUBMITTED)
        self.assertEqual(await self.repo.remaining_budget("engineering"), 2300)

    async def test_failed_revalidation_after_changed_confirmation_cannot_submit(self):
        await self.ready()
        self.manager.unregister("search_policy")
        changed = await self.decide({"expenses": [{"type": "hotel", "amount": 700, "receipt_present": True}]},
                                    ReimbursementAction.CONFIRM_SUBMIT, confirmed=True)
        self.assertFalse(changed.success)
        self.assertEqual(changed.case.state, ClaimState.REWORK_REQUIRED)
        blocked = await self.decide({}, ReimbursementAction.CONFIRM_SUBMIT, confirmed=True)
        self.assertFalse(blocked.success)
        self.assertFalse(any(trace.tool_name == "submit_claim" for trace in blocked.tool_calls))
        self.assertEqual(await self.repo.remaining_budget("engineering"), 3000)

    async def test_concurrent_status_cannot_overwrite_submission(self):
        await self.ready()
        status_at_save = asyncio.Event()
        release_status = asyncio.Event()
        original_save = self.repo.save_case
        status_task = None
        held = False

        async def hold_status_save(case):
            nonlocal held
            if asyncio.current_task() is status_task and not held:
                held = True
                status_at_save.set()
                await release_status.wait()
            return await original_save(case)

        self.repo.save_case = hold_status_save
        try:
            status_task = asyncio.create_task(
                self.decide({}, ReimbursementAction.GET_STATUS), name="status")
            await asyncio.wait_for(status_at_save.wait(), timeout=1)
            submit_task = asyncio.create_task(
                self.decide({}, ReimbursementAction.CONFIRM_SUBMIT, confirmed=True,
                            message="确认提交"), name="submit")
            await asyncio.wait({submit_task}, timeout=0.1)
            self.assertFalse(submit_task.done(), "submission must wait for the same-case status update")
            release_status.set()
            await asyncio.gather(status_task, submit_task)
        finally:
            release_status.set()
            self.repo.save_case = original_save
        stored = await self.repo.get_case(status_task.result().case.case_id)
        self.assertEqual(stored.state, ClaimState.SUBMITTED)
        self.assertTrue(stored.budget_reserved)
        self.assertEqual(await self.repo.remaining_budget("engineering"), 2400)

    async def test_invalid_hotel_limit_reworks_and_can_replace_expenses(self):
        bad = await self.decide({**FACTS, "expenses": [{"type": "hotel", "amount": 900, "receipt_present": True}]},
                                ReimbursementAction.VALIDATE)
        self.assertEqual(bad.case.state, ClaimState.REWORK_REQUIRED)
        self.assertIn("酒店", bad.content)
        fixed = await self.decide({"expenses": [{"type": "hotel", "amount": 700, "receipt_present": True}]},
                                  ReimbursementAction.VALIDATE)
        self.assertEqual(fixed.case.state, ClaimState.AWAITING_CONFIRMATION, fixed.content)
        self.assertEqual(fixed.case.expenses, [{"type": "hotel", "amount": 700, "receipt_present": True}])

    async def test_insufficient_budget_reworks_before_confirmation(self):
        result = await self.decide({**FACTS, "expenses": [{"type": "transport", "amount": 3200, "receipt_present": True}]},
                                   ReimbursementAction.VALIDATE)
        self.assertEqual(result.case.state, ClaimState.REWORK_REQUIRED)
        self.assertIn("预算", result.content)
        self.assertIn("check_budget", [trace.tool_name for trace in result.tool_calls])
        self.assertFalse(any(trace.tool_name == "submit_claim" for trace in result.tool_calls))

    async def test_manager_failure_keeps_prior_state_and_allows_retry(self):
        original = self.manager._tools["create_draft"]
        self.manager.unregister("create_draft")
        failed = await self.decide(FACTS, ReimbursementAction.VALIDATE)
        self.assertFalse(failed.success)
        self.assertEqual(failed.case.state, ClaimState.COLLECTING_INFO)
        self.assertEqual(failed.tool_calls[0].tool_name, "create_draft")
        self.assertFalse(failed.tool_calls[0].success)
        self.assertTrue(any(e.event_type == "tool_failure" for e in failed.case.events))
        self.assertEqual(self.harness.stats()["tool_failure_recoveries"], 1)
        self.manager.register(original)
        retried = await self.decide({}, ReimbursementAction.COLLECT_FACTS)
        self.assertEqual(retried.case.state, ClaimState.AWAITING_CONFIRMATION, retried.content)

    async def test_review_rejection_reworks_and_payment_requires_approval(self):
        submitted = await self.submitted()
        with self.assertRaises(ValueError):
            await self.harness.simulate_payment(submitted.case.case_id, "finance-001")
        waiting = await self.repo.get_case(submitted.case.case_id)
        self.assertTrue(any(e.event_type == "state_rejection" and e.detail["action"] == "simulate_payment"
                            for e in waiting.events))
        rejected = await self.harness.review_claim(submitted.case.case_id, "manager-001", False)
        self.assertEqual(rejected["state"], "rework_required")
        self.assertEqual(await self.repo.remaining_budget("engineering"), 3000)
        await self.decide({}, ReimbursementAction.VALIDATE)
        resubmitted = await self.decide({}, ReimbursementAction.CONFIRM_SUBMIT, confirmed=True,
                                        message="确认提交这笔报销")
        self.assertEqual(resubmitted.case.state, ClaimState.SUBMITTED)
        approved = await self.harness.review_claim(submitted.case.case_id, "manager-001", True)
        self.assertEqual(approved["state"], "submitted")
        paid = await self.harness.simulate_payment(submitted.case.case_id, "finance-001")
        self.assertEqual(paid["state"], "paid")
        self.assertEqual((await self.harness.get_case_view(submitted.case.case_id))["state"], "paid")

    async def test_governed_review_denial_keeps_submitted_state(self):
        submitted = await self.submitted()
        with self.assertRaises(ValueError):
            await self.harness.review_claim(submitted.case.case_id, "emp-001", False)
        current = await self.repo.get_case(submitted.case.case_id)
        self.assertEqual(current.state, ClaimState.SUBMITTED)
        self.assertTrue(current.budget_reserved)
        self.assertTrue(any(e.event_type == "tool_failure" and e.detail["tool"] == "review_claim"
                            for e in current.events))
        self.assertEqual(self.harness.stats()["tool_failure_recoveries"], 1)

    async def test_terminal_case_rejects_modification_and_records_rejection(self):
        submitted = await self.submitted()
        await self.harness.review_claim(submitted.case.case_id, "manager-001", True)
        await self.harness.simulate_payment(submitted.case.case_id, "finance-001")
        blocked = await self.decide({"project_code": "P2"})
        self.assertFalse(blocked.success)
        self.assertEqual(blocked.case.state, ClaimState.PAID)
        self.assertEqual(blocked.case.facts["project_code"], "P1")
        self.assertTrue(any(e.event_type == "state_rejection" for e in blocked.case.events))


if __name__ == "__main__":
    unittest.main()
