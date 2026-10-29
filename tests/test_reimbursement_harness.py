import asyncio
import sys
import unittest
from types import SimpleNamespace
from uuid import uuid4

sys.modules.setdefault("anthropic", SimpleNamespace(AsyncAnthropic=lambda **_: None))

from core.reimbursement_models import ClaimState, ReimbursementAction, ReimbursementDecision, append_event
from core.reimbursement_repository import InMemoryReimbursementRepository
from core.reimbursement_harness import ReimbursementCaseHarness
from core.auth import Actor
from core.operations import OperationVersionConflict
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
    return SimpleNamespace(message=message, user_id="emp-001", conv_id="conv-1", request_id=uuid4().hex)


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

    async def test_valid_claim_creates_preview_without_submit_trace(self):
        result = await self.ready()
        self.assertEqual(result.pending_operation.action_type, "reimbursement_submit")
        self.assertEqual(result.pending_operation.preview.details["total_amount"], 600.0)
        self.assertFalse(any(t.tool_name == "submit_claim" for t in result.tool_calls))
        self.assertEqual(await self.repo.remaining_budget("engineering"), 3000)

    async def test_changed_expense_rejects_old_confirmation_as_stale(self):
        first = await self.ready()
        await self.decide({"expenses": [{"type": "hotel", "amount": 700,
                                          "receipt_present": True}]},
                          ReimbursementAction.CONFIRM_SUBMIT, confirmed=True,
                          message="酒店改成 700，确认提交")
        with self.assertRaises(OperationVersionConflict):
            await self.harness.executor.confirm(
                Actor("emp-001", "employee"), first.pending_operation.operation_id, "submit-1")
        self.assertEqual(await self.repo.remaining_budget("engineering"), 3000)

    async def test_repository_expense_update_invalidates_preview(self):
        ready = await self.ready()
        case = await self.repo.get_case(ready.case.case_id)
        case.expenses[0]["amount"] = 700
        await self.repo.save_case(case)
        with self.assertRaises(OperationVersionConflict):
            await self.harness.executor.confirm(
                Actor("emp-001", "employee"), ready.pending_operation.operation_id,
                "submit-after-repository-edit")

    async def test_edit_between_confirm_precheck_and_commit_cannot_reserve_budget(self):
        ready = await self.ready()
        adapter = self.harness.executor.adapters["reimbursement_submit"]
        original_commit = adapter.commit

        async def edit_before_commit(actor, params, key):
            case = await self.repo.get_case(ready.case.case_id)
            case.expenses[0]["amount"] = 700
            await self.repo.save_case(case)
            return await original_commit(actor, params, key)

        adapter.commit = edit_before_commit
        with self.assertRaises(ValueError):
            await self.harness.executor.confirm(
                Actor("emp-001", "employee"), ready.pending_operation.operation_id,
                "submit-after-precheck")
        case = await self.repo.get_case(ready.case.case_id)
        self.assertEqual(case.state, ClaimState.AWAITING_CONFIRMATION)
        self.assertIsNone(case.claim_no)
        self.assertEqual(await self.repo.remaining_budget("engineering"), 3000)

    async def test_preview_uses_prechecked_expenses_if_case_changes_during_prepare(self):
        ready = await self.ready()
        adapter = self.harness.executor.adapters["reimbursement_submit"]
        actor = Actor("emp-001", "employee")
        validation = await adapter.precheck(actor, {"case_id": ready.case.case_id})
        case = await self.repo.get_case(ready.case.case_id)
        case.expenses[0]["amount"] = 700
        await self.repo.save_case(case)
        preview = await adapter.build_preview(actor, validation.normalized_params, validation)
        self.assertEqual(preview.details["items"][0]["amount"], 600)
        self.assertEqual(preview.details["total_amount"], 600.0)

    async def test_case_version_change_before_locked_reservation_rejects_submit(self):
        ready = await self.ready()
        original_call = self.manager.call

        async def add_event_before_tool(name, params, context=None, **kwargs):
            if name == "submit_claim":
                case = await self.repo.get_case(ready.case.case_id)
                append_event(case, "status_checked", {})
                await self.repo.save_case(case)
            return await original_call(name, params, context, **kwargs)

        self.manager.call = add_event_before_tool
        with self.assertRaises(ValueError):
            await self.harness.executor.confirm(
                Actor("emp-001", "employee"), ready.pending_operation.operation_id,
                "submit-after-status-event")
        case = await self.repo.get_case(ready.case.case_id)
        self.assertEqual(case.state, ClaimState.AWAITING_CONFIRMATION)
        self.assertEqual(await self.repo.remaining_budget("engineering"), 3000)

    async def test_unvalidated_case_cannot_create_submission_preview(self):
        case = await self.repo.get_or_create_active_case("emp-001", "conv-1")
        case.state = ClaimState.AWAITING_CONFIRMATION
        case.facts = {"trip_city": "上海", "trip_date": "2026-10-20", "project_code": "P1"}
        case.expenses = [{"type": "hotel", "amount": 600, "receipt_present": True}]
        await self.repo.save_case(case)
        from core.operations import ActionProposal, ValidationRejected
        with self.assertRaises(ValidationRejected):
            await self.harness.executor.prepare(
                Actor("emp-001", "employee"),
                ActionProposal("reimbursement_submit", {"case_id": case.case_id}))

    async def test_same_confirmation_key_reserves_budget_once(self):
        ready = await self.ready()
        actor = Actor("emp-001", "employee")
        first = await self.harness.executor.confirm(actor, ready.pending_operation.operation_id, "submit-1")
        second = await self.harness.executor.confirm(actor, ready.pending_operation.operation_id, "submit-1")
        self.assertEqual(first, second)
        self.assertEqual(first["state"], "submitted")
        self.assertEqual(await self.repo.remaining_budget("engineering"), 2400)

    async def submitted(self):
        ready = await self.ready()
        await self.harness.executor.confirm(Actor("emp-001", "employee"),
                                            ready.pending_operation.operation_id, uuid4().hex)
        return SimpleNamespace(case=await self.repo.get_case(ready.case.case_id))

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
        reply = await self.decide({}, ReimbursementAction.CONFIRM_SUBMIT, confirmed=True, message="确认提交")
        self.assertEqual(reply.case.state, ClaimState.AWAITING_CONFIRMATION)
        self.assertFalse(any(t.tool_name == "submit_claim" for t in reply.tool_calls))
        await self.harness.executor.confirm(Actor("emp-001", "employee"),
                                            ready.pending_operation.operation_id, "submit-valid")
        submitted = SimpleNamespace(case=await self.repo.get_case(ready.case.case_id))
        self.assertEqual(submitted.case.state, ClaimState.SUBMITTED)
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
        submitted = await self.harness.executor.confirm(
            Actor("emp-001", "employee"), changed.pending_operation.operation_id, "submit-revised")
        self.assertEqual(submitted["state"], "submitted")
        self.assertEqual(await self.repo.remaining_budget("engineering"), 2300)

    async def test_failed_revalidation_after_changed_confirmation_cannot_submit(self):
        ready = await self.ready()
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
        ready = await self.ready()
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
                self.harness.executor.confirm(Actor("emp-001", "employee"),
                                              ready.pending_operation.operation_id, "submit-race"),
                name="submit")
            await asyncio.wait({submit_task}, timeout=0.1)
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
        revalidated = await self.decide({}, ReimbursementAction.VALIDATE)
        resubmitted = await self.harness.executor.confirm(
            Actor("emp-001", "employee"), revalidated.pending_operation.operation_id, "submit-again")
        self.assertEqual(resubmitted["state"], "submitted")
        approved = await self.harness.review_claim(submitted.case.case_id, "manager-001", True)
        self.assertEqual(approved["state"], "submitted")
        paid = await self.harness.simulate_payment(submitted.case.case_id, "finance-001")
        self.assertEqual(paid["state"], "paid")
        self.assertEqual((await self.harness.get_case_view(submitted.case.case_id))["state"], "paid")

    async def test_review_retry_replays_same_request_after_state_and_events_change(self):
        submitted = await self.submitted()
        case_id = submitted.case.case_id
        first = await self.harness.review_claim(case_id, "manager-001", False,
                                                request_id="review-request-1")
        event_count = len((await self.repo.get_case(case_id)).events)
        replay = await self.harness.review_claim(case_id, "manager-001", False,
                                                 request_id="review-request-1")
        self.assertEqual(replay, first)
        self.assertEqual(len((await self.repo.get_case(case_id)).events), event_count)
        self.assertEqual(await self.repo.remaining_budget("engineering"), 3000)

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
