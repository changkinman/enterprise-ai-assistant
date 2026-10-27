"""Safety boundaries from the final whole-branch review."""
import asyncio
import unittest
from types import SimpleNamespace

from agents.agent_orchestrator import Request
from core.reimbursement_models import ClaimState, ReimbursementAction, ReimbursementDecision
from tests.test_reimbursement_harness import HarnessTests, DecisionProvider, request
from tests.test_reimbursement_orchestration import make_orchestrator, billing_task
from core.task_planner import ExecutionMode, TaskPlan


FACTS = {"trip_city": "上海", "trip_date": "2026-10-20", "project_code": "P1",
         "expenses": [{"type": "hotel", "amount": 600, "receipt_present": True}]}


class FinalSafetyTests(HarnessTests):
    async def ready(self, facts=None):
        return await super().ready(facts or FACTS)

    async def test_false_model_confirmation_cannot_authorize_submission(self):
        await self.ready()
        for message in ("好的", "不要确认提交", "我问的是如何确认提交", "确认提交吗？"):
            result = await self.decide({}, ReimbursementAction.CONFIRM_SUBMIT,
                                       confirmed=True, message=message)
            self.assertEqual(result.case.state, ClaimState.AWAITING_CONFIRMATION)
            self.assertFalse(any(t.tool_name == "submit_claim" for t in result.tool_calls))
        self.assertEqual(await self.repo.remaining_budget("engineering"), 3000)

    async def test_planner_evidence_cannot_replace_original_confirmation_message(self):
        await self.ready()
        provider = DecisionProvider(ReimbursementDecision({}, ReimbursementAction.CONFIRM_SUBMIT, True))
        orchestrator = make_orchestrator(self.harness)
        orchestrator._pool[next(k for k in orchestrator._pool if k.value == "billing")][0].decide_reimbursement = provider.decide_reimbursement
        task = billing_task()
        from dataclasses import replace
        task = replace(task, evidence="确认提交这笔报销")
        await orchestrator.run_plan(Request("好的", "emp-001", "conv-1"),
                                    TaskPlan(ExecutionMode.SINGLE, tasks=(task,)))
        case = await self.repo.get_or_create_active_case("emp-001", "conv-1")
        self.assertEqual(case.state, ClaimState.AWAITING_CONFIRMATION)

    async def test_hotel_alias_and_unknown_category_do_not_bypass_validation(self):
        for kind, expected in (("酒店", "hotel_over_limit"), ("HOTEL", "hotel_over_limit"),
                               ("住宿", "hotel_over_limit"), ("made-up", "invalid_expense_type")):
            with self.subTest(kind=kind):
                result = await self.decide({**FACTS, "expenses": [
                    {"type": kind, "amount": 900, "receipt_present": True}]}, ReimbursementAction.VALIDATE)
                self.assertEqual(result.case.state, ClaimState.REWORK_REQUIRED)
                self.assertIn(expected, [i["code"] for i in result.case.validation["issues"]])
                if kind != "made-up":
                    self.assertEqual(result.case.expenses[0]["type"], "hotel")

    async def test_missing_receipt_requires_rework_then_recovers_same_case(self):
        result = await self.decide({**FACTS, "expenses": [
            {"type": "hotel", "amount": 600, "receipt_present": False}]}, ReimbursementAction.VALIDATE)
        self.assertEqual(result.case.state, ClaimState.REWORK_REQUIRED)
        self.assertIs(result.case.expenses[0]["receipt_present"], False)
        self.assertIn("missing_receipt", [i["code"] for i in result.case.validation["issues"]])
        fixed = await self.ready()
        self.assertEqual(fixed.case.case_id, result.case.case_id)
        self.assertIs(fixed.case.expenses[0]["receipt_present"], True)

    async def test_tool_validator_canonicalizes_and_requires_receipts(self):
        from tools.reimbursement_tool import _validate
        case = await self.repo.get_or_create_active_case("emp-001", "conv-1")
        case.facts = {k: v for k, v in FACTS.items() if k != "expenses"}
        for item, code in (({"type": "酒店", "amount": 900, "receipt_present": True}, "hotel_over_limit"),
                           ({"type": "unknown", "amount": 10, "receipt_present": True}, "invalid_expense_type"),
                           ({"type": "hotel", "amount": 10}, "missing_receipt")):
            case.expenses = [item]
            result = _validate(case)
            self.assertFalse(result["valid"])
            self.assertIn(code, [i["code"] for i in result["issues"]])

    async def test_all_writes_require_case_and_idempotency_key(self):
        for name in ("create_draft", "add_expense_items", "validate_claim", "submit_claim",
                     "review_claim", "simulate_payment"):
            schema = self.manager._tools[name].schema
            self.assertTrue({"case_id", "idempotency_key"} <= set(schema["required"]), name)

    async def test_concurrent_replays_do_not_duplicate_writes_or_rewind_state(self):
        case = await self.repo.get_or_create_active_case("emp-001", "conv-1")
        owner = {"user_id": "emp-001", "conv_id": "conv-1"}
        reviewer = {"user_id": "manager-001", "conv_id": "conv-1"}
        finance = {"user_id": "finance-001", "conv_id": "conv-1"}
        async def call(name, extra, context):
            return await self.manager.call(name, {"case_id": case.case_id,
                "idempotency_key": name, **extra}, context, caller_agent="billing", use_cache=False)
        writes = [
            ("create_draft", {"facts": {k: v for k, v in FACTS.items() if k != "expenses"}}, owner),
            ("add_expense_items", {"items": FACTS["expenses"]}, owner),
            ("validate_claim", {}, owner),
            ("submit_claim", {}, {**owner, "confirmed": True}),
            ("review_claim", {"approved": True}, reviewer),
            ("simulate_payment", {}, finance),
        ]
        originals = []
        for name, extra, context in writes:
            results = await asyncio.gather(*(call(name, extra, context) for _ in range(5)))
            self.assertTrue(all(r.success for r in results), (name, [r.error for r in results]))
            self.assertTrue(all(r.data == results[0].data for r in results))
            originals.append(results[0].data)
        before = await self.repo.get_case(case.case_id)
        for (name, extra, context), original in zip(writes, originals):
            replay = await call(name, extra, context)
            self.assertTrue(replay.success, replay.error)
            self.assertEqual(replay.data, original)
        after = await self.repo.get_case(case.case_id)
        self.assertEqual(before, after)
        self.assertEqual(after.state, ClaimState.PAID)
        self.assertEqual(len(after.expenses), 1)
        self.assertEqual(await self.repo.remaining_budget("engineering"), 2400)
