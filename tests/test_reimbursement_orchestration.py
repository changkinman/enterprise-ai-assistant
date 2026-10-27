import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock


class DummyAsyncAnthropic:
    def __init__(self, **_kwargs):
        self.messages = SimpleNamespace()


sys.modules.setdefault("anthropic", SimpleNamespace(AsyncAnthropic=DummyAsyncAnthropic))

from agents.agent_orchestrator import (  # noqa: E402
    AgentOrchestrator, AgentResponse, AgentStats, AgentType, BillingAgent, Request,
    ToolCallTrace,
)
from core.reimbursement_models import (  # noqa: E402
    HarnessResult, HarnessToolTrace, ReimbursementAction, ReimbursementCase,
)
from core.task_planner import BusinessDomain, ExecutionMode, PlannedTask, TaskPlan  # noqa: E402


class FakeMessages:
    def __init__(self, content):
        self.content = content
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self.content)


class StubAgent:
    def __init__(self, agent_type, response):
        self.agent_type = agent_type
        self.response = response
        self.stats = AgentStats()
        self.handle = AsyncMock(return_value=response)


def make_orchestrator(harness=None):
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._reimbursement_harness = harness
    orchestrator._pool = {
        AgentType.GENERAL: [StubAgent(AgentType.GENERAL, AgentResponse(AgentType.GENERAL, "通用回复", True))],
        AgentType.TECHNICAL: [StubAgent(AgentType.TECHNICAL, AgentResponse(AgentType.TECHNICAL, "技术回复", True))],
        AgentType.BILLING: [StubAgent(AgentType.BILLING, AgentResponse(AgentType.BILLING, "旧循环回复", True))],
    }
    return orchestrator


def billing_task(task_id="t1"):
    return PlannedTask(task_id, BusinessDomain.BILLING, "报销上海出差费", "报销单", "报销", 0.9)


class BillingDecisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_structured_decision_uses_no_tools_and_known_facts(self):
        content = [{"type": "text", "text": json.dumps({
            "facts": {"trip_city": "上海", "trip_date": "2026-10-20",
                      "project_code": "PRJ-001", "expenses": [{"type": "hotel", "amount": 600}],
                      "secret": "ignored"},
            "action": "validate", "explicit_confirmation": False,
        }, ensure_ascii=False)}]
        messages = FakeMessages(content)
        agent = BillingAgent(SimpleNamespace(messages=messages), "fake")
        req = Request("上海出差住宿费 600 元", "u1", "c1")
        case = ReimbursementCase("case-1", "u1", "c1")

        decision = await agent.decide_reimbursement(req, case)

        self.assertEqual(decision.action, ReimbursementAction.VALIDATE)
        self.assertEqual(decision.facts["trip_city"], "上海")
        self.assertEqual(decision.facts["expenses"], [{"type": "hotel", "amount": 600}])
        self.assertNotIn("secret", decision.facts)
        self.assertFalse(decision.explicit_confirmation)
        self.assertEqual(decision.user_message, req.message)
        self.assertEqual(len(messages.calls), 1)
        self.assertNotIn("tools", messages.calls[0])

    async def test_bad_decision_json_becomes_safe_collect_facts_decision(self):
        agent = BillingAgent(SimpleNamespace(messages=FakeMessages(
            [{"type": "text", "text": "not json"}]
        )), "fake")
        req = Request("好的", "u1", "c1")

        decision = await agent.decide_reimbursement(req, ReimbursementCase("case-1", "u1", "c1"))

        self.assertEqual(decision.facts, {})
        self.assertEqual(decision.action, ReimbursementAction.COLLECT_FACTS)
        self.assertFalse(decision.explicit_confirmation)
        self.assertEqual(decision.user_message, req.message)

    async def test_invalid_decision_shapes_fall_back_safely(self):
        invalid = [
            {"facts": [], "action": "validate", "explicit_confirmation": False},
            {"facts": {}, "action": "submit_claim", "explicit_confirmation": True},
            {"facts": {"expenses": "hotel"}, "action": "validate", "explicit_confirmation": False},
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                agent = BillingAgent(SimpleNamespace(messages=FakeMessages(
                    [{"type": "text", "text": json.dumps(payload)}]
                )), "fake")
                decision = await agent.decide_reimbursement(
                    Request("好的", "u1", "c1"), ReimbursementCase("case-1", "u1", "c1")
                )
                self.assertEqual(decision.action, ReimbursementAction.COLLECT_FACTS)
                self.assertFalse(decision.explicit_confirmation)
                self.assertEqual(decision.facts, {})


class BillingRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_constructor_accepts_harness(self):
        harness = SimpleNamespace(handle=AsyncMock())

        orchestrator = AgentOrchestrator(
            api_key="test-key", client=SimpleNamespace(messages=FakeMessages([])),
            reimbursement_harness=harness,
        )

        self.assertIs(orchestrator._reimbursement_harness, harness)

    async def test_billing_task_uses_harness_not_base_agent_tool_loop(self):
        harness = SimpleNamespace(handle=AsyncMock(return_value=HarnessResult(
            "已创建草稿", ReimbursementCase("case-1", "u1", "c1"), [], True,
        )))
        orchestrator = make_orchestrator(harness)

        result = await orchestrator.run_plan(
            Request("报销上海出差费", "u1", "c1"),
            TaskPlan(ExecutionMode.SINGLE, tasks=(billing_task(),)),
        )

        harness.handle.assert_awaited_once()
        self.assertIs(harness.handle.await_args.args[1], orchestrator._pool[AgentType.BILLING][0])
        orchestrator._pool[AgentType.BILLING][0].handle.assert_not_awaited()
        self.assertEqual(result.response, "已创建草稿")
        self.assertEqual(result.agent_type, AgentType.BILLING)

    async def test_parallel_direct_billing_routes_through_harness_and_adapts_all_trace_fields(self):
        trace = HarnessToolTrace(2, "call-1", "validate_claim", {"case_id": "case-1"},
                                 False, {"valid": False}, "denied", 12.5)
        harness = SimpleNamespace(handle=AsyncMock(return_value=HarnessResult(
            "校验结果", ReimbursementCase("case-1", "u1", "c1"), [trace], True,
        )))
        orchestrator = make_orchestrator(harness)

        result = await orchestrator.run_parallel(
            Request("报销和 VPN", "u1", "c1"), [AgentType.TECHNICAL, AgentType.BILLING]
        )

        harness.handle.assert_awaited_once()
        orchestrator._pool[AgentType.BILLING][0].handle.assert_not_awaited()
        orchestrator._pool[AgentType.TECHNICAL][0].handle.assert_awaited_once()
        self.assertIn("校验结果", result.response)
        self.assertEqual(len(result.tool_calls), 1)
        adapted = result.tool_calls[0]
        self.assertIsInstance(adapted, ToolCallTrace)
        self.assertEqual(vars(adapted), vars(trace))

    async def test_failed_harness_result_falls_back_to_general_without_billing_retry(self):
        trace = HarnessToolTrace(1, "call-1", "submit_claim", {"case_id": "case-1"},
                                 False, None, "temporarily unavailable", 5.0)
        harness = SimpleNamespace(handle=AsyncMock(return_value=HarnessResult(
            "提交失败", ReimbursementCase("case-1", "u1", "c1"), [trace], False,
        )))
        orchestrator = make_orchestrator(harness)

        response = await orchestrator._execute(Request("确认提交", "u1", "c1"), AgentType.BILLING)

        self.assertEqual(response.content, "通用回复")
        self.assertEqual(vars(response.tool_calls[0]), vars(trace))
        harness.handle.assert_awaited_once()
        orchestrator._pool[AgentType.BILLING][0].handle.assert_not_awaited()
        orchestrator._pool[AgentType.GENERAL][0].handle.assert_awaited_once()

    async def test_sequential_billing_task_uses_harness(self):
        harness = SimpleNamespace(handle=AsyncMock(return_value=HarnessResult(
            "报销回复", ReimbursementCase("case-1", "u1", "c1"), [], True,
        )))
        orchestrator = make_orchestrator(harness)
        plan = TaskPlan(ExecutionMode.SEQUENTIAL, tasks=(
            PlannedTask("t1", BusinessDomain.TECHNICAL, "VPN 报错", "VPN", "排查", 0.9),
            PlannedTask("t2", BusinessDomain.BILLING, "报销", "报销单", "处理", 0.9,
                        depends_on=("t1",)),
        ))

        result = await orchestrator.run_plan(Request("VPN 和报销", "u1", "c1"), plan)

        harness.handle.assert_awaited_once()
        orchestrator._pool[AgentType.BILLING][0].handle.assert_not_awaited()
        self.assertIn("报销回复", result.response)

    async def test_without_harness_preserves_old_billing_path(self):
        orchestrator = make_orchestrator()

        response = await orchestrator._execute(Request("报销", "u1", "c1"), AgentType.BILLING)

        self.assertEqual(response.content, "旧循环回复")
        orchestrator._pool[AgentType.BILLING][0].handle.assert_awaited_once()
