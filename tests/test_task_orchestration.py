import sys
import unittest
from types import SimpleNamespace


class DummyAsyncAnthropic:
    def __init__(self, **_kwargs):
        self.messages = SimpleNamespace()


sys.modules.setdefault("anthropic", SimpleNamespace(AsyncAnthropic=DummyAsyncAnthropic))

from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    AgentStats,
    AgentType,
    Request,
)
from core.intent_recognizer import FunnelDecision, FunnelDecisionKind
from core.task_planner import BusinessDomain, DomainAnalysis, DomainCandidate, ExecutionMode, PlannedTask, TaskPlan


class StubAgent:
    def __init__(self, agent_type, content):
        self.agent_type = agent_type
        self.content = content
        self.stats = AgentStats()

    async def handle(self, _request):
        return AgentResponse(self.agent_type, self.content, True)


class TaskOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    def _orchestrator(self):
        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
        orchestrator._pool = {
            AgentType.GENERAL: [StubAgent(AgentType.GENERAL, "制度答复")],
            AgentType.TECHNICAL: [StubAgent(AgentType.TECHNICAL, "VPN 排查答复")],
            AgentType.BILLING: [StubAgent(AgentType.BILLING, "报销答复")],
        }
        return orchestrator

    async def test_parallel_plan_runs_each_task_and_returns_numbered_answers(self):
        plan = TaskPlan(
            ExecutionMode.PARALLEL,
            tasks=(
                PlannedTask("t1", BusinessDomain.TECHNICAL, "VPN 登录不上", "VPN", "排查", 0.91),
                PlannedTask("t2", BusinessDomain.BILLING, "报销单被退回", "报销单", "查询", 0.88),
            ),
        )

        result = await self._orchestrator().run_plan(
            Request("VPN 登录不上，同时报销单被退回", "u1", "c1"), plan
        )

        self.assertEqual(result.execution_mode, ExecutionMode.PARALLEL.value)
        self.assertIn("1. VPN 排查答复", result.response)
        self.assertIn("2. 报销答复", result.response)

    async def test_clarification_plan_does_not_run_an_agent(self):
        plan = TaskPlan(
            ExecutionMode.CLARIFY,
            clarification_question="这是系统报错还是审核退回？",
        )

        result = await self._orchestrator().run_plan(Request("报销退回", "u1", "c1"), plan)

        self.assertEqual(result.execution_mode, ExecutionMode.CLARIFY.value)
        self.assertEqual(result.response, "这是系统报错还是审核退回？")

    async def test_direct_reply_plan_does_not_run_an_agent(self):
        plan = TaskPlan(ExecutionMode.DIRECT, direct_response="你好！请告诉我需要处理什么问题。")

        result = await self._orchestrator().run_plan(Request("你好", "u1", "c1"), plan)

        self.assertEqual(result.execution_mode, ExecutionMode.DIRECT.value)
        self.assertEqual(result.response, "你好！请告诉我需要处理什么问题。")

    async def test_lightweight_funnel_skips_llm_analysis_and_task_decomposition(self):
        analysis = DomainAnalysis((
            DomainCandidate(BusinessDomain.TECHNICAL, 0.85, "VPN"),
            DomainCandidate(BusinessDomain.BILLING, 0.0, ""),
            DomainCandidate(BusinessDomain.ACCOUNT, 0.0, ""),
            DomainCandidate(BusinessDomain.GENERAL, 0.0, ""),
        ))

        class FakeRecognizer:
            def funnel_decision(self, _message):
                return FunnelDecision(FunnelDecisionKind.LIGHTWEIGHT, analysis)

            async def analyze_domains(self, *_args, **_kwargs):
                raise AssertionError("轻量请求不应调用 LLM 候选分析")

        class FakePlanner:
            def decide(self, message, candidates, tasks):
                self.message = message
                self.candidates = candidates
                self.tasks = tasks
                return TaskPlan(ExecutionMode.SINGLE, (
                    PlannedTask("t1", BusinessDomain.TECHNICAL, "VPN", "VPN", "排查", 0.85),
                ))

            async def plan(self, *_args, **_kwargs):
                raise AssertionError("轻量请求不应调用 LLM 任务拆分")

        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
        orchestrator._intent_recognizer = FakeRecognizer()
        orchestrator._task_planner = FakePlanner()

        plan = await orchestrator.plan_request(Request("VPN 登录失败", "u1", "c1"))

        self.assertEqual(plan.mode, ExecutionMode.SINGLE)
        self.assertEqual(orchestrator._task_planner.tasks, ())

    async def test_pending_clarification_is_combined_with_the_follow_up(self):
        class FakeRecognizer:
            async def analyze_domains(self, message, history):
                self.message = message
                self.history = history
                return "analysis"

        class FakePlanner:
            async def plan(self, message, analysis, history):
                self.message = message
                self.analysis = analysis
                self.history = history
                return TaskPlan(ExecutionMode.SINGLE, (
                    PlannedTask("t1", BusinessDomain.BILLING, "审核退回", "报销单", "查询原因", 0.9),
                ))

        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
        orchestrator._intent_recognizer = FakeRecognizer()
        orchestrator._task_planner = FakePlanner()
        request = Request("是审核退回", "u1", "c1")

        plan = await orchestrator.plan_request(
            request,
            pending_clarification={"original_message": "报销系统退回了，怎么办？"},
        )

        self.assertEqual(plan.mode, ExecutionMode.SINGLE)
        self.assertIn("报销系统退回了，怎么办？", orchestrator._intent_recognizer.message)
        self.assertIn("用户补充：是审核退回", orchestrator._task_planner.message)


if __name__ == "__main__":
    unittest.main()
