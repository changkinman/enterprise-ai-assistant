import asyncio
import json
import unittest
from types import SimpleNamespace
import sys


class DummyAsyncAnthropic:
    def __init__(self, **kwargs):
        self.messages = SimpleNamespace()


sys.modules.setdefault("anthropic", SimpleNamespace(AsyncAnthropic=DummyAsyncAnthropic))

from agents.agent_orchestrator import (
    AgentResponse,
    AgentStats,
    AgentType,
    BaseAgent,
    GeneralAgent,
    OrchestratorResult,
    Request,
    TechnicalAgent,
    ToolCallTrace,
    AgentOrchestrator,
)
try:
    from api.main import ChatResponse, ToolCallResponse
except ModuleNotFoundError:
    ChatResponse = ToolCallResponse = None
from mcp.tool_manager import MCPToolManager, Tool
from tools.loader import discover_agent_tools


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("fake model response exhausted")
        return SimpleNamespace(content=self.responses.pop(0))


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def manager_with_tools():
    manager = MCPToolManager(api_key="test-key")
    discover_agent_tools(manager)
    return manager


class ToolRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_rerank_empty_order_falls_back_to_retrieval_order(self):
        manager = MCPToolManager(api_key="test-key")
        manager._client = FakeClient([[SimpleNamespace(text="[]")]])
        items = [{"title": "first"}, {"title": "second"}]

        result = await manager._rerank("query", items, 1)

        self.assertEqual(result, [items[0]])

    async def test_discovery_and_agent_ownership(self):
        manager = manager_with_tools()
        self.assertEqual([t.name for t in manager.get_tools_for_agent("billing")], ["billing_lookup"])
        self.assertEqual([t.name for t in manager.get_tools_for_agent("technical")], ["fault_diagnose"])
        self.assertEqual(
            [t.name for t in manager.get_tools_for_agent("general")],
            ["billing_lookup", "fault_diagnose"],
        )
        with self.assertRaises(ValueError):
            discover_agent_tools(manager)

    async def test_mock_results_and_trusted_context(self):
        manager = manager_with_tools()
        bill = await manager.call(
            "billing_lookup", {"billing_no": "BILL-1001"}, caller_agent="billing"
        )
        self.assertTrue(bill.success)
        self.assertTrue(bill.data["mock"])
        self.assertEqual(bill.data["billing_no"], "BILL-1001")

        diagnosis = await manager.call(
            "fault_diagnose",
            {"symptom": "????"},
            {"user_id": "trusted-user"},
            caller_agent="technical",
        )
        self.assertTrue(diagnosis.success)
        self.assertEqual(diagnosis.data["user_id"], "trusted-user")
        self.assertTrue(diagnosis.data["mock"])

    async def test_server_side_permission_check(self):
        manager = manager_with_tools()
        denied = await manager.call(
            "fault_diagnose", {}, {"user_id": "u1"}, caller_agent="billing"
        )
        self.assertFalse(denied.success)
        self.assertIn("fault_diagnose", denied.error)


class AgentToolLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_tool_needed(self):
        client = FakeClient([[{"type": "text", "text": "??"}]])
        agent = GeneralAgent(client, "fake-model", manager_with_tools())
        response = await agent.handle(Request("??", "u1", "c1"))
        self.assertEqual(response.content, "??")
        self.assertEqual(response.tool_calls, [])

    async def test_multiple_tool_rounds(self):
        client = FakeClient([
            [{"type": "tool_use", "id": "call-1", "name": "billing_lookup", "input": {"billing_no": "B-1"}}],
            [{"type": "tool_use", "id": "call-2", "name": "fault_diagnose", "input": {"symptom": "??"}}],
            [{"type": "text", "text": "?????????????????"}],
        ])
        agent = GeneralAgent(client, "fake-model", manager_with_tools())
        response = await agent.handle(Request("?????????", "user-7", "conv-7"))
        self.assertTrue(response.success)
        self.assertEqual([t.round for t in response.tool_calls], [1, 2])
        self.assertEqual([t.tool_name for t in response.tool_calls], ["billing_lookup", "fault_diagnose"])
        self.assertEqual(response.tool_calls[1].result["user_id"], "user-7")

    async def test_same_round_parallel_tools_preserve_order(self):
        client = FakeClient([
            [
                {"type": "tool_use", "id": "a", "name": "fault_diagnose", "input": {}},
                {"type": "tool_use", "id": "b", "name": "billing_lookup", "input": {"billing_no": "B-2"}},
            ],
            [{"type": "text", "text": "????"}],
        ])
        agent = GeneralAgent(client, "fake-model", manager_with_tools())
        response = await agent.handle(Request("????", "u2", "c2"))
        self.assertEqual([t.tool_call_id for t in response.tool_calls], ["a", "b"])
        tool_result_message = client.messages.calls[1]["messages"][-1]
        self.assertEqual([b["tool_use_id"] for b in tool_result_message["content"]], ["a", "b"])

    async def test_unknown_tool_becomes_error_result(self):
        client = FakeClient([
            [{"type": "tool_use", "id": "x", "name": "missing_tool", "input": {}}],
            [{"type": "text", "text": "?????"}],
        ])
        agent = GeneralAgent(client, "fake-model", manager_with_tools())
        response = await agent.handle(Request("???????", "u", "c"))
        self.assertFalse(response.tool_calls[0].success)
        self.assertIn("missing_tool", response.tool_calls[0].error)

    async def test_max_rounds_forces_text_without_tools(self):
        client = FakeClient([
            [{"type": "tool_use", "id": "1", "name": "fault_diagnose", "input": {}}],
            [{"type": "tool_use", "id": "2", "name": "fault_diagnose", "input": {}}],
            [{"type": "text", "text": "?????????"}],
        ])
        agent = GeneralAgent(client, "fake-model", manager_with_tools(), max_tool_rounds=2)
        response = await agent.handle(Request("????", "u", "c"))
        self.assertEqual(len(response.tool_calls), 2)
        self.assertEqual(response.content, "?????????")
        self.assertNotIn("tools", client.messages.calls[-1])


class StubAgent:
    def __init__(self, response):
        self.response = response
        self.stats = AgentStats()

    async def handle(self, req):
        return self.response


class OrchestratorTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_agent_traces_are_merged(self):
        trace_a = ToolCallTrace(1, "a", "billing_lookup", {}, True, {"mock": True})
        trace_b = ToolCallTrace(1, "b", "fault_diagnose", {}, True, {"mock": True})
        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
        orchestrator._pool = {
            AgentType.BILLING: [StubAgent(AgentResponse(AgentType.BILLING, "??", True, tool_calls=[trace_a]))],
            AgentType.TECHNICAL: [StubAgent(AgentResponse(AgentType.TECHNICAL, "??", True, tool_calls=[trace_b]))],
        }
        req = Request("????", "u", "c")
        result = await orchestrator.run_parallel(req, [AgentType.BILLING, AgentType.TECHNICAL])
        self.assertEqual([t.tool_call_id for t in result.tool_calls], ["a", "b"])

    @unittest.skipIf(ChatResponse is None, "API dependencies not installed")
    async def test_chat_response_serializes_trace(self):
        response = ChatResponse(
            conv_id="c",
            response="ok",
            intent="query",
            agent_type="general",
            escalated=False,
            latency_ms=1.0,
            tool_calls=[ToolCallResponse(
                round=1,
                tool_call_id="id",
                tool_name="billing_lookup",
                arguments={"billing_no": "B-1"},
                success=True,
                result={"mock": True},
            )],
        )
        payload = response.model_dump()
        self.assertEqual(payload["tool_calls"][0]["tool_name"], "billing_lookup")


if __name__ == "__main__":
    unittest.main()
