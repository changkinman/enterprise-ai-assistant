"""Run synthetic reimbursement workflows through the real, process-local Harness."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# MCPToolManager imports the Anthropic SDK, but this benchmark never creates a client.
try:
    import anthropic  # noqa: F401
except ImportError:
    sys.modules["anthropic"] = SimpleNamespace(AsyncAnthropic=lambda **_: None)

from core.reimbursement_harness import ReimbursementCaseHarness
from core.reimbursement_models import ReimbursementAction, ReimbursementDecision
from core.reimbursement_repository import InMemoryReimbursementRepository
from mcp.tool_manager import MCPToolManager
from tools.loader import discover_agent_tools


DEFAULT_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "reimbursement_harness_cases.json"


class QueueDecisionProvider:
    def __init__(self, decisions: list[ReimbursementDecision]) -> None:
        self._decisions = list(decisions)

    async def decide_reimbursement(self, _request: Any, _case: Any) -> ReimbursementDecision:
        if not self._decisions:
            raise AssertionError("fixture exhausted its decision queue")
        return self._decisions.pop(0)


class ReimbursementHarnessBenchmark:
    def __init__(self, fixtures_path: str | Path = DEFAULT_FIXTURES) -> None:
        self.fixtures_path = Path(fixtures_path)

    async def _case(self, fixture: dict[str, Any]) -> dict[str, Any]:
        repository = InMemoryReimbursementRepository(
            budgets={"engineering": fixture.get("budget", 3000)})
        manager = MCPToolManager(api_key="offline-benchmark")
        discover_agent_tools(manager, dependencies={"reimbursement_repository": repository})
        harness = ReimbursementCaseHarness(repository, manager)
        decisions = [ReimbursementDecision(step.get("facts", {}), ReimbursementAction(step["action"]),
                                           explicit_confirmation=step.get("explicit_confirmation", False))
                     for step in fixture["steps"] if "action" in step]
        provider = QueueDecisionProvider(decisions)
        request = SimpleNamespace(message="离线报销测试", user_id="emp-001",
                                  conv_id=fixture["id"], request_id=fixture["id"])
        traces = []
        step_traces: list[list[Any]] = []
        case = None
        for step in fixture["steps"]:
            if "review" in step:
                if case is None:
                    raise AssertionError("review requires a submitted case")
                await harness.review_claim(case.case_id, "manager-001", step["review"])
                case = await repository.get_case(case.case_id)
                step_traces.append([])
                continue
            disabled = step.get("disable_tool")
            original = manager._tools.get(disabled) if disabled else None
            if disabled:
                manager.unregister(disabled)
            try:
                result = await harness.handle(request, provider)
            finally:
                if original is not None:
                    manager.register(original)
            case = result.case
            traces.extend(result.tool_calls)
            step_traces.append(result.tool_calls)
        if case is None:
            raise AssertionError("fixture must have at least one decision")
        stored = await repository.get_case(case.case_id)
        expected = fixture["expected"]
        names = [trace.tool_name for trace in traces]
        events = stored.events
        event_counts = {kind: sum(event.event_type == kind for event in events)
                        for kind in expected.get("event_counts", {})}
        stats = harness.stats()
        checks = {
            "task_completion": stored.state.value == expected["state"],
            "tool_presence": all(name in names or any(
                event.event_type == "tool_call" and event.detail.get("tool") == name
                for event in events) for name in expected.get("tools_present", [])),
            "tool_absence": all(name not in names and not any(
                event.event_type == "tool_call" and event.detail.get("tool") == name
                for event in events) for name in expected.get("tools_absent", [])),
            "event_counts": all(event_counts[kind] == count
                                for kind, count in expected.get("event_counts", {}).items()),
            "stats": all(stats[key] == count for key, count in expected.get("stats", {}).items()),
        }
        submits = [trace for trace in traces if trace.tool_name == "submit_claim"]
        explicit = sum(step.get("explicit_confirmation") is True for step in fixture["steps"])
        checks["confirmation_safety"] = (
            len(submits) <= explicit
            and all(not any(trace.tool_name == "submit_claim" for trace in step_trace)
                    for step, step_trace in zip(fixture["steps"], step_traces)
                    if step.get("action") == "confirm_submit"
                    and step.get("explicit_confirmation") is not True)
        )
        failed = [trace for trace in traces if not trace.success]
        checks["recovery"] = (
            stats["tool_failure_recoveries"] >= len(failed)
            and (not failed or any(trace.success and trace.tool_name == failed[0].tool_name
                                   for trace in traces[traces.index(failed[0]) + 1:]))
        )
        checks["policy_consistency"] = all(
            any(previous.tool_name == "search_policy" and previous.success
                for previous in traces[:index])
            for index, trace in enumerate(traces) if trace.tool_name == "validate_claim"
        ) and not (stored.validation.get("valid") is False and stored.state.value == "submitted")
        event_tool_ids = [event.detail.get("tool_call_id") for event in events
                          if event.event_type == "tool_call"]
        checks["trace_completeness"] = (
            all(trace.tool_call_id in event_tool_ids and trace.tool_call_id
                for trace in traces)
            and len(event_tool_ids) == len(set(event_tool_ids))
            and all(event.detail.get("tool") and isinstance(event.detail.get("success"), bool)
                    for event in events if event.event_type == "tool_call")
        )
        return {"id": fixture["id"], "passed": all(checks.values()), "checks": checks,
                "final_state": stored.state.value, "tool_calls": names,
                "event_counts": event_counts, "stats": stats}

    async def run(self) -> dict[str, Any]:
        fixtures = json.loads(self.fixtures_path.read_text(encoding="utf-8"))
        cases = [await self._case(fixture) for fixture in fixtures]
        total = len(cases)

        def rate(key: str) -> float:
            return sum(case["checks"][key] for case in cases) / total if total else 0.0

        return {"total": total, "passed": sum(case["passed"] for case in cases),
                "task_completion_rate": rate("task_completion"),
                "confirmation_safety_rate": rate("confirmation_safety"),
                "recovery_rate": rate("recovery"),
                "policy_consistency_rate": rate("policy_consistency"),
                "trace_completeness_rate": rate("trace_completeness"), "cases": cases}


def main() -> None:
    print(json.dumps(asyncio.run(ReimbursementHarnessBenchmark().run()),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
