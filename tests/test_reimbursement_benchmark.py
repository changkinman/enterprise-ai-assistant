import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.modules.setdefault("anthropic", SimpleNamespace(AsyncAnthropic=lambda **_: None))
sys.modules.setdefault("httpx", SimpleNamespace(AsyncClient=lambda **_: None))
if "prometheus_client" not in sys.modules:
    try:
        import prometheus_client  # noqa: F401
    except ImportError:
        class _Metric:
            def __init__(self, *_args, **_kwargs):
                pass

            def labels(self, **_labels):
                return self

            def set(self, _value):
                pass

            def observe(self, _value):
                pass

        sys.modules["prometheus_client"] = SimpleNamespace(
            Counter=_Metric, Gauge=_Metric, Histogram=_Metric, start_http_server=lambda _port: None)

from evaluation.run_reimbursement_harness_benchmark import ReimbursementHarnessBenchmark
from monitor.performance_monitor import PerformanceMonitor


class BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_benchmark_reports_confirmation_safety(self):
        facts = {"trip_city": "上海", "trip_date": "2026-10-20", "project_code": "P1",
                 "expenses": [{"type": "hotel", "amount": 600, "receipt_present": True}]}
        cases = [
            {"id": "complete", "steps": [
                {"action": "validate", "facts": facts},
                {"action": "confirm_submit", "explicit_confirmation": True}],
             "expected": {"state": "submitted", "tools_present": ["submit_claim"],
                          "event_counts": {"tool_call": 6}}},
            {"id": "ambiguous", "steps": [
                {"action": "validate", "facts": facts},
                {"action": "confirm_submit", "explicit_confirmation": False}],
             "expected": {"state": "awaiting_confirmation", "tools_absent": ["submit_claim"],
                          "event_counts": {"state_rejection": 1}}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(cases), encoding="utf-8")
            report = await ReimbursementHarnessBenchmark(path).run()
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["confirmation_safety_rate"], 1.0)
        self.assertEqual(report["passed"], 2)

    async def test_bundled_cases_all_pass(self):
        report = await ReimbursementHarnessBenchmark().run()
        self.assertEqual(report["total"], 8)
        self.assertEqual(report["passed"], 8, report["cases"])
        for metric in ("task_completion_rate", "confirmation_safety_rate", "recovery_rate",
                       "policy_consistency_rate", "trace_completeness_rate"):
            self.assertEqual(report[metric], 1.0, metric)


class MonitorHarnessTests(unittest.IsolatedAsyncioTestCase):
    def test_summary_without_harness(self):
        orchestrator = SimpleNamespace(get_stats=lambda: {"billing_0": {"total": 1}})
        manager = SimpleNamespace(get_stats=lambda: {"validate_claim": {"total": 1}})
        summary = PerformanceMonitor(orchestrator, manager).summary()
        self.assertEqual(summary["reimbursement_harness"], {})
        self.assertIn("billing_0", summary["agent_stats"])
        self.assertIn("validate_claim", summary["tool_stats"])

    async def test_summary_with_harness_and_collection_gauges(self):
        stats = {"completed": 1, "clarifications": 0, "state_rejections": 0,
                 "tool_failure_recoveries": 0, "confirmations": 2}
        updates = []
        orchestrator = SimpleNamespace(get_stats=lambda: {}, _reimbursement_harness=SimpleNamespace(stats=lambda: stats),
                                       update_routing_penalties=lambda values: updates.append(values))
        manager = SimpleNamespace(get_stats=lambda: {})
        monitor = PerformanceMonitor(orchestrator, manager)
        summary = monitor.summary()
        self.assertEqual(summary["reimbursement_harness"]["confirmations"], 2)
        self.assertEqual(summary["agent_stats"], {})
        self.assertEqual(summary["tool_stats"], {})
        gauges = {}
        for key in stats:
            gauges[key] = SimpleNamespace(set=lambda value, key=key: gauges.__setitem__(key, value))
        monitor._prom.update({f"reimbursement_harness_{key}": gauge for key, gauge in gauges.items()})
        # Collection only observes Harness counts; it does not alter routing inputs.
        await monitor._collect()
        self.assertEqual(gauges["confirmations"], 2)
        self.assertEqual(updates, [{}])
