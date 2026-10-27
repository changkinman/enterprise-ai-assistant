import sys
import unittest
from types import SimpleNamespace


class _DummyAsyncAnthropic:
    def __init__(self, *args, **kwargs):
        pass


sys.modules.setdefault("anthropic", SimpleNamespace(AsyncAnthropic=_DummyAsyncAnthropic))

from core.intent_recognizer import FunnelDecisionKind, IntentRecognizer


class FunnelRoutingTests(unittest.TestCase):
    def setUp(self):
        # 漏斗前两层必须是本地、确定性的，不依赖模型客户端。
        self.recognizer = IntentRecognizer.__new__(IntentRecognizer)

    def test_exact_greeting_returns_direct_reply(self):
        decision = self.recognizer.funnel_decision("你好")

        self.assertEqual(decision.kind, FunnelDecisionKind.DIRECT_REPLY)
        self.assertTrue(decision.reply)

    def test_clear_single_domain_uses_lightweight_path(self):
        decision = self.recognizer.funnel_decision("VPN 连不上，公司网络怎么处理？")

        self.assertEqual(decision.kind, FunnelDecisionKind.LIGHTWEIGHT)
        self.assertEqual(decision.analysis.candidates[0].domain.value, "technical")

    def test_multiple_business_domains_escalate_to_llm(self):
        decision = self.recognizer.funnel_decision("我的 VPN 连不上，同时本月报销什么时候到账？")

        self.assertEqual(decision.kind, FunnelDecisionKind.ESCALATE)
        self.assertIn("多", decision.reason)

    def test_low_confidence_message_escalates_to_llm(self):
        decision = self.recognizer.funnel_decision("帮我看一下这个情况")

        self.assertEqual(decision.kind, FunnelDecisionKind.ESCALATE)
        self.assertIn("低", decision.reason)


if __name__ == "__main__":
    unittest.main()
