import sys
import unittest
from types import SimpleNamespace


class DummyAsyncAnthropic:
    def __init__(self, **kwargs):
        self.messages = SimpleNamespace()


sys.modules.setdefault("anthropic", SimpleNamespace(AsyncAnthropic=DummyAsyncAnthropic))

from core.intent_recognizer import IntentCategory, IntentRecognizer


class IntentPatternRuleTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = object.__new__(IntentRecognizer)

    def test_billing_overrides_escalation(self):
        result = self.recognizer._pattern_recognize("请转人工处理这笔报销")
        self.assertEqual(result["intent"], IntentCategory.BILLING)

    def test_billing_overrides_request(self):
        result = self.recognizer._pattern_recognize("帮我提交餐费报销")
        self.assertEqual(result["intent"], IntentCategory.BILLING)

    def test_technical_overrides_generic_query_and_request(self):
        for message in ("VPN 权限申请需要找哪个部门？", "帮我开通项目代码仓库权限"):
            with self.subTest(message=message):
                result = self.recognizer._pattern_recognize(message)
                self.assertEqual(result["intent"], IntentCategory.TECHNICAL)

    def test_generic_service_requests_remain_generic(self):
        result = self.recognizer._pattern_recognize("帮我取消 A302 会议室预约")
        self.assertEqual(result["intent"], IntentCategory.REQUEST)

    def test_account_overrides_generic_query(self):
        result = self.recognizer._pattern_recognize("忘记邮箱密码怎么办？")
        self.assertEqual(result["intent"], IntentCategory.ACCOUNT)


if __name__ == "__main__":
    unittest.main()
