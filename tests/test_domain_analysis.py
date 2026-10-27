import unittest
from types import MethodType

from core.task_planner import BusinessDomain, DomainCandidate


class DomainAnalysisTests(unittest.IsolatedAsyncioTestCase):
    async def test_domain_analysis_keeps_multiple_credible_domains(self):
        from core.intent_recognizer import IntentRecognizer

        recognizer = IntentRecognizer.__new__(IntentRecognizer)

        async def fake_llm(_self, _message, _history):
            return {
                BusinessDomain.TECHNICAL: DomainCandidate(BusinessDomain.TECHNICAL, 0.90, "VPN 登录不上"),
                BusinessDomain.BILLING: DomainCandidate(BusinessDomain.BILLING, 0.84, "报销单被退回了"),
                BusinessDomain.ACCOUNT: DomainCandidate(BusinessDomain.ACCOUNT, 0.02, ""),
                BusinessDomain.GENERAL: DomainCandidate(BusinessDomain.GENERAL, 0.01, ""),
            }

        recognizer._llm_domain_candidates = MethodType(fake_llm, recognizer)
        recognizer._pattern_domain_scores = lambda _message: {
            BusinessDomain.TECHNICAL: 0.0,
            BusinessDomain.BILLING: 0.0,
            BusinessDomain.ACCOUNT: 0.0,
            BusinessDomain.GENERAL: 0.0,
        }
        recognizer._urgency = lambda _message, _intent: None

        analysis = await recognizer.analyze_domains("VPN 登录不上，同时报销单被退回了")

        scores = {candidate.domain: candidate.score for candidate in analysis.candidates}
        self.assertGreaterEqual(scores[BusinessDomain.TECHNICAL], 0.90)
        self.assertGreaterEqual(scores[BusinessDomain.BILLING], 0.84)
        self.assertEqual(analysis.candidates[0].domain, BusinessDomain.TECHNICAL)


if __name__ == "__main__":
    unittest.main()
