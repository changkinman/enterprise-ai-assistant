import unittest

from core.reimbursement_models import ClaimState
from core.reimbursement_repository import InMemoryReimbursementRepository


class ReimbursementRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_conversation_reuses_one_open_case(self):
        repo = InMemoryReimbursementRepository()
        first = await repo.get_or_create_active_case("emp-001", "conv-1")
        second = await repo.get_or_create_active_case("emp-001", "conv-1")
        self.assertEqual(first.case_id, second.case_id)
        self.assertEqual(first.state, ClaimState.COLLECTING_INFO)

    async def test_idempotency_result_is_stored_by_exact_key(self):
        repo = InMemoryReimbursementRepository()
        await repo.save_idempotent_result("case:submit:0", {"claim_no": "CLM-001"})
        self.assertEqual(
            await repo.get_idempotent_result("case:submit:0"),
            {"claim_no": "CLM-001"},
        )

    async def test_closed_case_is_not_reused_for_a_new_active_conversation(self):
        repo = InMemoryReimbursementRepository()
        case = await repo.get_or_create_active_case("emp-001", "conv-1")
        case.state = ClaimState.CLOSED
        await repo.save_case(case)
        replacement = await repo.get_or_create_active_case("emp-001", "conv-1")
        self.assertNotEqual(replacement.case_id, case.case_id)

    async def test_save_increments_version_once_and_reads_are_defensive(self):
        repo = InMemoryReimbursementRepository()
        case = await repo.get_or_create_active_case("emp-001", "conv-1")
        case.facts["merchant"] = "caller mutation"
        stored = await repo.get_case(case.case_id)
        self.assertNotIn("merchant", stored.facts)

        stored.state = ClaimState.DRAFT_CREATED
        saved = await repo.save_case(stored)
        self.assertEqual(saved.version, 1)
        reread = await repo.get_case(case.case_id)
        self.assertEqual(reread.version, 1)
        self.assertEqual(reread.state, ClaimState.DRAFT_CREATED)

    async def test_seeded_engineering_budget_is_available(self):
        repo = InMemoryReimbursementRepository()
        self.assertGreater(await repo.remaining_budget("engineering"), 0)


if __name__ == "__main__":
    unittest.main()
