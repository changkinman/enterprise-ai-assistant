import unittest

from core.task_planner import (
    BusinessDomain,
    DomainCandidate,
    ExecutionMode,
    PlannedTask,
    TaskPlanner,
)


class TaskPlannerDecisionTests(unittest.TestCase):
    def setUp(self):
        self.planner = TaskPlanner()

    def test_independent_cross_domain_tasks_run_in_parallel(self):
        message = "VPN 登录不上，同时报销单被退回了"
        plan = self.planner.decide(
            message=message,
            candidates=[
                DomainCandidate(BusinessDomain.TECHNICAL, 0.91, "VPN 登录不上"),
                DomainCandidate(BusinessDomain.BILLING, 0.88, "报销单被退回了"),
            ],
            tasks=[
                PlannedTask("t1", BusinessDomain.TECHNICAL, "VPN 登录不上", "VPN", "排查登录故障", 0.91),
                PlannedTask("t2", BusinessDomain.BILLING, "报销单被退回了", "报销单", "查询退回原因", 0.88),
            ],
        )

        self.assertIs(plan.mode, ExecutionMode.PARALLEL)
        self.assertEqual([task.task_id for task in plan.tasks], ["t1", "t2"])

    def test_dependent_tasks_run_sequentially(self):
        message = "先查报销单状态，如果被退回再告诉我补什么材料"
        plan = self.planner.decide(
            message=message,
            candidates=[DomainCandidate(BusinessDomain.BILLING, 0.92, message)],
            tasks=[
                PlannedTask("t1", BusinessDomain.BILLING, "查报销单状态", "报销单", "查询状态", 0.92),
                PlannedTask("t2", BusinessDomain.BILLING, "告诉我补什么材料", "报销单", "说明补充材料", 0.87, ("t1",)),
            ],
        )

        self.assertIs(plan.mode, ExecutionMode.SEQUENTIAL)

    def test_overlapping_evidence_requires_clarification(self):
        message = "报销系统退回了，怎么办"
        plan = self.planner.decide(
            message=message,
            candidates=[
                DomainCandidate(BusinessDomain.TECHNICAL, 0.75, "报销系统退回了"),
                DomainCandidate(BusinessDomain.BILLING, 0.78, "报销系统退回了"),
            ],
            tasks=[
                PlannedTask("t1", BusinessDomain.TECHNICAL, "报销系统退回了", "报销系统", "排查系统故障", 0.75),
                PlannedTask("t2", BusinessDomain.BILLING, "报销系统退回了", "报销单", "查询审核退回原因", 0.78),
            ],
        )

        self.assertIs(plan.mode, ExecutionMode.CLARIFY)
        self.assertIn("系统", plan.clarification_question)

    def test_no_credible_candidate_requires_clarification(self):
        plan = self.planner.decide(
            message="帮我处理这个事情",
            candidates=[
                DomainCandidate(BusinessDomain.GENERAL, 0.20, ""),
                DomainCandidate(BusinessDomain.TECHNICAL, 0.18, ""),
            ],
            tasks=[],
        )

        self.assertIs(plan.mode, ExecutionMode.CLARIFY)


if __name__ == "__main__":
    unittest.main()
