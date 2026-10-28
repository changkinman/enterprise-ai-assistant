"""Previewed reimbursement submission backed by the governed billing tool."""

from core.auth import Actor
from core.operations import PreviewPayload, ValidationResult
from core.reimbursement_models import ClaimState
from core.reimbursement_repository import InMemoryReimbursementRepository
from mcp.tool_manager import MCPToolManager
from tools.reimbursement_tool import _validate


class ReimbursementSubmitOperation:
    def __init__(self, repository: InMemoryReimbursementRepository,
                 tool_manager: MCPToolManager) -> None:
        self.repository = repository
        self.tool_manager = tool_manager

    async def precheck(self, actor: Actor, params: dict) -> ValidationResult:
        case = await self.repository.get_case(params.get("case_id", ""))
        if case is None or case.user_id != actor.id:
            return ValidationResult(False, reason="报销单不存在或无权访问")
        if case.state != ClaimState.AWAITING_CONFIRMATION:
            return ValidationResult(False, reason="报销单尚未等待确认")
        if case.validation.get("valid") is not True:
            return ValidationResult(False, reason="报销单尚未完成校验")
        validation = _validate(case)
        if not validation["valid"]:
            return ValidationResult(False, reason="报销单校验未通过")
        remaining = await self.repository.remaining_budget(case.department)
        if remaining < validation["required_budget"]:
            return ValidationResult(False, reason="部门预算不足")
        return ValidationResult(True, {"case_id": case.case_id, "conv_id": case.conv_id},
                                case.material_version)

    async def build_preview(self, actor: Actor, normalized_params: dict,
                            validation: ValidationResult) -> PreviewPayload:
        case = await self.repository.get_case(normalized_params["case_id"])
        checked = _validate(case)
        remaining = await self.repository.remaining_budget(case.department)
        return PreviewPayload(
            f"报销单待确认：共 {checked['total_amount']} 元，确认后将占用部门预算。",
            {"case_id": case.case_id, "items": case.expenses,
             "total_amount": checked["total_amount"],
             "policy_reference": "DEMO-TRAVEL-2026",
             "remaining_budget": remaining,
             "budget_after_submission": remaining - checked["required_budget"]},
        )

    async def commit(self, actor: Actor, normalized_params: dict,
                     idempotency_key: str) -> dict:
        result = await self.tool_manager.call(
            "submit_claim",
            {"case_id": normalized_params["case_id"], "idempotency_key": idempotency_key},
            {"user_id": actor.id, "conv_id": normalized_params["conv_id"], "confirmed": True},
            caller_agent="billing", executor_authorized=True, use_cache=False,
        )
        if not result.success:
            raise ValueError(result.error or "报销单提交失败")
        return result.data
