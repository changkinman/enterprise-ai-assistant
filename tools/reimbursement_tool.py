"""Deterministic, process-local reimbursement tools for the billing Agent."""

from dataclasses import asdict
import math
from typing import Any

from core.reimbursement_models import ClaimState, ReimbursementCase, canonical_expense_type
from core.reimbursement_repository import InMemoryReimbursementRepository, EMPLOYEES
from mcp.tool_manager import Tool

HOTEL_LIMITS = {"上海": 800.0, "北京": 800.0, "其他": 600.0}
REQUIRED_FACTS = ("trip_city", "trip_date", "project_code")


def _validate(case: ReimbursementCase) -> dict[str, Any]:
    issues = []
    for fact in REQUIRED_FACTS:
        if not case.facts.get(fact):
            issues.append({"code": f"missing_{fact}", "message": f"缺少{fact}"})
    city = str(case.facts.get("trip_city", "其他"))
    limit = HOTEL_LIMITS.get(city, HOTEL_LIMITS["其他"])
    total = 0.0
    for item in case.expenses:
        kind = canonical_expense_type(item.get("type"))
        if kind is None:
            issues.append({"code": "invalid_expense_type", "message": "费用类型不受支持"})
            continue
        amount = item.get("amount")
        if (not isinstance(amount, (int, float)) or isinstance(amount, bool)
                or not math.isfinite(amount) or amount <= 0):
            issues.append({"code": "invalid_amount", "message": "费用金额必须大于零"})
            continue
        total += float(amount)
        if item.get("receipt_present") is not True:
            issues.append({"code": "missing_receipt", "message": "缺少费用票据"})
        if kind == "hotel" and amount > limit:
            issues.append({"code": "hotel_over_limit", "message": f"酒店费用超过{city}标准{limit}元"})
    if not case.expenses:
        issues.append({"code": "missing_expenses", "message": "缺少费用明细"})
    if not math.isfinite(total):
        issues.append({"code": "invalid_amount", "message": "费用总额必须是有限正数"})
        total = 0.0
    total = round(total, 2)
    return {"valid": not issues, "issues": issues,
            "total_amount": total, "required_budget": total}


def build_tools(repository: InMemoryReimbursementRepository) -> list[Tool]:
    """Bind the shared repository to governed Tool definitions."""
    async def owned_case(params, context):
        context = context or {}
        case = await repository.get_case(params["case_id"])
        if (case is None or case.user_id != context.get("user_id")
                or case.conv_id != context.get("conv_id")):
            raise ValueError("报销单不存在或无权访问")
        return case

    async def search_policy(params, context):
        city = params["city"]
        return {"city": city, "hotel_limit": HOTEL_LIMITS.get(city, HOTEL_LIMITS["其他"]),
                "hotel_limits": dict(HOTEL_LIMITS), "policy_reference": "DEMO-TRAVEL-2026"}

    async def get_case(params, context):
        case = await owned_case(params, context)
        result = asdict(case)
        result["state"] = case.state.value
        return result

    async def check_budget(params, context):
        case = await owned_case(params, context)
        return {"department": case.department,
                "remaining_budget": await repository.remaining_budget(case.department)}

    async def create_draft(params, context):
        context = context or {}
        if not context.get("user_id") or not context.get("conv_id"):
            raise ValueError("缺少可信用户或会话上下文")
        def mutate(case):
            if case.state not in {ClaimState.COLLECTING_INFO, ClaimState.DRAFT_CREATED, ClaimState.REWORK_REQUIRED}:
                raise ValueError("当前状态不能创建草稿")
            case.facts.update(params.get("facts", {})); case.state = ClaimState.DRAFT_CREATED
            return {"case_id": case.case_id, "state": case.state.value}
        return await repository.atomic_write(params["case_id"], "create_draft", params["idempotency_key"], mutate,
            payload={"facts": params.get("facts", {})}, owner=(context["user_id"], context["conv_id"]))

    async def add_expense_items(params, context):
        case = await owned_case(params, context)
        items = params["items"]
        if not items:
            raise ValueError("费用明细不能为空")
        def mutate(current):
            if current.state not in {ClaimState.DRAFT_CREATED, ClaimState.REWORK_REQUIRED}:
                raise ValueError("当前状态不能添加费用")
            current.expenses.extend(items); current.validation = {}
            return {"case_id": current.case_id, "expense_count": len(current.expenses)}
        return await repository.atomic_write(case.case_id, "add_expense_items", params["idempotency_key"], mutate,
            payload={"items": items}, owner=(context["user_id"], context["conv_id"]))

    async def validate_claim(params, context):
        case = await owned_case(params, context)
        def mutate(current):
            if current.state not in {ClaimState.DRAFT_CREATED, ClaimState.VALIDATING,
                                     ClaimState.AWAITING_CONFIRMATION, ClaimState.REWORK_REQUIRED}:
                raise ValueError("当前状态不能校验")
            result = _validate(current); current.validation = result
            current.state = ClaimState.AWAITING_CONFIRMATION if result["valid"] else ClaimState.REWORK_REQUIRED
            return result
        return await repository.atomic_write(case.case_id, "validate_claim", params["idempotency_key"], mutate,
            owner=(context["user_id"], context["conv_id"]))

    async def submit_claim(params, context):
        context = context or {}
        if context.get("confirmed") is not True:
            raise ValueError("请先由用户确认提交报销单")
        if not context.get("user_id") or not context.get("conv_id"):
            raise ValueError("缺少可信用户或会话上下文")
        return await repository.submit_claim(
            params["case_id"], params["idempotency_key"],
            context["user_id"], context["conv_id"], _validate,
        )

    async def review_claim(params, context):
        context = context or {}
        role = EMPLOYEES.get(context.get("user_id"), {}).get("role")
        if role not in {"manager", "finance"}:
            raise ValueError("只有经理或财务可以审核")
        return await repository.review_claim(params["case_id"], params["approved"], params["idempotency_key"])

    async def simulate_payment(params, context):
        context = context or {}
        if EMPLOYEES.get(context.get("user_id"), {}).get("role") != "finance":
            raise ValueError("只有财务可以模拟付款")
        return await repository.mark_paid(params["case_id"], params["idempotency_key"])

    handlers = {
        "search_policy": (search_policy, {"city": "string"}, ["city"], 60.0),
        "get_case": (get_case, {"case_id": "string"}, ["case_id"], 1.0),
        "check_budget": (check_budget, {"case_id": "string"}, ["case_id"], 1.0),
        "create_draft": (create_draft, {"case_id": "string", "idempotency_key": "string", "facts": "object"}, ["case_id", "idempotency_key"], 0.0),
        "add_expense_items": (add_expense_items, {"case_id": "string", "idempotency_key": "string", "items": "array"}, ["case_id", "idempotency_key", "items"], 0.0),
        "validate_claim": (validate_claim, {"case_id": "string", "idempotency_key": "string"}, ["case_id", "idempotency_key"], 0.0),
        "submit_claim": (submit_claim, {"case_id": "string", "idempotency_key": "string"}, ["case_id", "idempotency_key"], 0.0),
        "review_claim": (review_claim, {"case_id": "string", "idempotency_key": "string", "approved": "boolean"}, ["case_id", "idempotency_key", "approved"], 0.0),
        "simulate_payment": (simulate_payment, {"case_id": "string", "idempotency_key": "string"}, ["case_id", "idempotency_key"], 0.0),
    }
    return [Tool(name=name, description=f"企业演示报销流程：{name}", handler=handler,
                 schema={"type": "object", "properties": {key: {"type": kind} for key, kind in fields.items()},
                         "required": required, "additionalProperties": False},
                 cache_ttl=ttl, allowed_agents=frozenset({"billing"}),
                 executor_only=name == "submit_claim")
            for name, (handler, fields, required, ttl) in handlers.items()]
