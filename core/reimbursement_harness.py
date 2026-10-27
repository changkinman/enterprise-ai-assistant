"""Deterministic case workflow around governed reimbursement tools."""

import asyncio
from copy import deepcopy
import math
from typing import Any
from uuid import uuid4

from core.reimbursement_models import (
    ClaimState, HarnessResult, HarnessToolTrace, ReimbursementAction,
    ReimbursementCase, ReimbursementDecision, append_event,
    canonical_expense_type, is_explicit_submission,
)
from core.reimbursement_repository import InMemoryReimbursementRepository
from mcp.tool_manager import MCPToolManager


REQUIRED_FACTS = ("trip_city", "trip_date", "project_code")
ALLOWED_ACTIONS = {
    ClaimState.COLLECTING_INFO: {ReimbursementAction.COLLECT_FACTS, ReimbursementAction.VALIDATE,
                                 ReimbursementAction.GET_STATUS},
    ClaimState.DRAFT_CREATED: {ReimbursementAction.COLLECT_FACTS, ReimbursementAction.VALIDATE,
                               ReimbursementAction.GET_STATUS},
    ClaimState.VALIDATING: {ReimbursementAction.VALIDATE, ReimbursementAction.GET_STATUS},
    ClaimState.AWAITING_CONFIRMATION: {ReimbursementAction.CONFIRM_SUBMIT,
                                       ReimbursementAction.COLLECT_FACTS, ReimbursementAction.GET_STATUS},
    ClaimState.SUBMITTED: {ReimbursementAction.GET_STATUS},
    ClaimState.REWORK_REQUIRED: {ReimbursementAction.COLLECT_FACTS, ReimbursementAction.VALIDATE,
                                 ReimbursementAction.GET_STATUS},
}
QUESTIONS = {
    "trip_city": "请问出差城市是哪里？",
    "trip_date": "请问出差日期是哪天？",
    "project_code": "请问本次报销的项目编号是什么？",
    "expenses": "请问费用类型和金额是多少？",
}


class ReimbursementCaseHarness:
    def __init__(self, repository: InMemoryReimbursementRepository,
                 tool_manager: MCPToolManager) -> None:
        self.repository = repository
        self.tool_manager = tool_manager
        self._stats = {"completed": 0, "clarifications": 0, "state_rejections": 0,
                       "tool_failure_recoveries": 0, "confirmations": 0}
        self._conversation_cases: dict[tuple[str, str], str] = {}
        self._case_locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, case_id: str) -> asyncio.Lock:
        return self._case_locks.setdefault(case_id, asyncio.Lock())

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    async def _event(self, case: ReimbursementCase, kind: str,
                     detail: dict[str, Any]) -> ReimbursementCase:
        append_event(case, kind, detail)
        return await self.repository.save_case(case)

    async def _transition(self, case: ReimbursementCase, state: ClaimState,
                          reason: str) -> ReimbursementCase:
        if case.state != state:
            before = case.state
            case.state = state
            case = await self._event(case, "state_transition",
                                     {"from": before.value, "to": state.value, "reason": reason})
        return case

    async def _call(self, case: ReimbursementCase, name: str, arguments: dict[str, Any],
                    context: dict[str, Any], traces: list[HarnessToolTrace]) -> tuple[ReimbursementCase, bool, Any]:
        if name in {"create_draft", "add_expense_items", "validate_claim", "submit_claim",
                    "review_claim", "simulate_payment"}:
            arguments = {"case_id": case.case_id,
                         "idempotency_key": f"{case.case_id}:{name}:{case.version}", **arguments}
        try:
            result = await self.tool_manager.call(name, arguments, context,
                                                  caller_agent="billing", use_cache=False)
            success, data, error, latency = result.success, result.data, result.error, result.latency_ms
        except Exception as exc:
            success, data, error, latency = False, None, str(exc), 0.0
        trace = HarnessToolTrace(round=len(traces) + 1, tool_call_id=uuid4().hex,
                                 tool_name=name, arguments=deepcopy(arguments),
                                 success=success, result=deepcopy(data), error=error,
                                 latency_ms=latency)
        traces.append(trace)
        updated = await self.repository.get_case(case.case_id)
        if updated is not None:
            if updated.state != case.state:
                append_event(updated, "state_transition", {"from": case.state.value,
                             "to": updated.state.value, "reason": name})
            case = updated
        case = await self._event(case, "tool_call", {"tool": name, "success": success,
                                                      "error": error, "tool_call_id": trace.tool_call_id})
        return case, success, data

    async def _failed(self, case: ReimbursementCase, traces: list[HarnessToolTrace],
                      start_state: ClaimState) -> HarnessResult:
        # Earlier draft/validation tool effects are reversible. Keep gathered facts for retry.
        case = await self._transition(case, start_state, "tool_failure_recovery")
        case = await self._event(case, "tool_failure", {"tool": traces[-1].tool_name,
                                                        "error": traces[-1].error})
        self._stats["tool_failure_recoveries"] += 1
        return HarnessResult("工具暂时不可用，报销信息已保留，请稍后重试。", case, traces, False)

    @staticmethod
    def _clean_facts(raw: Any) -> tuple[dict[str, str], list[dict[str, Any]] | None]:
        if not isinstance(raw, dict):
            return {}, None
        facts = {key: value.strip() for key in REQUIRED_FACTS
                 if isinstance((value := raw.get(key)), str) and value.strip()}
        expenses = None
        if isinstance(raw.get("expenses"), list):
            cleaned = []
            for item in raw["expenses"]:
                if not isinstance(item, dict):
                    continue
                kind, amount = item.get("type"), item.get("amount")
                if (isinstance(kind, str) and kind.strip()
                        and isinstance(amount, (int, float)) and not isinstance(amount, bool)
                        and math.isfinite(amount) and amount > 0):
                    cleaned.append({"type": canonical_expense_type(kind) or kind.strip(),
                                    "amount": amount,
                                    "receipt_present": item.get("receipt_present") is True})
            if cleaned:
                expenses = cleaned
        return facts, expenses

    async def handle(self, req: Any, agent: Any) -> HarnessResult:
        conversation = (req.user_id, req.conv_id)
        prior_id = self._conversation_cases.get(conversation)
        case = await self.repository.get_case(prior_id) if prior_id else None
        if case is None:
            case = await self.repository.get_or_create_active_case(*conversation)
        self._conversation_cases[conversation] = case.case_id
        async with self._lock_for(case.case_id):
            current = await self.repository.get_case(case.case_id)
            return await self._handle_locked(req, agent, current)

    async def _handle_locked(self, req: Any, agent: Any,
                             case: ReimbursementCase) -> HarnessResult:
        traces: list[HarnessToolTrace] = []
        start_state = case.state
        decision = await agent.decide_reimbursement(req, deepcopy(case))
        if not isinstance(decision, ReimbursementDecision):
            decision = ReimbursementDecision({}, ReimbursementAction.COLLECT_FACTS)
        try:
            action = ReimbursementAction(decision.action)
        except (ValueError, TypeError):
            action = None
        case = await self._event(case, "decision", {"action": action.value if action else "invalid",
                                                    "explicit_confirmation": decision.explicit_confirmation})
        if action not in ALLOWED_ACTIONS.get(case.state, set()):
            self._stats["state_rejections"] += 1
            case = await self._event(case, "state_rejection", {"action": str(action),
                                                                "state": case.state.value})
            return HarnessResult("当前报销状态不允许这项操作，请查询案件状态。", case, traces, False)

        context = {"user_id": req.user_id, "conv_id": req.conv_id, "agent_type": "billing"}
        if action == ReimbursementAction.GET_STATUS:
            case, ok, data = await self._call(case, "get_case", {"case_id": case.case_id}, context, traces)
            if not ok:
                return await self._failed(case, traces, start_state)
            return HarnessResult(f"报销单状态：{data['state']}。", case, traces, True)

        facts, expenses = self._clean_facts(decision.facts)
        if action == ReimbursementAction.CONFIRM_SUBMIT:
            supplied = decision.facts if isinstance(decision.facts, dict) else {}
            invalid_material = not isinstance(decision.facts, dict)
            invalid_material = invalid_material or any(
                key in supplied and key not in facts for key in REQUIRED_FACTS
            )
            if "expenses" in supplied:
                raw_expenses = supplied["expenses"]
                if (not isinstance(raw_expenses, list) or expenses is None
                        or len(expenses) != len(raw_expenses)):
                    invalid_material = True
            if invalid_material:
                self._stats["state_rejections"] += 1
                case = await self._event(case, "state_rejection", {"reason": "invalid_confirmation_facts"})
                return HarnessResult("修改的信息无效，请重新提供后确认提交。", case, traces, False)
            changed = (any(case.facts.get(key) != value for key, value in facts.items())
                       or (expenses is not None and expenses != case.expenses))
            if changed:
                # A confirmation containing new data confirms neither the old nor the new version.
                action = ReimbursementAction.COLLECT_FACTS
            elif (decision.explicit_confirmation is not True
                  or not is_explicit_submission(getattr(req, "original_message", None)
                                                if getattr(req, "original_message", None) is not None
                                                else req.message)):
                self._stats["state_rejections"] += 1
                case = await self._event(case, "state_rejection", {"reason": "confirmation_not_explicit"})
                return HarnessResult("请明确回复“确认提交这笔报销”后再提交。", case, traces, True)
            else:
                key = f"{case.case_id}:submit:{case.version}"
                case, ok, data = await self._call(case, "submit_claim",
                                                  {"case_id": case.case_id, "idempotency_key": key},
                                                  {**context, "confirmed": True}, traces)
                if not ok:
                    return await self._failed(case, traces, start_state)
                self._stats["confirmations"] += 1
                self._stats["completed"] += 1
                return HarnessResult(f"报销单已提交，单号 {data['claim_no']}。", case, traces, True)

        if facts or expenses is not None:
            case.facts.update(facts)
            if expenses is not None:
                case.expenses = expenses
            case.validation = {}
            case = await self.repository.save_case(case)
            case = await self._event(case, "facts_updated", {"fields": list(facts),
                                                               "expenses": expenses is not None})
            if start_state == ClaimState.AWAITING_CONFIRMATION:
                case = await self._transition(case, ClaimState.REWORK_REQUIRED, "facts_changed")
                # The prior confirmation no longer covers the revised data.
                start_state = ClaimState.REWORK_REQUIRED

        missing = next((key for key in (*REQUIRED_FACTS, "expenses")
                        if not (case.expenses if key == "expenses" else case.facts.get(key))), None)
        if missing:
            self._stats["clarifications"] += 1
            return HarnessResult(QUESTIONS[missing], case, traces, True)

        if case.state in {ClaimState.COLLECTING_INFO, ClaimState.REWORK_REQUIRED}:
            # The expense tool appends, so clear the staged items before using it.
            pending = deepcopy(case.expenses)
            case.expenses = []
            case = await self.repository.save_case(case)
            case, ok, _ = await self._call(case, "create_draft", {"facts": case.facts}, context, traces)
            if not ok:
                case.expenses = pending
                case = await self.repository.save_case(case)
                return await self._failed(case, traces, start_state)
            case, ok, _ = await self._call(case, "add_expense_items",
                                           {"case_id": case.case_id, "items": pending}, context, traces)
            if not ok:
                case.expenses = pending
                case = await self.repository.save_case(case)
                return await self._failed(case, traces, start_state)
        elif expenses is not None and case.state == ClaimState.DRAFT_CREATED:
            pending = deepcopy(case.expenses)
            case.expenses = []
            case = await self.repository.save_case(case)
            case, ok, _ = await self._call(case, "add_expense_items",
                                           {"case_id": case.case_id, "items": pending}, context, traces)
            if not ok:
                case.expenses = pending
                case = await self.repository.save_case(case)
                return await self._failed(case, traces, start_state)

        case, ok, policy = await self._call(case, "search_policy",
                                            {"city": case.facts["trip_city"]}, context, traces)
        if not ok:
            return await self._failed(case, traces, start_state)
        case = await self._transition(case, ClaimState.VALIDATING, "begin_validation")
        case, ok, validation = await self._call(case, "validate_claim",
                                                {"case_id": case.case_id}, context, traces)
        if not ok:
            return await self._failed(case, traces, start_state)
        if not validation["valid"]:
            issues = "；".join(issue["message"] for issue in validation["issues"])
            return HarnessResult(f"报销单需修改：{issues}。制度依据：{policy['policy_reference']}。",
                                 case, traces, True)
        case, ok, budget = await self._call(case, "check_budget",
                                            {"case_id": case.case_id}, context, traces)
        if not ok:
            return await self._failed(case, traces, start_state)
        if budget["remaining_budget"] < validation["required_budget"]:
            case.validation["valid"] = False
            case.validation["issues"].append({"code": "insufficient_budget", "message": "部门预算不足"})
            case = await self.repository.save_case(case)
            case = await self._transition(case, ClaimState.REWORK_REQUIRED, "insufficient_budget")
            return HarnessResult("部门预算不足，请调整费用后重新校验。", case, traces, True)
        return HarnessResult(f"报销单校验通过，共 {validation['total_amount']} 元。请明确确认提交。",
                             case, traces, True)

    async def get_case_view(self, case_id: str) -> dict[str, Any] | None:
        case = await self.repository.get_case(case_id)
        if case is None:
            return None
        amounts = [item.get("amount") for item in case.expenses if isinstance(item, dict)]
        total_amount = sum(
            amount for amount in amounts
            if isinstance(amount, (int, float)) and not isinstance(amount, bool)
            and math.isfinite(amount) and amount > 0
        )
        validation = case.validation
        issues = validation.get("issues")
        return {
            "case_id": case.case_id,
            "state": case.state.value,
            "claim_no": case.claim_no,
            "total_amount": total_amount,
            "validation_summary": {
                "valid": validation.get("valid") if isinstance(validation.get("valid"), bool) else None,
                "issue_count": len(issues) if isinstance(issues, list) else 0,
                "review_approved": validation.get("review_approved") is True,
            },
            "budget_reserved": case.budget_reserved,
            "version": case.version,
            "events": [
                {"at": event.at, "version": event.version, "event_type": event.event_type}
                for event in case.events
            ],
        }

    async def review_claim(self, case_id: str, reviewer_id: str, approved: bool,
                           comment: str = "") -> dict[str, Any]:
        async with self._lock_for(case_id):
            return await self._review_locked(case_id, reviewer_id, approved, comment)

    async def _review_locked(self, case_id: str, reviewer_id: str, approved: bool,
                             comment: str) -> dict[str, Any]:
        case = await self.repository.get_case(case_id)
        if case is None:
            raise ValueError("报销单不存在")
        if case.state != ClaimState.SUBMITTED:
            self._stats["state_rejections"] += 1
            await self._event(case, "state_rejection", {"action": "review_claim",
                                                         "state": case.state.value})
            raise ValueError("只有已提交的报销单可以审核")
        traces: list[HarnessToolTrace] = []
        case, ok, data = await self._call(case, "review_claim",
                                          {"case_id": case_id, "approved": approved},
                                          {"user_id": reviewer_id, "conv_id": case.conv_id}, traces)
        if not ok:
            await self._event(case, "tool_failure", {"tool": "review_claim", "error": traces[-1].error})
            self._stats["tool_failure_recoveries"] += 1
            raise ValueError(traces[-1].error or "审核失败")
        if comment:
            await self._event(case, "review_comment", {"comment": comment})
        return data

    async def simulate_payment(self, case_id: str, finance_user_id: str) -> dict[str, Any]:
        async with self._lock_for(case_id):
            return await self._payment_locked(case_id, finance_user_id)

    async def _payment_locked(self, case_id: str, finance_user_id: str) -> dict[str, Any]:
        case = await self.repository.get_case(case_id)
        if case is None:
            raise ValueError("报销单不存在")
        if case.state != ClaimState.SUBMITTED or not case.validation.get("review_approved"):
            self._stats["state_rejections"] += 1
            await self._event(case, "state_rejection", {"action": "simulate_payment",
                                                         "state": case.state.value})
            raise ValueError("报销单尚未审核通过")
        traces: list[HarnessToolTrace] = []
        case, ok, data = await self._call(case, "simulate_payment", {"case_id": case_id},
                                          {"user_id": finance_user_id, "conv_id": case.conv_id}, traces)
        if not ok:
            await self._event(case, "tool_failure", {"tool": "simulate_payment", "error": traces[-1].error})
            self._stats["tool_failure_recoveries"] += 1
            raise ValueError(traces[-1].error or "付款失败")
        return data
