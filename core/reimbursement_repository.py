"""Async, process-local persistence for reimbursement harness cases."""

import asyncio
import math
from copy import deepcopy
from typing import Any, Callable
from uuid import uuid4

from core.reimbursement_models import ClaimState, ReimbursementCase, append_event


ENGINEERING_BUDGET = 100_000.0
EMPLOYEES: dict[str, dict[str, str]] = {
    "emp-001": {"role": "employee", "department": "engineering"},
    "manager-001": {"role": "manager", "department": "engineering"},
    "finance-001": {"role": "finance", "department": "engineering"},
}


def seeded_budget_for(department: str) -> float:
    """Return the fixed process-local seed budget for a known department."""
    return ENGINEERING_BUDGET if department == "engineering" else 0.0


class InMemoryReimbursementRepository:
    def __init__(self, budgets: dict[str, float] | None = None) -> None:
        self._budgets = deepcopy(
            budgets if budgets is not None else {"engineering": seeded_budget_for("engineering")}
        )
        self._cases: dict[str, ReimbursementCase] = {}
        self._active_cases: dict[tuple[str, str], str] = {}
        self._idempotency_results: dict[str, dict[str, Any]] = {}
        self._write_payloads: dict[tuple[str, str, str], Any] = {}
        self._write_results: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._reserved_amounts: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_active_case(
        self, user_id: str, conv_id: str
    ) -> ReimbursementCase:
        key = (user_id, conv_id)
        async with self._lock:
            existing_id = self._active_cases.get(key)
            existing = self._cases.get(existing_id) if existing_id else None
            if existing is not None and existing.state not in {
                ClaimState.PAID,
                ClaimState.CLOSED,
            }:
                return deepcopy(existing)

            department = EMPLOYEES.get(user_id, {}).get("department", "engineering")
            case = ReimbursementCase(
                case_id=f"RMB-{uuid4().hex[:12].upper()}",
                user_id=user_id,
                conv_id=conv_id,
                department=department,
            )
            self._cases[case.case_id] = case
            self._active_cases[key] = case.case_id
            return deepcopy(case)

    async def get_case(self, case_id: str) -> ReimbursementCase | None:
        async with self._lock:
            case = self._cases.get(case_id)
            return deepcopy(case) if case is not None else None

    async def save_case(self, case: ReimbursementCase) -> ReimbursementCase:
        async with self._lock:
            stored = self._cases.get(case.case_id)
            saved = deepcopy(case)
            if (stored is not None and case.version < stored.version
                    and stored.state in {ClaimState.SUBMITTED, ClaimState.PAID}
                    and case.state == ClaimState.AWAITING_CONFIRMATION):
                # A status/event write may have waited while confirmation committed.
                # Keep the newer business state and add only events from that write.
                saved = deepcopy(stored)
                saved.events.extend(event for event in case.events if event not in stored.events)
            if stored is None:
                # A caller-created case starts at its supplied pre-save version.
                saved.version = case.version + 1
            else:
                saved.material_version = stored.material_version + int(
                    saved.facts != stored.facts or saved.expenses != stored.expenses)
                saved.version = stored.version
                changed = saved != stored
                saved.version = stored.version + int(changed)
            self._cases[case.case_id] = saved
            active_key = (saved.user_id, saved.conv_id)
            active_id = self._active_cases.get(active_key)
            if active_id is None or active_id == saved.case_id:
                self._active_cases[active_key] = saved.case_id
            return deepcopy(saved)

    async def remaining_budget(self, department: str) -> float:
        async with self._lock:
            return float(self._budgets.get(department, 0.0))

    async def get_idempotent_result(self, key: str) -> dict[str, Any] | None:
        async with self._lock:
            result = self._idempotency_results.get(key)
            return deepcopy(result) if result is not None else None

    async def save_idempotent_result(
        self, key: str, result: dict[str, Any]
    ) -> None:
        async with self._lock:
            self._idempotency_results[key] = deepcopy(result)

    async def submit_claim(
        self,
        case_id: str,
        idempotency_key: str,
        user_id: str,
        conv_id: str,
        validator: Callable[[ReimbursementCase], dict[str, Any]],
        *, expected_case_version: int, expected_material_version: int,
    ) -> dict[str, Any]:
        """Validate and submit under one lock so budget and idempotency cannot race."""
        async with self._lock:
            key = f"{case_id}:submit:{idempotency_key}"
            case = self._cases.get(case_id)
            if case is None or case.user_id != user_id or case.conv_id != conv_id:
                raise ValueError("报销单不存在或无权访问")
            prior = self._idempotency_results.get(key)
            if (prior is not None and case.state in {ClaimState.SUBMITTED, ClaimState.PAID}
                    and case.budget_reserved and prior.get("claim_no") == case.claim_no):
                return deepcopy(prior)
            if case.state != ClaimState.AWAITING_CONFIRMATION:
                raise ValueError("报销单尚未等待确认或已经提交")
            validation = validator(case)
            if not validation["valid"]:
                raise ValueError("报销单校验未通过")
            amount = float(validation["required_budget"])
            available = float(self._budgets.get(case.department, 0.0))
            if not math.isfinite(amount) or amount <= 0:
                raise ValueError("报销金额必须是有限正数")
            if not math.isfinite(available):
                raise ValueError("部门预算数据无效")
            if available < amount:
                raise ValueError("部门预算不足")
            if (case.version != expected_case_version
                    or case.material_version != expected_material_version):
                raise ValueError("报销单预览已过期")
            self._budgets[case.department] = available - amount
            self._reserved_amounts[case.case_id] = amount
            case.claim_no = f"CLM-{uuid4().hex[:12].upper()}"
            case.budget_reserved = True
            case.validation = deepcopy(validation)
            case.state = ClaimState.SUBMITTED
            case.version += 1
            append_event(case, "state_transition", {"from": ClaimState.AWAITING_CONFIRMATION.value,
                                                    "to": ClaimState.SUBMITTED.value,
                                                    "reason": "submit_claim"})
            result = {"case_id": case.case_id, "claim_no": case.claim_no,
                      "state": case.state.value, "total_amount": amount,
                      "budget_reserved": True}
            self._idempotency_results[key] = deepcopy(result)
            return deepcopy(result)

    async def atomic_write(
        self, case_id: str, action: str, idempotency_key: str,
        mutate: Callable[[ReimbursementCase], dict[str, Any]], *,
        payload: Any = None, owner: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        """Authenticate, replay or commit the mutation under the same lock."""
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValueError("缺少有效幂等键")
        async with self._lock:
            stored = self._cases.get(case_id)
            if stored is None or (owner is not None and owner != (stored.user_id, stored.conv_id)):
                raise ValueError("报销单不存在或无权访问")
            key = (case_id, action, idempotency_key)
            if key in self._write_results:
                if self._write_payloads[key] != payload:
                    raise ValueError("幂等键已用于不同的请求")
                return deepcopy(self._write_results[key])
            case = deepcopy(stored)
            result = mutate(case)
            case.material_version = stored.material_version + int(
                case.facts != stored.facts or case.expenses != stored.expenses)
            case.version = stored.version + int(case != stored)
            self._cases[case_id] = case
            self._write_payloads[key] = deepcopy(payload)
            self._write_results[key] = deepcopy(result)
            return deepcopy(result)

    async def get_write_result(self, case_id: str, action: str,
                               idempotency_key: str, payload: Any) -> dict[str, Any] | None:
        """Return a completed write only when the retry payload matches exactly."""
        async with self._lock:
            key = (case_id, action, idempotency_key)
            if key not in self._write_results:
                return None
            if self._write_payloads[key] != payload:
                raise ValueError("幂等键已用于不同的请求")
            return deepcopy(self._write_results[key])

    async def review_claim(self, case_id: str, approved: bool, idempotency_key: str) -> dict[str, Any]:
        """Approve or reject using the current stored state under one lock."""
        def mutate(case):
            if case.state != ClaimState.SUBMITTED or not case.budget_reserved:
                raise ValueError("报销单尚未提交或预算已经释放")
            if approved:
                case.validation["review_approved"] = True
            else:
                amount = self._reserved_amounts.pop(case_id)
                self._budgets[case.department] += amount
                case.budget_reserved = False
                case.validation.pop("review_approved", None)
                case.state = ClaimState.REWORK_REQUIRED
            return {"case_id": case.case_id, "approved": approved, "state": case.state.value}
        return await self.atomic_write(case_id, "review_claim", idempotency_key, mutate,
                                       payload={"approved": approved})

    async def mark_paid(self, case_id: str, idempotency_key: str) -> dict[str, Any]:
        """Pay only an approved, still-reserved submission under the lock."""
        def mutate(case):
            if (case.state != ClaimState.SUBMITTED
                    or not case.budget_reserved or not case.validation.get("review_approved")):
                raise ValueError("报销单尚未审核通过")
            case.state = ClaimState.PAID
            return {"case_id": case.case_id, "state": case.state.value, "claim_no": case.claim_no}
        return await self.atomic_write(case_id, "simulate_payment", idempotency_key, mutate)
