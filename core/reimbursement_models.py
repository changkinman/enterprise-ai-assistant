"""Shared domain types for the process-local reimbursement harness."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from core.operations import OperationPreview


EXPENSE_TYPES = {
    "hotel": "hotel", "酒店": "hotel", "住宿": "hotel", "住宿费": "hotel",
    "transport": "transport", "交通": "transport", "交通费": "transport",
    "meal": "meal", "meals": "meal", "餐饮": "meal", "餐费": "meal",
}


def canonical_expense_type(value: Any) -> str | None:
    return EXPENSE_TYPES.get(value.strip().lower()) if isinstance(value, str) else None


def is_explicit_submission(message: Any) -> bool:
    """Accept complete confirmation utterances, never substrings or model evidence."""
    return isinstance(message, str) and message.strip().rstrip("。！!").strip() in {
        "确认提交", "确认提交报销", "确认提交这笔报销", "确认提交这笔报销单",
        "我确认提交这笔报销", "请提交这笔报销", "确认提交报销单",
    }


class ClaimState(str, Enum):
    COLLECTING_INFO = "collecting_info"
    DRAFT_CREATED = "draft_created"
    VALIDATING = "validating"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    SUBMITTED = "submitted"
    REWORK_REQUIRED = "rework_required"
    PAID = "paid"
    CLOSED = "closed"


class ReimbursementAction(str, Enum):
    COLLECT_FACTS = "collect_facts"
    VALIDATE = "validate"
    CONFIRM_SUBMIT = "confirm_submit"
    GET_STATUS = "get_status"


@dataclass
class ReimbursementDecision:
    facts: dict[str, Any]
    action: ReimbursementAction
    explicit_confirmation: bool = False
    user_message: str = ""


@dataclass
class HarnessToolTrace:
    round: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    success: bool
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class HarnessResult:
    content: str
    case: "ReimbursementCase"
    tool_calls: list[HarnessToolTrace]
    success: bool
    pending_operation: OperationPreview | None = None


@dataclass
class ExecutionEvent:
    at: str
    version: int
    event_type: str
    detail: dict[str, Any]


@dataclass
class ReimbursementCase:
    case_id: str
    user_id: str
    conv_id: str
    department: str = "engineering"
    state: ClaimState = ClaimState.COLLECTING_INFO
    facts: dict[str, Any] = field(default_factory=dict)
    expenses: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    budget_reserved: bool = False
    claim_no: str | None = None
    version: int = 0
    material_version: int = 0
    events: list[ExecutionEvent] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def append_event(
    case: ReimbursementCase, event_type: str, detail: dict[str, Any]
) -> ExecutionEvent:
    """Append an event tagged with the case's current state version."""
    event = ExecutionEvent(
        at=datetime.now(timezone.utc).isoformat(),
        version=case.version,
        event_type=event_type,
        detail=dict(detail),
    )
    case.events.append(event)
    return event
