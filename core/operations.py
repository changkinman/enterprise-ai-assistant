"""Value types and domain errors for previewed side-effect operations."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from core.auth import Actor


class OperationState(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    EXPIRED = "expired"


class OperationError(Exception):
    """Base class for typed operation failures."""


class OperationNotFound(OperationError):
    pass


class OperationForbidden(OperationError):
    pass


class OperationExpired(OperationError):
    pass


class OperationStateConflict(OperationError):
    pass


class OperationVersionConflict(OperationError):
    pass


class IdempotencyConflict(OperationError):
    pass


class ValidationRejected(OperationError):
    pass


@dataclass(frozen=True)
class ActionProposal:
    action_type: str
    params: dict[str, Any]


@dataclass(frozen=True)
class ValidationResult:
    allowed: bool
    normalized_params: dict[str, Any] = field(default_factory=dict)
    resource_version: int = 0
    reason: str | None = None


@dataclass(frozen=True)
class PreviewPayload:
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OperationPreview:
    operation_id: str
    action_type: str
    preview: PreviewPayload
    state: OperationState
    expires_at: datetime


@dataclass(frozen=True)
class StoredOperation:
    operation_id: str
    actor_id: str
    action_type: str
    params: dict[str, Any]
    preview: PreviewPayload
    snapshot_hash: str
    resource_version: int
    state: OperationState
    expires_at: datetime

    def as_preview(self) -> OperationPreview:
        return OperationPreview(
            self.operation_id, self.action_type, self.preview, self.state, self.expires_at,
        )


class SideEffectOperation(Protocol):
    async def precheck(self, actor: Actor, params: dict[str, Any]) -> ValidationResult: ...

    async def build_preview(
        self, actor: Actor, normalized_params: dict[str, Any], validation: ValidationResult,
    ) -> PreviewPayload: ...

    async def commit(
        self, actor: Actor, normalized_params: dict[str, Any], idempotency_key: str,
    ) -> dict[str, Any]: ...
