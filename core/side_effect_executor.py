"""Server-owned preview and confirmation for deterministic side effects."""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from core.auth import Actor
from core.operation_repository import OperationRepository
from core.operations import (
    ActionProposal, IdempotencyConflict, OperationExpired, OperationNotFound, OperationPreview,
    OperationState, OperationStateConflict, OperationVersionConflict,
    SideEffectOperation, ValidationRejected,
)


def _digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _snapshot(action_type: str, actor: Actor, params: dict[str, Any], version: int) -> str:
    return _digest({"action_type": action_type, "actor_id": actor.id,
                    "params": params, "resource_version": version})


class SideEffectExecutor:
    """Own the authority to turn a previewed proposal into one commit."""

    def __init__(
        self, repository: OperationRepository, adapters: Mapping[str, SideEffectOperation],
        *, preview_ttl: timedelta = timedelta(seconds=600),
    ) -> None:
        self.repository = repository
        self.adapters = dict(adapters)
        self.preview_ttl = preview_ttl

    async def prepare(self, actor: Actor, proposal: ActionProposal) -> OperationPreview:
        await self.repository.append_audit(None, "proposal_received", actor,
                                           {"action_type": proposal.action_type})
        adapter = self.adapters.get(proposal.action_type)
        if adapter is None:
            await self.repository.append_audit(None, "precheck_rejected", actor,
                                               {"action_type": proposal.action_type, "reason": "unknown_action"})
            raise ValidationRejected("Unknown action type")

        validation = await adapter.precheck(actor, proposal.params)
        if not validation.allowed:
            await self.repository.append_audit(None, "precheck_rejected", actor,
                                               {"action_type": proposal.action_type,
                                                "reason": validation.reason or "rejected"})
            raise ValidationRejected(validation.reason or "Precheck rejected")

        normalized = validation.normalized_params
        preview = await adapter.build_preview(actor, normalized, validation)
        snapshot_hash = _snapshot(proposal.action_type, actor, normalized,
                                  validation.resource_version)
        stored = await self.repository.create_pending(
            actor, proposal.action_type, normalized, preview,
            validation.resource_version, snapshot_hash,
            datetime.now(timezone.utc) + self.preview_ttl,
        )
        await self.repository.append_audit(stored.operation_id, "preview_created", actor,
                                           {"action_type": proposal.action_type})
        return stored.as_preview()

    async def get_preview(self, actor: Actor, operation_id: str) -> OperationPreview:
        operation = await self.repository.get_for_actor(operation_id, actor)
        if operation is None:
            raise OperationNotFound("Operation not found")
        return operation.as_preview()

    async def confirm(
        self, actor: Actor, operation_id: str, idempotency_key: str,
    ) -> dict[str, Any]:
        if not idempotency_key or not idempotency_key.strip():
            raise ValueError("Idempotency-Key is required")

        commit_started = False
        try:
            async with self.repository.transaction() as connection:
                operation = await self.repository.lock_for_confirmation(operation_id, actor, connection)
                if operation is None:
                    raise OperationNotFound("Operation not found")

                request_hash = _digest({"operation_id": operation.operation_id,
                                        "snapshot_hash": operation.snapshot_hash})
                replay = await self.repository.load_idempotency(
                    actor, operation.action_type, idempotency_key, request_hash, connection,
                )
                if replay is not None:
                    await self.repository.append_audit(operation_id, "idempotency_replayed", actor, {}, connection)
                    return replay
                if operation.state is not OperationState.PENDING:
                    raise OperationStateConflict("Operation is not pending")

                adapter = self.adapters.get(operation.action_type)
                if adapter is None:
                    raise OperationStateConflict("Action adapter is unavailable")
                validation = await adapter.precheck(actor, operation.params)
                if not validation.allowed:
                    raise ValidationRejected(validation.reason or "Precheck rejected")
                current_hash = _snapshot(operation.action_type, actor,
                                         validation.normalized_params, validation.resource_version)
                if (validation.resource_version != operation.resource_version
                        or current_hash != operation.snapshot_hash):
                    raise OperationVersionConflict("Operation preview is stale")

                commit_started = True
                response = await adapter.commit(actor, operation.params, idempotency_key)
                await self.repository.save_idempotency(
                    actor, operation.action_type, idempotency_key, request_hash, response, connection,
                )
                await self.repository.save_completed(operation_id, actor, connection)
                await self.repository.append_audit(operation_id, "commit_succeeded", actor, {}, connection)
                return response
        except OperationExpired:
            # Repositories persist the expiry transition after the failed transaction.
            raise
        except (IdempotencyConflict, OperationNotFound, OperationStateConflict,
                OperationVersionConflict, ValidationRejected) as exc:
            await self.repository.append_audit(operation_id, "confirmation_rejected", actor,
                                               {"reason": type(exc).__name__})
            raise
        except Exception as exc:
            await self.repository.append_audit(
                operation_id, "tool_failed" if commit_started else "confirmation_rejected", actor,
                {"reason": type(exc).__name__},
            )
            raise
