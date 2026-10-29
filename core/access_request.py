"""Deterministic local IT access request, without an IAM integration."""

import asyncio
from copy import deepcopy

from core.auth import Actor
from core.operations import PreviewPayload, ValidationResult


class InMemoryAccessRequestStore:
    """Local submitted tickets; the executor owns confirmation and replay."""

    def __init__(self) -> None:
        self._requests: dict[str, dict] = {}
        self._keys: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()

    async def submit(self, actor: Actor, params: dict, idempotency_key: str) -> dict:
        async with self._lock:
            identity = (actor.id, idempotency_key)
            existing = self._keys.get(identity)
            if existing is not None:
                return deepcopy(self._requests[existing])
            request_id = f"ACCESS-{len(self._requests) + 1:06d}"
            request = {
                "request_id": request_id, "owner_id": actor.id,
                "system_name": params["system_name"], "permission": params["permission"],
                "reason": params["reason"], "state": "submitted", "version": 1,
            }
            self._requests[request_id] = request
            self._keys[identity] = request_id
            return deepcopy(request)

    async def list_for_owner(self, owner_id: str) -> list[dict]:
        async with self._lock:
            return [deepcopy(item) for item in self._requests.values()
                    if item["owner_id"] == owner_id]


class AccessRequestOperation:
    SYSTEMS = frozenset({"vpn", "gitlab", "finance_portal"})
    PERMISSIONS = frozenset({"read", "write"})

    def __init__(self, store: InMemoryAccessRequestStore) -> None:
        self.store = store

    async def precheck(self, actor: Actor, params: dict) -> ValidationResult:
        system = params.get("system_name")
        permission = params.get("permission")
        reason = params.get("reason")
        if not isinstance(system, str) or system not in self.SYSTEMS:
            return ValidationResult(False, reason="Unsupported system")
        if not isinstance(permission, str) or permission not in self.PERMISSIONS:
            return ValidationResult(False, reason="Unsupported permission")
        if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 200:
            return ValidationResult(False, reason="Reason must contain 1 to 200 characters")
        if actor.role not in {"employee", "manager"}:
            return ValidationResult(False, reason="Actor role cannot request access")
        if permission == "write" and actor.role != "manager":
            return ValidationResult(False, reason="Write access requires manager role")
        return ValidationResult(True, {
            "system_name": system, "permission": permission, "reason": reason.strip(),
        }, resource_version=1)

    async def build_preview(self, actor: Actor, normalized_params: dict,
                            validation: ValidationResult) -> PreviewPayload:
        return PreviewPayload(
            f"申请 {normalized_params['system_name']} 的 {normalized_params['permission']} 权限，确认后提交本地工单。",
            deepcopy(normalized_params),
        )

    async def commit(self, actor: Actor, normalized_params: dict,
                     idempotency_key: str) -> dict:
        return await self.store.submit(actor, normalized_params, idempotency_key)
