"""Signed JWT identity for operation APIs and the side-effect executor."""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Header, HTTPException


class AuthenticationError(ValueError):
    """The request has no valid, signed actor identity."""


@dataclass(frozen=True)
class Actor:
    id: str
    role: str


def _secret(secret: str | None) -> str:
    value = secret if secret is not None else os.environ.get("JWT_SECRET_KEY", "")
    if not value:
        raise AuthenticationError("JWT secret is not configured")
    return value


def create_access_token(
    actor: Actor, *, secret: str | None = None,
    expires_in: timedelta = timedelta(hours=1),
) -> str:
    """Sign exactly the actor and expiry claims with HS256."""
    claims = {
        "sub": actor.id,
        "role": actor.role,
        "exp": datetime.now(timezone.utc) + expires_in,
    }
    return jwt.encode(claims, _secret(secret), algorithm="HS256")


def decode_actor(token: str, *, secret: str | None = None) -> Actor:
    try:
        claims = jwt.decode(
            token, _secret(secret), algorithms=["HS256"],
            options={"require": ["sub", "role", "exp"]},
        )
    except (jwt.PyJWTError, TypeError, ValueError) as exc:
        raise AuthenticationError("Invalid or expired bearer token") from exc
    if set(claims) != {"sub", "role", "exp"}:
        raise AuthenticationError("Unexpected token claims")
    if not isinstance(claims["sub"], str) or not claims["sub"].strip():
        raise AuthenticationError("Invalid token subject")
    if not isinstance(claims["role"], str) or not claims["role"].strip():
        raise AuthenticationError("Invalid token role")
    return Actor(claims["sub"], claims["role"])


async def get_current_actor(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> Actor:
    parts = authorization.split() if isinstance(authorization, str) else []
    if len(parts) != 2 or parts[0] != "Bearer":
        raise HTTPException(status_code=401, detail="Bearer token required")
    try:
        return decode_actor(parts[1])
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def create_dev_access_token(actor: Actor) -> str:
    """Issue a local demo token only in the explicitly named development env."""
    if os.environ.get("APP_ENV") != "development":
        raise AuthenticationError("Development token issuance is disabled")
    return create_access_token(actor)
