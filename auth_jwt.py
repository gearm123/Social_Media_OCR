"""JWT access tokens (HS256). Requires PyJWT."""

from __future__ import annotations

import os
import secrets
import time
from typing import Optional

import jwt

_ephemeral_secret: Optional[str] = None


def jwt_secret() -> str:
    """Use JWT_SECRET env; otherwise ephemeral (dev single-process only)."""
    global _ephemeral_secret
    s = os.environ.get("JWT_SECRET", "").strip()
    if s:
        return s
    if _ephemeral_secret is None:
        _ephemeral_secret = secrets.token_hex(32)
        print(
            "[AUTH] JWT_SECRET not set; using ephemeral signing key (tokens lost on restart). "
            "Set JWT_SECRET in production.",
            flush=True,
        )
    return _ephemeral_secret


def create_access_token(user_id: str, ttl_seconds: int = 7 * 24 * 3600) -> str:
    now = int(time.time())
    payload = {"sub": user_id, "iat": now, "exp": now + ttl_seconds}
    return jwt.encode(payload, jwt_secret(), algorithm="HS256")


def decode_access_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, jwt_secret(), algorithms=["HS256"])
        sub = payload.get("sub")
        return str(sub) if sub else None
    except jwt.PyJWTError:
        return None
