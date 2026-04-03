i"""Password hashing (stdlib only — PBKDF2-SHA256)."""

from __future__ import annotations

import base64
import hashlib
import secrets

_ITERATIONS = 310_000
_PREFIX = "pbkdf2_sha256"


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return (
        f"{_PREFIX}${_ITERATIONS}$"
        f"{base64.b64encode(salt).decode('ascii')}$"
        f"{base64.b64encode(dk).decode('ascii')}"
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, it_s, salt_s, hash_s = stored.split("$", 3)
        if algo != _PREFIX:
            return False
        iterations = int(it_s)
        salt = base64.b64decode(salt_s.encode("ascii"))
        expected = base64.b64decode(hash_s.encode("ascii"))
    except Exception:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return secrets.compare_digest(dk, expected)
