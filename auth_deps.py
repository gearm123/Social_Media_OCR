"""FastAPI dependencies for optional / required auth."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Dict, Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth_jwt import decode_access_token
from user_store import UserRecord, UserStore, user_store_from_env

BASE_DIR = Path(__file__).resolve().parent

_user_store_singleton: Optional[UserStore] = None


def get_user_store() -> UserStore:
    global _user_store_singleton
    if _user_store_singleton is None:
        _user_store_singleton = user_store_from_env(BASE_DIR)
    return _user_store_singleton


bearer_scheme = HTTPBearer(auto_error=False)


def require_auth_for_jobs() -> bool:
    return os.environ.get("REQUIRE_AUTH_FOR_JOBS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


async def get_current_user_optional(
    creds: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)],
    store: Annotated[UserStore, Depends(get_user_store)],
) -> Optional[UserRecord]:
    if not creds or not creds.credentials:
        return None
    uid = decode_access_token(creds.credentials)
    if not uid:
        return None
    return store.get_by_id(uid)


async def get_current_user_required(
    user: Annotated[Optional[UserRecord], Depends(get_current_user_optional)],
) -> UserRecord:
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_job_user(
    user: Annotated[Optional[UserRecord], Depends(get_current_user_optional)],
) -> Optional[UserRecord]:
    """When REQUIRE_AUTH_FOR_JOBS is on, same as required user; else optional."""
    if require_auth_for_jobs():
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="Authentication required for jobs",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    return user


def assert_job_readable(status: Dict[str, Any], user: Optional[UserRecord]) -> None:
    """If job auth is required, caller must be logged in and own the job."""
    if not require_auth_for_jobs():
        return
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    owner = status.get("user_id")
    if not owner or owner != user.id:
        raise HTTPException(status_code=404, detail="Job not found")
