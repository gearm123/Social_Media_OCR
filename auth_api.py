"""HTTP routes for registration, login, OAuth, and current user."""

from __future__ import annotations

import os
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth_deps import get_current_user_required, get_user_store
from auth_jwt import create_access_token
from auth_oauth import (
    OAuthError,
    is_reserved_facebook_placeholder_email,
    verify_facebook_access_token,
    verify_google_access_token,
    verify_google_id_token,
)
from auth_password import hash_password, verify_password
from user_store import (
    OAUTH_PASSWORD_SENTINEL,
    UserRecord,
    UserStore,
    suggest_username_from_email_or_name,
    validate_email,
    validate_password,
    validate_username,
)

router = APIRouter()


class RegisterBody(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=8, max_length=256)


class LoginBody(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=1, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserPublic(BaseModel):
    id: str
    email: str
    username: Optional[str] = None
    created_at: str


class GoogleTokenBody(BaseModel):
    """Send either ``id_token`` (GIS button) or ``access_token`` (OAuth2 token client + userinfo)."""

    id_token: Optional[str] = None
    access_token: Optional[str] = None


class FacebookTokenBody(BaseModel):
    access_token: str = Field(..., min_length=10)


@router.get("/providers")
def auth_providers():
    """Public OAuth client IDs for the frontend (safe to expose)."""
    return {
        "google_client_id": os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "").strip(),
        "facebook_app_id": os.environ.get("FACEBOOK_APP_ID", "").strip(),
    }


def _user_public(u: UserRecord) -> UserPublic:
    return UserPublic(
        id=u.id, email=u.email, username=u.username, created_at=u.created_at
    )


def _oauth_sign_in(store: UserStore, provider: str, profile: dict) -> TokenResponse:
    sub = profile["sub"]
    email = profile["email"]
    name = profile.get("name")

    existing = store.get_by_oauth(provider, sub)
    if existing:
        return TokenResponse(access_token=create_access_token(existing.id))

    row = store.get_by_email(email)
    if row:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists. Sign in with email and password.",
        )

    base = suggest_username_from_email_or_name(email, name)
    suffix = 0
    last_err: Exception | None = None
    while suffix < 100:
        candidate = base if suffix == 0 else f"{base[:24]}_{suffix}"
        candidate = candidate[:32]
        try:
            user = store.create_oauth_user(provider, sub, email, candidate)
            return TokenResponse(access_token=create_access_token(user.id))
        except ValueError as e:
            last_err = e
            msg = str(e).lower()
            if "username" in msg and "taken" in msg:
                suffix += 1
                continue
            if "email" in msg:
                raise HTTPException(status_code=409, detail=str(e)) from e
            raise HTTPException(status_code=400, detail=str(e)) from e
    raise HTTPException(status_code=500, detail=str(last_err) if last_err else "username allocation failed")


@router.post("/register", response_model=UserPublic)
def register(body: RegisterBody, store: Annotated[UserStore, Depends(get_user_store)]):
    try:
        validate_username(body.username)
        validate_email(body.email)
        validate_password(body.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if is_reserved_facebook_placeholder_email(body.email):
        raise HTTPException(
            status_code=400,
            detail="This email is reserved for Facebook sign-in. Use Continue with Facebook or pick another address.",
        )
    pw_hash = hash_password(body.password)
    try:
        user = store.create_user_with_password(body.email, pw_hash, body.username.strip())
    except ValueError as e:
        msg = str(e).lower()
        if "email" in msg:
            raise HTTPException(status_code=409, detail="Email already registered") from e
        if "username" in msg:
            raise HTTPException(status_code=409, detail="Username already taken") from e
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _user_public(user)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginBody, store: Annotated[UserStore, Depends(get_user_store)]):
    row = store.get_by_email(body.email)
    if not row:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    user, pw_hash = row
    if pw_hash == OAUTH_PASSWORD_SENTINEL:
        raise HTTPException(
            status_code=401,
            detail="This account uses social sign-in. Use Google or Facebook.",
        )
    if not verify_password(body.password, pw_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/oauth/google", response_model=TokenResponse)
def oauth_google(
    body: GoogleTokenBody, store: Annotated[UserStore, Depends(get_user_store)]
):
    tid = (body.id_token or "").strip()
    tac = (body.access_token or "").strip()
    if bool(tid) == bool(tac):
        raise HTTPException(
            status_code=400,
            detail="Send exactly one of id_token or access_token",
        )
    try:
        profile = verify_google_id_token(tid) if tid else verify_google_access_token(tac)
    except OAuthError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e
    return _oauth_sign_in(store, "google", profile)


@router.post("/oauth/facebook", response_model=TokenResponse)
def oauth_facebook(
    body: FacebookTokenBody, store: Annotated[UserStore, Depends(get_user_store)]
):
    try:
        profile = verify_facebook_access_token(body.access_token)
    except OAuthError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e
    return _oauth_sign_in(store, "facebook", profile)


@router.get("/me", response_model=UserPublic)
def me(user: Annotated[UserRecord, Depends(get_current_user_required)]):
    return _user_public(user)
