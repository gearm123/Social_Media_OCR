"""HTTP routes for registration, login, and current user."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth_deps import get_current_user_required, get_user_store
from auth_jwt import create_access_token
from auth_password import hash_password, verify_password
from user_store import UserRecord, UserStore, validate_email, validate_password

router = APIRouter()


class RegisterBody(BaseModel):
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
    created_at: str


@router.post("/register", response_model=UserPublic)
def register(body: RegisterBody, store: Annotated[UserStore, Depends(get_user_store)]):
    try:
        validate_email(body.email)
        validate_password(body.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    pw_hash = hash_password(body.password)
    try:
        user = store.create_user(body.email, pw_hash)
    except ValueError as e:
        if "already registered" in str(e).lower():
            raise HTTPException(status_code=409, detail="Email already registered") from e
        raise
    return UserPublic(id=user.id, email=user.email, created_at=user.created_at)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginBody, store: Annotated[UserStore, Depends(get_user_store)]):
    row = store.get_by_email(body.email)
    if not row:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    user, pw_hash = row
    if not verify_password(body.password, pw_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserPublic)
def me(user: Annotated[UserRecord, Depends(get_current_user_required)]):
    return UserPublic(id=user.id, email=user.email, created_at=user.created_at)
