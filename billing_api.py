"""Paddle Billing: checkout links, customer portal, webhooks (Israel-friendly MoR)."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from auth_deps import get_current_user_required
from billing_store import get_billing_store, normalize_guest_key
from paddle_client import (
    PaddleAPIError,
    paddle_configured,
    paddle_create_address,
    paddle_create_customer,
    paddle_create_portal_session,
    paddle_get_subscription,
    paddle_post_transaction,
)
from user_store import UserRecord

BASE_DIR = Path(__file__).resolve().parent

router = APIRouter()

VALID_PLANS = frozenset({"single", "day", "month", "sixmo"})
PAYMENT_PLANS = frozenset({"single", "day"})
SUBSCRIPTION_PLANS = frozenset({"month", "sixmo"})

_PRICE_ENV = {
    "single": "PADDLE_PRICE_SINGLE",
    "day": "PADDLE_PRICE_DAY",
    "month": "PADDLE_PRICE_MONTH",
    "sixmo": "PADDLE_PRICE_SIXMO",
}


def _price_id(plan: str) -> Optional[str]:
    env = _PRICE_ENV.get(plan)
    if not env:
        return None
    pid = os.environ.get(env, "").strip()
    return pid or None


def _frontend_base() -> str:
    return os.environ.get("FRONTEND_URL", "http://localhost:5173").strip().rstrip("/")


def _checkout_country() -> str:
    return os.environ.get("PADDLE_CHECKOUT_COUNTRY", "IL").strip().upper()[:2]


def _checkout_postal() -> str:
    return os.environ.get("PADDLE_CHECKOUT_POSTAL_CODE", "6100001").strip()


def _checkout_region() -> Optional[str]:
    r = os.environ.get("PADDLE_CHECKOUT_REGION", "").strip()
    return r or None


def _checkout_city() -> Optional[str]:
    c = os.environ.get("PADDLE_CHECKOUT_CITY", "").strip()
    return c or None


class CheckoutBody(BaseModel):
    plan: str = Field(..., min_length=3, max_length=16)


class CheckoutResponse(BaseModel):
    url: str


class PortalResponse(BaseModel):
    url: str


def _webhook_secret() -> str:
    return os.environ.get("PADDLE_WEBHOOK_SECRET", "").strip()


def _verify_paddle_signature(raw_body: bytes, sig_header: Optional[str]) -> bool:
    secret = _webhook_secret()
    if not secret or not sig_header:
        return False
    parts: dict[str, str] = {}
    for segment in sig_header.split(";"):
        segment = segment.strip()
        if "=" in segment:
            k, v = segment.split("=", 1)
            parts[k.strip()] = v.strip()
    ts = parts.get("ts")
    h1 = parts.get("h1")
    if not ts or not h1:
        return False
    try:
        ts_int = int(ts)
    except ValueError:
        return False
    tol = int(os.environ.get("PADDLE_WEBHOOK_TOLERANCE_SEC", "300"))
    if abs(int(time.time()) - ts_int) > tol:
        return False
    signed = ts.encode("utf-8") + b":" + raw_body
    expected = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, h1)


def _custom_str(data: dict[str, Any], key: str) -> Optional[str]:
    v = data.get(key)
    if v is None:
        return None
    return str(v).strip() or None


def _ensure_paddle_customer_and_address(store, user: UserRecord) -> tuple[str, str]:
    ent = store.get_entitlements(user.id)
    cid = ent.get("paddle_customer_id")
    aid = ent.get("paddle_address_id")
    if cid and aid:
        return cid, aid
    display = (user.username or user.email.split("@")[0] or "Customer").strip()
    if not cid:
        try:
            cres = paddle_create_customer(user.email, display, user.id)
        except PaddleAPIError as e:
            raise HTTPException(status_code=502, detail=f"Paddle customer error: {e}") from e
        cdata = cres.get("data") or cres
        cid = cdata.get("id")
        if not cid:
            raise HTTPException(status_code=502, detail="Paddle did not return customer id")
        store.set_paddle_customer(user.id, cid)
    if not aid:
        try:
            ares = paddle_create_address(
                cid,
                _checkout_country(),
                _checkout_postal(),
                region=_checkout_region(),
                city=_checkout_city(),
            )
        except PaddleAPIError as e:
            raise HTTPException(status_code=502, detail=f"Paddle address error: {e}") from e
        adata = ares.get("data") or ares
        aid = adata.get("id")
        if not aid:
            raise HTTPException(status_code=502, detail="Paddle did not return address id")
        store.set_paddle_address(user.id, aid)
    return cid, aid


@router.get("/status")
def billing_status():
    return {
        "provider": "paddle",
        "paddle_configured": paddle_configured(),
        "paddle_sandbox": os.environ.get("PADDLE_SANDBOX", "").strip().lower()
        in ("1", "true", "yes", "on"),
        "webhook_configured": bool(_webhook_secret()),
        "prices": {k: bool(os.environ.get(v, "").strip()) for k, v in _PRICE_ENV.items()},
    }


@router.get("/guest-status")
def billing_guest_status(
    x_guest_billing_id: Annotated[Optional[str], Header(alias="X-Guest-Billing-Id")] = None,
):
    gk = normalize_guest_key(x_guest_billing_id)
    if not gk:
        raise HTTPException(
            status_code=400,
            detail="Missing or invalid X-Guest-Billing-Id (8–64 hex characters)",
        )
    store = get_billing_store(BASE_DIR)
    g = store.get_guest_entitlements(gk)
    return {
        "guest_key": gk,
        "free_runs_used": g["free_runs_used"],
        "free_runs_remaining": g["free_runs_remaining"],
        "free_runs_max": 3,
        "note": "Guests: one image per job; sign in + checkout for multi-image or paid plans.",
    }


@router.get("/me")
def billing_me(
    user: Annotated[UserRecord, Depends(get_current_user_required)],
):
    store = get_billing_store(BASE_DIR)
    e = store.get_entitlements(user.id)
    return {
        "user_id": user.id,
        "access_until": e.get("access_until"),
        "has_unlimited": bool(e.get("has_unlimited")),
        "paid_job_credits": int(e.get("paid_job_credits") or 0),
        "free_runs_used": int(e.get("free_runs_used") or 0),
        "free_runs_remaining": int(e.get("free_runs_remaining") or 0),
        "free_runs_max": 3,
        "paddle_customer_id": e.get("paddle_customer_id"),
        "paddle_subscription_id": e.get("paddle_subscription_id"),
    }


@router.post("/checkout-session", response_model=CheckoutResponse)
def create_checkout_session(
    body: CheckoutBody,
    user: Annotated[UserRecord, Depends(get_current_user_required)],
):
    plan = body.plan.strip().lower()
    if plan not in VALID_PLANS:
        raise HTTPException(status_code=400, detail="Unknown plan")
    price = _price_id(plan)
    if not price:
        raise HTTPException(
            status_code=503,
            detail=f"Price not configured for plan {plan} (set {_PRICE_ENV[plan]})",
        )
    if not paddle_configured():
        raise HTTPException(status_code=503, detail="Paddle is not configured (PADDLE_API_KEY)")

    store = get_billing_store(BASE_DIR)
    store.ensure_row(user.id)
    customer_id, address_id = _ensure_paddle_customer_and_address(store, user)

    txn_body: dict[str, Any] = {
        "customer_id": customer_id,
        "address_id": address_id,
        "collection_mode": "automatic",
        "items": [{"price_id": price, "quantity": 1}],
        "custom_data": {"user_id": str(user.id), "plan": plan},
    }
    try:
        res = paddle_post_transaction(txn_body)
    except PaddleAPIError as e:
        raise HTTPException(status_code=502, detail=f"Paddle transaction error: {e}") from e

    data = res.get("data") or res
    checkout = data.get("checkout") or {}
    url = checkout.get("url")
    if not url:
        raise HTTPException(
            status_code=502,
            detail="Paddle did not return checkout.url — check price IDs and sandbox/live mode",
        )
    return CheckoutResponse(url=url)


@router.post("/portal-session", response_model=PortalResponse)
def create_portal_session(
    user: Annotated[UserRecord, Depends(get_current_user_required)],
):
    if not paddle_configured():
        raise HTTPException(status_code=503, detail="Paddle is not configured")
    store = get_billing_store(BASE_DIR)
    ent = store.get_entitlements(user.id)
    cid = ent.get("paddle_customer_id")
    if not cid:
        raise HTTPException(
            status_code=400,
            detail="No Paddle customer yet — complete checkout once to open the customer portal",
        )
    sub = ent.get("paddle_subscription_id")
    sub_ids = [sub] if sub else None
    try:
        res = paddle_create_portal_session(cid, subscription_ids=sub_ids)
    except PaddleAPIError as e:
        raise HTTPException(status_code=502, detail=f"Paddle portal error: {e}") from e
    data = res.get("data") or res
    urls = data.get("urls") or {}
    general = urls.get("general") or {}
    overview = general.get("overview")
    if not overview:
        raise HTTPException(status_code=502, detail="Paddle did not return portal overview URL")
    return PortalResponse(url=overview)


def _apply_transaction_completed(data: dict[str, Any]) -> None:
    store = get_billing_store(BASE_DIR)
    custom = data.get("custom_data") or {}
    uid = _custom_str(custom, "user_id")
    plan = (_custom_str(custom, "plan") or "").lower()
    cid = data.get("customer_id")
    if cid and uid:
        store.set_paddle_customer(uid, cid)

    if not uid:
        return

    if plan == "single":
        store.add_job_credits(uid, 1)
    elif plan == "day":
        store.extend_access_hours(uid, 24)

    sub_id = data.get("subscription_id")
    if sub_id:
        try:
            sub_res = paddle_get_subscription(sub_id)
        except PaddleAPIError:
            return
        sub = sub_res.get("data") or sub_res
        _apply_subscription_entity(sub)


def _apply_subscription_entity(sub: dict[str, Any]) -> None:
    store = get_billing_store(BASE_DIR)
    cid = sub.get("customer_id")
    uid = store.get_user_id_by_paddle_customer(cid) if cid else None
    if not uid:
        return
    sub_id = sub.get("id")
    period = sub.get("current_billing_period") or {}
    ends = period.get("ends_at")
    if ends:
        store.set_subscription_access_iso(uid, sub_id, ends)


@router.post("/webhook")
async def paddle_webhook(request: Request):
    if not _webhook_secret():
        raise HTTPException(status_code=503, detail="PADDLE_WEBHOOK_SECRET not configured")

    raw = await request.body()
    sig = request.headers.get("Paddle-Signature") or request.headers.get("paddle-signature")
    if not _verify_paddle_signature(raw, sig):
        raise HTTPException(status_code=400, detail="Invalid Paddle-Signature")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from e

    event_id = payload.get("event_id") or payload.get("notification_id")
    if not event_id:
        return {"received": True}

    store = get_billing_store(BASE_DIR)
    if not store.try_claim_webhook_event(event_id):
        return {"received": True, "duplicate": True}

    etype = (payload.get("event_type") or "").strip().lower()
    data = payload.get("data") or {}

    try:
        if etype == "transaction.completed":
            _apply_transaction_completed(data)
        elif etype in ("subscription.updated", "subscription.created", "subscription.canceled", "subscription.activated"):
            _apply_subscription_entity(data)
    except Exception:
        import traceback

        traceback.print_exc()
        store.release_webhook_event(event_id)
        raise HTTPException(status_code=500, detail="Webhook handler error")

    return {"received": True}
