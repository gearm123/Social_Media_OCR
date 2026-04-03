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
from billing_store import (
    GUEST_FREE_RUNS_MAX,
    USER_FREE_RUNS_MAX,
    get_billing_store,
    normalize_guest_key,
    subscription_runs_cap,
)
from paddle_client import (
    PaddleAPIError,
    paddle_configured,
    paddle_create_portal_session,
    paddle_get_or_create_address_id_for_checkout,
    paddle_get_or_create_customer_id_for_checkout,
    paddle_get_subscription,
    paddle_post_transaction,
)
from user_store import UserRecord

BASE_DIR = Path(__file__).resolve().parent

router = APIRouter()

VALID_PLANS = frozenset({"single", "debug", "month", "sixmo", "year"})
# One-time checkout only — no saved subscription.
PAYMENT_PLANS = frozenset({"single", "debug"})
# Recurring in Paddle: monthly, every 6 months, every 12 months (year billed annually).
SUBSCRIPTION_PLANS = frozenset({"month", "sixmo", "year"})

_PRICE_ENV = {
    "single": "PADDLE_PRICE_SINGLE",
    "debug": "PADDLE_PRICE_DEBUG",
    "month": "PADDLE_PRICE_MONTH",
    "sixmo": "PADDLE_PRICE_SIXMO",
    "year": "PADDLE_PRICE_YEAR",
}


def _price_id(plan: str) -> Optional[str]:
    env = _PRICE_ENV.get(plan)
    if not env:
        return None
    pid = os.environ.get(env, "").strip()
    return pid or None


def _checkout_url_from_transaction_response(res: dict[str, Any]) -> Optional[str]:
    """Read checkout payment link from Paddle POST /transactions response (shape may vary slightly)."""
    if not isinstance(res, dict):
        return None
    data = res.get("data")
    if isinstance(data, dict):
        checkout = data.get("checkout")
        if isinstance(checkout, dict):
            u = checkout.get("url")
            if isinstance(u, str) and u.strip():
                return u.strip()
        for key in ("checkout_url", "payment_url"):
            u = data.get(key)
            if isinstance(u, str) and u.strip().startswith("http"):
                return u.strip()
    return None


def _frontend_base() -> str:
    """Site origin for Paddle checkout return URL; must be an absolute http(s) URL."""
    u = os.environ.get("FRONTEND_URL", "http://localhost:5173").strip().rstrip("/")
    if u and not u.lower().startswith(("http://", "https://")):
        u = f"https://{u}"
    return u


def _paddle_checkout_page_url() -> str:
    """Paddle appends ?_ptxn=… to this URL; domain must be allowlisted in Paddle."""
    return f"{_frontend_base()}/pay"


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


class GuestCheckoutBody(BaseModel):
    plan: str = Field(..., min_length=3, max_length=16)
    email: str = Field(..., min_length=3, max_length=320)


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


def _normalize_custom_data(raw: Any) -> dict[str, Any]:
    """Paddle may send JSON object or string; dashboard/API may use camelCase keys."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            raw = json.loads(s)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = dict(raw)
    if "guest_key" not in out and out.get("guestKey") is not None:
        out["guest_key"] = out.get("guestKey")
    if "user_id" not in out and out.get("userId") is not None:
        out["user_id"] = out.get("userId")
    return out


def _ensure_paddle_customer_and_address(store, user: UserRecord) -> tuple[str, str]:
    ent = store.get_entitlements(user.id)
    cid = ent.get("paddle_customer_id")
    aid = ent.get("paddle_address_id")
    if cid and aid:
        return cid, aid
    display = (user.username or user.email.split("@")[0] or "Customer").strip()
    if not cid:
        try:
            cid = paddle_get_or_create_customer_id_for_checkout(
                user.email, display, user_id=user.id
            )
        except PaddleAPIError as e:
            raise HTTPException(status_code=502, detail=f"Paddle customer error: {e}") from e
        store.set_paddle_customer(user.id, cid)
    if not aid:
        try:
            aid = paddle_get_or_create_address_id_for_checkout(
                cid,
                _checkout_country(),
                _checkout_postal(),
                region=_checkout_region(),
                city=_checkout_city(),
            )
        except PaddleAPIError as e:
            raise HTTPException(status_code=502, detail=f"Paddle address error: {e}") from e
        store.set_paddle_address(user.id, aid)
    return cid, aid


def _ensure_guest_paddle_customer_and_address(
    store, guest_key: str, email: str
) -> tuple[str, str]:
    g = store.get_guest_entitlements(guest_key)
    cid = g.get("paddle_customer_id")
    aid = g.get("paddle_address_id")
    if cid and aid:
        return str(cid), str(aid)
    em = email.strip()
    if not em or "@" not in em:
        raise HTTPException(status_code=400, detail="Valid email is required for checkout")
    display = (em.split("@")[0] or "Guest").strip() or "Guest"
    if not cid:
        try:
            cid = paddle_get_or_create_customer_id_for_checkout(
                em, display, guest_key=guest_key
            )
        except PaddleAPIError as e:
            raise HTTPException(status_code=502, detail=f"Paddle customer error: {e}") from e
        store.set_guest_paddle_customer(guest_key, cid)
    if not aid:
        try:
            aid = paddle_get_or_create_address_id_for_checkout(
                cid,
                _checkout_country(),
                _checkout_postal(),
                region=_checkout_region(),
                city=_checkout_city(),
            )
        except PaddleAPIError as e:
            raise HTTPException(status_code=502, detail=f"Paddle address error: {e}") from e
        store.set_guest_paddle_address(guest_key, aid)
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
        "subscription_runs_per_month": subscription_runs_cap(),
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
        "free_runs_max": GUEST_FREE_RUNS_MAX,
        "paid_job_credits": int(g.get("paid_job_credits") or 0),
        "note": "Guests: one free single-image try; then one-time purchase (no account). Credits allow multi-image runs.",
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
        "subscription_active": bool(e.get("subscription_active")),
        "subscription_runs_cap": int(e.get("subscription_runs_cap") or 0),
        "subscription_runs_used_this_month": int(e.get("subscription_runs_used_this_month") or 0),
        "subscription_runs_remaining": int(e.get("subscription_runs_remaining") or 0),
        "subscription_quota_month": e.get("subscription_quota_month"),
        "paid_job_credits": int(e.get("paid_job_credits") or 0),
        "free_runs_used": int(e.get("free_runs_used") or 0),
        "free_runs_remaining": int(e.get("free_runs_remaining") or 0),
        "free_runs_max": USER_FREE_RUNS_MAX,
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
        "checkout": {"url": _paddle_checkout_page_url()},
    }
    try:
        res = paddle_post_transaction(txn_body)
    except PaddleAPIError as e:
        raise HTTPException(status_code=502, detail=f"Paddle transaction error: {e}") from e

    url = _checkout_url_from_transaction_response(res)
    if not url:
        raise HTTPException(
            status_code=502,
            detail="Paddle did not return checkout.url — check FRONTEND_URL (allowed in Paddle), price IDs, and API key",
        )
    return CheckoutResponse(url=url)


@router.post("/guest-checkout-session", response_model=CheckoutResponse)
def create_guest_checkout_session(
    body: GuestCheckoutBody,
    x_guest_billing_id: Annotated[Optional[str], Header(alias="X-Guest-Billing-Id")] = None,
):
    gk = normalize_guest_key(x_guest_billing_id)
    if not gk:
        raise HTTPException(
            status_code=400,
            detail="Missing or invalid X-Guest-Billing-Id (8–64 hex characters)",
        )
    plan = body.plan.strip().lower()
    if plan not in PAYMENT_PLANS:
        raise HTTPException(
            status_code=400,
            detail="Guests may only purchase one-time plans (single or debug)",
        )
    price = _price_id(plan)
    if not price:
        raise HTTPException(
            status_code=503,
            detail=f"Price not configured for plan {plan} (set {_PRICE_ENV[plan]})",
        )
    if not paddle_configured():
        raise HTTPException(status_code=503, detail="Paddle is not configured (PADDLE_API_KEY)")

    store = get_billing_store(BASE_DIR)
    store.ensure_guest_row(gk)
    customer_id, address_id = _ensure_guest_paddle_customer_and_address(store, gk, body.email)

    txn_body: dict[str, Any] = {
        "customer_id": customer_id,
        "address_id": address_id,
        "collection_mode": "automatic",
        "items": [{"price_id": price, "quantity": 1}],
        "custom_data": {"guest_key": gk, "plan": plan},
        "checkout": {"url": _paddle_checkout_page_url()},
    }
    try:
        res = paddle_post_transaction(txn_body)
    except PaddleAPIError as e:
        raise HTTPException(status_code=502, detail=f"Paddle transaction error: {e}") from e

    url = _checkout_url_from_transaction_response(res)
    if not url:
        raise HTTPException(
            status_code=502,
            detail="Paddle did not return checkout.url — check FRONTEND_URL (allowed in Paddle), price IDs, and API key",
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
    custom = _normalize_custom_data(data.get("custom_data"))
    gk_raw = _custom_str(custom, "guest_key")
    gk = normalize_guest_key(gk_raw) if gk_raw else None
    uid = _custom_str(custom, "user_id")
    plan = (_custom_str(custom, "plan") or "").lower()
    tid = data.get("id")
    tid = str(tid).strip() if tid else None

    cid = data.get("customer_id")
    if isinstance(cid, str):
        cid = cid.strip() or None

    if not gk and cid:
        gk = store.get_guest_key_by_paddle_customer_id(cid)

    if gk:
        if cid:
            store.set_guest_paddle_customer(gk, cid)
        # Guest checkout only sells one-time run products; credit if plan matches or webhook omitted plan.
        grant = plan in ("single", "debug") or not plan
        if grant:
            if not store.try_claim_one_time_txn_credit(tid):
                return
            store.guest_add_job_credits(gk, 1)
        return

    if cid and uid:
        store.set_paddle_customer(uid, cid)

    if not uid:
        return

    if plan in ("single", "debug"):
        if not store.try_claim_one_time_txn_credit(tid):
            return
        store.add_job_credits(uid, 1)

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
        if etype in ("transaction.completed", "transaction.paid"):
            _apply_transaction_completed(data)
        elif etype in ("subscription.updated", "subscription.created", "subscription.canceled", "subscription.activated"):
            _apply_subscription_entity(data)
    except Exception:
        import traceback

        traceback.print_exc()
        store.release_webhook_event(event_id)
        raise HTTPException(status_code=500, detail="Webhook handler error")

    return {"received": True}
