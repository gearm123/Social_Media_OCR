"""Stripe Checkout + webhooks; entitlements in SQLite (see billing_store)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Optional

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from auth_deps import get_current_user_required
from billing_store import get_billing_store, normalize_guest_key
from user_store import UserRecord

BASE_DIR = Path(__file__).resolve().parent

router = APIRouter()

VALID_PLANS = frozenset({"single", "day", "month", "sixmo"})
PAYMENT_PLANS = frozenset({"single", "day"})
SUBSCRIPTION_PLANS = frozenset({"month", "sixmo"})

_PRICE_ENV = {
    "single": "STRIPE_PRICE_SINGLE",
    "day": "STRIPE_PRICE_DAY",
    "month": "STRIPE_PRICE_MONTH",
    "sixmo": "STRIPE_PRICE_SIXMO",
}


def _configure_stripe() -> None:
    key = os.environ.get("STRIPE_SECRET_KEY", "").strip()
    if key:
        stripe.api_key = key


def stripe_configured() -> bool:
    return bool(os.environ.get("STRIPE_SECRET_KEY", "").strip())


def _price_id(plan: str) -> Optional[str]:
    env = _PRICE_ENV.get(plan)
    if not env:
        return None
    pid = os.environ.get(env, "").strip()
    return pid or None


def _frontend_base() -> str:
    return os.environ.get("FRONTEND_URL", "http://localhost:5173").strip().rstrip("/")


class CheckoutBody(BaseModel):
    plan: str = Field(..., min_length=3, max_length=16)


class CheckoutResponse(BaseModel):
    url: str


class PortalResponse(BaseModel):
    """Stripe Customer Portal URL (manage payment method, cancel subscription)."""

    url: str


@router.get("/status")
def billing_status():
    """Whether Stripe keys and webhook are configured (no secrets returned)."""
    return {
        "stripe_configured": stripe_configured(),
        "webhook_configured": bool(os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()),
        "prices": {
            k: bool(os.environ.get(v, "").strip()) for k, v in _PRICE_ENV.items()
        },
    }


@router.get("/guest-status")
def billing_guest_status(
    x_guest_billing_id: Annotated[Optional[str], Header(alias="X-Guest-Billing-Id")] = None,
):
    """Anonymous free-tier usage for this guest id (send same header as POST /jobs)."""
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
    _configure_stripe()
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
        "stripe_customer_id": e.get("stripe_customer_id"),
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
    if not stripe_configured():
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    _configure_stripe()
    store = get_billing_store(BASE_DIR)
    store.ensure_row(user.id)
    ent = store.get_entitlements(user.id)
    customer_id = ent.get("stripe_customer_id")

    if not customer_id:
        cus = stripe.Customer.create(
            email=user.email,
            metadata={"user_id": user.id},
        )
        customer_id = cus.id
        store.set_stripe_customer(user.id, customer_id)

    base = _frontend_base()
    success_url = f"{base}/?checkout=success&session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{base}/?checkout=cancel"

    meta = {"user_id": user.id, "plan": plan}

    if plan in PAYMENT_PLANS:
        session = stripe.checkout.Session.create(
            mode="payment",
            customer=customer_id,
            client_reference_id=user.id,
            metadata=meta,
            line_items=[{"price": price, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            payment_intent_data={"metadata": meta},
        )
    else:
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            client_reference_id=user.id,
            metadata=meta,
            line_items=[{"price": price, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            subscription_data={"metadata": meta},
        )

    return CheckoutResponse(url=session.url)


@router.post("/portal-session", response_model=PortalResponse)
def create_portal_session(
    user: Annotated[UserRecord, Depends(get_current_user_required)],
):
    """Stripe Customer Portal — update card, cancel subscription, see invoices."""
    _configure_stripe()
    if not stripe_configured():
        raise HTTPException(status_code=503, detail="Stripe is not configured")
    store = get_billing_store(BASE_DIR)
    ent = store.get_entitlements(user.id)
    cid = ent.get("stripe_customer_id")
    if not cid:
        raise HTTPException(
            status_code=400,
            detail="No Stripe customer yet — complete checkout once to open the portal",
        )
    base = _frontend_base()
    session = stripe.billing_portal.Session.create(
        customer=cid,
        return_url=f"{base}/",
    )
    return PortalResponse(url=session.url)


def _apply_checkout_session_completed(session: dict[str, Any]) -> None:
    store = get_billing_store(BASE_DIR)
    meta = session.get("metadata") or {}
    uid = meta.get("user_id") or session.get("client_reference_id")
    if not uid:
        return

    customer = session.get("customer")
    if customer:
        store.set_stripe_customer(uid, customer)

    mode = session.get("mode")
    plan = (meta.get("plan") or "").strip().lower()

    if mode == "payment":
        if plan == "single":
            store.add_job_credits(uid, 1)
        elif plan == "day":
            store.extend_access_hours(uid, 24)
        return

    if mode == "subscription":
        sub_id = session.get("subscription")
        if not sub_id:
            return
        sub = stripe.Subscription.retrieve(sub_id)
        cpe = sub.get("current_period_end")
        if cpe:
            store.update_subscription_fields(uid, sub_id, int(cpe))


def _apply_subscription_updated(sub: dict[str, Any]) -> None:
    store = get_billing_store(BASE_DIR)
    customer_id = sub.get("customer")
    if not customer_id:
        return
    uid = store.get_user_id_by_stripe_customer(customer_id)
    if not uid:
        return
    status = (sub.get("status") or "").lower()
    sub_id = sub.get("id")
    cpe = sub.get("current_period_end")
    if status in ("active", "trialing", "past_due") and cpe:
        store.update_subscription_fields(uid, sub_id, int(cpe))
    elif cpe:
        # canceled / unpaid — keep period end if Stripe still returns it
        store.update_subscription_fields(uid, sub_id, int(cpe))


def _apply_invoice_paid(invoice: dict[str, Any]) -> None:
    """Refresh subscription period after renewals (redundant with subscription.updated)."""
    sub_id = invoice.get("subscription")
    if not sub_id:
        return
    _configure_stripe()
    sub = stripe.Subscription.retrieve(sub_id)
    _apply_subscription_updated(sub)


@router.post("/webhook")
async def stripe_webhook(request: Request):
    _configure_stripe()
    wh_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()
    if not wh_secret:
        raise HTTPException(status_code=503, detail="Webhook secret not configured")

    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    if not sig:
        raise HTTPException(status_code=400, detail="Missing Stripe-Signature")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload, sig_header=sig, secret=wh_secret
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}") from e
    except Exception as e:
        msg = str(e).lower()
        if "signature" in msg or "sig_header" in msg:
            raise HTTPException(status_code=400, detail="Invalid Stripe signature") from e
        raise HTTPException(status_code=400, detail=str(e)) from e

    store = get_billing_store(BASE_DIR)
    eid = event.get("id")
    if not eid:
        return {"received": True}

    if not store.try_claim_webhook_event(eid):
        return {"received": True, "duplicate": True}

    etype = event.get("type")
    obj = event.get("data", {}).get("object") or {}

    try:
        if etype == "checkout.session.completed":
            _apply_checkout_session_completed(obj)
        elif etype == "customer.subscription.updated":
            _apply_subscription_updated(obj)
        elif etype == "customer.subscription.deleted":
            _apply_subscription_updated(obj)
        elif etype == "invoice.paid":
            _apply_invoice_paid(obj)
    except Exception:
        import traceback

        traceback.print_exc()
        store.release_webhook_event(eid)
        raise HTTPException(status_code=500, detail="Webhook handler error")

    return {"received": True}
