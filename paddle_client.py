"""Minimal Paddle Billing API client (urllib — no extra dependencies)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional


def paddle_api_base() -> str:
    if os.environ.get("PADDLE_SANDBOX", "").strip().lower() in ("1", "true", "yes", "on"):
        return "https://sandbox-api.paddle.com"
    return os.environ.get("PADDLE_API_BASE", "https://api.paddle.com").rstrip("/")


def paddle_api_key() -> str:
    return os.environ.get("PADDLE_API_KEY", "").strip()


def paddle_configured() -> bool:
    return bool(paddle_api_key())


class PaddleAPIError(Exception):
    def __init__(self, message: str, status: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


def _request(
    method: str,
    path: str,
    *,
    json_body: Optional[dict[str, Any]] = None,
    query: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    key = paddle_api_key()
    if not key:
        raise PaddleAPIError("PADDLE_API_KEY is not set")

    base = paddle_api_base()
    url = f"{base}{path}"
    if query:
        from urllib.parse import urlencode

        url = f"{url}?{urlencode(query)}"

    data = None
    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, method=method.upper(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                return {}
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        try:
            j = json.loads(err_body)
            detail = j.get("error", j)
            msg = json.dumps(detail) if isinstance(detail, (dict, list)) else str(detail)
        except json.JSONDecodeError:
            msg = err_body or str(e)
        raise PaddleAPIError(msg, status=e.code, body=err_body) from e


def paddle_post_transaction(body: dict[str, Any]) -> dict[str, Any]:
    # Do not use query include=checkout — Paddle only allows include values like
    # customer, address, discount, etc. Checkout URL is returned on data.checkout for
    # automatic collection_mode transactions.
    return _request("POST", "/transactions", json_body=body)


def paddle_get_subscription(subscription_id: str) -> dict[str, Any]:
    return _request("GET", f"/subscriptions/{subscription_id}")


def paddle_create_customer(
    email: str,
    name: Optional[str],
    *,
    user_id: Optional[str] = None,
    guest_key: Optional[str] = None,
) -> dict[str, Any]:
    if not user_id and not guest_key:
        raise ValueError("paddle_create_customer requires user_id or guest_key")
    custom: dict[str, str] = {}
    if user_id:
        custom["user_id"] = str(user_id)
    if guest_key:
        custom["guest_key"] = str(guest_key)
    payload: dict[str, Any] = {"email": email, "custom_data": custom}
    if name and name.strip():
        payload["name"] = name.strip()
    return _request("POST", "/customers", json_body=payload)


def paddle_create_address(
    customer_id: str,
    country_code: str,
    postal_code: str,
    *,
    region: Optional[str] = None,
    city: Optional[str] = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "country_code": country_code.upper(),
        "postal_code": postal_code.strip(),
        "description": "Checkout",
    }
    if region:
        body["region"] = region
    if city:
        body["city"] = city
    return _request("POST", f"/customers/{customer_id}/addresses", json_body=body)


def paddle_create_portal_session(customer_id: str, subscription_ids: Optional[list[str]] = None) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if subscription_ids:
        body["subscription_ids"] = subscription_ids
    return _request("POST", f"/customers/{customer_id}/portal-sessions", json_body=body)
