"""Verify OAuth tokens from Google and Facebook."""

from __future__ import annotations

import os
from typing import Any, Dict

import requests

# Facebook: we intentionally do not request the `email` permission so Meta does not treat
# sign-in as requiring advanced “email” access / login review for many apps. The DB still
# needs a unique `email` column, so we store a stable synthetic address per Facebook user id.
FACEBOOK_OAUTH_EMAIL_SUFFIX = "@fb-oauth.internal"

try:
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests
except ImportError:  # pragma: no cover
    google_id_token = None  # type: ignore
    google_requests = None  # type: ignore


class OAuthError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def verify_google_id_token(id_token_str: str) -> Dict[str, Any]:
    client_id = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "").strip()
    if not client_id:
        raise OAuthError("Google sign-in is not configured on this server", 503)
    if google_id_token is None or google_requests is None:
        raise OAuthError("Google auth library not installed", 503)
    try:
        info = google_id_token.verify_oauth2_token(
            id_token_str, google_requests.Request(), client_id
        )
    except ValueError as e:
        raise OAuthError(f"Invalid Google token: {e}") from e
    if not info.get("sub"):
        raise OAuthError("Invalid Google token: missing subject")
    if not info.get("email"):
        raise OAuthError("Google did not return an email; try another sign-in method")
    return {
        "sub": str(info["sub"]),
        "email": str(info["email"]).strip().lower(),
        "name": (info.get("name") or "").strip() or None,
    }


def verify_google_access_token(access_token: str) -> Dict[str, Any]:
    """Validate a Google OAuth 2.0 access token via userinfo (custom Sign-in button / token client)."""
    if not os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "").strip():
        raise OAuthError("Google sign-in is not configured on this server", 503)
    tok = access_token.strip()
    if len(tok) < 10:
        raise OAuthError("Invalid Google token", 400)
    r = requests.get(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {tok}"},
        timeout=15,
    )
    if r.status_code != 200:
        raise OAuthError("Invalid or expired Google token", 401)
    data = r.json()
    sub = data.get("sub")
    if not sub:
        raise OAuthError("Google did not return a user id", 401)
    email = (data.get("email") or "").strip().lower()
    if not email:
        raise OAuthError(
            "Google did not return an email. Ensure the token includes email scope (openid email profile).",
            400,
        )
    return {
        "sub": str(sub),
        "email": email,
        "name": (data.get("name") or "").strip() or None,
    }


def facebook_placeholder_email(facebook_user_id: str) -> str:
    raw = "".join(c for c in str(facebook_user_id).strip() if c.isdigit())
    if not raw:
        raw = "0"
    return f"fb{raw}{FACEBOOK_OAUTH_EMAIL_SUFFIX}"


def is_reserved_facebook_placeholder_email(email: str) -> bool:
    return email.strip().lower().endswith(FACEBOOK_OAUTH_EMAIL_SUFFIX)


def verify_facebook_access_token(access_token: str) -> Dict[str, Any]:
    app_id = os.environ.get("FACEBOOK_APP_ID", "").strip()
    app_secret = os.environ.get("FACEBOOK_APP_SECRET", "").strip()
    if not app_id:
        raise OAuthError("Facebook sign-in is not configured on this server", 503)
    # Only `id` — enough for account linking and synthetic email. Do not request `email`
    # (see module note above). We do not read name/picture from Graph.
    params: Dict[str, str] = {
        "fields": "id",
        "access_token": access_token,
    }
    if app_secret:
        import hashlib
        import hmac

        proof = hmac.new(
            app_secret.encode("utf-8"),
            msg=access_token.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()
        params["appsecret_proof"] = proof
    r = requests.get(
        "https://graph.facebook.com/v18.0/me",
        params=params,
        timeout=15,
    )
    if r.status_code != 200:
        raise OAuthError("Invalid Facebook token", 401)
    data = r.json()
    if "error" in data:
        raise OAuthError("Invalid Facebook token", 401)
    uid = data.get("id")
    if not uid:
        raise OAuthError("Facebook did not return a user id", 401)
    uid_s = str(uid)
    # If `email` is ever present (legacy token / app config), prefer it; else synthetic.
    email = (data.get("email") or "").strip().lower()
    if not email:
        email = facebook_placeholder_email(uid_s)
    return {
        "sub": uid_s,
        "email": email,
        "name": None,
    }
