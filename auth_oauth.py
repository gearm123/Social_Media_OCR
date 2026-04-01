"""Verify OAuth tokens from Google, Facebook, and Apple (Sign in with Apple)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import jwt
import requests

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


def verify_facebook_access_token(access_token: str) -> Dict[str, Any]:
    app_id = os.environ.get("FACEBOOK_APP_ID", "").strip()
    app_secret = os.environ.get("FACEBOOK_APP_SECRET", "").strip()
    if not app_id:
        raise OAuthError("Facebook sign-in is not configured on this server", 503)
    params: Dict[str, str] = {
        "fields": "id,email,name",
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
    email = (data.get("email") or "").strip().lower()
    if not email:
        raise OAuthError(
            "Facebook did not return an email. Ensure the app requests the email permission "
            "and that your Facebook account has an email on file.",
            400,
        )
    return {
        "sub": str(uid),
        "email": email,
        "name": (data.get("name") or "").strip() or None,
    }


def verify_apple_id_token(id_token_str: str) -> Dict[str, Any]:
    client_id = os.environ.get("APPLE_CLIENT_ID", "").strip()
    if not client_id:
        raise OAuthError("Apple sign-in is not configured on this server", 503)
    try:
        jwks_client = jwt.PyJWKClient("https://appleid.apple.com/auth/keys")
        signing_key = jwks_client.get_signing_key_from_jwt(id_token_str)
        data = jwt.decode(
            id_token_str,
            signing_key.key,
            algorithms=["RS256"],
            audience=client_id,
            issuer="https://appleid.apple.com",
        )
    except jwt.PyJWTError as e:
        raise OAuthError(f"Invalid Apple token: {e}") from e
    sub = data.get("sub")
    if not sub:
        raise OAuthError("Invalid Apple token: missing subject")
    email = (data.get("email") or "").strip().lower()
    if not email:
        raise OAuthError(
            "Apple did not include an email in this sign-in. "
            "Use “Share my email” on first sign-in, or sign in with Google/email.",
            400,
        )
    return {
        "sub": str(sub),
        "email": email,
        "name": None,
    }
