import json
import logging
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Annotated, Optional
from uuid import uuid4

BASE_DIR = Path(__file__).resolve().parent
if os.environ.get("RENDER", "").strip().lower() != "true":
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from starlette.responses import Response

from activity_log import actor_fields, read_activity_summary, read_recent_activity, write_activity
from auth_api import router as auth_router
from auth_deps import assert_job_readable, get_current_user_optional, get_job_user
from billing_api import paddle_webhook_handler, router as billing_router
from billing_store import (
    billing_enforce_enabled,
    billing_exempt_user,
    get_billing_store,
    normalize_guest_key,
)
from db_postgres import raw_database_url_from_environment
from rate_limit import RateLimitMiddleware
from user_store import UserRecord
from usage_report import (
    note_algorithm_completed,
    note_algorithm_failed,
    note_algorithm_started,
    note_free_trial_attempt,
    read_usage_report,
)

_log = logging.getLogger("translate_chat.web")

_sentry_dsn = os.environ.get("SENTRY_DSN", "").strip()
if _sentry_dsn:
    import sentry_sdk

    sentry_sdk.init(
        dsn=_sentry_dsn,
        traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0") or "0"),
    )

TERMS_HTML_PATH = BASE_DIR / "legal" / "terms.html"
PRIVACY_HTML_PATH = BASE_DIR / "legal" / "privacy.html"
JOBS_DIR = BASE_DIR / "jobs"
SERVER_TEST_INPUTS = BASE_DIR / "server_test_inputs"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _cors_allow_origins() -> list[str]:
    """Origins for CORSMiddleware. Browsers send Origin without a trailing slash."""
    raw = os.environ.get("CORS_ORIGINS", "").strip()
    if raw:
        out = [x.strip().rstrip("/") for x in raw.split(",") if x.strip()]
        if out:
            return out
        # e.g. CORS_ORIGINS="," or whitespace-only entries — treat as unset (see README).
    fu = os.environ.get("FRONTEND_URL", "").strip().rstrip("/")
    if fu:
        return [fu]
    return ["*"]


app = FastAPI(title="Translate Chat API", version="0.1.0")


@app.on_event("startup")
def _log_runtime_env_for_debug():
    """Confirms deploy sees DATABASE_URL (value is never logged)."""
    has_db = bool(raw_database_url_from_environment())
    _log.info(
        "Runtime env: RENDER=%r DATABASE_URL_non_empty=%s RENDER_GIT_COMMIT=%r",
        os.environ.get("RENDER"),
        has_db,
        os.environ.get("RENDER_GIT_COMMIT"),
    )
    if os.environ.get("RENDER", "").strip().lower() == "true" and not has_db:
        _log.warning(
            "DATABASE_URL is empty. Link translate-chat-db to this web service or set DATABASE_URL "
            "to the Postgres internal URL."
        )


app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(billing_router, prefix="/billing", tags=["billing"])


@app.post("/webhook/paddle")
async def paddle_webhook_alias(request: Request):
    """Some Paddle setups use this path; canonical URL is POST /billing/webhook."""
    return await paddle_webhook_handler(request)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"


def _load_status(job_id: str) -> dict:
    path = _status_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _make_pipeline_stage_writer(job_id: str):
    """Emit structured pipeline phases; refresh phase_started_at when phase id changes."""
    last_phase = {"v": None}

    def _cb(payload: dict) -> None:
        phase = str(payload.get("phase") or "running")
        extra: dict = {}
        if phase != last_phase["v"] or bool(payload.get("reset_phase_started_at")):
            last_phase["v"] = phase
            extra["phase_started_at"] = _utc_now()
        _write_status(
            job_id,
            status="running",
            stage=phase,
            stage_label=payload.get("label"),
            progress=payload.get("progress"),
            pipeline_elapsed_sec=payload.get("elapsed_sec"),
            eta_extra_sec=payload.get("eta_extra_sec"),
            **extra,
        )

    return _cb


def _write_status(job_id: str, **fields) -> dict:
    path = _status_path(job_id)
    current = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(fields)
    current["job_id"] = job_id
    current["updated_at"] = _utc_now()
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    return current


def _artifact_urls(job_id: str, artifacts: dict) -> dict:
    out = {}
    for key in artifacts or {}:
        out[key] = f"/jobs/{job_id}/artifacts/{key}"
    return out


def _cancel_flag_path(job_id: str) -> Path:
    return _job_dir(job_id) / ".cancel_requested"


def _parse_job_difficulty(raw: Optional[str]) -> int:
    """Multipart ``difficulty`` field; invalid or missing → 3 (matches CLI default)."""
    if raw is None or not str(raw).strip():
        return 3
    try:
        n = int(str(raw).strip())
    except ValueError:
        return 3
    return max(1, min(3, n))


def _parse_hurry_up_form(raw: Optional[str]) -> bool:
    """Same semantics as CLI ``--hurry-up``: only explicit truthy values enable hurry mode."""
    if raw is None:
        return False
    s = str(raw).strip().lower()
    if not s:
        return False
    return s in ("1", "true", "yes", "on")


def _run_job(
    job_id: str,
    image_paths: list[str],
    language: Optional[str],
    bubble_summary_text: Optional[str],
    billing_user_id: Optional[str] = None,
    billing_consumption: Optional[str] = None,
    billing_guest_key: Optional[str] = None,
    difficulty: int = 3,
    hurry_up: bool = False,
):
    # Import here so the web process binds to $PORT quickly (Render port scan).
    # Loading ``main`` pulls CV2, Gemini, Vision — too slow for module-level import.
    from main import JobCancelledError, run_pipeline_job

    try:
        note_algorithm_started()
        _write_status(
            job_id,
            status="running",
            stage="starting",
            stage_label="Starting…",
            progress=0.0,
            pipeline_elapsed_sec=0.0,
            eta_extra_sec=0.0,
            phase_started_at=_utc_now(),
        )
        result = run_pipeline_job(
            image_paths=image_paths,
            work_dir=_job_dir(job_id),
            language=language,
            bubble_summary_text=bubble_summary_text,
            on_stage=_make_pipeline_stage_writer(job_id),
            cancel_check=lambda: _cancel_flag_path(job_id).exists(),
            difficulty=difficulty,
            hurry_up=hurry_up,
        )
        _write_status(
            job_id,
            status="completed",
            stage="completed",
            stage_label="Final image ready",
            progress=1.0,
            pipeline_elapsed_sec=result.get("total_runtime_sec"),
            result=result,
            artifact_urls=_artifact_urls(job_id, result.get("artifacts") or {}),
        )
        store = get_billing_store(BASE_DIR)
        if billing_guest_key and billing_consumption in ("guest_free", "guest_credit"):
            store.guest_apply_successful_job(
                billing_guest_key, billing_consumption, job_id=job_id
            )
        elif billing_user_id and billing_consumption:
            store.apply_successful_job(
                billing_user_id, billing_consumption, job_id=job_id
            )
        write_activity(
            "job_completed",
            job_id=job_id,
            images_count=len(image_paths),
            language=language,
            duration_sec=result.get("total_runtime_sec"),
            billing_consumption=billing_consumption,
            artifacts_count=len(result.get("artifacts") or {}),
            **actor_fields(user_id=billing_user_id, guest_key=billing_guest_key),
        )
        note_algorithm_completed(result.get("pass_outcomes"))
    except JobCancelledError:
        _write_status(
            job_id,
            status="cancelled",
            stage="cancelled",
            error="Cancelled by user",
        )
        flag = _cancel_flag_path(job_id)
        if flag.exists():
            try:
                flag.unlink()
            except OSError:
                pass
    except Exception as exc:
        pass_outcomes = {}
        try:
            from ocr_translate import get_gemini_pass_outcomes

            raw_outcomes = get_gemini_pass_outcomes()
            pass_outcomes = {
                f"pass{pn}": meta for pn, meta in raw_outcomes.items() if isinstance(meta, dict)
            }
        except Exception:
            pass
        prior_status = {}
        try:
            prior_status = _load_status(job_id)
        except HTTPException:
            prior_status = {}
        if pass_outcomes:
            try:
                print(
                    "[pipeline] job failure pass outcomes: "
                    + json.dumps(pass_outcomes, ensure_ascii=False, sort_keys=True),
                    flush=True,
                )
            except Exception:
                pass
        _write_status(
            job_id,
            status="failed",
            stage="failed",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        write_activity(
            "job_failed",
            job_id=job_id,
            images_count=len(image_paths),
            language=language,
            billing_consumption=billing_consumption,
            failure_stage=prior_status.get("stage"),
            error_summary=str(exc),
            pass_outcomes=pass_outcomes,
            **actor_fields(user_id=billing_user_id, guest_key=billing_guest_key),
        )
        note_algorithm_failed(pass_outcomes)


def _billing_block_detail(code: str) -> dict:
    messages = {
        "free_exhausted": "No free runs remaining; purchase a plan or credits.",
        "multi_requires_plan": "Multiple images require an active subscription, quota, or job credits.",
        "quota_exhausted": "Monthly plan runs used. Try again next calendar month or buy a single-run credit.",
        "no_files": "At least one image is required.",
    }
    return {"code": code, "message": messages.get(code, code)}


def _check_smoke_secret(x_smoke_secret: Optional[str]) -> None:
    expected = os.environ.get("SMOKE_TEST_SECRET", "").strip()
    if not expected:
        return
    if (x_smoke_secret or "").strip() != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Smoke-Secret")


def _check_monitor_token(x_monitor_token: Optional[str]) -> None:
    expected = os.environ.get("MONITOR_READ_TOKEN", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="MONITOR_READ_TOKEN is not configured")
    if (x_monitor_token or "").strip() != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Monitor-Token")


@app.head("/")
def root_head():
    """Render port checks may send HEAD /; avoid 405 so the instance is detected as listening."""
    return Response(status_code=200)


@app.get("/")
def root():
    return {
        "service": "translate-chat-api",
        "health": "GET /health",
        "docs": "GET /docs",
        "auth": {
            "providers": "GET /auth/providers — OAuth client IDs for the SPA",
            "register": "POST /auth/register JSON {username, email, password}",
            "login": "POST /auth/login JSON {email, password} → Bearer token",
            "oauth_google": "POST /auth/oauth/google or /auth/oauth/gsi JSON {id_token} or {access_token}",
            "oauth_facebook": "POST /auth/oauth/facebook or /auth/oauth/fb JSON {access_token}",
            "me": "GET /auth/me Authorization: Bearer <token>",
        },
        "create_job": "POST /jobs (multipart: files + optional language, bubble_summary_text, difficulty 1–3, hurry_up 1/true when set — default hurry_up off)",
        "cancel_job": "POST /jobs/{job_id}/cancel — request cooperative cancel (checked between pipeline stages)",
        "job_auth": "Set REQUIRE_AUTH_FOR_JOBS=1 to require Bearer token; jobs are scoped to the user",
        "billing": "GET /billing/status, GET /billing/me, GET /billing/guest-status (X-Guest-Billing-Id), POST /billing/checkout-session, POST /billing/guest-checkout-session (one-time, guest + email), POST /billing/guest-claim-transaction, POST /billing/user-claim-transaction (Bearer + txn, single/debug), POST /billing/portal-session, POST /billing/webhook (alias POST /webhook/paddle). BILLING_ENFORCE=1: POST /jobs needs Bearer or X-Guest-Billing-Id",
        "legal": "GET /legal/terms, GET /legal/privacy (set PUBLIC_CONTACT_EMAIL)",
        "smoke_test": "POST /test/smoke (optional Form: language; Header X-Smoke-Secret if env SMOKE_TEST_SECRET is set)",
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "database_url_configured": bool(raw_database_url_from_environment()),
        "render_git_commit": os.environ.get("RENDER_GIT_COMMIT"),
    }


@app.get("/monitor/activity")
def monitor_activity(
    limit: int = Query(default=100, ge=1, le=500),
    event: Optional[str] = Query(default=None),
    x_monitor_token: Optional[str] = Header(default=None, alias="X-Monitor-Token"),
):
    _check_monitor_token(x_monitor_token)
    events = read_recent_activity(limit=limit, event=event)
    return {
        "ok": True,
        "storage": "postgres",
        "events": events,
        "summary": read_activity_summary(event=event),
    }


@app.get("/monitor/usage")
def monitor_usage(
    x_monitor_token: Optional[str] = Header(default=None, alias="X-Monitor-Token"),
):
    _check_monitor_token(x_monitor_token)
    usage = read_usage_report()
    return {
        "ok": True,
        "database_configured": bool(usage),
        "usage": usage,
    }


@app.get("/legal/terms", response_class=HTMLResponse)
def terms_of_service():
    """Public Terms of Service for storefront / payment provider onboarding (e.g. Paddle)."""
    if not TERMS_HTML_PATH.is_file():
        raise HTTPException(status_code=404, detail="Terms page not found")
    html = TERMS_HTML_PATH.read_text(encoding="utf-8")
    contact = os.environ.get("PUBLIC_CONTACT_EMAIL", "").strip() or "UPDATE_YOUR_EMAIL@example.com"
    return html.replace("{{CONTACT_EMAIL}}", contact)


@app.get("/legal/privacy", response_class=HTMLResponse)
def privacy_policy():
    """Public privacy policy (starter template)."""
    if not PRIVACY_HTML_PATH.is_file():
        raise HTTPException(status_code=404, detail="Privacy page not found")
    html = PRIVACY_HTML_PATH.read_text(encoding="utf-8")
    contact = os.environ.get("PUBLIC_CONTACT_EMAIL", "").strip() or "UPDATE_YOUR_EMAIL@example.com"
    return html.replace("{{CONTACT_EMAIL}}", contact)


def _max_job_files() -> int:
    try:
        n = int(os.environ.get("MAX_JOB_FILES", "30"))
    except ValueError:
        n = 30
    return max(1, min(n, 80))


def _max_job_upload_bytes() -> int:
    try:
        mb = int(os.environ.get("MAX_JOB_UPLOAD_MB", "80"))
    except ValueError:
        mb = 80
    mb = max(1, min(mb, 500))
    return mb * 1024 * 1024


@app.post("/jobs")
async def create_job(
    job_user: Annotated[Optional[UserRecord], Depends(get_job_user)],
    files: list[UploadFile] = File(...),
    language: Optional[str] = Form(default=None),
    bubble_summary_text: Optional[str] = Form(default=None),
    difficulty: Optional[str] = Form(default=None),
    hurry_up: Optional[str] = Form(default=None),
    x_guest_billing_id: Annotated[Optional[str], Header(alias="X-Guest-Billing-Id")] = None,
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one image file is required")

    difficulty_i = _parse_job_difficulty(difficulty)
    hurry_up_b = _parse_hurry_up_form(hurry_up)

    max_files = _max_job_files()
    if len(files) > max_files:
        raise HTTPException(
            status_code=400,
            detail=f"Too many images in one job (maximum {max_files}).",
        )

    billing_consumption: Optional[str] = None
    billing_guest_key: Optional[str] = None
    bill = get_billing_store(BASE_DIR)
    guest_key_opt = normalize_guest_key(x_guest_billing_id) if job_user is None else None

    # Signed-in: always run entitlement checks and set billing_consumption so successful jobs
    # call apply_successful_job (free_runs_used, credits, sub quota). This applies even when
    # BILLING_ENFORCE is off — otherwise accounts could run unlimited single-image jobs.
    if job_user is not None:
        if billing_exempt_user(job_user.email, job_user.username):
            billing_consumption = "unlimited"
        else:
            ok, consumption, err = bill.can_start_job(job_user.id, len(files))
            if not ok:
                raise HTTPException(
                    status_code=402,
                    detail=_billing_block_detail(err),
                )
            billing_consumption = consumption
    elif billing_enforce_enabled():
        if not guest_key_opt:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Sign in with Authorization: Bearer, or send header X-Guest-Billing-Id "
                    "(8–64 hex characters, e.g. UUID without dashes) for guest free runs"
                ),
            )
        ok, consumption, err = bill.guest_can_start_job(guest_key_opt, len(files))
        if not ok:
            raise HTTPException(
                status_code=402,
                detail=_billing_block_detail(err),
            )
        billing_guest_key = guest_key_opt
        billing_consumption = consumption
    elif guest_key_opt:
        # Guest jobs: record Paddle-style consumption on success even when BILLING_ENFORCE is off,
        # so the SPA and GET /billing/guest-status stay aligned with completed runs.
        ok, consumption, _err = bill.guest_can_start_job(guest_key_opt, len(files))
        billing_guest_key = guest_key_opt
        if ok:
            billing_consumption = consumption

    job_id = uuid4().hex
    job_dir = _job_dir(job_id)
    input_dir = job_dir / "input_images"
    max_bytes = _max_job_upload_bytes()
    image_paths: list[str] = []
    intake_rows: list[tuple[int, str, int]] = []
    total_written = 0

    try:
        input_dir.mkdir(parents=True, exist_ok=True)
        for idx, upload in enumerate(files):
            orig_name = upload.filename or f"image_{idx}.png"
            suffix = Path(orig_name).suffix or ".png"
            safe_name = f"{idx:03d}{suffix.lower()}"
            dest = input_dir / safe_name
            with dest.open("wb") as f:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    total_written += len(chunk)
                    if total_written > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Total upload exceeds {max_bytes // (1024 * 1024)} MB limit.",
                        )
                    f.write(chunk)
            nbytes = dest.stat().st_size
            image_paths.append(str(dest))
            intake_rows.append((idx, orig_name, nbytes))
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    bubble_preview = (bubble_summary_text or "").strip()
    bubble_line_count = len([ln for ln in bubble_preview.splitlines() if ln.strip()])
    print(
        f"[JOB_INTAKE] job_id={job_id} images_saved={len(image_paths)} language={language!r} "
        f"difficulty={difficulty_i} hurry_up={hurry_up_b}",
        flush=True,
    )
    for idx, orig_name, nbytes in intake_rows:
        print(f"[JOB_INTAKE]   input[{idx}] original_name={orig_name!r} bytes_on_disk={nbytes}", flush=True)
    if bubble_preview:
        print(
            f"[JOB_INTAKE] bubble_summary_text lines={bubble_line_count} chars={len(bubble_preview)}:",
            flush=True,
        )
        print(bubble_preview, flush=True)
    else:
        print("[JOB_INTAKE] bubble_summary_text: (none)", flush=True)

    status = _write_status(
        job_id,
        status="queued",
        stage="queued",
        stage_label="Queued — waiting to start…",
        progress=0.0,
        pipeline_elapsed_sec=0.0,
        eta_extra_sec=0.0,
        phase_started_at=_utc_now(),
        created_at=_utc_now(),
        language=language,
        images_count=len(image_paths),
        bubble_summary_text=bubble_summary_text,
        difficulty=difficulty_i,
        hurry_up=hurry_up_b,
        user_id=(job_user.id if job_user else None),
        billing_consumption=billing_consumption,
        billing_guest_key=billing_guest_key,
    )
    Thread(
        target=_run_job,
        args=(
            job_id,
            image_paths,
            language,
            bubble_summary_text,
            job_user.id if job_user else None,
            billing_consumption,
            billing_guest_key,
            difficulty_i,
            hurry_up_b,
        ),
        daemon=True,
    ).start()
    status["artifact_urls"] = {}
    if billing_consumption in ("free", "guest_free"):
        note_free_trial_attempt()
    write_activity(
        "job_created",
        job_id=job_id,
        images_count=len(image_paths),
        language=language,
        difficulty=difficulty_i,
        hurry_up=hurry_up_b,
        billing_consumption=billing_consumption,
        **actor_fields(user_id=(job_user.id if job_user else None), guest_key=billing_guest_key),
    )
    return status


@app.post("/test/smoke")
async def smoke_job(
    job_user: Annotated[Optional[UserRecord], Depends(get_job_user)],
    language: Optional[str] = Form(default=None),
    x_smoke_secret: Optional[str] = Header(default=None, alias="X-Smoke-Secret"),
):
    """Run one pipeline job using committed files under ``server_test_inputs/`` (for deployed checks)."""
    _check_smoke_secret(x_smoke_secret)
    if not SERVER_TEST_INPUTS.is_dir():
        raise HTTPException(status_code=500, detail="server_test_inputs directory is missing")

    candidates = sorted(
        p
        for p in SERVER_TEST_INPUTS.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not candidates:
        raise HTTPException(
            status_code=500,
            detail="No sample images in server_test_inputs (add .png / .jpg files)",
        )

    job_id = uuid4().hex
    job_dir = _job_dir(job_id)
    input_dir = job_dir / "input_images"
    input_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for idx, src in enumerate(candidates):
        suffix = src.suffix.lower() or ".png"
        dest = input_dir / f"{idx:03d}{suffix}"
        shutil.copy2(src, dest)
        image_paths.append(str(dest))

    src_pass1 = SERVER_TEST_INPUTS / "pass1_bubble_input.txt"
    if src_pass1.exists():
        shutil.copy2(src_pass1, job_dir / "pass1_bubble_input.txt")

    status = _write_status(
        job_id,
        status="queued",
        stage="queued",
        created_at=_utc_now(),
        language=language,
        images_count=len(image_paths),
        smoke_test=True,
        user_id=(job_user.id if job_user else None),
    )
    Thread(
        target=_run_job,
        args=(job_id, image_paths, language, None, None, None, None, 3, False),
        daemon=True,
    ).start()
    status["artifact_urls"] = {}
    status["note"] = "Uses files from server_test_inputs/; poll GET /jobs/{job_id} until status is completed or failed"
    return status


@app.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    user: Annotated[Optional[UserRecord], Depends(get_current_user_optional)],
    x_guest_billing_id: Annotated[Optional[str], Header(alias="X-Guest-Billing-Id")] = None,
):
    status = _load_status(job_id)
    assert_job_readable(status, user, x_guest_billing_id)
    return status


@app.post("/jobs/{job_id}/cancel")
def cancel_job(
    job_id: str,
    user: Annotated[Optional[UserRecord], Depends(get_current_user_optional)],
    x_guest_billing_id: Annotated[Optional[str], Header(alias="X-Guest-Billing-Id")] = None,
):
    """Signal the worker thread to stop at the next cancel check (between major pipeline stages)."""
    status = _load_status(job_id)
    assert_job_readable(status, user, x_guest_billing_id)
    st = status.get("status")
    if st in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=409, detail=f"Job already finished ({st})")
    _cancel_flag_path(job_id).touch()
    return {"job_id": job_id, "status": "cancelling"}


@app.get("/jobs/{job_id}/results")
def get_job_results(
    job_id: str,
    user: Annotated[Optional[UserRecord], Depends(get_current_user_optional)],
    x_guest_billing_id: Annotated[Optional[str], Header(alias="X-Guest-Billing-Id")] = None,
):
    status = _load_status(job_id)
    assert_job_readable(status, user, x_guest_billing_id)
    if status.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed yet")
    return {
        "job_id": job_id,
        "status": status.get("status"),
        "result": status.get("result"),
        "artifact_urls": status.get("artifact_urls") or {},
    }


@app.get("/jobs/{job_id}/artifacts/{artifact_name}")
def get_job_artifact(
    job_id: str,
    artifact_name: str,
    user: Annotated[Optional[UserRecord], Depends(get_current_user_optional)],
    x_guest_billing_id: Annotated[Optional[str], Header(alias="X-Guest-Billing-Id")] = None,
):
    status = _load_status(job_id)
    assert_job_readable(status, user, x_guest_billing_id)
    result = status.get("result") or {}
    artifacts = result.get("artifacts") or {}
    path = artifacts.get(artifact_name)
    if not path:
        raise HTTPException(status_code=404, detail="Artifact not found")
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file is missing")
    return FileResponse(artifact_path)
