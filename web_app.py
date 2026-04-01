import json
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Annotated, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from auth_api import router as auth_router
from auth_deps import assert_job_readable, get_current_user_optional, get_job_user
from main import run_pipeline_job
from user_store import UserRecord


BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
SERVER_TEST_INPUTS = BASE_DIR / "server_test_inputs"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Translate Chat API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth", tags=["auth"])


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


def _run_job(job_id: str, image_paths: list[str], language: Optional[str], bubble_summary_text: Optional[str]):
    try:
        _write_status(job_id, status="running", stage="starting")
        result = run_pipeline_job(
            image_paths=image_paths,
            work_dir=_job_dir(job_id),
            language=language,
            bubble_summary_text=bubble_summary_text,
            on_stage=lambda s: _write_status(job_id, status="running", stage=s),
        )
        _write_status(
            job_id,
            status="completed",
            stage="completed",
            result=result,
            artifact_urls=_artifact_urls(job_id, result.get("artifacts") or {}),
        )
    except Exception as exc:
        _write_status(
            job_id,
            status="failed",
            stage="failed",
            error=str(exc),
            traceback=traceback.format_exc(),
        )


def _check_smoke_secret(x_smoke_secret: Optional[str]) -> None:
    expected = os.environ.get("SMOKE_TEST_SECRET", "").strip()
    if not expected:
        return
    if (x_smoke_secret or "").strip() != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Smoke-Secret")


@app.get("/")
def root():
    return {
        "service": "translate-chat-api",
        "health": "GET /health",
        "docs": "GET /docs",
        "auth": {
            "register": "POST /auth/register JSON {email, password}",
            "login": "POST /auth/login JSON {email, password} → Bearer token",
            "me": "GET /auth/me Authorization: Bearer <token>",
        },
        "create_job": "POST /jobs (multipart: files + optional language, bubble_summary_text)",
        "job_auth": "Set REQUIRE_AUTH_FOR_JOBS=1 to require Bearer token; jobs are scoped to the user",
        "smoke_test": "POST /test/smoke (optional Form: language; Header X-Smoke-Secret if env SMOKE_TEST_SECRET is set)",
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/jobs")
async def create_job(
    job_user: Annotated[Optional[UserRecord], Depends(get_job_user)],
    files: list[UploadFile] = File(...),
    language: Optional[str] = Form(default=None),
    bubble_summary_text: Optional[str] = Form(default=None),
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one image file is required")

    job_id = uuid4().hex
    job_dir = _job_dir(job_id)
    input_dir = job_dir / "input_images"
    input_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    intake_rows = []
    for idx, upload in enumerate(files):
        orig_name = upload.filename or f"image_{idx}.png"
        suffix = Path(orig_name).suffix or ".png"
        safe_name = f"{idx:03d}{suffix.lower()}"
        dest = input_dir / safe_name
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        image_paths.append(str(dest))
        intake_rows.append((idx, orig_name, dest.stat().st_size))

    bubble_preview = (bubble_summary_text or "").strip()
    bubble_line_count = len([ln for ln in bubble_preview.splitlines() if ln.strip()])
    print(f"[JOB_INTAKE] job_id={job_id} images_saved={len(image_paths)} language={language!r}", flush=True)
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
        created_at=_utc_now(),
        language=language,
        images_count=len(image_paths),
        bubble_summary_text=bubble_summary_text,
        user_id=(job_user.id if job_user else None),
    )
    Thread(
        target=_run_job,
        args=(job_id, image_paths, language, bubble_summary_text),
        daemon=True,
    ).start()
    status["artifact_urls"] = {}
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
        args=(job_id, image_paths, language, None),
        daemon=True,
    ).start()
    status["artifact_urls"] = {}
    status["note"] = "Uses files from server_test_inputs/; poll GET /jobs/{job_id} until status is completed or failed"
    return status


@app.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    user: Annotated[Optional[UserRecord], Depends(get_current_user_optional)],
):
    status = _load_status(job_id)
    assert_job_readable(status, user)
    return status


@app.get("/jobs/{job_id}/results")
def get_job_results(
    job_id: str,
    user: Annotated[Optional[UserRecord], Depends(get_current_user_optional)],
):
    status = _load_status(job_id)
    assert_job_readable(status, user)
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
):
    status = _load_status(job_id)
    assert_job_readable(status, user)
    result = status.get("result") or {}
    artifacts = result.get("artifacts") or {}
    path = artifacts.get(artifact_name)
    if not path:
        raise HTTPException(status_code=404, detail="Artifact not found")
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file is missing")
    return FileResponse(artifact_path)
