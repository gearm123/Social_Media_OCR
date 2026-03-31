import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from main import run_pipeline_job


BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Translate Chat API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        _write_status(job_id, status="running", stage="pipeline")
        result = run_pipeline_job(
            image_paths=image_paths,
            work_dir=_job_dir(job_id),
            language=language,
            bubble_summary_text=bubble_summary_text,
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


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/jobs")
async def create_job(
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
    for idx, upload in enumerate(files):
        suffix = Path(upload.filename or f"image_{idx}.png").suffix or ".png"
        safe_name = f"{idx:03d}{suffix.lower()}"
        dest = input_dir / safe_name
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        image_paths.append(str(dest))

    status = _write_status(
        job_id,
        status="queued",
        stage="queued",
        created_at=_utc_now(),
        language=language,
        images_count=len(image_paths),
        bubble_summary_text=bubble_summary_text,
    )
    Thread(
        target=_run_job,
        args=(job_id, image_paths, language, bubble_summary_text),
        daemon=True,
    ).start()
    status["artifact_urls"] = {}
    return status


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    return _load_status(job_id)


@app.get("/jobs/{job_id}/results")
def get_job_results(job_id: str):
    status = _load_status(job_id)
    if status.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed yet")
    return {
        "job_id": job_id,
        "status": status.get("status"),
        "result": status.get("result"),
        "artifact_urls": status.get("artifact_urls") or {},
    }


@app.get("/jobs/{job_id}/artifacts/{artifact_name}")
def get_job_artifact(job_id: str, artifact_name: str):
    status = _load_status(job_id)
    result = status.get("result") or {}
    artifacts = result.get("artifacts") or {}
    path = artifacts.get(artifact_name)
    if not path:
        raise HTTPException(status_code=404, detail="Artifact not found")
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file is missing")
    return FileResponse(artifact_path)
