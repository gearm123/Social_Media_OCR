import argparse
import json
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from web_app import app


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = BASE_DIR / "input_images"
LOCAL_TEST_INPUT_DIR = BASE_DIR / "tests" / "local_inputs"


def _load_images(images_dir: Path):
    images = [
        p for p in sorted(images_dir.glob("*"))
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ]
    if not images:
        raise FileNotFoundError(f"No image files found in {images_dir}")
    return images


def _default_bubble_summary():
    bubble_file = BASE_DIR / "pass1_bubble_input.txt"
    if bubble_file.exists():
        return bubble_file.read_text(encoding="utf-8").strip()
    return None


def main():
    parser = argparse.ArgumentParser(description="Run a local backend smoke test against /jobs")
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Directory of local sample screenshots. Default: tests/local_inputs, then input_images",
    )
    parser.add_argument(
        "--source-language",
        default="th",
        help="Optional source language hint sent to the backend as form field 'language'",
    )
    parser.add_argument(
        "--bubble-summary-file",
        default=None,
        help="Optional path to a bubble summary text file",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="Polling interval while waiting for job completion",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=900.0,
        help="Max time to wait for the job to complete",
    )
    args = parser.parse_args()

    if args.images_dir:
        images_dir = Path(args.images_dir)
    elif any(LOCAL_TEST_INPUT_DIR.iterdir()):
        images_dir = LOCAL_TEST_INPUT_DIR
    else:
        images_dir = DEFAULT_INPUT_DIR

    images = _load_images(images_dir)
    if args.bubble_summary_file:
        bubble_summary_text = Path(args.bubble_summary_file).read_text(encoding="utf-8").strip()
    else:
        bubble_summary_text = _default_bubble_summary()

    print(f"[TEST] Using images from: {images_dir}")
    print(f"[TEST] Found {len(images)} image(s)")

    files = []
    handles = []
    try:
        for image_path in images:
            fh = image_path.open("rb")
            handles.append(fh)
            files.append(("files", (image_path.name, fh, "application/octet-stream")))

        data = {}
        if args.source_language:
            data["language"] = args.source_language
        if bubble_summary_text:
            data["bubble_summary_text"] = bubble_summary_text

        client = TestClient(app)
        response = client.post("/jobs", files=files, data=data)
        response.raise_for_status()
        payload = response.json()
        print("[TEST] Job created:")
        print(json.dumps(payload, indent=2, ensure_ascii=False))

        job_id = payload["job_id"]
        deadline = time.time() + args.timeout_seconds
        while time.time() < deadline:
            status_resp = client.get(f"/jobs/{job_id}")
            status_resp.raise_for_status()
            status = status_resp.json()
            print(f"[TEST] status={status.get('status')} stage={status.get('stage')}")
            if status.get("status") == "completed":
                results_resp = client.get(f"/jobs/{job_id}/results")
                results_resp.raise_for_status()
                results = results_resp.json()
                print("[TEST] Job completed:")
                print(json.dumps(results, indent=2, ensure_ascii=False))
                return
            if status.get("status") == "failed":
                print("[TEST] Job failed:")
                print(json.dumps(status, indent=2, ensure_ascii=False))
                sys.exit(1)
            time.sleep(args.poll_seconds)
    finally:
        for fh in handles:
            fh.close()

    print("[TEST] Timed out waiting for job completion")
    sys.exit(1)


if __name__ == "__main__":
    main()
