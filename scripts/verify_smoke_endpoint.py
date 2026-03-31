"""Local check: OpenAPI lists POST /test/smoke and POST returns queued job (no Gemini run required for this script)."""
import os
import sys

# Repo root = parent of scripts/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.chdir(ROOT)


def main() -> int:
    from fastapi.testclient import TestClient

    import web_app

    if os.environ.get("SMOKE_TEST_SECRET", "").strip():
        print("Unset SMOKE_TEST_SECRET for this script, or pass header in your own test.")
        return 1

    client = TestClient(web_app.app)
    spec = client.get("/openapi.json")
    assert spec.status_code == 200, spec.text
    paths = spec.json().get("paths") or {}
    if "/test/smoke" not in paths:
        print("FAIL: /test/smoke missing from OpenAPI. Registered paths:", sorted(paths.keys()))
        return 1
    post_op = paths["/test/smoke"].get("post")
    if not post_op:
        print("FAIL: /test/smoke has no POST operation")
        return 1

    r = client.post("/test/smoke", data={})
    if r.status_code != 200:
        print("FAIL: POST /test/smoke ->", r.status_code, r.text)
        return 1
    body = r.json()
    if not body.get("job_id"):
        print("FAIL: no job_id in response", body)
        return 1
    if body.get("status") != "queued":
        print("WARN: expected status queued, got", body.get("status"))

    print("OK: POST /test/smoke is registered and accepts requests.")
    print("    job_id:", body.get("job_id"))
    print("    (Pipeline runs in a background thread; poll GET /jobs/{job_id} if needed.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
