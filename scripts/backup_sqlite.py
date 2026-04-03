#!/usr/bin/env python3
"""Copy the billing/users SQLite DB to a timestamped file. For cron or manual backups.

Environment:
  USER_DB_PATH  — source DB (default: ../data/users.sqlite3 relative to repo root)
  BACKUP_DIR    — destination directory (default: current working directory)

Example (Render shell / cron):
  USER_DB_PATH=/data/users.sqlite3 BACKUP_DIR=/data/backups python scripts/backup_sqlite.py
"""

from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    raw = os.environ.get("USER_DB_PATH", "").strip()
    db = Path(raw) if raw else REPO_ROOT / "data" / "users.sqlite3"
    if not db.is_file():
        print(f"No database file at {db}", file=sys.stderr)
        return 1
    out_dir = Path(os.environ.get("BACKUP_DIR", ".").strip() or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = out_dir / f"users_backup_{stamp}.sqlite3"
    shutil.copy2(db, dest)
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
