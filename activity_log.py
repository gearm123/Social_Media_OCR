"""Backend activity monitor stored in PostgreSQL."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from psycopg.rows import dict_row

from db_postgres import get_pool

_WRITE_LOCK = Lock()
_MAX_STRING_LEN = 240
_MAX_LIST_ITEMS = 20
_MAX_DEPTH = 4
_MAX_EVENT_ROWS = 2000
_MONITOR_KEY = "backend"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(value: Any, max_len: int = _MAX_STRING_LEN) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def guest_ref(guest_key: Optional[str]) -> Optional[str]:
    if not guest_key or not str(guest_key).strip():
        return None
    digest = hashlib.sha256(str(guest_key).strip().lower().encode("utf-8")).hexdigest()
    return f"guest_{digest[:12]}"


def actor_fields(
    user_id: Optional[str] = None, guest_key: Optional[str] = None
) -> dict[str, Any]:
    uid = str(user_id or "").strip()
    if uid:
        return {"actor_type": "user", "user_id": uid}
    gref = guest_ref(guest_key)
    if gref:
        return {"actor_type": "guest", "guest_ref": gref}
    return {"actor_type": "anonymous"}


def _sanitize(value: Any, depth: int = 0) -> Any:
    if depth >= _MAX_DEPTH:
        return _clip(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _clip(value)
    if isinstance(value, Path):
        return _clip(value.as_posix())
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            clean_key = _clip(key, 64)
            out[clean_key] = _sanitize(item, depth + 1)
        return out
    if isinstance(value, (list, tuple, set)):
        items = list(value)[:_MAX_LIST_ITEMS]
        return [_sanitize(item, depth + 1) for item in items]
    return _clip(value)


class ActivityStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS backend_activity_summary (
                            monitor_key TEXT PRIMARY KEY,
                            total_events BIGINT NOT NULL DEFAULT 0,
                            latest_event_at TEXT,
                            latest_event JSONB
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS backend_activity_event_stats (
                            monitor_key TEXT NOT NULL,
                            event TEXT NOT NULL,
                            count BIGINT NOT NULL DEFAULT 0,
                            latest_at TEXT,
                            PRIMARY KEY (monitor_key, event)
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS backend_activity_events (
                            id BIGSERIAL PRIMARY KEY,
                            ts TEXT NOT NULL,
                            event TEXT NOT NULL,
                            payload JSONB NOT NULL
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_backend_activity_events_id_desc
                        ON backend_activity_events(id DESC)
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_backend_activity_events_event_id_desc
                        ON backend_activity_events(event, id DESC)
                        """
                    )
                conn.commit()

    def write(self, record: dict[str, Any]) -> None:
        payload_json = json.dumps(record, ensure_ascii=False)
        ts = str(record.get("ts") or "")
        event = str(record.get("event") or "")
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO backend_activity_events (ts, event, payload)
                        VALUES (%s, %s, %s::jsonb)
                        """,
                        (ts, event, payload_json),
                    )
                    cur.execute(
                        """
                        INSERT INTO backend_activity_summary (
                            monitor_key, total_events, latest_event_at, latest_event
                        )
                        VALUES (%s, 1, %s, %s::jsonb)
                        ON CONFLICT (monitor_key) DO UPDATE
                        SET total_events = backend_activity_summary.total_events + 1,
                            latest_event_at = EXCLUDED.latest_event_at,
                            latest_event = EXCLUDED.latest_event
                        """,
                        (_MONITOR_KEY, ts, payload_json),
                    )
                    cur.execute(
                        """
                        INSERT INTO backend_activity_event_stats (
                            monitor_key, event, count, latest_at
                        )
                        VALUES (%s, %s, 1, %s)
                        ON CONFLICT (monitor_key, event) DO UPDATE
                        SET count = backend_activity_event_stats.count + 1,
                            latest_at = EXCLUDED.latest_at
                        """,
                        (_MONITOR_KEY, event, ts),
                    )
                    cur.execute(
                        """
                        DELETE FROM backend_activity_events
                        WHERE id IN (
                            SELECT id
                            FROM backend_activity_events
                            ORDER BY id DESC
                            OFFSET %s
                        )
                        """,
                        (_MAX_EVENT_ROWS,),
                    )
                conn.commit()

    def recent_events(
        self, limit: int = 100, event: Optional[str] = None
    ) -> list[dict[str, Any]]:
        wanted = max(1, min(int(limit or 100), 500))
        event_filter = str(event or "").strip()
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    if event_filter:
                        cur.execute(
                            """
                            SELECT payload::text AS payload_json
                            FROM backend_activity_events
                            WHERE event = %s
                            ORDER BY id DESC
                            LIMIT %s
                            """,
                            (event_filter, wanted),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT payload::text AS payload_json
                            FROM backend_activity_events
                            ORDER BY id DESC
                            LIMIT %s
                            """,
                            (wanted,),
                        )
                    rows = cur.fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"])
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                out.append(payload)
        return out

    def summary(self, event: Optional[str] = None) -> dict[str, Any]:
        event_filter = str(event or "").strip()
        with self._lock:
            pool = get_pool()
            with pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    if event_filter:
                        cur.execute(
                            """
                            SELECT count, latest_at
                            FROM backend_activity_event_stats
                            WHERE monitor_key = %s AND event = %s
                            """,
                            (_MONITOR_KEY, event_filter),
                        )
                        stats_row = cur.fetchone()
                        cur.execute(
                            """
                            SELECT payload::text AS payload_json
                            FROM backend_activity_events
                            WHERE event = %s
                            ORDER BY id DESC
                            LIMIT 1
                            """,
                            (event_filter,),
                        )
                        latest_row = cur.fetchone()
                        latest_event = None
                        if latest_row and latest_row.get("payload_json"):
                            try:
                                latest_event = json.loads(latest_row["payload_json"])
                            except (TypeError, json.JSONDecodeError):
                                latest_event = None
                        total = int(stats_row["count"] or 0) if stats_row else 0
                        latest_at = (stats_row.get("latest_at") if stats_row else None) or (
                            latest_event.get("ts") if latest_event else None
                        )
                        counts_by_event = {event_filter: total} if stats_row else {}
                        latest_by_event = {event_filter: latest_at} if latest_at else {}
                        return {
                            "total_events": total,
                            "counts_by_event": counts_by_event,
                            "latest_by_event": latest_by_event,
                            "latest_event_at": latest_at,
                            "latest_event": latest_event,
                            "storage": "postgres",
                        }

                    cur.execute(
                        """
                        SELECT total_events, latest_event_at, latest_event::text AS latest_event_json
                        FROM backend_activity_summary
                        WHERE monitor_key = %s
                        """,
                        (_MONITOR_KEY,),
                    )
                    summary_row = cur.fetchone()
                    cur.execute(
                        """
                        SELECT event, count, latest_at
                        FROM backend_activity_event_stats
                        WHERE monitor_key = %s
                        ORDER BY event ASC
                        """,
                        (_MONITOR_KEY,),
                    )
                    stat_rows = cur.fetchall()

        latest_event = None
        if summary_row and summary_row.get("latest_event_json"):
            try:
                latest_event = json.loads(summary_row["latest_event_json"])
            except (TypeError, json.JSONDecodeError):
                latest_event = None
        counts_by_event = {
            str(row.get("event") or ""): int(row.get("count") or 0)
            for row in stat_rows
            if str(row.get("event") or "").strip()
        }
        latest_by_event = {
            str(row.get("event") or ""): row.get("latest_at")
            for row in stat_rows
            if str(row.get("event") or "").strip() and row.get("latest_at")
        }
        return {
            "total_events": int(summary_row["total_events"] or 0) if summary_row else 0,
            "counts_by_event": counts_by_event,
            "latest_by_event": latest_by_event,
            "latest_event_at": (summary_row.get("latest_event_at") if summary_row else None),
            "latest_event": latest_event,
            "storage": "postgres",
        }


_activity_store: Optional[ActivityStore] = None
_activity_store_lock = Lock()


def get_activity_store() -> ActivityStore:
    global _activity_store
    with _activity_store_lock:
        if _activity_store is None:
            _activity_store = ActivityStore()
        return _activity_store


def write_activity(event: str, **payload: Any) -> bool:
    record = {
        "ts": _utc_now_iso(),
        "event": _clip(event, 64),
    }
    record.update(_sanitize(payload))
    try:
        with _WRITE_LOCK:
            get_activity_store().write(record)
    except Exception:
        return False
    return True


def read_recent_activity(limit: int = 100, event: Optional[str] = None) -> list[dict[str, Any]]:
    try:
        return get_activity_store().recent_events(limit=limit, event=event)
    except Exception:
        return []


def read_activity_summary(event: Optional[str] = None) -> dict[str, Any]:
    try:
        return get_activity_store().summary(event=event)
    except Exception:
        return {
            "total_events": 0,
            "counts_by_event": {},
            "latest_by_event": {},
            "latest_event_at": None,
            "latest_event": None,
            "storage": "postgres",
        }


def summarize_activity(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    latest_by_event: dict[str, str] = {}
    for row in events:
        event = str(row.get("event") or "").strip()
        if not event:
            continue
        counts[event] = counts.get(event, 0) + 1
        latest_by_event.setdefault(event, str(row.get("ts") or ""))
    return {
        "total_events": len(events),
        "counts_by_event": dict(sorted(counts.items())),
        "latest_by_event": latest_by_event,
        "latest_event_at": (events[0].get("ts") if events else None),
    }
