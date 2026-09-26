"""Mock stand-in for the target system's own `postgres_connection` module.

A real Postgres connection pool (imported in scripts/persist_diagram.py as
`from postgres_connection import PostgresConnection`) already exists in that
target system and is not reimplemented here. This module exists only so
persist_diagram.py's real SQL statements can be executed end to end -- via
the same `pool.connection() -> conn.transaction() -> conn.cursor() ->
cur.execute(sql, params)` shape a real psycopg pool exposes -- without
opening an actual database socket. `_MockCursor.execute` dispatches on the
SQL text itself, so persist_diagram.py's own SQL never has to change when
this module is swapped for the real one at deployment.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional


def _dispatch(tables: dict[str, Any], sql: str, params: Any) -> Optional[tuple]:
    text = " ".join(sql.split())

    if text == "SELECT 1":
        # Trivial connectivity probe used by check_connection() below.
        return (1,)

    if "SELECT status FROM iag_diagrams" in text:
        (diagram_id,) = params
        rows = [r for r in tables["iag_diagrams"] if r["id"] == diagram_id]
        if not rows:
            return None
        latest = max(rows, key=lambda r: r["version"])
        return (latest["status"],)

    if "FROM iag_diagram_versions" in text and "MAX(version)" in text:
        (diagram_id,) = params
        existing = [v["version"] for v in tables["iag_diagram_versions"] if v["diagram_id"] == diagram_id]
        return (max(existing, default=0) + 1,)

    if "INSERT INTO iag_diagrams" in text:
        # One row per version -- current_version doubles as this row's own
        # version number (see persistence.py's module docstring).
        tables["iag_diagrams"].append({**params, "status": "DRAFT", "version": params["version"]})
        return None

    if "INSERT INTO iag_diagram_versions" in text:
        diagram_id, version, xml, calm_json, geometry_json, description, change_summary, custom_metadata, created_by = params
        tables["iag_diagram_versions"].append({
            "diagram_id": diagram_id,
            "version": version,
            "xml": xml,
            "calm_json": calm_json,
            "geometry_json": geometry_json,
            "description": description,
            "change_summary": change_summary,
            "custom_metadata": custom_metadata,
            "created_by": created_by,
        })
        return None

    if "INSERT INTO iag_diagram_approval_events" in text:
        diagram_id, version, from_status, actor, comment = params
        tables["iag_diagram_approval_events"].append({
            "diagram_id": diagram_id,
            "version": version,
            "from_status": from_status,
            "to_status": "DRAFT",
            "actor": actor,
            "comment": comment,
        })
        return None

    if "INSERT INTO iag_diagram_permissions" in text:
        perm_id, user_id, diagram_id, granted_by = params
        tables["iag_diagram_permissions"].append({
            "id": perm_id,
            "user_id": user_id,
            "role": "OWNER",
            "diagram_id": diagram_id,
            "granted_by": granted_by,
        })
        return None

    raise NotImplementedError(f"mock postgres_connection: no handler for SQL: {text[:80]}...")


class _MockCursor:
    def __init__(self, tables: dict[str, Any]) -> None:
        self._tables = tables
        self._last_result: Optional[tuple] = None

    def execute(self, sql: str, params: Any = None) -> None:
        self._last_result = _dispatch(self._tables, sql, params)

    def fetchone(self) -> Optional[tuple]:
        return self._last_result

    def __enter__(self) -> "_MockCursor":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class _MockConnection:
    def __init__(self, tables: dict[str, Any]) -> None:
        self._tables = tables

    @contextmanager
    def transaction(self):
        # No real commit/rollback semantics needed for the mock -- a raised
        # exception still propagates out of persist_diagram's `with` block.
        yield

    def cursor(self) -> _MockCursor:
        return _MockCursor(self._tables)

    def __enter__(self) -> "_MockConnection":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class _MockPool:
    def __init__(self) -> None:
        self.tables: dict[str, Any] = {
            "iag_diagrams": [],
            "iag_diagram_versions": [],
            "iag_diagram_approval_events": [],
            "iag_diagram_permissions": [],
        }

    @contextmanager
    def connection(self):
        yield _MockConnection(self.tables)


_pool: Optional[_MockPool] = None


class PostgresConnection:
    """Same interface as the target system's real `postgres_connection.PostgresConnection`."""

    @classmethod
    def get_pool(cls) -> _MockPool:
        global _pool
        if _pool is None:
            _pool = _MockPool()
        return _pool


def check_connection() -> tuple[bool, Optional[str]]:
    """Probe whether the persistence layer is reachable.

    Runs the same pool -> connection -> cursor -> execute round trip
    persist_diagram uses, with a trivial `SELECT 1`. Against this mock that
    round trip can't actually fail, but once `PostgresConnection` is swapped
    for the real client this becomes a genuine connectivity check with no
    other code changes needed.

    Returns:
        (True, None) if reachable, otherwise (False, <error message>).
    """
    try:
        pool = PostgresConnection.get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True, None
    except Exception as exc:
        return False, str(exc)
