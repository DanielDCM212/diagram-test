"""Persists the finished diagram into Postgres, mirroring the table flow
from scripts/persist_diagram.py (kept in this repo as a reference of an
existing, similar pipeline).

Only `postgres_connection.PostgresConnection` and the embedding call are
mocked (see postgres_connection.py's docstring, and `_create_embedding`
below) -- both already exist as real services in the target system.
Everything else here, in particular the actual SQL statements and the
version/history/approval-event flow, is genuine and runs unchanged once
`postgres_connection.PostgresConnection` is swapped for the real one.

Domain rows (iag_domains) are assumed to already exist -- unlike the
reference, we deliberately do not upsert them here.

`iag_diagrams` holds one row PER VERSION (confirmed against the real
schema -- `id` is not unique there), not a single current-state row per
diagram. So every persist is a plain append, never an update-in-place,
and none of this relies on a unique/exclusion constraint existing on
`id`. `current_version` on that row doubles as this row's own version
number.

One transaction, in this order:
    1. iag_diagrams               INSERT, one new row per version
    2. iag_diagram_versions       append-only per (diagram_id, version)
    3. iag_diagram_approval_events one row per persist (audit trail)
    4. iag_diagram_permissions    OWNER row, first create only
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import BaseModel, Field

from .postgres_connection import PostgresConnection, check_connection
from .tools import _DIAGRAM_JSON_KEY, _DRAWIO_XML_KEY  # noqa: F401 (diagram_json reserved for future use)

_STATE_DIAGRAM_INPUT_KEY = "diagram_input"
_STATE_PERSIST_RESULT_KEY = "persist_result"


def check_db_before_run(callback_context):
    """before_agent_callback: verify the persistence layer is reachable before recreating anything.

    Runs once, before the vision/tool-calling loop starts. Recreating a
    diagram is expensive (several LLM turns, tool calls); there is no point
    running all of that only to fail at the very last step (persist_diagram)
    because the database was never reachable. Returning non-None Content
    here skips the agent run entirely and sends that content back as the
    response instead (see BaseAgent.before_agent_callback in the installed
    google-adk package).
    """
    available, error = check_connection()
    if available:
        return None
    return types.Content(
        role="model",
        parts=[types.Part(text=(
            "Cannot start: the diagram persistence database is not reachable "
            f"({error}). No diagram recreation was attempted."
        ))],
    )


class DiagramInput(BaseModel):
    """Shape of the `diagram_input` object a caller sends in `state_delta`."""

    diagram_id: str
    title: str
    domain_id: str
    diagram_type: str = "architecture"
    format: str = "drawio"
    tags: list[str] = Field(default_factory=list)
    source_type: str = "manual"
    source_ref: Optional[str] = None
    source_url: Optional[str] = None
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
    soeid: str = Field(description="Actor identity stamped on created_by/updated_by/actor/granted_by.")
    description: str = Field(description="Natural-language diagram description, stored alongside the XML.")


class PersistResult(BaseModel):
    status: str
    diagram_id: Optional[str] = None
    version: Optional[int] = None
    domain_id: Optional[str] = None
    reason: Optional[str] = None
    detail: Optional[str] = None


def _create_embedding(text: str) -> Optional[list[float]]:
    """Mock for the target system's real embedding service call.

    Stands in for something like `CompatEmbeddingAndPdfService`, which isn't
    wired up in this repo -- always returns None (stored as NULL in
    iag_diagrams.embedding) until the real call is dropped in here.
    """
    return None


def _error(reason: str, detail: str) -> dict[str, Any]:
    return PersistResult(status="error", reason=reason, detail=detail).model_dump(exclude_none=True)


# No ORDER BY column other than current_version distinguishes rows for the
# same id (there's one row per version) -- take the highest version's status
# as "previous". FOR UPDATE only locks that one row; it doesn't serialize
# concurrent inserts of a brand-new diagram_id (nothing to lock yet).
_SQL_LOOKUP_PREVIOUS_STATUS = """
SELECT status FROM iag_diagrams
WHERE id = %s
ORDER BY current_version DESC
LIMIT 1
FOR UPDATE
"""

_SQL_NEXT_VERSION = """
SELECT COALESCE(MAX(version), 0) + 1
FROM iag_diagram_versions
WHERE diagram_id = %s
"""

# Plain INSERT, one new row per version -- no ON CONFLICT, so this doesn't
# depend on `id` (or any other column here) having a unique/exclusion
# constraint. `inserted` (first version vs. a later one) is decided by the
# previous-status lookup above, before this ever runs.
_SQL_DIAGRAM_INSERT = """
INSERT INTO iag_diagrams (
    id, title, description, format, diagram_type, xml,
    calm_json, geometry_json,
    domain_id, tags, status, current_version,
    created_by, updated_by, source_type, source_ref, source_url,
    custom_metadata, tsv, embedding
)
VALUES (
    %(id)s, %(title)s, %(description)s, %(format)s, %(diagram_type)s, %(xml)s,
    %(calm_json)s::jsonb, %(geometry_json)s::jsonb,
    %(domain_id)s, %(tags)s::jsonb, 'DRAFT', %(version)s,
    %(soeid)s, %(soeid)s, %(source_type)s, %(source_ref)s, %(source_url)s,
    %(custom_metadata)s::jsonb,
    setweight(to_tsvector('english', coalesce(%(title)s, '')), 'A') ||
    setweight(to_tsvector('english', coalesce(%(description)s, '')), 'B'),
    %(embedding)s
)
"""

_SQL_VERSION_INSERT = """
INSERT INTO iag_diagram_versions (
    diagram_id, version, xml, calm_json, geometry_json,
    description, change_summary, custom_metadata, created_by
)
VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s::jsonb, %s)
"""

_SQL_APPROVAL_EVENT = """
INSERT INTO iag_diagram_approval_events (
    diagram_id, version, from_status, to_status, actor, comment
)
VALUES (%s, %s, %s, 'DRAFT', %s, %s)
"""

_SQL_PERM_INSERT = """
INSERT INTO iag_diagram_permissions (
    id, user_id, role, diagram_id, granted_by
)
VALUES (%s, %s, 'OWNER', %s, %s)
ON CONFLICT DO NOTHING
"""


def persist_diagram(tool_context: ToolContext) -> dict[str, Any]:
    """Persist the finished diagram (from save_diagram/render_drawio) into Postgres.

    Call this last, only after validate_drawio reports the drawio XML is
    valid and sanity_plot's preview matches the source image. Reads
    everything it needs from session state -- `diagram_input` (the caller's
    metadata, sent via state_delta) and the `drawio_xml` this pipeline
    already produced -- so nothing here is something you supply as an
    argument.
    """
    try:
        diagram_input_raw = tool_context.state.get(_STATE_DIAGRAM_INPUT_KEY)
        if not diagram_input_raw:
            return _error(
                "missing_diagram_input",
                f"state[{_STATE_DIAGRAM_INPUT_KEY!r}] is empty — expected in the run's state_delta",
            )
        diagram_input = DiagramInput.model_validate(
            json.loads(diagram_input_raw) if isinstance(diagram_input_raw, str) else diagram_input_raw
        )

        xml = tool_context.state.get(_DRAWIO_XML_KEY)
        if not xml:
            return _error(
                "missing_xml",
                f"state[{_DRAWIO_XML_KEY!r}] is empty — call render_drawio first",
            )

        embedding = _create_embedding(f"{diagram_input.title}\n\n{diagram_input.description}")
        diagram_id = diagram_input.diagram_id
        custom_metadata_json = json.dumps(diagram_input.custom_metadata)

        params: dict[str, Any] = {
            "id": diagram_id,
            "title": diagram_input.title,
            "description": diagram_input.description,
            "format": diagram_input.format,
            "diagram_type": diagram_input.diagram_type,
            "xml": xml,
            "calm_json": None,
            "geometry_json": None,
            "domain_id": diagram_input.domain_id,
            "tags": json.dumps(diagram_input.tags),
            "version": 0,  # overwritten below, once next_version is known
            "soeid": diagram_input.soeid,
            "source_type": diagram_input.source_type,
            "source_ref": diagram_input.source_ref,
            "source_url": diagram_input.source_url,
            "custom_metadata": custom_metadata_json,
            "embedding": embedding,
        }

        # ---- One transaction for everything -----------------------------------
        pool = PostgresConnection.get_pool()
        with pool.connection() as conn, conn.transaction():
            with conn.cursor() as cur:
                # 1. Determine previous status (for the approval-event audit trail)
                #    and lock the row against concurrent re-uploads of the same
                #    diagram while we compute the next version.
                cur.execute(_SQL_LOOKUP_PREVIOUS_STATUS, (diagram_id,))
                prev_row = cur.fetchone()
                previous_status: Optional[str] = prev_row[0] if prev_row else None

                cur.execute(_SQL_NEXT_VERSION, (diagram_id,))
                next_version = cur.fetchone()[0]
                params["version"] = next_version

                # 2. Diagram row for this version -- always a fresh insert.
                #    `inserted` (first version vs. a later one) was already
                #    decided by whether step 1 found a prior row.
                inserted = previous_status is None
                cur.execute(_SQL_DIAGRAM_INSERT, params)

                # 3. Version history (append-only).
                cur.execute(
                    _SQL_VERSION_INSERT,
                    (
                        diagram_id,
                        next_version,
                        xml,
                        params["calm_json"],
                        params["geometry_json"],
                        diagram_input.description,
                        "initial" if inserted else "update",
                        custom_metadata_json,
                        diagram_input.soeid,
                    ),
                )

                # 4. Approval event -- always one per persist, for audit.
                cur.execute(
                    _SQL_APPROVAL_EVENT,
                    (
                        diagram_id,
                        next_version,
                        previous_status,
                        diagram_input.soeid,
                        "auto-created" if inserted else "new version",
                    ),
                )

                # 5. OWNER permission seed on first create only.
                if inserted:
                    cur.execute(
                        _SQL_PERM_INSERT,
                        (
                            f"diagram_perm_{uuid.uuid4().hex}",
                            diagram_input.soeid,
                            diagram_id,
                            diagram_input.soeid,
                        ),
                    )

        result = PersistResult(
            status="ok",
            diagram_id=diagram_id,
            version=next_version,
            domain_id=diagram_input.domain_id,
        )
        tool_context.state[_STATE_PERSIST_RESULT_KEY] = result.model_dump(exclude_none=True)
        return result.model_dump(exclude_none=True)

    except Exception as exc:
        return _error("persist_failed", str(exc))
