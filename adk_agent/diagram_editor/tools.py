"""ADK tools + callback for the diagram_editor agent.

The working diagram lives in session state as draw.io XML under `drawio_xml`
(the same key diagram_recreator uses, so a parent agent can read either
sub-agent's result the same way). The LLM never sees or writes raw XML:
it reads `describe_diagram` and acts through `create_diagram` / `edit_diagram`.
"""
from __future__ import annotations

import hashlib
import io
import re
from typing import Any, Optional

from google.adk.tools.tool_context import ToolContext
from pydantic import TypeAdapter, ValidationError

from . import drawio_doc as dd
from . import export_client
from . import overlaps as ov
from . import references as refs
from .edit_ops import apply_ops
from .layout import LayoutError, build_diagram
from .schema import ArchitectureSpec, EditOp

DRAWIO_XML_KEY = "drawio_xml"  # working + final diagram
_INPUT_HASH_KEY = "_input_xml_sha"

_XML_SPAN_RE = re.compile(r"<mxfile\b.*?</mxfile>|<mxGraphModel\b.*?</mxGraphModel>", re.DOTALL)
_XML_MIMES = ("text/xml", "application/xml", "text/plain", "application/octet-stream",
              "application/vnd.jgraph.mxfile", "application/x-drawio")
_ops_adapter = TypeAdapter(list[EditOp])


# --------------------------------------------------------------------------
# Input capture
# --------------------------------------------------------------------------

def _placeholder(doc: dd.Doc) -> str:
    return (f"[draw.io XML attached by the user: {len(doc.cells())} cells, {len(doc.pages)} page(s). "
            "It is loaded in the session -- call describe_diagram to inspect it; never ask the user to paste it again.]")


def _store_input(callback_context, raw: str) -> Optional[dd.Doc]:
    try:
        doc = dd.load(raw)
    except dd.DrawioParseError as exc:
        callback_context.state["input_xml_error"] = str(exc)
        return None
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if callback_context.state.get(_INPUT_HASH_KEY) != sha:
        callback_context.state[_INPUT_HASH_KEY] = sha
        callback_context.state[DRAWIO_XML_KEY] = dd.serialize(doc)
        callback_context.state["input_xml_error"] = ""  # ADK State has no pop/del; "" means no error
    return doc


def capture_input_xml(callback_context, llm_request):
    """before_model_callback: lift draw.io XML out of the user's messages into session state.

    Runs before every model call over the whole history, so it is idempotent (keyed on a hash
    of the raw XML). The XML is replaced in the outgoing request by a short placeholder: the
    model works through describe_diagram/edit_diagram instead of carrying a huge blob in every
    turn, and never has to re-emit it as a tool argument. A newly pasted XML later in the
    conversation replaces the working diagram.
    """
    for content in llm_request.contents or []:
        if content.role != "user":
            continue
        for part in content.parts or []:
            text = getattr(part, "text", None)
            if text and ("<mxfile" in text or "<mxGraphModel" in text):
                def swap(m):
                    doc = _store_input(callback_context, m.group(0))
                    return _placeholder(doc) if doc else "[draw.io XML that could not be parsed -- see input_xml_error]"
                new = _XML_SPAN_RE.sub(swap, text)
                part.text = new
                continue
            inline = getattr(part, "inline_data", None)
            if inline and inline.data and (inline.mime_type or "") in _XML_MIMES:
                try:
                    raw = inline.data.decode("utf-8-sig")
                except UnicodeDecodeError:
                    continue
                if "<mxfile" in raw or "<mxGraphModel" in raw:
                    doc = _store_input(callback_context, raw)
                    part.inline_data = None
                    part.text = _placeholder(doc) if doc else "[attached draw.io file could not be parsed -- see input_xml_error]"
    return None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _working_doc(tool_context: ToolContext) -> tuple[Optional[dd.Doc], Optional[dict]]:
    xml = tool_context.state.get(DRAWIO_XML_KEY)
    if not xml:
        err = tool_context.state.get("input_xml_error")
        return None, {"error": ("the attached XML could not be parsed: " + err) if err
                      else "no diagram yet -- the user gave no XML; use create_diagram for a new one"}
    return dd.load(xml), None


def _save(tool_context: ToolContext, doc: dd.Doc) -> None:
    tool_context.state[DRAWIO_XML_KEY] = dd.serialize(doc)


def _coerce(model_or_raw: Any, adapter: TypeAdapter):
    """ADK sometimes hands nested tool args through as dicts or even a JSON string."""
    if isinstance(model_or_raw, (str, bytes)):
        return adapter.validate_json(model_or_raw)
    return adapter.validate_python(model_or_raw)


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------

def list_references(tool_context: ToolContext) -> dict:
    """List the house-style reference diagrams (the company's standard look and layout).

    Call this first in create mode, and when adding components in edit mode. Pick the
    reference whose structure is closest to what the user wants, then call get_reference.
    """
    items = refs.list_references()
    if not items:
        return {"references": [], "warning": "no reference diagrams are installed; styles can't be copied"}
    return {"references": items}


def get_reference(tool_context: ToolContext, name: str) -> dict:
    """Show one reference diagram and its derived style/layout conventions.

    Returns the diagram's cells (ids, labels, positions) so you can choose which cells to
    imitate via `like`, plus a profile: node kinds (style clusters with example cell ids),
    zone kinds, edge kinds, and layout metrics (zone arrangement, padding, gaps, grid).

    Args:
        name: File name from list_references.
    """
    try:
        doc = refs.load_reference(name)
    except refs.ReferenceLookupError as exc:
        return {"error": str(exc)}
    return {"name": name, "diagram": dd.summarize(doc), "profile": refs.derive_profile(doc)}


def create_diagram(tool_context: ToolContext, spec: ArchitectureSpec) -> dict:
    """Create a NEW diagram from structure, laid out in the house style. Replaces the working diagram.

    You give zones, components and connections and say which reference cells to imitate
    (`like`); positions are computed from the layout reference's conventions -- do not try
    to specify coordinates. Every problem in the spec is reported at once, so fix them all
    and call again.

    Args:
        spec: The architecture: title, layout_from (reference file), zones, components, connections.
    """
    try:
        spec = _coerce(spec, TypeAdapter(ArchitectureSpec))
    except ValidationError as exc:
        return {"created": False, "errors": [f"invalid spec: {exc.errors()[:5]}"]}
    try:
        doc, stats = build_diagram(spec)
    except LayoutError as exc:
        return {"created": False, "errors": exc.problems}
    _save(tool_context, doc)
    return {"created": True, **stats}


def describe_diagram(tool_context: ToolContext, page: int = 0) -> dict:
    """Describe the working diagram: every node (id, label, absolute x/y/w/h, container, style digest) and edge.

    Use the ids returned here in edit_diagram ops. Coordinates are absolute canvas
    coordinates even for nodes nested inside containers.

    Args:
        page: Page index for multi-page files (default 0).
    """
    doc, err = _working_doc(tool_context)
    if err:
        return err
    page = int(float(page))
    if not 0 <= page < len(doc.pages):
        return {"error": f"page {page} out of range (0..{len(doc.pages) - 1})"}
    return dd.summarize(doc, page)


def get_cells_xml(tool_context: ToolContext, ids: list[str], include_children: bool = False, page: int = 0) -> dict:
    """Return the raw XML of specific cells, to see exactly what is stored (full style strings, extra attributes).

    Use before raw XML edits (upsert_xml / set_attrs / set_geometry) or when describe_diagram's
    style digest isn't enough, e.g. to see grIcon, opacity or a custom property. Only the cells
    you ask for are returned, never the whole file. Geometry is raw: relative to each cell's parent.

    Args:
        ids: Cell ids (from describe_diagram).
        include_children: Also return every cell nested inside the given ones (a whole zone).
        page: Page index (default 0).
    """
    from xml.etree import ElementTree as ET

    doc, err = _working_doc(tool_context)
    if err:
        return err
    page = int(float(page))
    if not 0 <= page < len(doc.pages):
        return {"error": f"page {page} out of range"}
    if isinstance(ids, str):
        ids = [ids]
    by_id = {c.get("id"): c for c in doc.cells(page)}
    wanted = [i for i in ids if i in by_id]
    if include_children:
        frontier = list(wanted)
        while frontier:
            cur = frontier.pop()
            for c in doc.cells(page):
                if dd.attr(c, "parent") == cur and c.get("id") not in wanted:
                    wanted.append(c.get("id"))
                    frontier.append(c.get("id"))
    out, total, truncated = [], 0, False
    for cid in wanted:
        xml = ET.tostring(by_id[cid], encoding="unicode")
        if total + len(xml) > 30_000:
            truncated = True
            break
        total += len(xml)
        out.append({"id": cid, "xml": xml})
    return {"cells": out, "unknown_ids": [i for i in ids if i not in by_id], "truncated": truncated}


def edit_diagram(tool_context: ToolContext, ops: list[EditOp]) -> dict:
    """Apply edit operations to the working diagram, all-or-nothing.

    Ops run in order on a copy; if any fails nothing is changed and the failing op is
    reported. The result is validated before it is saved. Only change what the user asked.

    Prefer the high-level ops when they fit: update_label, move, resize, set_style, delete,
    add_component (auto-placed; set `parent`/`near`, and `like` to copy an existing style/size
    -- a similar cell from describe_diagram or a RefCell from get_reference), add_connection.

    For anything else use the XML-level ops, which can express any change: upsert_xml (add or
    replace one <mxCell>), set_attrs (any attribute: style, value, parent, source, target),
    set_geometry (raw, parent-relative), reorder (draw order: backgrounds go 'back'). Read the
    current cells with get_cells_xml first, and copy style keys from existing cells or
    references rather than inventing them. High-level and XML-level ops can be mixed in one call.

    Args:
        ops: The operations, in order.
    """
    doc, err = _working_doc(tool_context)
    if err:
        return err
    try:
        ops = _coerce(ops, _ops_adapter)
    except ValidationError as exc:
        return {"applied": False, "errors": [f"invalid ops: {exc.errors()[:5]}"]}
    new, results = apply_ops(doc, ops, refs=refs.resolve_ref_cell)
    if new is None:
        return {"applied": False, "results": results}
    check = dd.validate(new)
    if not check["valid"]:
        return {"applied": False, "results": results, "validation": check}
    _save(tool_context, new)
    return {"applied": True, "results": results, "cell_count": check["cell_count"]}


def find_overlaps(tool_context: ToolContext, page: int = 0) -> dict:
    """Report layout problems in the working diagram: overlapping nodes, labels that overflow their box,
    and children sticking out of their container.

    Nesting is not an overlap: a node inside a container/zone, or a label on a box, is fine. Each
    overlap has a severity (1.0 = the smaller box is completely covered). Use before and after
    resolve_overlaps, and fix the worst first.

    Args:
        page: Page index (default 0).
    """
    doc, err = _working_doc(tool_context)
    if err:
        return err
    page = int(float(page))
    if not 0 <= page < len(doc.pages):
        return {"error": f"page {page} out of range"}
    return ov.find_overlaps(doc, page)


def resolve_overlaps(
    tool_context: ToolContext,
    lock: Optional[list[str]] = None,
    only: Optional[list[str]] = None,
    gap: float = 10,
    max_shift: float = 80,
    allow_move: bool = True,
    allow_resize: bool = True,
    min_scale: float = 0.75,
    allow_font_shrink: bool = True,
    min_font: int = 8,
    fix_overflow: bool = False,
    dry_run: bool = False,
    page: int = 0,
) -> dict:
    """Automatically remove overlaps, changing as little as possible. Saved only if the result validates.

    Escalates in order, each step bounded: (1) MOVE overlapping nodes apart (each at most `max_shift`
    px in total; a moved node carries everything inside it, containers grow to fit their contents),
    (2) SHRINK leaf boxes (not below `min_scale` of their original size), (3) shrink FONTS of resized
    boxes so labels still fit (not below `min_font`; fonts are never enlarged). Whatever can't be solved
    within these limits is listed under `remaining`, never forced -- then loosen a limit, move things
    yourself with edit_diagram, or tell the user. Run with dry_run=true first when unsure: it returns
    the same report without saving.

    Args:
        lock: Node ids that must not move or resize (title, anchor nodes, anything the user said to keep).
        only: If given, ONLY these nodes (and what is inside them) may change; everything else is fixed.
            Use to limit the clean-up to one area.
        gap: Clearance in px left between nodes that were pushed apart.
        max_shift: Most a node may move in total, px.
        allow_move: Allow step 1.
        allow_resize: Allow step 2.
        min_scale: Smallest size factor for shrunk boxes, 0..1.
        allow_font_shrink: Allow step 3.
        min_font: Smallest font size step 3 may use.
        fix_overflow: Also shrink fonts of labels that already overflow before any resize.
        dry_run: Report only; do not save.
        page: Page index (default 0).
    """
    doc, err = _working_doc(tool_context)
    if err:
        return err
    page = int(float(page))
    if not 0 <= page < len(doc.pages):
        return {"error": f"page {page} out of range"}
    if isinstance(lock, str):
        lock = [lock]
    if isinstance(only, str):
        only = [only]
    try:
        params = ov.Params(
            gap=float(gap), max_shift=float(max_shift), allow_move=bool(allow_move),
            allow_resize=bool(allow_resize), min_scale=min(1.0, max(0.1, float(min_scale))),
            allow_font_shrink=bool(allow_font_shrink), min_font=int(float(min_font)),
            fix_overflow=bool(fix_overflow), lock=frozenset(lock or ()),
            only=frozenset(only) if only else None,
        )
        new, report = ov.resolve_overlaps(doc, page, params)
    except ValueError as exc:
        return {"saved": False, "error": str(exc)}
    check = dd.validate(new)
    if not check["valid"]:
        return {"saved": False, "validation": check, **report}
    if dry_run:
        return {"saved": False, "dry_run": True, **report}
    _save(tool_context, new)
    return {"saved": True, **report}


def validate_drawio(tool_context: ToolContext) -> dict:
    """Check the working diagram: parses, no duplicate ids, no dangling source/target/parent references."""
    xml = tool_context.state.get(DRAWIO_XML_KEY)
    if not xml:
        return {"error": "no working diagram"}
    try:
        return dd.validate(dd.load(xml))
    except dd.DrawioParseError as exc:
        return {"valid": False, "parse_error": str(exc)}


def _render_python(doc: dd.Doc, page: int) -> bytes:
    """Schematic fallback: boxes, labels and straight edges via matplotlib (no draw.io fidelity)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    cells = doc.cells(page)
    boxes = {}
    for c in cells:
        b = dd.absolute_box(doc, c, page)
        if b:
            boxes[c.get("id")] = (c, b)
    min_x = min(0.0, min(b[0] for _, b in boxes.values()))  # nodes can sit at negative coordinates
    min_y = min(0.0, min(b[1] for _, b in boxes.values()))
    max_x = max(b[0] + b[2] for _, b in boxes.values())
    max_y = max(b[1] + b[3] for _, b in boxes.values())
    # keep the PNG under the model's per-image edge limit
    fig_w, fig_h = (max_x - min_x) / 100, (max_y - min_y) / 100
    dpi = max(40, min(150, int(2400 / max(fig_w, fig_h, 1))))

    def color(v):
        return v if v and v.startswith("#") else None

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for c, (x, y, w, h) in sorted(boxes.values(), key=lambda t: t[1][2] * t[1][3], reverse=True):
        st = dd.parse_style(dd.attr(c, "style"))
        if "text" in st and st.get("fillColor") in (None, "none"):
            pass
        elif "ellipse" in st or st.get("shape") == "ellipse":
            ax.add_patch(patches.Ellipse((x + w / 2, y + h / 2), w, h, facecolor=color(st.get("fillColor")) or "none",
                                         edgecolor=color(st.get("strokeColor")) or "#333333", lw=0.8))
        else:
            ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=color(st.get("fillColor")) or "none",
                                           edgecolor=color(st.get("strokeColor")) or "#333333", lw=0.8))
        label = dd.plain_label(dd.attr(c, "value"))
        if label:
            top = st.get("verticalAlign") == "top"
            ax.text(x + w / 2, y + (4 if top else h / 2), label, ha="center", va="top" if top else "center",
                    fontsize=6, color=color(st.get("fontColor")) or "#000000", clip_on=True)
    for c in cells:
        if dd.attr(c, "edge") != "1" or dd.attr(c, "source") not in boxes or dd.attr(c, "target") not in boxes:
            continue
        sx, sy, sw, sh = boxes[dd.attr(c, "source")][1]
        tx, ty, tw, th = boxes[dd.attr(c, "target")][1]
        dashed = dd.parse_style(dd.attr(c, "style")).get("dashed") == "1"
        ax.annotate("", xy=(tx + tw / 2, ty + th / 2), xytext=(sx + sw / 2, sy + sh / 2),
                    arrowprops=dict(arrowstyle="->", linestyle="--" if dashed else "-", color="#444444", lw=0.7))
        label = dd.plain_label(dd.attr(c, "value"))
        if label:
            ax.text((sx + sw / 2 + tx + tw / 2) / 2, (sy + sh / 2 + ty + th / 2) / 2, label, fontsize=5, color="#444444")
    ax.set_xlim(min_x - 10, max_x + 10)
    ax.set_ylim(max_y + 10, min_y - 10)
    ax.axis("off")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def preview_diagram(tool_context: ToolContext, page: int = 0) -> dict:
    """Render a PNG of the working diagram, returned to you to LOOK at.

    Uses the draw.io export server when available (real draw.io rendering: actual shapes and
    edge routing). If that server is unconfigured, down or fails, a schematic Python render is
    used instead (boxes, labels, straight edge lines) and `renderer` says so. Check for
    overlaps, things outside their container, and that the picture matches the request, then
    fix with edit_diagram.

    Args:
        page: Page index (default 0).
    """
    from google.genai import types

    doc, err = _working_doc(tool_context)
    if err:
        return err
    page = int(float(page))
    if not 0 <= page < len(doc.pages):
        return {"error": f"page {page} out of range"}
    cells = doc.cells(page)
    nodes = sum(1 for c in cells if dd.local_box(c))
    if not nodes:
        return {"error": "diagram has no nodes to draw"}

    result: dict = {"node_count": nodes, "edge_count": sum(1 for c in cells if dd.attr(c, "edge") == "1")}
    png = None
    if export_client.is_configured():
        try:
            png = export_client.export_png(tool_context.state[DRAWIO_XML_KEY], page)
            result["renderer"] = "export_server"
        except export_client.ExportError as exc:
            result["fallback_reason"] = str(exc)
    if png is None:
        png = _render_python(doc, page)
        result["renderer"] = "python_fallback"
        result.setdefault("fallback_reason", "DRAWIO_EXPORT_URL is not set")
        result["note"] = "schematic render: straight edge lines and no stencil shapes; real draw.io output will look different"
    result["image"] = types.Part.from_bytes(data=png, mime_type="image/png")
    return result
