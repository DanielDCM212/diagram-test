"""Apply structured edit ops to a draw.io document, atomically.

Pure logic on `drawio_doc.Doc` -- no ADK. `apply_ops` works on a deep copy and
only returns it if every op succeeded, so a half-applied edit can never reach
session state. Untouched cells are not rewritten, so their attributes (and any
style keys this code does not know about) survive byte-for-byte.
"""
from __future__ import annotations

import copy
import re
from typing import Callable, Optional
from xml.etree import ElementTree as ET

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

from . import drawio_doc as dd
from .schema import (
    AddComponent, AddConnection, Delete, Move, RefCell, Reorder, Resize, ScaleFonts, SetAttrs, SetGeometry,
    SetStyle, UpdateLabel, UpsertXml,
)

# (style, w, h) for a RefCell, or None if it can't be found.
RefResolver = Callable[[RefCell], Optional[tuple[str, float, float]]]

DEFAULT_W, DEFAULT_H = 120.0, 60.0
DEFAULT_STYLE = "rounded=0;whiteSpace=wrap;html=1;"
DEFAULT_EDGE_STYLE = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;"
GAP = 20.0
SNAP = 10.0


class OpError(Exception):
    pass


def clone(doc: dd.Doc) -> dd.Doc:
    mxfile = copy.deepcopy(doc.mxfile)
    return dd.Doc(mxfile, mxfile.findall("diagram"))


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:24] or "node"


def _unique_id(doc: dd.Doc, base: str) -> str:
    taken = {c.get("id") for c in doc.cells()}
    if base not in taken:
        return base
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


def _require(doc: dd.Doc, cell_id: str, what: str = "cell") -> ET.Element:
    c = doc.cell(cell_id)
    if c is None or cell_id in ("0", "1"):
        raise OpError(f"unknown {what} id {cell_id!r}")
    return c


def _set_geom(cell: ET.Element, x=None, y=None, w=None, h=None) -> None:
    node = dd.inner(cell)  # a wrapped cell keeps its geometry on the inner mxCell
    if node is None:
        raise OpError(f"cell {cell.get('id')!r} has no inner <mxCell>")
    g = node.find("mxGeometry")
    if g is None:
        g = ET.SubElement(node, "mxGeometry")
        g.set("as", "geometry")
    for k, v in (("x", x), ("y", y), ("width", w), ("height", h)):
        if v is not None:
            g.set(k, _fmt(v))


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


def _descendants(doc: dd.Doc, cell_id: str) -> set[str]:
    out, frontier = {cell_id}, [cell_id]
    cells = doc.cells()
    while frontier:
        cur = frontier.pop()
        for c in cells:
            if dd.attr(c, "parent") == cur and c.get("id") not in out:
                out.add(c.get("id"))
                frontier.append(c.get("id"))
    return out


def _parent_abs_origin(doc: dd.Doc, parent_id: str) -> tuple[float, float]:
    if parent_id in (None, "1", "0"):
        return 0.0, 0.0
    p = doc.cell(parent_id)
    box = dd.absolute_box(doc, p) if p is not None else None
    return (box[0], box[1]) if box else (0.0, 0.0)


def _resolve_like(doc: dd.Doc, like, refs: Optional[RefResolver]) -> Optional[tuple[str, Optional[float], Optional[float]]]:
    if like is None:
        return None
    if isinstance(like, RefCell):
        res = refs(like) if refs else None
        if res is None:
            raise OpError(f"reference cell {like.reference}:{like.cell} not found")
        return res
    c = doc.cell(like)
    if c is None:
        raise OpError(f"'like' cell {like!r} not found in the diagram")
    box = dd.local_box(c)
    return dd.attr(c, "style") or "", (box[2] if box else None), (box[3] if box else None)


# -- free-slot placement ------------------------------------------------------

def _siblings_boxes(doc: dd.Doc, parent_id: str) -> list[tuple[float, float, float, float]]:
    """Absolute boxes of every vertex that shares the same parent."""
    out = []
    for c in doc.cells():
        if dd.attr(c, "vertex") == "1" and dd.attr(c, "parent") == parent_id:
            b = dd.absolute_box(doc, c)
            if b:
                out.append(b)
    return out


def _collides(box, others, pad=GAP / 2) -> bool:
    x, y, w, h = box
    for ox, oy, ow, oh in others:
        if x < ox + ow + pad and ox < x + w + pad and y < oy + oh + pad and oy < y + h + pad:
            return True
    return False


def _snap(v: float) -> float:
    return round(v / SNAP) * SNAP


def find_free_slot(doc: dd.Doc, parent_id: str, w: float, h: float, near: Optional[str] = None) -> tuple[float, float]:
    """Absolute (x, y) for a new w x h box: right of `near`, below it, then a row-major scan.

    Inside a container the scan is bounded by the container; if nothing fits, the slot
    is below the lowest sibling (the caller grows the container).
    """
    others = _siblings_boxes(doc, parent_id)
    cands: list[tuple[float, float]] = []
    if near:
        nb = dd.absolute_box(doc, doc.cell(near)) if doc.cell(near) is not None else None
        if nb:
            cands += [(nb[0] + nb[2] + GAP, nb[1]), (nb[0], nb[1] + nb[3] + GAP)]
    pbox = None
    if parent_id not in ("1", "0"):
        pc = doc.cell(parent_id)
        pbox = dd.absolute_box(doc, pc) if pc is not None else None

    if pbox:
        pad_top = 40.0 if not others else GAP  # leave room for a container title
        x0, y0 = pbox[0] + GAP, pbox[1] + pad_top
        x1, y1 = pbox[0] + pbox[2] - GAP, pbox[1] + pbox[3] - GAP
    else:
        allb = [dd.absolute_box(doc, c) for c in doc.cells() if dd.attr(c, "vertex") == "1"]
        allb = [b for b in allb if b]
        x0, y0 = (min(b[0] for b in allb), min(b[1] for b in allb)) if allb else (40.0, 40.0)
        x1 = max((b[0] + b[2] for b in allb), default=x0 + 800)
        y1 = max((b[1] + b[3] for b in allb), default=y0 + 600)

    step_x, step_y = w + GAP, h + GAP
    y = y0
    while y + h <= y1:
        x = x0
        while x + w <= x1:
            cands.append((x, y))
            x += step_x
        y += step_y

    for cx, cy in cands:
        cx, cy = _snap(cx), _snap(cy)
        inside = True if not pbox else (
            cx >= pbox[0] and cy >= pbox[1] and cx + w <= pbox[0] + pbox[2] and cy + h <= pbox[1] + pbox[3])
        if inside and not _collides((cx, cy, w, h), others):
            return cx, cy

    # Nothing free: below everything in the same parent.
    if others:
        return _snap(min(b[0] for b in others)), _snap(max(b[1] + b[3] for b in others) + GAP)
    return _snap(x0), _snap(y0)


def _grow_to_contain(doc: dd.Doc, parent_id: str) -> None:
    """Enlarge container ancestors so their children fit (never shrinks)."""
    cur = parent_id
    while cur not in (None, "0", "1"):
        pc = doc.cell(cur)
        if pc is None:
            return
        pbox = dd.absolute_box(doc, pc)
        kids = [dd.absolute_box(doc, c) for c in doc.cells() if dd.attr(c, "parent") == cur and dd.attr(c, "vertex") == "1"]
        kids = [k for k in kids if k]
        if pbox and kids:
            need_r = max(k[0] + k[2] for k in kids) + GAP
            need_b = max(k[1] + k[3] for k in kids) + GAP
            _set_geom(pc, w=max(pbox[2], need_r - pbox[0]), h=max(pbox[3], need_b - pbox[1]))
        cur = dd.attr(pc, "parent")


# -- individual ops -----------------------------------------------------------

def _op_update_label(doc, op: UpdateLabel, refs):
    dd.set_attr(_require(doc, op.id), "value", op.label)


def _op_move(doc, op: Move, refs):
    c = _require(doc, op.id)
    if dd.attr(c, "vertex") != "1":
        raise OpError(f"{op.id!r} is not a node; edges can't be moved")
    box = dd.absolute_box(doc, c)
    w, h = box[2], box[3]
    new_parent = op.parent or dd.attr(c, "parent")
    if op.parent:
        if op.parent in _descendants(doc, op.id):
            raise OpError(f"can't move {op.id!r} into itself or its own child {op.parent!r}")
        if op.parent != "1":
            _require(doc, op.parent, "container")
    if op.x is None and op.y is None and op.parent:
        ax, ay = find_free_slot(doc, new_parent, w, h)
    else:
        ax = op.x if op.x is not None else box[0]
        ay = op.y if op.y is not None else box[1]
    ox, oy = _parent_abs_origin(doc, new_parent)
    dd.set_attr(c, "parent", new_parent)
    _set_geom(c, x=ax - ox, y=ay - oy)
    _grow_to_contain(doc, new_parent)


def _op_resize(doc, op: Resize, refs):
    c = _require(doc, op.id)
    if dd.attr(c, "vertex") != "1":
        raise OpError(f"{op.id!r} is not a node")
    _set_geom(c, w=op.w, h=op.h)
    _grow_to_contain(doc, dd.attr(c, "parent"))


def _op_set_style(doc, op: SetStyle, refs):
    c = _require(doc, op.id)
    if op.like is not None:
        style = dd.parse_style(_resolve_like(doc, op.like, refs)[0])
    else:
        style = dd.parse_style(dd.attr(c, "style"))
    for k in op.remove:
        style.pop(k, None)
    style.update(op.set)
    dd.set_attr(c, "style", dd.format_style(style))


def _op_delete(doc, op: Delete, refs):
    _require(doc, op.id)
    gone = _descendants(doc, op.id)
    root = doc.root()
    for c in [e for e in root if e.tag in dd.CELL_TAGS]:
        if c.get("id") in gone or dd.attr(c, "source") in gone or dd.attr(c, "target") in gone:
            root.remove(c)
    # An edge that only touched a deleted edge's endpoint is removed above; edges whose
    # *edge-label parent* was removed are handled the same way via `gone`.


def _op_add_component(doc, op: AddComponent, refs):
    parent = op.parent or "1"
    if parent != "1":
        _require(doc, parent, "container")
    style, w, h = DEFAULT_STYLE, DEFAULT_W, DEFAULT_H
    res = _resolve_like(doc, op.like, refs)
    if res:
        style = res[0] or style
        w, h = res[1] or w, res[2] or h
    w, h = op.w or w, op.h or h
    new_id = op.id or _unique_id(doc, _slug(op.label))
    if doc.cell(new_id) is not None:
        raise OpError(f"id {new_id!r} already exists")

    if op.fit_to:
        boxes = []
        for cid in op.fit_to:
            c = _require(doc, cid, "fit_to cell")
            b = dd.absolute_box(doc, c)
            if b is None:
                raise OpError(f"fit_to cell {cid!r} is not a node")
            boxes.append(b)
        pad = op.padding
        ax = min(b[0] for b in boxes) - pad
        ay = min(b[1] for b in boxes) - pad
        w = max(b[0] + b[2] for b in boxes) + pad - ax
        h = max(b[1] + b[3] for b in boxes) + pad - ay
    elif op.x is not None and op.y is not None:
        ax, ay = op.x, op.y
    else:
        ax, ay = find_free_slot(doc, parent, w, h, near=op.near)
    ox, oy = _parent_abs_origin(doc, parent)
    cell = ET.SubElement(doc.root(), "mxCell", id=new_id, value=op.label, style=style,
                         vertex="1", parent=parent)
    _set_geom(cell, x=ax - ox, y=ay - oy, w=w, h=h)
    cell.find("mxGeometry").set("as", "geometry")
    if not op.fit_to:  # a fit_to box deliberately surrounds others; don't resize its container for it
        _grow_to_contain(doc, parent)


def _op_add_connection(doc, op: AddConnection, refs):
    for end, cid in (("source", op.source), ("target", op.target)):
        c = _require(doc, cid, end)
        if dd.attr(c, "vertex") != "1":
            raise OpError(f"{end} {cid!r} is not a node")
    res = _resolve_like(doc, op.like, refs)
    style = res[0] if res and res[0] else DEFAULT_EDGE_STYLE
    new_id = op.id or _unique_id(doc, f"e_{op.source}_{op.target}")
    if doc.cell(new_id) is not None:
        raise OpError(f"id {new_id!r} already exists")
    cell = ET.SubElement(doc.root(), "mxCell", id=new_id, value=op.label, style=style,
                         edge="1", parent="1", source=op.source, target=op.target)
    ET.SubElement(cell, "mxGeometry", relative="1").set("as", "geometry")


def _op_scale_fonts(doc, op: ScaleFonts, refs):
    from .text_fit import DEFAULT_FONT

    if op.ids is not None:
        targets = [_require(doc, i) for i in op.ids]
    else:
        targets = [c for c in doc.cells() if c.get("id") not in ("0", "1") and dd.attr(c, "value")]
    for c in targets:
        style = dd.parse_style(dd.attr(c, "style"))
        try:
            cur = float(style.get("fontSize", DEFAULT_FONT))
        except ValueError:
            cur = float(DEFAULT_FONT)
        style["fontSize"] = str(max(op.min_size, min(op.max_size, round(cur * op.factor))))
        dd.set_attr(c, "style", dd.format_style(style))


# -- generic XML-level ops ----------------------------------------------------

_MAX_FRAGMENT = 20_000
_RESERVED = ("0", "1")


def _op_upsert_xml(doc, op: UpsertXml, refs):
    if len(op.xml) > _MAX_FRAGMENT:
        raise OpError(f"xml fragment too large ({len(op.xml)} chars, max {_MAX_FRAGMENT}); split it")
    try:
        new = SafeET.fromstring(op.xml.strip())
    except (ET.ParseError, DefusedXmlException) as exc:
        raise OpError(f"upsert_xml: not well-formed XML ({exc})") from exc
    if new.tag not in dd.CELL_TAGS:
        raise OpError(f"upsert_xml: expected a single <mxCell> (or <object>/<UserObject> wrapping one), got <{new.tag}>")
    if new.tag == "mxCell":
        node = new
    else:
        inner_cells = [c for c in new if c.tag == "mxCell"]
        if len(inner_cells) != 1 or len(new) != 1:
            raise OpError(f"upsert_xml: <{new.tag}> must contain exactly one <mxCell>")
        node = inner_cells[0]
        if node.get("id"):
            raise OpError(f"upsert_xml: the id belongs on <{new.tag}>, not on its inner <mxCell>")
    for child in node:
        if child.tag != "mxGeometry":
            raise OpError(f"upsert_xml: unexpected child <{child.tag}> (only <mxGeometry> is allowed)")
    cid = new.get("id")
    if not cid:
        raise OpError("upsert_xml: the cell needs an id")
    if cid in _RESERVED:
        raise OpError(f"upsert_xml: id {cid!r} is reserved")
    if node.get("vertex") != "1" and node.get("edge") != "1":
        raise OpError("upsert_xml: the cell needs vertex=\"1\" or edge=\"1\"")
    if not node.get("parent"):
        node.set("parent", "1")
    root = doc.root()
    for i, existing in enumerate(list(root)):
        if existing.tag in dd.CELL_TAGS and existing.get("id") == cid:
            root.remove(existing)
            root.insert(i, new)  # replaced in place: draw order is preserved
            return
    root.append(new)


def _op_set_attrs(doc, op: SetAttrs, refs):
    c = _require(doc, op.id)
    for k, v in op.attrs.items():
        if k == "id":
            raise OpError("set_attrs: the id can't be changed")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", k):
            raise OpError(f"set_attrs: invalid attribute name {k!r}")
        try:
            # routed to the right element: for a wrapped cell, style/parent/... go to the inner
            # mxCell, `value` becomes the wrapper's `label`, anything else is a custom property
            dd.set_attr(c, k, v)
        except ValueError as exc:
            raise OpError(f"set_attrs: {exc}") from exc


def _op_set_geometry(doc, op: SetGeometry, refs):
    c = _require(doc, op.id)
    if dd.geometry(c) is None and dd.attr(c, "vertex") != "1":
        raise OpError(f"{op.id!r} has no geometry to set")
    _set_geom(c, x=op.x, y=op.y, w=op.width, h=op.height)


def _op_reorder(doc, op: Reorder, refs):
    c = _require(doc, op.id)
    root = doc.root()
    if op.to in ("before", "after"):
        if not op.ref:
            raise OpError(f"reorder: '{op.to}' needs `ref`")
        ref = _require(doc, op.ref, "ref")
        if ref is c:
            raise OpError("reorder: a cell can't be ordered relative to itself")
    root.remove(c)
    kids = list(root)
    if op.to == "back":
        # behind every cell, but after the structural cells 0 and 1 that must stay first
        idx = max((i + 1 for i, k in enumerate(kids) if k.get("id") in _RESERVED), default=0)
    elif op.to == "front":
        idx = len(kids)
    else:
        ref_idx = next(i for i, k in enumerate(kids) if k is ref)
        idx = ref_idx if op.to == "before" else ref_idx + 1
    root.insert(idx, c)


_DISPATCH = {
    UpdateLabel: _op_update_label, Move: _op_move, Resize: _op_resize, SetStyle: _op_set_style,
    Delete: _op_delete, AddComponent: _op_add_component, AddConnection: _op_add_connection,
    ScaleFonts: _op_scale_fonts, UpsertXml: _op_upsert_xml, SetAttrs: _op_set_attrs, SetGeometry: _op_set_geometry, Reorder: _op_reorder,
}


def apply_ops(doc: dd.Doc, ops: list, refs: Optional[RefResolver] = None) -> tuple[Optional[dd.Doc], list[dict]]:
    """Apply `ops` in order to a copy of `doc`.

    Returns (new_doc, results). If any op fails, new_doc is None and `results`
    says which op failed and why; the input doc is never modified.
    """
    work = clone(doc)
    results: list[dict] = []
    failed = False
    for i, op in enumerate(ops):
        try:
            _DISPATCH[type(op)](work, op, refs)
            results.append({"index": i, "op": op.op, "ok": True})
        except OpError as exc:
            failed = True
            results.append({"index": i, "op": op.op, "ok": False, "error": str(exc)})
            break
    return (None if failed else work), results
