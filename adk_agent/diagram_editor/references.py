"""Reads the house-style reference diagrams at request time and derives what the
layout engine needs from them -- no offline profile, nothing to keep in sync.

`derive_profile(doc)` is deterministic: it clusters vertex/edge styles and
measures zone arrangement, padding and gaps straight from the geometry.
Zones are detected both structurally (cells that are another cell's parent)
and geometrically (a big unparented box that fully contains >= 2 others),
because many real diagrams are flat -- every cell parented to the root layer.
"""
from __future__ import annotations

import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from . import drawio_doc as dd
from .schema import RefCell

REFERENCE_DIR = Path(os.environ.get("DIAGRAM_REFERENCE_DIR", Path(__file__).parent / "reference_diagrams"))

_NODE_KEYS = ("shape", "rounded", "dashed", "fillColor", "strokeColor", "fontSize", "fontStyle", "fontColor", "strokeWidth")
_EDGE_KEYS = ("edgeStyle", "dashed", "endArrow", "startArrow", "strokeColor", "strokeWidth", "curved")


class ReferenceLookupError(Exception):
    pass


def _ref_path(name: str, directory: Optional[Path] = None) -> Path:
    directory = Path(directory or REFERENCE_DIR)
    if not name or Path(name).name != name or not name.lower().endswith((".drawio", ".xml")):
        raise ReferenceLookupError(f"invalid reference name {name!r}")
    path = directory / name
    if not path.is_file():
        raise ReferenceLookupError(f"no reference named {name!r}")
    return path


def load_reference(name: str, directory: Optional[Path] = None) -> dd.Doc:
    try:
        return dd.load(_ref_path(name, directory).read_text(encoding="utf-8"))
    except dd.DrawioParseError as exc:
        raise ReferenceLookupError(f"reference {name!r} is not valid draw.io XML: {exc}") from exc


# -- structure detection ------------------------------------------------------

def _is_text(style: dict) -> bool:
    return "text" in style or style.get("fillColor") == "none" and style.get("strokeColor") == "none"


def _vertices(doc: dd.Doc, page: int = 0):
    out = []
    for c in doc.cells(page):
        if dd.attr(c, "vertex") == "1" and c.get("id") not in ("0", "1"):
            box = dd.absolute_box(doc, c, page)
            if box:
                out.append((c, box))
    return out


def find_zones(doc: dd.Doc, page: int = 0) -> dict[str, list[str]]:
    """zone id -> ids of vertices inside it (by parent link or full geometric containment)."""
    verts = _vertices(doc, page)
    boxes = {c.get("id"): b for c, b in verts}
    members: dict[str, set[str]] = defaultdict(set)
    parent_ids = {dd.attr(c, "parent") for c, _ in verts}
    for c, _ in verts:
        if c.get("id") in parent_ids:
            members[c.get("id")]
    for c, _ in verts:
        if dd.attr(c, "parent") in boxes:
            members[dd.attr(c, "parent")].add(c.get("id"))
    for zid, (zx, zy, zw, zh) in boxes.items():
        if zw * zh < 4000:
            continue
        inside = [i for i, (x, y, w, h) in boxes.items()
                  if i != zid and x >= zx - 1 and y >= zy - 1 and x + w <= zx + zw + 1 and y + h <= zy + zh + 1
                  and w * h < zw * zh]
        if len(inside) >= 2:
            members[zid].update(inside)
    return {z: sorted(m) for z, m in members.items()}


def _median(values, default=0.0):
    values = list(values)
    return statistics.median(values) if values else default


def _digest(style: dict, keys) -> tuple:
    return tuple((k, style.get(k)) for k in keys if style.get(k) is not None)


# -- profile ------------------------------------------------------------------

def derive_profile(doc: dd.Doc, page: int = 0) -> dict:
    zones = find_zones(doc, page)
    verts = _vertices(doc, page)
    cells = {c.get("id"): c for c, _ in verts}
    boxes = {c.get("id"): b for c, b in verts}
    zone_ids = set(zones)

    # top-level zones = not contained in another zone
    contained = {i for members in zones.values() for i in members}
    top_zones = [z for z in zone_ids if z not in contained]
    top_zones.sort(key=lambda z: (boxes[z][1], boxes[z][0]))

    # node kinds (leaf, non-text vertices)
    groups: dict[tuple, list[str]] = defaultdict(list)
    text_cells = []
    for cid, c in cells.items():
        if cid in zone_ids:
            continue
        st = dd.parse_style(dd.attr(c, "style"))
        if _is_text(st):
            text_cells.append(cid)
            continue
        groups[_digest(st, _NODE_KEYS)].append(cid)
    node_kinds = []
    for digest, ids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        ex = ids[0]
        node_kinds.append({
            "example_cell": ex,
            "count": len(ids),
            "style": dd.attr(cells[ex], "style") or "",
            "w": round(_median(boxes[i][2] for i in ids)),
            "h": round(_median(boxes[i][3] for i in ids)),
            "sample_labels": [dd.plain_label(dd.attr(cells[i], "value")) for i in ids[:4]],
        })

    zone_kinds = []
    zgroups: dict[tuple, list[str]] = defaultdict(list)
    for zid in zone_ids:
        zgroups[_digest(dd.parse_style(dd.attr(cells[zid], "style")), _NODE_KEYS)].append(zid)
    for digest, ids in sorted(zgroups.items(), key=lambda kv: -len(kv[1])):
        ex = ids[0]
        zone_kinds.append({
            "example_cell": ex, "count": len(ids),
            "style": dd.attr(cells[ex], "style") or "",
            "sample_labels": [dd.plain_label(dd.attr(cells[i], "value")) for i in ids[:4]],
        })

    # edge kinds
    egroups: dict[tuple, list] = defaultdict(list)
    for c in doc.cells(page):
        if dd.attr(c, "edge") == "1":
            egroups[_digest(dd.parse_style(dd.attr(c, "style")), _EDGE_KEYS)].append(c)
    edge_kinds = [{
        "example_cell": es[0].get("id"), "count": len(es), "style": dd.attr(es[0], "style") or "",
        "dashed": dd.parse_style(dd.attr(es[0], "style")).get("dashed") == "1",
    } for _, es in sorted(egroups.items(), key=lambda kv: -len(kv[1]))]

    # layout metrics --------------------------------------------------------
    arrangement = "rows"
    zone_gap = 20.0
    if len(top_zones) >= 2:
        cx = [boxes[z][0] + boxes[z][2] / 2 for z in top_zones]
        cy = [boxes[z][1] + boxes[z][3] / 2 for z in top_zones]
        arrangement = "columns" if (max(cx) - min(cx)) >= (max(cy) - min(cy)) else "rows"
        axis = 0 if arrangement == "columns" else 1
        order = sorted(top_zones, key=lambda z: boxes[z][axis])
        gaps = []
        for a, b in zip(order, order[1:]):
            ga = boxes[b][axis] - (boxes[a][axis] + boxes[a][2 + axis])
            if ga >= 0:
                gaps.append(ga)
        zone_gap = _median(gaps, 20.0)

    pad_left, pad_top, comp_gaps, inner_cols = [], [], [], []
    for zid, members in zones.items():
        kids = [m for m in members if m not in zone_ids and m not in text_cells]
        if not kids:
            continue
        zx, zy = boxes[zid][0], boxes[zid][1]
        pad_left.append(min(boxes[k][0] for k in kids) - zx)
        pad_top.append(min(boxes[k][1] for k in kids) - zy)
        inner_cols.append(len({round(boxes[k][0] / 10) for k in kids}))
        for k in kids:
            near = [max(boxes[o][0] - (boxes[k][0] + boxes[k][2]), boxes[k][0] - (boxes[o][0] + boxes[o][2]),
                        boxes[o][1] - (boxes[k][1] + boxes[k][3]), boxes[k][1] - (boxes[o][1] + boxes[o][3]))
                    for o in kids if o != k]
            near = [g for g in near if g >= 0]
            if near:
                comp_gaps.append(min(near))

    all_boxes = list(boxes.values())
    snap_votes = Counter(10 if all(round(v) % 10 == 0 for v in (b[0], b[1])) else 1 for b in all_boxes)

    title = None
    if text_cells and top_zones:
        top_y = min(boxes[z][1] for z in top_zones)
        above = [t for t in text_cells if boxes[t][1] + boxes[t][3] <= top_y + 5 and dd.plain_label(dd.attr(cells[t], "value"))]
        if above:
            t = max(above, key=lambda i: boxes[i][2] * boxes[i][3])
            title = {"example_cell": t, "style": dd.attr(cells[t], "style") or "", "h": round(boxes[t][3])}

    return {
        "zones": len(zone_ids), "top_level_zones": len(top_zones),
        "node_kinds": node_kinds[:12], "zone_kinds": zone_kinds[:6], "edge_kinds": edge_kinds[:6],
        "layout": {
            "zone_arrangement": arrangement,
            "zone_gap": round(zone_gap),
            "zone_padding_left": round(_median(pad_left, 20.0)),
            "zone_padding_top": round(_median(pad_top, 40.0)),
            "component_gap": round(_median(comp_gaps, 20.0)),
            "components_per_row": max(1, round(_median(inner_cols, 1))),
            "grid": snap_votes.most_common(1)[0][0] if all_boxes else 10,
            "canvas": [round(max(b[0] + b[2] for b in all_boxes)), round(max(b[1] + b[3] for b in all_boxes))] if all_boxes else [0, 0],
            "title": title,
        },
    }


def list_references(directory: Optional[Path] = None) -> list[dict]:
    directory = Path(directory or REFERENCE_DIR)
    out = []
    for path in sorted(directory.glob("*")) if directory.is_dir() else []:
        if path.suffix.lower() not in (".drawio", ".xml"):
            continue
        try:
            doc = dd.load(path.read_text(encoding="utf-8"))
        except (dd.DrawioParseError, OSError) as exc:
            out.append({"name": path.name, "error": str(exc)})
            continue
        zones = find_zones(doc)
        cells = {c.get("id"): c for c in doc.cells()}
        s = dd.summarize(doc)
        leaf = [v["label"] for v in s["vertices"] if v["id"] not in zones and v.get("label")]
        out.append({
            "name": path.name,
            "title": s["page"],
            "nodes": len(s["vertices"]), "edges": len(s["edges"]),
            "zones": [dd.plain_label(dd.attr(cells[z], "value")) for z in zones if dd.plain_label(dd.attr(cells[z], "value"))][:8],
            "sample_labels": leaf[:8],
        })
    return out


def resolve_ref_cell(ref: RefCell, directory: Optional[Path] = None) -> Optional[tuple[str, float, float]]:
    """RefResolver for edit_ops: (style, w, h) of a reference cell, or None."""
    try:
        doc = load_reference(ref.reference, directory)
    except ReferenceLookupError:
        return None
    c = doc.cell(ref.cell)
    if c is None:
        return None
    box = dd.local_box(c)
    return (dd.attr(c, "style") or "", box[2] if box else 0.0, box[3] if box else 0.0)
