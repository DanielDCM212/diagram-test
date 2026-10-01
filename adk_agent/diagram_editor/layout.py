"""Deterministic layout for create mode: ArchitectureSpec -> draw.io document.

The LLM states structure (zones, components, connections) and which reference
cells to imitate. Every coordinate here comes from the metrics derived from the
chosen `layout_from` reference (zone arrangement, padding, gaps, grid), and
every style/size from the referenced cells -- so two requests for similar
diagrams come out with the same look and layout.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from . import drawio_doc as dd
from .edit_ops import DEFAULT_EDGE_STYLE, DEFAULT_H, DEFAULT_STYLE, DEFAULT_W
from .references import (
    ReferenceLookupError, derive_profile, load_reference, resolve_ref_cell,
)
from .schema import ArchitectureSpec

MARGIN = 20.0
_TITLE_STYLE = "text;html=1;align=left;verticalAlign=middle;whiteSpace=wrap;fontSize=16;fontStyle=1;"


class LayoutError(Exception):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def _snap(v: float, grid: int) -> float:
    return float(round(v / grid) * grid) if grid > 1 else float(round(v))


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


def _check_spec(spec: ArchitectureSpec) -> list[str]:
    problems = []
    ids = [z.id for z in spec.zones] + [c.id for c in spec.components]
    for i in {x for x in ids if ids.count(x) > 1}:
        problems.append(f"duplicate id {i!r}")
    for i in ids:
        if i in ("0", "1", "title"):
            problems.append(f"id {i!r} is reserved")
    zone_ids = {z.id for z in spec.zones}
    for z in spec.zones:
        if z.parent and z.parent not in zone_ids:
            problems.append(f"zone {z.id!r}: unknown parent zone {z.parent!r}")
        seen, cur = {z.id}, z.parent
        while cur and cur in zone_ids:
            if cur in seen:
                problems.append(f"zone {z.id!r}: parent cycle")
                break
            seen.add(cur)
            cur = next(p.parent for p in spec.zones if p.id == cur)
    for c in spec.components:
        if c.zone and c.zone not in zone_ids:
            problems.append(f"component {c.id!r}: unknown zone {c.zone!r}")
    all_ids = set(ids)
    for e in spec.connections:
        for end in (e.source, e.target):
            if end not in all_ids:
                problems.append(f"connection {e.source}->{e.target}: unknown id {end!r}")
            elif end in zone_ids:
                problems.append(f"connection {e.source}->{e.target}: {end!r} is a zone, connect components")
    return problems


def build_diagram(spec: ArchitectureSpec, directory: Optional[Path] = None) -> tuple[dd.Doc, dict]:
    problems = _check_spec(spec)
    try:
        profile = derive_profile(load_reference(spec.layout_from, directory))
    except ReferenceLookupError as exc:
        raise LayoutError(problems + [str(exc)]) from exc
    L = profile["layout"]
    grid = int(L["grid"]) or 10
    pad_l, pad_t = float(L["zone_padding_left"]), float(L["zone_padding_top"])
    gap = float(L["component_gap"]) or 20.0
    per_row = int(L["components_per_row"]) or 1
    zone_gap = float(L["zone_gap"]) or 20.0

    # resolve `like` cells (style, w, h), collecting every miss at once
    def resolve(ref, what):
        res = resolve_ref_cell(ref, directory)
        if res is None:
            problems.append(f"{what}: reference cell {ref.reference}:{ref.cell} not found")
        return res

    zstyle = {z.id: resolve(z.like, f"zone {z.id!r}") for z in spec.zones}
    cstyle = {c.id: resolve(c.like, f"component {c.id!r}") for c in spec.components}
    estyle = [resolve(e.like, f"connection {e.source}->{e.target}") if e.like else None for e in spec.connections]
    if problems:
        raise LayoutError(problems)

    def comp_size(cid):
        _, w, h = cstyle[cid]
        return (w or DEFAULT_W), (h or DEFAULT_H)

    # ---- sizing (bottom-up) -------------------------------------------------
    sub_zones: dict[Optional[str], list[str]] = {}
    for z in spec.zones:
        sub_zones.setdefault(z.parent, []).append(z.id)
    comps_in: dict[Optional[str], list[str]] = {}
    for c in spec.components:
        comps_in.setdefault(c.zone, []).append(c.id)

    placed: dict[str, tuple[Optional[str], float, float, float, float]] = {}  # id -> (parent, x, y, w, h) relative

    def layout_zone(zid: str) -> tuple[float, float]:
        rows: list[list[tuple[str, float, float]]] = []
        cur: list[tuple[str, float, float]] = []
        for cid in comps_in.get(zid, []):
            w, h = comp_size(cid)
            cur.append((cid, w, h))
            if len(cur) >= per_row:
                rows.append(cur)
                cur = []
        if cur:
            rows.append(cur)
        for sz in sub_zones.get(zid, []):
            w, h = layout_zone(sz)
            rows.append([(sz, w, h)])
        y = pad_t
        max_w = 0.0
        for row in rows:
            x = pad_l
            for item, w, h in row:
                placed[item] = (zid, _snap(x, grid), _snap(y, grid), w, h)
                x += w + gap
            max_w = max(max_w, x - gap)
            y += max(h for _, _, h in row) + gap
        zw = max(max_w + pad_l, pad_l * 2 + DEFAULT_W) if rows else pad_l * 2 + DEFAULT_W
        zh = (y - gap + pad_l) if rows else pad_t + pad_l
        placed.setdefault(zid, (None, 0, 0, 0, 0))
        return zw, zh

    top = sub_zones.get(None, [])
    sizes = {z: layout_zone(z) for z in top}

    # ---- top-level placement (arrangement from the reference) ---------------
    title_h = float((L.get("title") or {}).get("h") or 30) if spec.title else 0.0
    ox = oy = _snap(MARGIN, grid)
    if spec.title:
        oy = _snap(MARGIN + title_h + MARGIN / 2, grid)
    columns = L["zone_arrangement"] == "columns"
    cross = max((s[1] if columns else s[0]) for s in sizes.values()) if sizes else 0.0
    cursor = 0.0
    for z in top:
        w, h = sizes[z]
        if columns:
            placed[z] = (None, _snap(ox + cursor, grid), oy, w, cross)
            cursor += w + zone_gap
        else:
            placed[z] = (None, ox, _snap(oy + cursor, grid), cross, h)
            cursor += h + zone_gap
    # free-standing components: a strip after the zones
    free = comps_in.get(None, [])
    strip = cursor if top else 0.0
    fx = 0.0
    for cid in free:
        w, h = comp_size(cid)
        if columns or not top:
            placed[cid] = (None, _snap(ox + fx, grid), _snap(oy + strip, grid) if top else oy, w, h)
            fx += w + gap
        else:
            placed[cid] = (None, _snap(ox + strip, grid), _snap(oy + fx, grid), w, h)
            fx += h + gap
    # the equalised cross-axis size must also propagate to nested children that were sized bottom-up
    # (only top-level zones are stretched; nested zones keep their content size)

    # ---- emit ---------------------------------------------------------------
    mxfile = ET.Element("mxfile", host="app.diagrams.net")
    diagram = ET.SubElement(mxfile, "diagram", id="page-1", name=spec.title or "Page-1")
    model = ET.SubElement(diagram, "mxGraphModel", dx="800", dy="600", grid="1", gridSize=str(grid), guides="1",
                          tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1",
                          pageWidth="1100", pageHeight="850", math="0", shadow="0")
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    def add_vertex(cid, label, style, parent, x, y, w, h):
        cell = ET.SubElement(root, "mxCell", id=cid, value=label, style=style, vertex="1", parent=parent or "1")
        geo = ET.SubElement(cell, "mxGeometry", x=_fmt(x), y=_fmt(y), width=_fmt(w), height=_fmt(h))
        geo.set("as", "geometry")

    if spec.title:
        t = L.get("title")
        add_vertex("title", spec.title, (t and t["style"]) or _TITLE_STYLE, None,
                   ox, _snap(MARGIN, grid), max(300.0, cursor if not columns else 600.0), title_h)

    def zone_style(zid):
        st = dd.parse_style(zstyle[zid][0] or DEFAULT_STYLE)
        if not any(k in st for k in ("verticalAlign", "swimlane", "startSize")):
            st["verticalAlign"] = "top"
        return dd.format_style(st)

    def emit_zone(zid):  # parents before children
        parent, x, y, w, h = placed[zid]
        add_vertex(zid, next(z.label for z in spec.zones if z.id == zid), zone_style(zid), parent, x, y, w, h)
        for c in comps_in.get(zid, []):
            p, cx, cy, cw, ch = placed[c]
            add_vertex(c, next(k.label for k in spec.components if k.id == c), cstyle[c][0] or DEFAULT_STYLE, p, cx, cy, cw, ch)
        for sz in sub_zones.get(zid, []):
            emit_zone(sz)

    for z in top:
        emit_zone(z)
    for c in free:
        p, cx, cy, cw, ch = placed[c]
        add_vertex(c, next(k.label for k in spec.components if k.id == c), cstyle[c][0] or DEFAULT_STYLE, None, cx, cy, cw, ch)

    default_edge = (profile["edge_kinds"][0]["style"] if profile["edge_kinds"] else DEFAULT_EDGE_STYLE) or DEFAULT_EDGE_STYLE
    used: set[str] = set()
    for i, (e, es) in enumerate(zip(spec.connections, estyle)):
        eid = f"e_{e.source}_{e.target}"
        if eid in used:
            eid = f"{eid}_{i}"
        used.add(eid)
        cell = ET.SubElement(root, "mxCell", id=eid, value=e.label, style=(es[0] if es and es[0] else default_edge),
                             edge="1", parent="1", source=e.source, target=e.target)
        ET.SubElement(cell, "mxGeometry", relative="1").set("as", "geometry")

    doc = dd.load(dd.serialize(dd.Doc(mxfile, [diagram])))
    stats = {"zones": len(spec.zones), "components": len(spec.components), "connections": len(spec.connections),
             "arrangement": L["zone_arrangement"], "layout_from": spec.layout_from}
    return doc, stats
