"""Find and resolve overlapping nodes in a draw.io diagram.

What counts as an overlap: two nodes whose boxes intersect where neither contains the other.
Nesting is intentional and never flagged -- a node inside a container (by `parent`, or
geometrically fully inside a bigger box, as in flat diagrams), or a label sitting on a box.

Resolution escalates, least invasive first, and stays bounded and deterministic:
  1. MOVE   push overlapping peers apart along the axis of least penetration, each by at most
            `max_shift` in total. A moved node carries everything inside it; containers grow to
            keep their contents.
  2. SHRINK if moving was not enough, shrink leaf boxes toward the side facing the other box,
            never below `min_scale` of their original size.
  3. FONT   shrink fonts of resized boxes (never grow) so labels still fit, down to `min_font`.
Anything that can't be resolved within those limits is reported, not forced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional
from xml.etree import ElementTree as ET

from . import drawio_doc as dd
from . import text_fit
from .edit_ops import _fmt, clone

TOL = 1.0               # px of intersection ignored (touching borders, rounding)
MIN_W, MIN_H = 24.0, 16.0
SHRINK_CLEARANCE = 2.0  # px left between boxes separated by shrinking
GROW_PAD = 10.0
MAX_ITERS = 80
REPORT_CAP = 40


@dataclass
class Node:
    id: str
    cell: ET.Element
    x: float
    y: float
    w: float
    h: float
    parent_attr: Optional[str]
    ox: float = 0.0
    oy: float = 0.0
    ow: float = 0.0
    oh: float = 0.0
    container: Optional[str] = None
    children: list[str] = field(default_factory=list)
    depth: int = 0
    locked: bool = False
    shift: float = 0.0          # own accumulated displacement (|dx|+|dy|), bounded by max_shift
    dx: float = 0.0             # own accumulated displacement vector (for the report)
    dy: float = 0.0
    resized: bool = False
    grown: bool = False

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class Params:
    gap: float = 10.0
    max_shift: float = 80.0
    allow_move: bool = True
    allow_resize: bool = True
    min_scale: float = 0.75
    allow_font_shrink: bool = True
    min_font: int = 8
    fix_overflow: bool = False
    allow_grow: bool = True
    lock: frozenset = frozenset()
    only: Optional[frozenset] = None


def _contains(m: Node, n: Node, tol: float = TOL) -> bool:
    return (m.x - tol <= n.x and m.y - tol <= n.y
            and n.x + n.w <= m.x + m.w + tol and n.y + n.h <= m.y + m.h + tol)


def penetration(a: Node, b: Node) -> tuple[float, float]:
    """(x, y) overlap extents; either can be <= 0 when the boxes are apart on that axis."""
    px = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    py = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    return px, py


class Model:
    """Absolute-coordinate working copy of every node, plus the containment tree."""

    def __init__(self, doc: dd.Doc, page: int = 0):
        self.doc, self.page = doc, page
        self.nodes: dict[str, Node] = {}
        cells = doc.cells(page)
        edge_ids = {c.get("id") for c in cells if dd.attr(c, "edge") == "1"}
        for c in cells:
            cid = c.get("id")
            if cid in ("0", "1") or dd.attr(c, "vertex") != "1":
                continue
            if dd.attr(c, "parent") in edge_ids:  # edge labels: coordinates are relative to the edge
                continue
            box = dd.absolute_box(doc, c, page)
            if not box or box[2] <= 0 or box[3] <= 0:
                continue
            n = Node(cid, c, *box, parent_attr=None)
            n.ox, n.oy, n.ow, n.oh = box
            self.nodes[cid] = n
        for n in self.nodes.values():
            pa = dd.attr(n.cell, "parent")
            n.parent_attr = pa if pa in self.nodes and pa != n.id else None
        self._link()

    # -- containment tree -----------------------------------------------------
    def _explicit_ancestors(self, nid: str) -> set[str]:
        out, cur = set(), self.nodes[nid].parent_attr
        while cur and cur not in out:
            out.add(cur)
            cur = self.nodes[cur].parent_attr
        return out

    def _link(self) -> None:
        nodes = self.nodes
        explicit_anc = {i: self._explicit_ancestors(i) for i in nodes}
        for n in nodes.values():
            if n.parent_attr and n.id not in explicit_anc[n.parent_attr]:
                n.container = n.parent_attr
                continue
            best: Optional[Node] = None
            for m in nodes.values():
                if m is n or m.area <= n.area + 1e-6:
                    continue
                if n.id in explicit_anc[m.id]:  # m is an explicit descendant of n: would make a cycle
                    continue
                if _contains(m, n) and (best is None or m.area < best.area):
                    best = m
            n.container = best.id if best else None
        for n in nodes.values():
            n.children.clear()
        for n in nodes.values():
            if n.container:
                nodes[n.container].children.append(n.id)
        for n in nodes.values():
            d, cur, seen = 0, n.container, {n.id}
            while cur and cur not in seen:
                seen.add(cur)
                d += 1
                cur = nodes[cur].container
            n.depth = d

    def ancestors(self, nid: str) -> set[str]:
        out, cur = set(), self.nodes[nid].container
        while cur and cur not in out:
            out.add(cur)
            cur = self.nodes[cur].container
        return out

    def group(self, nid: str) -> list[str]:
        out, stack = [], [nid]
        while stack:
            cur = stack.pop()
            out.append(cur)
            stack.extend(self.nodes[cur].children)
        return out

    def translate(self, nid: str, dx: float, dy: float) -> None:
        for i in self.group(nid):
            self.nodes[i].x += dx
            self.nodes[i].y += dy

    # -- queries --------------------------------------------------------------
    def overlap_pairs(self, tol: float = TOL) -> list[tuple[Node, Node, float, float]]:
        out = []
        ids = sorted(self.nodes)
        anc = {i: self.ancestors(i) for i in ids}
        for ai, a_id in enumerate(ids):
            a = self.nodes[a_id]
            for b_id in ids[ai + 1:]:
                if b_id in anc[a_id] or a_id in anc[b_id]:
                    continue
                b = self.nodes[b_id]
                px, py = penetration(a, b)
                if px > tol and py > tol:
                    # one box strictly inside the other is nesting, not an overlap (identical boxes are)
                    if (a.area > b.area + 1e-6 and _contains(a, b)) or (b.area > a.area + 1e-6 and _contains(b, a)):
                        continue
                    out.append((a, b, px, py))
        return out

    def levels(self) -> list[list[str]]:
        """Peer groups (children of one container, or the roots), deepest containers first."""
        groups: dict[Optional[str], list[str]] = {}
        for n in self.nodes.values():
            groups.setdefault(n.container, []).append(n.id)
        key = lambda c: -(self.nodes[c].depth + 1) if c else 0  # noqa: E731
        return [sorted(groups[c]) for c in sorted(groups, key=key)]


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def _label(n: Node) -> str:
    return dd.plain_label(dd.attr(n.cell, "value"))[:40]


def find_overlaps(doc: dd.Doc, page: int = 0) -> dict:
    m = Model(doc, page)
    pairs = m.overlap_pairs()
    rows = []
    for a, b, px, py in pairs:
        rows.append({
            "a": a.id, "a_label": _label(a), "b": b.id, "b_label": _label(b),
            "overlap_w": round(px, 1), "overlap_h": round(py, 1),
            "severity": round(px * py / min(a.area, b.area), 2),  # 1.0 = the smaller box is fully covered
        })
    rows.sort(key=lambda r: -r["severity"])

    overflow = []
    for n in m.nodes.values():
        style = dd.parse_style(dd.attr(n.cell, "style"))
        fit = text_fit.best_font(dd.attr(n.cell, "value"), style, n.w, n.h, min_font=6)
        if fit is not None:
            overflow.append({"id": n.id, "label": _label(n), "font": text_fit.current_font(style),
                             "box": [round(n.w), round(n.h)], "suggested_font": fit[0], "fits_then": fit[1]})
    outside = [{"id": n.id, "container": n.parent_attr}
               for n in m.nodes.values()
               if n.parent_attr and not _contains(m.nodes[n.parent_attr], n)]
    return {
        "node_count": len(m.nodes),
        "overlap_count": len(rows),
        "overlaps": rows[:REPORT_CAP],
        "label_overflow_count": len(overflow),
        "label_overflow": overflow[:REPORT_CAP],
        "outside_container": outside[:REPORT_CAP],
        "truncated": len(rows) > REPORT_CAP or len(overflow) > REPORT_CAP,
    }


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------

class _Resolver:
    def __init__(self, doc: dd.Doc, page: int, p: Params):
        self.m = Model(doc, page)
        self.p = p
        missing = [i for i in (set(p.lock) | set(p.only or ())) if i not in self.m.nodes]
        if missing:
            raise ValueError(f"unknown node id(s): {sorted(missing)}")
        for i in p.lock:
            self.m.nodes[i].locked = True
        self.allowed: Optional[set[str]] = None
        if p.only is not None:
            self.allowed = set()
            for i in p.only:
                self.allowed.update(self.m.group(i))
        self._frozen_cache: dict[str, bool] = {}

    # -- permissions ----------------------------------------------------------
    def _frozen(self, nid: str) -> bool:
        n = self.m.nodes[nid]
        return n.locked or (self.allowed is not None and nid not in self.allowed)

    def can_move(self, nid: str) -> bool:
        if nid not in self._frozen_cache:
            self._frozen_cache[nid] = any(self._frozen(i) for i in self.m.group(nid))
        return not self._frozen_cache[nid]

    def can_resize(self, nid: str) -> bool:
        return not self._frozen(nid)

    # -- phase 1: move --------------------------------------------------------
    def _budget(self, n: Node) -> float:
        return max(0.0, self.p.max_shift - n.shift) if self.can_move(n.id) else 0.0

    def _separate(self, a: Node, b: Node) -> bool:
        px, py = penetration(a, b)
        if px <= TOL or py <= TOL:
            return False
        axis = 0 if px <= py else 1
        need = (px if axis == 0 else py) + self.p.gap
        ca = (a.x + a.w / 2) if axis == 0 else (a.y + a.h / 2)
        cb = (b.x + b.w / 2) if axis == 0 else (b.y + b.h / 2)
        sign = 1.0 if cb >= ca else -1.0
        ra, rb = self._budget(a), self._budget(b)
        da, db = min(need / 2, ra), min(need / 2, rb)
        rest = need - da - db
        if rest > 0:
            extra = min(rest, ra - da)
            da, rest = da + extra, rest - extra
            db += min(rest, rb - db)
        if da + db < 0.5:
            return False
        for n, d, s in ((a, da, -sign), (b, db, sign)):
            if d <= 0:
                continue
            vec = (s * d, 0.0) if axis == 0 else (0.0, s * d)
            self.m.translate(n.id, *vec)
            n.shift += d
            n.dx += vec[0]
            n.dy += vec[1]
        return True

    def grow_containers(self) -> bool:
        changed = False
        for n in sorted((x for x in self.m.nodes.values() if x.children), key=lambda x: -x.depth):
            if not self.can_resize(n.id):
                continue
            kids = [self.m.nodes[c] for c in n.children]
            x0, y0 = min(k.x for k in kids), min(k.y for k in kids)
            x1, y1 = max(k.x + k.w for k in kids), max(k.y + k.h for k in kids)
            if x0 >= n.x - TOL and y0 >= n.y - TOL and x1 <= n.x + n.w + TOL and y1 <= n.y + n.h + TOL:
                continue
            nx0, ny0 = min(n.x, x0 - GROW_PAD), min(n.y, y0 - GROW_PAD)
            nx1, ny1 = max(n.x + n.w, x1 + GROW_PAD), max(n.y + n.h, y1 + GROW_PAD)
            n.x, n.y, n.w, n.h = nx0, ny0, nx1 - nx0, ny1 - ny0
            n.grown = changed = True
        return changed

    def move_phase(self) -> None:
        for _ in range(MAX_ITERS):
            changed = self.grow_containers() if self.p.allow_grow else False
            for level in self.m.levels():
                for i, a_id in enumerate(level):
                    for b_id in level[i + 1:]:
                        if self._separate(self.m.nodes[a_id], self.m.nodes[b_id]):
                            changed = True
            if not changed:
                break

    # -- phase 2: shrink ------------------------------------------------------
    def _capacity(self, n: Node, axis: int) -> float:
        if n.children or not self.can_resize(n.id):
            return 0.0
        if axis == 0:
            return max(0.0, n.w - max(MIN_W, n.ow * self.p.min_scale))
        return max(0.0, n.h - max(MIN_H, n.oh * self.p.min_scale))

    def _shrink_pair(self, a: Node, b: Node, axis: int, need: float) -> float:
        """Shrink toward each other along `axis`; returns the amount still unresolved."""
        lo, hi = (a, b) if ((a.x + a.w / 2) if axis == 0 else (a.y + a.h / 2)) <= \
                           ((b.x + b.w / 2) if axis == 0 else (b.y + b.h / 2)) else (b, a)
        cl, ch = self._capacity(lo, axis), self._capacity(hi, axis)
        tl, th = min(need / 2, cl), min(need / 2, ch)
        rest = need - tl - th
        if rest > 0:
            extra = min(rest, cl - tl)
            tl, rest = tl + extra, rest - extra
            th += min(rest, ch - th)
        if tl > 0:
            if axis == 0:
                lo.w -= tl
            else:
                lo.h -= tl
            lo.resized = True
        if th > 0:
            if axis == 0:
                hi.x, hi.w = hi.x + th, hi.w - th
            else:
                hi.y, hi.h = hi.y + th, hi.h - th
            hi.resized = True
        return need - tl - th

    def shrink_phase(self) -> None:
        for _ in range(MAX_ITERS):
            progressed = False
            for a, b, _px, _py in self.m.overlap_pairs():
                px, py = penetration(a, b)  # earlier shrinks this pass may already have separated them
                if px <= TOL or py <= TOL:
                    continue
                for axis, pen in sorted(((0, px), (1, py)), key=lambda t: t[1]):
                    need = pen + SHRINK_CLEARANCE
                    if self._capacity(a, axis) + self._capacity(b, axis) >= need:
                        self._shrink_pair(a, b, axis, need)
                        progressed = True
                        break  # fully resolved on this axis; otherwise try the other, else leave it reported
            if not progressed:
                break

    # -- phase 3: fonts -------------------------------------------------------
    def font_phase(self) -> tuple[list[dict], list[dict]]:
        changed, still = [], []
        for n in self.m.nodes.values():
            if not (n.resized or self.p.fix_overflow):
                continue
            style = dd.parse_style(dd.attr(n.cell, "style"))
            fit = text_fit.best_font(dd.attr(n.cell, "value"), style, n.w, n.h, self.p.min_font)
            if fit is None:
                continue
            size, ok = fit
            old = text_fit.current_font(style)
            if size < old:
                style["fontSize"] = str(size)
                dd.set_attr(n.cell, "style", dd.format_style(style))
                changed.append({"id": n.id, "from": old, "to": size})
            if not ok:
                still.append({"id": n.id, "label": _label(n), "font": size, "box": [round(n.w), round(n.h)]})
        return changed, still

    # -- write back -----------------------------------------------------------
    def write_back(self) -> None:
        for n in self.m.nodes.values():
            if (abs(n.x - n.ox) < 0.05 and abs(n.y - n.oy) < 0.05
                    and abs(n.w - n.ow) < 0.05 and abs(n.h - n.oh) < 0.05):
                continue
            px, py = 0.0, 0.0
            if n.parent_attr:
                parent = self.m.nodes[n.parent_attr]
                px, py = parent.x, parent.y
            g = dd.geometry(n.cell)
            if g is None:
                continue
            for key, val in (("x", n.x - px), ("y", n.y - py), ("width", n.w), ("height", n.h)):
                old = float(g.get(key, 0))
                if abs(val - old) > 0.05:
                    g.set(key, _fmt(round(val, 1)))


def resolve_overlaps(doc: dd.Doc, page: int = 0, params: Optional[Params] = None) -> tuple[dd.Doc, dict]:
    """Return (new_doc, report). Never mutates `doc`. Raises ValueError for unknown lock/only ids."""
    p = params or Params()
    work = clone(doc)
    r = _Resolver(work, page, p)
    before = r.m.overlap_pairs()
    if p.allow_grow:
        r.grow_containers()
    if p.allow_move:
        r.move_phase()
    if p.allow_resize and r.m.overlap_pairs():
        r.shrink_phase()
    font_changed, font_still = ([], [])
    if p.allow_font_shrink:
        font_changed, font_still = r.font_phase()
    r.write_back()
    after = Model(work, page).overlap_pairs()

    nodes = r.m.nodes.values()
    moved = [{"id": n.id, "dx": round(n.dx, 1), "dy": round(n.dy, 1)} for n in nodes if n.shift > 0]
    resized = [{"id": n.id, "from": [round(n.ow), round(n.oh)], "to": [round(n.w), round(n.h)]}
               for n in nodes if n.resized]
    grown = [n.id for n in nodes if n.grown]
    cells = work.cells(page)
    free_edges = sum(1 for c in cells if dd.attr(c, "edge") == "1"
                     and (not dd.attr(c, "source") or not dd.attr(c, "target")))
    report = {
        "overlaps_before": len(before),
        "overlaps_after": len(after),
        "moved": moved[:REPORT_CAP],
        "resized": resized[:REPORT_CAP],
        "fonts_changed": font_changed[:REPORT_CAP],
        "containers_grown": grown[:REPORT_CAP],
        "remaining": [{"a": a.id, "b": b.id, "overlap_w": round(px, 1), "overlap_h": round(py, 1)}
                      for a, b, px, py in after][:REPORT_CAP],
        "labels_still_overflowing": font_still[:REPORT_CAP],
    }
    if (moved or grown) and free_edges:
        report["warning"] = (f"{free_edges} edge(s) have a free end (no source/target node) and do not follow "
                             "moved nodes; check them in the preview")
    return work, report
