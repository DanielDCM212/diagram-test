"""Deterministic change summary between two draw.io documents.

Computed by code, never by the model: the agent quotes this instead of describing its own work,
so "what changed" is a fact a reviewer can rely on. Cells are matched by id (page by page);
wrapper cells (<object>/<UserObject>) are handled through the same helpers as everywhere else.

Position changes are measured in absolute canvas coordinates. A node that only moved because its
container moved (same offset inside it, or fully inside a zone that moved by the same amount) is
counted as *carried*, not as a separate move, so moving a zone is one change, not twenty.
"""
from __future__ import annotations

from typing import Optional

from . import drawio_doc as dd

POS_TOL = 0.5          # px; smaller differences are rounding noise
LINES_CAP = 30
DETAIL_CAP = 60

_ORDER = ("added", "removed", "renamed", "moved", "resized", "restyled",
          "reparented", "reconnected", "properties", "reordered")


def _snapshot(doc: dd.Doc, page: int) -> tuple[dict[str, dict], list[str]]:
    cells: dict[str, dict] = {}
    order: list[str] = []
    for c in doc.cells(page):
        cid = c.get("id")
        if cid in ("0", "1"):
            continue
        kind = "edge" if dd.attr(c, "edge") == "1" else "node" if dd.attr(c, "vertex") == "1" else "other"
        raw = dd.attr(c, "value") or ""
        cells[cid] = {
            "kind": kind,
            "raw": raw,
            "label": dd.plain_label(raw),
            "style": dd.parse_style(dd.attr(c, "style")),
            "parent": dd.attr(c, "parent"),
            "source": dd.attr(c, "source"),
            "target": dd.attr(c, "target"),
            "props": dd.custom_props(c),
            "abs": dd.absolute_box(doc, c, page) if kind == "node" else None,
            "local": dd.local_box(c) if kind == "node" else None,
        }
        order.append(cid)
    return cells, order


def _reordered(before_order: list[str], after_order: list[str]) -> list[str]:
    """Cells whose draw order changed relative to the others.

    Repeatedly picks the cell that moved furthest (ties: earlier in the new order), takes it out of
    both orders and repeats until the rest agree -- so sending one box to the back reports that box,
    not the many cells it jumped over.
    """
    a, b, out = list(before_order), list(after_order), []
    while a != b and len(out) < len(before_order):
        pos_a = {i: n for n, i in enumerate(a)}
        worst = max(b, key=lambda i: (abs(pos_a[i] - b.index(i)), -b.index(i)))
        out.append(worst)
        a.remove(worst)
        b.remove(worst)
    return out


def _num(v: float) -> str:
    v = round(v, 1)
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _signed(v: float) -> str:
    return ("+" if round(v, 1) > 0 else "") + _num(v)


def _contains(outer: tuple, inner: tuple, tol: float = 1.0) -> bool:
    return (outer[0] - tol <= inner[0] and outer[1] - tol <= inner[1]
            and inner[0] + inner[2] <= outer[0] + outer[2] + tol
            and inner[1] + inner[3] <= outer[1] + outer[3] + tol
            and outer[2] * outer[3] > inner[2] * inner[3])


def _diff_page(before: dict, border: list, after: dict, aorder: list, page_label: Optional[str]) -> dict:
    tag = (lambda i: f"{page_label}:{i}") if page_label else (lambda i: i)  # noqa: E731
    d: dict[str, list] = {k: [] for k in _ORDER}
    d["carried"] = []
    lines: list[str] = []
    common = [i for i in border if i in after]

    for i in aorder:
        if i not in before:
            c = after[i]
            d["added"].append({"id": tag(i), "kind": c["kind"], "label": c["label"]})
            lines.append(f'added {c["kind"]} {tag(i)}' + (f' "{c["label"]}"' if c["label"] else "")
                         + (f' ({c["source"]} -> {c["target"]})' if c["kind"] == "edge" else ""))
    for i in border:
        if i not in after:
            c = before[i]
            d["removed"].append({"id": tag(i), "kind": c["kind"], "label": c["label"]})
            lines.append(f'removed {c["kind"]} {tag(i)}' + (f' "{c["label"]}"' if c["label"] else ""))

    changed: set[str] = set()
    deltas: dict[str, tuple[float, float]] = {}
    for i in common:
        b, a = before[i], after[i]
        if b["label"] != a["label"]:
            d["renamed"].append({"id": tag(i), "from": b["label"], "to": a["label"]})
            lines.append(f'renamed {tag(i)}: "{b["label"]}" -> "{a["label"]}"')
            changed.add(i)

        style_changes = {}
        for k in sorted(set(b["style"]) | set(a["style"])):
            if b["style"].get(k) != a["style"].get(k):
                style_changes[k] = [b["style"].get(k), a["style"].get(k)]
        if b["raw"] != a["raw"] and b["label"] == a["label"]:
            style_changes["label_formatting"] = [None, None]  # same words, different markup
        if style_changes:
            d["restyled"].append({"id": tag(i), "changes": dict(list(style_changes.items())[:12])})
            shown = ", ".join(f"{k} {old} -> {new}" if k != "label_formatting" else k
                              for k, (old, new) in list(style_changes.items())[:4])
            lines.append(f"restyled {tag(i)}: {shown}" + (" ..." if len(style_changes) > 4 else ""))
            changed.add(i)

        if b["parent"] != a["parent"]:
            d["reparented"].append({"id": tag(i), "from": b["parent"], "to": a["parent"]})
            lines.append(f'moved {tag(i)} from container {b["parent"]} to {a["parent"]}')
            changed.add(i)
        if (b["source"], b["target"]) != (a["source"], a["target"]):
            d["reconnected"].append({"id": tag(i), "from": [b["source"], b["target"]], "to": [a["source"], a["target"]]})
            lines.append(f'reconnected {tag(i)}: {b["source"]} -> {b["target"]} became {a["source"]} -> {a["target"]}')
            changed.add(i)
        if b["props"] != a["props"]:
            keys = sorted(k for k in set(b["props"]) | set(a["props"]) if b["props"].get(k) != a["props"].get(k))
            d["properties"].append({"id": tag(i), "keys": keys})
            lines.append(f'changed properties of {tag(i)}: {", ".join(keys)}')
            changed.add(i)

        if b["abs"] and a["abs"]:
            (bx, by, bw, bh), (ax, ay, aw, ah) = b["abs"], a["abs"]
            if abs(aw - bw) > POS_TOL or abs(ah - bh) > POS_TOL:
                d["resized"].append({"id": tag(i), "from": [round(bw, 1), round(bh, 1)], "to": [round(aw, 1), round(ah, 1)]})
                lines.append(f"resized {tag(i)}: {_num(bw)}x{_num(bh)} -> {_num(aw)}x{_num(ah)}")
                changed.add(i)
            if abs(ax - bx) > POS_TOL or abs(ay - by) > POS_TOL:
                deltas[i] = (ax - bx, ay - by)

    # moved vs carried along by a container that moved
    boxes_before = {i: before[i]["abs"] for i in common if before[i]["abs"]}
    for i, (dx, dy) in deltas.items():
        carried = False
        c = after[i]
        parent = c["parent"]
        if parent in deltas and before[i]["local"] and c["local"] \
                and abs(before[i]["local"][0] - c["local"][0]) <= POS_TOL \
                and abs(before[i]["local"][1] - c["local"][1]) <= POS_TOL and before[i]["parent"] == c["parent"]:
            carried = True
        elif c["parent"] in (None, "1") or c["parent"] not in deltas:
            # flat diagrams: a zone is just a bigger box around it; same offset as that zone => carried
            for j, (jx, jy) in deltas.items():
                if j != i and boxes_before.get(j) and _contains(boxes_before[j], boxes_before[i]) \
                        and abs(jx - dx) <= POS_TOL and abs(jy - dy) <= POS_TOL:
                    carried = True
                    break
        if carried:
            d["carried"].append({"id": tag(i)})
        else:
            d["moved"].append({"id": tag(i), "dx": round(dx, 1), "dy": round(dy, 1)})
            lines.append(f"moved {tag(i)} by ({_signed(dx)}, {_signed(dy)})")
            changed.add(i)
    carried_ids = {x["id"] for x in d["carried"]}

    # draw order
    for i in _reordered(common, [i for i in aorder if i in before]):
        d["reordered"].append({"id": tag(i)})
        lines.append(f"changed draw order of {tag(i)}")
        changed.add(i)

    d["unchanged"] = sum(1 for i in common if i not in changed and tag(i) not in carried_ids)
    d["lines"] = lines
    return d


def diff_docs(before: Optional[dd.Doc], after: dd.Doc) -> dict:
    """Compare `before` (None = an empty diagram) with `after`.

    Returns {"summary": str, "counts": {...}, "lines": [...], "details": {...}, "since": "input"|"empty"}.
    """
    since = "empty" if before is None else "input"
    pages_b = len(before.pages) if before else 0
    pages_a = len(after.pages)
    multi = max(pages_a, pages_b) > 1
    merged: dict[str, list] = {k: [] for k in _ORDER}
    merged["carried"] = []
    unchanged, lines = 0, []
    for p in range(max(pages_a, pages_b)):
        pname = (after.pages[p].get("name") if p < pages_a else before.pages[p].get("name")) or f"page {p + 1}"
        b_cells, b_order = _snapshot(before, p) if before is not None and p < pages_b else ({}, [])
        a_cells, a_order = _snapshot(after, p) if p < pages_a else ({}, [])
        part = _diff_page(b_cells, b_order, a_cells, a_order, pname if multi else None)
        for k in merged:
            merged[k].extend(part[k])
        unchanged += part["unchanged"]
        lines.extend(part["lines"])

    counts = {k: len(merged[k]) for k in _ORDER}
    counts["added_nodes"] = sum(1 for x in merged["added"] if x["kind"] == "node")
    counts["added_edges"] = sum(1 for x in merged["added"] if x["kind"] == "edge")
    counts["removed_nodes"] = sum(1 for x in merged["removed"] if x["kind"] == "node")
    counts["removed_edges"] = sum(1 for x in merged["removed"] if x["kind"] == "edge")
    counts["carried"] = len(merged["carried"])
    counts["unchanged"] = unchanged

    if since == "empty":
        summary = f'New diagram: {_plural(counts["added_nodes"], "node")}, {_plural(counts["added_edges"], "edge")}.'
    else:
        parts = []
        for k in _ORDER:
            n = counts[k]
            if not n:
                continue
            text = f"{k} {n}"
            if k in ("added", "removed"):
                text += f' ({_plural(counts[k + "_nodes"], "node")}, {_plural(counts[k + "_edges"], "edge")})'
            if k == "moved" and counts["carried"]:
                text += f' (+{counts["carried"]} carried along with their container)'
            parts.append(text)
        if not parts and counts["carried"]:
            parts.append(f'{counts["carried"]} carried along with a moved container')
        summary = ("; ".join(parts) if parts else "No changes") + f". {unchanged} cells unchanged."
        summary = summary[0].upper() + summary[1:]

    shown = lines[:LINES_CAP]
    if len(lines) > LINES_CAP:
        shown.append(f"... and {len(lines) - LINES_CAP} more")
    details = {k: v[:DETAIL_CAP] for k, v in merged.items()}
    return {"since": since, "summary": summary, "counts": counts, "lines": shown, "details": details,
            "truncated": len(lines) > LINES_CAP or any(len(v) > DETAIL_CAP for v in merged.values())}


def compact(result: dict) -> dict:
    """The small form returned from each editing tool call: summary + non-zero counts."""
    return {"summary": result["summary"], "counts": {k: v for k, v in result["counts"].items() if v}}
