"""Pure-python helpers for reading, summarising and serialising draw.io (mxGraph) XML.

No ADK imports here, so this module is unit-testable on its own and reusable
by the offline style-profile builder (scripts/build_style_profile.py).

Editing works on the raw XML cells rather than converting to the flat
`diagram_recreator.schema.Diagram` model, because that schema has no notion of
parents/groups/waypoints/images and a round trip would silently drop parts of
a user's diagram. Anything this module does not understand is left untouched.
"""
from __future__ import annotations

import base64
import re
import urllib.parse
import zlib
from dataclasses import dataclass, field
from html import unescape
from typing import Optional
from xml.etree import ElementTree as ET

from defusedxml import ElementTree as SafeET  # user-supplied XML: blocks entity-expansion attacks
from defusedxml.common import DefusedXmlException


class DrawioParseError(ValueError):
    """Raised when the input is not recognisable draw.io XML."""


_TAG_RE = re.compile(r"<[^>]+>")
_BR_RE = re.compile(r"<br\s*/?>|</div>|</p>", re.IGNORECASE)


def plain_label(value: Optional[str]) -> str:
    """Strip HTML from a cell value, keeping line breaks as spaces."""
    if not value:
        return ""
    text = _BR_RE.sub(" ", value)
    text = _TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def parse_style(style: Optional[str]) -> dict[str, str]:
    """'a=1;b=2;rounded' -> {'a': '1', 'b': '2', 'rounded': ''} (order preserved)."""
    out: dict[str, str] = {}
    for part in (style or "").split(";"):
        if not part:
            continue
        key, sep, value = part.partition("=")
        out[key] = value if sep else ""
    return out


# draw.io wraps a cell in <object>/<UserObject> when it carries custom data, a link, a tooltip or
# placeholders. The wrapper owns `id` and the label (`label`, not `value`) plus every custom
# property; the inner <mxCell> keeps style/parent/source/target/vertex/edge and the geometry and
# has no id of its own. Everywhere in this package a "cell" is the LOGICAL element -- the wrapper
# if there is one, else the mxCell -- and these helpers hide which element holds what.
CELL_TAGS = ("mxCell", "object", "UserObject")
INNER_ATTRS = frozenset({"style", "parent", "source", "target", "vertex", "edge", "connectable", "collapsed", "visible"})
_WRAPPER_RESERVED = frozenset({"id", "label", "placeholders"})


def inner(cell: ET.Element) -> Optional[ET.Element]:
    """The element that holds style/parent/geometry: the cell itself, or a wrapper's inner mxCell."""
    return cell if cell.tag == "mxCell" else cell.find("mxCell")


def attr(cell: ET.Element, name: str, default: Optional[str] = None) -> Optional[str]:
    if cell.tag == "mxCell" or name == "id":
        return cell.get(name, default)
    if name == "value":
        return cell.get("label", cell.get("value", default))
    if name in INNER_ATTRS:
        node = inner(cell)
        return node.get(name, default) if node is not None else default
    return cell.get(name, default)


def set_attr(cell: ET.Element, name: str, value: Optional[str]) -> None:
    """Set (or, with None, remove) an attribute on whichever element owns it."""
    if cell.tag == "mxCell" or name == "id":
        target = cell
    elif name == "value":
        target, name = cell, "label"
    elif name in INNER_ATTRS:
        target = inner(cell)
        if target is None:
            raise ValueError(f"wrapper {cell.get('id')!r} has no inner <mxCell>")
    else:
        target = cell
    if value is None:
        target.attrib.pop(name, None)
    else:
        target.set(name, value)


def custom_props(cell: ET.Element) -> dict[str, str]:
    """Custom properties of a wrapped cell (everything on the wrapper except id/label/placeholders)."""
    if cell.tag == "mxCell":
        return {}
    return {k: v for k, v in cell.attrib.items() if k not in _WRAPPER_RESERVED}


def format_style(style: dict[str, str]) -> str:
    parts = [f"{k}={v}" if v != "" else k for k, v in style.items()]
    return ";".join(parts) + (";" if parts else "")


def _inflate_diagram_text(text: str) -> ET.Element:
    """Decode draw.io's compressed <diagram> payload (base64 -> raw deflate -> url-encoded XML)."""
    try:
        raw = zlib.decompress(base64.b64decode(text.strip()), -15)
        return SafeET.fromstring(urllib.parse.unquote(raw.decode("utf-8")))
    except Exception as exc:  # noqa: BLE001 - any failure here means "not a valid payload"
        raise DrawioParseError(f"could not decompress <diagram> content: {exc}") from exc


def compress_model(model: ET.Element) -> str:
    """Inverse of _inflate_diagram_text; used by tests to build compressed fixtures."""
    xml = urllib.parse.quote(ET.tostring(model, encoding="unicode"), safe="")
    comp = zlib.compressobj(9, zlib.DEFLATED, -15)
    return base64.b64encode(comp.compress(xml.encode("utf-8")) + comp.flush()).decode("ascii")


@dataclass
class Doc:
    """A parsed draw.io file. `mxfile` always holds fully expanded (uncompressed) pages."""

    mxfile: ET.Element
    pages: list[ET.Element] = field(default_factory=list)  # the <diagram> elements

    def model(self, page: int = 0) -> ET.Element:
        return self.pages[page].find("mxGraphModel")

    def root(self, page: int = 0) -> ET.Element:
        return self.model(page).find("root")

    def cells(self, page: int = 0) -> list[ET.Element]:
        return [e for e in self.root(page) if e.tag in CELL_TAGS]

    def cell(self, cell_id: str, page: int = 0) -> Optional[ET.Element]:
        for c in self.cells(page):
            if c.get("id") == cell_id:
                return c
        return None


def load(xml: str) -> Doc:
    """Parse an <mxfile>, a bare <mxGraphModel>, or a compressed-<diagram> file.

    Raises DrawioParseError with a readable message on anything else.
    """
    if not xml or not xml.strip():
        raise DrawioParseError("empty XML")
    try:
        root = SafeET.fromstring(xml.strip())
    except (ET.ParseError, DefusedXmlException) as exc:
        raise DrawioParseError(f"not well-formed or unsafe XML: {exc}") from exc

    if root.tag == "mxGraphModel":
        mxfile = ET.Element("mxfile", host="app.diagrams.net")
        diagram = ET.SubElement(mxfile, "diagram", id="page-1", name="Page-1")
        diagram.append(root)
        return Doc(mxfile, [diagram])

    if root.tag != "mxfile":
        raise DrawioParseError(f"expected <mxfile> or <mxGraphModel>, got <{root.tag}>")

    pages = root.findall("diagram")
    if not pages:
        raise DrawioParseError("<mxfile> has no <diagram> pages")
    for page in pages:
        if page.find("mxGraphModel") is None:
            text = (page.text or "").strip()
            if not text:
                raise DrawioParseError(f"page {page.get('name')!r} is empty")
            page.text = None
            page.append(_inflate_diagram_text(text))
        if page.find("mxGraphModel/root") is None:
            raise DrawioParseError(f"page {page.get('name')!r} has no <root>")
    return Doc(root, pages)


def serialize(doc: Doc) -> str:
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(doc.mxfile, encoding="unicode")


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def geometry(cell: ET.Element) -> Optional[ET.Element]:
    node = inner(cell)
    return node.find("mxGeometry") if node is not None else None


def local_box(cell: ET.Element) -> Optional[tuple[float, float, float, float]]:
    """(x, y, w, h) relative to the cell's parent, or None (edges / no geometry)."""
    g = geometry(cell)
    if g is None or attr(cell, "vertex") != "1":
        return None
    return (
        float(g.get("x", 0)), float(g.get("y", 0)),
        float(g.get("width", 0)), float(g.get("height", 0)),
    )


def absolute_box(doc: Doc, cell: ET.Element, page: int = 0) -> Optional[tuple[float, float, float, float]]:
    """(x, y, w, h) in canvas coordinates: local offsets summed through every ancestor."""
    box = local_box(cell)
    if box is None:
        return None
    x, y, w, h = box
    by_id = {c.get("id"): c for c in doc.cells(page)}
    seen = {cell.get("id")}
    parent = by_id.get(attr(cell, "parent"))
    while parent is not None and parent.get("id") not in seen:
        seen.add(parent.get("id"))
        pbox = local_box(parent)
        if pbox is not None:
            x += pbox[0]
            y += pbox[1]
        parent = by_id.get(attr(parent, "parent"))
    return (x, y, w, h)


# --------------------------------------------------------------------------
# Summary (what the LLM reads instead of raw XML)
# --------------------------------------------------------------------------

def summarize(doc: Doc, page: int = 0) -> dict:
    """Compact, LLM-friendly view of one page.

    Vertices carry absolute x/y/w/h, parent (container) id and a short style
    digest; edges carry source/target ids and label. Structural cells 0/1 are
    omitted. Containers are cells that other cells name as their parent.
    """
    cells = doc.cells(page)
    parent_ids = {attr(c, "parent") for c in cells}
    vertices, edges = [], []
    for c in cells:
        cid = c.get("id")
        if cid in ("0", "1"):
            continue
        style = parse_style(attr(c, "style"))
        props = custom_props(c)
        if attr(c, "edge") == "1":
            e = {
                "id": cid,
                "source": attr(c, "source"),
                "target": attr(c, "target"),
                "label": plain_label(attr(c, "value")),
                "dashed": style.get("dashed") == "1",
            }
            if props:
                e["props"] = props
            edges.append(e)
        elif attr(c, "vertex") == "1":
            box = absolute_box(doc, c, page)
            v = {
                "id": cid,
                "label": plain_label(attr(c, "value")),
                "parent": attr(c, "parent") if attr(c, "parent") != "1" else None,
                "container": cid in parent_ids,
                "x": round(box[0], 1), "y": round(box[1], 1),
                "w": round(box[2], 1), "h": round(box[3], 1),
                "style": {k: style[k] for k in ("shape", "fillColor", "strokeColor", "fontSize")
                          if k in style},
            }
            vertices.append({k: val for k, val in v.items() if val not in (None, "")})
            if "label" not in v:
                vertices[-1]["label"] = ""
            if props:  # custom data attached via draw.io's "Edit Data" (owner, link, tooltip, ...)
                vertices[-1]["props"] = props
    return {
        "page": doc.pages[page].get("name"),
        "page_count": len(doc.pages),
        "vertices": vertices,
        "edges": edges,
    }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(doc: Doc) -> dict:
    """Dangling source/target/parent references, duplicate ids, missing roots, per page."""
    problems = []
    cell_count = 0
    for i in range(len(doc.pages)):
        cells = doc.cells(i)
        cell_count += len(cells)
        ids = [c.get("id") for c in cells]
        id_set = set(ids)
        for dup in {x for x in ids if ids.count(x) > 1}:
            problems.append({"page": i, "cell": dup, "problem": "duplicate id"})
        for c in cells:
            if c.tag != "mxCell" and inner(c) is None:
                problems.append({"page": i, "cell": c.get("id"), "problem": f"<{c.tag}> has no inner <mxCell>"})
        for c in cells:
            for name in ("source", "target", "parent"):
                v = attr(c, name)
                if v and v not in id_set:
                    problems.append({"page": i, "cell": c.get("id"), "problem": f"dangling {name}", "missing_id": v})
        parent_of = {c.get("id"): attr(c, "parent") for c in cells}
        for cid in parent_of:
            seen, cur = {cid}, parent_of.get(cid)
            while cur in parent_of:
                if cur in seen:
                    problems.append({"page": i, "cell": cid, "problem": "parent cycle"})
                    break
                seen.add(cur)
                cur = parent_of[cur]
    return {"valid": not problems, "cell_count": cell_count, "problems": problems}
