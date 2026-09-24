"""Deterministic renderer: structured diagram JSON -> draw.io (.drawio / mxGraph) XML.

This is stage 2 of a two-stage image-to-drawio pipeline:
  stage 1 (vision/LLM): image -> structured JSON {nodes, edges}
  stage 2 (this script, no LLM involved): JSON -> mxGraph XML

Keeping stage 2 deterministic avoids the failure mode where an LLM hand-writes
mxGraph XML directly and produces dangling edges / mismatched ids.

Usage:
    python json_to_drawio.py <input.json> <output.drawio>
"""
import json
import sys
import uuid
from xml.etree import ElementTree as ET


def build_vertex_style(node: dict) -> str:
    if node.get("raw_style") is not None:
        return node["raw_style"]

    shape = node.get("shape", "rect")
    fill = node.get("fill")
    stroke = node.get("stroke")
    font_color = node.get("font_color", "#000000")
    font_size = node.get("font_size", 12)
    bold = node.get("bold", False)
    align = node.get("align", "center")
    vertical_text = node.get("vertical_text", False)

    parts = []
    if shape == "rect":
        parts.append("rounded=1" if node.get("rounded", True) else "rounded=0")
        parts.append("whiteSpace=wrap")
        parts.append("html=1")
    elif shape == "ellipse":
        parts.append("ellipse")
        parts.append("whiteSpace=wrap")
        parts.append("html=1")
    elif shape == "text":
        parts.append("text")
        parts.append("html=1")
        parts.append(f"align={align}")
        parts.append("verticalAlign=middle")
        parts.append("whiteSpace=wrap")

    if shape != "text":
        parts.append(f"fillColor={fill}" if fill else "fillColor=none")
        parts.append(f"strokeColor={stroke}" if stroke else "strokeColor=none")
        if align != "center":
            parts.append(f"align={align}")

    parts.append(f"fontColor={font_color}")
    parts.append(f"fontSize={font_size}")
    if bold:
        parts.append("fontStyle=1")
    if vertical_text:
        parts.append("horizontal=0")

    return ";".join(parts) + ";"


def build_edge_style(edge: dict) -> str:
    if edge.get("raw_style") is not None:
        return edge["raw_style"]

    parts = ["html=1", "edgeStyle=orthogonalEdgeStyle" if edge.get("orthogonal", True) else "none"]
    parts.append("endArrow=block" if edge.get("end_arrow", True) else "endArrow=none")
    parts.append("startArrow=block" if edge.get("start_arrow", False) else "startArrow=none")
    if edge.get("dashed"):
        parts.append("dashed=1")
    parts.append(f"strokeColor={edge.get('color', '#000000')}")
    parts.append(f"fontColor={edge.get('color', '#000000')}")
    parts.append(f"fontSize={edge.get('font_size', 10)}")
    parts.append("rounded=0")
    return ";".join(parts) + ";"


def render(diagram: dict) -> str:
    mxfile = ET.Element("mxfile", host="app.diagrams.net")
    diagram_el = ET.SubElement(
        mxfile, "diagram", id=str(uuid.uuid4()), name=diagram.get("name", "Page-1")
    )
    model = ET.SubElement(
        diagram_el,
        "mxGraphModel",
        dx="800", dy="600", grid="1", gridSize="10", guides="1",
        tooltips="1", connect="1", arrows="1", fold="1", page="1",
        pageScale="1", pageWidth="850", pageHeight="420", math="0", shadow="0",
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    node_ids = {n["id"] for n in diagram["nodes"]}

    for node in diagram["nodes"]:
        cell = ET.SubElement(
            root, "mxCell",
            id=node["id"],
            value=node.get("label", ""),
            style=build_vertex_style(node),
            vertex="1",
            parent="1",
        )
        ET.SubElement(
            cell, "mxGeometry",
            x=str(node["x"]), y=str(node["y"]),
            width=str(node["w"]), height=str(node["h"]),
        ).set("as", "geometry")

    for i, edge in enumerate(diagram["edges"]):
        eid = edge.get("id", f"e{i}")
        attrs = {
            "id": eid,
            "value": edge.get("label", ""),
            "style": build_edge_style(edge),
            "edge": "1",
            "parent": "1",
        }
        src = edge.get("source")
        tgt = edge.get("target")
        if src:
            assert src in node_ids, f"edge {eid}: unknown source id {src}"
            attrs["source"] = src
        if tgt:
            assert tgt in node_ids, f"edge {eid}: unknown target id {tgt}"
            attrs["target"] = tgt
        cell = ET.SubElement(root, "mxCell", **attrs)
        geom = ET.SubElement(cell, "mxGeometry", relative="1")
        geom.set("as", "geometry")
        if not src:
            sp = ET.SubElement(geom, "mxPoint", x=str(edge["source_point"][0]), y=str(edge["source_point"][1]))
            sp.set("as", "sourcePoint")
        if not tgt:
            tp = ET.SubElement(geom, "mxPoint", x=str(edge["target_point"][0]), y=str(edge["target_point"][1]))
            tp.set("as", "targetPoint")
        if edge.get("label_offset"):
            dx, dy = edge["label_offset"]
            off = ET.SubElement(geom, "mxPoint", x=str(dx), y=str(dy))
            off.set("as", "offset")

    xml_str = ET.tostring(mxfile, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str


def main():
    if len(sys.argv) != 3:
        print("usage: python json_to_drawio.py <input.json> <output.drawio>")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        diagram = json.load(f)
    xml_str = render(diagram)
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        f.write(xml_str)
    print(f"wrote {sys.argv[2]} ({len(diagram['nodes'])} nodes, {len(diagram['edges'])} edges)")


if __name__ == "__main__":
    main()
