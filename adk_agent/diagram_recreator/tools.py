"""Tools for the diagram-recreation agent.

These mirror the manual steps used to build diagrams/*.drawio by hand:
  1. measure exact pixel colors / bounding boxes instead of eyeballing
     (sample_color_at, find_bbox_of_color, get_image_size)
  2. save the extracted structure as schema-validated JSON (save_diagram)
  3. render it with the existing deterministic renderer, no LLM involved
     (render_drawio)
  4. validate the resulting XML has no dangling edge references
     (validate_drawio)
  5. render a rough layout preview to catch gross mistakes before finishing
     (sanity_plot)
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = PROJECT_ROOT / "images"
DIAGRAMS_DIR = PROJECT_ROOT / "diagrams"
DIAGRAMS_DIR.mkdir(exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.json_to_drawio import render as render_mxgraph_xml  # noqa: E402

from .schema import Diagram  # noqa: E402


def _resolve_image(image_path: str) -> Path:
    p = Path(image_path)
    if not p.is_absolute():
        candidate = PROJECT_ROOT / image_path
        if candidate.exists():
            return candidate
        candidate = IMAGES_DIR / Path(image_path).name
        if candidate.exists():
            return candidate
    return p


def get_image_size(image_path: str) -> dict:
    """Return the pixel width/height of a source image.

    Args:
        image_path: Path to the image, relative to the project root or just the filename in images/.
    """
    from PIL import Image

    path = _resolve_image(image_path)
    if not path.exists():
        return {"error": f"image not found: {image_path}"}
    with Image.open(path) as im:
        return {"width": im.width, "height": im.height}


def sample_color_at(image_path: str, x: int, y: int) -> dict:
    """Sample the exact pixel color at (x, y) in the source image.

    Use this instead of guessing a fill/stroke color by eye — pick a point
    well inside the shape you're measuring, away from its border/text.

    Args:
        image_path: Path to the image, relative to the project root or just the filename in images/.
        x: Pixel x-coordinate.
        y: Pixel y-coordinate.
    """
    from PIL import Image

    path = _resolve_image(image_path)
    if not path.exists():
        return {"error": f"image not found: {image_path}"}
    with Image.open(path) as im:
        im = im.convert("RGB")
        if not (0 <= x < im.width and 0 <= y < im.height):
            return {"error": f"({x},{y}) is outside the image ({im.width}x{im.height})"}
        r, g, b = im.getpixel((x, y))
        return {"hex": f"#{r:02X}{g:02X}{b:02X}"}


def find_bbox_of_color(
    image_path: str,
    hex_color: str,
    region: Optional[list[int]] = None,
    tolerance: int = 12,
) -> dict:
    """Find the tight bounding box of a fill color, to get exact shape geometry instead of eyeballing it.

    Scans pixels matching hex_color (within `tolerance` per channel) and returns
    the bounding box of all matches. Restrict `region` to roughly where you
    expect the shape to be — otherwise anti-aliased edges of unrelated same-ish
    colored elements elsewhere in the image can pollute the result; if the
    returned box looks too big/sparse (low pixel_count relative to its area),
    narrow `region` and retry.

    Args:
        image_path: Path to the image, relative to the project root or just the filename in images/.
        hex_color: Color to search for, e.g. "#248D45".
        region: Optional [x0, y0, x1, y1] to restrict the search to.
        tolerance: Max per-channel difference to still count as a match.
    """
    from PIL import Image

    path = _resolve_image(image_path)
    if not path.exists():
        return {"error": f"image not found: {image_path}"}
    with Image.open(path) as im:
        im = im.convert("RGB")
        x0, y0, x1, y1 = region if region else (0, 0, im.width, im.height)
        crop = im.crop((x0, y0, x1, y1))
        target = tuple(int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

        w, h = crop.size
        pixels = crop.load()
        min_x, min_y, max_x, max_y, count = w, h, -1, -1, 0
        for yy in range(h):
            for xx in range(w):
                px = pixels[xx, yy]
                if all(abs(px[i] - target[i]) <= tolerance for i in range(3)):
                    count += 1
                    if xx < min_x:
                        min_x = xx
                    if xx > max_x:
                        max_x = xx
                    if yy < min_y:
                        min_y = yy
                    if yy > max_y:
                        max_y = yy
        if count == 0:
            return {"found": False}
        return {
            "found": True,
            "x": x0 + min_x,
            "y": y0 + min_y,
            "w": max_x - min_x + 1,
            "h": max_y - min_y + 1,
            "pixel_count": count,
        }


def save_diagram(diagram: Diagram, output_name: str) -> dict:
    """Validate and save the extracted diagram structure as JSON.

    Call this once you've identified every node and edge. It checks that
    every edge's source/target refers to a node id that actually exists
    (beyond the basic schema validation ADK already applies) before writing
    the file, so mistakes surface here rather than as a broken .drawio later.

    Args:
        diagram: The full extracted diagram (nodes + edges).
        output_name: Base filename to save as, e.g. "img_7" -> diagrams/img_7.diagram.json.
    """
    node_ids = {n.id for n in diagram.nodes}
    errors = []
    for e in diagram.edges:
        if e.source and e.source not in node_ids:
            errors.append(f"edge {e.id!r}: unknown source id {e.source!r}")
        if e.target and e.target not in node_ids:
            errors.append(f"edge {e.id!r}: unknown target id {e.target!r}")
        if not e.source and not e.source_point:
            errors.append(f"edge {e.id!r}: needs either source or source_point")
        if not e.target and not e.target_point:
            errors.append(f"edge {e.id!r}: needs either target or target_point")
    if errors:
        return {"saved": False, "errors": errors}

    out_path = DIAGRAMS_DIR / f"{output_name}.diagram.json"
    out_path.write_text(json.dumps(diagram.model_dump(exclude_none=True), indent=2), encoding="utf-8")
    return {"saved": True, "path": str(out_path), "nodes": len(diagram.nodes), "edges": len(diagram.edges)}


def render_drawio(output_name: str) -> dict:
    """Render a previously saved diagram JSON into a .drawio (mxGraph XML) file.

    This step is plain deterministic code, not a model guess — it cannot by
    itself introduce a dangling reference or malformed XML as long as the
    JSON passed schema validation in save_diagram.

    Args:
        output_name: Base filename used in save_diagram, e.g. "img_7".
    """
    json_path = DIAGRAMS_DIR / f"{output_name}.diagram.json"
    if not json_path.exists():
        return {"error": f"{json_path} does not exist — call save_diagram first"}
    diagram = json.loads(json_path.read_text(encoding="utf-8"))
    xml_str = render_mxgraph_xml(diagram)
    out_path = DIAGRAMS_DIR / f"{output_name}.drawio"
    out_path.write_text(xml_str, encoding="utf-8")
    return {"path": str(out_path), "nodes": len(diagram["nodes"]), "edges": len(diagram["edges"])}


def validate_drawio(output_name: str) -> dict:
    """Parse the rendered .drawio XML and confirm every edge's source/target id resolves to a real node.

    Always call this after render_drawio. If it reports dangling ids or a
    parse error, fix diagrams/<output_name>.diagram.json and call
    save_diagram/render_drawio again — don't hand-patch the XML.

    Args:
        output_name: Base filename used in render_drawio, e.g. "img_7".
    """
    path = DIAGRAMS_DIR / f"{output_name}.drawio"
    if not path.exists():
        return {"error": f"{path} does not exist — call render_drawio first"}
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        return {"valid": False, "parse_error": str(e)}

    cells = tree.getroot().findall(".//mxCell")
    ids = {c.get("id") for c in cells}
    dangling = []
    for c in cells:
        for attr in ("source", "target"):
            v = c.get(attr)
            if v and v not in ids:
                dangling.append({"cell": c.get("id"), "attr": attr, "missing_id": v})
    return {"valid": not dangling, "cell_count": len(cells), "dangling": dangling}


def sanity_plot(output_name: str) -> dict:
    """Render a rough, schematic PNG preview of the saved diagram JSON for a final visual sanity check.

    This is a quick matplotlib plot of node/edge positions — colors and
    rough placement only, not final draw.io fidelity (edges are drawn
    straight, not orthogonally routed). Use it to catch gross mistakes:
    overlapping boxes, a node in the wrong region, a missing node — compare
    it mentally against what you read from the source image.

    Args:
        output_name: Base filename used in save_diagram, e.g. "img_7".
    """
    json_path = DIAGRAMS_DIR / f"{output_name}.diagram.json"
    if not json_path.exists():
        return {"error": f"{json_path} does not exist — call save_diagram first"}
    diagram = json.loads(json_path.read_text(encoding="utf-8"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    nodes = {n["id"]: n for n in diagram["nodes"]}
    max_x = max((n["x"] + n["w"] for n in diagram["nodes"]), default=800)
    max_y = max((n["y"] + n["h"] for n in diagram["nodes"]), default=400)

    fig, ax = plt.subplots(figsize=(max_x / 100, max_y / 100))
    for n in diagram["nodes"]:
        x, y, w, h = n["x"], n["y"], n["w"], n["h"]
        fill = n.get("fill") or "none"
        if n.get("shape") == "ellipse":
            ax.add_patch(patches.Ellipse((x + w / 2, y + h / 2), w, h, facecolor=fill, edgecolor=n.get("stroke") or "black"))
        elif n.get("shape") == "text" or n.get("raw_style"):
            pass
        else:
            ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=fill, edgecolor=n.get("stroke") or "none"))
        if n.get("label"):
            ax.text(x + w / 2, y + h / 2, n["label"], ha="center", va="center", fontsize=6, color=n.get("font_color", "#000"))

    def center(n):
        return (n["x"] + n["w"] / 2, n["y"] + n["h"] / 2)

    for e in diagram["edges"]:
        if e.get("source") and e.get("target") and e["source"] in nodes and e["target"] in nodes:
            sx, sy = center(nodes[e["source"]])
            tx, ty = center(nodes[e["target"]])
        elif e.get("source_point") and e.get("target_point"):
            sx, sy = e["source_point"]
            tx, ty = e["target_point"]
        else:
            continue
        ax.annotate("", xy=(tx, ty), xytext=(sx, sy), arrowprops=dict(arrowstyle="->", linestyle="--" if e.get("dashed") else "-", color=e.get("color", "#000"), lw=0.7))
        if e.get("label"):
            ax.text((sx + tx) / 2, (sy + ty) / 2, e["label"], fontsize=5, color=e.get("color", "#000"))

    ax.set_xlim(-10, max_x + 10)
    ax.set_ylim(max_y + 10, -10)
    ax.axis("off")
    plt.tight_layout()
    out_path = DIAGRAMS_DIR / f"{output_name}.sanity.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"path": str(out_path)}
