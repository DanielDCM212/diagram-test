"""Tools for the diagram-recreation agent.

These mirror the manual steps used to build diagrams/*.drawio by hand:
  1. zoom into dense/small regions and actually look before transcribing text
     (inspect_region)
  2. measure exact pixel colors / bounding boxes instead of eyeballing
     (sample_color_at, find_bbox_of_color, get_image_size)
  3. save the extracted structure as schema-validated JSON (save_diagram)
  4. render it with the existing deterministic renderer, no LLM involved
     (render_drawio)
  5. validate the resulting XML has no dangling edge references
     (validate_drawio)
  6. render a rough layout preview and actually look at it to catch gross
     mistakes before finishing (sanity_plot)

inspect_region and sanity_plot return a `google.genai.types.Part` holding
image bytes alongside their dict payload. ADK's tool-result handling detects
any `types.Part` value in a returned dict/list and pulls it out as a separate
multimodal part of the FunctionResponse (see
flows/llm_flows/_tool_caller.py::_extract_multimodal_parts in the installed
google-adk package) — and for Claude specifically (models/anthropic_llm.py::
_function_response_media_blocks) that part is converted into a real
ImageBlockParam inside the tool_result content, which Claude actually sees on
the next turn. This is what makes a real zoom-and-reread loop possible here,
not just a numeric measurement — confirmed against the installed ADK source,
not assumed.
"""
import base64
import io
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from google.adk.tools.tool_context import ToolContext

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.json_to_drawio import render as render_mxgraph_xml  # noqa: E402

from .schema import Diagram  # noqa: E402


# Claude 4.7+ models (including the claude-sonnet-5 this agent uses) are in
# Anthropic's "high-resolution" tier: max long edge 2576px, max 4784 visual
# tokens -- not the 1568px "standard tier" value older code/docs assume.
# 2576px is also the exact per-image ceiling Anthropic enforces once a
# request carries multiple images (confirmed by the literal "2576 pixels" in
# the many-image-request 400 this constant exists to prevent), so clamping
# every image handed to the model to this edge stays safely under every
# applicable limit while preserving as much detail as the model can use.
# Relying on the model to self-limit is exactly what caused that 400.
MAX_IMAGE_EDGE = 2576


def _clamp_image_bytes(png_bytes: bytes, max_edge: int = MAX_IMAGE_EDGE) -> bytes:
    """Downscale PNG bytes so neither dimension exceeds max_edge, preserving aspect ratio."""
    from PIL import Image

    with Image.open(io.BytesIO(png_bytes)) as im:
        if im.width <= max_edge and im.height <= max_edge:
            return png_bytes
        ratio = min(max_edge / im.width, max_edge / im.height)
        new_size = (max(1, round(im.width * ratio)), max(1, round(im.height * ratio)))
        im = im.convert("RGB").resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_SOURCE_IMAGE_B64_KEY = "_source_image_b64"
_SOURCE_IMAGE_MIME_KEY = "_source_image_mime"

# Working artifacts are kept in ADK session state rather than written to a
# local folder, so that whatever backs the session (SQLite by default under
# `adk web`/`adk api_server`, Postgres once the session service is pointed
# there) is the single place this data lives -- no separate local-disk save
# step, and it survives a server restart for any session whose tool calls
# already completed. A later persistence step (writing the finished diagram
# to Postgres proper, alongside the `diagram_input` metadata a caller passes
# via state_delta) reads these same keys off the session; it does not exist
# yet and is intentionally out of scope here.
_DIAGRAM_JSON_KEY = "diagram_json"
_DRAWIO_XML_KEY = "drawio_xml"


def clamp_request_images(callback_context, llm_request):
    """ADK before_model_callback: capture the session's source image, downscale oversized inline images.

    Two jobs, both needing the one choke point every request passes through
    regardless of entry point (CLI attachment via run_pipeline.py, or a real
    upload under `adk web`):

    1. Capture the source image. The pixel-measurement tools (get_image_size,
       inspect_region, sample_color_at, find_bbox_of_color) used to read the
       image from a local images/ folder by filename -- which only worked if
       whatever the user actually attached in the conversation happened to
       already exist on disk under a matching name. They now read from
       session state instead, populated here from the first inline image
       seen in the whole conversation (which is always the original
       attachment, since any later inline image only shows up from a tool
       result on a subsequent turn). Captured once, before any clamping
       below, so the copy tools measure against stays at full native
       resolution regardless of what gets downscaled for the API request.
       Stored as base64 text (not raw bytes) since session state can be
       JSON-persisted (e.g. `adk web`'s SQLite-backed session store), which
       raw bytes aren't serializable for.
    2. Downscale any oversized inline image actually going to the model.
       Every image these tools *return* (inspect_region, sanity_plot) already
       goes through _clamp_image_bytes, but the source image itself is handed
       in from outside this module, so it can slip through unclamped. A
       source image over Anthropic's per-image limit works fine alone, but
       once a second image (e.g. from inspect_region) joins the request, the
       stricter multi-image size cap applies and the request is rejected with
       a 400.
    """
    captured = _SOURCE_IMAGE_B64_KEY in callback_context.state
    for content in llm_request.contents or []:
        for part in content.parts or []:
            inline = getattr(part, "inline_data", None)
            if not inline or not inline.data or not (inline.mime_type or "").startswith("image/"):
                continue
            if not captured:
                callback_context.state[_SOURCE_IMAGE_B64_KEY] = base64.b64encode(inline.data).decode("ascii")
                callback_context.state[_SOURCE_IMAGE_MIME_KEY] = inline.mime_type
                captured = True
            clamped = _clamp_image_bytes(inline.data)
            if clamped is not inline.data:
                inline.data = clamped
                if clamped[:8] == _PNG_MAGIC:
                    inline.mime_type = "image/png"
    return None


def _get_source_image_bytes(tool_context: ToolContext) -> Optional[bytes]:
    b64 = tool_context.state.get(_SOURCE_IMAGE_B64_KEY)
    if not b64:
        return None
    return base64.b64decode(b64)


def _to_int(value) -> int:
    """Coerce a possibly-stringified numeric tool argument to int.

    Claude's tool calls occasionally serialize a numeric argument as a JSON
    string (e.g. "4" instead of 4) even when the declared schema type is
    integer. ADK's plain-function tools unpack the raw JSON args straight
    into the Python call (unlike Pydantic-model tool params, which DO coerce)
    so every numeric arg on these simple tools needs this guard, or a stray
    string blows up an arithmetic comparison deep inside the tool.
    """
    return int(float(value)) if isinstance(value, str) else int(value)


def _to_float(value) -> float:
    """Coerce a possibly-stringified numeric tool argument to float.

    Unlike _to_int, this preserves fractional values (e.g. a sub-1 `scale`
    meant to downscale a large region) instead of truncating them to 0.
    """
    return float(value)


def get_image_size(tool_context: ToolContext) -> dict:
    """Return the pixel width/height of the diagram image attached to this session."""
    from PIL import Image

    data = _get_source_image_bytes(tool_context)
    if data is None:
        return {"error": "no source image found for this session -- attach the diagram image in your first message"}
    with Image.open(io.BytesIO(data)) as im:
        return {"width": im.width, "height": im.height}


def inspect_region(tool_context: ToolContext, x: int, y: int, w: int, h: int, scale: float = 4) -> dict:
    """Crop a region of the source image and zoom in, so you can actually read small/dense detail.

    Use this whenever a region has small text, a table, a legend with many
    numbers, a Venn diagram, or several small boxes packed close together —
    anywhere you're not fully confident reading it from the full image. The
    cropped, upscaled image is returned to you directly (not just saved to
    disk) — look at it before transcribing labels/numbers from that area.
    Prefer several small, targeted crops (roughly 150-300px per side before
    scaling) over one huge crop — it reads more clearly, costs less, and
    won't get silently downscaled (the result's `scale` tells you the scale
    actually used, which is reduced automatically if w*scale or h*scale
    would exceed a safe size for the API — request a smaller region rather
    than a smaller scale if a crop comes back less sharp than expected).

    Args:
        x: Left edge of the region, in source-image pixels.
        y: Top edge of the region, in source-image pixels.
        w: Width of the region, in source-image pixels.
        h: Height of the region, in source-image pixels.
        scale: Upscale factor applied after cropping (3-5 works well for small text).
            Can be fractional (e.g. 0.4) to downscale a large region instead —
            useful for a quick overview crop rather than a full-resolution zoom.
    """
    from PIL import Image
    from google.genai import types

    x, y, w, h = _to_int(x), _to_int(y), _to_int(w), _to_int(h)
    scale = _to_float(scale)

    data = _get_source_image_bytes(tool_context)
    if data is None:
        return {"error": "no source image found for this session -- attach the diagram image in your first message"}
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(im.width, x + w), min(im.height, y + h)
        if x1 <= x0 or y1 <= y0:
            return {"error": f"region ({x},{y},{w},{h}) does not overlap the image ({im.width}x{im.height})"}
        crop = im.crop((x0, y0, x1, y1))

        target_w, target_h = crop.width * scale, crop.height * scale
        effective_scale = scale
        if target_w > MAX_IMAGE_EDGE or target_h > MAX_IMAGE_EDGE:
            effective_scale = min(MAX_IMAGE_EDGE / crop.width, MAX_IMAGE_EDGE / crop.height)
            target_w = crop.width * effective_scale
            target_h = crop.height * effective_scale
        # Floor unconditionally, not just in the branch above -- a small/fractional
        # `scale` (e.g. 0.4 requested to downscale a huge region) can land here with
        # target_w/target_h under 1 without ever exceeding MAX_IMAGE_EDGE, and
        # Image.resize raises "height and width must be > 0" on a (0, 0) target.
        target_w, target_h = max(1, round(target_w)), max(1, round(target_h))
        crop = crop.resize((target_w, target_h), Image.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, format="PNG")

    return {
        "region": [x0, y0, x1 - x0, y1 - y0],
        "scale": round(effective_scale, 2),
        "image": types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
    }


def sample_color_at(tool_context: ToolContext, x: int, y: int) -> dict:
    """Sample the exact pixel color at (x, y) in the source image.

    Use this instead of guessing a fill/stroke color by eye — pick a point
    well inside the shape you're measuring, away from its border/text.

    Args:
        x: Pixel x-coordinate.
        y: Pixel y-coordinate.
    """
    from PIL import Image

    x, y = _to_int(x), _to_int(y)

    data = _get_source_image_bytes(tool_context)
    if data is None:
        return {"error": "no source image found for this session -- attach the diagram image in your first message"}
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        if not (0 <= x < im.width and 0 <= y < im.height):
            return {"error": f"({x},{y}) is outside the image ({im.width}x{im.height})"}
        r, g, b = im.getpixel((x, y))
        return {"hex": f"#{r:02X}{g:02X}{b:02X}"}


def find_bbox_of_color(
    tool_context: ToolContext,
    hex_color: str,
    x: Optional[int] = None,
    y: Optional[int] = None,
    w: Optional[int] = None,
    h: Optional[int] = None,
    tolerance: int = 12,
) -> dict:
    """Find the tight bounding box of a fill color, to get exact shape geometry instead of eyeballing it.

    Scans pixels matching hex_color (within `tolerance` per channel) and returns
    the bounding box of all matches. Restrict the search to roughly where you
    expect the shape to be — via x/y/w/h, same convention as inspect_region —
    otherwise anti-aliased edges of unrelated same-ish colored elements
    elsewhere in the image can pollute the result; if the returned box looks
    too big/sparse (low pixel_count relative to its area), narrow the region
    and retry.

    Args:
        hex_color: Color to search for, e.g. "#248D45".
        x: Left edge of the region to restrict the search to, in source-image pixels. Omit to search the whole image.
        y: Top edge of the region, in source-image pixels.
        w: Width of the region, in source-image pixels.
        h: Height of the region, in source-image pixels.
        tolerance: Max per-channel difference to still count as a match.
    """
    from PIL import Image

    tolerance = _to_int(tolerance)
    has_region = None not in (x, y, w, h)

    data = _get_source_image_bytes(tool_context)
    if data is None:
        return {"error": "no source image found for this session -- attach the diagram image in your first message"}
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        if has_region:
            x0, y0 = max(0, _to_int(x)), max(0, _to_int(y))
            x1, y1 = min(im.width, x0 + _to_int(w)), min(im.height, y0 + _to_int(h))
        else:
            x0, y0, x1, y1 = 0, 0, im.width, im.height
        if x1 <= x0 or y1 <= y0:
            return {
                "error": (
                    f"region (x={x}, y={y}, w={w}, h={h}) does not overlap the image "
                    f"({im.width}x{im.height})"
                )
            }
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


# Node/edge geometry is extracted directly in native source-image pixel
# coordinates (inspect_region, find_bbox_of_color, sample_color_at all work
# in that same space, and the model places x/y/w/h to match). But font_size
# (schema.py DiagramNode/DiagramEdge) is a small fixed range, ~10-14, that is
# never scaled to the source resolution -- it only looks right when the
# source image is roughly normal size. For a large source (e.g. a 5760x3240
# HiDPI screenshot) shapes come out several times bigger than that
# calibration while text stays the same absolute size, so the result reads
# as "same canvas, unreadably small text" even though font_size itself is
# fine. Rescaling geometry down to a consistent logical size restores the
# shape-to-text ratio regardless of the source image's actual resolution.
# Chosen to sit above every existing diagram's extent in this repo (~1630px
# max) so normal-sized diagrams are left untouched.
_MAX_DIAGRAM_EDGE = 2000.0

# raw_style properties that describe a decorative/structural pixel size (not
# text metrics) and should scale down with geometry so borders stay
# proportional to their now-smaller shapes -- e.g. AWS group/container boxes
# commonly set strokeWidth=2, which looks chunky relative to a shape shrunk
# to a third of its extracted size. Deliberately excludes fontSize/spacing*,
# which must stay fixed for the same reason font_size itself isn't scaled.
_SCALE_SENSITIVE_STYLE_KEYS = ("strokeWidth", "startSize", "endSize")
_STYLE_KEY_RE = re.compile(
    r"(" + "|".join(_SCALE_SENSITIVE_STYLE_KEYS) + r")=([\d.]+)"
)


def _scale_raw_style(raw_style: str, scale: float) -> str:
    return _STYLE_KEY_RE.sub(
        lambda m: f"{m.group(1)}={float(m.group(2)) * scale:.2f}", raw_style
    )


# The uniform scale factor is derived only from the diagram's overall extent,
# not from any individual node -- so a small element (e.g. a thin connector
# label) shrinks by the same factor as everything else, which can push it
# below what its (deliberately unscaled) font needs to render one line
# without overflowing into a neighboring shape. Floor matches the smallest
# node size actually seen working in this repo's existing diagrams (24x20 in
# diagrams/img_1.diagram.json), rather than a guess.
_MIN_NODE_W, _MIN_NODE_H = 24.0, 20.0


def _rescale_diagram(diagram: Diagram) -> Optional[float]:
    """Uniformly downscale node/edge geometry if the diagram's extent exceeds _MAX_DIAGRAM_EDGE.

    Returns the scale factor applied, or None if no rescale was needed.
    """
    max_extent = max(
        (n.x + n.w for n in diagram.nodes), default=0.0,
    )
    max_extent = max(max_extent, max((n.y + n.h for n in diagram.nodes), default=0.0))
    for e in diagram.edges:
        for pt in (e.source_point, e.target_point):
            if pt:
                max_extent = max(max_extent, pt[0], pt[1])

    if max_extent <= _MAX_DIAGRAM_EDGE:
        return None

    scale = _MAX_DIAGRAM_EDGE / max_extent
    for n in diagram.nodes:
        n.x *= scale
        n.y *= scale
        n.w *= scale
        n.h *= scale
        if n.w < _MIN_NODE_W:
            n.x -= (_MIN_NODE_W - n.w) / 2
            n.w = _MIN_NODE_W
        if n.h < _MIN_NODE_H:
            n.y -= (_MIN_NODE_H - n.h) / 2
            n.h = _MIN_NODE_H
        if n.raw_style:
            n.raw_style = _scale_raw_style(n.raw_style, scale)
    for e in diagram.edges:
        if e.source_point:
            e.source_point = (e.source_point[0] * scale, e.source_point[1] * scale)
        if e.target_point:
            e.target_point = (e.target_point[0] * scale, e.target_point[1] * scale)
        if e.label_offset:
            e.label_offset = (e.label_offset[0] * scale, e.label_offset[1] * scale)
        if e.raw_style:
            e.raw_style = _scale_raw_style(e.raw_style, scale)
    return scale


# draw.io has no auto-shrink-font-to-fit-box feature: with whiteSpace=wrap, a
# label whose wrapped text is taller than its box just overflows past the
# box edges into whatever's below, and a font picked without regard to box
# size is just as likely to be needlessly tiny in a roomy box. Search for the
# largest font size in this range whose wrapped text actually measures inside
# the node's w/h before falling back to _MIN_FONT_SIZE -- Pillow's real text
# metrics, not a guess, though still an approximation of draw.io's own
# browser text rendering. Bounded above by 14 to match the range this
# pipeline's diagrams already use in practice (schema.py DiagramNode default).
_MIN_FONT_SIZE, _MAX_FONT_SIZE = 6, 14
_LABEL_PADDING = 4.0  # px, rough match for draw.io's default internal label margin
_LINE_HEIGHT_FACTOR = 1.2  # draw.io's approximate default line-height multiplier


def _wrap_text_lines(label: str, font, max_width: float) -> list[str]:
    words = label.split()
    if not words:
        return []
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.getlength(candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _label_fits(label: str, font_size: int, w: float, h: float) -> bool:
    from PIL import ImageFont

    font = ImageFont.load_default(size=font_size)
    max_width = max(1.0, w - 2 * _LABEL_PADDING)
    lines = _wrap_text_lines(label, font, max_width)
    if not lines:
        return True
    total_height = len(lines) * font_size * _LINE_HEIGHT_FACTOR
    max_line_width = max(font.getlength(line) for line in lines)
    return max_line_width <= max_width and total_height <= (h - 2 * _LABEL_PADDING)


def _autofit_font_sizes(diagram: Diagram) -> None:
    """Pick the largest font size that fits each node's label inside its box.

    Skips nodes with no label or a raw_style (raw_style's own embedded
    fontSize, if any, takes over and this pipeline doesn't rewrite it).
    If even _MIN_FONT_SIZE doesn't fit, that node's label is too long for its
    box -- left at _MIN_FONT_SIZE and accepted as a documented edge case
    rather than growing the box (which risks colliding with neighbors).
    """
    for n in diagram.nodes:
        if not n.label or n.raw_style:
            continue
        for size in range(_MAX_FONT_SIZE, _MIN_FONT_SIZE - 1, -1):
            if _label_fits(n.label, size, n.w, n.h):
                n.font_size = size
                break
        else:
            n.font_size = _MIN_FONT_SIZE


def save_diagram(tool_context: ToolContext, diagram: Diagram) -> dict:
    """Validate the extracted diagram structure and save it into session state.

    Call this once you've identified every node and edge. It checks that
    every edge's source/target refers to a node id that actually exists
    (beyond the basic schema validation ADK already applies) before storing
    it, so mistakes surface here rather than as a broken .drawio later.

    Args:
        diagram: The full extracted diagram (nodes + edges).
    """
    # For a large/deeply-nested tool argument like this one, Claude occasionally
    # emits the whole structure as an escaped JSON string instead of a native
    # nested object. ADK's preprocess_args tries Diagram.model_validate(...) on
    # it, that raises (a str isn't a mapping), and it only logs a warning and
    # silently leaves the raw string in place rather than erroring -- so this
    # tool must defend itself or it hits "str object has no attribute 'nodes'".
    if isinstance(diagram, str):
        diagram = Diagram.model_validate_json(diagram)

    applied_scale = _rescale_diagram(diagram)
    _autofit_font_sizes(diagram)

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

    tool_context.state[_DIAGRAM_JSON_KEY] = diagram.model_dump(exclude_none=True)
    result = {"saved": True, "nodes": len(diagram.nodes), "edges": len(diagram.edges)}
    if applied_scale is not None:
        result["rescaled"] = round(applied_scale, 4)
    return result


def render_drawio(tool_context: ToolContext) -> dict:
    """Render the diagram JSON saved by save_diagram into mxGraph XML, stored in session state.

    This step is plain deterministic code, not a model guess — it cannot by
    itself introduce a dangling reference or malformed XML as long as the
    JSON passed schema validation in save_diagram.
    """
    diagram = tool_context.state.get(_DIAGRAM_JSON_KEY)
    if diagram is None:
        return {"error": "no saved diagram in session state — call save_diagram first"}
    xml_str = render_mxgraph_xml(diagram)
    tool_context.state[_DRAWIO_XML_KEY] = xml_str
    return {"nodes": len(diagram["nodes"]), "edges": len(diagram["edges"])}


def validate_drawio(tool_context: ToolContext) -> dict:
    """Parse the rendered drawio XML and confirm every edge's source/target id resolves to a real node.

    Always call this after render_drawio. If it reports dangling ids or a
    parse error, fix the diagram and call save_diagram/render_drawio again —
    don't hand-patch the XML.
    """
    xml_str = tool_context.state.get(_DRAWIO_XML_KEY)
    if xml_str is None:
        return {"error": "no rendered drawio XML in session state — call render_drawio first"}
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        return {"valid": False, "parse_error": str(e)}

    cells = root.findall(".//mxCell")
    ids = {c.get("id") for c in cells}
    dangling = []
    for c in cells:
        for attr in ("source", "target"):
            v = c.get(attr)
            if v and v not in ids:
                dangling.append({"cell": c.get("id"), "attr": attr, "missing_id": v})
    return {"valid": not dangling, "cell_count": len(cells), "dangling": dangling}


def sanity_plot(tool_context: ToolContext) -> dict:
    """Render a rough, schematic PNG preview of the saved diagram JSON, returned to you for a real visual check.

    This is a quick matplotlib plot of node/edge positions — colors and
    rough placement only, not final draw.io fidelity (edges are drawn
    straight, not orthogonally routed). The image is returned to you
    directly: actually look at it and compare it against the source image
    (use inspect_region on the source again if you need a refresher on any
    area) before finishing. If you spot a missing node, wrong region, wrong
    count, or a badly-shaped element (e.g. a Venn diagram that came out as
    disconnected circles instead of overlapping ones), fix the diagram and
    call save_diagram -> render_drawio -> validate_drawio -> sanity_plot
    again rather than accepting a mismatch.
    """
    diagram = tool_context.state.get(_DIAGRAM_JSON_KEY)
    if diagram is None:
        return {"error": "no saved diagram in session state — call save_diagram first"}

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
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)

    from google.genai import types

    return {
        "node_count": len(diagram["nodes"]),
        "edge_count": len(diagram["edges"]),
        "image": types.Part.from_bytes(data=_clamp_image_bytes(buf.getvalue()), mime_type="image/png"),
    }
