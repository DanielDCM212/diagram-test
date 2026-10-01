"""Client for the draw.io image export server (jgraph/draw-image-export2).

That server renders real draw.io output (actual shapes, stencils, edge routing), which is
what the preview should show. It is optional infrastructure: when it is unconfigured,
unreachable or returns anything that is not a PNG, callers fall back to the schematic
Python renderer -- so every failure mode here raises ExportError with a short reason and
nothing else.

Config (env):
    DRAWIO_EXPORT_URL      server URL, e.g. http://host:8000/ (unset -> server rendering disabled).
                           The server answers on any path, so the path is irrelevant.
    DRAWIO_EXPORT_TIMEOUT  seconds, default 15

No authentication. Request: POST application/x-www-form-urlencoded with `format=png`,
`xml=<raw diagram XML>`, `from` = page index, `border`, `bg`. Verified against the server's
export.js: `xml` is used as-is (only URL-decoded if it starts with "%3C"; `xmldata` is the
deflate+base64 variant, which we don't need). Response: 200 with the PNG bytes.
"""
from __future__ import annotations

import io
import os
import urllib.error
import urllib.parse
import urllib.request

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MAX_IMAGE_EDGE = 2576  # per-image limit for the model; larger renders are downscaled


class ExportError(Exception):
    """The export server could not produce a usable PNG (reason in the message)."""


def is_configured() -> bool:
    return bool(os.environ.get("DRAWIO_EXPORT_URL", "").strip())


def _clamp(png: bytes, max_edge: int = MAX_IMAGE_EDGE) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(png)) as im:
        if im.width <= max_edge and im.height <= max_edge:
            return png
        ratio = min(max_edge / im.width, max_edge / im.height)
        im = im.convert("RGB").resize((max(1, round(im.width * ratio)), max(1, round(im.height * ratio))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


def export_png(xml: str, page: int = 0) -> bytes:
    """Render `xml` (one page) to PNG bytes via the export server.

    Raises ExportError if the server is unconfigured, unreachable, slow, answers with an
    error status, or returns something that is not a PNG.
    """
    url = os.environ.get("DRAWIO_EXPORT_URL", "").strip()
    if not url:
        raise ExportError("DRAWIO_EXPORT_URL is not set")
    try:
        timeout = float(os.environ.get("DRAWIO_EXPORT_TIMEOUT", "15"))
    except ValueError:
        timeout = 15.0

    body = urllib.parse.urlencode({
        "format": "png", "xml": xml, "from": page, "border": 10, "bg": "#ffffff",
    }).encode("utf-8")
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "image/png"}

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        raise ExportError(f"export server returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ExportError(f"export server unreachable: {getattr(exc, 'reason', exc)}") from exc
    if not data.startswith(_PNG_MAGIC):
        raise ExportError("export server did not return a PNG")
    try:
        return _clamp(data)
    except Exception as exc:  # noqa: BLE001 - corrupt image
        raise ExportError(f"export server returned an unreadable PNG: {exc}") from exc
