"""Approximate "does this label fit its box" checks, using Pillow's real text metrics.

draw.io has no shrink-to-fit, so a label taller/wider than its box just spills out. This is an
approximation of draw.io's own browser rendering (default font, 4px padding, 1.2 line height),
good enough to pick a font size that avoids obvious overflow -- not a pixel-exact guarantee.
"""
from __future__ import annotations

import re
from html import unescape
from typing import Optional

from . import drawio_doc as dd

DEFAULT_FONT = 12
PADDING = 4.0
LINE_HEIGHT = 1.2
_BREAK_RE = re.compile(r"<br\s*/?>|</div>|</p>|</li>|</h\d>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def label_lines(raw: Optional[str]) -> list[str]:
    """Visible text lines of a (possibly HTML) label."""
    if not raw:
        return []
    text = _TAG_RE.sub("", _BREAK_RE.sub("\n", raw))
    return [re.sub(r"[ \t]+", " ", ln).strip() for ln in unescape(text).split("\n") if ln.strip()]


def label_outside_box(style: dict[str, str]) -> bool:
    """True when the label is drawn outside the shape (AWS-style icons), so box size is irrelevant."""
    return (style.get("verticalLabelPosition") in ("bottom", "top")
            or style.get("labelPosition") in ("left", "right"))


def _usable_area(style: dict[str, str], w: float, h: float) -> tuple[float, float]:
    shape = style.get("shape", "")
    if "ellipse" in style or shape == "ellipse" or "rhombus" in style or shape == "rhombus":
        w, h = w * 0.7, h * 0.7  # text area of a round shape is well inside its bounding box
    return max(1.0, w - 2 * PADDING), max(1.0, h - 2 * PADDING)


def fits(raw: Optional[str], style: dict[str, str], w: float, h: float, font_size: float) -> bool:
    from PIL import ImageFont

    lines = label_lines(raw)
    if not lines:
        return True
    max_w, max_h = _usable_area(style, w, h)
    font = ImageFont.load_default(size=max(1, round(font_size)))
    bold = 1.08 if (int(style.get("fontStyle", "0") or 0) & 1) else 1.0
    wrap = style.get("whiteSpace") == "wrap"
    out: list[str] = []
    for line in lines:
        if not wrap:
            out.append(line)
            continue
        words, cur = line.split(" "), ""
        for word in words:
            cand = f"{cur} {word}".strip()
            if cur and font.getlength(cand) * bold > max_w:
                out.append(cur)
                cur = word
            else:
                cur = cand
        out.append(cur)
    widest = max(font.getlength(ln) for ln in out) * bold
    return widest <= max_w and len(out) * font_size * LINE_HEIGHT <= max_h


def current_font(style: dict[str, str]) -> int:
    try:
        return int(float(style.get("fontSize", DEFAULT_FONT)))
    except ValueError:
        return DEFAULT_FONT


def best_font(raw: Optional[str], style: dict[str, str], w: float, h: float, min_font: int) -> Optional[tuple[int, bool]]:
    """Largest font <= the current one that fits, never below `min_font`, never larger than now.

    Returns None if the label already fits (or has no in-box label). Otherwise
    (new_size, fully_fits) where fully_fits is False if even `min_font` still overflows.
    """
    if label_outside_box(style) or not label_lines(raw):
        return None
    cur = current_font(style)
    if fits(raw, style, w, h, cur):
        return None
    floor = min(min_font, cur)
    for size in range(cur - 1, floor - 1, -1):
        if fits(raw, style, w, h, size):
            return size, True
    return floor, False
