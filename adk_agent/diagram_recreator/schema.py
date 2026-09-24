"""Structured diagram schema shared between the agent's tools and
scripts/json_to_drawio.py (the deterministic JSON -> mxGraph XML renderer).

Keeping this as a Pydantic model (rather than a free-form dict) means ADK
builds a real JSON-schema contract for the LLM's tool call, so the model is
constrained to emit fields the renderer actually understands instead of
inventing its own shape.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Shape = Literal["rect", "ellipse", "text"]


class DiagramNode(BaseModel):
    id: str = Field(description="Unique id, referenced by edges as source/target.")
    shape: Shape = "rect"
    label: str = ""
    x: float
    y: float
    w: float
    h: float
    fill: Optional[str] = Field(None, description="Hex fill color, e.g. #248D45. Omit for no fill.")
    stroke: Optional[str] = Field(None, description="Hex stroke color. Omit for no border.")
    font_color: str = "#000000"
    font_size: int = 12
    bold: bool = False
    rounded: bool = Field(True, description="Rounded corners for shape=rect. Most real boxes are sharp-cornered — check the source image, don't default blindly.")
    vertical_text: bool = Field(False, description="Rotate the label 90 degrees, for narrow vertical bars like a 'Data lake' bar.")
    align: Literal["left", "center", "right"] = "center"
    raw_style: Optional[str] = Field(
        None,
        description=(
            "Verbatim mxGraph style string, used INSTEAD of the shape/fill/stroke fields above. "
            "Use this for draw.io's native stencil shapes when they exist and fit better than a plain "
            "rect/ellipse: AWS icons (shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.<name>;...), "
            "UML components (shape=component;...), UML interaction frames (shape=umlFrame;...), etc. "
            "Leave null for plain shapes."
        ),
    )


class DiagramEdge(BaseModel):
    id: Optional[str] = None
    source: Optional[str] = Field(None, description="Source node id. Omit if using source_point instead.")
    target: Optional[str] = Field(None, description="Target node id. Omit if using target_point instead.")
    source_point: Optional[tuple[float, float]] = Field(
        None, description="[x, y] to use instead of `source` when the line starts at a bare point, not a node (e.g. a sequence-diagram message, or a decorative divider)."
    )
    target_point: Optional[tuple[float, float]] = Field(None, description="[x, y] to use instead of `target`.")
    label: str = ""
    dashed: bool = False
    end_arrow: bool = True
    start_arrow: bool = Field(False, description="Set true for bidirectional arrows.")
    color: str = "#000000"
    font_size: int = 10
    label_offset: Optional[tuple[float, float]] = Field(
        None,
        description=(
            "[dx, dy] pixel offset to push this edge's label off the raw midpoint. Use this whenever two "
            "edges run between the same two points (e.g. opposite-direction arrows) so their labels don't "
            "stack on top of each other — match how the source image visually separates them (usually into "
            "two rows)."
        ),
    )
    raw_style: Optional[str] = Field(None, description="Verbatim mxGraph edge style string override, e.g. for UML dependency arrows: 'html=1;endArrow=open;dashed=1;...'.")


class Diagram(BaseModel):
    name: str = Field(description="Human-readable diagram title.")
    nodes: list[DiagramNode]
    edges: list[DiagramEdge]
