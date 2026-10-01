"""Tool-call contracts for the diagram_editor agent.

Pydantic models (not free-form dicts) so ADK builds a real JSON schema for the
LLM, like diagram_recreator/schema.py. The LLM only ever states *structure*
and *intent*; coordinates for new elements are computed by layout code.

`like` fields name a cell whose style/size should be cloned:
  * in an op acting on the working diagram -> a cell id in that diagram
  * `RefCell` -> a cell id inside one of the files in reference_diagrams/
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class RefCell(BaseModel):
    reference: str = Field(description="File name from list_references, e.g. 'payments.drawio'.")
    cell: str = Field(description="Cell id inside that reference (see get_reference).")


# ---- create mode -----------------------------------------------------------

class SpecZone(BaseModel):
    id: str
    label: str
    like: RefCell = Field(description="Reference container whose look to copy (a zone/tier/group).")
    parent: Optional[str] = Field(None, description="Id of an enclosing zone, for nested zones.")


class SpecComponent(BaseModel):
    id: str
    label: str
    zone: Optional[str] = Field(None, description="Zone id this component sits in. Omit only for free-standing items.")
    like: RefCell = Field(description="Reference node whose shape/style/size to copy.")


class SpecConnection(BaseModel):
    source: str
    target: str
    label: str = ""
    like: Optional[RefCell] = Field(None, description="Reference edge to copy the line style from. Omit for the reference's most common edge style.")


class ArchitectureSpec(BaseModel):
    title: str
    layout_from: str = Field(description="Reference file whose zone arrangement, spacing and canvas conventions to follow.")
    zones: list[SpecZone] = Field(default_factory=list)
    components: list[SpecComponent]
    connections: list[SpecConnection] = Field(default_factory=list)


# ---- edit mode -------------------------------------------------------------

class UpdateLabel(BaseModel):
    op: Literal["update_label"] = "update_label"
    id: str
    label: str


class Move(BaseModel):
    op: Literal["move"] = "move"
    id: str
    x: Optional[float] = Field(None, description="Absolute canvas x. Omit to keep.")
    y: Optional[float] = Field(None, description="Absolute canvas y. Omit to keep.")
    parent: Optional[str] = Field(None, description="Move into this container (its id). Position is then auto-placed unless x/y given.")


class Resize(BaseModel):
    op: Literal["resize"] = "resize"
    id: str
    w: Optional[float] = None
    h: Optional[float] = None


class SetStyle(BaseModel):
    op: Literal["set_style"] = "set_style"
    id: str
    set: dict[str, str] = Field(default_factory=dict, description="Style keys to set, e.g. {'fillColor': '#DAE8FC'}.")
    remove: list[str] = Field(default_factory=list, description="Style keys to remove.")
    like: Optional[Union[str, RefCell]] = Field(None, description="Replace the whole style with that of this cell (working-diagram id or RefCell).")


class Delete(BaseModel):
    op: Literal["delete"] = "delete"
    id: str = Field(description="Removes the cell, its children, and every edge touching any of them.")


class AddComponent(BaseModel):
    op: Literal["add_component"] = "add_component"
    id: Optional[str] = Field(None, description="New id. Omit to auto-generate from the label.")
    label: str
    parent: Optional[str] = Field(None, description="Container id to place it in.")
    like: Optional[Union[str, RefCell]] = Field(None, description="Cell whose style and size to copy (working-diagram id or RefCell). Strongly preferred.")
    near: Optional[str] = Field(None, description="Existing cell id to place next to (right first, then below).")
    x: Optional[float] = Field(None, description="Absolute x. Omit to auto-place.")
    y: Optional[float] = Field(None, description="Absolute y. Omit to auto-place.")
    w: Optional[float] = Field(None, description="Width. Omit to use the size of `like`.")
    h: Optional[float] = Field(None, description="Height. Omit to use the size of `like`.")
    fit_to: list[str] = Field(
        default_factory=list,
        description=(
            "Cell ids to wrap: the new node is placed and sized to surround all of them plus `padding` "
            "(x/y/w/h are then ignored). Use for backgrounds, highlights and group boxes. "
            "Follow with a `reorder` to_back op so it sits behind them."
        ),
    )
    padding: float = Field(20, description="Margin around `fit_to` cells.")


class AddConnection(BaseModel):
    op: Literal["add_connection"] = "add_connection"
    id: Optional[str] = None
    source: str
    target: str
    label: str = ""
    like: Optional[Union[str, RefCell]] = Field(None, description="Edge whose line style to copy (working-diagram id or RefCell).")


class ScaleFonts(BaseModel):
    op: Literal["scale_fonts"] = "scale_fonts"
    factor: float = Field(ge=0.3, le=3.0, description="Multiply each font size by this (0.9 = 10% smaller).")
    ids: Optional[list[str]] = Field(None, description="Cells to change. Omit for every labelled node and edge.")
    min_size: int = Field(6, description="Never go below this font size.")
    max_size: int = Field(48, description="Never go above this font size.")


# ---- generic, XML-level operations (escape hatch for anything the ops above don't cover) ----

class UpsertXml(BaseModel):
    op: Literal["upsert_xml"] = "upsert_xml"
    xml: str = Field(description=(
        "ONE <mxCell ...>...</mxCell> element (with its <mxGeometry> child if any), as stored in the file. "
        "An existing cell with the same id is replaced in place; otherwise it is appended on top. "
        "`parent` defaults to '1'. Geometry of a cell inside a container is RELATIVE to that container."
    ))


class SetAttrs(BaseModel):
    op: Literal["set_attrs"] = "set_attrs"
    id: str
    attrs: dict[str, Optional[str]] = Field(description=(
        "mxCell attributes to set, e.g. {'value': 'New', 'style': '...', 'parent': 'z1', 'source': 'a'}. "
        "null removes the attribute. `id` can't be changed."
    ))


class SetGeometry(BaseModel):
    op: Literal["set_geometry"] = "set_geometry"
    id: str
    x: Optional[float] = None
    y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    # raw, as stored: relative to the cell's parent (use `move` for absolute canvas coordinates)


class Reorder(BaseModel):
    op: Literal["reorder"] = "reorder"
    id: str
    to: Literal["back", "front", "before", "after"] = Field(description=(
        "Draw order among the canvas-level cells: 'back' = behind everything, 'front' = on top, "
        "'before'/'after' = just behind/in front of `ref`."
    ))
    ref: Optional[str] = Field(None, description="Required for 'before'/'after'.")


EditOp = Annotated[
    Union[UpdateLabel, Move, Resize, SetStyle, Delete, AddComponent, AddConnection, ScaleFonts,
          UpsertXml, SetAttrs, SetGeometry, Reorder],
    Field(discriminator="op"),
]
