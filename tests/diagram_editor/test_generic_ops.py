from xml.etree import ElementTree as ET

import pytest

from adk_agent.diagram_editor import drawio_doc as dd
from adk_agent.diagram_editor import tools
from adk_agent.diagram_editor.edit_ops import apply_ops
from adk_agent.diagram_editor.schema import (
    AddComponent, Reorder, SetAttrs, SetGeometry, UpsertXml,
)
from tests.diagram_editor.test_drawio_doc import PLAIN
from tests.diagram_editor.test_tools import Ctx

AWS = """<mxfile><diagram name="Arch"><mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>
<mxCell id="user" value="User" style="shape=actor;" vertex="1" parent="1"><mxGeometry x="20" y="120" width="30" height="60" as="geometry"/></mxCell>
<mxCell id="aws" value="AWS Cloud" style="shape=mxgraph.aws4.group;grIcon=mxgraph.aws4.group_aws_cloud;strokeColor=#232F3E;fillColor=none;verticalAlign=top;" vertex="1" parent="1">
  <mxGeometry x="100" y="40" width="400" height="260" as="geometry"/></mxCell>
<mxCell id="vpc" value="VPC" style="shape=mxgraph.aws4.group;grIcon=mxgraph.aws4.group_vpc;strokeColor=#248814;fillColor=none;" vertex="1" parent="aws">
  <mxGeometry x="20" y="40" width="360" height="200" as="geometry"/></mxCell>
<mxCell id="ec2" value="EC2" style="shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.ec2;" vertex="1" parent="vpc"><mxGeometry x="40" y="60" width="60" height="60" as="geometry"/></mxCell>
<mxCell id="rds" value="RDS" style="shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.rds;" vertex="1" parent="vpc"><mxGeometry x="240" y="60" width="60" height="60" as="geometry"/></mxCell>
<mxCell id="e1" value="" style="endArrow=block;" edge="1" parent="1" source="ec2" target="rds"><mxGeometry relative="1" as="geometry"/></mxCell>
<mxCell id="e2" value="" style="endArrow=block;" edge="1" parent="1" source="user" target="ec2"><mxGeometry relative="1" as="geometry"/></mxCell>
</root></mxGraphModel></diagram></mxfile>"""


def _ctx(xml=AWS):
    c = Ctx()
    c.state["drawio_xml"] = dd.serialize(dd.load(xml))
    return c


def _order(ctx):
    return [c.get("id") for c in dd.load(ctx.state["drawio_xml"]).cells()]


def _v(ctx, cid):
    return next(v for v in tools.describe_diagram(ctx)["vertices"] if v["id"] == cid)


def _cell_xml(xml, cid):
    return ET.tostring(dd.load(xml).cell(cid), encoding="unicode")


def test_aws_orange_transparent_background_behind_the_zone():
    """'identify the AWS zone and add an orange, transparent background behind it'."""
    ctx = _ctx()
    aws = _v(ctx, "aws")  # found by label / shape in describe_diagram
    assert aws["label"] == "AWS Cloud" and aws["style"]["shape"] == "mxgraph.aws4.group"
    res = tools.edit_diagram(ctx, [
        {"op": "add_component", "id": "aws_bg", "label": "", "fit_to": ["aws"], "padding": 24},
        {"op": "set_attrs", "id": "aws_bg", "attrs": {
            "style": "rounded=0;whiteSpace=wrap;html=1;fillColor=#FF9900;fillOpacity=25;strokeColor=#FF9900;strokeOpacity=60;"}},
        {"op": "reorder", "id": "aws_bg", "to": "back"},
    ])
    assert res["applied"], res
    bg = _v(ctx, "aws_bg")
    assert (bg["x"], bg["y"], bg["w"], bg["h"]) == (76, 16, 448, 308)  # surrounds the zone with 24px padding
    order = _order(ctx)
    assert order[:3] == ["0", "1", "aws_bg"] and order.index("aws_bg") < order.index("aws")
    xml = tools.get_cells_xml(ctx, ["aws_bg"])["cells"][0]["xml"]
    assert "fillOpacity=25" in xml and "#FF9900" in xml
    assert tools.validate_drawio(ctx)["valid"]
    # untouched cells are unchanged
    assert tools.get_cells_xml(ctx, ["ec2"])["cells"][0]["xml"] == _cell_xml(AWS, "ec2")


def test_get_cells_xml_children_unknown_and_no_diagram():
    ctx = _ctx()
    r = tools.get_cells_xml(ctx, ["aws", "nope"], include_children=True)
    assert {c["id"] for c in r["cells"]} == {"aws", "vpc", "ec2", "rds"}
    assert r["unknown_ids"] == ["nope"] and "grIcon=mxgraph.aws4.group_aws_cloud" in r["cells"][0]["xml"]
    assert "error" in tools.get_cells_xml(Ctx(), ["x"])


def test_upsert_replaces_in_place_and_appends_on_top():
    doc = dd.load(PLAIN)
    frag = ('<mxCell id="api" value="Gateway" style="shape=rect;fillColor=#000000;" vertex="1" parent="zone">'
            '<mxGeometry x="1" y="2" width="30" height="40" as="geometry"/></mxCell>')
    new, res = apply_ops(doc, [UpsertXml(xml=frag)])
    assert res[0]["ok"]
    assert [c.get("id") for c in new.cells()] == [c.get("id") for c in doc.cells()]  # draw order preserved
    assert new.cell("api").get("value") == "Gateway"
    fresh = '<mxCell id="fresh" value="F" vertex="1"><mxGeometry x="0" y="0" width="10" height="10" as="geometry"/></mxCell>'
    new, _ = apply_ops(doc, [UpsertXml(xml=fresh)])
    assert [c.get("id") for c in new.cells()][-1] == "fresh" and new.cell("fresh").get("parent") == "1"


@pytest.mark.parametrize("xml,msg", [
    ("<mxCell id='a'", "not well-formed"),
    ("<div id='a'/>", "expected a single <mxCell>"),
    ("<mxCell value='x' vertex='1'/>", "needs an id"),
    ("<mxCell id='1' vertex='1'/>", "reserved"),
    ("<mxCell id='z'/>", "vertex"),
    ("<mxCell id='z' vertex='1'><script/></mxCell>", "unexpected child"),
    ('<!DOCTYPE x [<!ENTITY a "b">]><mxCell id="z" vertex="1" value="&a;"/>', "not well-formed"),
    ("<mxCell id='z' vertex='1' value='" + "x" * 25000 + "'/>", "too large"),
])
def test_upsert_rejects_bad_fragments(xml, msg):
    new, res = apply_ops(dd.load(PLAIN), [UpsertXml(xml=xml)])
    assert new is None and msg in res[0]["error"]


def test_set_attrs_and_set_geometry():
    doc = dd.load(PLAIN)
    new, res = apply_ops(doc, [SetAttrs(id="api", attrs={"value": "X", "tooltip": "hi"}),
                               SetAttrs(id="api", attrs={"tooltip": None}),
                               SetGeometry(id="api", x=50, width=200)])
    assert all(r["ok"] for r in res)
    c = new.cell("api")
    assert c.get("value") == "X" and c.get("tooltip") is None
    g = c.find("mxGeometry")
    assert (g.get("x"), g.get("y"), g.get("width")) == ("50", "30", "200")
    for bad in (SetAttrs(id="api", attrs={"id": "z"}), SetAttrs(id="api", attrs={"on click": "x"}),
                SetAttrs(id="ghost", attrs={}), SetGeometry(id="ghost", x=1)):
        assert apply_ops(doc, [bad])[0] is None


def test_reorder_variants():
    doc = dd.load(PLAIN)  # order: 0 1 zone api db e1

    def ids(ops):
        new, res = apply_ops(doc, ops)
        assert new is not None, res
        return [c.get("id") for c in new.cells()]

    assert ids([Reorder(id="e1", to="back")]) == ["0", "1", "e1", "zone", "api", "db"]
    assert ids([Reorder(id="zone", to="front")]) == ["0", "1", "api", "db", "e1", "zone"]
    assert ids([Reorder(id="db", to="before", ref="api")]) == ["0", "1", "zone", "db", "api", "e1"]
    assert ids([Reorder(id="zone", to="after", ref="db")]) == ["0", "1", "api", "db", "zone", "e1"]
    for bad in (Reorder(id="api", to="before"), Reorder(id="api", to="after", ref="api"), Reorder(id="1", to="back")):
        assert apply_ops(doc, [bad])[0] is None


def test_bad_raw_edits_are_caught_by_validation_and_not_saved():
    ctx = _ctx(PLAIN)
    before = ctx.state["drawio_xml"]
    dangling = tools.edit_diagram(ctx, [{"op": "set_attrs", "id": "e1", "attrs": {"target": "ghost"}}])
    cyc = tools.edit_diagram(ctx, [{"op": "set_attrs", "id": "zone", "attrs": {"parent": "api"}}])
    assert dangling["applied"] is False and cyc["applied"] is False
    assert any(p["problem"] == "parent cycle" for p in cyc["validation"]["problems"])
    assert ctx.state["drawio_xml"] == before


def test_add_component_size_and_fit_to():
    doc = dd.load(PLAIN)
    new, _ = apply_ops(doc, [AddComponent(id="big", label="Big", x=10, y=10, w=300, h=90)])
    b = next(v for v in dd.summarize(new)["vertices"] if v["id"] == "big")
    assert (b["w"], b["h"]) == (300.0, 90.0)
    assert apply_ops(doc, [AddComponent(label="x", fit_to=["ghost"])])[0] is None
    assert apply_ops(doc, [AddComponent(label="x", fit_to=["e1"])])[0] is None  # edges have no box
    # fit_to a cell nested in a container still works in absolute coordinates
    new, _ = apply_ops(doc, [AddComponent(id="hl", label="", fit_to=["api"], padding=10)])
    hl = next(v for v in dd.summarize(new)["vertices"] if v["id"] == "hl")
    assert (hl["x"], hl["y"], hl["w"], hl["h"]) == (110, 70, 140, 80)


def test_mixed_high_and_low_level_ops_are_atomic():
    ctx = _ctx(PLAIN)
    before = ctx.state["drawio_xml"]
    res = tools.edit_diagram(ctx, [{"op": "update_label", "id": "api", "label": "GW"},
                                   {"op": "reorder", "id": "zone", "to": "front"},
                                   {"op": "upsert_xml", "xml": "<nope"}])
    assert res["applied"] is False and ctx.state["drawio_xml"] == before
