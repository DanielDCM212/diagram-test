"""draw.io wraps a cell in <object>/<UserObject> when it has custom data, a link or a tooltip.

The wrapper owns id/label/custom props; the inner <mxCell> owns style/parent/source/target/geometry.
"""
import pytest
from defusedxml import ElementTree as SafeET

from adk_agent.diagram_editor import drawio_doc as dd
from adk_agent.diagram_editor import references as refs
from adk_agent.diagram_editor import tools
from adk_agent.diagram_editor.edit_ops import apply_ops
from adk_agent.diagram_editor.schema import (
    AddComponent, AddConnection, Delete, Move, Reorder, Resize, SetAttrs, SetGeometry, SetStyle, UpdateLabel,
    UpsertXml,
)
from tests.diagram_editor.test_tools import Ctx

WRAPPED = """<mxfile><diagram name="W"><mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>
<object id="zone" label="&lt;b&gt;Backend&lt;/b&gt;" owner="team-core" placeholders="1">
  <mxCell style="rounded=0;fillColor=#EEEEEE;verticalAlign=top;" vertex="1" parent="1">
    <mxGeometry x="100" y="50" width="400" height="200" as="geometry"/></mxCell></object>
<object id="api" label="Orders API" owner="team-payments" tier="1" link="https://wiki.example.com/api" tooltip="Handles checkout">
  <mxCell style="rounded=1;fillColor=#DAE8FC;customKeep=1;" vertex="1" parent="zone">
    <mxGeometry x="20" y="40" width="120" height="60" as="geometry"/></mxCell></object>
<UserObject id="db" label="Orders DB" tooltip="Postgres">
  <mxCell style="shape=cylinder3;fillColor=#D5E8D4;" vertex="1" parent="zone">
    <mxGeometry x="240" y="40" width="80" height="60" as="geometry"/></mxCell></UserObject>
<mxCell id="plain" value="Plain" style="rounded=0;" vertex="1" parent="1"><mxGeometry x="600" y="50" width="80" height="40" as="geometry"/></mxCell>
<object id="e1" label="reads" tooltip="sql"><mxCell style="endArrow=block;dashed=1;" edge="1" parent="1" source="api" target="db">
  <mxGeometry relative="1" as="geometry"/></mxCell></object>
<mxCell id="e2" value="" style="endArrow=block;" edge="1" parent="1" source="plain" target="api"><mxGeometry relative="1" as="geometry"/></mxCell>
</root></mxGraphModel></diagram></mxfile>"""


def _doc():
    return dd.load(WRAPPED)


def _v(doc, cid):
    return next(v for v in dd.summarize(doc)["vertices"] if v["id"] == cid)


def _ids(doc):
    return [c.get("id") for c in doc.cells()]


def _apply(ops, doc=None):
    new, res = apply_ops(doc or _doc(), ops)
    assert new is not None, res
    return new


def test_summarize_sees_wrapped_cells_with_props_and_absolute_geometry():
    s = dd.summarize(_doc())
    by = {v["id"]: v for v in s["vertices"]}
    assert set(by) == {"zone", "api", "db", "plain"}
    assert by["zone"]["label"] == "Backend" and by["zone"]["container"] is True
    assert (by["api"]["x"], by["api"]["y"], by["api"]["w"]) == (120.0, 90.0, 120.0)  # zone offset applied
    assert by["api"]["parent"] == "zone" and by["api"]["style"]["fillColor"] == "#DAE8FC"
    assert by["api"]["props"]["owner"] == "team-payments" and by["api"]["props"]["link"].startswith("https://")
    assert "label" not in by["api"]["props"] and "id" not in by["api"]["props"]
    assert "props" not in by["plain"]
    e = {x["id"]: x for x in s["edges"]}
    assert (e["e1"]["source"], e["e1"]["target"], e["e1"]["label"], e["e1"]["dashed"]) == ("api", "db", "reads", True)
    assert e["e1"]["props"] == {"tooltip": "sql"}
    assert (e["e2"]["source"], e["e2"]["target"]) == ("plain", "api")


def test_validate_wrapped_ok_and_detects_problems():
    assert dd.validate(_doc())["valid"]
    d = _doc()
    dd.set_attr(d.cell("e1"), "target", "ghost")  # lives on the inner cell
    assert [p["problem"] for p in dd.validate(d)["problems"]] == ["dangling target"]
    d = _doc()
    dd.set_attr(d.cell("api"), "parent", "nope")
    assert [p["problem"] for p in dd.validate(d)["problems"]] == ["dangling parent"]
    d = _doc()
    d.root().append(SafeET.fromstring('<mxCell id="api" vertex="1" parent="1"/>'))  # id clash wrapper vs plain
    assert "duplicate id" in {p["problem"] for p in dd.validate(d)["problems"]}
    d = _doc()
    d.root().append(SafeET.fromstring('<object id="hollow" label="x"/>'))
    assert any("no inner" in p["problem"] for p in dd.validate(d)["problems"])


def test_roundtrip_keeps_wrapper_structure_and_every_attribute():
    out = dd.load(dd.serialize(_doc()))
    api = out.cell("api")
    assert api.tag == "object" and api.get("owner") == "team-payments" and api.get("tooltip") == "Handles checkout"
    assert api.find("mxCell").get("style").endswith("customKeep=1;") and api.find("mxCell").get("id") is None
    assert out.cell("db").tag == "UserObject"


def test_update_label_sets_label_on_wrapper_and_keeps_props():
    d = _apply([UpdateLabel(id="api", label="Gateway"), UpdateLabel(id="plain", label="P2")])
    api = d.cell("api")
    assert api.get("label") == "Gateway" and api.get("value") is None and api.find("mxCell").get("value") is None
    assert api.get("owner") == "team-payments" and api.get("link").startswith("https://")
    assert d.cell("plain").get("value") == "P2"
    assert _v(d, "api")["label"] == "Gateway"


def test_move_resize_geometry_style_on_wrapped_cells():
    d = _apply([Move(id="api", x=300, y=100), Resize(id="api", w=200),
                SetGeometry(id="db", height=90),
                SetStyle(id="api", set={"fillColor": "#FFFFFF"}, remove=["customKeep"])])
    api = _v(d, "api")
    assert (api["x"], api["y"], api["w"]) == (300.0, 100.0, 200.0)
    assert d.cell("api").find("mxCell").get("style") == "rounded=1;fillColor=#FFFFFF;"
    assert d.cell("api").find("mxGeometry") is None  # geometry is never put on the wrapper
    assert _v(d, "db")["h"] == 90.0
    assert d.cell("api").get("owner") == "team-payments"


def test_set_attrs_routes_each_attribute_to_the_right_element():
    d = _apply([SetAttrs(id="api", attrs={"value": "New", "owner": "team-x", "tier": None, "style": "rounded=0;",
                                          "parent": "1"})])
    w, inner = d.cell("api"), d.cell("api").find("mxCell")
    assert w.get("label") == "New" and w.get("owner") == "team-x" and w.get("tier") is None
    assert inner.get("style") == "rounded=0;" and inner.get("parent") == "1"
    assert w.get("style") is None and w.get("parent") is None and inner.get("owner") is None
    assert dd.validate(d)["valid"]


def test_delete_wrapped_container_cascades_through_wrapped_children_and_edges():
    d = _apply([Delete(id="zone")])
    assert _ids(d) == ["0", "1", "plain"]  # api/db (wrapped children), e1 (wrapped edge), e2 (touches api) all gone
    d = _apply([Delete(id="db")])
    assert "e1" not in _ids(d) and "api" in _ids(d)
    assert dd.validate(d)["valid"]


def test_add_component_and_connection_with_wrapped_like_and_endpoints():
    d = _apply([AddComponent(id="cache", label="Cache", parent="zone", like="api", near="api"),
                AddConnection(source="cache", target="db", like="e1", label="warm")])
    c = _v(d, "cache")
    assert (c["w"], c["h"]) == (120.0, 60.0) and c["style"]["fillColor"] == "#DAE8FC" and c["parent"] == "zone"
    edge = next(e for e in dd.summarize(d)["edges"] if e["label"] == "warm")
    assert (edge["source"], edge["target"], edge["dashed"]) == ("cache", "db", True)  # style copied from wrapped edge
    # new component avoids the wrapped siblings
    for other in ("api", "db"):
        o = _v(d, other)
        assert not (c["x"] < o["x"] + o["w"] and o["x"] < c["x"] + c["w"] and c["y"] < o["y"] + o["h"] and o["y"] < c["y"] + c["h"])
    # fit_to a wrapped nested cell
    d = _apply([AddComponent(id="hl", label="", fit_to=["api"], padding=10)])
    assert (_v(d, "hl")["x"], _v(d, "hl")["w"]) == (110.0, 140.0)


def test_upsert_xml_wrapper_forms():
    frag = ('<object id="api" label="API v2" owner="team-z"><mxCell style="rounded=1;" vertex="1" parent="zone">'
            '<mxGeometry x="5" y="5" width="50" height="50" as="geometry"/></mxCell></object>')
    d = _apply([UpsertXml(xml=frag)])
    assert _ids(d) == _ids(_doc())  # replaced in place
    assert d.cell("api").get("owner") == "team-z" and _v(d, "api")["label"] == "API v2"
    # a plain cell can be replaced by a wrapper (id keeps its position)
    d = _apply([UpsertXml(xml=frag.replace('id="api"', 'id="plain"').replace('parent="zone"', 'parent="1"'))])
    assert d.cell("plain").tag == "object" and _ids(d) == _ids(_doc())
    # new wrapped cell appended
    d = _apply([UpsertXml(xml=frag.replace('id="api"', 'id="n1"'))])
    assert _ids(d)[-1] == "n1" and dd.validate(d)["valid"]


@pytest.mark.parametrize("xml,msg", [
    ('<object id="a" label="x"><mxCell id="a" vertex="1"/></object>', "id belongs on"),
    ('<object id="a" label="x"/>', "exactly one <mxCell>"),
    ('<object id="a"><mxCell vertex="1"/><mxCell vertex="1"/></object>', "exactly one <mxCell>"),
    ('<object label="x"><mxCell vertex="1"/></object>', "needs an id"),
    ('<UserObject id="a"><mxCell/></UserObject>', "vertex"),
    ('<object id="a"><mxCell vertex="1"><evil/></mxCell></object>', "unexpected child"),
])
def test_upsert_rejects_bad_wrapper_fragments(xml, msg):
    new, res = apply_ops(_doc(), [UpsertXml(xml=xml)])
    assert new is None and msg in res[0]["error"]


def test_reorder_wrapped_cells():
    d = _apply([Reorder(id="e1", to="back")])
    assert _ids(d)[:3] == ["0", "1", "e1"]
    d = _apply([Reorder(id="zone", to="after", ref="plain")])
    ids = _ids(d)
    assert ids.index("zone") == ids.index("plain") + 1


def test_get_cells_xml_returns_wrapper_and_children_through_wrapped_parent():
    ctx = Ctx()
    ctx.state["drawio_xml"] = dd.serialize(_doc())
    r = tools.get_cells_xml(ctx, ["zone"], include_children=True)
    assert {c["id"] for c in r["cells"]} == {"zone", "api", "db"}
    api = next(c for c in r["cells"] if c["id"] == "api")["xml"]
    assert api.startswith("<object") and 'owner="team-payments"' in api and "customKeep=1" in api


def test_tools_end_to_end_on_wrapped_diagram_and_preview():
    ctx = Ctx()
    ctx.state["drawio_xml"] = dd.serialize(_doc())
    res = tools.edit_diagram(ctx, [
        {"op": "update_label", "id": "api", "label": "Gateway"},
        {"op": "add_component", "id": "bg", "label": "", "fit_to": ["zone"], "padding": 16},
        {"op": "set_attrs", "id": "bg", "attrs": {"style": "fillColor=#FF9900;fillOpacity=25;"}},
        {"op": "reorder", "id": "bg", "to": "back"},
    ])
    assert res["applied"], res
    assert tools.validate_drawio(ctx)["valid"]
    assert tools.describe_diagram(ctx)["vertices"][0]["id"] == "bg"
    p = tools.preview_diagram(ctx)
    assert p["node_count"] == 5 and p["edge_count"] == 2 and p["image"].inline_data.data[:4] == b"\x89PNG"


def test_references_handle_wrapped_cells(tmp_path):
    (tmp_path / "wrapped.drawio").write_text(WRAPPED, encoding="utf-8")
    info = refs.list_references(tmp_path)[0]
    assert info["zones"] == ["Backend"] and info["nodes"] == 4 and info["edges"] == 2
    assert "Orders API" in info["sample_labels"]
    style, w, h = refs.resolve_ref_cell(refs.RefCell(reference="wrapped.drawio", cell="api"), tmp_path)
    assert style.startswith("rounded=1") and (w, h) == (120.0, 60.0)
    profile = refs.derive_profile(refs.load_reference("wrapped.drawio", tmp_path))
    assert profile["zones"] == 1 and any(k["example_cell"] == "api" for k in profile["node_kinds"])
