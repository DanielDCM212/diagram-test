import pytest

from adk_agent.diagram_editor import drawio_doc as dd
from adk_agent.diagram_editor import references as refs
from adk_agent.diagram_editor.layout import LayoutError, build_diagram
from adk_agent.diagram_editor.schema import ArchitectureSpec, RefCell

ZONE = "rounded=0;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor=#666666;verticalAlign=top;"
SVC = "rounded=1;whiteSpace=wrap;html=1;fillColor=#DAE8FC;strokeColor=#6C8EBF;fontSize=12;"
DB = "shape=cylinder3;whiteSpace=wrap;html=1;fillColor=#D5E8D4;strokeColor=#82B366;fontSize=12;"


def _v(i, label, style, x, y, w, h):
    return (f'<mxCell id="{i}" value="{label}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def _reference_xml():
    cells = [_v("title", "Platform", "text;html=1;fontSize=16;fontStyle=1;", 20, 10, 300, 30)]
    for n, name in enumerate(["Edge", "Core", "Data"]):
        zx = 20 + n * 240
        cells.append(_v(f"z{n}", name, ZONE, zx, 60, 220, 300))
        cells.append(_v(f"a{n}", f"{name} A", SVC if n < 2 else DB, zx + 20, 100, 180, 60))
        cells.append(_v(f"b{n}", f"{name} B", SVC if n < 2 else DB, zx + 20, 180, 180, 60))
    cells.append('<mxCell id="ea" value="" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;" edge="1" parent="1" source="a0" target="a1"><mxGeometry relative="1" as="geometry"/></mxCell>')
    cells.append('<mxCell id="eb" value="" style="edgeStyle=orthogonalEdgeStyle;endArrow=block;" edge="1" parent="1" source="a1" target="a2"><mxGeometry relative="1" as="geometry"/></mxCell>')
    cells.append('<mxCell id="ec" value="" style="edgeStyle=orthogonalEdgeStyle;endArrow=open;dashed=1;" edge="1" parent="1" source="b1" target="b2"><mxGeometry relative="1" as="geometry"/></mxCell>')
    return ('<mxfile><diagram name="Ref"><mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
            + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")


@pytest.fixture()
def refdir(tmp_path):
    (tmp_path / "platform.drawio").write_text(_reference_xml(), encoding="utf-8")
    (tmp_path / "broken.drawio").write_text("<nope", encoding="utf-8")
    return tmp_path


def test_list_references_reports_good_and_broken(refdir):
    by = {r["name"]: r for r in refs.list_references(refdir)}
    assert by["platform.drawio"]["zones"] == ["Edge", "Core", "Data"]
    assert by["platform.drawio"]["nodes"] == 10 and by["platform.drawio"]["edges"] == 3
    assert "error" in by["broken.drawio"]


def test_reference_name_cannot_escape_directory(refdir):
    for bad in ("../x.drawio", "sub/x.drawio", "..", "platform.txt", ""):
        with pytest.raises(refs.ReferenceLookupError):
            refs.load_reference(bad, refdir)


def test_derive_profile_finds_flat_zones_and_metrics(refdir):
    p = refs.derive_profile(refs.load_reference("platform.drawio", refdir))
    L = p["layout"]
    assert p["zones"] == 3 and p["top_level_zones"] == 3
    assert L["zone_arrangement"] == "columns"
    assert L["zone_gap"] == 20
    assert (L["zone_padding_left"], L["zone_padding_top"]) == (20, 40)
    assert L["component_gap"] == 20 and L["components_per_row"] == 1 and L["grid"] == 10
    assert L["title"]["example_cell"] == "title"
    assert {k["count"] for k in p["node_kinds"]} == {4, 2}
    assert p["edge_kinds"][0]["count"] == 2 and p["edge_kinds"][1]["dashed"] is True


def _spec(**over):
    base = dict(
        title="Orders",
        layout_from="platform.drawio",
        zones=[
            {"id": "edge", "label": "Edge", "like": {"reference": "platform.drawio", "cell": "z0"}},
            {"id": "core", "label": "Core", "like": {"reference": "platform.drawio", "cell": "z1"}},
            {"id": "data", "label": "Data", "like": {"reference": "platform.drawio", "cell": "z2"}},
        ],
        components=[
            {"id": "web", "label": "Web", "zone": "edge", "like": {"reference": "platform.drawio", "cell": "a0"}},
            {"id": "api", "label": "API", "zone": "core", "like": {"reference": "platform.drawio", "cell": "a1"}},
            {"id": "svc2", "label": "Billing", "zone": "core", "like": {"reference": "platform.drawio", "cell": "b1"}},
            {"id": "svc3", "label": "Stock", "zone": "core", "like": {"reference": "platform.drawio", "cell": "b1"}},
            {"id": "pg", "label": "Postgres", "zone": "data", "like": {"reference": "platform.drawio", "cell": "a2"}},
        ],
        connections=[
            {"source": "web", "target": "api", "label": "https"},
            {"source": "api", "target": "pg", "like": {"reference": "platform.drawio", "cell": "ec"}},
        ],
    )
    base.update(over)
    return ArchitectureSpec.model_validate(base)


def _boxes(doc):
    return {v["id"]: v for v in dd.summarize(doc)["vertices"]}


def test_build_diagram_follows_reference_layout(refdir):
    doc, stats = build_diagram(_spec(), refdir)
    assert dd.validate(doc)["valid"] and stats["arrangement"] == "columns"
    b = _boxes(doc)
    # zones side by side, left to right in spec order, separated by the reference gap
    assert b["edge"]["x"] < b["core"]["x"] < b["data"]["x"]
    assert b["core"]["x"] - (b["edge"]["x"] + b["edge"]["w"]) == 20
    assert b["edge"]["h"] == b["core"]["h"] == b["data"]["h"]  # equalised cross-axis
    # title sits above every zone
    assert b["title"]["y"] + b["title"]["h"] <= min(b[z]["y"] for z in ("edge", "core", "data"))
    # components inside their zone, padded like the reference, stacked one per row with the reference gap
    for cid, zid in (("web", "edge"), ("api", "core"), ("svc2", "core"), ("svc3", "core"), ("pg", "data")):
        c, z = b[cid], b[zid]
        assert c["parent"] == zid
        assert z["x"] <= c["x"] and c["x"] + c["w"] <= z["x"] + z["w"]
        assert z["y"] <= c["y"] and c["y"] + c["h"] <= z["y"] + z["h"]
    assert (b["api"]["x"] - b["core"]["x"], b["api"]["y"] - b["core"]["y"]) == (20, 40)
    assert b["svc2"]["y"] - (b["api"]["y"] + b["api"]["h"]) == 20
    assert b["svc3"]["y"] > b["svc2"]["y"] and b["svc3"]["x"] == b["svc2"]["x"]
    # cloned style + size, not default
    assert b["api"]["style"]["fillColor"] == "#DAE8FC" and (b["api"]["w"], b["api"]["h"]) == (180, 60)
    assert b["pg"]["style"]["shape"] == "cylinder3"


def test_build_diagram_edges(refdir):
    doc, _ = build_diagram(_spec(), refdir)
    edges = {(e["source"], e["target"]): e for e in dd.summarize(doc)["edges"]}
    assert edges[("web", "api")]["label"] == "https" and edges[("web", "api")]["dashed"] is False
    assert edges[("api", "pg")]["dashed"] is True  # cloned from the reference's dashed edge


def test_build_diagram_is_deterministic(refdir):
    a, _ = build_diagram(_spec(), refdir)
    b, _ = build_diagram(_spec(), refdir)
    assert dd.summarize(a) == dd.summarize(b)


def test_nested_zone_and_free_component(refdir):
    spec = _spec(
        zones=_spec().model_dump()["zones"] + [
            {"id": "cache_tier", "label": "Cache", "parent": "core",
             "like": {"reference": "platform.drawio", "cell": "z1"}}],
        components=_spec().model_dump()["components"] + [
            {"id": "redis", "label": "Redis", "zone": "cache_tier",
             "like": {"reference": "platform.drawio", "cell": "a2"}},
            {"id": "user", "label": "User", "like": {"reference": "platform.drawio", "cell": "a0"}}],
    )
    doc, _ = build_diagram(spec, refdir)
    b = _boxes(doc)
    assert dd.validate(doc)["valid"]
    inner, outer = b["cache_tier"], b["core"]
    assert inner["parent"] == "core"
    assert outer["x"] <= inner["x"] and inner["x"] + inner["w"] <= outer["x"] + outer["w"]
    assert outer["y"] <= inner["y"] and inner["y"] + inner["h"] <= outer["y"] + outer["h"]
    assert b["redis"]["parent"] == "cache_tier"
    assert "parent" not in b["user"]
    assert b["user"]["y"] >= b["core"]["y"] + b["core"]["h"]  # free strip sits after the zones


def test_bad_spec_reports_every_problem(refdir):
    spec = _spec(
        layout_from="missing.drawio",
        components=_spec().model_dump()["components"] + [
            {"id": "web", "label": "dup", "zone": "nowhere", "like": {"reference": "platform.drawio", "cell": "nope"}}],
        connections=[{"source": "web", "target": "ghost"}],
    )
    with pytest.raises(LayoutError) as exc:
        build_diagram(spec, refdir)
    text = str(exc.value)
    assert "duplicate id 'web'" in text and "unknown zone 'nowhere'" in text and "ghost" in text
    assert "missing.drawio" in text


def test_unknown_like_cell_is_reported(refdir):
    spec = _spec(components=[{"id": "x", "label": "X", "zone": "edge",
                              "like": {"reference": "platform.drawio", "cell": "nope"}}], connections=[])
    with pytest.raises(LayoutError) as exc:
        build_diagram(spec, refdir)
    assert "nope" in str(exc.value)
