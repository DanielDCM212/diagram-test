import pytest

from adk_agent.diagram_editor import drawio_doc as dd

PLAIN = """<mxfile host="app.diagrams.net"><diagram id="p1" name="Main"><mxGraphModel><root>
<mxCell id="0"/><mxCell id="1" parent="0"/>
<mxCell id="zone" value="&lt;b&gt;Backend&lt;/b&gt;&lt;br&gt;tier" style="rounded=0;fillColor=#EEEEEE;" vertex="1" parent="1">
  <mxGeometry x="100" y="50" width="400" height="200" as="geometry"/></mxCell>
<mxCell id="api" value="API" style="shape=rect;fillColor=#FFFFFF;fontSize=12;customKeep=1;" vertex="1" parent="zone">
  <mxGeometry x="20" y="30" width="120" height="60" as="geometry"/></mxCell>
<mxCell id="db" value="DB" style="shape=cylinder3;" vertex="1" parent="zone">
  <mxGeometry x="200" y="30" width="80" height="60" as="geometry"/></mxCell>
<mxCell id="e1" value="reads" style="dashed=1;" edge="1" parent="1" source="api" target="db">
  <mxGeometry relative="1" as="geometry"/></mxCell>
</root></mxGraphModel></diagram></mxfile>"""


def test_load_plain_and_summarize():
    doc = dd.load(PLAIN)
    s = dd.summarize(doc)
    by_id = {v["id"]: v for v in s["vertices"]}
    assert set(by_id) == {"zone", "api", "db"}
    assert by_id["zone"]["label"] == "Backend tier"
    assert by_id["zone"]["container"] is True
    # child geometry is absolute: zone (100,50) + local (20,30)
    assert (by_id["api"]["x"], by_id["api"]["y"]) == (120.0, 80.0)
    assert by_id["api"]["parent"] == "zone"
    assert s["edges"] == [{"id": "e1", "source": "api", "target": "db", "label": "reads", "dashed": True}]


def test_roundtrip_preserves_unknown_style_keys():
    out = dd.serialize(dd.load(PLAIN))
    assert "customKeep=1" in out
    assert dd.summarize(dd.load(out)) == dd.summarize(dd.load(PLAIN))


def test_load_compressed_diagram():
    model = dd.load(PLAIN).model()
    wrapped = f'<mxfile><diagram name="Packed">{dd.compress_model(model)}</diagram></mxfile>'
    doc = dd.load(wrapped)
    assert {v["id"] for v in dd.summarize(doc)["vertices"]} == {"zone", "api", "db"}
    # serialising expands it: output is readable without decompression
    assert "<mxGraphModel" in dd.serialize(doc)


def test_load_bare_model():
    inner = dd.serialize(dd.load(PLAIN))
    bare = inner[inner.index("<mxGraphModel"):inner.index("</mxGraphModel>") + len("</mxGraphModel>")]
    assert len(dd.load(bare).pages) == 1


@pytest.mark.parametrize("bad", ["", "   ", "<not-closed", "<html/>", "<mxfile/>",
                                 '<mxfile><diagram name="x">@@notbase64@@</diagram></mxfile>'])
def test_load_rejects_garbage(bad):
    with pytest.raises(dd.DrawioParseError):
        dd.load(bad)


def test_load_rejects_entity_bomb():
    bomb = ('<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;">]>'
            "<mxfile>&b;</mxfile>")
    with pytest.raises(dd.DrawioParseError):
        dd.load(bomb)


def test_validate_reports_dangling_and_duplicates():
    doc = dd.load(PLAIN)
    assert dd.validate(doc)["valid"]
    doc.cell("e1").set("target", "ghost")
    dup = dd.load(PLAIN)
    dup.root().append(dup.cell("db"))  # same element appended again -> duplicate id
    assert {p["problem"] for p in dd.validate(doc)["problems"]} == {"dangling target"}
    assert {p["problem"] for p in dd.validate(dup)["problems"]} == {"duplicate id"}


def test_style_parse_format_roundtrip():
    s = "rounded=1;whiteSpace=wrap;text;fillColor=#fff;"
    assert dd.format_style(dd.parse_style(s)) == s
