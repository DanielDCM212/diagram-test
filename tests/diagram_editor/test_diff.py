import json

import pytest
from google.genai import types

from adk_agent.diagram_editor import diff as df
from adk_agent.diagram_editor import drawio_doc as dd
from adk_agent.diagram_editor import references as refs
from adk_agent.diagram_editor import tools
from adk_agent.diagram_editor.edit_ops import apply_ops
from adk_agent.diagram_editor.schema import (
    AddComponent, AddConnection, Delete, Move, Reorder, Resize, ScaleFonts, SetAttrs, SetGeometry, SetStyle,
    UpdateLabel,
)
from tests.diagram_editor.test_drawio_doc import PLAIN
from tests.diagram_editor.test_layout import _spec, refdir  # noqa: F401 (fixture)
from tests.diagram_editor.test_tools import Ctx, _request

STYLE = "rounded=0;whiteSpace=wrap;html=1;fontSize=12;"


def N(i, label, x, y, w, h, parent="1", style=STYLE):
    return (f'<mxCell id="{i}" value="{label}" style="{style}" vertex="1" parent="{parent}">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def D(*cells):
    return dd.load('<mxfile><diagram name="T"><mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                   + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")


EDGE = ('<mxCell id="e1" value="reads" style="endArrow=block;dashed=1;" edge="1" parent="1" source="a" target="b">'
        '<mxGeometry relative="1" as="geometry"/></mxCell>')


def base():  # flat: zone is just a bigger box around a and b
    return D(N("zone", "Zone", 0, 0, 300, 200), N("a", "API", 20, 40, 100, 50), N("b", "DB", 160, 40, 100, 50),
             N("n", "Other", 400, 50, 100, 50), EDGE)


def run(*ops, doc=None):
    before = doc or base()
    after, res = apply_ops(before, list(ops))
    assert after is not None, res
    return df.diff_docs(before, after)


def test_no_changes_and_baseline_none():
    r = df.diff_docs(base(), base())
    assert r["summary"] == "No changes. 5 cells unchanged." and r["lines"] == []
    assert all(v == 0 for k, v in r["counts"].items() if k != "unchanged")
    assert df.diff_docs(None, base())["summary"] == "New diagram: 4 nodes, 1 edge."


def test_rename_add_remove():
    r = run(UpdateLabel(id="a", label="Gateway"))
    assert r["summary"] == "Renamed 1. 4 cells unchanged."
    assert r["details"]["renamed"] == [{"id": "a", "from": "API", "to": "Gateway"}]
    assert r["lines"] == ['renamed a: "API" -> "Gateway"']

    r = run(AddComponent(id="c", label="Cache", x=500, y=300, like="a"), AddConnection(source="a", target="c", label="warm"))
    assert r["summary"] == "Added 2 (1 node, 1 edge). 5 cells unchanged."
    assert 'added node c "Cache"' in r["lines"] and 'added edge e_a_c "warm" (a -> c)' in r["lines"]

    r = run(Delete(id="b"))  # takes the edge with it
    assert r["summary"] == "Removed 2 (1 node, 1 edge). 3 cells unchanged."
    assert 'removed node b "DB"' in r["lines"]


def test_move_resize_restyle_and_tolerance():
    r = run(Move(id="n", x=420))
    assert r["summary"] == "Moved 1. 4 cells unchanged." and r["details"]["moved"] == [{"id": "n", "dx": 20.0, "dy": 0.0}]
    assert r["lines"] == ["moved n by (+20, 0)"]

    r = run(Resize(id="a", w=150))
    assert r["summary"] == "Resized 1. 4 cells unchanged." and r["details"]["resized"][0]["to"] == [150.0, 50.0]
    assert r["lines"] == ["resized a: 100x50 -> 150x50"]

    r = run(SetStyle(id="a", set={"fillColor": "#FF0000"}, remove=["fontSize"]))
    assert r["details"]["restyled"][0]["changes"] == {"fillColor": [None, "#FF0000"], "fontSize": ["12", None]}
    assert r["lines"] == ["restyled a: fillColor None -> #FF0000, fontSize 12 -> None"]

    r = run(SetGeometry(id="a", x=20.3))  # rounding noise
    assert r["summary"] == "No changes. 5 cells unchanged."


def test_label_markup_only_change_is_restyle_not_rename():
    r = run(SetAttrs(id="a", attrs={"value": "<b>API</b>"}))
    assert r["counts"]["renamed"] == 0 and r["counts"]["restyled"] == 1
    assert r["details"]["restyled"][0]["changes"] == {"label_formatting": [None, None]}


def test_carried_with_flat_zone_vs_independent_move():
    r = run(Move(id="zone", x=30), Move(id="a", x=50), Move(id="b", x=190))
    assert r["summary"] == "Moved 1 (+2 carried along with their container). 2 cells unchanged."
    assert [m["id"] for m in r["details"]["moved"]] == ["zone"] and {c["id"] for c in r["details"]["carried"]} == {"a", "b"}
    # same zone move, but `a` also went somewhere else on its own: that one is a real move
    r = run(Move(id="zone", x=30), Move(id="a", x=60, y=100), Move(id="b", x=190))
    assert {m["id"] for m in r["details"]["moved"]} == {"zone", "a"} and [c["id"] for c in r["details"]["carried"]] == ["b"]


def test_carried_with_explicit_parent():
    doc = D(N("z", "Zone", 100, 100, 300, 200), N("c1", "A", 10, 40, 100, 50, parent="z"), N("c2", "B", 150, 40, 100, 50, parent="z"))
    r = run(Move(id="z", x=150), doc=doc)
    assert r["summary"] == "Moved 1 (+2 carried along with their container). 0 cells unchanged."
    assert [m["id"] for m in r["details"]["moved"]] == ["z"]
    # moving one child inside the zone is a real move of that child
    r = run(Move(id="c1", x=130, y=150), doc=doc)
    assert [m["id"] for m in r["details"]["moved"]] == ["c1"] and r["counts"]["carried"] == 0


def test_reparent_reconnect_props_reorder():
    doc = D(N("z", "Zone", 100, 100, 300, 200), N("c1", "A", 10, 40, 100, 50, parent="z"), N("c2", "B", 150, 40, 100, 50, parent="z"),
            '<mxCell id="e" value="" style="endArrow=block;" edge="1" parent="1" source="c1" target="c2"><mxGeometry relative="1" as="geometry"/></mxCell>')
    r = run(SetAttrs(id="c1", attrs={"parent": "1"}), doc=doc)
    assert r["details"]["reparented"] == [{"id": "c1", "from": "z", "to": "1"}]
    assert "moved c1 from container z to 1" in r["lines"]

    r = run(SetAttrs(id="e", attrs={"target": "c1", "source": "c2"}), doc=doc)
    assert r["details"]["reconnected"] == [{"id": "e", "from": ["c1", "c2"], "to": ["c2", "c1"]}]

    wrapped = D('<object id="w" label="Box" owner="team-a" tier="1"><mxCell style="rounded=0;" vertex="1" parent="1">'
                '<mxGeometry x="0" y="0" width="80" height="40" as="geometry"/></mxCell></object>')
    r = run(SetAttrs(id="w", attrs={"owner": "team-b", "tier": None, "link": "https://x"}), doc=wrapped)
    assert r["details"]["properties"] == [{"id": "w", "keys": ["link", "owner", "tier"]}]
    assert r["counts"]["renamed"] == 0

    r = run(Reorder(id="a", to="back"))  # reports the cell that was moved, not those it jumped over
    assert [x["id"] for x in r["details"]["reordered"]] == ["a"] and r["lines"] == ["changed draw order of a"]
    r = run(Reorder(id="a", to="front"))
    assert [x["id"] for x in r["details"]["reordered"]] == ["a"]
    assert df.diff_docs(base(), apply_ops(base(), [Reorder(id="zone", to="back")])[0])["counts"]["reordered"] == 0


def test_scale_fonts_shows_as_restyle_of_labelled_cells():
    r = run(ScaleFonts(factor=0.5, min_size=6))
    assert r["counts"]["restyled"] == 5  # zone, a, b, n and the edge (its label "reads" is text too)
    assert {x["id"] for x in r["details"]["restyled"]} == {"zone", "a", "b", "n", "e1"}
    by = {x["id"]: x["changes"] for x in r["details"]["restyled"]}
    assert by["a"] == by["zone"] == by["b"] == by["n"] == {"fontSize": ["12", "6"]}  # explicit 12 before
    assert by["e1"] == {"fontSize": [None, "6"]}  # the edge had no explicit size


def test_multi_page_ids_are_tagged_and_added_pages_count():
    def two(extra=""):
        return dd.load('<mxfile><diagram name="Main"><mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                       + N("a", "A", 0, 0, 80, 40) + '</root></mxGraphModel></diagram><diagram name="Second"><mxGraphModel><root>'
                       '<mxCell id="0"/><mxCell id="1" parent="0"/>' + N("p", "P", 0, 0, 80, 40) + extra
                       + "</root></mxGraphModel></diagram></mxfile>")

    r = df.diff_docs(two(), two(N("q", "Q", 200, 0, 80, 40)))
    assert r["summary"] == "Added 1 (1 node, 0 edges). 2 cells unchanged." and r["details"]["added"][0]["id"] == "Second:q"
    one = dd.load('<mxfile><diagram name="Main"><mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                  + N("a", "A", 0, 0, 80, 40) + "</root></mxGraphModel></diagram></mxfile>")
    r = df.diff_docs(one, two())  # a whole page appeared
    assert r["counts"]["added_nodes"] == 1 and r["details"]["added"][0]["id"] == "Second:p"


def test_line_cap_and_exact_counts():
    ops = [AddComponent(id=f"x{i}", label=f"X{i}", x=500 + i * 10, y=300, w=5, h=5) for i in range(40)]
    r = run(*ops)
    assert r["counts"]["added"] == 40 and len(r["lines"]) == df.LINES_CAP + 1 and r["lines"][-1] == "... and 10 more"
    assert r["truncated"] is True and r["summary"].startswith("Added 40 (40 nodes, 0 edges).")


def test_summary_is_deterministic_and_wrapped_cells_work():
    a = run(Move(id="n", x=420), UpdateLabel(id="a", label="G"))
    b = run(Move(id="n", x=420), UpdateLabel(id="a", label="G"))
    assert a == b and a["summary"] == "Renamed 1; moved 1. 3 cells unchanged."
    wrapped = D('<object id="w" label="Box" owner="x"><mxCell style="rounded=0;" vertex="1" parent="1">'
                '<mxGeometry x="0" y="0" width="80" height="40" as="geometry"/></mxCell></object>')
    r = run(UpdateLabel(id="w", label="Renamed"), Resize(id="w", w=120), doc=wrapped)
    assert r["counts"]["renamed"] == 1 and r["counts"]["resized"] == 1 and r["counts"]["added"] == 0


# ---- through the tools, with ADK's real State ---------------------------------------------

@pytest.fixture(autouse=True)
def _refs(refdir, monkeypatch):  # noqa: F811
    monkeypatch.setattr(refs, "REFERENCE_DIR", refdir)


def _pasted(xml=PLAIN):
    ctx = Ctx()
    tools.capture_input_xml(ctx, _request(types.Part(text=f"please edit\n{xml}")))
    return ctx


def test_edit_results_carry_per_call_changes_and_session_total_accumulates():
    ctx = _pasted()
    assert ctx.state["change_summary"]["summary"] == "No changes. 4 cells unchanged."
    r1 = tools.edit_diagram(ctx, [{"op": "update_label", "id": "api", "label": "GW"}])
    assert r1["changes"]["summary"] == "Renamed 1. 3 cells unchanged."
    r2 = tools.edit_diagram(ctx, [{"op": "add_component", "label": "Cache", "parent": "zone", "like": "api"}])
    assert r2["changes"]["counts"]["added_nodes"] == 1 and "renamed" not in r2["changes"]["counts"]  # this call only
    total = tools.get_change_summary(ctx)
    assert total["since"] == "input" and total["counts"]["renamed"] == 1 and total["counts"]["added_nodes"] == 1
    assert total["summary"] == ctx.state["change_summary"]["summary"]  # what callers read from state
    assert 'renamed api: "API" -> "GW"' in total["lines"]
    detail = tools.get_change_summary(ctx, detail=True)
    assert detail["details"]["renamed"] == [{"id": "api", "from": "API", "to": "GW"}]
    json.dumps(ctx.state.to_dict())  # session state must stay JSON-serialisable


def test_new_paste_resets_the_baseline():
    ctx = _pasted()
    tools.edit_diagram(ctx, [{"op": "update_label", "id": "api", "label": "GW"}])
    assert tools.get_change_summary(ctx)["counts"]["renamed"] == 1
    tools.capture_input_xml(ctx, _request(types.Part(text=PLAIN.replace('value="API"', 'value="Other"'))))
    assert tools.get_change_summary(ctx)["summary"] == "No changes. 4 cells unchanged."


def test_create_mode_summary_is_a_new_diagram():
    ctx = Ctx()
    res = tools.create_diagram(ctx, _spec())
    n = len(tools.describe_diagram(ctx)["vertices"])
    assert res["changes"]["summary"] == f"New diagram: {n} nodes, 2 edges."
    assert tools.get_change_summary(ctx)["since"] == "empty"
    tools.edit_diagram(ctx, [{"op": "add_component", "label": "Queue", "parent": "core", "like": "svc2"}])
    assert tools.get_change_summary(ctx)["summary"] == f"New diagram: {n + 1} nodes, 2 edges."


def test_state_set_from_outside_gets_a_baseline_on_first_edit():
    ctx = Ctx()
    ctx.state["drawio_xml"] = dd.serialize(dd.load(PLAIN))  # no capture callback ran
    assert tools.get_change_summary(ctx)["since"] == "unknown"
    tools.edit_diagram(ctx, [{"op": "update_label", "id": "api", "label": "GW"}])
    assert tools.get_change_summary(ctx)["summary"] == "Renamed 1. 3 cells unchanged."
    assert "error" in tools.get_change_summary(Ctx())


def test_resolve_overlaps_reports_its_changes_through_the_same_summary():
    ctx = Ctx()
    ctx.state["drawio_xml"] = dd.serialize(D(N("a", "A", 0, 0, 100, 60), N("b", "B", 80, 20, 100, 60), N("c", "C", 500, 0, 50, 50)))
    res = tools.resolve_overlaps(ctx)
    assert res["saved"] and res["changes"]["counts"]["moved"] == 2
    assert "Moved 2" in res["changes"]["summary"] and res["changes"]["summary"].endswith("1 cells unchanged.")
    assert tools.get_change_summary(ctx)["counts"]["moved"] == 2
    dry = tools.resolve_overlaps(ctx, dry_run=True)
    assert "changes" not in dry  # nothing was saved, nothing to summarise
