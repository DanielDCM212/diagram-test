import time

import pytest

from adk_agent.diagram_editor import drawio_doc as dd
from adk_agent.diagram_editor import overlaps as ov
from adk_agent.diagram_editor import text_fit, tools
from adk_agent.diagram_editor.edit_ops import apply_ops
from adk_agent.diagram_editor.schema import ScaleFonts
from tests.diagram_editor.test_tools import Ctx

STYLE = "rounded=0;whiteSpace=wrap;html=1;fontSize=12;"


def node(i, x, y, w, h, label="", style=STYLE, parent="1"):
    return (f'<mxCell id="{i}" value="{label}" style="{style}" vertex="1" parent="{parent}">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def diagram(*cells):
    return dd.load('<mxfile><diagram name="T"><mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                   + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")


def box(doc, cid):
    v = next(v for v in dd.summarize(doc)["vertices"] if v["id"] == cid)
    return v["x"], v["y"], v["w"], v["h"]


def resolve(doc, **kw):
    return ov.resolve_overlaps(doc, 0, ov.Params(**kw))


# ---- detection ---------------------------------------------------------------------

def test_nesting_and_touching_are_not_overlaps():
    doc = diagram(
        node("zone", 0, 0, 400, 300, "Zone", parent="1"),
        node("in_geo", 20, 40, 100, 50, "geometrically inside"),            # flat zone: fully inside by position
        node("lbl", 30, 50, 60, 20, "label on a box", style="text;html=1;"),  # label sitting on a box
        node("c", 200, 40, 100, 50, "explicit child", parent="zone"),
        node("touch_a", 500, 0, 100, 60),
        node("touch_b", 600, 0, 100, 60),                                    # shares only an edge
    )
    assert ov.find_overlaps(doc)["overlap_count"] == 0


def test_partial_overlap_identical_boxes_and_severity():
    doc = diagram(node("a", 0, 0, 100, 60), node("b", 80, 20, 100, 60), node("c", 500, 0, 50, 50), node("d", 500, 0, 50, 50))
    r = ov.find_overlaps(doc)
    by = {(o["a"], o["b"]): o for o in r["overlaps"]}
    assert set(by) == {("a", "b"), ("c", "d")}
    assert by[("c", "d")]["severity"] == 1.0 and by[("a", "b")]["severity"] == round(20 * 40 / (100 * 60), 2)
    assert r["overlaps"][0]["a"] == "c"  # sorted worst first


def test_edge_label_vertices_are_ignored():
    edge = ('<mxCell id="e" value="" style="endArrow=block;" edge="1" parent="1" source="a" target="b">'
            '<mxGeometry relative="1" as="geometry"/></mxCell>')
    lbl = node("elbl", 0, 0, 40, 20, "x", parent="e")  # edge-label vertex: coordinates relative to the edge
    doc = diagram(node("a", 0, 0, 100, 60), node("b", 300, 0, 100, 60), edge, lbl)
    assert ov.find_overlaps(doc)["overlap_count"] == 0
    new, rep = resolve(doc)
    assert dd.serialize(new) == dd.serialize(doc) and rep["overlaps_before"] == 0


def test_overflow_and_outside_container_reported():
    doc = diagram(node("z", 0, 0, 100, 100), node("kid", 80, 80, 60, 60, parent="z"),
                  node("t", 300, 0, 60, 24, "A very long label that cannot fit in here"))
    r = ov.find_overlaps(doc)
    assert [o["id"] for o in r["label_overflow"]] == ["t"] and r["label_overflow"][0]["suggested_font"] < 12
    assert r["outside_container"] == [{"id": "kid", "container": "z"}]


# ---- move ------------------------------------------------------------------------

def test_move_separates_pair_with_gap_and_leaves_input_untouched():
    doc = diagram(node("a", 0, 0, 100, 60), node("b", 80, 20, 100, 60))
    before = dd.serialize(doc)
    new, rep = resolve(doc, gap=10)
    assert dd.serialize(doc) == before
    assert rep["overlaps_before"] == 1 and rep["overlaps_after"] == 0 and not rep["resized"]
    a, b = box(new, "a"), box(new, "b")
    assert b[0] - (a[0] + a[2]) == pytest.approx(10, abs=0.2)
    assert {m["id"] for m in rep["moved"]} == {"a", "b"}
    assert (a[2], a[3], b[2], b[3]) == (100, 60, 100, 60)  # moving never resizes


def test_lock_and_only_restrict_what_may_change():
    doc = diagram(node("a", 0, 0, 100, 60), node("b", 80, 20, 100, 60))
    new, rep = resolve(doc, lock=frozenset({"a"}))
    assert box(new, "a") == (0, 0, 100, 60) and rep["overlaps_after"] == 0
    assert box(new, "b")[0] == pytest.approx(110, abs=0.2)  # b took the whole push
    new, rep = resolve(doc, only=frozenset({"b"}))
    assert box(new, "a") == (0, 0, 100, 60) and rep["overlaps_after"] == 0
    new, rep = resolve(doc, lock=frozenset({"a", "b"}))
    assert rep["overlaps_after"] == 1 and rep["moved"] == []
    with pytest.raises(ValueError, match="unknown node"):
        resolve(doc, lock=frozenset({"nope"}))


def test_dense_grid_resolves_within_shift_budget():
    cells = [node(f"n{r}{c}", c * 70, r * 40, 100, 60, f"N{r}{c}") for r in range(3) for c in range(3)]
    doc = diagram(*cells)
    new, rep = resolve(doc, max_shift=200, allow_resize=False, allow_font_shrink=False)
    assert rep["overlaps_before"] > 0 and rep["overlaps_after"] == 0, rep["remaining"]
    assert all(abs(m["dx"]) + abs(m["dy"]) <= 200.01 for m in rep["moved"])
    assert ov.find_overlaps(new)["overlap_count"] == 0


def test_max_shift_is_a_hard_limit_and_leftovers_are_reported_not_forced():
    doc = diagram(node("a", 0, 0, 100, 60), node("b", 50, 0, 100, 60))
    new, rep = resolve(doc, max_shift=5, allow_resize=False, allow_font_shrink=False)
    assert rep["overlaps_after"] == 1 and rep["remaining"][0]["overlap_w"] == pytest.approx(40, abs=0.2)
    assert all(abs(m["dx"]) + abs(m["dy"]) <= 5.01 for m in rep["moved"])


# ---- shrink + font -----------------------------------------------------------------

def _threshold_width(label, h=24):
    style = dd.parse_style(STYLE)
    for w in range(60, 260):
        if text_fit.fits(label, style, w, h, 12) and not text_fit.fits(label, style, w - 10, h, 12):
            return w
    pytest.skip("no width threshold found for this Pillow font")


def test_shrink_then_font_when_moving_is_not_allowed():
    label = "Payment gateway service"
    w0 = _threshold_width(label)
    doc = diagram(node("a", 0, 0, w0, 24, label), node("b", w0 - 10, 0, w0, 24, "Other"))
    new, rep = resolve(doc, allow_move=False)
    assert rep["overlaps_after"] == 0 and rep["moved"] == []
    a = box(new, "a")
    assert a[2] < w0 and a[2] >= max(ov.MIN_W, w0 * 0.75)  # shrank, within the floor
    changed = {f["id"]: f for f in rep["fonts_changed"]}
    assert "a" in changed and 8 <= changed["a"]["to"] < 12 == changed["a"]["from"]
    assert "b" not in changed  # its short label still fits
    style = dd.parse_style(dd.attr(new.cell("a"), "style"))
    assert text_fit.fits(label, style, a[2], a[3], int(style["fontSize"]))


def test_shrink_respects_min_scale_and_reports_what_remains():
    doc = diagram(node("a", 0, 0, 100, 60), node("b", 20, 0, 100, 60))
    new, rep = resolve(doc, allow_move=False, min_scale=0.95)
    assert rep["overlaps_after"] == 1  # needs ~22px of shrink, only ~10 allowed
    for cid in ("a", "b"):
        assert box(new, cid)[2] == 100  # untouched: a shrink that can't resolve the pair isn't applied
    new, rep = resolve(doc, allow_move=False, allow_resize=False)
    assert rep["overlaps_after"] == 1 and rep["resized"] == []


def test_font_never_grows_and_min_font_is_a_floor():
    label = "Payment gateway service"
    w0 = _threshold_width(label)
    doc = diagram(node("a", 0, 0, w0, 24, label), node("b", w0 - 10, 0, w0, 24, "Other"))
    new, rep = resolve(doc, allow_move=False, min_font=12)
    assert rep["fonts_changed"] == []
    assert [f["id"] for f in rep["labels_still_overflowing"]] == ["a"]
    new, rep = resolve(doc, allow_move=False, allow_font_shrink=False)
    assert rep["fonts_changed"] == [] and dd.parse_style(dd.attr(new.cell("a"), "style"))["fontSize"] == "12"


def test_fix_overflow_option():
    style = "rounded=0;whiteSpace=wrap;html=1;fontSize=14;"
    doc = diagram(node("t", 0, 0, 80, 24, "A long label that overflows its box", style=style))
    _, rep = resolve(doc)
    assert rep["fonts_changed"] == []  # no overlap involved -> untouched by default
    new, rep = resolve(doc, fix_overflow=True)
    assert rep["fonts_changed"] and rep["fonts_changed"][0]["to"] < 14


# ---- containers, flat zones, wrapped cells -------------------------------------------

def test_container_grows_and_children_stay_inside():
    z = node("z", 100, 100, 200, 100, "Zone")
    doc = diagram(z, node("c1", 10, 40, 120, 60, "A", parent="z"), node("c2", 50, 60, 120, 60, "B", parent="z"))
    new, rep = resolve(doc, allow_resize=False, allow_font_shrink=False)
    assert rep["overlaps_after"] == 0 and "z" in rep["containers_grown"]
    assert ov.find_overlaps(new)["outside_container"] == []
    assert dd.validate(new)["valid"]
    for cid in ("c1", "c2"):
        assert new.cell(cid).find("mxGeometry") is not None and dd.attr(new.cell(cid), "parent") == "z"


def test_flat_zone_carries_its_contents_when_it_is_pushed():
    doc = diagram(
        node("z", 0, 0, 300, 200, "Zone"),
        node("c1", 20, 40, 100, 50, "A"), node("c2", 160, 40, 100, 50, "B"),  # inside the zone by position only
        node("n", 280, 60, 100, 60, "N"),
    )
    new, rep = resolve(doc, lock=frozenset({"n"}))
    assert rep["overlaps_after"] == 0
    z, c1, c2 = box(new, "z"), box(new, "c1"), box(new, "c2")
    assert z[0] < 0 and (c1[0] - z[0], c1[1] - z[1]) == (20, 40) and (c2[0] - z[0], c2[1] - z[1]) == (160, 40)
    assert box(new, "n") == (280, 60, 100, 60)
    assert [m["id"] for m in rep["moved"]] == ["z"]  # contents moved as part of the zone, not on their own


def test_wrapped_cells_are_resolved_and_keep_their_data():
    wrapped = ('<object id="w1" label="One" owner="team-a"><mxCell style="rounded=0;" vertex="1" parent="1">'
               '<mxGeometry x="0" y="0" width="100" height="60" as="geometry"/></mxCell></object>'
               '<UserObject id="w2" label="Two" tooltip="t"><mxCell style="rounded=0;" vertex="1" parent="1">'
               '<mxGeometry x="60" y="10" width="100" height="60" as="geometry"/></mxCell></UserObject>')
    doc = dd.load('<mxfile><diagram name="T"><mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                  + wrapped + "</root></mxGraphModel></diagram></mxfile>")
    new, rep = resolve(doc)
    assert rep["overlaps_before"] == 1 and rep["overlaps_after"] == 0
    assert new.cell("w1").get("owner") == "team-a" and new.cell("w2").get("tooltip") == "t"
    assert new.cell("w2").find("mxGeometry") is None and new.cell("w2").find("mxCell/mxGeometry") is not None
    assert dd.validate(new)["valid"]


def test_free_edges_trigger_a_warning_when_things_move():
    edge = ('<mxCell id="e" value="" style="endArrow=block;" edge="1" parent="1" source="a">'
            '<mxGeometry relative="1" as="geometry"><mxPoint x="500" y="50" as="targetPoint"/></mxGeometry></mxCell>')
    doc = diagram(node("a", 0, 0, 100, 60), node("b", 80, 20, 100, 60), edge)
    _, rep = resolve(doc)
    assert "free end" in rep["warning"]


# ---- the tools ---------------------------------------------------------------------

def _ctx(doc):
    c = Ctx()
    c.state["drawio_xml"] = dd.serialize(doc)
    return c


def test_tools_find_dry_run_resolve_and_errors():
    ctx = _ctx(diagram(node("a", 0, 0, 100, 60), node("b", 80, 20, 100, 60)))
    before = ctx.state["drawio_xml"]
    assert tools.find_overlaps(ctx)["overlap_count"] == 1
    dry = tools.resolve_overlaps(ctx, dry_run=True)
    assert dry["dry_run"] and dry["saved"] is False and dry["overlaps_after"] == 0 and ctx.state["drawio_xml"] == before
    res = tools.resolve_overlaps(ctx, lock=["a"], max_shift="60", min_scale="0.8", min_font="8")  # stringified numbers
    assert res["saved"] is True and tools.find_overlaps(ctx)["overlap_count"] == 0
    assert tools.validate_drawio(ctx)["valid"]
    bad = tools.resolve_overlaps(ctx, lock=["ghost"])
    assert bad["saved"] is False and "ghost" in bad["error"]
    assert "error" in tools.find_overlaps(Ctx()) and "error" in tools.resolve_overlaps(Ctx())
    assert "out of range" in tools.find_overlaps(ctx, page=3)["error"]


def test_tool_limits_leave_leftovers_in_the_report():
    ctx = _ctx(diagram(node("a", 0, 0, 100, 60), node("b", 50, 0, 100, 60)))
    res = tools.resolve_overlaps(ctx, max_shift=5, allow_resize=False)
    assert res["saved"] is True and res["overlaps_after"] == 1 and res["remaining"]


def test_performance_on_a_dense_diagram():
    cells = [node(f"n{r}_{c}", c * 80, r * 45, 100, 60, f"N{r}{c}") for r in range(12) for c in range(12)]
    doc = diagram(*cells)
    t0 = time.time()
    new, rep = resolve(doc, max_shift=400)
    assert time.time() - t0 < 20
    assert rep["overlaps_after"] <= rep["overlaps_before"]
    assert ov.find_overlaps(new)["overlap_count"] == rep["overlaps_after"]


# ---- scale_fonts op ------------------------------------------------------------------

def test_scale_fonts_op():
    doc = diagram(node("a", 0, 0, 100, 60, "A", style="rounded=0;"),               # no fontSize -> default 12
                  node("b", 200, 0, 100, 60, "B", style="rounded=0;fontSize=20;"),
                  node("c", 400, 0, 100, 60, "", style="rounded=0;"))               # no label
    new, res = apply_ops(doc, [ScaleFonts(factor=0.9)])
    assert res[0]["ok"]
    fs = lambda i: dd.parse_style(dd.attr(new.cell(i), "style")).get("fontSize")  # noqa: E731
    assert (fs("a"), fs("b"), fs("c")) == ("11", "18", None)
    new, _ = apply_ops(doc, [ScaleFonts(factor=0.5, ids=["b"], min_size=12)])
    assert dd.parse_style(dd.attr(new.cell("b"), "style"))["fontSize"] == "12" and "fontSize" not in dd.attr(new.cell("a"), "style")
    new, _ = apply_ops(doc, [ScaleFonts(factor=3, ids=["b"], max_size=30)])
    assert dd.parse_style(dd.attr(new.cell("b"), "style"))["fontSize"] == "30"
    assert apply_ops(doc, [ScaleFonts(factor=0.9, ids=["ghost"])])[0] is None
