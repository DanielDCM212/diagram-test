import pytest

from adk_agent.diagram_editor import drawio_doc as dd
from adk_agent.diagram_editor.edit_ops import apply_ops
from adk_agent.diagram_editor.schema import (
    AddComponent, AddConnection, Delete, Move, RefCell, Resize, SetStyle, UpdateLabel,
)
from tests.diagram_editor.test_drawio_doc import PLAIN


def _doc():
    return dd.load(PLAIN)


def _v(doc, cid):
    return next(v for v in dd.summarize(doc)["vertices"] if v["id"] == cid)


def _overlap(a, b):
    return a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"] and a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"]


def test_failure_is_atomic_and_input_untouched():
    doc = _doc()
    before = dd.serialize(doc)
    new, res = apply_ops(doc, [UpdateLabel(id="api", label="X"), Delete(id="nope")])
    assert new is None and res[-1]["ok"] is False and "nope" in res[-1]["error"]
    assert dd.serialize(doc) == before


def test_update_label_keeps_other_cells_identical():
    doc = _doc()
    new, _ = apply_ops(doc, [UpdateLabel(id="api", label="Gateway")])
    assert _v(new, "api")["label"] == "Gateway"
    assert "customKeep=1" in dd.serialize(new)
    assert dd.summarize(new)["edges"] == dd.summarize(doc)["edges"]


def test_delete_cascades_to_children_and_edges():
    new, _ = apply_ops(_doc(), [Delete(id="zone")])
    s = dd.summarize(new)
    assert s["vertices"] == [] and s["edges"] == []
    assert dd.validate(new)["valid"]


def test_delete_node_removes_its_edges_only():
    new, _ = apply_ops(_doc(), [Delete(id="db")])
    s = dd.summarize(new)
    assert {v["id"] for v in s["vertices"]} == {"zone", "api"} and s["edges"] == []


def test_add_component_clones_style_and_size_and_avoids_overlap():
    doc = _doc()
    new, res = apply_ops(doc, [AddComponent(label="Cache", parent="zone", like="api")])
    assert res[0]["ok"], res
    cache = _v(new, "cache")
    assert (cache["w"], cache["h"]) == (120.0, 60.0)
    assert "customKeep" not in cache.get("style", {}) and cache["style"]["fillColor"] == "#FFFFFF"
    assert cache["parent"] == "zone"
    for other in ("api", "db"):
        assert not _overlap(cache, _v(new, other))
    z, c = _v(new, "zone"), cache
    assert z["x"] <= c["x"] and c["x"] + c["w"] <= z["x"] + z["w"]
    assert z["y"] <= c["y"] and c["y"] + c["h"] <= z["y"] + z["h"]
    assert dd.validate(new)["valid"]


def test_add_component_near_prefers_right_then_below():
    # zone is 400 wide: right of db (x=380..500) would overflow it, so it goes below db
    new, _ = apply_ops(_doc(), [AddComponent(id="n", label="N", parent="zone", like="api", near="db")])
    db, n = _v(new, "db"), _v(new, "n")
    assert n["y"] >= db["y"] + db["h"] and not _overlap(n, db)
    # an unparented canvas has room on the right
    new, _ = apply_ops(_doc(), [AddComponent(id="m", label="M", like="api", near="zone", x=None)])
    z, m = _v(new, "zone"), _v(new, "m")
    assert not _overlap(z, m)


def test_add_component_grows_full_container():
    ops = [AddComponent(label=f"S{i}", parent="zone", like="api") for i in range(8)]
    new, res = apply_ops(_doc(), ops)
    assert all(r["ok"] for r in res)
    kids = [v for v in dd.summarize(new)["vertices"] if v.get("parent") == "zone"]
    assert len(kids) == 10
    for i, a in enumerate(kids):
        for b in kids[i + 1:]:
            assert not _overlap(a, b)
    z = _v(new, "zone")
    for k in kids:
        assert k["y"] + k["h"] <= z["y"] + z["h"] and k["x"] + k["w"] <= z["x"] + z["w"]


def test_add_component_from_reference_resolver():
    def refs(rc: RefCell):
        return ("shape=cylinder3;fillColor=#F5F5F5;", 50.0, 70.0) if rc.cell == "db" else None

    new, res = apply_ops(_doc(), [AddComponent(label="Store", like=RefCell(reference="r.drawio", cell="db"))], refs)
    assert res[0]["ok"]
    assert _v(new, "store")["style"]["shape"] == "cylinder3" and _v(new, "store")["w"] == 50.0
    _, res = apply_ops(_doc(), [AddComponent(label="Store", like=RefCell(reference="r.drawio", cell="zzz"))], refs)
    assert not res[0]["ok"]


def test_add_connection_and_unique_ids():
    new, res = apply_ops(_doc(), [AddConnection(source="db", target="api", label="back", like="e1")])
    assert res[0]["ok"]
    edge = next(e for e in dd.summarize(new)["edges"] if e["label"] == "back")
    assert (edge["source"], edge["target"], edge["dashed"]) == ("db", "api", True)
    new2, res2 = apply_ops(new, [AddConnection(source="db", target="api")])
    assert res2[0]["ok"] and dd.validate(new2)["valid"]
    _, bad = apply_ops(_doc(), [AddConnection(source="api", target="missing")])
    assert not bad[0]["ok"]


def test_move_absolute_and_reparent():
    doc = _doc()
    new, _ = apply_ops(doc, [Move(id="api", x=500, y=400)])
    assert (_v(new, "api")["x"], _v(new, "api")["y"]) == (500.0, 400.0)
    new, res = apply_ops(doc, [Move(id="api", parent="1")])  # out of the zone, auto-placed
    assert res[0]["ok"] and "parent" not in _v(new, "api")  # summary omits parent "1"
    assert dd.validate(new)["valid"]
    _, bad = apply_ops(doc, [Move(id="zone", parent="api")])
    assert not bad[0]["ok"]


def test_resize_and_set_style():
    new, _ = apply_ops(_doc(), [Resize(id="api", w=200), SetStyle(id="api", set={"fillColor": "#00FF00"}, remove=["fontSize"])])
    v = _v(new, "api")
    assert v["w"] == 200.0 and v["style"]["fillColor"] == "#00FF00" and "fontSize" not in v["style"]
    new, _ = apply_ops(_doc(), [SetStyle(id="api", like="db")])
    assert _v(new, "api")["style"]["shape"] == "cylinder3"
