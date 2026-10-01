from types import SimpleNamespace

import pytest
from google.adk.sessions.state import State
from google.genai import types

from adk_agent.diagram_editor import drawio_doc as dd
from adk_agent.diagram_editor import references as refs
from adk_agent.diagram_editor import tools
from tests.diagram_editor.test_drawio_doc import PLAIN
from tests.diagram_editor.test_layout import _reference_xml, _spec, refdir  # noqa: F401 (fixture)


class Ctx(SimpleNamespace):
    """Stand-in for ToolContext/CallbackContext. Uses ADK's real State (not a dict): it has no
    pop()/del/items(), and a dict here once let a State.pop() call through to production."""

    def __init__(self):
        super().__init__(state=State({}, {}))


@pytest.fixture(autouse=True)
def _refs(refdir, monkeypatch):  # noqa: F811
    monkeypatch.setattr(refs, "REFERENCE_DIR", refdir)


def _request(*parts, role="user"):
    return SimpleNamespace(contents=[types.Content(role=role, parts=list(parts))])


def test_capture_from_pasted_text_replaces_xml_with_placeholder():
    ctx = Ctx()
    req = _request(types.Part(text=f"Rename API to Gateway:\n```xml\n{PLAIN}\n```"))
    tools.capture_input_xml(ctx, req)
    sent = req.contents[0].parts[0].text
    assert "<mxCell" not in sent and "describe_diagram" in sent and "Rename API to Gateway" in sent
    assert "customKeep=1" in ctx.state["drawio_xml"]
    # idempotent across repeated model calls; does not clobber later edits
    ctx.state["drawio_xml"] = "EDITED"
    tools.capture_input_xml(ctx, _request(types.Part(text=f"Rename:\n{PLAIN}")))
    assert ctx.state["drawio_xml"] == "EDITED"
    # a *different* XML pasted later replaces the working diagram
    other = PLAIN.replace('value="API"', 'value="Other"')
    tools.capture_input_xml(ctx, _request(types.Part(text=other)))
    assert "Other" in ctx.state["drawio_xml"]


def test_capture_from_attached_file():
    ctx = Ctx()
    part = types.Part.from_bytes(data=PLAIN.encode(), mime_type="application/xml")
    req = _request(part, types.Part(text="add a cache"))
    tools.capture_input_xml(ctx, req)
    assert "drawio_xml" in ctx.state
    assert req.contents[0].parts[0].inline_data is None and "loaded in the session" in req.contents[0].parts[0].text


def test_capture_bad_xml_sets_error_not_state():
    ctx = Ctx()
    tools.capture_input_xml(ctx, _request(types.Part(text="<mxfile><diagram name='x'>@@@</diagram></mxfile>")))
    assert "drawio_xml" not in ctx.state and ctx.state["input_xml_error"]
    assert "could not be parsed" in tools.describe_diagram(ctx)["error"]


def test_bad_xml_error_is_cleared_by_a_later_good_xml():
    ctx = Ctx()
    tools.capture_input_xml(ctx, _request(types.Part(text="<mxfile><diagram name='x'>@@@</diagram></mxfile>")))
    assert ctx.state["input_xml_error"] and "drawio_xml" not in ctx.state
    tools.capture_input_xml(ctx, _request(types.Part(text=f"fixed it:\n{PLAIN}")))  # real State: no pop() available
    assert not ctx.state["input_xml_error"] and "drawio_xml" in ctx.state
    assert tools.describe_diagram(ctx)["vertices"]


def test_capture_ignores_model_turns_and_plain_text():
    ctx = Ctx()
    tools.capture_input_xml(ctx, _request(types.Part(text=PLAIN), role="model"))
    tools.capture_input_xml(ctx, _request(types.Part(text="make a diagram of an order system")))
    assert ctx.state.to_dict() == {}


def test_edit_diagram_tool_with_dict_ops_and_atomic_failure():
    ctx = Ctx()
    ctx.state["drawio_xml"] = dd.serialize(dd.load(PLAIN))
    bad = tools.edit_diagram(ctx, [{"op": "update_label", "id": "api", "label": "GW"}, {"op": "delete", "id": "zzz"}])
    assert bad["applied"] is False and "GW" not in ctx.state["drawio_xml"]
    ok = tools.edit_diagram(ctx, [{"op": "update_label", "id": "api", "label": "GW"},
                                  {"op": "add_component", "label": "Cache", "parent": "zone", "like": "api", "near": "api"},
                                  {"op": "add_connection", "source": "api", "target": "cache", "like": "e1"}])
    assert ok["applied"] is True
    s = dd.summarize(dd.load(ctx.state["drawio_xml"]))
    assert {"api", "cache"} <= {v["id"] for v in s["vertices"]}
    assert tools.validate_drawio(ctx)["valid"]


def test_edit_diagram_tool_reference_like_and_invalid_ops():
    ctx = Ctx()
    ctx.state["drawio_xml"] = dd.serialize(dd.load(PLAIN))
    ok = tools.edit_diagram(ctx, [{"op": "add_component", "label": "Store", "like": {"reference": "platform.drawio", "cell": "a2"}}])
    assert ok["applied"]
    assert any(v["style"].get("shape") == "cylinder3" for v in dd.summarize(dd.load(ctx.state["drawio_xml"]))["vertices"]
               if v["label"] == "Store")
    assert tools.edit_diagram(ctx, [{"op": "explode", "id": "x"}])["applied"] is False
    assert "error" in tools.edit_diagram(Ctx(), [])  # no diagram at all


def test_create_then_edit_roundtrip_through_state():
    ctx = Ctx()
    made = tools.create_diagram(ctx, _spec())
    assert made["created"] is True and ctx.state["drawio_xml"]
    desc = tools.describe_diagram(ctx)
    assert {"api", "pg", "core"} <= {v["id"] for v in desc["vertices"]}
    ok = tools.edit_diagram(ctx, [{"op": "add_component", "label": "Queue", "parent": "core", "like": "svc2"}])
    assert ok["applied"]
    kids = [v for v in tools.describe_diagram(ctx)["vertices"] if v.get("parent") == "core"]
    assert len(kids) == 4
    core = next(v for v in tools.describe_diagram(ctx)["vertices"] if v["id"] == "core")
    assert all(k["y"] + k["h"] <= core["y"] + core["h"] for k in kids)  # container grew to fit


def test_create_diagram_reports_problems_and_accepts_dict():
    ctx = Ctx()
    spec = _spec().model_dump(mode="json")
    spec["layout_from"] = "nope.drawio"
    res = tools.create_diagram(ctx, spec)
    assert res["created"] is False and any("nope.drawio" in e for e in res["errors"])
    assert "drawio_xml" not in ctx.state
    assert tools.create_diagram(ctx, {"title": "x"})["created"] is False  # invalid spec shape


def test_reference_tools():
    ctx = Ctx()
    assert [r["name"] for r in tools.list_references(ctx)["references"] if "error" not in r] == ["platform.drawio"]
    g = tools.get_reference(ctx, "platform.drawio")
    assert g["profile"]["layout"]["zone_arrangement"] == "columns" and g["diagram"]["vertices"]
    assert "error" in tools.get_reference(ctx, "../secrets.drawio")


def test_preview_returns_image():
    ctx = Ctx()
    tools.create_diagram(ctx, _spec())
    res = tools.preview_diagram(ctx)
    assert res["node_count"] >= 8 and res["image"].inline_data.data[:8] == b"\x89PNG\r\n\x1a\n"
