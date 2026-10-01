import io
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

from adk_agent.diagram_editor import drawio_doc as dd
from adk_agent.diagram_editor import export_client as ec
from adk_agent.diagram_editor import references as refs
from adk_agent.diagram_editor import tools
from tests.diagram_editor.test_drawio_doc import PLAIN
from tests.diagram_editor.test_layout import _reference_xml, _spec, refdir  # noqa: F401 (fixture)
from tests.diagram_editor.test_tools import Ctx


def _png(w=40, h=20, color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


class _Server:
    """Tiny stand-in for the export server; records requests, behaviour set per test."""

    def __init__(self):
        self.requests = []
        self.status, self.body, self.delay = 200, _png(), 0.0
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                outer.requests.append({"path": self.path, "form": urllib.parse.parse_qs(raw.decode()),
                                       "ctype": self.headers.get("Content-Type")})
                time.sleep(outer.delay)
                self.send_response(outer.status)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                try:
                    self.wfile.write(outer.body)
                except BrokenPipeError:
                    pass

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/ImageExport4/export"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture()
def server(monkeypatch):
    s = _Server()
    monkeypatch.setenv("DRAWIO_EXPORT_URL", s.url)
    yield s
    s.close()


def test_export_png_sends_form_request_and_returns_png(server):
    out = ec.export_png("<mxfile/>", page=2)
    assert out.startswith(b"\x89PNG")
    r = server.requests[0]
    assert r["path"] == "/ImageExport4/export" and r["ctype"] == "application/x-www-form-urlencoded"
    assert r["form"]["format"] == ["png"] and r["form"]["xml"] == ["<mxfile/>"]
    assert r["form"]["from"] == ["2"]


def test_export_png_downscales_huge_render(server):
    server.body = _png(6000, 3000)
    out = Image.open(io.BytesIO(ec.export_png("<x/>")))
    assert max(out.size) <= ec.MAX_IMAGE_EDGE


@pytest.mark.parametrize("status,body,msg", [(500, b"boom", "HTTP 500"), (200, b"<html>nope</html>", "not return a PNG"),
                                             (200, b"\x89PNG\r\n\x1a\ngarbage", "unreadable")])
def test_export_png_failures(server, status, body, msg):
    server.status, server.body = status, body
    with pytest.raises(ec.ExportError, match=msg):
        ec.export_png("<x/>")


def test_export_png_unset_unreachable_and_timeout(monkeypatch, server):
    server.delay = 1.5
    monkeypatch.setenv("DRAWIO_EXPORT_TIMEOUT", "0.3")
    with pytest.raises(ec.ExportError, match="unreachable"):
        ec.export_png("<x/>")
    monkeypatch.setenv("DRAWIO_EXPORT_URL", "http://127.0.0.1:9/export")  # nothing listens on port 9
    with pytest.raises(ec.ExportError, match="unreachable"):
        ec.export_png("<x/>")
    monkeypatch.delenv("DRAWIO_EXPORT_URL")
    assert not ec.is_configured()
    with pytest.raises(ec.ExportError, match="not set"):
        ec.export_png("<x/>")


# ---- preview_diagram: server first, Python fallback ----------------------------------

@pytest.fixture()
def ctx():
    c = Ctx()
    c.state["drawio_xml"] = dd.serialize(dd.load(PLAIN))
    return c


def _pixel(res):
    from PIL import Image
    return Image.open(io.BytesIO(res["image"].inline_data.data)).convert("RGB").getpixel((1, 1))


def test_preview_uses_server_when_available(server, ctx):
    res = tools.preview_diagram(ctx)
    assert res["renderer"] == "export_server" and "fallback_reason" not in res and _pixel(res) == (200, 30, 30)
    assert "<mxCell" in server.requests[0]["form"]["xml"][0]  # the working diagram XML was sent
    assert res["node_count"] == 3 and res["edge_count"] == 1


@pytest.mark.parametrize("break_server", [
    lambda s, mp: setattr(s, "status", 503),
    lambda s, mp: setattr(s, "body", b"not an image"),
    lambda s, mp: mp.setenv("DRAWIO_EXPORT_URL", "http://127.0.0.1:9/export"),
])
def test_preview_falls_back_to_python(server, ctx, monkeypatch, break_server):
    break_server(server, monkeypatch)
    res = tools.preview_diagram(ctx)
    assert res["renderer"] == "python_fallback" and res["fallback_reason"] and "schematic" in res["note"]
    assert res["image"].inline_data.data.startswith(b"\x89PNG") and _pixel(res) != (200, 30, 30)


def test_preview_python_when_unconfigured(monkeypatch, ctx):
    monkeypatch.delenv("DRAWIO_EXPORT_URL", raising=False)
    res = tools.preview_diagram(ctx)
    assert res["renderer"] == "python_fallback" and "DRAWIO_EXPORT_URL" in res["fallback_reason"]


def test_preview_errors_do_not_call_server(server, ctx):
    assert "error" in tools.preview_diagram(Ctx())
    assert "out of range" in tools.preview_diagram(ctx, page=5)["error"]
    assert server.requests == []
