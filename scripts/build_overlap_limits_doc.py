"""Builds docs/overlap_resolution_limits.drawio from REAL runs of the overlap tools.

Every 'after' panel is the actual output of resolve_overlaps; the numbers on each page come
from its report, so the document can't drift from the tool's behaviour (re-run to refresh).
"""
import json
import sys
import uuid
from html import escape
from pathlib import Path

from adk_agent.diagram_editor import drawio_doc as dd
from adk_agent.diagram_editor import overlaps as ov

OUT = Path(__file__).resolve().parent.parent / "docs" / "overlap_resolution_limits.drawio"
PW, PH = 520, 400           # panel size
AX, BX, PY = 20, 580, 130   # before-panel x, after-panel x, panels' y
CONTENT_DX, CONTENT_DY = 40, 80             # content origin inside a panel: room for the tool to move things up

BLUE = "rounded=1;whiteSpace=wrap;html=1;fillColor=#DAE8FC;strokeColor=#6C8EBF;fontSize=12;"
GREEN = "rounded=1;whiteSpace=wrap;html=1;fillColor=#D5E8D4;strokeColor=#82B366;fontSize=12;"
ORANGE = "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFE6CC;strokeColor=#D79B00;fontSize=12;"
GREY = "rounded=1;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor=#666666;fontSize=12;"
NOTE_Y = "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF2CC;strokeColor=#D6B656;align=left;verticalAlign=top;spacing=8;fontSize=13;"
NOTE_R = "rounded=1;whiteSpace=wrap;html=1;fillColor=#F8CECC;strokeColor=#B85450;align=left;verticalAlign=top;spacing=8;fontSize=13;"
NOTE_G = "rounded=1;whiteSpace=wrap;html=1;fillColor=#D5E8D4;strokeColor=#82B366;align=left;verticalAlign=top;spacing=8;fontSize=13;"
FRAME = "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#999999;dashed=1;verticalAlign=top;align=left;spacing=6;fontStyle=1;fontSize=13;fontColor=#555555;"
TITLE = "text;html=1;align=left;verticalAlign=middle;whiteSpace=wrap;fontSize=22;fontStyle=1;"
SUB = "text;html=1;align=left;verticalAlign=top;whiteSpace=wrap;fontSize=13;fontColor=#444444;"
REPORT = "rounded=0;whiteSpace=wrap;html=1;fillColor=#E1D5E7;strokeColor=#9673A6;align=left;spacing=8;fontSize=12;fontFamily=Courier New;"
EDGE = "endArrow=block;html=1;strokeColor=#333333;"


class Page:
    def __init__(self, name):
        self.name, self.cells, self.n = name, [], 0

    def uid(self, p="c"):
        self.n += 1
        return f"{p}{self.n}"

    def v(self, label, style, x, y, w, h, cid=None):
        cid = cid or self.uid()
        self.cells.append(
            f'<mxCell id="{cid}" value="{escape(label, quote=True)}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return cid

    def edge(self, cid, source, target, style, label="", src_pt=None, tgt_pt=None):
        attrs = f'id="{cid}" value="{escape(label, quote=True)}" style="{style}" edge="1" parent="1"'
        if source:
            attrs += f' source="{source}"'
        if target:
            attrs += f' target="{target}"'
        pts = ""
        if src_pt:
            pts += f'<mxPoint x="{src_pt[0]}" y="{src_pt[1]}" as="sourcePoint"/>'
        if tgt_pt:
            pts += f'<mxPoint x="{tgt_pt[0]}" y="{tgt_pt[1]}" as="targetPoint"/>'
        self.cells.append(f'<mxCell {attrs}><mxGeometry relative="1" as="geometry">{pts}</mxGeometry></mxCell>')

    def xml(self):
        return (f'<diagram id="{uuid.uuid4()}" name="{escape(self.name, quote=True)}"><mxGraphModel dx="1200" dy="800" grid="1" '
                f'gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
                f'pageWidth="1100" pageHeight="850" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                + "".join(self.cells) + "</root></mxGraphModel></diagram>")


# ---- scenario docs (flat, plain cells in panel-local coordinates) ------------------------

def N(i, label, x, y, w, h, style=BLUE):
    return (f'<mxCell id="{i}" value="{escape(label, quote=True)}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def E(i, s, t, style=EDGE, src_pt=None, tgt_pt=None):
    pts = ""
    if src_pt:
        pts += f'<mxPoint x="{src_pt[0]}" y="{src_pt[1]}" as="sourcePoint"/>'
    if tgt_pt:
        pts += f'<mxPoint x="{tgt_pt[0]}" y="{tgt_pt[1]}" as="targetPoint"/>'
    a = (f' source="{s}"' if s else "") + (f' target="{t}"' if t else "")
    return (f'<mxCell id="{i}" value="" style="{style}" edge="1" parent="1"{a}>'
            f'<mxGeometry relative="1" as="geometry">{pts}</mxGeometry></mxCell>')


def D(*cells):
    return dd.load('<mxfile><diagram name="s"><mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                   + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")


def put(page, doc, prefix, ox, oy, mark=()):
    """Copy a scenario doc into a page panel at (ox, oy); `mark` ids get a thick red outline."""
    for c in doc.cells():
        cid = c.get("id")
        if cid in ("0", "1"):
            continue
        g = dd.geometry(c)
        if dd.attr(c, "vertex") == "1":
            style = dd.attr(c, "style")
            if cid in mark:
                style += "strokeColor=#FF0000;strokeWidth=3;"
            page.v(dd.attr(c, "value") or "", style, round(float(g.get("x", 0)) + ox, 1), round(float(g.get("y", 0)) + oy, 1),
                   float(g.get("width")), float(g.get("height")), cid=prefix + cid)
        else:
            sp = g.find("mxPoint[@as='sourcePoint']") if g is not None else None
            tp = g.find("mxPoint[@as='targetPoint']") if g is not None else None
            page.edge(prefix + cid, prefix + dd.attr(c, "source") if dd.attr(c, "source") else None,
                      prefix + dd.attr(c, "target") if dd.attr(c, "target") else None, dd.attr(c, "style"),
                      src_pt=(float(sp.get("x")) + ox, float(sp.get("y")) + oy) if sp is not None else None,
                      tgt_pt=(float(tp.get("x")) + ox, float(tp.get("y")) + oy) if tp is not None else None)


def scenario(title, subtitle, before, params, see, expect, tone, check=None):
    after, rep = ov.resolve_overlaps(before, 0, params)
    if check:
        assert check(rep), f'{title}: the tool no longer behaves as this page claims: {json.dumps(rep)[:400]}'
    page = Page(title.split(" - ")[0])
    page.v(title, TITLE, 20, 10, 1040, 36)
    page.v(subtitle, SUB, 20, 48, 1040, 40)
    remaining = {i for r in rep["remaining"] for i in (r["a"], r["b"])} | {f["id"] for f in rep["labels_still_overflowing"]}
    summary = (f"overlaps: {rep['overlaps_before']} -> {rep['overlaps_after']}   moved: {len(rep['moved'])}   "
               f"resized: {len(rep['resized'])}   fonts changed: {len(rep['fonts_changed'])}   "
               f"still overlapping: {rep['overlaps_after']}   labels still overflowing: {len(rep['labels_still_overflowing'])}")
    page.v("REAL TOOL REPORT  " + summary, REPORT, 20, 92, 1000, 28)
    page.v("BEFORE", FRAME, AX, PY, PW, PH)
    page.v("AFTER (actual tool output)", FRAME, BX, PY, PW, PH)
    put(page, before, "b_", AX + CONTENT_DX, PY + CONTENT_DY)
    put(page, after, "a_", BX + CONTENT_DX, PY + CONTENT_DY, mark=remaining)
    page.v("What happens<br>" + see, NOTE_Y, AX, PY + PH + 20, PW, 130)
    page.v("What the business should expect<br>" + expect, {"ok": NOTE_G, "limit": NOTE_R}[tone], BX, PY + PH + 20, PW, 130)
    return page, rep


pages, reports = [], {}

# 1 ------------------------------------------------------------------ what works
before = D(N("gw", "API Gateway", 40, 20, 120, 50), N("auth", "Auth Service", 120, 50, 120, 50),
           N("ord", "Orders Service", 60, 90, 120, 50), N("bill", "Billing Service", 220, 80, 120, 50),
           N("db", "Orders DB", 370, 40, 80, 60, GREEN), E("e1", "gw", "ord"), E("e2", "ord", "db"))
p, r = scenario("2. Works well - typical overlap clean-up",
                "Several boxes slightly on top of each other: the common case. This is what the tool is built for.",
                before, ov.Params(), "Overlapping boxes are pushed apart a little, each by a small bounded amount. Arrows attached to boxes follow them. Sizes and fonts stay as they were.",
                "Reliable for light and medium overlap: a few boxes touching or partly covering each other. Result is validated before it is saved.", "ok", check=lambda r: r["overlaps_after"] == 0 and not r["resized"] and not r["fonts_changed"])
pages.append(p); reports["works"] = r

# 2 ------------------------------------------------------------------ too crowded
before = D(*[N(f"s{i}", f"Service {chr(65 + i)}", 40 + i * 18, 10 + i * 14, 170, 60) for i in range(8)])
p, r = scenario("3. Too crowded - the movement budget runs out",
                "A pile of boxes dropped almost on top of each other (e.g. copy-paste). Default limits: move at most 80 px per box, shrink to 75%.",
                before, ov.Params(), "Each box may only move a limited distance and shrink a limited amount. Boxes outlined in red are still overlapping after the tool did everything it is allowed to do.",
                "Heavy overlap cannot be fully fixed within the limits. The tool reports what is left instead of scattering the diagram. Options: raise the limits (more movement, smaller boxes), or re-layout manually.", "limit", check=lambda r: r["overlaps_after"] > 0 and r["overlaps_after"] < r["overlaps_before"])
pages.append(p); reports["crowded"] = r

# 3 ------------------------------------------------------------------ locked
before = D(N("pri", "Primary DB (keep in place)", 60, 30, 150, 70, GREEN), N("rep", "Replica DB (keep in place)", 140, 70, 150, 70, GREEN),
           N("cache", "Cache", 330, 50, 100, 50), N("note", "Reporting", 20, 150, 120, 50))
p, r = scenario("4. Locked elements - cannot move what the user said to keep",
                "The request says both databases must stay exactly where they are. They overlap each other.",
                before, ov.Params(lock=frozenset({"pri", "rep"})), "Locked boxes are never moved or resized, so an overlap between two locked boxes cannot be fixed. Only unlocked neighbours may move.",
                "Contradictory instructions (keep both fixed + remove the overlap) end with a report, not a guess. The agent must tell the user which constraint to relax.", "limit", check=lambda r: r["overlaps_after"] == 1 and not r["moved"])
pages.append(p); reports["locked"] = r

# 4 ------------------------------------------------------------------ labels
long_a = "Customer Identity Verification and Fraud Screening Service"
before = D(N("v", long_a, 20, 20, 90, 30), N("pg", "Payment Gateway Adapter", 150, 20, 100, 40, BLUE.replace("fontSize=12", "fontSize=14")),
           N("ok", "Fits fine", 300, 20, 100, 40), N("n4", "Billing", 20, 120, 100, 40))
p, r = scenario("5. Long labels - the font has a floor",
                "Not an overlap problem but a related one: text that does not fit its box. Tool run with fix_overflow=true and minimum font 8.",
                before, ov.Params(fix_overflow=True, min_font=8), "Fonts shrink just enough to fit, never below the minimum and never larger. A label too long for a small box overflows even at the minimum size (outlined in red).",
                "Moderate overflow is fixed automatically. Very long text in a small box is only reported: shorten the label or enlarge the box. Fit is estimated with standard font metrics, so it is approximate, not pixel exact.", "limit", check=lambda r: r["labels_still_overflowing"] and len(r["fonts_changed"]) >= 1 and r["overlaps_before"] == 0)
pages.append(p); reports["labels"] = r

# 5 ------------------------------------------------------------------ free edges
before = D(N("part", "External Partner", 290, 30, 100, 60, ORANGE), N("fw", "Firewall", 360, 35, 90, 60, GREY),
           N("ord", "Order Service", 30, 40, 120, 60), E("e1", "ord", None, tgt_pt=(290, 60)))
p, r = scenario("6. Arrows with a loose end - they do not follow",
                "The arrow starts at Order Service and its loose end was drawn by hand to touch External Partner (it is not connected to it).",
                before, ov.Params(), "Arrows connected to boxes at both ends follow when boxes move. An arrow with a loose end keeps its loose end at the old position, so it can end in empty space or inside another box.",
                "Diagrams drawn with hand-placed arrow ends need a visual check after any move. The tool warns when such arrows exist. Fix: connect the arrow end to the box.", "limit", check=lambda r: "free end" in r.get("warning", "")
                 and any(m["id"] == "part" and abs(m["dx"]) >= 5 for m in r["moved"]))  # partner moves sideways, tip stays
pages.append(p); reports["free_edges"] = r

# 6 ------------------------------------------------------------------ edges crossing boxes
before = D(N("a", "Web App", 20, 70, 100, 50), N("b", "Cache", 190, 55, 100, 80, GREY), N("c", "Orders API", 360, 70, 100, 50),
           E("e1", "a", "c"))
p, r = scenario("7. Arrows through boxes - not detected",
                "A straight arrow from Web App to Orders API goes right through the Cache box. Boxes do not overlap each other.",
                before, ov.Params(), "The tool only looks at boxes. It does not see arrows passing through boxes, arrows crossing each other, or labels sitting on lines. It reports 0 overlaps and changes nothing.",
                "A clean report does not mean a clean diagram. The preview (and a human look) is still needed for arrow routing. Edge routing clean-up is not part of this tool.", "limit", check=lambda r: r["overlaps_before"] == 0 and not r["moved"])
pages.append(p); reports["edges"] = r

# 7 ------------------------------------------------------------------ bounding boxes
before = D(N("c1", "Cloud", 40, 20, 120, 120, "ellipse;whiteSpace=wrap;html=1;fillColor=#DAE8FC;strokeColor=#6C8EBF;fontSize=12;"),
           N("c2", "On-prem", 130, 110, 120, 120, "ellipse;whiteSpace=wrap;html=1;fillColor=#FFE6CC;strokeColor=#D79B00;fontSize=12;"),
           N("d1", "Rotated", 330, 40, 100, 50, "rhombus;whiteSpace=wrap;html=1;fillColor=#E1D5E7;strokeColor=#9673A6;fontSize=12;"),
           N("d2", "Decision", 380, 90, 100, 50, "rhombus;whiteSpace=wrap;html=1;fillColor=#D5E8D4;strokeColor=#82B366;fontSize=12;"))
p, r = scenario("8. Round shapes - treated as rectangles",
                "Circles and diamonds whose corners (bounding boxes) overlap, while the visible shapes do not touch.",
                before, ov.Params(), "Overlap is measured on each shape's bounding rectangle. Round or diamond shapes placed diagonally are reported as overlapping and get moved even though they were fine.",
                "Some unnecessary movement, mostly with circles, diamonds and rotated shapes. Harmless but visible. Use lock for shapes that were placed deliberately.", "limit", check=lambda r: r["overlaps_before"] >= 1 and r["moved"])
pages.append(p); reports["shapes"] = r

# 8 ------------------------------------------------------------------ ripple
before = D(N("z1", "Zone 1", 10, 10, 140, 230, GREY.replace("rounded=1", "rounded=0") + "verticalAlign=top;"),
           N("z2", "Zone 2", 150, 10, 140, 230, GREY.replace("rounded=1", "rounded=0") + "verticalAlign=top;"),
           N("z3", "Zone 3", 290, 10, 140, 230, GREY.replace("rounded=1", "rounded=0") + "verticalAlign=top;"),
           N("a1", "Service A", 20, 40, 70, 50), N("a2", "Service B", 60, 45, 80, 50),
           N("b1", "Service C", 160, 40, 120, 50), N("c1", "Service D", 300, 40, 120, 50))
p, r = scenario("9. Ripple effect - one fix can move neighbours",
                "Two services overlap inside Zone 1. The three zones are packed edge to edge, with no free space between them.",
                before, ov.Params(), "Fixing the overlap inside Zone 1 makes the zone grow. The grown zone pushes Zone 2, which pushes Zone 3: a small local problem moves several areas. Pushed neighbours can also end up closer together than the usual spacing, because clearance is only added when a pair is pushed.",
                "On tightly packed diagrams a fix can shift more than the overlapping boxes. Each box moves at most the limit, and the report lists everything that moved. Use 'only' to restrict the clean-up to one area.", "limit", check=lambda r: {"z2", "z3"} <= {m["id"] for m in r["moved"]} and r["overlaps_after"] == 0)
pages.append(p); reports["ripple"] = r

# ------------------------------------------------------------------ summary page
summary = Page("1. Summary")
summary.v("Diagram editor agent: overlap clean-up - what to expect", TITLE, 20, 10, 1100, 36)
summary.v("Built from real runs of the tool. Each numbered page shows one situation: the diagram BEFORE, and the tool's actual result AFTER. "
          "Green = works well, red = a known limit the team should plan for.", SUB, 20, 48, 1100, 40)
cols = [("#", 30), ("Situation", 210), ("What you will see", 400), ("Impact", 80), ("What to do", 380)]
x = 20
for name, w in cols:
    summary.v(name, "rounded=0;whiteSpace=wrap;html=1;fillColor=#333333;fontColor=#FFFFFF;fontStyle=1;fontSize=13;align=left;spacing=6;", x, 100, w, 32)
    x += w
rows = [
    ("2", "Typical overlap", "Boxes pushed apart a little; arrows follow; sizes and fonts unchanged.", "None", "Nothing: this is the normal case.", "#D5E8D4"),
    ("3", "Very crowded pile", "Some boxes still overlap after moving and shrinking within the limits.", "Medium", "Raise limits or re-layout by hand. The tool reports what is left.", "#FFE6CC"),
    ("4", "Locked elements", "Two kept-in-place boxes that overlap each other cannot be fixed.", "Low", "The agent must ask which constraint to relax.", "#FFF2CC"),
    ("5", "Long labels", "Text shrinks to a minimum size; very long text in a small box still overflows.", "Medium", "Shorten the label or enlarge the box. Fit check is approximate.", "#FFE6CC"),
    ("6", "Arrows with a loose end", "Loose arrow ends stay behind when boxes move.", "Medium", "Visual check after moves. Connect arrow ends to boxes.", "#FFE6CC"),
    ("7", "Arrows through boxes", "Not detected: report says 0 overlaps although an arrow crosses a box.", "Medium", "Look at the preview. Arrow routing is not handled.", "#FFE6CC"),
    ("8", "Circles and diamonds", "Treated as rectangles: some unneeded movement.", "Low", "Lock deliberately placed shapes.", "#FFF2CC"),
    ("9", "Packed neighbours", "One fix can push several neighbouring zones.", "Medium", "Restrict the clean-up to one area; read the list of moved items.", "#FFE6CC"),
    ("10", "Other limits", "See page 10: text-only items that cannot be shown in a picture.", "Varies", "Read page 10.", "#F5F5F5"),
]
y = 132
for n, sit, see, imp, todo, color in rows:
    x = 20
    for (name, w), val in zip(cols, (n, sit, see, imp, todo)):
        st = f"rounded=0;whiteSpace=wrap;html=1;fillColor={color if name == 'Impact' else '#FFFFFF'};strokeColor=#BBBBBB;align=left;verticalAlign=middle;spacing=6;fontSize=12;"
        summary.v(val, st, x, y, w, 56)
        x += w
    y += 56

# ------------------------------------------------------------------ other limits (text only)
other = Page("10. Other limits")
other.v("10. Other limits that are not easy to draw", TITLE, 20, 10, 1040, 36)
items = [
    ("Preview accuracy", "Without the draw.io export server the preview is a schematic drawing (straight arrows, no icons). Real draw.io rendering needs the export server to be reachable. Falls back automatically, and says which one was used."),
    ("Text fit is an estimate", "Label fit uses standard font measurements, not draw.io's own renderer. Expect occasional labels that still look tight or are shrunk more than needed."),
    ("Only the first page of a file is edited", "Describe and preview accept a page number, but edit operations act on the first page only."),
    ("New diagrams depend on reference diagrams", "Creating a diagram in the house style needs the reference diagrams installed. Without them the agent must say it is not in house style."),
    ("Judgement calls stay with the model", "Requests like 'identify the AWS zone' rely on the labels and shapes in the file. With several candidates or unusual names the agent can pick the wrong one: ask it to state what it picked."),
    ("Placeholders are shown raw", "Labels such as %owner% (draw.io placeholders) are shown as written, not resolved to the property value."),
    ("Not a layout designer", "The tool removes overlaps and keeps the existing design. It does not re-arrange a diagram, align things to a grid or improve readability."),
]
y = 60
for head, body in items:
    other.v(head, "rounded=0;whiteSpace=wrap;html=1;fillColor=#333333;fontColor=#FFFFFF;fontStyle=1;fontSize=13;align=left;spacing=6;", 20, y, 260, 64)
    other.v(body, "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#BBBBBB;align=left;verticalAlign=middle;spacing=8;fontSize=12;", 280, y, 780, 64)
    y += 64

# ------------------------------------------------------------------ write
order = [summary] + pages + [other]
xml = '<mxfile host="app.diagrams.net">' + "".join(p.xml() for p in order) + "</mxfile>"
dd.load(xml)  # parses cleanly
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(xml, encoding="utf-8")
for k, v in reports.items():
    print(f"{k:10} before={v['overlaps_before']} after={v['overlaps_after']} moved={len(v['moved'])} resized={len(v['resized'])} "
          f"fonts={len(v['fonts_changed'])} overflow_left={len(v['labels_still_overflowing'])} warn={'warning' in v}")
from adk_agent.diagram_editor import text_fit
final = dd.load(xml)
bad = 0
for i, pg in enumerate(final.pages):
    res = ov.find_overlaps(final, i)
    deliberate = [o for o in res["overlaps"] if not (o["a"].startswith("b_") and o["b"].startswith("b_"))
                  and not (o["a"].startswith("a_") and o["b"].startswith("a_"))]
    if deliberate:
        bad += 1
        print("UNEXPECTED overlap on", pg.get("name"), [(o["a"], o["b"]) for o in deliberate])
    for o in res["label_overflow"]:
        if not (pg.get("name").startswith("5.") and o["id"].startswith("b_")):  # page 5 BEFORE overflows on purpose
            bad += 1
            print("TEXT OVERFLOW on", pg.get("name"), o["id"], o["label"][:30], o["box"], "font", o["font"], "->", o["suggested_font"])
print("self-check problems:", bad)
print("wrote", OUT, len(order), "pages")
