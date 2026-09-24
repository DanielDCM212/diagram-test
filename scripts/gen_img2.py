"""One-off generator for diagrams/img_2.diagram.json (ITS national architecture style diagram).
Builds the structured JSON programmatically (grid layout) instead of hand-typing ~35 boxes.
"""
import json

PANEL = "#4FFEDF"
BOX = "#FFFFCC"
STROKE = "#663300"
BAND = "#04DEC6"

nodes = []
edges = []
_id = [0]


def nid(prefix):
    _id[0] += 1
    return f"{prefix}{_id[0]}"


def panel(x, y, w, h):
    nodes.append({"id": nid("panel"), "shape": "rect", "label": "", "x": x, "y": y, "w": w, "h": h,
                  "fill": PANEL, "rounded": False, "font_size": 1})


def title(text, x, y, w, h=26, size=16):
    nodes.append({"id": nid("title"), "shape": "text", "label": text, "x": x, "y": y, "w": w, "h": h,
                  "bold": True, "font_size": size})


def box(text, x, y, w=140, h=42):
    i = nid("box")
    nodes.append({"id": i, "shape": "rect", "label": text, "x": x, "y": y, "w": w, "h": h,
                  "fill": BOX, "stroke": STROKE, "rounded": True, "font_size": 10, "font_color": "#000000"})
    return i


def hband(text, x, y, w, h=24):
    i = nid("hband")
    nodes.append({"id": i, "shape": "rect", "label": text, "x": x, "y": y, "w": w, "h": h,
                  "fill": BAND, "rounded": True, "font_size": 10, "bold": True, "font_color": "#000000"})
    return i


def vband(text, x, y, w=16, h=170):
    i = nid("vband")
    nodes.append({"id": i, "shape": "rect", "label": text, "x": x, "y": y, "w": w, "h": h,
                  "fill": BAND, "rounded": True, "vertical_text": True, "font_size": 9, "bold": True,
                  "font_color": "#000000"})
    return i


def edge(src, tgt, dashed=False):
    edges.append({"id": nid("e"), "source": src, "target": tgt, "end_arrow": False, "dashed": dashed,
                  "color": "#333333", "orthogonal": True})


# ---- Travelers ----
panel(0, 0, 190, 170)
title("Travelers", 10, 6, 170)
t1 = box("Traveler Information\nDevices", 15, 45)
t2 = box("Regional Kiosks", 15, 95)
edge(t1, t2)

# ---- vertical band: Commercial Wireline/Wide Area Wireless ----
cwwaw = vband("Commercial Wireline/\nWide Area Wireless", 205, 0, 25, 340)
edge(t1, cwwaw)
edge(t2, cwwaw)

# ---- Vehicles (tall combined panel) ----
panel(0, 180, 390, 460)
v1 = box("Vehicle Information\nSystems", 15, 195)
v2 = box("Vehicle Mayday\nSystems", 15, 245)
edge(v1, v2)

waw_band = hband("Wide Area Wireless - Trunked/Dedicated Radio Systems", 15, 300, 355)
edge(v2, waw_band)

tvs = box("Transit Vehicle\nSystems", 15, 335)
mvs = box("Maintenance Vehicle\nSystems", 200, 335)
evs = box("Emergency Vehicle\nSystems", 15, 385)
cvs = box("Commercial Vehicle\nSystems", 200, 385)
ynp = box("Yellowstone NP\nElectronic Tags", 200, 435)
edge(waw_band, tvs)
edge(waw_band, mvs)
edge(tvs, evs)
edge(mvs, cvs)
edge(cvs, ynp)

title("Vehicles", 15, 590, 150)

# ---- Centers ----
panel(245, 0, 800, 290)
title("Centers", 245, 6, 800, size=18)

top_x = [255, 415, 575, 735, 895]
top_labels = ["Visitor Info\nServices", "Traveler Info\nServices", "Kiosk\nData Server",
              "Cable\nTelevision", "Traveler Service\nProviders"]
top_ids = [box(lbl, x, 45) for lbl, x in zip(top_labels, top_x)]

b_band = hband("Wireline Communications (B)", 255, 95, 780)
for tid in top_ids:
    edge(tid, b_band)

col_x = [255, 431, 607, 783]
band_x = [405, 581, 757]
band_labels = ["Wireline\nCommunications (D)", "Wireline\nCommunications (A)", "Wireline\nCommunications (C)"]
rows_y = [130, 182, 234]

col_labels = [
    ["Weather\nService", "Fleet Dispatch/\nOperations", "Transit Operations\nCenters"],
    ["Regional\nServer", "Incident Command/\nDispatch Systems", "Maintenance Mgmt\nSystems"],
    ["Mayday Message\nAnswering Points", "Public Safety\nAnswering Point", "State DOT\nOperations Centers"],
    ["Rail Operations\nCenters", "Park Entrance Fee\nAdministration", "State Commercial\nVehicle Admin"],
]

col_first_ids = []
for ci, cx in enumerate(col_x):
    ids = [box(col_labels[ci][ri], cx, rows_y[ri]) for ri in range(3)]
    col_first_ids.append(ids[0])
    edge(ids[0], ids[1])
    edge(ids[1], ids[2])

for i, bx in enumerate(band_x):
    vb = vband(band_labels[i], bx, 130, 16, 146)
    edge(b_band, vb)
    edge(vb, col_first_ids[i + 1])
edge(b_band, col_first_ids[0])

# ---- DSRC vertical band (Vehicles -> Roadside) ----
dsrc = vband("DSRC", 400, 350, 18, 300)
edge(cvs, dsrc)
edge(ynp, dsrc)

# ---- Roadside ----
panel(410, 505, 900, 230)
title("Roadside", 420, 700, 150)

c_band = hband("Wireline Communications (C)", 420, 520, 780)
edge(dsrc, c_band)

rcol_x = [420, 596, 746, 922]
rrows_y = [560, 612, 664]
rcol_labels = [
    ["National Park\nParking Systems", "Weigh Station\nSystems", "Entrance Gate\nAVI Systems"],
    ["CCTV\nRoadCams", "Intersection\nWarning Systems", None],
    ["Environmental\nSensor Stations", "Highway Advisory\nRadio Systems", "Automated\nGates"],
    ["Animal-Vehicle\nWarning Systems", "Dynamic\nWarning VMS", "Variable Message\nSigns"],
]

rcol_first_ids = []
rcol_all_ids = []
for ci, cx in enumerate(rcol_x):
    ids = []
    for ri in range(3):
        lbl = rcol_labels[ci][ri]
        if lbl is None:
            ids.append(None)
            continue
        ids.append(box(lbl, cx, rrows_y[ri]))
    rcol_all_ids.append(ids)
    rcol_first_ids.append(ids[0])
    for a, b in zip(ids, ids[1:]):
        if a and b:
            edge(a, b)
    edge(c_band, ids[0])

rc_vband = vband("Wireline\nCommunications (C)", 570, 560, 16, 170)
edge(rcol_first_ids[0], rc_vband)
edge(rc_vband, rcol_first_ids[1])

edge(rcol_all_ids[1][0], rcol_all_ids[2][0], dashed=True)

wwaw_vband = vband("Wireline/\nWide Area Wireless", 896, 560, 16, 170)
edge(rcol_first_ids[2], wwaw_vband)
edge(wwaw_vband, rcol_first_ids[3])

diagram = {"name": "ITS National Architecture", "nodes": nodes, "edges": edges}
with open("diagrams/img_2.diagram.json", "w", encoding="utf-8") as f:
    json.dump(diagram, f, indent=2)
print(f"nodes={len(nodes)} edges={len(edges)}")
