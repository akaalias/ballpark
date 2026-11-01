"""Build exp 17's architecture figure (ring pooling). Writes figure.json + fig.svg here."""
import json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
INK, CAP, ANN, RED, TRN = "#111111", "#6b6a60", "#9b998c", "#8c2f1f", "#8a6a1e"
F = "Palatino,Georgia,serif"
W, H = 980, 620
CY = H // 2
o = []


def T(x, y, s, size=9.5, fill=CAP, w=400, anchor="middle", italic=False, extra=""):
    st = " font-style='italic'" if italic else ""
    s = s.replace("&", "&amp;")
    o.append(f"<text x='{x:.1f}' y='{y:.1f}' font-family='{F}' font-size='{size}' fill='{fill}' "
             f"font-weight='{w}' text-anchor='{anchor}'{st}{extra}>{s}</text>")


def L(x1, y1, x2, y2, c=ANN, sw=1, op=None, dash=None):
    a = f" opacity='{op}'" if op is not None else ""
    d = f" stroke-dasharray='{dash}'" if dash else ""
    o.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{c}' stroke-width='{sw}'{a}{d}/>")


def arrow(x1, y, x2, c=ANN):
    L(x1, y, x2 - 5, y, c)
    o.append(f"<path d='M {x2:.1f},{y:.1f} l -6,-3 v 6 Z' fill='{c}'/>")


def grid(x, y, s, n, c=INK, op=0.45, sw=1.2):
    o.append(f"<rect x='{x}' y='{y}' width='{s}' height='{s}' fill='none' stroke='{c}' stroke-width='{sw}'/>")
    st = s / n
    for i in range(1, n):
        L(x + i * st, y, x + i * st, y + s, c, 0.45, op)
        L(x, y + i * st, x + s, y + i * st, c, 0.45, op)


def slab(x, y, s, depth, c=INK):
    dx, dy = depth * 0.92, depth * 0.5
    o.append(f"<polygon points='{x},{y} {x+dx:.1f},{y-dy:.1f} {x+s+dx:.1f},{y-dy:.1f} {x+s},{y}' fill='#00000010' stroke='{c}' stroke-width='1'/>")
    o.append(f"<polygon points='{x+s},{y} {x+s+dx:.1f},{y-dy:.1f} {x+s+dx:.1f},{y+s-dy:.1f} {x+s},{y+s}' fill='#0000001c' stroke='{c}' stroke-width='1'/>")
    o.append(f"<rect x='{x}' y='{y}' width='{s}' height='{s}' fill='#00000006' stroke='{c}' stroke-width='1.2'/>")


# headings
T(8, 20, "INFERENCE PATH — WHAT FLIES", 9, ANN, 600, "start", extra=" letter-spacing='1.8'")
T(8, 612, "TRAINING SIGNALS — NEVER FLY", 9, TRN, 600, "start", extra=" letter-spacing='1.8'")

# camera frame (verbatim glyph)
Y = CY - 38
o.append(f"""<g id='cam-terrain'><rect id='frozen-input' x='26' y='{Y}' width='76' height='76' fill='#f6f4ea' stroke='#9b998c' stroke-width='1.6'/><path d='M31 {Y+47} L97 {Y+27}' stroke='#e6e3d4' stroke-width='5' fill='none'/><path d='M59 {Y+6} L49 {Y+70}' stroke='#e6e3d4' stroke-width='3.5' fill='none'/><rect x='34' y='{Y+9}' width='12' height='8' fill='#d9d5c3' transform='rotate(-8 40 {Y+13})'/><rect x='78' y='{Y+8}' width='10' height='11' fill='#cfccbd'/><rect x='35' y='{Y+57}' width='13' height='8' fill='#d9d5c3'/><rect x='75' y='{Y+50}' width='10' height='9' fill='#cfccbd' transform='rotate(6 80 {Y+54})'/><rect x='54' y='{Y+31}' width='9' height='8' fill='#d9d5c3' opacity='.85'/><ellipse cx='86' cy='{Y+63}' rx='8' ry='6' fill='#8a6a1e' opacity='.12'/><circle cx='40' cy='{Y+28}' r='.7' fill='#6b6a60' opacity='.5'/><circle cx='70' cy='{Y+14}' r='.7' fill='#6b6a60' opacity='.5'/><circle cx='92' cy='{Y+36}' r='.7' fill='#6b6a60' opacity='.5'/><circle cx='52' cy='{Y+52}' r='.7' fill='#6b6a60' opacity='.5'/><circle cx='82' cy='{Y+70}' r='.7' fill='#6b6a60' opacity='.5'/><circle cx='31' cy='{Y+42}' r='.7' fill='#6b6a60' opacity='.5'/></g>""")
T(64, Y - 6, "128²×3", 9, ANN, italic=True)
T(64, Y + 90, "camera frame", 10.5, ANN, 600)
T(64, Y + 102, "~1 m/px crop", 9.5, ANN)
T(64, Y + 114, "frozen contract", 8.5, ANN, italic=True)

# resize
arrow(102, CY, 116)
grid(118, CY - 15, 30, 8, INK, 0.35, 1.1)
grid(166, CY - 28, 56, 14, INK, 0.3, 1.2)
L(148, CY - 15, 166, CY - 28, INK, 0.55, 0.5)
L(148, CY + 15, 166, CY + 28, INK, 0.55, 0.5)
T(133, CY - 21, "128²", 9, CAP, italic=True)
T(194, CY - 34, "224²×3", 9, CAP, italic=True)
T(170, CY + 46, "resize to 224²", 10.5, INK, 600)
T(170, CY + 58, "bilinear, in-graph", 9.5)
T(170, CY + 70, "0 params · ImageNet's scale", 9, CAP, italic=True)

# split bus
MA, MB = CY - 140, CY + 120  # member centre lines: 170, 430
L(222, CY, 240, CY)
o.append(f"<circle cx='240' cy='{CY}' r='1.8' fill='{ANN}'/>")
L(240, MA, 240, MB)
arrow(240, MA, 260)
arrow(240, MB, 260)

RX, RY = 482, 0  # ring glyph centre x
BAR_X = 516
COS_X = 554
BEL_X = 576


def member(yc, tag):
    # trunk slabs (unchanged → ink)
    specs = [(262, 30, 10, "24×56²"), (318, 24, 14, "32×28²"), (368, 18, 18, "96×14²"), (414, 12, 22, "320×7²")]
    for x, s, d, lab in specs:
        slab(x, yc - s / 2, s, d)
        T(x + s / 2 + d * 0.46, yc - s / 2 - d * 0.5 - 6, lab, 9, CAP, italic=True)
    # kernel projections
    for (kx, ks, tx) in [(283, 6, 320), (334.2, 4.8, 370), (379, 4, 416)]:
        ky = yc - 5 if ks == 6 else (yc - 4.4 if ks == 4.8 else yc - 4)
        o.append(f"<rect x='{kx}' y='{ky:.1f}' width='{ks}' height='{ks}' fill='none' stroke='{INK}' stroke-width='.9'/>")
        o.append(f"<rect x='{tx}' y='{yc-2.5}' width='3' height='3' fill='{INK}' fill-opacity='.5'/>")
        for px in (kx, kx + ks):
            for py in (ky, ky + ks):
                L(px, py, tx + 1.5, yc - 1, INK, 0.55, 0.4)
    # ring glyph: lifted map with concentric rings (NEW, red)
    gs = 40
    gx, gy = RX - gs / 2, yc - gs / 2
    o.append(f"<rect x='{gx}' y='{gy}' width='{gs}' height='{gs}' fill='{RED}' fill-opacity='.04' stroke='{RED}' stroke-width='1.2'/>")
    for i in range(1, 7):
        L(gx + i * gs / 7, gy, gx + i * gs / 7, gy + gs, RED, 0.4, 0.3)
        L(gx, gy + i * gs / 7, gx + gs, gy + i * gs / 7, RED, 0.4, 0.3)
    o.append(f"<circle cx='{RX}' cy='{yc}' r='{gs/3:.1f}' fill='{RED}' fill-opacity='.06' stroke='{RED}' stroke-width='1.1'/>")
    o.append(f"<circle cx='{RX}' cy='{yc}' r='{gs/6:.1f}' fill='{RED}' fill-opacity='.12' stroke='{RED}' stroke-width='1.1'/>")
    # inputs into the ring pool: deep (direct), mid (L-route under)
    arrow(448, yc, gx - 1, RED)
    o.append(f"<path d='M 377,{yc+9} V {yc+30} H {RX} V {gy+gs+6}' fill='none' stroke='{RED}' stroke-width='0.9'/>")
    o.append(f"<path d='M {RX},{gy+gs+1} l -3,6 h 6 Z' fill='{RED}'/>")
    # bar 336 — 6 segments (3 rings × 2 depths)
    bh = 54
    by = yc - bh / 2
    o.append(f"<rect x='{BAR_X}' y='{by}' width='7' height='{bh}' fill='none' stroke='{RED}' stroke-width='1.2'/>")
    for k in range(1, 6):
        L(BAR_X, by + k * bh / 6, BAR_X + 7, by + k * bh / 6, RED, 0.5 if k != 3 else 0.9, 0.6)
    T(BAR_X + 3.5, by - 5, "336", 9, RED, italic=True)
    # converge lines: ring0 centre, ring1 point, ring2 corner → their bar segments
    srcs = [(RX, yc), (RX + gs / 4, yc - gs / 4 + 6), (gx + gs - 3, gy + 3)]
    for k, (sx, sy) in enumerate(srcs):
        for half in (0, 3):
            L(sx, sy, BAR_X, by + (k + half + 0.5) * bh / 6, RED, 0.55, 0.45)
    # cosine
    arrow(BAR_X + 9, yc, COS_X - 14)
    o.append(f"<circle cx='{COS_X}' cy='{yc}' r='12' fill='none' stroke='{INK}' stroke-width='1.1'/>")
    for a in (-70, -40, 10, 65, 130):
        r = math.radians(a)
        L(COS_X + 3.6 * math.cos(r), yc + 3.6 * math.sin(r), COS_X + 11 * math.cos(r), yc + 11 * math.sin(r), INK, 0.7, 0.5)
    L(COS_X, yc, COS_X + 10, yc + 7, INK, 1.4)
    o.append(f"<circle cx='{COS_X}' cy='{yc}' r='1.4' fill='{INK}'/>")
    arrow(COS_X + 14, yc, BEL_X - 2)
    # belief field
    bx, byy = BEL_X, yc - 28
    grid(bx, byy, 56, 8, INK, 0.5)
    for (cx, cy, op) in [(5, 2, .55), (4, 2, .28), (5, 1, .28), (6, 2, .18), (5, 3, .18), (2, 5, .12), (1, 6, .1)]:
        o.append(f"<rect x='{bx+cx*7}' y='{byy+cy*7}' width='7' height='7' fill='{INK}' fill-opacity='{op}'/>")
    T(bx + 28, byy - 8, f"belief field {tag}", 10.5, INK, 600)
    # captions
    T(354, yc + 44, f"MobileNetV2 trunk {tag}", 10.5, INK, 600)
    T(354, yc + 56, "17 blocks · 96×14², 320×7² → 56 ch", 9.5)
    T(COS_X, yc + 44, "cosine tile match", 10.5, INK, 600)
    T(COS_X, yc + 56, "336-d vs 2,970 protos, s=30", 9.5)
    T(RX - 20, yc + 74, "NEW — concentric-ring pooling", 10.5, RED, 600)
    T(RX - 20, yc + 86, "centre · middle · edge, replaces the global average", 9.5, RED)


member(MA, "A")
member(MB, "B")

# averaged belief
AX = 686
E = BEL_X + 56
L(E, MA, AX - 4, CY - 22, INK, 1)
o.append(f"<path d='M {AX},{CY-18} l -3.5,-6.5 l 5.5,1 Z' fill='{INK}'/>")
L(E, MB, AX - 4, CY + 22, INK, 1)
o.append(f"<path d='M {AX},{CY+18} l -3.5,6.5 l 5.5,-1 Z' fill='{INK}'/>")
grid(AX, CY - 28, 56, 8, INK, 0.5)
for (cx, cy, op) in [(5, 2, .5), (4, 2, .22), (5, 1, .22), (5, 3, .14)]:
    o.append(f"<rect x='{AX+cx*7}' y='{CY-28+cy*7}' width='7' height='7' fill='{INK}' fill-opacity='{op}'/>")
o.append(f"<rect x='{AX+28}' y='{CY-21}' width='21' height='21' fill='none' stroke='{INK}' stroke-width='1.8'/>")
T(AX, CY - 84, "averaged belief p̄", 10.5, INK, 600, "start")
T(AX, CY - 72, "mean of both softmaxes", 9.5, CAP, 400, "start")
for (cx, cy) in [(4.5, 1.5), (6.5, 1.5), (5.5, 2.5), (6.5, 3.5), (4.5, 3.5)]:
    L(AX + cx * 7, CY - 28 + cy * 7, 806, CY, INK, 0.5, 0.4)
T(776, CY - 50, "pooled 3×3 centroid", 10.5, INK, 600)
T(776, CY - 38, "+ mean offset nudge", 9.5)

# output (frozen)
L(805, CY, 819, CY, ANN, 1.4)
L(812, CY - 7, 812, CY + 7, ANN, 1.4)
o.append(f"<circle cx='812' cy='{CY}' r='3.9' fill='none' stroke='{ANN}' stroke-width='1.4'/>")
o.append(f"<circle cx='812' cy='{CY}' r='1.5' fill='{ANN}'/>")
T(828, CY - 16, "frozen contract", 8.5, ANN, italic=True, anchor="start")
o.append(f"<text id='frozen-output' x='828' y='{CY+5}' font-family='{F}' font-size='13' fill='{ANN}' font-weight='600' text-anchor='start'>(lat, lon, confidence)</text>")
T(828, CY + 21, "position fix + confidence", 9, ANN, anchor="start")

# confidence gate
GX, GY = 752, CY + 96
L(AX + 31.5, CY - 0, GX - 13, GY - 6, INK, 1, dash="2 4")
L(AX + 45.5, CY - 0, GX + 13, GY - 6, INK, 1, dash="2 4")
o.append(f"<path d='M {GX-14},{GY} A 14 14 0 0 1 {GX+14},{GY}' fill='none' stroke='{INK}' stroke-width='1.2'/>")
L(GX, GY, GX + 6.3, GY - 9, INK, 1.4)
o.append(f"<circle cx='{GX}' cy='{GY}' r='1.6' fill='{INK}'/>")
T(GX + 20, GY + 20, "confidence = min(mA, mB)", 10.5, INK, 600)
T(GX + 20, GY + 32, "both members must back the 3×3 block", 9.5)
T(GX + 20, GY + 43, "else the drone abstains", 8.5, CAP, italic=True)

# training lane
o.append(f"<path d='M 262,{MB+8} H 247 V 536 H 270' fill='none' stroke='{TRN}' stroke-width='1' stroke-dasharray='2 4'/>")
o.append(f"<path d='M 270,536 A 12 12 0 0 1 294,536' fill='none' stroke='{TRN}' stroke-width='1.2'/>")
o.append(f"<circle cx='282' cy='536' r='1.4' fill='{TRN}'/>")
T(304, 534, "MixStyle after blocks 3, 6, 10 · CosFace m=0.25 · trunk at 0.1× LR — unchanged from exp 16", 10, TRN, 600, "start")
T(304, 547, "paired three-survey batches, re-photographing augmentation, offset loss; MixStyle is identity at export", 9.5, TRN, 400, "start")
T(304, 566, "ring masks are fixed buffers, not trained · training views pass the same in-graph 128→224 resize", 9.5, TRN, 400, "start", italic=True)
T(972, 612, "only the pooling changed: global average → 3 concentric rings (heading-invariant, 0 params) · 112-d → 336-d", 9, ANN, anchor="end", italic=True)

svg = f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' role='img'>" + "".join(o) + "</svg>"
(HERE / "fig.svg").write_text(svg)
(HERE / "figure.json").write_text(json.dumps({"architecture_svg": svg}))
print("ok", len(svg))
