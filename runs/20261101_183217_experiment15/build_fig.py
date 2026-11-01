"""Experiment 15 architecture figure generator (scratch, this run only)."""
import json
from pathlib import Path

INK, MUT, FAINT, RED, OCH = "#111111", "#6b6a60", "#9b998c", "#8c2f1f", "#8a6a1e"
FONT = "Palatino,Georgia,serif"
W, H = 980, 620
C = H // 2  # 310
out = []


def t(x, y, s, size=10, color=MUT, w=400, anchor="middle", italic=False, tid=None, ls=None):
    a = f"id='{tid}' " if tid else ""
    st = " font-style='italic'" if italic else ""
    l = f" letter-spacing='{ls}'" if ls else ""
    out.append(f"<text {a}x='{x:.1f}' y='{y:.1f}' font-family='{FONT}' font-size='{size}' "
               f"fill='{color}' font-weight='{w}' text-anchor='{anchor}'{st}{l}>{s}</text>")


def cap(x, y, name, sub=None, color=INK, subcolor=MUT, anchor="middle"):
    t(x, y, name, 10.5, color, 600, anchor)
    if sub:
        t(x, y + 12, sub, 9.5, subcolor, 400, anchor)


def line(x1, y1, x2, y2, color=FAINT, w=1, dash=None, op=None):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    o = f" opacity='{op}'" if op else ""
    out.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}' "
               f"stroke-width='{w}'{d}{o}/>")


def path(d, color, w=1, dash=None):
    ds = f" stroke-dasharray='{dash}'" if dash else ""
    out.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='{w}'{ds}/>")


def harrow(x1, x2, y, color=FAINT):
    line(x1, y, x2 - 5, y, color)
    out.append(f"<path d='M {x2:.1f},{y:.1f} l -6,-3 v 6 Z' fill='{color}'/>")


def grid(x, y, s, n, color=INK, lw=1.2, op=.5):
    out.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{s}' height='{s}' fill='none' stroke='{color}' stroke-width='{lw}'/>")
    for i in range(1, n):
        p = i * s / n
        line(x + p, y, x + p, y + s, color, .45, op=op)
        line(x, y + p, x + s, y + p, color, .45, op=op)


def slab(x, yc, s, d, color=INK):
    y = yc - s / 2
    dx, dy = d * .92, d * .5
    out.append(f"<polygon points='{x:.1f},{y:.1f} {x+dx:.1f},{y-dy:.1f} {x+dx+s:.1f},{y-dy:.1f} {x+s:.1f},{y:.1f}' fill='#00000010' stroke='{color}' stroke-width='1'/>")
    out.append(f"<polygon points='{x+s:.1f},{y:.1f} {x+dx+s:.1f},{y-dy:.1f} {x+dx+s:.1f},{y-dy+s:.1f} {x+s:.1f},{y+s:.1f}' fill='#0000001c' stroke='{color}' stroke-width='1'/>")
    out.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{s}' height='{s}' fill='#00000006' stroke='{color}' stroke-width='1.2'/>")


def kernel(x1, y1, k, x2, y2, color=INK):
    """kernel square on slab face (x1,y1,k) projecting to a cell at (x2,y2)."""
    out.append(f"<rect x='{x1:.1f}' y='{y1:.1f}' width='{k}' height='{k}' fill='none' stroke='{color}' stroke-width='.9'/>")
    out.append(f"<rect x='{x2-1.5:.1f}' y='{y2-1.5:.1f}' width='3' height='3' fill='{color}' fill-opacity='.5'/>")
    for (a, b) in ((0, 0), (k, 0), (0, k), (k, k)):
        line(x1 + a, y1 + b, x2, y2, color, .55, op=.45)


def tickbar(x, yc, h, n, color=INK):
    out.append(f"<rect x='{x:.1f}' y='{yc-h/2:.1f}' width='7' height='{h}' fill='none' stroke='{color}' stroke-width='1.2'/>")
    for i in range(1, n):
        line(x, yc - h / 2 + i * h / n, x + 7, yc - h / 2 + i * h / n, color, .5, op=.5)


# ---- headers
t(8, 20, "INFERENCE PATH — WHAT FLIES", 9, FAINT, 600, "start", ls=1.8)
t(8, H - 8, "TRAINING SIGNALS — NEVER FLY", 9, OCH, 600, "start", ls=1.8)

# ---- frozen camera frame (verbatim glyph)
Y = C - 38
def yy(n): return f"{Y+n}"
out.append(
 f"<g id='cam-terrain'><rect id='frozen-input' x='26' y='{Y}' width='76' height='76' fill='#f6f4ea' stroke='#9b998c' stroke-width='1.6'/>"
 f"<path d='M31 {yy(47)} L97 {yy(27)}' stroke='#e6e3d4' stroke-width='5' fill='none'/>"
 f"<path d='M59 {yy(6)} L49 {yy(70)}' stroke='#e6e3d4' stroke-width='3.5' fill='none'/>"
 f"<rect x='34' y='{yy(9)}' width='12' height='8' fill='#d9d5c3' transform='rotate(-8 40 {yy(13)})'/>"
 f"<rect x='78' y='{yy(8)}' width='10' height='11' fill='#cfccbd'/>"
 f"<rect x='35' y='{yy(57)}' width='13' height='8' fill='#d9d5c3'/>"
 f"<rect x='75' y='{yy(50)}' width='10' height='9' fill='#cfccbd' transform='rotate(6 80 {yy(54)})'/>"
 f"<rect x='54' y='{yy(31)}' width='9' height='8' fill='#d9d5c3' opacity='.85'/>"
 f"<ellipse cx='86' cy='{yy(63)}' rx='8' ry='6' fill='#8a6a1e' opacity='.12'/>"
 f"<circle cx='40' cy='{yy(28)}' r='.7' fill='#6b6a60' opacity='.5'/>"
 f"<circle cx='70' cy='{yy(14)}' r='.7' fill='#6b6a60' opacity='.5'/>"
 f"<circle cx='92' cy='{yy(36)}' r='.7' fill='#6b6a60' opacity='.5'/>"
 f"<circle cx='52' cy='{yy(52)}' r='.7' fill='#6b6a60' opacity='.5'/>"
 f"<circle cx='82' cy='{yy(70)}' r='.7' fill='#6b6a60' opacity='.5'/>"
 f"<circle cx='31' cy='{yy(42)}' r='.7' fill='#6b6a60' opacity='.5'/></g>")
t(64, Y - 6, "128²×3", 9, FAINT, italic=True)
cap(64, C + 52, "camera frame", "~1 m/px crop", FAINT, FAINT)
t(64, C + 76, "frozen contract", 8.5, FAINT, italic=True)

# ---- NEW: in-graph resize 128 -> 224
harrow(102, 116, C)
sx, ss = 118, 30
grid(sx, C - ss / 2, ss, 8, RED, 1.1, .35)
bx, bs = 166, 56
grid(bx, C - bs / 2, bs, 14, RED, 1.2, .3)
for b in (0, 1):
    line(sx + ss, C - ss / 2 + b * ss, bx, C - bs / 2 + b * bs, RED, .55, op=.5)
t(sx + ss / 2, C - ss / 2 - 6, "128²", 9, RED, italic=True)
t(bx + bs / 2, C - bs / 2 - 6, "224²×3", 9, RED, italic=True)
cap(170, C + 46, "NEW — resize to 224²", "bilinear, in-graph", RED, RED)
t(170, C + 70, "0 params · ImageNet's scale", 9, RED, italic=True)

# ---- split to two members
J = 240
line(bx + bs, C, J, C, FAINT)
out.append(f"<circle cx='{J}' cy='{C}' r='1.8' fill='{FAINT}'/>")
YA, YB = C - 140, C + 120
line(J, YA, J, YB, FAINT)
harrow(J, 260, YA)
harrow(J, 260, YB)


def member(yc, tag):
    # trunk slabs: (x, s, d, shape)
    slabs = [(262, 30, 10, "16×112²"), (318, 24, 14, "24×28²"), (368, 18, 18, "48×14²"), (414, 12, 22, "96×7²")]
    for (x, s, d, lab) in slabs:
        slab(x, yc, s, d)
        t(x + s / 2 + d * .46, yc - s / 2 - d * .5 - 6, lab, 9, RED, italic=True)
    # kernel projections between consecutive slabs
    for (a, b) in zip(slabs, slabs[1:]):
        (x1, s1, _, _), (x2, s2, _, _) = a, b
        k = max(4, s1 / 5)
        kernel(x1 + s1 - k - 3, yc - k / 2 - 2, k, x2 + 3.5, yc - 1, INK)
    # two read depths: mid (slab3) tapped from below, deep (slab4) straight
    gx = 456
    path(f"M {368+9},{yc+9} V {yc+22} H {gx} V {yc+2.5}", INK, .9)
    line(414 + 12 + 1, yc, gx - 2, yc, INK, .9)
    out.append(f"<circle cx='{gx}' cy='{yc}' r='2.2' fill='{INK}'/>")
    t(gx + 2, yc + 34, "", 1)  # placeholder (no-op)
    harrow(gx + 3, 470, yc)
    tickbar(471, yc, 46, 8)
    t(474.5, yc - 28, "112", 9, MUT, italic=True)
    harrow(480, 496, yc)
    # cosine compass
    cx = 510
    out.append(f"<circle cx='{cx}' cy='{yc}' r='12' fill='none' stroke='{INK}' stroke-width='1.1'/>")
    for ang in (-70, -40, 10, 65, 130):
        import math
        r = math.radians(ang)
        line(cx + 3.6 * math.cos(r), yc + 3.6 * math.sin(r), cx + 11 * math.cos(r), yc + 11 * math.sin(r), INK, .7, op=.5)
    line(cx, yc, cx + 10, yc + 7, INK, 1.4)
    out.append(f"<circle cx='{cx}' cy='{yc}' r='1.4' fill='{INK}'/>")
    harrow(524, 534, yc)
    # belief grid
    gx0, gs = 536, 56
    grid(gx0, yc - gs / 2, gs, 8)
    for (i, j, o) in ((5, 2, .55), (4, 2, .28), (5, 1, .28), (6, 2, .18), (5, 3, .18), (2, 5, .12), (1, 6, .1)):
        out.append(f"<rect x='{gx0+i*7:.1f}' y='{yc-gs/2+j*7:.1f}' width='7' height='7' fill='{INK}' fill-opacity='{o}'/>")
    t(gx0 + gs / 2, yc - gs / 2 - 8, f"belief field {tag}", 10.5, INK, 600)
    # captions
    cap(362, yc + 44, f"trunk {tag} (pretrained), read at 2 depths", "11 MobileNetV3-S blocks · GAP of 14², 7²")
    cap(538, yc + 44, "cosine tile match", "112-d vs 2,970 protos, s=30")
    return gx0 + gs


ea = member(YA, "A")
eb = member(YB, "B")

# ---- average on centre line
ax, asz = 646, 56
line(ea, YA, ax - 4, C - 22, INK)
out.append(f"<path d='M {ax-1},{C-18} l -3.5,-6.5 l 5.5,1 Z' fill='{INK}'/>")
line(eb, YB, ax - 4, C + 22, INK)
out.append(f"<path d='M {ax-1},{C+18} l -3.5,6.5 l 5.5,-1 Z' fill='{INK}'/>")
grid(ax, C - asz / 2, asz, 8)
for (i, j, o) in ((5, 2, .5), (4, 2, .22), (5, 1, .22), (5, 3, .14)):
    out.append(f"<rect x='{ax+i*7:.1f}' y='{C-asz/2+j*7:.1f}' width='7' height='7' fill='{INK}' fill-opacity='{o}'/>")
bxk, byk = ax + 4 * 7, C - asz / 2 + 1 * 7
out.append(f"<rect x='{bxk}' y='{byk}' width='21' height='21' fill='none' stroke='{INK}' stroke-width='1.8'/>")
cap(ax, C - asz / 2 - 42, "averaged belief p̄", "mean of both softmaxes", anchor="start")
# converge from the 3x3 block cells to crosshair
for (px, py) in ((bxk + 3.5, byk + 3.5), (bxk + 17.5, byk + 3.5), (bxk + 10.5, byk + 10.5), (bxk + 17.5, byk + 17.5), (bxk + 3.5, byk + 17.5)):
    line(px, py, 806, C, INK, .5, op=.4)
cap(752, C - 40, "pooled 3×3 centroid", "+ mean offset nudge")
# crosshair + frozen output
line(805, C, 819, C, FAINT, 1.4)
line(812, C - 7, 812, C + 7, FAINT, 1.4)
out.append(f"<circle cx='812' cy='{C}' r='3.9' fill='none' stroke='{FAINT}' stroke-width='1.4'/>")
out.append(f"<circle cx='812' cy='{C}' r='1.5' fill='{FAINT}'/>")
t(828, C - 16, "frozen contract", 8.5, FAINT, italic=True, anchor="start")
out.append(f"<text id='frozen-output' x='828' y='{C+5}' font-family='{FONT}' font-size='13' fill='{FAINT}' font-weight='600' text-anchor='start'>(lat, lon, confidence)</text>")
t(828, C + 21, "position fix + confidence", 9, FAINT, anchor="start")
# confidence gauge
gcx, gcy = 712, C + 92
line(bxk + 4, byk + 21, gcx - 13, gcy - 6, INK, 1, "2 4")
line(bxk + 18, byk + 21, gcx + 13, gcy - 6, INK, 1, "2 4")
path(f"M {gcx-14},{gcy} A 14 14 0 0 1 {gcx+14},{gcy}", INK, 1.2)
line(gcx, gcy, gcx + 6.3, gcy - 9, INK, 1.4)
out.append(f"<circle cx='{gcx}' cy='{gcy}' r='1.6' fill='{INK}'/>")
cap(gcx, gcy + 20, "confidence = min(mA, mB)", "both members must back the 3×3 block")
t(gcx, gcy + 43, "else the drone abstains", 8.5, MUT, italic=True)

# ---- training band
TB = 536
lx = 247
path(f"M 262,{YB+8} H {lx} V {TB} H 270", OCH, 1, "2 4")
path(f"M 270,{TB} A 12 12 0 0 1 294,{TB}", OCH, 1.2)
out.append(f"<circle cx='282' cy='{TB}' r='1.4' fill='{OCH}'/>")
t(304, TB - 2, "MixStyle after blocks 1, 3, 6 · CosFace margin m=0.25 · trunk at 0.1× LR — unchanged from exp 14", 10, OCH, 600, "start")
t(304, TB + 11, "paired three-survey batches, re-photographing augmentation, offset loss; MixStyle is identity at export", 9.5, OCH, 400, "start")
t(304, TB + 30, "NEW (by construction) — training views pass the same in-graph 128→224 resize, so train and flight see one scale", 9.5, RED, 400, "start", italic=True)
t(972, H - 8, "trunk weights, decode, gate, losses unchanged · ONNX I/O still 1×3×128²", 9, FAINT, 400, "end", italic=True)

svg = (f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' role='img'>" + "".join(o for o in out if "font-size='1'" not in o) + "</svg>")
here = Path(__file__).parent
(here / "fig.svg").write_text(svg)
(here / "figure.json").write_text(json.dumps({"architecture_svg": svg}))
print(len(svg))
