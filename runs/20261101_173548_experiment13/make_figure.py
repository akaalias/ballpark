"""Experiment 13 architecture figure: deepens each committee member's
pretrained trunk from features[0:9] to features[0:11], reading TWO depth
levels (48ch@8x8, unchanged; 96ch@4x4, new) and concatenating their lifted
GAP vectors into the same 112-d descriptor. Everything downstream (cosine
head, belief pooling, offset head, confidence gate) is unchanged from
experiment 12. Standalone script, run locally only to emit figure.json."""
import json
import math
from pathlib import Path

INK, MUT, FAINT, ACC, OCH = "#111111", "#6b6a60", "#9b998c", "#8c2f1f", "#8a6a1e"
FONT = "Palatino,Georgia,serif"


def txt(x, y, s, size=10, color=MUT, w=400, anchor="middle", style="", tid=None):
    tid_attr = f"id='{tid}' " if tid else ""
    xs = "828" if abs(x - 828) < 0.05 else f"{x:.1f}"
    return (f"<text {tid_attr}x='{xs}' y='{y:.1f}' font-family='{FONT}' font-size='{size}' "
            f"fill='{color}' font-weight='{w}' text-anchor='{anchor}' {style}>{s}</text>")


def cap(xc, y, name, sub=None, color=MUT, name_color=None):
    out = [txt(xc, y, name, 10.5, name_color or INK, 600)]
    if sub:
        out.append(txt(xc, y + 12, sub, 9.5, color))
    return "".join(out)


def harrow(x1, x2, y, label=None, color=FAINT):
    out = [f"<line x1='{x1:.1f}' y1='{y:.1f}' x2='{x2 - 5:.1f}' y2='{y:.1f}' stroke='{color}' stroke-width='1'/>",
           f"<path d='M {x2:.1f},{y:.1f} l -6,-3 v 6 Z' fill='{color}'/>"]
    if label:
        out.append(txt((x1 + x2) / 2, y - 6, label, 9, FAINT, style="font-style='italic'"))
    return "".join(out)


def diag_arrow(x1, y1, x2, y2, color=FAINT):
    ang = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - 7 * math.cos(ang), y2 - 7 * math.sin(ang)
    perp = ang + math.pi / 2
    ax1 = bx + 4 * math.cos(perp) - 3 * math.cos(ang)
    ay1 = by + 4 * math.sin(perp) - 3 * math.sin(ang)
    ax2 = bx - 4 * math.cos(perp) - 3 * math.cos(ang)
    ay2 = by - 4 * math.sin(perp) - 3 * math.sin(ang)
    return (f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{bx:.1f}' y2='{by:.1f}' stroke='{color}' stroke-width='1'/>"
            f"<path d='M {x2:.1f},{y2:.1f} L {ax1:.1f},{ay1:.1f} L {ax2:.1f},{ay2:.1f} Z' fill='{color}'/>")


def leader(x1, y1, x2, y2, color):
    return (f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}' "
            f"stroke-width='1' stroke-dasharray='2 4'/>")


def slab(x, yc, s, d, color=INK):
    y = yc - s / 2
    dx, dy = d, d * 0.55
    out = [
        f"<polygon points='{x:.1f},{y:.1f} {x+dx:.1f},{y-dy:.1f} {x+dx+s:.1f},{y-dy:.1f} {x+s:.1f},{y:.1f}' "
        f"fill='#00000010' stroke='{color}' stroke-width='1'/>",
        f"<polygon points='{x+s:.1f},{y:.1f} {x+dx+s:.1f},{y-dy:.1f} {x+dx+s:.1f},{y-dy+s:.1f} {x+s:.1f},{y+s:.1f}' "
        f"fill='#0000001c' stroke='{color}' stroke-width='1'/>",
        f"<rect x='{x:.1f}' y='{y:.1f}' width='{s}' height='{s}' fill='#00000006' "
        f"stroke='{color}' stroke-width='1.2'/>",
    ]
    return "".join(out), s + dx, x + s, y


def kernel_proj(x1, s1, yc1, x2, s2, yc2, color):
    k = max(5, s1 * 0.30)
    kx, ky = x1 + s1 * 0.60, yc1 - s1 / 2 + s1 * 0.18
    c = max(3, s2 * 0.14)
    cx, cy = x2 + s2 * 0.22, yc2 - s2 / 2 + s2 * 0.26
    out = [f"<rect x='{kx:.1f}' y='{ky:.1f}' width='{k:.1f}' height='{k:.1f}' "
           f"fill='none' stroke='{color}' stroke-width='0.9'/>",
           f"<rect x='{cx:.1f}' y='{cy:.1f}' width='{c:.1f}' height='{c:.1f}' "
           f"fill='{color}' fill-opacity='0.5' stroke='none'/>"]
    for (px, py) in ((kx, ky), (kx + k, ky), (kx, ky + k), (kx + k, ky + k)):
        out.append(f"<line x1='{px:.1f}' y1='{py:.1f}' x2='{cx + c / 2:.1f}' "
                   f"y2='{cy + c / 2:.1f}' stroke='{color}' stroke-width='0.55' "
                   f"opacity='0.45'/>")
    return "".join(out)


def slab_chain(x0, yc, sizes, color, gap=13, prev_face=None):
    out, x, faces = [], x0, []
    for (s, d) in sizes:
        g, w, fx, _ = slab(x, yc, s, d, color)
        out.append(g)
        faces.append((fx - s, s))
        x += w + gap
    x -= gap
    if prev_face:
        out.append(kernel_proj(prev_face[0], prev_face[1], prev_face[2], faces[0][0], faces[0][1], yc, prev_face[3]))
    for (x1, s1), (x2, s2) in zip(faces, faces[1:]):
        out.append(kernel_proj(x1, s1, yc, x2, s2, yc, MUT if color == INK else color))
    return "".join(out), x, (faces[-1][0], faces[-1][1], yc, color)


def bar(x, yc, h, color=INK, ticks=7):
    y = yc - h / 2
    out = [f"<rect x='{x:.1f}' y='{y:.1f}' width='7' height='{h:.1f}' fill='none' "
           f"stroke='{color}' stroke-width='1.2'/>"]
    for i in range(1, ticks):
        yy = y + i * h / ticks
        out.append(f"<line x1='{x:.1f}' y1='{yy:.1f}' x2='{x+7:.1f}' "
                   f"y2='{yy:.1f}' stroke='{color}' stroke-width='0.5' opacity='0.5'/>")
    return "".join(out), 7


def split_bar(x, yc, h_top, h_bot, color_top=INK, color_bot=ACC, ticks=6):
    y_top = yc - (h_top + h_bot) / 2
    y_bot = y_top + h_top
    out = [f"<rect x='{x:.1f}' y='{y_top:.1f}' width='7' height='{h_top:.1f}' fill='none' "
           f"stroke='{color_top}' stroke-width='1.2'/>",
           f"<rect x='{x:.1f}' y='{y_bot:.1f}' width='7' height='{h_bot:.1f}' fill='none' "
           f"stroke='{color_bot}' stroke-width='1.2'/>"]
    for i in range(1, ticks):
        yy = y_top + i * h_top / ticks
        out.append(f"<line x1='{x:.1f}' y1='{yy:.1f}' x2='{x+7:.1f}' y2='{yy:.1f}' "
                   f"stroke='{color_top}' stroke-width='0.5' opacity='0.5'/>")
        yy2 = y_bot + i * h_bot / ticks
        out.append(f"<line x1='{x:.1f}' y1='{yy2:.1f}' x2='{x+7:.1f}' y2='{yy2:.1f}' "
                   f"stroke='{color_bot}' stroke-width='0.5' opacity='0.5'/>")
    out.append(f"<line x1='{x:.1f}' y1='{y_bot:.1f}' x2='{x+7:.1f}' y2='{y_bot:.1f}' "
               f"stroke='{MUT}' stroke-width='0.7'/>")
    return "".join(out), 7


def gridsq(x, yc, s, n, color=INK, bumps=(), lw=1.2):
    y = yc - s / 2
    out = [f"<rect x='{x:.1f}' y='{y:.1f}' width='{s}' height='{s}' fill='none' "
           f"stroke='{color}' stroke-width='{lw}'/>"]
    for i in range(1, n):
        out.append(f"<line x1='{x + i * s / n:.1f}' y1='{y:.1f}' x2='{x + i * s / n:.1f}' "
                   f"y2='{y + s:.1f}' stroke='{color}' stroke-width='0.45' opacity='0.5'/>")
        out.append(f"<line x1='{x:.1f}' y1='{y + i * s / n:.1f}' x2='{x + s:.1f}' "
                   f"y2='{y + i * s / n:.1f}' stroke='{color}' stroke-width='0.45' opacity='0.5'/>")
    for (gx, gy, o) in bumps:
        out.append(f"<rect x='{x + gx * s / n:.1f}' y='{y + gy * s / n:.1f}' "
                   f"width='{s / n:.1f}' height='{s / n:.1f}' fill='{color}' fill-opacity='{o}'/>")
    return "".join(out), s


BUMP = ((5, 2, .55), (4, 2, .28), (5, 1, .28), (6, 2, .18), (5, 3, .18), (2, 5, .12), (1, 6, .1))


def crosspt(x, y, color, r=7):
    return (f"<line x1='{x - r}' y1='{y}' x2='{x + r}' y2='{y}' stroke='{color}' stroke-width='1.4'/>"
            f"<line x1='{x}' y1='{y - r}' x2='{x}' y2='{y + r}' stroke='{color}' stroke-width='1.4'/>"
            f"<circle cx='{x}' cy='{y}' r='{r * 0.55:.1f}' fill='none' stroke='{color}' stroke-width='1.4'/>"
            f"<circle cx='{x}' cy='{y}' r='1.5' fill='{color}'/>")


def cam_terrain(y_top):
    Y = y_top
    return (f"<g id='cam-terrain'><rect id='frozen-input' x='26' y='{Y:.1f}' width='76' "
            f"height='76' fill='#f6f4ea' stroke='#9b998c' stroke-width='1.6'/>"
            f"<path d='M31 {Y+47:.1f} L97 {Y+27:.1f}' stroke='#e6e3d4' stroke-width='5' fill='none'/>"
            f"<path d='M59 {Y+6:.1f} L49 {Y+70:.1f}' stroke='#e6e3d4' stroke-width='3.5' fill='none'/>"
            f"<rect x='34' y='{Y+9:.1f}' width='12' height='8' fill='#d9d5c3' transform='rotate(-8 40 {Y+13:.1f})'/>"
            f"<rect x='78' y='{Y+8:.1f}' width='10' height='11' fill='#cfccbd'/>"
            f"<rect x='35' y='{Y+57:.1f}' width='13' height='8' fill='#d9d5c3'/>"
            f"<rect x='75' y='{Y+50:.1f}' width='10' height='9' fill='#cfccbd' transform='rotate(6 80 {Y+54:.1f})'/>"
            f"<rect x='54' y='{Y+31:.1f}' width='9' height='8' fill='#d9d5c3' opacity='.85'/>"
            f"<ellipse cx='86' cy='{Y+63:.1f}' rx='8' ry='6' fill='#8a6a1e' opacity='.12'/>"
            f"<circle cx='40' cy='{Y+28:.1f}' r='.7' fill='#6b6a60' opacity='.5'/>"
            f"<circle cx='70' cy='{Y+14:.1f}' r='.7' fill='#6b6a60' opacity='.5'/>"
            f"<circle cx='92' cy='{Y+36:.1f}' r='.7' fill='#6b6a60' opacity='.5'/>"
            f"<circle cx='52' cy='{Y+52:.1f}' r='.7' fill='#6b6a60' opacity='.5'/>"
            f"<circle cx='82' cy='{Y+70:.1f}' r='.7' fill='#6b6a60' opacity='.5'/>"
            f"<circle cx='31' cy='{Y+42:.1f}' r='.7' fill='#6b6a60' opacity='.5'/></g>")


# ---------------------------------------------------------------- geometry
TRUNK_X0 = 164.0             # trunk start, right of the cam-caption clearance zone
Y0_A, Y1_A = 92.0, 190.0     # member A: mid lane / deep sub-lane (needs >=82px gap for 2 caption bands)
Y0_B, Y1_B = 300.0, 398.0    # member B: mid lane / deep sub-lane
YC_A = (Y0_A + Y1_A) / 2     # 141
YC_B = (Y0_B + Y1_B) / 2     # 349
CAM_CY = (YC_A + YC_B) / 2   # 245
CAM_Y_TOP = CAM_CY - 38

body = []
body.append(txt(8, 20, "INFERENCE PATH — WHAT FLIES", 9, FAINT, 600, "start", "letter-spacing='1.8'"))
body.append(txt(8, 592, "TRAINING SIGNALS — NEVER FLY", 9, OCH, 600, "start", "letter-spacing='1.8'"))

body.append(cam_terrain(CAM_Y_TOP))
body.append(txt(64, CAM_Y_TOP - 6, "128²×3", 9, FAINT))
body.append(txt(64, CAM_Y_TOP + 86, "camera frame", 10.5, INK, 600))
body.append(txt(64, CAM_Y_TOP + 98, "one TRAIN survey", 9.5, MUT))
body.append(txt(64, CAM_Y_TOP + 110, "frozen contract", 8.5, FAINT, style="font-style='italic'"))

body.append(diag_arrow(102, CAM_CY, 128, Y0_A, FAINT))
body.append(diag_arrow(102, CAM_CY, 128, Y0_B, FAINT))
body.append(txt(119, CAM_CY - 86, "splits —", 8.5, FAINT, style="font-style='italic'"))
body.append(txt(119, CAM_CY - 75, "no shared", 8.5, FAINT, style="font-style='italic'"))
body.append(txt(119, CAM_CY - 64, "weights", 8.5, FAINT, style="font-style='italic'"))


def member(y0, y1, yc, letter):
    b = []
    g, x_end, mid_face = slab_chain(TRUNK_X0, y0, [(22, 6), (16, 8), (11, 10), (8, 12)], INK)
    b.append(g)
    b.append(txt((TRUNK_X0 + x_end) / 2, y0 - 30, "feat[0:9]·8²×48", 8, FAINT))
    b.append(cap((TRUNK_X0 + x_end) / 2, y0 + 38, f"trunk {letter} (pretrained)"))
    x_tap = x_end + 7
    b.append(f"<circle cx='{x_tap:.1f}' cy='{y0:.1f}' r='2' fill='{INK}'/>")

    # branch 1 (ink, unchanged): on to lift_mid -> GAP(56-d) at y0
    x = x_tap + 24
    b.append(harrow(x_tap, x, y0))
    g, w, _, _ = slab(x, y0, 11, 5, INK)
    b.append(g)
    b.append(txt(x + 2, y0 + 38, "lift 48→56", 10.5, INK, 600, "start"))
    x2 = x + w + 26
    b.append(harrow(x + w + 2, x2, y0))
    g, bw = bar(x2, y0, 40, INK)
    b.append(g)
    b.append(txt(x2 + 3, y0 - 30, "56d GAP", 8, FAINT))
    mid_vec_x = x2

    # branch 2 (red, NEW): fork down to deep sub-lane, blocks 9-10, lift_deep, GAP(56-d)
    xf = x_tap + 15
    b.append(diag_arrow(x_tap, y0 + 4, xf, y1 - 4, ACC))
    g, x_end2, _ = slab_chain(xf + 4, y1, [(9, 9), (6, 11)], ACC,
                               prev_face=(mid_face[0], mid_face[1], y0, ACC))
    b.append(g)
    b.append(txt((xf + x_end2) / 2, y1 - 30, "NEW feat[9:11]·4²×96", 8, ACC))
    b.append(cap((xf + x_end2) / 2, y1 + 38, "NEW: blocks 9–10", name_color=ACC))
    xd = x_end2 + 7
    b.append(f"<circle cx='{xd:.1f}' cy='{y1:.1f}' r='2' fill='{ACC}'/>")
    xdl = xd + 22
    b.append(harrow(xd, xdl, y1, color=ACC))
    g, wd, _, _ = slab(xdl, y1, 9, 4, ACC)
    b.append(g)
    b.append(txt(xdl + 2, y1 + 38, "NEW lift 96→56", 10.5, ACC, 600, "start"))
    xd2 = xdl + wd + 24
    b.append(harrow(xdl + wd + 2, xd2, y1, color=ACC))
    g, bw2 = bar(xd2, y1, 40, ACC)
    b.append(g)
    b.append(txt(xd2 + 3, y1 - 30, "56d GAP", 8, ACC))
    deep_vec_x = xd2

    # concat: both 56-d vectors converge into one 112-d split bar at yc
    xcat = max(mid_vec_x, deep_vec_x) + 32
    b.append(f"<line x1='{mid_vec_x+8:.1f}' y1='{y0:.1f}' x2='{xcat-6:.1f}' y2='{yc-10:.1f}' stroke='{INK}' stroke-width='0.8'/>")
    b.append(f"<line x1='{deep_vec_x+8:.1f}' y1='{y1:.1f}' x2='{xcat-6:.1f}' y2='{yc+10:.1f}' stroke='{ACC}' stroke-width='0.8'/>")
    g, bwc = split_bar(xcat, yc, 40, 40, INK, ACC)
    b.append(g)
    b.append(cap(xcat + 3, yc - 42, "112-d", name_color=ACC))
    b.append(txt(xcat + 3, yc + 56, "concat: 56(8²)+56(4²)", 8.5, ACC))

    # unchanged downstream: BN -> unit embed/cosine -> belief field
    x = xcat + 28
    b.append(harrow(xcat + bwc + 3, x, yc))
    g, wn = bar(x, yc, 20, INK, ticks=3)
    b.append(g)
    b.append(txt(x + 3.5, yc - 24, "BN", 9, INK, 600))
    x2 = x + wn + 24
    b.append(harrow(x + wn + 2, x2, yc))
    cxc = x2 + 12
    b.append(f"<circle cx='{cxc:.1f}' cy='{yc:.1f}' r='12' fill='none' stroke='{INK}' stroke-width='1.1'/>")
    for (ax1, ay1, ax2, ay2) in ((3.8, -10.3, 5.5, -15), (8.4, -7.1, 12.3, -10.3),
                                 (10.3, 3.8, 15, 5.5), (-1.9, 10.8, -2.8, 15.8),
                                 (-9.5, 5.5, -13.9, 8)):
        b.append(f"<line x1='{cxc+ax1:.1f}' y1='{yc+ay1:.1f}' x2='{cxc+ax2:.1f}' y2='{yc+ay2:.1f}' "
                 f"stroke='{INK}' stroke-width='0.7' opacity='0.5'/>")
    b.append(f"<line x1='{cxc:.1f}' y1='{yc:.1f}' x2='{cxc+10:.1f}' y2='{yc+7:.1f}' stroke='{INK}' stroke-width='1.4'/>")
    b.append(f"<circle cx='{cxc:.1f}' cy='{yc:.1f}' r='1.4' fill='{INK}'/>")
    b.append(txt(cxc, yc - 40, "unit embed.→cos", 9.5, INK, 600))
    b.append(txt(cxc, yc - 28, "2970 protos", 8.5, MUT))
    x3 = cxc + 46
    b.append(harrow(x2 + 24, x3, yc))
    gx = x3 + 10
    g, gs = gridsq(gx, yc, 58, 8, INK, BUMP)
    b.append(g)
    b.append(txt(gx + 29, yc - 40, f"belief field {letter}", 10.5, INK, 600))
    b.append(txt(gx + 29, yc - 28, "cosine×s=30", 8.5, MUT))
    return b, gx, gs, yc


bA, gxA, gsA, ycA = member(Y0_A, Y1_A, YC_A, "A")
body += bA
bB, gxB, gsB, ycB = member(Y0_B, Y1_B, YC_B, "B")
body += bB

# ---- average, pooled centroid, confidence gate, frozen output (unchanged)
mx = gxA + gsA + 56
body.append(f"<line x1='{gxA+gsA:.1f}' y1='{ycA:.1f}' x2='{mx-6:.1f}' y2='{(ycA+ycB)/2-9:.1f}' stroke='{INK}' stroke-width='1'/>")
body.append(f"<path d='M {mx:.1f},{(ycA+ycB)/2-15:.1f} l -5,2.5 l 0,-5 Z' fill='{INK}'/>")
body.append(f"<line x1='{gxB+gsB:.1f}' y1='{ycB:.1f}' x2='{mx-6:.1f}' y2='{(ycA+ycB)/2+9:.1f}' stroke='{INK}' stroke-width='1'/>")
body.append(f"<path d='M {mx:.1f},{(ycA+ycB)/2+15:.1f} l -5,-2.5 l 0,5 Z' fill='{INK}'/>")
mcy = (ycA + ycB) / 2
body.append(txt(mx + 6, mcy - 46, "average", 9, INK, 600, "start", style="font-style='italic'"))
g, gsm = gridsq(mx + 4, mcy, 58, 8, INK, ((5, 2, .5), (4, 2, .22), (5, 1, .22), (5, 3, .14)))
body.append(g)
body.append(txt(mx + 4, mcy - 68, "averaged belief p̄", 10.5, INK, 600, "start"))
body.append(txt(mx + 4, mcy - 32, "mean of both members' softmax", 9.5, INK, 400, "start"))

winx = mx + 4 + 3 * 58 / 8
winy = mcy - 29 + 3 * 58 / 8
body.append(f"<rect x='{winx:.1f}' y='{winy:.1f}' width='{2*58/8:.1f}' height='{2*58/8:.1f}' fill='none' stroke='{INK}' stroke-width='1.8'/>")

tx = 828 - 16
cvy = mcy
for dxp, dyp in ((0, 0), (1, 0), (0, 1), (1.2, 1.2)):
    sx, sy = winx + dxp * 58 / 8 + 7, winy + dyp * 58 / 8 + 7
    body.append(f"<line x1='{sx:.1f}' y1='{sy:.1f}' x2='{tx-10:.1f}' y2='{cvy:.1f}' stroke='{INK}' stroke-width='0.5' opacity='0.4'/>")
body.append(f"<line x1='{tx-10:.1f}' y1='{cvy-10:.1f}' x2='{tx+10:.1f}' y2='{cvy+10:.1f}' stroke='{INK}' stroke-width='1.4'/>")
body.append(crosspt(tx, cvy, INK))
body.append(txt(mx + 4 + 58 + 90, mcy - 42, "pooled centroid of the 3×3 block", 10.5, INK, 600, "start"))
body.append(txt(mx + 4 + 58 + 90, mcy - 30, "+ mean of both members' offset nudges", 9.5, INK, 400, "start"))

body.append(txt(828, cvy - 18, "frozen contract", 8.5, FAINT, anchor="start", style="font-style='italic'"))
body.append(txt(828, cvy - 2, "(lat, lon, confidence)", 13, MUT, 600, "start", tid="frozen-output"))
body.append(txt(828, cvy + 12, "position fix + confidence", 9, FAINT, anchor="start"))

# confidence gate
gcx, gcy = mx + 4 + 58 / 2, mcy + 100
body.append(leader(mx + 4 + 20, mcy + 29, gcx - 14, gcy - 10, INK))
body.append(leader(mx + 4 + 38, mcy + 29, gcx + 14, gcy - 10, INK))
body.append(f"<path d='M {gcx-14:.1f},{gcy:.1f} A 14 14 0 0 1 {gcx+14:.1f},{gcy:.1f}' fill='none' stroke='{INK}' stroke-width='1.2'/>")
body.append(f"<line x1='{gcx:.1f}' y1='{gcy:.1f}' x2='{gcx+6.3:.1f}' y2='{gcy-9:.1f}' stroke='{INK}' stroke-width='1.4'/>")
body.append(f"<circle cx='{gcx:.1f}' cy='{gcy:.1f}' r='1.6' fill='{INK}'/>")
body.append(cap(gcx, gcy + 22, "confidence = min(mᴀ, mʙ)", "both must independently back it"))
body.append(txt(gcx, gcy + 47, "or the drone abstains", 8.3, INK))
body.append(f"<path d='M {gcx:.1f},{gcy+54:.1f} V {gcy+74:.1f} H {tx+40:.1f} V {cvy+16:.1f}' fill='none' stroke='{INK}' stroke-width='0.8'/>")
body.append(f"<path d='M {tx+40:.1f},{cvy+10:.1f} l -3,6 h 6 Z' fill='{INK}'/>")
body.append(txt((gcx + tx + 40) / 2, gcy + 86, "may abstain instead of guessing", 8.5, INK, style="font-style='italic'"))

# ---- training lane notes
ty1 = 560
body.append(f"<circle cx='{gxA+gsA/2:.1f}' cy='{YC_A-54:.1f}' r='1.6' fill='{OCH}'/>")
body.append(f"<path d='M {gxA+gsA/2:.1f},{YC_A-54:.1f} V {YC_A-80:.1f} H 108 V {ty1:.1f} H 206' fill='none' stroke='{OCH}' stroke-width='1' stroke-dasharray='2 4'/>")
body.append(f"<path d='M 193,{ty1:.1f} A 13 13 0 0 1 219,{ty1:.1f}' fill='none' stroke='{OCH}' stroke-width='1.2'/>")
body.append(f"<line x1='206' y1='{ty1:.1f}' x2='198.5' y2='{ty1-10.6:.1f}' stroke='{OCH}' stroke-width='1.3'/>")
body.append(f"<line x1='206' y1='{ty1:.1f}' x2='210.4' y2='{ty1-12.2:.1f}' stroke='{OCH}' stroke-width='1.3' stroke-dasharray='1.5 1.5'/>")
body.append(f"<circle cx='206' cy='{ty1:.1f}' r='1.4' fill='{OCH}'/>")
body.append(txt(236, ty1 - 18, "additive cosine margin m=0.25 on the target cell (unchanged)", 10, OCH, 600, "start"))
body.append(txt(236, ty1 - 6, "CosFace-style, subtracted from cos before softmax — shared head, not this experiment", 9.5, OCH, 400, "start"))

ty2 = 606
body.append(f"<circle cx='{TRUNK_X0+10:.1f}' cy='{Y0_A-54:.1f}' r='1.6' fill='{ACC}'/>")
body.append(f"<path d='M {TRUNK_X0+10:.1f},{Y0_A-54:.1f} V {ty2:.1f} H 222.0' fill='none' stroke='{ACC}' stroke-width='1' stroke-dasharray='2 4'/>")
body.append(txt(236, ty2 - 4, "all eleven blocks fine-tuned at 0.1× the head's LR, same cosine schedule", 10, ACC, 600, "start"))
body.append(txt(236, ty2 + 8, "blocks 9–10 join the fine-tune for the first time — added to ImageNet, not overwritten", 9.5, ACC, 400, "start"))

body.append(txt(968, 540, "head, pooling, offset head, confidence gate, cosine margin, paired batches unchanged", 9, FAINT, 400, "end", style="font-style='italic'"))

HEIGHT = 636
svg = (f"<svg viewBox='0 0 980 {HEIGHT}' xmlns='http://www.w3.org/2000/svg' role='img'>"
       + "".join(body) + "</svg>")

out_path = Path(__file__).parent / "figure.json"
out_path.write_text(json.dumps({"architecture_svg": svg}))
print("wrote", out_path, len(svg), "bytes")
