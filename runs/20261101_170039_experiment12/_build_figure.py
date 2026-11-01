"""Scratch builder for experiment 12's architecture figure. Adapted from the
visual language in archive/arch_svg_reference.py (read, not executed) with
helpers parameterized by an explicit lane-center `ic` so two member lanes can
be drawn at different y. Output: figure.json next to this script."""
import json
import math
from pathlib import Path

INK, MUT, FAINT, RED, OCH = "#111111", "#6b6a60", "#9b998c", "#8c2f1f", "#8a6a1e"
FONT = "Palatino,Georgia,serif"
HERE = Path(__file__).parent


def txt(x, y, s, size=10, color=MUT, w=400, anchor="middle", style="", tid=None):
    tid_attr = f"id='{tid}' " if tid else ""
    return (f"<text {tid_attr}x='{x:g}' y='{y:g}' font-family='{FONT}' font-size='{size}' "
            f"fill='{color}' font-weight='{w}' text-anchor='{anchor}' {style}>{s}</text>")


def cap(xc, y, name, sub=None, color=MUT, name_color=None):
    out = [txt(xc, y, name, 10.5, name_color or (INK if color == MUT else color), 600)]
    if sub:
        out.append(txt(xc, y + 12, sub, 9.5, color))
    return "".join(out)


def harrow(x1, x2, y, label=None, color=FAINT):
    out = [f"<line x1='{x1:.1f}' y1='{y:.1f}' x2='{x2 - 5:.1f}' y2='{y:.1f}' stroke='{color}' stroke-width='1'/>",
           f"<path d='M {x2:.1f},{y:.1f} l -6,-3 v 6 Z' fill='{color}'/>"]
    if label:
        out.append(txt((x1 + x2) / 2, y - 6, label, 9, FAINT, style="font-style='italic'"))
    return "".join(out)


def leader(x1, y1, x2, y2, color):
    return (f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}' "
            f"stroke-width='1' stroke-dasharray='2 4'/>")


def slab(x, s, d, ic, color=INK):
    y = ic - s / 2
    dx, dy = d, d * 0.55
    out = [
        f"<polygon points='{x:.1f},{y:.1f} {x+dx:.1f},{y-dy:.1f} {x+dx+s:.1f},{y-dy:.1f} {x+s:.1f},{y:.1f}' "
        f"fill='#00000010' stroke='{color}' stroke-width='1'/>",
        f"<polygon points='{x+s:.1f},{y:.1f} {x+dx+s:.1f},{y-dy:.1f} {x+dx+s:.1f},{y-dy+s:.1f} {x+s:.1f},{y+s:.1f}' "
        f"fill='#0000001c' stroke='{color}' stroke-width='1'/>",
        f"<rect x='{x:.1f}' y='{y:.1f}' width='{s:.1f}' height='{s:.1f}' fill='#00000006' "
        f"stroke='{color}' stroke-width='1.2'/>",
    ]
    return "".join(out), s + dx


def bar(x, h, ic, color=INK, ticks=9):
    y = ic - h / 2
    out = [f"<rect x='{x:.1f}' y='{y:.1f}' width='7' height='{h:.1f}' fill='none' "
           f"stroke='{color}' stroke-width='1.2'/>"]
    for i in range(1, ticks):
        out.append(f"<line x1='{x:.1f}' y1='{y + i * h / ticks:.1f}' x2='{x+7:.1f}' "
                   f"y2='{y + i * h / ticks:.1f}' stroke='{color}' stroke-width='0.5' opacity='0.5'/>")
    return "".join(out), 7


def gridsq(x, s, n, ic, color=INK, bumps=(), lw=1.2):
    y = ic - s / 2
    out = [f"<rect x='{x:.1f}' y='{y:.1f}' width='{s:.1f}' height='{s:.1f}' fill='none' "
           f"stroke='{color}' stroke-width='{lw}'/>"]
    for i in range(1, n):
        out.append(f"<line x1='{x + i*s/n:.1f}' y1='{y:.1f}' x2='{x + i*s/n:.1f}' "
                   f"y2='{y+s:.1f}' stroke='{color}' stroke-width='0.45' opacity='0.5'/>")
        out.append(f"<line x1='{x:.1f}' y1='{y + i*s/n:.1f}' x2='{x+s:.1f}' "
                   f"y2='{y + i*s/n:.1f}' stroke='{color}' stroke-width='0.45' opacity='0.5'/>")
    for (gx, gy, o) in bumps:
        out.append(f"<rect x='{x + gx*s/n:.1f}' y='{y + gy*s/n:.1f}' "
                   f"width='{s/n:.1f}' height='{s/n:.1f}' fill='{color}' fill-opacity='{o}'/>")
    return "".join(out), s


BUMP = ((5, 2, .55), (4, 2, .28), (5, 1, .28), (6, 2, .18), (5, 3, .18), (2, 5, .12), (1, 6, .1))


def crosspt(x, y, color, r=7):
    return (f"<line x1='{x-r:.1f}' y1='{y:.1f}' x2='{x+r:.1f}' y2='{y:.1f}' stroke='{color}' stroke-width='1.4'/>"
            f"<line x1='{x:.1f}' y1='{y-r:.1f}' x2='{x:.1f}' y2='{y+r:.1f}' stroke='{color}' stroke-width='1.4'/>"
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r*0.55:.1f}' fill='none' stroke='{color}' stroke-width='1.4'/>"
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='1.5' fill='{color}'/>")


def kernel_proj(x1, s1, x2, s2, ic, color):
    k = max(5, s1 * 0.30)
    kx, ky = x1 + s1 * 0.60, ic - s1 / 2 + s1 * 0.18
    c = max(3, s2 * 0.14)
    cx, cy = x2 + s2 * 0.22, ic - s2 / 2 + s2 * 0.26
    out = [f"<rect x='{kx:.1f}' y='{ky:.1f}' width='{k:.1f}' height='{k:.1f}' "
           f"fill='none' stroke='{color}' stroke-width='0.9'/>",
           f"<rect x='{cx:.1f}' y='{cy:.1f}' width='{c:.1f}' height='{c:.1f}' "
           f"fill='{color}' fill-opacity='0.5' stroke='none'/>"]
    for (px, py) in ((kx, ky), (kx + k, ky), (kx, ky + k), (kx + k, ky + k)):
        out.append(f"<line x1='{px:.1f}' y1='{py:.1f}' x2='{cx + c/2:.1f}' "
                   f"y2='{cy + c/2:.1f}' stroke='{color}' stroke-width='0.55' opacity='0.45'/>")
    return "".join(out)


def gauge(x, yc, color=MUT, r=14):
    a = math.radians(55)
    return (f"<path d='M {x-r:.1f},{yc:.1f} A {r} {r} 0 0 1 {x+r:.1f},{yc:.1f}' fill='none' "
            f"stroke='{color}' stroke-width='1.2'/>"
            f"<line x1='{x:.1f}' y1='{yc:.1f}' x2='{x + (r-3)*math.cos(a):.1f}' "
            f"y2='{yc - (r-3)*math.sin(a):.1f}' stroke='{color}' stroke-width='1.4'/>"
            f"<circle cx='{x:.1f}' cy='{yc:.1f}' r='1.6' fill='{color}'/>")


CAM_TERRAIN = """<g id='cam-terrain'><rect id='frozen-input' x='26' y='{y:.1f}' width='76'
   height='76' fill='#f6f4ea' stroke='#9b998c' stroke-width='1.6'/>
<path d='M31 {y47:.1f} L97 {y27:.1f}' stroke='#e6e3d4' stroke-width='5' fill='none'/>
<path d='M59 {y6:.1f} L49 {y70:.1f}' stroke='#e6e3d4' stroke-width='3.5' fill='none'/>
<rect x='34' y='{y9:.1f}' width='12' height='8' fill='#d9d5c3' transform='rotate(-8 40 {y13:.1f})'/>
<rect x='78' y='{y8:.1f}' width='10' height='11' fill='#cfccbd'/>
<rect x='35' y='{y57:.1f}' width='13' height='8' fill='#d9d5c3'/>
<rect x='75' y='{y50:.1f}' width='10' height='9' fill='#cfccbd' transform='rotate(6 80 {y54:.1f})'/>
<rect x='54' y='{y31:.1f}' width='9' height='8' fill='#d9d5c3' opacity='.85'/>
<ellipse cx='86' cy='{y63:.1f}' rx='8' ry='6' fill='#8a6a1e' opacity='.12'/>
<circle cx='40' cy='{y28:.1f}' r='.7' fill='#6b6a60' opacity='.5'/>
<circle cx='70' cy='{y14:.1f}' r='.7' fill='#6b6a60' opacity='.5'/>
<circle cx='92' cy='{y36:.1f}' r='.7' fill='#6b6a60' opacity='.5'/>
<circle cx='52' cy='{y52:.1f}' r='.7' fill='#6b6a60' opacity='.5'/>
<circle cx='82' cy='{y70b:.1f}' r='.7' fill='#6b6a60' opacity='.5'/>
<circle cx='31' cy='{y42:.1f}' r='.7' fill='#6b6a60' opacity='.5'/></g>"""


def cam_terrain(y):
    return CAM_TERRAIN.format(y=y, y47=y+47, y27=y+27, y6=y+6, y70=y+70, y9=y+9, y13=y+13,
                               y8=y+8, y57=y+57, y50=y+50, y54=y+54, y31=y+31, y63=y+63,
                               y28=y+28, y14=y+14, y36=y+36, y52=y+52, y70b=y+70, y42=y+42)


def output_part(x, ic, sub="position fix + confidence"):
    return (txt(x, ic - 18, "frozen contract", 8.5, FAINT, anchor="start", style="font-style='italic'")
            + txt(x, ic - 2, "(lat, lon, confidence)", 13, MUT, 600, "start", tid="frozen-output")
            + txt(x, ic + 12, sub, 9, FAINT, anchor="start"))


def main():
    H = 580
    FRAME_C = 170.0
    IC_A, IC_B = 82.0, 258.0
    b = []
    b.append(txt(8, 26, "INFERENCE PATH — WHAT FLIES", 9, FAINT, 600, "start", "letter-spacing='1.8'"))
    b.append(txt(8, 436, "TRAINING SIGNALS — NEVER FLY", 9, OCH, 600, "start", "letter-spacing='1.8'"))

    # camera frame (shared, frozen)
    fy = FRAME_C - 38
    b.append(cam_terrain(fy))
    b.append(txt(53, fy - 6, "128²×3", 9, FAINT))
    b.append(cap(53, fy + 88, "camera frame", "one of three TRAIN surveys"))
    b.append(txt(53, fy + 111, "frozen contract", 8.5, FAINT, style="font-style='italic'"))

    # split to two members
    fx = 26 + 76
    b.append(f"<line x1='{fx}' y1='{FRAME_C}' x2='128.3' y2='{IC_A+6.8:.1f}' stroke='{FAINT}' stroke-width='1'/>")
    b.append(f"<path d='M 130,{IC_A:.1f} L 131.2,{IC_A+7.5:.1f} L 125.4,{IC_A+6.1:.1f} Z' fill='{FAINT}'/>")
    b.append(f"<line x1='{fx}' y1='{FRAME_C}' x2='128.3' y2='{IC_B-6.8:.1f}' stroke='{FAINT}' stroke-width='1'/>")
    b.append(f"<path d='M 130,{IC_B:.1f} L 125.4,{IC_B-6.1:.1f} L 131.2,{IC_B-7.5:.1f} Z' fill='{FAINT}'/>")
    b.append(txt(119, 40, "splits — no shared weights", 8.5, FAINT, style="font-style='italic'"))

    gate_mA = {}
    gate_mB = {}
    avg = {}

    def lane(ic, letter, store):
        x = 130
        # pretrained trunk: MobileNetV3-Small stem + 8 inverted-residual
        # blocks (features[0:9]) — CHANGED this experiment, drawn in red
        sizes = [(24, 6), (17, 9), (12, 11), (9, 13)]
        faces = []
        for i, (s, d) in enumerate(sizes):
            g, w = slab(x, s, d, ic, RED)
            store.setdefault("svg", []).append(g)
            faces.append((x, s))
            x += w + 11
        x -= 11
        store["svg"].append(kernel_proj(26, 76, faces[0][0], faces[0][1], ic, FAINT))
        for (x1, s1), (x2, s2) in zip(faces, faces[1:]):
            store["svg"].append(kernel_proj(x1, s1, x2, s2, ic, RED))
        xc_trunk = (130 + x) / 2
        store["svg"].append(cap(xc_trunk, ic + 58, f"member {letter} trunk — pretrained",
                                 "MobileNetV3-S features[0:9], BSD-3", name_color=RED))
        store["svg"].append(txt(faces[0][0] + faces[0][1] / 2, ic - faces[0][1] / 2 - 6, "64²×16", 8, FAINT))
        store["svg"].append(txt(faces[-1][0] + faces[-1][1] / 2, ic - faces[-1][1] / 2 - 6, "8²×48", 8, FAINT))

        # 1x1 lift 48 -> 112, trainable, red (new)
        x += 14
        store["svg"].append(harrow(x - 8, x + 6, ic))
        lx = x + 8
        g, w = slab(lx, 9, 15, ic, RED)
        store["svg"].append(g)
        store["svg"].append(kernel_proj(faces[-1][0], faces[-1][1], lx, 9, ic, RED))
        store["svg"].append(cap(lx + 10, ic + 115, "1×1 lift", "48→112ch, BN+ReLU, trainable", name_color=RED))
        x = lx + w

        # GAP -> 112-d vector (unchanged)
        x += 10
        store["svg"].append(harrow(x, x + 20, ic))
        x += 22
        g, _ = bar(x, 64, ic, INK, ticks=11)
        store["svg"].append(g)
        store["svg"].append(cap(x + 3.5, ic - 44, "112-d", "GAP"))
        bar_x = x
        x += 7

        # BatchNorm1d (unchanged)
        x += 10
        store["svg"].append(harrow(x, x + 12, ic))
        x += 14
        store["svg"].append(f"<rect x='{x:.1f}' y='{ic-10:.1f}' width='7' height='20' fill='none' "
                             f"stroke='{INK}' stroke-width='1.2'/>")
        for i in range(1, 3):
            yy = ic - 10 + i * 20 / 3
            store["svg"].append(f"<line x1='{x:.1f}' y1='{yy:.1f}' x2='{x+7:.1f}' y2='{yy:.1f}' "
                                 f"stroke='{INK}' stroke-width='0.5' opacity='0.5'/>")
        store["svg"].append(txt(x + 3.5, ic + 34, "BN", 9, INK, 600))
        store["svg"].append(txt(x + 3.5, ic + 45, "BatchNorm1d", 8, INK))
        x += 13

        # unit-embed -> cosine head (unchanged)
        x += 10
        store["svg"].append(harrow(x, x + 12, ic))
        x += 20
        cx = x
        store["svg"].append(f"<circle cx='{cx:.1f}' cy='{ic:.1f}' r='12' fill='none' stroke='{INK}' stroke-width='1.1'/>")
        for ang in (-70, -40, 20, 100, 150):
            a = math.radians(ang)
            store["svg"].append(f"<line x1='{cx+11*math.cos(a):.1f}' y1='{ic+11*math.sin(a):.1f}' "
                                 f"x2='{cx+16*math.cos(a):.1f}' y2='{ic+16*math.sin(a):.1f}' "
                                 f"stroke='{INK}' stroke-width='0.7' opacity='0.5'/>")
        store["svg"].append(f"<line x1='{cx:.1f}' y1='{ic:.1f}' x2='{cx+10:.1f}' y2='{ic+7:.1f}' stroke='{INK}' stroke-width='1.4'/>")
        store["svg"].append(f"<circle cx='{cx:.1f}' cy='{ic:.1f}' r='1.4' fill='{INK}'/>")
        store["svg"].append(cap(cx, ic - 26, "unit embed. → cos", "vs 2970 prototypes"))
        x = cx + 12

        # belief field (unchanged)
        x += 12
        store["svg"].append(harrow(x, x + 10, ic))
        x += 24
        g, gs = gridsq(x, 58, 8, ic, INK, BUMP)
        store["svg"].append(g)
        store["svg"].append(cap(x + gs / 2, ic - 50, f"belief field {letter}", "2970 cells, scaled cosine × s=30"))
        store["gx"], store["gs"], store["ic"], store["bar_x"] = x, gs, ic, bar_x
        return x + gs

    end_x = lane(IC_A, "A", gate_mA)
    lane(IC_B, "B", gate_mB)
    b.extend(gate_mA["svg"])
    b.extend(gate_mB["svg"])

    # ---- average, pooled centroid, confidence gate, output (all unchanged) ----
    gxA, gsA, icA = gate_mA["gx"], gate_mA["gs"], gate_mA["ic"]
    gxB, gsB, icB = gate_mB["gx"], gate_mB["gs"], gate_mB["ic"]
    avg_x = end_x + 56
    kink_x = 572.0  # right edge of the belief-field caption's text box — the
    # merge lines must not angle upward/downward past the caption until clear of it
    b.append(f"<path d='M {gxA+gsA:.1f},{icA:.1f} L {kink_x:.1f},{icA:.1f} L {avg_x:.1f},{(icA+icB)/2 - 16:.1f}' fill='none' stroke='{INK}' stroke-width='1'/>")
    b.append(f"<path d='M {avg_x+6:.1f},{(icA+icB)/2-22:.1f} L {avg_x-0.4:.1f},{(icA+icB)/2-26.5:.1f} L {avg_x+4.6:.1f},{(icA+icB)/2-29.9:.1f} Z' fill='{INK}'/>")
    b.append(f"<path d='M {gxB+gsB:.1f},{icB:.1f} L {kink_x:.1f},{icB:.1f} L {avg_x:.1f},{(icA+icB)/2 + 16:.1f}' fill='none' stroke='{INK}' stroke-width='1'/>")
    b.append(f"<path d='M {avg_x+6:.1f},{(icA+icB)/2+22:.1f} L {avg_x+4.6:.1f},{(icA+icB)/2+29.9:.1f} L {avg_x-0.4:.1f},{(icA+icB)/2+26.5:.1f} Z' fill='{INK}'/>")
    mid = (icA + icB) / 2
    b.append(txt(avg_x + 14, mid - 48, "average", 9, INK, 600, style="font-style='italic'"))
    agx = avg_x + 6
    g, gs2 = gridsq(agx, 58, 8, mid, INK, ((5, 2, .5), (4, 2, .22), (5, 1, .22), (5, 3, .14)))
    b.append(g)
    b.append(txt(agx, mid - 37, "averaged belief  p̄", 10.5, INK, 600, anchor="start"))
    b.append(txt(agx, mid - 25, "mean of both members' softmax", 9.5, INK, anchor="start"))
    wx, wy = agx + 3 * gs2 / 8, mid - gs2 / 2 + 2 * gs2 / 8
    b.append(f"<rect x='{wx:.1f}' y='{wy:.1f}' width='{3*gs2/8:.1f}' height='{3*gs2/8:.1f}' fill='none' stroke='{INK}' stroke-width='1.8'/>")
    tx = 812
    b.append(f"<line x1='{wx+1.5:.1f}' y1='{wy+1.5:.1f}' x2='{tx}' y2='{mid:.1f}' stroke='{INK}' stroke-width='0.5' opacity='0.4'/>")
    b.append(f"<line x1='{wx+3*gs2/16:.1f}' y1='{wy+1.5:.1f}' x2='{tx}' y2='{mid:.1f}' stroke='{INK}' stroke-width='0.5' opacity='0.4'/>")
    b.append(f"<line x1='{wx+1.5:.1f}' y1='{wy+3*gs2/16:.1f}' x2='{tx}' y2='{mid:.1f}' stroke='{INK}' stroke-width='0.5' opacity='0.4'/>")
    b.append(f"<line x1='{wx+3*gs2/8-1.5:.1f}' y1='{wy+3*gs2/8-1.5:.1f}' x2='{tx}' y2='{mid:.1f}' stroke='{INK}' stroke-width='0.5' opacity='0.4'/>")
    b.append(crosspt(tx, mid, INK))
    b.append(txt((agx + gs2 + tx) / 2 + 40, mid - 74, "pooled centroid of the 3×3 block", 10.5, INK, 600))
    b.append(txt((agx + gs2 + tx) / 2 + 40, mid - 62, "+ mean of both members' offset nudges", 9.5, INK))
    b.append(output_part(828, mid))

    # confidence gate (unchanged)
    gx0 = agx + gs2 / 2
    gy0 = mid + gs2 / 2 + 70
    b.append(f"<line x1='{agx+3*gs2/16:.1f}' y1='{mid+gs2/2-2:.1f}' x2='{gx0-14:.1f}' y2='{gy0:.1f}' stroke='{INK}' stroke-width='1' stroke-dasharray='2 4'/>")
    b.append(f"<line x1='{agx+gs2/8+3*gs2/16:.1f}' y1='{mid+gs2/2-2:.1f}' x2='{gx0+14:.1f}' y2='{gy0:.1f}' stroke='{INK}' stroke-width='1' stroke-dasharray='2 4'/>")
    b.append(f"<path d='M {gx0-14:.1f},{gy0:.1f} A 14 14 0 0 1 {gx0+14:.1f},{gy0:.1f}' fill='none' stroke='{INK}' stroke-width='1.2'/>")
    b.append(f"<line x1='{gx0:.1f}' y1='{gy0:.1f}' x2='{gx0+6.3:.1f}' y2='{gy0-9:.1f}' stroke='{INK}' stroke-width='1.4'/>")
    b.append(f"<circle cx='{gx0:.1f}' cy='{gy0:.1f}' r='1.6' fill='{INK}'/>")
    b.append(txt(gx0, gy0 + 20, "confidence = min(mᴀ, mʙ)", 9.5, INK, 600))
    b.append(txt(gx0, gy0 + 32, "EACH member's own mass on the winning block", 8.3, INK))
    b.append(txt(gx0, gy0 + 43, "both must independently back it, or the drone abstains", 8.3, INK))
    abstain_y = gy0 + 62
    b.append(f"<path d='M {gx0:.1f},{abstain_y:.1f} V {abstain_y+20:.1f} H {858} V {mid+26:.1f}' fill='none' stroke='{INK}' stroke-width='0.8'/>")
    b.append(f"<path d='M 858,{mid+20:.1f} l -3,6 h 6 Z' fill='{INK}'/>")
    b.append(txt((gx0 + 858) / 2, abstain_y + 14, "may abstain instead of guessing", 8.5, INK, style="font-style='italic'"))

    # ---- training-only lane ----
    # cosine margin loss: unchanged this experiment -> ochre, not red
    anchor_ax, anchor_ay = gxA + gsA / 2, icA + gsA / 2 + 32
    anchor_bx, anchor_by = gxB + gsB / 2, icB - gsB / 2 - 32
    lab_x, lab_y = 236, 426
    b.append(f"<circle cx='{anchor_ax:.1f}' cy='{anchor_ay:.1f}' r='1.6' fill='{OCH}'/>")
    b.append(f"<path d='M {anchor_ax:.1f},{anchor_ay:.1f} V {anchor_ay+20:.1f} H 112 V 444 H 206' fill='none' stroke='{OCH}' stroke-width='1' stroke-dasharray='2 4'/>")
    b.append(f"<circle cx='{anchor_bx:.1f}' cy='{anchor_by:.1f}' r='1.6' fill='{OCH}'/>")
    b.append(f"<path d='M {anchor_bx:.1f},{anchor_by:.1f} V {anchor_by-20:.1f} H 100 V 444 H 206' fill='none' stroke='{OCH}' stroke-width='1' stroke-dasharray='2 4'/>")
    b.append(f"<path d='M 193,444 A 13 13 0 0 1 219,444' fill='none' stroke='{OCH}' stroke-width='1.2'/>")
    b.append(f"<line x1='206' y1='444' x2='198.5' y2='433.4' stroke='{OCH}' stroke-width='1.3'/>")
    b.append(f"<line x1='206' y1='444' x2='210.4' y2='431.8' stroke='{OCH}' stroke-width='1.3' stroke-dasharray='1.5 1.5'/>")
    b.append(f"<circle cx='206' cy='444' r='1.4' fill='{OCH}'/>")
    b.append(txt(lab_x, lab_y, "additive cosine margin m=0.25 on the target cell (unchanged)", 10, OCH, 600, "start"))
    b.append(txt(lab_x, lab_y + 12, "CosFace-style, subtracted from cos before softmax — shared head, not this experiment", 9.5, OCH, 400, "start"))

    # trunk learning-rate schedule: CHANGED this experiment -> red
    lrx, lry = 236, lab_y + 32
    tr_anchor_x, tr_anchor_y = 130 + 10, icA + 58 + 24
    b.append(f"<circle cx='{tr_anchor_x:.1f}' cy='{tr_anchor_y:.1f}' r='1.6' fill='{RED}'/>")
    b.append(f"<path d='M {tr_anchor_x:.1f},{tr_anchor_y:.1f} V {lry-4:.1f} H {lrx-14:.1f}' fill='none' stroke='{RED}' stroke-width='1' stroke-dasharray='2 4'/>")
    b.append(txt(lrx, lry, "pretrained trunk fine-tuned at 0.1× the head's learning rate", 10, RED, 600, "start"))
    b.append(txt(lrx, lry + 12, "both decay on the SAME cosine schedule — Berlin is added to ImageNet, not overwritten", 9.5, RED, 400, "start"))

    b.append(txt(968, 414, "head, pooling, offset head, confidence gate, cosine margin, paired batches unchanged", 9, FAINT, anchor="end", style="font-style='italic'"))

    svg = f"<svg viewBox='0 0 980 {H}' xmlns='http://www.w3.org/2000/svg' role='img'>{''.join(b)}</svg>"
    out = HERE / "figure.json"
    out.write_text(json.dumps({"architecture_svg": svg}))
    print("wrote", len(svg), "bytes to", out)


if __name__ == "__main__":
    main()
