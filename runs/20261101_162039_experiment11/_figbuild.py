"""Scratch figure builder for experiment 11 (cosine-prototype head).
Lives inside this run's own directory per the figure prompt's rule.
Helper geometry ported by hand from archive/arch_svg_reference.py (read,
not executed) so no IC-global coupling and two lanes can be drawn."""
import json
import math
from pathlib import Path

INK, MUT, FAINT, ACC, OCH = "#111111", "#6b6a60", "#9b998c", "#8c2f1f", "#8a6a1e"
FONT = "Palatino,Georgia,serif"
LOSS_Y = 470


def _n(v):
    v = round(v, 1)
    return f"{int(v)}" if v == int(v) else f"{v:.1f}"


def txt(x, y, s, size=10, color=MUT, w=400, anchor="middle", style="", tid=None):
    tid_attr = f"id='{tid}' " if tid else ""
    return (f"<text {tid_attr}x='{_n(x)}' y='{_n(y)}' font-family='{FONT}' font-size='{size}' "
            f"fill='{color}' font-weight='{w}' text-anchor='{anchor}' {style}>{s}</text>")


def cap(xc, y, name, sub=None, color=MUT, name_color=None):
    out = [txt(xc, y, name, 10.5, name_color or (INK if color == MUT else color), 600)]
    if sub:
        out.append(txt(xc, y + 12, sub, 9.5, color))
    return "".join(out)


def harrow(x1, x2, y, label=None, color=FAINT, lcolor=None):
    out = [f"<line x1='{x1}' y1='{y}' x2='{x2 - 5}' y2='{y}' stroke='{color}' stroke-width='1'/>",
           f"<path d='M {x2},{y} l -6,-3 v 6 Z' fill='{color}'/>"]
    if label:
        out.append(txt((x1 + x2) / 2, y - 6, label, 9, lcolor or FAINT, style="font-style='italic'"))
    return "".join(out)


def leader(x1, y1, x2, y2, color):
    return (f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' "
            f"stroke-width='1' stroke-dasharray='2 4'/>")


def slab(x, ic, s, d, color=INK):
    y = ic - s / 2
    dx, dy = d, d * 0.55
    out = [
        f"<polygon points='{x},{y} {x+dx},{y-dy} {x+dx+s},{y-dy} {x+s},{y}' "
        f"fill='#00000010' stroke='{color}' stroke-width='1'/>",
        f"<polygon points='{x+s},{y} {x+dx+s},{y-dy} {x+dx+s},{y-dy+s} {x+s},{y+s}' "
        f"fill='#0000001c' stroke='{color}' stroke-width='1'/>",
        f"<rect x='{x}' y='{y}' width='{s}' height='{s}' fill='#00000006' "
        f"stroke='{color}' stroke-width='1.2'/>",
    ]
    return "".join(out), s + dx


def bar(x, ic, h, color=INK, ticks=9):
    y = ic - h / 2
    out = [f"<rect x='{x}' y='{y:.1f}' width='7' height='{h}' fill='none' "
           f"stroke='{color}' stroke-width='1.2'/>"]
    for i in range(1, ticks):
        out.append(f"<line x1='{x}' y1='{y + i * h / ticks:.1f}' x2='{x + 7}' "
                   f"y2='{y + i * h / ticks:.1f}' stroke='{color}' stroke-width='0.5' opacity='0.5'/>")
    return "".join(out), 7


def gridsq(x, yc, s, n, color=INK, bumps=(), lw=1.2):
    y = yc - s / 2
    out = [f"<rect x='{x}' y='{y:.1f}' width='{s}' height='{s}' fill='none' "
           f"stroke='{color}' stroke-width='{lw}'/>"]
    for i in range(1, n):
        out.append(f"<line x1='{x + i * s / n:.1f}' y1='{y:.1f}' x2='{x + i * s / n:.1f}' "
                   f"y2='{y + s:.1f}' stroke='{color}' stroke-width='0.45' opacity='0.5'/>")
        out.append(f"<line x1='{x}' y1='{y + i * s / n:.1f}' x2='{x + s}' "
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


def kernel_proj(x1, ic1, s1, x2, ic2, s2, color):
    k = max(5, s1 * 0.30)
    kx, ky = x1 + s1 * 0.60, ic1 - s1 / 2 + s1 * 0.18
    c = max(3, s2 * 0.14)
    cx, cy = x2 + s2 * 0.22, ic2 - s2 / 2 + s2 * 0.26
    out = [f"<rect x='{kx:.1f}' y='{ky:.1f}' width='{k:.1f}' height='{k:.1f}' "
           f"fill='none' stroke='{color}' stroke-width='0.9'/>",
           f"<rect x='{cx:.1f}' y='{cy:.1f}' width='{c:.1f}' height='{c:.1f}' "
           f"fill='{color}' fill-opacity='0.5' stroke='none'/>"]
    for (px, py) in ((kx, ky), (kx + k, ky), (kx, ky + k), (kx + k, ky + k)):
        out.append(f"<line x1='{px:.1f}' y1='{py:.1f}' x2='{cx + c / 2:.1f}' "
                   f"y2='{cy + c / 2:.1f}' stroke='{color}' stroke-width='0.55' "
                   f"opacity='0.45'/>")
    return "".join(out)


def conv_group(x0, ic, prev=None):
    sizes = [(26, 7), (19, 6), (14, 5), (10, 4)]
    out, x, faces = [], x0, []
    for (s, d) in sizes:
        g, w = slab(x, ic, s, d, INK)
        out.append(g)
        faces.append((x, s))
        x += w + 13
    x -= 13
    if prev:
        out.append(kernel_proj(prev[0], ic, prev[1], faces[0][0], ic, faces[0][1], prev[2]))
    for (x1, s1), (x2, s2) in zip(faces, faces[1:]):
        out.append(kernel_proj(x1, ic, s1, x2, ic, s2, MUT))
    out.append(txt(x0 + 23, ic - 46, "64²×16", 9, FAINT))
    out.append(txt(x - 20, ic - 42, "8²×112", 9, FAINT))
    return "".join(out), x


def cam_terrain(y_top):
    Y = y_top
    return (f"<g id='cam-terrain'><rect id='frozen-input' x='26' y='{Y}' width='76' "
            f"height='76' fill='#f6f4ea' stroke='#9b998c' stroke-width='1.6'/>"
            f"<path d='M31 {Y+47} L97 {Y+27}' stroke='#e6e3d4' stroke-width='5' fill='none'/>"
            f"<path d='M59 {Y+6} L49 {Y+70}' stroke='#e6e3d4' stroke-width='3.5' fill='none'/>"
            f"<rect x='34' y='{Y+9}' width='12' height='8' fill='#d9d5c3' transform='rotate(-8 40 {Y+13})'/>"
            f"<rect x='78' y='{Y+8}' width='10' height='11' fill='#cfccbd'/>"
            f"<rect x='35' y='{Y+57}' width='13' height='8' fill='#d9d5c3'/>"
            f"<rect x='75' y='{Y+50}' width='10' height='9' fill='#cfccbd' transform='rotate(6 80 {Y+54})'/>"
            f"<rect x='54' y='{Y+31}' width='9' height='8' fill='#d9d5c3' opacity='.85'/>"
            f"<ellipse cx='86' cy='{Y+63}' rx='8' ry='6' fill='#8a6a1e' opacity='.12'/>"
            f"<circle cx='40' cy='{Y+28}' r='.7' fill='#6b6a60' opacity='.5'/>"
            f"<circle cx='70' cy='{Y+14}' r='.7' fill='#6b6a60' opacity='.5'/>"
            f"<circle cx='92' cy='{Y+36}' r='.7' fill='#6b6a60' opacity='.5'/>"
            f"<circle cx='52' cy='{Y+52}' r='.7' fill='#6b6a60' opacity='.5'/>"
            f"<circle cx='82' cy='{Y+70}' r='.7' fill='#6b6a60' opacity='.5'/>"
            f"<circle cx='31' cy='{Y+42}' r='.7' fill='#6b6a60' opacity='.5'/></g>")


def output_part(x, ic, sub="position fix + confidence"):
    return (txt(x, ic - 18, "frozen contract", 8.5, FAINT, anchor="start",
                style="font-style='italic'")
            + txt(x, ic - 2, "(lat, lon, confidence)", 13, MUT, 600, "start",
                  tid="frozen-output")
            + txt(x, ic + 12, sub, 9, FAINT, anchor="start"))


def gauge(x, yc, color=MUT, r=14):
    a = math.radians(55)
    return (f"<path d='M {x - r},{yc} A {r} {r} 0 0 1 {x + r},{yc}' fill='none' "
            f"stroke='{color}' stroke-width='1.2'/>"
            f"<line x1='{x}' y1='{yc}' x2='{x + (r - 3) * math.cos(a):.1f}' "
            f"y2='{yc - (r - 3) * math.sin(a):.1f}' stroke='{color}' stroke-width='1.4'/>"
            f"<circle cx='{x}' cy='{yc}' r='1.6' fill='{color}'/>")


def compass(cx, cy, r, color, hi_angle=35):
    """Unit-circle glyph: normalised embedding as a radius to the rim,
    a few prototype directions as short rim ticks, the matching one in
    solid color — 'compass, not lookup table'."""
    out = [f"<circle cx='{cx}' cy='{cy}' r='{r}' fill='none' stroke='{color}' stroke-width='1.1'/>"]
    for ang in (-70, -20, 60, 150, 210):
        a = math.radians(ang)
        x1, y1 = cx + (r - 3) * math.cos(a), cy + (r - 3) * math.sin(a)
        x2, y2 = cx + (r + 2) * math.cos(a), cy + (r + 2) * math.sin(a)
        out.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
                   f"stroke='{color}' stroke-width='0.7' opacity='0.5'/>")
    a = math.radians(hi_angle)
    out.append(f"<line x1='{cx}' y1='{cy}' x2='{cx + r * math.cos(a):.1f}' "
               f"y2='{cy + r * math.sin(a):.1f}' stroke='{color}' stroke-width='1.4'/>")
    out.append(f"<circle cx='{cx}' cy='{cy}' r='1.4' fill='{color}'/>")
    return "".join(out)


def lane(ic, letter, frame_right, split_y_frame, member_colors):
    """One member's path from the trunk-input split to the belief-field
    grid; returns (svg, grid_x, grid_s)."""
    b = []
    sign = -1 if ic < split_y_frame else 1
    ex, ey = 128.3, ic + sign * 0  # split endpoint handled by caller
    x = 130
    trunk, x_end = conv_group(x, ic, prev=(26, 76, FAINT))
    b.append(trunk)
    b.append(cap((x + x_end) / 2, ic + 58, f"member {letter} trunk", "conv 3×3 ×4, 112ch top"))
    x = x_end
    b.append(harrow(x + 8, x + 34, ic))
    x += 34
    g, _ = bar(x, ic, 64, INK, ticks=11)
    b.append(g)
    b.append(cap(x + 4, ic - 46, "112-d", "GAP"))
    bar_x = x
    x += 14
    # --- NEW cosine-prototype head (red: this experiment's change) ---
    b.append(harrow(x, x + 13, ic))
    x += 15
    bg, _ = bar(x, ic, 20, ACC, ticks=4)
    b.append(bg)
    bn_dy = 24 if letter == "A" else -24
    b.append(txt(x + 3, ic + bn_dy, "BN", 9, ACC, 600))
    b.append(txt(x + 3, ic + bn_dy + 11, "BatchNorm1d", 8, ACC))
    x += 13
    b.append(harrow(x, x + 12, ic))
    x += 14
    ccx = x + 12
    b.append(compass(ccx, ic, 12, ACC))
    cap_dy = -26 if letter == "A" else 26
    b.append(txt(ccx, ic + cap_dy, "unit embed. → cos", 9, ACC, 600))
    b.append(txt(ccx, ic + cap_dy + 11, "vs 2970 prototypes", 8.5, ACC))
    x = ccx + 12
    b.append(harrow(x, x + 36, ic, "× s=30", lcolor=ACC))
    x += 38
    grid_x = x
    gg, gs = gridsq(grid_x, ic, 58, 8, ACC, BUMP)
    b.append(gg)
    field_dy = -48 if letter == "A" else 70
    field_sub_dy = field_dy + 12
    b.append(txt(grid_x + 29, ic + field_dy, f"belief field {letter}", 10.5, ACC, 600))
    b.append(txt(grid_x + 29, ic + field_sub_dy, "2970 cells, scaled cosine", 9.5, ACC))
    x = grid_x + gs
    return "".join(b), grid_x, gs, bar_x, x


def build():
    H = 580
    IC_A, IC_B = 82, 258
    IC_F = (IC_A + IC_B) / 2
    b = [
        txt(8, 26, "INFERENCE PATH — WHAT FLIES", 9, FAINT, 600, "start", "letter-spacing='1.8'"),
        txt(8, LOSS_Y - 34, "TRAINING SIGNALS — NEVER FLY", 9, OCH, 600, "start", "letter-spacing='1.8'"),
        cam_terrain(IC_F - 38),
        txt(53, IC_F - 44, "128²×3", 9, FAINT),
        txt(53, IC_F + 48, "camera frame", 10.5, MUT, 600),
        txt(53, IC_F + 60, "one of three TRAIN surveys", 9.5, FAINT),
        txt(53, IC_F + 73, "frozen contract", 8.5, FAINT, style="font-style='italic'"),
    ]
    # splits
    b.append(f"<line x1='108' y1='{IC_F}' x2='128.3' y2='{IC_A+6.8}' stroke='{FAINT}' stroke-width='1'/>")
    b.append(f"<path d='M 130,{IC_A} L 131.2,{IC_A+7.5} L 125.4,{IC_A+6.1} Z' fill='{FAINT}'/>")
    b.append(f"<line x1='108' y1='{IC_F}' x2='128.3' y2='{IC_B-6.8}' stroke='{FAINT}' stroke-width='1'/>")
    b.append(f"<path d='M 130,{IC_B} L 125.4,{IC_B-6.1} L 131.2,{IC_B-7.5} Z' fill='{FAINT}'/>")
    b.append(txt(119, 48, "splits — no shared weights", 8.5, FAINT, style="font-style='italic'"))

    lane_a, gx_a, gs_a, barx_a, end_a = lane(IC_A, "A", 102, IC_F, None)
    lane_b, gx_b, gs_b, barx_b, end_b = lane(IC_B, "B", 102, IC_F, None)
    b.append(lane_a)
    b.append(lane_b)

    # average + pooling + gate + output (unchanged: ink), anchored same x
    # as prior experiments in this era for continuity
    avg_x = max(end_a, end_b) + 56
    b.append(f"<line x1='{end_a}' y1='{IC_A}' x2='{avg_x-6}' y2='{IC_A+74}' stroke='{INK}' stroke-width='1'/>")
    b.append(f"<path d='M {avg_x},{IC_A+80} L {avg_x-6.4},{IC_A+75.9} L {avg_x-1.4},{IC_A+72.5} Z' fill='{INK}'/>")
    b.append(f"<line x1='{end_b}' y1='{IC_B}' x2='{avg_x-6}' y2='{IC_B-74}' stroke='{INK}' stroke-width='1'/>")
    b.append(f"<path d='M {avg_x},{IC_B-80} L {avg_x-1.4},{IC_B-72.5} L {avg_x-6.4},{IC_B-75.9} Z' fill='{INK}'/>")
    avg_ic = (IC_A + IC_B) / 2
    b.append(txt((max(end_a, end_b) + avg_x) / 2, avg_ic - 70, "average", 9, INK, 600, style="font-style='italic'"))
    g, gs = gridsq(avg_x, avg_ic, 58, 8, INK, ((4, 2, .5), (3, 2, .22), (4, 1, .22), (4, 3, .14)))
    b.append(g)
    b.append(cap(avg_x + 3, avg_ic - 41, "averaged belief  p̄", "mean of both members' softmax", name_color=INK, color=INK))
    b[-1] = (txt(avg_x + 3, avg_ic - 41, "averaged belief  p̄", 10.5, INK, 600, "start")
             + txt(avg_x + 3, avg_ic - 29, "mean of both members' softmax", 9.5, INK, 400, "start"))
    win_x, win_y = avg_x + 29 - 10.9, avg_ic - 10.9
    b.append(f"<rect x='{win_x:.1f}' y='{win_y:.1f}' width='21.8' height='21.8' fill='none' stroke='{INK}' stroke-width='1.8'/>")
    tx, ty = 812, avg_ic
    for (fx, fy) in ((win_x + 3.6, win_y + 3.6), (win_x + 10.9, win_y + 3.6), (win_x + 18.1, win_y + 3.6),
                     (win_x + 3.6, win_y + 10.9), (win_x + 10.9, win_y + 10.9), (win_x + 18.1, win_y + 10.9),
                     (win_x + 3.6, win_y + 18.1), (win_x + 10.9, win_y + 18.1), (win_x + 18.1, win_y + 18.1)):
        b.append(f"<line x1='{fx:.1f}' y1='{fy:.1f}' x2='{tx}' y2='{ty}' stroke='{INK}' stroke-width='0.5' opacity='0.4'/>")
    b.append(crosspt(tx, ty, INK))
    b.append(txt((avg_x + gs + tx) / 2 + 4, avg_ic - 54 - 24, "pooled centroid of the 3×3 block", 10.5, INK, 600))
    b.append(txt((avg_x + gs + tx) / 2 + 4, avg_ic - 42 - 24, "+ mean of both members' offset nudges", 9.5, INK, 400))
    b.append(output_part(828, avg_ic))

    # confidence gate glyph below the averaged-belief grid
    gate_cx, gate_cy = avg_x + 10, avg_ic + 95
    b.append(f"<line x1='{win_x+10.9:.1f}' y1='{ty}' x2='{gate_cx-14:.1f}' y2='{gate_cy-13}' stroke='{INK}' stroke-width='1' stroke-dasharray='2 4'/>")
    b.append(f"<line x1='{win_x+10.9:.1f}' y1='{ty}' x2='{gate_cx+14:.1f}' y2='{gate_cy-13}' stroke='{INK}' stroke-width='1' stroke-dasharray='2 4'/>")
    b.append(gauge(gate_cx, gate_cy, INK))
    b.append(txt(gate_cx, gate_cy + 20, "confidence = min(mA, mB)", 9.5, INK, 600))
    b.append(txt(gate_cx, gate_cy + 32, "EACH member's own mass on the winning block", 8.3, INK))
    b.append(txt(gate_cx, gate_cy + 43, "both must independently back it, or the drone abstains", 8.3, INK))
    ex2 = tx + 46
    b.append(f"<path d='M {gate_cx},{gate_cy+50} V {gate_cy+70} H {ex2} V {avg_ic+26}' fill='none' stroke='{INK}' stroke-width='0.8'/>")
    b.append(f"<path d='M {ex2},{avg_ic+20} l -3,6 h 6 Z' fill='{INK}'/>")
    b.append(txt((gate_cx + ex2) / 2, gate_cy + 64, "may abstain instead of guessing", 8.5, INK, style="font-style='italic'"))

    # training-only margin note (red: train_only + changed), leader from
    # each member's cosine-compare glyph down into the training lane
    comp_a_x = gx_a + gs_a / 2
    comp_b_x = gx_b + gs_b / 2
    start_a_y = IC_A + gs_a / 2 + 3
    start_b_y = IC_B - gs_b / 2 - 3
    b.append(f"<circle cx='{comp_a_x:.1f}' cy='{start_a_y:.1f}' r='1.6' fill='{ACC}'/>")
    b.append(f"<path d='M {comp_a_x:.1f},{start_a_y:.1f} V {IC_A+76} H 112 V {LOSS_Y-26} H 206' fill='none' stroke='{ACC}' stroke-width='1' stroke-dasharray='2 4'/>")
    b.append(f"<circle cx='{comp_b_x:.1f}' cy='{start_b_y:.1f}' r='1.6' fill='{ACC}'/>")
    b.append(f"<path d='M {comp_b_x:.1f},{start_b_y:.1f} V {IC_B-76} H 100 V {LOSS_Y-26} H 206' fill='none' stroke='{ACC}' stroke-width='1' stroke-dasharray='2 4'/>")

    mcx, mcy = 206, LOSS_Y - 26
    b.append(f"<path d='M {mcx-13},{mcy} A 13 13 0 0 1 {mcx+13},{mcy}' fill='none' stroke='{ACC}' stroke-width='1.2'/>")
    a1 = math.radians(235)
    b.append(f"<line x1='{mcx}' y1='{mcy}' x2='{mcx+13*math.cos(a1):.1f}' y2='{mcy+13*math.sin(a1):.1f}' stroke='{ACC}' stroke-width='1.3'/>")
    a2 = math.radians(290)
    b.append(f"<line x1='{mcx}' y1='{mcy}' x2='{mcx+13*math.cos(a2):.1f}' y2='{mcy+13*math.sin(a2):.1f}' stroke='{ACC}' stroke-width='1.3' stroke-dasharray='1.5 1.5'/>")
    b.append(f"<circle cx='{mcx}' cy='{mcy}' r='1.4' fill='{ACC}'/>")
    b.append(txt(236, LOSS_Y - 44, "additive cosine margin m=0.25 on the target cell (CosFace-style)", 10, ACC, 600, "start"))
    b.append(txt(236, LOSS_Y - 32, "subtracted from cos before softmax, shared across tent neighbours by target weight", 9.5, ACC, 400, "start"))
    b.append(txt(236, LOSS_Y - 20, "train-only: removed at inference — norms are fixed, so only angle can lower the loss", 9.5, ACC, 400, "start"))
    b.append(txt(968, LOSS_Y - 56, "trunk, pooling, offset head, confidence gate, paired batches unchanged", 9, FAINT, 400, "end", "font-style='italic'"))

    svg = f"<svg viewBox='0 0 980 {H}' xmlns='http://www.w3.org/2000/svg' role='img'>{''.join(b)}</svg>"
    return svg


if __name__ == "__main__":
    svg = build()
    out = Path(__file__).parent / "figure.json"
    out.write_text(json.dumps({"architecture_svg": svg}))
    print(f"wrote {len(svg)} bytes to {out}")
