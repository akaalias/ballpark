import json
import math

INK, MUT, FAINT, ACC, OCH = "#111111", "#6b6a60", "#9b998c", "#8c2f1f", "#8a6a1e"
FONT = "Palatino,Georgia,serif"
H = 640
MID = H / 2  # 320


def txt(x, y, s, size=10, color=MUT, w=400, anchor="middle", style=""):
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-family='{FONT}' font-size='{size}' "
            f"fill='{color}' font-weight='{w}' text-anchor='{anchor}' {style}>{s}</text>")


def cap(xc, y, name, sub=None, color=MUT, name_color=None, anchor="middle"):
    out = [txt(xc, y, name, 10.5, name_color or INK, 600, anchor)]
    if sub:
        out.append(txt(xc, y + 12, sub, 9.5, color, 400, anchor))
    return "".join(out)


def harrow(x1, x2, y, color=FAINT):
    return (f"<line x1='{x1:.1f}' y1='{y:.1f}' x2='{x2-5:.1f}' y2='{y:.1f}' stroke='{color}' stroke-width='1'/>"
            f"<path d='M {x2:.1f},{y:.1f} l -6,-3 v 6 Z' fill='{color}'/>")


def lpath(d, color, dash="2 4", w=1):
    return f"<path d='{d}' fill='none' stroke='{color}' stroke-width='{w}' stroke-dasharray='{dash}'/>"


def slab(x, yc, s, color=INK):
    d = s * 0.92
    y = yc - s / 2
    dx, dy = d, d * 0.55
    out = [
        f"<polygon points='{x},{y:.1f} {x+dx:.1f},{y-dy:.1f} {x+dx+s:.1f},{y-dy:.1f} {x+s},{y:.1f}' "
        f"fill='#00000010' stroke='{color}' stroke-width='1'/>",
        f"<polygon points='{x+s},{y:.1f} {x+dx+s:.1f},{y-dy:.1f} {x+dx+s:.1f},{y-dy+s:.1f} {x+s},{y+s:.1f}' "
        f"fill='#0000001c' stroke='{color}' stroke-width='1'/>",
        f"<rect x='{x}' y='{y:.1f}' width='{s}' height='{s}' fill='#00000006' "
        f"stroke='{color}' stroke-width='1.2'/>",
    ]
    return "".join(out), s + dx


def kernel_proj(x1, yc, s1, x2, s2, color):
    k = max(5, s1 * 0.30)
    kx, ky = x1 + s1 * 0.60, yc - s1 / 2 + s1 * 0.18
    c = max(3, s2 * 0.14)
    cx, cy = x2 + s2 * 0.22, yc - s2 / 2 + s2 * 0.26
    out = [f"<rect x='{kx:.1f}' y='{ky:.1f}' width='{k:.1f}' height='{k:.1f}' "
           f"fill='none' stroke='{color}' stroke-width='0.9'/>",
           f"<rect x='{cx:.1f}' y='{cy:.1f}' width='{c:.1f}' height='{c:.1f}' "
           f"fill='{color}' fill-opacity='0.5' stroke='none'/>"]
    for (px, py) in ((kx, ky), (kx + k, ky), (kx, ky + k), (kx + k, ky + k)):
        out.append(f"<line x1='{px:.1f}' y1='{py:.1f}' x2='{cx+c/2:.1f}' "
                   f"y2='{cy+c/2:.1f}' stroke='{color}' stroke-width='0.55' opacity='0.45'/>")
    return "".join(out), (kx + k / 2, ky)


def bar(x, yc, h, color=INK, ticks=8):
    y = yc - h / 2
    out = [f"<rect x='{x}' y='{y:.1f}' width='7' height='{h}' fill='none' stroke='{color}' stroke-width='1.2'/>"]
    for i in range(1, ticks):
        yy = y + i * h / ticks
        out.append(f"<line x1='{x}' y1='{yy:.1f}' x2='{x+7}' y2='{yy:.1f}' stroke='{color}' stroke-width='0.5' opacity='0.5'/>")
    return "".join(out), 7


def gridsq(x, yc, s, n, color=INK, bumps=()):
    y = yc - s / 2
    out = [f"<rect x='{x}' y='{y:.1f}' width='{s}' height='{s}' fill='none' stroke='{color}' stroke-width='1.2'/>"]
    for i in range(1, n):
        out.append(f"<line x1='{x+i*s/n:.1f}' y1='{y:.1f}' x2='{x+i*s/n:.1f}' y2='{y+s:.1f}' stroke='{color}' stroke-width='0.45' opacity='0.5'/>")
        out.append(f"<line x1='{x}' y1='{y+i*s/n:.1f}' x2='{x+s}' y2='{y+i*s/n:.1f}' stroke='{color}' stroke-width='0.45' opacity='0.5'/>")
    for (gx, gy, o) in bumps:
        out.append(f"<rect x='{x+gx*s/n:.1f}' y='{y+gy*s/n:.1f}' width='{s/n:.1f}' height='{s/n:.1f}' fill='{color}' fill-opacity='{o}'/>")
    return "".join(out), s


BUMP = ((5, 2, .55), (4, 2, .28), (5, 1, .28), (6, 2, .18), (5, 3, .18), (2, 5, .12), (1, 6, .1))


def crosspt(x, y, color, r=7):
    return (f"<line x1='{x-r}' y1='{y}' x2='{x+r}' y2='{y}' stroke='{color}' stroke-width='1.4'/>"
            f"<line x1='{x}' y1='{y-r}' x2='{x}' y2='{y+r}' stroke='{color}' stroke-width='1.4'/>"
            f"<circle cx='{x}' cy='{y}' r='{r*0.55:.1f}' fill='none' stroke='{color}' stroke-width='1.4'/>"
            f"<circle cx='{x}' cy='{y}' r='1.5' fill='{color}'/>")


def cos_head(x, yc, color=INK):
    out = [f"<circle cx='{x}' cy='{yc}' r='12' fill='none' stroke='{color}' stroke-width='1.1'/>"]
    for ang in (-70, -40, 10, 65, 130):
        a = math.radians(ang)
        out.append(f"<line x1='{x+12*math.cos(a)*0.3:.1f}' y1='{yc+12*math.sin(a)*0.3:.1f}' "
                   f"x2='{x+12*math.cos(a)*0.95:.1f}' y2='{yc+12*math.sin(a)*0.95:.1f}' "
                   f"stroke='{color}' stroke-width='0.7' opacity='0.5'/>")
    out.append(f"<line x1='{x}' y1='{yc}' x2='{x+10}' y2='{yc+7}' stroke='{color}' stroke-width='1.4'/>")
    out.append(f"<circle cx='{x}' cy='{yc}' r='1.4' fill='{color}'/>")
    return "".join(out)


def cam_terrain(Y):
    return (f"<g id='cam-terrain'><rect id='frozen-input' x='26' y='{Y}' width='76' height='76' "
            f"fill='#f6f4ea' stroke='#9b998c' stroke-width='1.6'/>"
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


def member_lane(yc, label, row_dir):
    """One committee member, single row: trunk (4 slabs, 3 MixStyle
    junctions after blocks 1/3/6) -> 112-d embedding (GAP+concat of the
    mid 8²x48 and deep 4²x96 taps, unchanged) -> BN -> unit embed -> cosine
    tile head -> averaged-softmax belief field. row_dir=+1 captions sit
    below each row, -1 above (so lane B's captions don't crowd the frame)."""
    b = []
    sizes = [20, 15, 11, 8]
    gaps = [20, 17, 14]
    x = 164
    faces = []
    for i, s in enumerate(sizes):
        g, w = slab(x, yc, s, INK)
        b.append(g)
        faces.append((x, s))
        if i < 3:
            x += w + gaps[i]
    end_x = faces[-1][0] + faces[-1][1] + faces[-1][1] * 0.92
    mix_pts = []
    for i in range(3):
        (x1, s1), (x2, s2) = faces[i], faces[i + 1]
        g, pt = kernel_proj(x1, yc, s1, x2, s2, ACC)
        b.append(g)
        mix_pts.append(pt)
    trunk_cap_x = (faces[0][0] + end_x) / 2
    row_main = yc + row_dir * 36
    row_low = yc + row_dir * 54
    b.append(cap(trunk_cap_x, row_main, "trunk (pretrained)", "11 MobileNetV3-S blocks, ImageNet init"))

    tap_x = end_x + 6
    b.append(f"<circle cx='{tap_x:.1f}' cy='{yc}' r='2' fill='{INK}'/>")
    bx = tap_x + 24
    b.append(harrow(tap_x + 2, bx - 2, yc))
    g, _ = bar(bx, yc, 46, INK, ticks=7); b.append(g)

    bnx = bx + 30
    b.append(harrow(bx + 9, bnx - 2, yc))
    g, _ = bar(bnx, yc, 20, INK, ticks=2); b.append(g)

    cosx = bnx + 34
    b.append(harrow(bnx + 9, cosx - 14, yc))
    b.append(cos_head(cosx, yc, INK))
    b.append(cap(cosx, row_main, "cosine tile match", "112-d embed, 2970 protos, s=30"))

    gx = cosx + 24
    b.append(harrow(cosx + 14, gx - 2, yc))
    g, gs = gridsq(gx, yc, 56, 8, INK, BUMP); b.append(g)
    b.append(cap(gx + gs / 2, row_low, f"belief field {label}"))
    return "".join(b), gx, gs, mix_pts


def output_part(x, y_mid):
    base = y_mid + 13 * 0.35
    return (txt(x, base - 16, "frozen contract", 8.5, FAINT, 400, "start", style="font-style='italic'")
            + f"<text id='frozen-output' x='{x}' y='{base:.1f}' font-family='{FONT}' font-size='13' "
              f"fill='{MUT}' font-weight='600' text-anchor='start'>(lat, lon, confidence)</text>"
            + txt(x, base + 14, "position fix + confidence", 9, FAINT, 400, "start"))


def build():
    b = [txt(8, 20, "INFERENCE PATH — WHAT FLIES", 9, FAINT, 600, "start", "letter-spacing='1.8'"),
         txt(8, H - 10, "TRAINING SIGNALS — NEVER FLY", 9, OCH, 600, "start", "letter-spacing='1.8'")]

    frame_y = MID - 38
    b.append(cam_terrain(frame_y))
    b.append(txt(64, frame_y - 6, "128²×3", 9, FAINT))
    b.append(txt(64, frame_y + 90, "camera frame", 10.5, MUT, 600))
    b.append(txt(64, frame_y + 102, "one TRAIN survey", 9.5, FAINT))
    b.append(txt(64, frame_y + 114, "frozen contract", 8.5, FAINT, style="font-style='italic'"))

    ycA, ycB = 160, 480
    b.append(f"<line x1='102' y1='{MID-10:.0f}' x2='126' y2='{ycA+6}' stroke='{FAINT}' stroke-width='1'/>"
              f"<path d='M 128,{ycA} L 130.3,{ycA+10.5} L 122.4,{ycA+9.2} Z' fill='{FAINT}'/>")
    b.append(f"<line x1='102' y1='{MID+10:.0f}' x2='126' y2='{ycB-6}' stroke='{FAINT}' stroke-width='1'/>"
              f"<path d='M 128,{ycB} L 120.1,{ycB-7.3} L 127.3,{ycB-10.8} Z' fill='{FAINT}'/>")

    gA, gxA, gsA, mixA = member_lane(ycA, "A", 1)
    gB, gxB, gsB, mixB = member_lane(ycB, "B", -1)
    b.append(gA); b.append(gB)

    avx = gxA + gsA + 70
    b.append(f"<line x1='{gxA+gsA}' y1='{ycA}' x2='{avx-6}' y2='{MID-30}' stroke='{INK}' stroke-width='1'/>")
    b.append(f"<path d='M {avx},{MID-34} l -5,2.5 l 0,-5 Z' fill='{INK}'/>")
    b.append(f"<line x1='{gxB+gsB}' y1='{ycB}' x2='{avx-6}' y2='{MID+30}' stroke='{INK}' stroke-width='1'/>")
    b.append(f"<path d='M {avx},{MID+34} l -5,-2.5 l 0,5 Z' fill='{INK}'/>")
    b.append(txt(avx + 2, MID - 76, "average", 9, INK, 600, "start", style="font-style='italic'"))
    g, gsv = gridsq(avx, MID, 56, 8, INK, ((5, 2, .5), (4, 2, .22), (5, 1, .22), (5, 3, .14))); b.append(g)
    b.append(txt(avx, MID - 46, "averaged belief p̄", 10.5, INK, 600, anchor="start"))
    b.append(txt(avx, MID - 34, "mean of both members' softmax", 9.5, MUT, anchor="start"))

    winx, winy = avx + 21, MID - 21
    b.append(f"<rect x='{winx:.1f}' y='{winy:.1f}' width='14' height='14' fill='none' stroke='{INK}' stroke-width='1.8'/>")
    decx = 812
    for (dxp, dyp) in ((winx + 4, winy + 4), (winx + 14, winy + 4), (winx + 4, winy + 14), (winx + 10, winy + 1)):
        b.append(f"<line x1='{dxp:.1f}' y1='{dyp:.1f}' x2='{decx-10}' y2='{MID}' stroke='{INK}' stroke-width='0.5' opacity='0.4'/>")
    b.append(crosspt(decx, MID, INK))
    b.append(txt(decx, MID - 48, "pooled centroid, 3×3 block", 10.5, INK, 600, anchor="middle"))
    b.append(txt(decx, MID - 36, "+ mean of both offset nudges", 9.5, MUT, anchor="middle"))
    b.append(output_part(828, MID))

    gatex, gatey = decx - 40, MID + 80
    b.append(f"<line x1='{winx+2}' y1='{winy+14}' x2='{gatex-14}' y2='{gatey-9}' stroke='{INK}' stroke-width='1' stroke-dasharray='2 4'/>")
    b.append(f"<line x1='{winx+12}' y1='{winy+14}' x2='{gatex+14}' y2='{gatey-9}' stroke='{INK}' stroke-width='1' stroke-dasharray='2 4'/>")
    b.append(f"<path d='M {gatex-14},{gatey+5} A 14 14 0 0 1 {gatex+14},{gatey+5}' fill='none' stroke='{INK}' stroke-width='1.2'/>")
    b.append(f"<line x1='{gatex}' y1='{gatey+5}' x2='{gatex+6.3}' y2='{gatey-4}' stroke='{INK}' stroke-width='1.4'/>")
    b.append(f"<circle cx='{gatex}' cy='{gatey+5}' r='1.6' fill='{INK}'/>")
    b.append(cap(gatex, gatey + 27, "confidence = min(mᴀ, mʙ)", "both must independently back it"))
    b.append(txt(gatex, gatey + 50, "else the drone abstains", 8.3, INK, style="font-style='italic'"))

    # ---- training-only lane (bottom) ----
    b.append(f"<circle cx='230' cy='250' r='1.6' fill='{OCH}'/>")
    b.append(lpath("M 230,250 H 150 V 540 H 206", OCH))
    b.append(f"<path d='M 193,540 A 13 13 0 0 1 219,540' fill='none' stroke='{OCH}' stroke-width='1.2'/>")
    b.append(f"<circle cx='206' cy='540' r='1.4' fill='{OCH}'/>")
    b.append(txt(236, 546, "cosine margin m=0.25 (CosFace) + fine-tune all 11 blocks at 0.1× head LR", 10, OCH, 600, "start"))
    b.append(txt(236, 558, "shared head, same schedule as experiment 13 — not this experiment", 9.5, OCH, anchor="start"))

    mdotx, mdoty = 240, 95
    b.append(f"<circle cx='{mdotx}' cy='{mdoty}' r='1.8' fill='{ACC}'/>")
    for (mx, my) in mixA:
        b.append(lpath(f"M {mx:.1f},{my:.1f} V {mdoty-4} H {mdotx}", ACC, dash="1.5 3"))
    b.append(lpath(f"M {mdotx},{mdoty} V 580 H 300", ACC))
    for (mx, my) in mixB:
        b.append(lpath(f"M {mx:.1f},{my+2:.1f} V 580 H 340", ACC, dash="1.5 3"))
    b.append(f"<path d='M 287,580 A 13 13 0 0 1 313,580' fill='none' stroke='{ACC}' stroke-width='1.3'/>")
    b.append(f"<circle cx='300' cy='580' r='1.6' fill='{ACC}'/>")
    b.append(txt(236, 598, "NEW — MixStyle: cross-survey style mixing after trunk blocks 1, 3, 6 (both members)", 10.5, ACC, 600, "start"))
    b.append(txt(236, 610, "each view's early-layer channel mean/std blended with a random batch partner's, λ~Beta(0.1,0.1)", 9.5, ACC, anchor="start"))
    b.append(txt(236, 622, "partner usually another survey (batches are already survey-paired); identity at eval/export — ONNX unchanged", 9.5, ACC, anchor="start", style="font-style='italic'"))

    b.append(txt(972, 634, "pooling, offset head, confidence gate, cosine margin, paired batches unchanged", 9, FAINT, 400, "end", style="font-style='italic'"))

    body = "".join(b)
    return f"<svg viewBox='0 0 980 {H}' xmlns='http://www.w3.org/2000/svg' role='img'>{body}</svg>"


if __name__ == "__main__":
    svg = build()
    with open("runs/20261101_180233_experiment14/figure.json", "w") as f:
        json.dump({"architecture_svg": svg}, f)
    print(len(svg), "MID", MID)
