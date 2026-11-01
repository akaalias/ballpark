"""Build experiment 14 figure: cross-survey style mixing (MixStyle) inside
the pretrained trunk. Same committee architecture as experiment 13
(two independent trunk+cosine-head members, averaged belief, local pooling,
offset, min-confidence gate) -- that whole inference path is UNCHANGED here,
so it is drawn entirely in ink. The only change is a training-only
feature-statistic mixing applied after trunk blocks 1, 3 and 6; it is the
identity at eval/export, so it never appears in the flown graph."""
import json
import math

INK, MUT, FAINT, ACC, OCH = "#111111", "#6b6a60", "#9b998c", "#8c2f1f", "#8a6a1e"
FONT = "Palatino,Georgia,serif"


def txt(x, y, s, size=10, color=MUT, w=400, anchor="middle", style="", tid=None):
    tid_attr = f"id='{tid}' " if tid else ""
    return (f"<text {tid_attr}x='{x:.1f}' y='{y:.1f}' font-family='{FONT}' font-size='{size}' "
            f"fill='{color}' font-weight='{w}' text-anchor='{anchor}' {style}>{s}</text>")


def cap(xc, y, name, sub=None, color=MUT, name_color=None):
    out = [txt(xc, y, name, 10.5, name_color or (INK if color == MUT else color), 600)]
    if sub:
        out.append(txt(xc, y + 12, sub, 9.5, color))
    return "".join(out)


def harrow(x1, x2, y, label=None, color=FAINT):
    out = [f"<line x1='{x1}' y1='{y}' x2='{x2 - 5}' y2='{y}' stroke='{color}' stroke-width='1'/>",
           f"<path d='M {x2},{y} l -6,-3 v 6 Z' fill='{color}'/>"]
    if label:
        out.append(txt((x1 + x2) / 2, y - 6, label, 9, FAINT, style="font-style='italic'"))
    return "".join(out)


def leader(x1, y1, x2, y2, color, dash="2 4"):
    return (f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' "
            f"stroke-width='1' stroke-dasharray='{dash}'/>")


def lpath(d, color, dash="2 4"):
    return f"<path d='{d}' fill='none' stroke='{color}' stroke-width='1' stroke-dasharray='{dash}'/>"


def slab(x, yc, s, d, color=INK):
    y = yc - s / 2
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


def kernel_proj(x1, yc1, s1, x2, yc2, s2, color, star=False):
    k = max(5, s1 * 0.30)
    kx, ky = x1 + s1 * 0.60, yc1 - s1 / 2 + s1 * 0.18
    c = max(3, s2 * 0.14)
    cx, cy = x2 + s2 * 0.22, yc2 - s2 / 2 + s2 * 0.26
    out = [f"<rect x='{kx:.1f}' y='{ky:.1f}' width='{k:.1f}' height='{k:.1f}' "
           f"fill='none' stroke='{color}' stroke-width='{1.3 if star else 0.9}'/>",
           f"<rect x='{cx:.1f}' y='{cy:.1f}' width='{c:.1f}' height='{c:.1f}' "
           f"fill='{color}' fill-opacity='0.5' stroke='none'/>"]
    for (px, py) in ((kx, ky), (kx + k, ky), (kx, ky + k), (kx + k, ky + k)):
        out.append(f"<line x1='{px:.1f}' y1='{py:.1f}' x2='{cx + c / 2:.1f}' "
                   f"y2='{cy + c / 2:.1f}' stroke='{color}' stroke-width='0.55' "
                   f"opacity='0.45'/>")
    if star:
        # small red tick radiating from the kernel square's outer corner --
        # marks "MixStyle runs on this block's output"
        sx, sy = kx - 3, ky - 3
        for ang in (0, 60, 120, 180, 240, 300):
            a = math.radians(ang)
            out.append(f"<line x1='{sx:.1f}' y1='{sy:.1f}' x2='{sx + 4.5*math.cos(a):.1f}' "
                       f"y2='{sy + 4.5*math.sin(a):.1f}' stroke='{color}' stroke-width='0.8'/>")
    return "".join(out), (kx, ky, k)


def bar(x, yc, h, color=INK, ticks=9):
    y = yc - h / 2
    out = [f"<rect x='{x}' y='{y}' width='7' height='{h}' fill='none' "
           f"stroke='{color}' stroke-width='1.2'/>"]
    for i in range(1, ticks):
        out.append(f"<line x1='{x}' y1='{y + i * h / ticks:.1f}' x2='{x + 7}' "
                   f"y2='{y + i * h / ticks:.1f}' stroke='{color}' stroke-width='0.5' opacity='0.5'/>")
    return "".join(out), 7


def gridsq(x, yc, s, n, color=INK, bumps=(), lw=1.2):
    y = yc - s / 2
    out = [f"<rect x='{x}' y='{y}' width='{s}' height='{s}' fill='none' "
           f"stroke='{color}' stroke-width='{lw}'/>"]
    for i in range(1, n):
        out.append(f"<line x1='{x + i * s / n:.1f}' y1='{y}' x2='{x + i * s / n:.1f}' "
                   f"y2='{y + s}' stroke='{color}' stroke-width='0.45' opacity='0.5'/>")
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


def cos_head(x, yc, color=INK):
    out = [f"<circle cx='{x}' cy='{yc}' r='12' fill='none' stroke='{color}' stroke-width='1.1'/>"]
    for ang in (-70, -40, 10, 65, 130):
        a = math.radians(ang)
        out.append(f"<line x1='{x + 12*math.cos(a)*0.3:.1f}' y1='{yc + 12*math.sin(a)*0.3:.1f}' "
                   f"x2='{x + 12*math.cos(a)*0.95:.1f}' y2='{yc + 12*math.sin(a)*0.95:.1f}' "
                   f"stroke='{color}' stroke-width='0.7' opacity='0.5'/>")
    out.append(f"<line x1='{x}' y1='{yc}' x2='{x+10}' y2='{yc+7}' stroke='{color}' stroke-width='1.4'/>")
    out.append(f"<circle cx='{x}' cy='{yc}' r='1.4' fill='{color}'/>")
    return "".join(out)


def cam_terrain(Y):
    out = [f"<g id='cam-terrain'><rect id='frozen-input' x='26' y='{Y}' width='76' height='76' "
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
           f"<circle cx='31' cy='{Y+42}' r='.7' fill='#6b6a60' opacity='.5'/></g>"]
    return "".join(out)


def member_lane(yc, label):
    """One committee member: trunk (4 representative slabs, 3 of whose
    inter-block junctions are MixStyle insertion points) -> mid tap -> lift
    56 -> deep tap (blocks 9-10) -> lift 56 -> concat 112 -> BN -> unit
    embed -> cosine head -> belief grid 8x8."""
    b = []
    mix_pts = []  # (x, y) of the 3 MixStyle junctions, for leader routing

    sizes = [22, 16, 11, 8]
    x = 164
    faces = []
    for s in sizes:
        g, w = slab(x, yc, s, s * 0.95, INK)
        b.append(g)
        faces.append((x, s))
        x += w + 15 if s == 22 else (w + 11 if s == 16 else w + 10 if s == 11 else w)
    # recompute exact faces/x precisely to avoid drift: redo spacing cleanly
    b.clear()
    faces = []
    x = 164
    gaps = [19, 16, 13]
    for i, s in enumerate(sizes):
        g, w = slab(x, yc, s, s * 0.95, INK)
        b.append(g)
        faces.append((x, s))
        if i < 3:
            x += w + gaps[i]
    end_x = faces[-1][0] + faces[-1][1] + faces[-1][1] * 0.95

    # junctions: frame->slab0 drawn by caller; slab0-1, slab1-2, slab2-3 are
    # the 3 MixStyle insertion points (after blocks 1, 3, 6)
    for i in range(3):
        (x1, s1), (x2, s2) = faces[i], faces[i + 1]
        g, (kx, ky, k) = kernel_proj(x1, yc, s1, x2, yc, s2, ACC, star=True)
        b.append(g)
        mix_pts.append((kx + k / 2, ky - 2))

    cap_x = (faces[0][0] + end_x) / 2
    b.append(cap(cap_x, yc + 38, f"trunk {label} (pretrained)", "feat[0:9] . 8sq x48", color=MUT))

    # mid tap -> lift 48->56
    tap_x = end_x + 6
    b.append(f"<circle cx='{tap_x:.1f}' cy='{yc}' r='2' fill='{INK}'/>")
    b.append(harrow(tap_x, tap_x + 23, yc))
    lx = tap_x + 25
    g, w = slab(lx, yc, 11, 11, INK); b.append(g)
    b.append(cap(lx + 2, yc + 38, "lift 48→56", color=MUT, name_color=INK))
    bx = lx + w + 17
    b.append(harrow(lx + w + 2, bx - 2, yc))
    g, _ = bar(bx, yc, 40, INK, ticks=6); b.append(g)
    b.append(txt(bx + 3, yc - 30, "56d GAP", 8, FAINT))

    # deep tap (blocks 9-10, already fine-tuned since exp13 -- unchanged now)
    deep_y = yc + 108
    (dx0, ds0) = faces[1]
    b.append(f"<line x1='{dx0+2:.1f}' y1='{yc+ds0/2+2:.1f}' x2='{dx0+14:.1f}' y2='{deep_y-7:.1f}' "
             f"stroke='{INK}' stroke-width='1'/>")
    b.append(f"<path d='M {dx0+16:.1f},{deep_y:.1f} L {dx0+10:.1f},{deep_y-9:.1f} L {dx0+18:.1f},{deep_y-10:.1f} Z' fill='{INK}'/>")
    gdx = dx0 + 20
    g, w = slab(gdx, deep_y, 9, 9, INK); b.append(g)
    g2, w2 = slab(gdx + w + 6, deep_y, 6, 6, INK); b.append(g2)
    b.append(txt(gdx + w / 2 + 10, deep_y - 20, "feat[9:11] . 4sq x96", 8, FAINT))
    b.append(cap(gdx + 6, deep_y + 24, "blocks 9–10", color=MUT, name_color=INK))
    dtap_x = gdx + w + w2 + 10
    b.append(f"<circle cx='{dtap_x:.1f}' cy='{deep_y:.1f}' r='2' fill='{INK}'/>")
    b.append(harrow(dtap_x, dtap_x + 17, deep_y))
    dlx = dtap_x + 19
    g, dw = slab(dlx, deep_y, 9, 9, INK); b.append(g)
    b.append(cap(dlx + 1, deep_y + 24, "lift 96→56", color=MUT, name_color=INK))
    dbx = dlx + dw + 17
    b.append(harrow(dlx + dw + 2, dbx - 2, deep_y))
    g, _ = bar(dbx, deep_y, 40, INK, ticks=6); b.append(g)
    b.append(txt(dbx + 3, deep_y - 30, "56d GAP", 8, FAINT))

    # concat 112-d
    cx0 = max(bx, dbx) + 24
    b.append(f"<line x1='{bx+8:.1f}' y1='{yc:.1f}' x2='{cx0-2:.1f}' y2='{yc+19:.1f}' stroke='{INK}' stroke-width='0.8'/>")
    b.append(f"<line x1='{dbx+8:.1f}' y1='{deep_y:.1f}' x2='{cx0-2:.1f}' y2='{yc+21:.1f}' stroke='{INK}' stroke-width='0.8'/>")
    ccy = yc + 20
    g, _ = bar(cx0, ccy, 40, INK, ticks=6); b.append(g)
    b.append(txt(cx0 + 3, ccy - 30, "112-d", 10.5, INK, 600))
    b.append(txt(cx0 + 3, ccy + 34, "concat: 56+56", 8.5, MUT))

    # BN -> normalize -> cosine head
    bnx = cx0 + 20
    b.append(harrow(cx0 + 9, bnx - 2, ccy))
    g, _ = bar(bnx, ccy, 20, INK, ticks=2); b.append(g)
    b.append(txt(bnx + 3, ccy - 16, "BN", 9, INK, 600))
    cosx = bnx + 32
    b.append(harrow(bnx + 9, cosx - 14, ccy))
    b.append(cos_head(cosx, ccy, INK))
    b.append(txt(cosx, ccy - 18, "unit embed.→cos", 9.5, INK, 600))
    b.append(txt(cosx, ccy - 6, "2970 protos", 8.5, MUT))

    # belief grid
    gx = cosx + 22
    b.append(harrow(cosx + 14, gx - 2, ccy))
    g, gs = gridsq(gx, ccy, 58, 8, INK, BUMP); b.append(g)
    b.append(cap(gx + gs / 2, ccy - 44, f"belief field {label}", "cosine x s=30", color=MUT, name_color=INK))

    return "".join(b), gx, gs, ccy, mix_pts


def output_part(x):
    return (txt(x, 229, "frozen contract", 8.5, FAINT, anchor="start", style="font-style='italic'")
            + txt(x, 245, "(lat, lon, confidence)", 13, MUT, 600, "start", tid="frozen-output")
            + txt(x, 259, "position fix + confidence", 9, FAINT, anchor="start"))


def build():
    H = 660
    b = [txt(8, 20, "INFERENCE PATH — WHAT FLIES", 9, FAINT, 600, "start", "letter-spacing='1.8'"),
         txt(8, H - 44, "TRAINING SIGNALS — NEVER FLY", 9, OCH, 600, "start", "letter-spacing='1.8'")]

    b.append(cam_terrain(99))
    b.append(txt(64, 93, "128²×3", 9, FAINT))
    b.append(txt(64, 185, "camera frame", 10.5, MUT, 600))
    b.append(txt(64, 197, "one TRAIN survey", 9.5, FAINT))
    b.append(txt(64, 209, "frozen contract", 8.5, FAINT, style="font-style='italic'"))

    # two branch arrows to lane A (y~92) and lane B (y~300)
    b.append(f"<line x1='102' y1='137' x2='126.8' y2='96' stroke='{FAINT}' stroke-width='1'/>"
              f"<path d='M 128,89 L 130.3,99.5 L 122.4,98.2 Z' fill='{FAINT}'/>")
    b.append(f"<line x1='102' y1='137' x2='125.0' y2='298' stroke='{FAINT}' stroke-width='1'/>"
              f"<path d='M 128,305 L 120.1,297.7 L 127.3,294.2 Z' fill='{FAINT}'/>")
    b.append(txt(118, 155, "splits —", 8.5, FAINT, style="font-style='italic'"))
    b.append(txt(118, 166, "no shared", 8.5, FAINT, style="font-style='italic'"))
    b.append(txt(118, 177, "weights", 8.5, FAINT, style="font-style='italic'"))

    gA, gxA, gsA, ccyA, mixA = member_lane(92, "A")
    gB, gxB, gsB, ccyB, mixB = member_lane(300, "B")
    b.append(gA); b.append(gB)

    # converge both belief fields -> averaged belief -> decode
    avx, avy = 781, 196
    b.append(f"<line x1='{gxA+gsA}' y1='{ccyA}' x2='{avx-6}' y2='{avy-34}' stroke='{INK}' stroke-width='1'/>")
    b.append(f"<path d='M {avx},{avy-40} l -5,2.5 l 0,-5 Z' fill='{INK}'/>")
    b.append(f"<line x1='{gxB+gsB}' y1='{ccyB}' x2='{avx-6}' y2='{avy-18}' stroke='{INK}' stroke-width='1'/>")
    b.append(f"<path d='M {avx},{avy-12} l -5,-2.5 l 0,5 Z' fill='{INK}'/>")
    b.append(txt(avx + 2, avy - 55, "average", 9, INK, 600, "start", style="font-style='italic'"))
    g, gsv = gridsq(avx, avy, 58, 8, INK, ((5, 2, .5), (4, 2, .22), (5, 1, .22), (5, 3, .14))); b.append(g)
    b.append(txt(avx, avy - 74, "averaged belief p̄", 10.5, INK, 600, anchor="start"))
    b.append(txt(avx, avy - 60, "mean of both members' softmax", 9.5, MUT, anchor="start"))

    winx, winy = avx + 21.75, avy + 21.75
    b.append(f"<rect x='{winx:.1f}' y='{winy:.1f}' width='14.5' height='14.5' fill='none' stroke='{INK}' stroke-width='1.8'/>")
    decx, decy = 812, avy + 29
    for (dxp, dyp) in ((winx + 7, winy + 7), (winx + 14.5, winy + 7), (winx + 7, winy + 14.5), (winx + 2, winy + 17)):
        b.append(f"<line x1='{dxp:.1f}' y1='{dyp:.1f}' x2='{decx-10}' y2='{decy}' stroke='{INK}' stroke-width='0.5' opacity='0.4'/>")
    b.append(crosspt(decx, decy, INK))
    b.append(txt(decx + 49, decy - 42, "pooled centroid of the 3×3 block", 10.5, INK, 600, anchor="start"))
    b.append(txt(decx + 49, decy - 30, "+ mean of both members' offset nudges", 9.5, MUT, anchor="start"))
    b.append(output_part(828))

    # confidence gate below decode
    gatex, gatey = decx - 70, decy + 100
    b.append(f"<line x1='{winx+4}' y1='{winy+14.5}' x2='{gatex-14}' y2='{gatey-9}' stroke='{INK}' stroke-width='1' stroke-dasharray='2 4'/>")
    b.append(f"<line x1='{winx+10.5}' y1='{winy+14.5}' x2='{gatex+14}' y2='{gatey-9}' stroke='{INK}' stroke-width='1' stroke-dasharray='2 4'/>")
    b.append(f"<path d='M {gatex-14},{gatey+5} A 14 14 0 0 1 {gatex+14},{gatey+5}' fill='none' stroke='{INK}' stroke-width='1.2'/>")
    b.append(f"<line x1='{gatex}' y1='{gatey+5}' x2='{gatex+6.3}' y2='{gatey-4}' stroke='{INK}' stroke-width='1.4'/>")
    b.append(f"<circle cx='{gatex}' cy='{gatey+5}' r='1.6' fill='{INK}'/>")
    b.append(cap(gatex, gatey + 27, "confidence = min(mᴀ, mʙ)", "both must independently back it",
                 color=MUT, name_color=INK))
    b.append(txt(gatex, gatey + 52, "or the drone abstains", 8.3, INK))
    b.append(f"<path d='M {gatex},{gatey+58} V {gatey+78} H {decx+40} V {decy+14}' fill='none' stroke='{INK}' stroke-width='0.8'/>")
    b.append(f"<path d='M {decx+40},{decy+8} l -3,6 h 6 Z' fill='{INK}'/>")
    b.append(txt(decx - 15, gatey + 72, "may abstain instead of guessing", 8.5, INK, style="font-style='italic'"))

    # ---- training-only lane ----
    LY = H - 96
    # unchanged ochre notes, same substance as experiment 13: cosine margin
    # and fine-tune LR, now both fully background facts, not THIS change
    b.append(f"<circle cx='624' cy='87' r='1.6' fill='{OCH}'/>")
    b.append(lpath(f"M 624,87 V 61 H 108 V {LY-18} H 206", OCH))
    b.append(f"<path d='M 193,{LY-18} A 13 13 0 0 1 219,{LY-18}' fill='none' stroke='{OCH}' stroke-width='1.2'/>")
    b.append(f"<line x1='206' y1='{LY-18}' x2='198.5' y2='{LY-28.6}' stroke='{OCH}' stroke-width='1.3'/>")
    b.append(f"<circle cx='206' cy='{LY-18}' r='1.4' fill='{OCH}'/>")
    b.append(txt(236, LY - 36, "additive cosine margin m=0.25 on the target cell (unchanged)", 10, OCH, 600, "start"))
    b.append(txt(236, LY - 24, "CosFace-style, shared head — not this experiment", 9.5, OCH, anchor="start"))

    b.append(f"<circle cx='174' cy='38' r='1.6' fill='{OCH}'/>")
    b.append(lpath(f"M 174,38 V {LY+4} H 222", OCH))
    b.append(txt(236, LY, "all eleven blocks fine-tuned at 0.1× the head's LR (unchanged since exp 13)", 10, OCH, 600, "start"))
    b.append(txt(236, LY + 12, "same cosine schedule, same paired three-survey batches", 9.5, OCH, anchor="start"))

    # NEW red: MixStyle, with leaders from the 6 insertion-junction markers
    # (3 per lane) routed orthogonally down into one description
    mly = LY + 34
    b.append(f"<circle cx='230' cy='40' r='1.8' fill='{ACC}'/>")
    for i, (mx, my) in enumerate(mixA):
        b.append(lpath(f"M {mx:.1f},{my:.1f} V 28 H 230 V 40", ACC, dash="1.5 3"))
    b.append(lpath(f"M 230,40 V {mly-6}", ACC))
    for i, (mx, my) in enumerate(mixB):
        b.append(lpath(f"M {mx:.1f},{my+2:.1f} V {mly-6} H 280", ACC, dash="1.5 3"))
    b.append(f"<path d='M 274,{mly-16} A 13 13 0 0 1 300,{mly-16}' fill='none' stroke='{ACC}' stroke-width='1.3'/>")
    b.append(f"<line x1='287' y1='{mly-16}' x2='281' y2='{mly-27}' stroke='{ACC}' stroke-width='1.4'/>")
    b.append(f"<line x1='287' y1='{mly-16}' x2='293' y2='{mly-27}' stroke='{ACC}' stroke-width='1.4' stroke-dasharray='1.4 1.6'/>")
    b.append(f"<circle cx='287' cy='{mly-16}' r='1.6' fill='{ACC}'/>")
    b.append(txt(236, mly - 2, "NEW — cross-survey style mixing (MixStyle), after blocks 1, 3, 6", 10.5, ACC, 600, "start"))
    b.append(txt(236, mly + 11, "per-channel feature mean/std blended with a random batch partner's, λ~Beta(0.1,0.1)", 9.5, ACC, anchor="start"))
    b.append(txt(236, mly + 23, "partner usually a different survey (batches are already survey-paired) — spatial layout untouched", 9.5, ACC, anchor="start"))
    b.append(txt(236, mly + 35, "identity in eval / export — the flown graph above is pixel-for-pixel unchanged", 9.5, ACC, anchor="start", style="font-style='italic'"))
    # small red asterisk legend near the two "style mix" marks already drawn on the trunks
    b.append(txt(968, LY - 58, "head, pooling, offset head, confidence gate, cosine margin unchanged", 9, FAINT, 400, "end", style="font-style='italic'"))

    body = "".join(b)
    return f"<svg viewBox='0 0 980 {H}' xmlns='http://www.w3.org/2000/svg' role='img'>{body}</svg>"


if __name__ == "__main__":
    svg = build()
    out = {"architecture_svg": svg}
    with open("runs/20261101_180233_experiment14/figure.json", "w") as f:
        json.dump(out, f)
    print(len(svg))
