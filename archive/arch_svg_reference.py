"""Technical architecture diagrams for experiments 1-6, drawn in tensor-native
visual language: images/fields as real grids, feature maps as 3D slabs,
vectors as tick-bars, FC as connection fans, decodes as lines converging to a
point. Captions sit under elements; no labeled boxes."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoresearch.db import connect
from terrain_frame_glyph import terrain_frame

INK, MUT, FAINT, ACC, OCH = "#111111", "#6b6a60", "#9b998c", "#8c2f1f", "#8a6a1e"
FONT = "Palatino,Georgia,serif"
IC = 112          # inference lane center y
LOSS_Y = 292      # loss-annotation row baseline


def txt(x, y, s, size=10, color=MUT, w=400, anchor="middle", style="", tid=None):
    tid_attr = f"id='{tid}' " if tid else ""
    return (f"<text {tid_attr}x='{x:.0f}' y='{y:.0f}' font-family='{FONT}' font-size='{size}' "
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


def leader(x1, y1, x2, y2, color):
    """Slanted dashed annotation leader, paper-margin style."""
    return (f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' "
            f"stroke-width='1' stroke-dasharray='2 4'/>")


def slab(x, s, d, color=INK):
    """Pseudo-3D feature-map slab: front s x s, depth d. Returns (svg, width)."""
    y = IC - s / 2
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


def imgsq(x, s, color=INK, tid=None):
    """Input image: square with a sparse pixel-grid texture."""
    y = IC - s / 2
    tid_attr = f"id='{tid}' " if tid else ""
    out = [f"<rect {tid_attr}x='{x}' y='{y}' width='{s}' height='{s}' fill='#00000008' "
           f"stroke='{color}' stroke-width='1.4'/>"]
    n = 6
    for i in range(1, n):
        out.append(f"<line x1='{x + i * s / n:.0f}' y1='{y}' x2='{x + i * s / n:.0f}' "
                   f"y2='{y + s}' stroke='{color}' stroke-width='0.4' opacity='0.3'/>")
        out.append(f"<line x1='{x}' y1='{y + i * s / n:.0f}' x2='{x + s}' "
                   f"y2='{y + i * s / n:.0f}' stroke='{color}' stroke-width='0.4' opacity='0.3'/>")
    for (a, b, o) in ((1, 2, .35), (3, 1, .5), (2, 4, .4), (4, 3, .3), (0, 4, .25), (4, 0, .3)):
        out.append(f"<rect x='{x + a * s / n:.0f}' y='{y + b * s / n:.0f}' "
                   f"width='{s / n:.0f}' height='{s / n:.0f}' fill='{color}' opacity='{o * 0.35}'/>")
    return "".join(out), s


def bar(x, h, color=INK, ticks=9):
    """1-D vector as a thin vertical tick-bar."""
    y = IC - h / 2
    out = [f"<rect x='{x}' y='{y}' width='7' height='{h}' fill='none' "
           f"stroke='{color}' stroke-width='1.2'/>"]
    for i in range(1, ticks):
        out.append(f"<line x1='{x}' y1='{y + i * h / ticks:.1f}' x2='{x + 7}' "
                   f"y2='{y + i * h / ticks:.1f}' stroke='{color}' stroke-width='0.5' opacity='0.5'/>")
    return "".join(out), 7


def fan(x1, h1, x2, h2, color=FAINT, n=5):
    """Fully-connected layer as a fan of crossing connections."""
    out = []
    for i in range(n):
        ya = IC - h1 / 2 + (i + 0.5) * h1 / n
        for j in range(n):
            yb = IC - h2 / 2 + (j + 0.5) * h2 / n
            out.append(f"<line x1='{x1}' y1='{ya:.1f}' x2='{x2}' y2='{yb:.1f}' "
                       f"stroke='{color}' stroke-width='0.5' opacity='0.45'/>")
    return "".join(out)


def gridsq(x, s, n, color=INK, bumps=(), yc=None, lw=1.2):
    yc = IC if yc is None else yc
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


def converge(gx, gs, tx, color, cells="all", n=8):
    """Decode drawn as cell-centers voting: lines from grid cells to a point."""
    out = []
    src = []
    y0 = IC - gs / 2
    if cells == "all":
        xs = ys = range(n)
    else:
        # 5x5 refinement window: cols 3-7, rows 1-5 — must match the red
        # window rect exp3 draws (wx = gx+3gs/8, wy = y0+1gs/8), so every
        # vote line starts at a cell center INSIDE the drawn window.
        xs, ys = range(3, 8), range(1, 6)
    for gyy in ys:
        for gxx in xs:
            if cells == "all" and (gxx + gyy) % 2:
                continue
            src.append((gx + (gxx + 0.5) * gs / n, y0 + (gyy + 0.5) * gs / n))
    for (sx, sy) in src:
        out.append(f"<line x1='{sx:.0f}' y1='{sy:.0f}' x2='{tx}' y2='{IC}' "
                   f"stroke='{color}' stroke-width='0.55' opacity='0.4'/>")
    out.append(crosspt(tx, IC, color))
    return "".join(out)


def gauge(x, yc, color=MUT, r=14):
    a = math.radians(55)
    return (f"<path d='M {x - r},{yc} A {r} {r} 0 0 1 {x + r},{yc}' fill='none' "
            f"stroke='{color}' stroke-width='1.2'/>"
            f"<line x1='{x}' y1='{yc}' x2='{x + (r - 3) * math.cos(a):.1f}' "
            f"y2='{yc - (r - 3) * math.sin(a):.1f}' stroke='{color}' stroke-width='1.4'/>"
            f"<circle cx='{x}' cy='{yc}' r='1.6' fill='{color}'/>")


def lanes(h):
    return (txt(8, 26, "INFERENCE PATH — WHAT FLIES", 9, FAINT, 600, "start",
                "letter-spacing='1.8'")
            + txt(8, LOSS_Y - 34, "TRAINING SIGNALS — NEVER FLY", 9, OCH, 600, "start",
                  "letter-spacing='1.8'"))


def svgwrap(h, body):
    return (f"<svg viewBox='0 0 980 {h}' xmlns='http://www.w3.org/2000/svg' role='img'>"
            f"{body}</svg>")


def kernel_proj(x1, s1, x2, s2, color):
    """Conv connection: small kernel square on layer i's face, faint lines
    converging to a cell on layer i+1 — computed-from, not standing-next-to."""
    k = max(5, s1 * 0.30)
    kx, ky = x1 + s1 * 0.60, IC - s1 / 2 + s1 * 0.18
    c = max(3, s2 * 0.14)
    cx, cy = x2 + s2 * 0.22, IC - s2 / 2 + s2 * 0.26
    out = [f"<rect x='{kx:.1f}' y='{ky:.1f}' width='{k:.1f}' height='{k:.1f}' "
           f"fill='none' stroke='{color}' stroke-width='0.9'/>",
           f"<rect x='{cx:.1f}' y='{cy:.1f}' width='{c:.1f}' height='{c:.1f}' "
           f"fill='{color}' fill-opacity='0.5' stroke='none'/>"]
    for (px, py) in ((kx, ky), (kx + k, ky), (kx, ky + k), (kx + k, ky + k)):
        out.append(f"<line x1='{px:.1f}' y1='{py:.1f}' x2='{cx + c / 2:.1f}' "
                   f"y2='{cy + c / 2:.1f}' stroke='{color}' stroke-width='0.55' "
                   f"opacity='0.45'/>")
    return "".join(out)


def conv_group(x0, colors=None, prev=None):
    """The shared encoder: the four conv OUTPUT tensors as slabs (64²×16 →
    8²×128); each of the 4 convs is a kernel projection — the first from the
    input image itself (prev=(x, s, color))."""
    sizes = [(46, 7), (34, 11), (24, 15), (17, 21)]
    out, x, faces = [], x0, []
    for i, (s, d) in enumerate(sizes):
        g, w = slab(x, s, d, (colors or [INK] * 4)[i])
        out.append(g)
        faces.append((x, s))
        x += w + 15
    x -= 15
    if prev:
        out.append(kernel_proj(prev[0], prev[1], faces[0][0], faces[0][1], prev[2]))
    for (x1, s1), (x2, s2) in zip(faces, faces[1:]):
        out.append(kernel_proj(x1, s1, x2, s2, MUT))
    xc = (x0 + x) / 2
    out.append(cap(xc, IC + 58, "convolutional encoder", "Conv 3×3, stride 2 ×4 · BN + ReLU"))
    out.append(txt(x0 + 23, IC - 36, "64²×16", 9, FAINT))
    out.append(txt(x - 20, IC - 32, "8²×128", 9, FAINT))
    return "".join(out), x


def frame_part():
    # canonical 76x76 nadir-terrain glyph (prompt.md: copy verbatim),
    # centered on the inference lane; captions stay centered on x=53
    g = terrain_frame(IC - 38)
    out = (g + txt(53, IC - 44, "128²×3", 9, FAINT)
           + txt(53, IC + 48, "camera frame", 10.5, MUT, 600)
           + txt(53, IC + 60, "one night exposure", 9.5, FAINT)
           + txt(53, IC + 73, "frozen contract", 8.5, FAINT,
                 style="font-style='italic'"))
    return out, 26 + 76


def conf_branch(bar_x, out_x, color=FAINT):
    gx = bar_x + 4
    ex = out_x + 28
    out = [leader(gx, IC + 64, gx, 216, FAINT),
           gauge(gx, 230, MUT),
           txt(gx, 248, "confidence 0–1", 9, MUT),
           f"<path d='M {gx + 18},230 H {ex} V {IC + 26}' fill='none' "
           f"stroke='{FAINT}' stroke-width='0.8'/>",
           f"<path d='M {ex},{IC + 20} l -3,6 h 6 Z' fill='{FAINT}'/>",
           txt((gx + ex) / 2, 224, "may abstain instead of guessing", 8.5, FAINT,
               style="font-style='italic'")]
    return "".join(out)


def output_part(x, sub="position fix + confidence"):
    return (txt(x, IC - 18, "frozen contract", 8.5, FAINT, anchor="start",
                style="font-style='italic'")
            + txt(x, IC - 2, "(lat, lon, confidence)", 13, MUT, 600, "start",
                  tid="frozen-output")
            + txt(x, IC + 12, sub, 9, FAINT, anchor="start"))


def loss_note(x_from, y_from, x_lab, text, color, glyph=None):
    out = [leader(x_from, y_from, x_lab - 8, LOSS_Y - 16, color)]
    if glyph:
        out.append(glyph)
    out.append(txt(x_lab, LOSS_Y - 12, text[0], 10, color, 600, "start"))
    if len(text) > 1:
        out.append(txt(x_lab, LOSS_Y, text[1], 9.5, color, 400, "start"))
    return "".join(out)


# ---------------------------------------------------------------- exp 1
def exp1():
    b = [lanes(330)]
    g, x = frame_part(); b.append(g)
    b.append(harrow(x + 6, x + 30, IC)); x += 32
    g, x = conv_group(x, prev=(26, 76, FAINT)); b.append(g)
    b.append(harrow(x + 8, x + 30, IC)); x += 34
    g, w = bar(x, 64, ticks=11); b.append(g)
    b.append(cap(x + 4, IC + 46, "128-d vector", "global avg pool"))
    fx = x + 12
    g, w2 = bar(fx + 56, 18, ticks=3)
    b.append(fan(fx, 64, fx + 56, 18)); b.append(g)
    b.append(cap(fx + 60, IC + 84, "FC 128 → 3", "sigmoid"))
    x = fx + 70
    b.append(harrow(x, 796, IC, "one shot"))
    b.append(crosspt(812, IC, INK))
    b.append(output_part(828))
    b.append(loss_note(fx + 60, IC + 14, 380, ("mean-squared error on (u, v)",
             "the whole map supervises one number pair"), OCH))
    b.append(loss_note(812, IC + 20, 660, ("BCE ×0.1 on confidence",), OCH))
    b.append(txt(968, LOSS_Y - 12, "starting design — nothing changed yet", 9.5, FAINT,
                 anchor="end", style="font-style='italic'"))
    return svgwrap(330, "".join(b))


# ------------------------------------------------ shared probability-field chain
def field_chain(fc_color, field_color, dec_color, dec_cells, dec_name, dec_sub):
    b = []
    g, x = frame_part(); b.append(g)
    b.append(harrow(x + 6, x + 30, IC)); x += 32
    g, x = conv_group(x, prev=(26, 76, FAINT)); b.append(g)
    b.append(harrow(x + 8, x + 30, IC)); x += 34
    g, _ = bar(x, 64, ticks=11); b.append(g)
    b.append(cap(x + 4, IC + 46, "128-d", "GAP"))
    bar_x = x
    fx = x + 12
    g, _ = bar(fx + 60, 92, fc_color, ticks=15)
    b.append(fan(fx, 64, fx + 60, 92, FAINT if fc_color == INK else fc_color))
    b.append(g)
    b.append(cap(fx + 64, IC + 84, "FC → 1024 logits", "softmax",
                 name_color=fc_color if fc_color == ACC else None))
    x = fx + 74
    b.append(harrow(x, x + 26, IC))
    gx = x + 28
    g, gs = gridsq(gx, 76, 8, field_color, BUMP)
    b.append(g)
    b.append(cap(gx + 38, IC + 56, "probability field", "32×32 cells over the map",
                 name_color=field_color if field_color == ACC else None))
    tx = 812  # shared right anchor: every figure's output lines up
    b.append(converge(gx, 76, tx, dec_color, cells=dec_cells))
    b.append(cap((gx + gs + tx) / 2 + 4, IC - 54, dec_name, dec_sub,
                 name_color=dec_color if dec_color == ACC else None))
    b.append(output_part(828))
    b.append(conf_branch(bar_x, tx + 18))
    return b, gx, gs, tx, bar_x


# ---------------------------------------------------------------- exp 2
def exp2():
    b, gx, gs, tx, _ = field_chain(ACC, ACC, ACC, "all",
                                   "soft-argmax over ALL cells",
                                   "answer = Σ probability · cell-center")
    body = [lanes(330)] + b
    tiny, _ = gridsq(486, 22, 6, ACC, ((3, 2, .6), (2, 2, .25), (3, 1, .25)), yc=LOSS_Y - 12, lw=1)
    body.append(loss_note(gx + 38, IC + 76, 516,
                          ("cross-entropy: match a Gaussian bump on the true cell",
                           "σ = 1.5 cells — every cell gets a hotter/colder gradient"),
                          ACC, glyph=tiny))
    body.append(loss_note(tx, IC + 12, 856, ("L2 on the decoded (u, v)",), ACC))
    return svgwrap(330, "".join(body))


# ---------------------------------------------------------------- exp 3
def exp3():
    b, gx, gs, tx, _ = field_chain(INK, INK, ACC, "window",
                                   "argmax → 5×5 window soft-argmax",
                                   "commit to the hottest cell, refine locally")
    body = [lanes(330)] + b
    wx, wy = gx + 3 * gs / 8, IC - gs / 2 + 1 * gs / 8
    body.append(f"<rect x='{wx:.0f}' y='{wy:.0f}' width='{5 * gs / 8:.0f}' "
                f"height='{5 * gs / 8:.0f}' fill='none' stroke='{ACC}' stroke-width='2'/>")
    tiny, _ = gridsq(486, 22, 6, OCH, ((3, 2, .6),), yc=LOSS_Y - 12, lw=1)
    body.append(loss_note(gx + 38, IC + 76, 516,
                          ("cross-entropy vs Gaussian cell target (unchanged)",), OCH, glyph=tiny))
    body.append(loss_note(tx, IC + 12, 790,
                          ("L2 on the WINDOWED decode", "train on exactly what flies"), ACC))
    return svgwrap(330, "".join(body))


# ---------------------------------------------------------------- exp 4
def exp4():
    b = [lanes(340)]
    g, x = frame_part(); b.append(g)
    b.append(harrow(x + 6, x + 30, IC)); x += 36
    g, x = conv_group(x, prev=(26, 76, FAINT)); b.append(g)
    b.append(harrow(x + 8, x + 34, IC)); x += 38
    dil_x = x
    g, w = slab(x, 17, 21, ACC); b.append(g)
    b.append(cap(dil_x + 19, IC + 48, "dilated conv 3×3, d2",
                 "~95 px context per cell", name_color=ACC))
    x += w + 40
    b.append(harrow(dil_x + w + 6, x - 6, IC))
    gx = x
    g, gs = gridsq(gx, 96, 8, ACC)
    b.append(g)
    dots = []
    for i in range(8):
        for j in range(8):
            ox = ((i * 37 + j * 53) % 9 - 4) * 0.9
            oy = ((i * 61 + j * 29) % 9 - 4) * 0.9
            dots.append((i, j, gx + (j + 0.5) * gs / 8 + ox,
                         IC - gs / 2 + (i + 0.5) * gs / 8 + oy))
    for (_, _, cx, cy) in dots:
        b.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='1.3' fill='{ACC}'/>")
    b.append(cap(gx + gs / 2, IC + 72, "64 per-patch coordinates",
                 "1×1 conv + σ (8×8×2)", name_color=ACC))
    tx = 812  # shared right anchor
    # committee decode: every vote line starts at an actual per-patch dot
    for (i, j, cx, cy) in dots:
        if (i + j) % 2:
            continue
        b.append(f"<line x1='{cx:.1f}' y1='{cy:.1f}' x2='{tx}' y2='{IC}' "
                 f"stroke='{ACC}' stroke-width='0.55' opacity='0.4'/>")
    b.append(crosspt(tx, IC, ACC))
    b.append(cap((gx + gs + tx) / 2, IC - 54, "mean of 64 answers",
                 "one committee answer from 64 votes", name_color=ACC))
    b.append(output_part(828))
    # confidence branch, anchored under the dilated-context stage
    cx0 = dil_x + 19
    ex = tx + 46
    b.append(leader(cx0, IC + 66, cx0, 216, FAINT))
    b.append(gauge(cx0, 230, MUT))
    b.append(txt(cx0, 248, "confidence 0–1", 9, MUT))
    b.append(f"<path d='M {cx0 + 18},230 H {ex} V {IC + 26}' fill='none' "
             f"stroke='{FAINT}' stroke-width='0.8'/>")
    b.append(f"<path d='M {ex},{IC + 20} l -3,6 h 6 Z' fill='{FAINT}'/>")
    b.append(txt((cx0 + ex) / 2, 224, "may abstain instead of guessing", 8.5, FAINT,
                 style="font-style='italic'"))
    b.append(leader(gx + gs / 2, IC + 92, gx + gs / 2, LOSS_Y - 20, ACC))
    b.append(txt(gx + gs / 2, LOSS_Y - 12,
                 "smooth-L1: each patch vs its OWN true coordinate", 10, ACC, 600))
    b.append(txt(gx + gs / 2, LOSS_Y,
                 "crop center + rotated offset → 64× denser supervision", 9.5, ACC))
    return svgwrap(340, "".join(b))


# ---------------------------------------------------------------- exp 5
def exp5():
    b, gx, gs, tx, _ = field_chain(INK, INK, INK, "all",
                                   "soft-argmax over ALL cells",
                                   "answer = Σ probability · cell-center")
    body = [lanes(340)] + b
    px = 430
    for i, (s, n) in enumerate(((30, 8), (22, 4), (16, 2))):
        g, _ = gridsq(px, s, n, ACC, ((n - 3 if n > 2 else 0, n // 4, .5),), yc=LOSS_Y - 16, lw=1)
        body.append(g)
        if i < 2:
            body.append(f"<path d='M {px + s + 4},{LOSS_Y - 16} h 10' stroke='{ACC}' stroke-width='0.8'/>"
                        f"<path d='M {px + s + 17},{LOSS_Y - 16} l -5,-2.5 v 5 Z' fill='{ACC}'/>")
        px += s + 22
    body.append(leader(gx + 38, IC + 76, 448, LOSS_Y - 34, ACC))
    body.append(txt(px + 6, LOSS_Y - 20, "sum-pool the SAME field to 16², 8², 4² — Gaussian-CE at every scale",
                    10, ACC, 600, "start"))
    body.append(txt(px + 6, LOSS_Y - 8, "coarse cells collect 10–40 positives each and steer the fine field (+ L2 on decode, unchanged)",
                    9.5, ACC, 400, "start"))
    return svgwrap(340, "".join(body))


# ---------------------------------------------------------------- exp 6
def exp6():
    b, gx, gs, tx, _ = field_chain(INK, INK, INK, "all",
                                   "soft-argmax over ALL cells",
                                   "answer = Σ probability · cell-center")
    body = [lanes(340)] + b
    # rotation glyph: one square re-drawn at three concentric angles
    # (fainter = earlier epoch), circled by a single arc arrow that stays
    # clear of the squares.
    cx0, cy0, s = 100, LOSS_Y - 6, 20
    for i, ang in enumerate((0, 30, 60)):
        body.append(f"<rect x='{cx0 - s / 2}' y='{cy0 - s / 2}' width='{s}' height='{s}' "
                    f"fill='none' stroke='{ACC}' stroke-width='1' opacity='{0.3 + i * 0.3:.1f}' "
                    f"transform='rotate({ang} {cx0} {cy0})'/>")
    r = 24
    body.append(f"<path d='M {cx0 - r},{cy0} A {r} {r} 0 0 1 {cx0 + r},{cy0}' fill='none' "
                f"stroke='{ACC}' stroke-width='1.2'/>"
                f"<path d='M {cx0 + r},{cy0 + 7} l -3.5,-7 h 7 Z' fill='{ACC}'/>")
    body.append(leader(53, IC + 80, 78, LOSS_Y - 22, ACC))
    body.append(txt(184, LOSS_Y - 8, "crop rotations re-drawn EVERY epoch", 10, ACC, 600, "start"))
    body.append(txt(184, LOSS_Y + 4, "fresh views of the same places each pass, not one frozen tensor reused 8×",
                    9.5, ACC, 400, "start"))
    tiny, _ = gridsq(586, 22, 6, OCH, ((3, 2, .6),), yc=LOSS_Y - 12, lw=1)
    body.append(loss_note(gx + 38, IC + 76, 616,
                          ("Gaussian-CE + L2 (unchanged)",), OCH, glyph=tiny))
    return svgwrap(340, "".join(body))


SVGS = {1: exp1(), 2: exp2(), 3: exp3(), 4: exp4(), 5: exp5(), 6: exp6()}

# ARCHIVED (mission era, experiments 1-6 of THAT era). Figure agents read this
# file as a style reference and may run it; on 2026-10-31 one did, and the DB
# write below silently replaced the capture-holdout era's figures for
# experiments 1-4 with these mission-era drawings. The write now needs an
# explicit flag; without it the script only reports sizes.
if __name__ == "__main__":
    if "--write-db" in sys.argv:
        conn = connect()
        for exp_id, s in SVGS.items():
            conn.execute("UPDATE experiments SET arch_svg=? WHERE id=?", (s, exp_id))
            print(f"exp {exp_id}: {len(s)} bytes written")
        conn.commit()
        conn.close()
    else:
        for exp_id, s in SVGS.items():
            print(f"exp {exp_id}: {len(s)} bytes (dry run; --write-db to store)")
