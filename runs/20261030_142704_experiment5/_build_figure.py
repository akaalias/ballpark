import json
import sys
from pathlib import Path

REPO = Path("/Users/alexisrondeau/Workshop/low-light-geolocalization-autoresearch")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "archive"))
from arch_svg_reference import (
    INK, MUT, FAINT, ACC, OCH, FONT, IC, LOSS_Y,
    txt, cap, harrow, leader, slab, bar, fan, gridsq, BUMP, crosspt,
    converge, gauge, lanes, svgwrap, kernel_proj, frame_part, conf_branch,
    output_part, loss_note,
)

H = 340
b = [lanes(H)]

g, x = frame_part()
b.append(g)
b.append(harrow(x + 6, x + 28, IC))
x += 30

# ---- unchanged 4-stage conv encoder (3->32->64->128->160, 128px -> 8px) ----
sizes = [(42, 7), (31, 10), (22, 14), (16, 19)]
faces = []
x0 = x
for i, (s, d) in enumerate(sizes):
    sg, w = slab(x, s, d, INK)
    b.append(sg)
    faces.append((x, s))
    x += w + 11
x -= 11
b.append(kernel_proj(26, 76, faces[0][0], faces[0][1], FAINT))
for (x1, s1), (x2, s2) in zip(faces, faces[1:]):
    b.append(kernel_proj(x1, s1, x2, s2, MUT))
b.append(cap((x0 + x) / 2, IC + 54, "convolutional encoder", "Conv 3×3, stride 2 ×4 · BN + ReLU (unchanged)"))
b.append(txt(x0 + 21, IC - 32, "64²×32", 9, FAINT))
last4_x, last4_s = faces[-1][0] + (sizes[-1][0] + sizes[-1][1]) - sizes[-1][0], sizes[-1][0]
b.append(txt(x - 16, IC - 30, "8²×160", 9, FAINT))

# ---- CHANGED: two new depthwise-separable stride-2 stages, red ----
b.append(harrow(x + 6, x + 22, IC, color=ACC))
x += 24
new_sizes = [(11, 19), (7, 19)]
new_faces = []
prev_face = faces[-1]
x0b = x
for i, (s, d) in enumerate(new_sizes):
    sg, w = slab(x, s, d, ACC)
    b.append(sg)
    new_faces.append((x, s))
    x += w + 11
x -= 11
b.append(kernel_proj(prev_face[0], prev_face[1], new_faces[0][0], new_faces[0][1], ACC))
for (x1, s1), (x2, s2) in zip(new_faces, new_faces[1:]):
    b.append(kernel_proj(x1, s1, x2, s2, ACC))
b.append(cap((x0b + x) / 2, IC + 54, "+2 depthwise-separable stride-2 stages",
             "depthwise 3×3 g=160 + pointwise 1×1 · BN+ReLU ×2", color=ACC, name_color=ACC))
b.append(txt(x0b + 10, IC - 28, "4²×160", 9, ACC))
b.append(txt(x - 10, IC - 26, "2²×160", 9, ACC))
b.append(txt((x0b + x) / 2, IC - 86, "receptive field 31 → 63 → 127 px — whole frame, not a texture patch", 9.5, ACC,
             w=600, style="font-style='italic'"))

# ---- GAP (now over 2x2, mechanism unchanged) ----
b.append(harrow(x + 6, x + 22, IC))
x += 24
gap_x = x
g, w = bar(x, 56, ticks=10)
b.append(g)
b.append(cap(x + 3, IC + 42, "160-d", "GAP over 2×2 (was 8×8)"))
x += w + 10

# ---- FC fan -> 2,970-way cell logits ----
fx = x
g, w2 = bar(fx + 46, 78, ticks=13)
b.append(fan(fx, 56, fx + 46, 78))
b.append(g)
b.append(cap(fx + 50, IC + 68, "FC → 2,970 cell logits", "softmax"))
x = fx + 56

b.append(harrow(x + 4, x + 20, IC))
x += 22

# ---- probability field grid ----
gx = x
g, gs = gridsq(gx, 66, 8, INK, BUMP)
b.append(g)
b.append(cap(gx + gs / 2, IC + 48, "probability field", "55×54 map cells · 128 m each"))

tx = 812  # fixed right anchor
b.append(converge(gx, gs, tx, INK, cells="all"))
b.append(cap((gx + gs + tx) / 2 - 2, IC - 46, "3×3 pooled centroid",
             "balance point of winning cell + 8 neighbours"))

b.append(output_part(828))
b.append(conf_branch(gap_x, tx + 10))

# ---- training signal lane (unchanged from experiment 4, now ochre) ----
b.append(leader(102, IC + 22, 213, LOSS_Y - 14, OCH))
b.append(f"<circle cx='213' cy='{LOSS_Y - 14}' r='2' fill='{OCH}'/>")
tiny, _ = gridsq(214, 14, 2, OCH, ((1, 0, .5),), yc=LOSS_Y - 20, lw=1)
b.append(tiny)
b.append(txt(240, LOSS_Y - 18, "one training crop, re-photographed at random every step", 10, OCH, 600, "start"))
b.append(txt(240, LOSS_Y - 6, "brightness · contrast · gamma · colour balance · saturation · blur · sensor noise (unchanged from exp 4)", 9.5, OCH, 400, "start"))

b.append(txt(968, LOSS_Y + 14, "trunk gains whole-frame context; everything downstream of GAP is byte-identical", 9.5, FAINT,
             anchor="end", style="font-style='italic'"))

svg = svgwrap(H, "".join(b))

out_path = REPO / "runs/20261030_142704_experiment5/figure.json"
out_path.write_text(json.dumps({"architecture_svg": svg}))
print(len(svg), "bytes written to", out_path)
