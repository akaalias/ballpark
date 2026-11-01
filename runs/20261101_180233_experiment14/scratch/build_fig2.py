import json
import re

SRC = 'runs/20261101_180233_experiment14/scratch/exp13.svg'
OUT = 'runs/20261101_180233_experiment14/figure.json'

ACC = '#8c2f1f'
MUT = '#6b6a60'

s = open(SRC).read()

# 1. neutralize experiment 13's own change (blocks 9-10 / lift 96->56 / the
# 112-d concat) back to ink: it is background, unchanged fact for exp 14.
s = s.replace(ACC, '#111111')

# 2. drop the "NEW" wording now that it's background, not this experiment
s = s.replace('NEW feat', 'feat')
s = s.replace('NEW: blocks', 'blocks')
s = s.replace('NEW lift', 'lift')
s = s.replace(
    "blocks 9–10 join the fine-tune for the first time — added to ImageNet, not overwritten",
    "blocks 9–10 fine-tune alongside the rest (added in experiment 13)",
)

# 3. recolor the 3 inter-block trunk junctions per lane (where blocks 1, 3, 6
# sit, collapsed into the 4-slab trunk glyph) red -- this IS where MixStyle
# runs. Each anchor below is unique in the document (one per lane).
anchors = [
    "x='177.2' y='85.0'", "x='214.6' y='86.9'", "x='248.6' y='88.5'",
    "x='177.2' y='293.0'", "x='214.6' y='294.9'", "x='248.6' y='296.5'",
]
mix_dots = []  # (x, y) just above/at each junction, for the leader source
for a in anchors:
    i = s.find(a)
    assert i != -1, a
    end = i + 620
    block = s[i:end]
    assert block.count(MUT) == 6, (a, block.count(MUT))
    block = block.replace(MUT, ACC)
    s = s[:i] + block + s[end:]
    mx = float(re.search(r"x='([\d.]+)'", a).group(1))
    my = float(re.search(r"y='([\d.]+)'", a).group(1))
    mix_dots.append((mx + 3, my))

# 4. add the MixStyle training-only annotation in the one genuinely empty
# band of the figure: below lane B's belief field (ends ~y=390) and above
# the existing ochre notes (start ~y=542), left of the decode/output column
# (which stays above y=420).
extra = []
extra.append(f"<circle cx='216' cy='45' r='1.8' fill='{ACC}'/>")
extra.append(f"<path d='M 216,45 H 155 V 450 H 226' fill='none' stroke='{ACC}' "
             f"stroke-width='1' stroke-dasharray='2 4'/>")
extra.append(f"<path d='M 226,450 l 9,-3.5 v 7 Z' fill='{ACC}'/>")
for (mx, my) in mix_dots:
    extra.append(f"<circle cx='{mx:.1f}' cy='{my:.1f}' r='1.4' fill='{ACC}'/>")

FONT = "Palatino,Georgia,serif"


def t(x, y, text, size, weight=400, style=""):
    return (f"<text x='{x}' y='{y}' font-family='{FONT}' font-size='{size}' "
            f"fill='{ACC}' font-weight='{weight}' text-anchor='start' {style}>{text}</text>")


extra.append(t(240, 441, "MixStyle — cross-survey style mixing, train only", 10.5, 600))
extra.append(t(240, 454, "after trunk blocks 1, 3, 6 (both members, independent draws): each view's "
                         "per-channel feature mean/std is blended", 9.5))
extra.append(t(240, 466, "with a random batch partner's, λ ~ Beta(0.1,0.1) — partner usually a "
                         "different survey (batches are already survey-paired)", 9.5))
extra.append(t(240, 478, "spatial layout untouched — identity at eval / export, so the graph drawn "
                         "above and the ONNX are pixel-for-pixel unchanged", 9.5, style="font-style='italic'"))

s = s.replace("</svg>", "".join(extra) + "</svg>")

with open(OUT, "w") as f:
    json.dump({"architecture_svg": s}, f)
print(len(s))
