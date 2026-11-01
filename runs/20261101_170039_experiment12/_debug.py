import json
import sys

sys.path.insert(0, ".")
from autoresearch import figcheck as fc

d = json.load(open("runs/20261101_170039_experiment12/figure.json"))
svg = d["architecture_svg"]
boxes = fc._text_boxes(svg)
chains = fc._polylines(svg)
want = [
    "both must independently back it, or ",
    "2970 cells, scaled cosine",
    "1×1 lift",
    "48→112ch, BN+ReLU, trainable",
    "member B trunk",
    "averaged belief",
    "mean of both members",
]
for b in boxes:
    if any(b[4].startswith(w[:18]) for w in want):
        print("BOX", b)
        for chain in chains:
            clipped = sum(fc._clip_len((p1[0], p1[1], p2[0], p2[1]), b) for p1, p2 in zip(chain, chain[1:]))
            if clipped > 12:
                tip = (fc._inside(chain[0], b) or fc._inside(chain[-1], b)) and clipped <= 16.0
                if not tip:
                    print("   crossed by chain start=", chain[0], "end=", chain[-1], "clipped=", round(clipped, 1))
