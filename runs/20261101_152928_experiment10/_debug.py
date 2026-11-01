import json, sys
sys.path.insert(0, "autoresearch")
import figcheck as fc

svg = json.load(open("runs/20261101_152928_experiment10/figure.json"))["architecture_svg"]
boxes = fc._text_boxes(svg)
byname = {b[4]: b for b in boxes}
for name in ["2 conv+upsample, 112", "MSE", "3 training", "once per run"]:
    for b in boxes:
        if b[4].startswith(name):
            print(name, "->", b)

print("---overlaps---")
for i in range(len(boxes)):
    for j in range(i+1, len(boxes)):
        a, b = boxes[i], boxes[j]
        ox = min(a[2], b[2]) - max(a[0], b[0])
        oy = min(a[3], b[3]) - max(a[1], b[1])
        if ox > 3.0 and oy > 3.0:
            print("OVERLAP", a[4], "||", b[4], a, b)

print("---line crosses---")
chains = fc._polylines(svg)
for chain in chains:
    for box in boxes:
        clipped = sum(fc._clip_len((p1[0], p1[1], p2[0], p2[1]), box) for p1, p2 in zip(chain, chain[1:]))
        if clipped <= 12.0:
            continue
        tip = ((fc._inside(chain[0], box) or fc._inside(chain[-1], box)) and clipped <= 16.0)
        if tip:
            continue
        print("CROSS", box[4], "chain=", chain, "clipped=", clipped)
