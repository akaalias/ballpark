import json, sys
src, out = sys.argv[1], sys.argv[2]
s = json.load(open(src))["architecture_svg"]
reps = [
    ("32×28²", "40×28²"),
    ("96×14²</text>", "112×14²</text>"),
    ("NEW — MobileNetV2 trunk A", "NEW — EfficientNet-B0 trunk A"),
    ("NEW — MobileNetV2 trunk B", "NEW — EfficientNet-B0 trunk B"),
    ("17 blocks · GAP of 96×14², 320×7²", "7 MBConv+SE stages · GAP of 112×14², 320×7²"),
    ("MixStyle after blocks 3, 6, 10 (same strides) · CosFace m=0.25 · trunk at 0.1× LR — unchanged from exp 15",
     "MixStyle after stages 2, 3, 4 (same strides) · CosFace m=0.25 · trunk at 0.1× LR — unchanged from exp 16"),
    ("only the trunk changed: ImageNet MobileNetV2 (BSD-3) · decode, gate, losses unchanged",
     "only the trunk changed: ImageNet EfficientNet-B0 (Apache-2.0) · decode, gate, losses unchanged"),
]
for a, b in reps:
    assert a in s, a
    s = s.replace(a, b)
json.dump({"architecture_svg": s}, open(out, "w"), ensure_ascii=False)
open(out.replace(".json", ".svg"), "w").write(s)
