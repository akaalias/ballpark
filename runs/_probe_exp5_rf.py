"""Design-stage probe (experiment 5): receptive field and parameter budget of
the current trunk vs. one extended to whole-frame receptive field. Not part of
the record; no training, no eval capture touched."""
import json
import torch
import torch.nn as nn

from model.model import build_model

meta = json.load(open("data/berlin/meta.json"))
print("raster", meta["width"], "x", meta["height"])
m = build_model(meta)
n = sum(p.numel() for p in m.parameters())
trunk = sum(p.numel() for p in m.features.parameters())
print("params total", n, "trunk", trunk, "cls_head", m.cls_head.weight.numel())

rf, j = 1, 1
for _ in range(4):
    rf += 2 * j
    j *= 2
print("RF current (4 x 3x3 s2):", rf, "px")
for _ in range(2):
    rf += 2 * j
    j *= 2
print("RF with 2 more 3x3 s2 layers:", rf, "px")


def dwsep(c):
    return nn.Sequential(nn.Conv2d(c, c, 3, 2, 1, groups=c, bias=False), nn.BatchNorm2d(c),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c),
                         nn.ReLU(inplace=True))


extra = nn.Sequential(dwsep(160), dwsep(160))
ne = sum(p.numel() for p in extra.parameters())
print("extra params (2 dw-sep 160->160):", ne, "-> est onnx bytes",
      3079093 + ne * 4, "gate", 4 * 1024 * 1024)
x = torch.zeros(1, 3, 128, 128)
f = m.features(x)
print("feature map now", tuple(f.shape), "-> after extra", tuple(extra(f).shape))
