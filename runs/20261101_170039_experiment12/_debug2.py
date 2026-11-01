import json
import re

d = json.load(open("runs/20261101_170039_experiment12/figure.json"))
svg = d["architecture_svg"]
for m in re.finditer(r"<(line|path)[^>]*>", svg):
    s = m.group(0)
    if "279" in s or "280" in s or "239" in s:
        print(s)
