import re, json
svg = json.load(open("runs/20261101_152928_experiment10/figure.json"))["architecture_svg"]
texts = re.findall(r"<text[^>]*x='([-\d.]+)' y='([-\d.]+)'[^>]*text-anchor='(\w+)'[^>]*>([^<]*)</text>", svg)
rows = sorted(((float(y), float(x), a, t) for x,y,a,t in texts))
for y,x,a,t in rows:
    print(f"{y:7.1f} {x:7.1f} {a:7s} {t}")
