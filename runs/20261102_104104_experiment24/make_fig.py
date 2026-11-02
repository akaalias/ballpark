import json, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
s = open('fig20.svg').read().strip()
R = '#8c2f1f'


def recolor(s, start, end):
    i = s.index(start); j = s.index(end, i)
    seg = (s[i:j].replace('#111111', R)
           .replace("fill='#00000010'", "fill='#8c2f1f10'")
           .replace("fill='#0000001c'", "fill='#8c2f1f1c'")
           .replace("fill='#00000006'", "fill='#8c2f1f06'"))
    return s[:i] + seg + s[j:]


s = recolor(s, "<polygon points='262.0,155.0", "<line x1='459.0' y1='170.0'")
s = recolor(s, "<polygon points='262.0,415.0", "<line x1='459.0' y1='430.0'")
for a, y in (('A', 214), ('B', 474)):
    old = (f"<text x='362.0' y='{y}.0' font-family='Palatino,Georgia,serif' font-size='10.5' fill='#111111' font-weight='600' text-anchor='middle'>EfficientNet-B0 trunk {a}</text>"
           f"<text x='362.0' y='{y+12}.0' font-family='Palatino,Georgia,serif' font-size='9.5' fill='#111111'")
    assert old in s, a
    new = (f"<text x='362.0' y='{y}.0' font-family='Palatino,Georgia,serif' font-size='10.5' fill='{R}' font-weight='600' text-anchor='middle'>EfficientNet-B0 trunk {a} · Noisy Student</text>"
           f"<text x='362.0' y='{y+24}.0' font-family='Palatino,Georgia,serif' font-size='9' fill='{R}' font-style='italic' text-anchor='middle'>NEW init weights · TF-same pad, eps 1e-3</text>"
           f"<text x='362.0' y='{y+12}.0' font-family='Palatino,Georgia,serif' font-size='9.5' fill='#6b6a60'")
    s = s.replace(old, new)

# exp 20's red SWA glyph is now an unchanged training element: drop it, fold into gold text
i = s.index("<ellipse cx='672' cy='552'")
s = s[:i]
old = "<text x='304.0' y='566.0' font-family='Palatino,Georgia,serif' font-size='9.5' fill='#8a6a1e' font-weight='400' text-anchor='start' font-style='italic'>8 epochs · MixStyle is identity at export</text>"
assert old in s
s = s.replace(old,
  "<text x='304.0' y='560.0' font-family='Palatino,Georgia,serif' font-size='9.5' fill='#8a6a1e' font-weight='400' text-anchor='start'>weight averaging epochs 5–8, BN stats re-measured</text>"
  "<text x='304.0' y='575.0' font-family='Palatino,Georgia,serif' font-size='9.5' fill='#8a6a1e' font-weight='400' text-anchor='start' font-style='italic'>8 epochs · MixStyle is identity at export</text>")
old = "only which weights fly changed · graph, size (31.7 MB), decode and gate identical to exp 19"
assert old in s
s = s.replace(old, "only the trunks' starting weights changed · graph, size (~31.7 MB), decode and gate identical to exp 20")

g = []
for dx, dy in [(12, -8), (6, -4), (0, 0)]:
    g.append(f"<rect x='{626+dx}' y='{540+dy}' width='36' height='28' fill='#f6f4ea' stroke='{R}' stroke-width='1.1'/>")
x, y = 626, 540
g.append(f"<path d='M{x+3} {y+20} L{x+33} {y+9}' stroke='{R}' stroke-width='1.6' opacity='.35' fill='none'/>")
g.append(f"<rect x='{x+6}' y='{y+4}' width='8' height='6' fill='{R}' opacity='.12'/>")
g.append(f"<rect x='{x+22}' y='{y+17}' width='8' height='7' fill='{R}' opacity='.12'/>")
for px, py in [(5, 14), (11, 23), (17, 6), (26, 12), (30, 24), (20, 20), (9, 3), (33, 4)]:
    g.append(f"<circle cx='{x+px}' cy='{y+py}' r='.8' fill='{R}' opacity='.6'/>")
g.append(f"<text x='650' y='584' font-family='Palatino,Georgia,serif' font-size='9' fill='{R}' font-style='italic' text-anchor='middle'>300M noised photos</text>")
g.append(f"<path d='M 650,531 V 512 H 362 V 503' fill='none' stroke='{R}' stroke-width='1' stroke-dasharray='2 4'/>")
lines = [("NEW — Noisy Student trunk weights", 10, 600, ''),
         ("tf_efficientnet_b0.ns_jft_in1k, Apache-2.0", 9.5, 400, ''),
         ("student trained under RandAugment, dropout, stoch. depth", 9.5, 400, ''),
         ("ported to torchvision · numerically checked vs timm", 9.5, 400, ''),
         ("init only — same graph, same size", 9.5, 400, 'italic')]
yy = 534
for t, fs, fw, st in lines:
    g.append(f"<text x='700' y='{yy}' font-family='Palatino,Georgia,serif' font-size='{fs}' fill='{R}' font-weight='{fw}' text-anchor='start'"
             + (f" font-style='{st}'" if st else '') + f">{t}</text>")
    yy += 13
s = s + ''.join(g) + '</svg>'
open('fig24.svg', 'w').write(s)
json.dump({'architecture_svg': s}, open('figure.json', 'w'))
