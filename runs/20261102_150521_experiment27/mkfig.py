import json, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
s = open('ref20.svg').read()
s = s[:s.index("<path d='M 262,438 H 247")]
F = "font-family='Palatino,Georgia,serif'"
R = '#8c2f1f'; O = '#8a6a1e'


def t(x, y, txt, size=9.5, fill=R, w=400, anchor='start', it=False):
    st = " font-style='italic'" if it else ''
    return f"<text x='{x}' y='{y}' {F} font-size='{size}' fill='{fill}' font-weight='{w}' text-anchor='{anchor}'{st}>{txt}</text>"


def terrain(x, y, s, tint, trees):
    k = s / 44
    road = '#f6f4ea' if tint == '#e6e3d4' else '#e6e3d4'
    g = [f"<rect x='{x}' y='{y}' width='{s}' height='{s}' fill='{tint}' stroke='#6b6a60' stroke-width='1'/>",
         f"<path d='M{x+2} {y+27*k:.1f} L{x+s-2} {y+15*k:.1f}' stroke='{road}' stroke-width='2.2' fill='none'/>",
         f"<path d='M{x+19*k:.1f} {y+2} L{x+13*k:.1f} {y+s-2}' stroke='{road}' stroke-width='2' fill='none'/>",
         f"<rect x='{x+4*k:.1f}' y='{y+4*k:.1f}' width='{7*k:.1f}' height='{5*k:.1f}' fill='#d9d5c3'/>",
         f"<rect x='{x+30*k:.1f}' y='{y+4*k:.1f}' width='{6*k:.1f}' height='{7*k:.1f}' fill='#cfccbd'/>",
         f"<rect x='{x+4*k:.1f}' y='{y+33*k:.1f}' width='{8*k:.1f}' height='{5*k:.1f}' fill='#d9d5c3'/>",
         f"<rect x='{x+28*k:.1f}' y='{y+29*k:.1f}' width='{6*k:.1f}' height='{6*k:.1f}' fill='#cfccbd'/>"]
    for cx, cy, r in trees:
        g.append(f"<circle cx='{x+cx*k:.1f}' cy='{y+cy*k:.1f}' r='{r*k:.1f}' fill='#8a6a1e' opacity='.12'/>")
    return ''.join(g)


Y = 518; S = 44
out = []
out.append(terrain(30, Y, S, '#f6f4ea', [(36, 38, 5), (22, 20, 3)]))
out.append(t(52, Y+S+12, 'view, survey a', 8.5, '#6b6a60', it=True, anchor='middle'))
out.append(terrain(92, Y, S, '#e6e3d4', [(36, 38, 2), (10, 22, 4)]))
out.append(t(114, Y+S+12, 'same ground,', 8.5, '#6b6a60', it=True, anchor='middle'))
out.append(t(114, Y+S+23, 'survey b', 8.5, '#6b6a60', it=True, anchor='middle'))
out.append(f"<line x1='140' y1='{Y+22}' x2='151' y2='{Y+22}' stroke='#9b998c' stroke-width='1'/><path d='M 156,{Y+22} l -6,-3 v 6 Z' fill='#9b998c'/>")
c = 160
out.append(terrain(c, Y, S, '#f6f4ea', [(36, 38, 5), (22, 20, 3)]))
out.append(f"<rect x='{c+6}' y='{Y+20}' width='16' height='14' fill='#e6e3d4' stroke='{R}' stroke-width='1'/>")
out.append(f"<path d='M{c+7} {Y+26} L{c+21} {Y+23}' stroke='#f6f4ea' stroke-width='2.2' fill='none'/>")
out.append(f"<rect x='{c+27}' y='{Y+30}' width='14' height='11' fill='#e6e3d4' stroke='{R}' stroke-width='1'/>")
out.append(f"<circle cx='{c+36}' cy='{Y+38}' r='2' fill='#8a6a1e' opacity='.12'/>")
out.append(f"<rect x='{c+26}' y='{Y+5}' width='13' height='12' fill='{R}' fill-opacity='.08' stroke='{R}' stroke-width='1' stroke-dasharray='2 1.5'/>")
out.append(f"<path d='M{c+28} {Y+8} L{c+37} {Y+14}' stroke='{R}' stroke-width='.6' opacity='.5'/><path d='M{c+37} {Y+8} L{c+30} {Y+15}' stroke='{R}' stroke-width='.6' opacity='.5'/>")
out.append(t(182, Y+S+12, 'composite view', 8.5, R, it=True, anchor='middle'))
out.append(t(182, Y+S+23, 'label unchanged', 8.5, R, it=True, anchor='middle'))
out.append(f"<path d='M 182,{Y} V 504 H 64 V 392' fill='none' stroke='{R}' stroke-width='1' stroke-dasharray='2 4'/>")
tx = 226
for y, txt, sz, w, it in [(530, 'NEW — patchwork surveys', 10, 600, False),
                          (544, '60% of views: up to 3 rectangles, 12–48 px a side', 9.5, 400, False),
                          (557, '¾ refilled from another year at the same ground,', 9.5, 400, False),
                          (570, '¼ from unrelated ground (another crop in the batch)', 9.5, 400, False),
                          (585, 'recognise the place from what survives local change', 9.5, 400, True)]:
    out.append(t(tx, y, txt, sz, R, w, it=it))
gx = 572
out.append(f"<path d='M {gx},528 V 492' fill='none' stroke='{O}' stroke-width='1' stroke-dasharray='2 4'/>")
out.append(f"<path d='M {gx-12},540 A 12 12 0 0 1 {gx+12},540' fill='none' stroke='{O}' stroke-width='1.2'/><circle cx='{gx}' cy='540' r='1.4' fill='{O}'/>")
ox = gx + 22
for y, txt, sz, w, it in [(536, 'unchanged from exp 20', 10, 600, False),
                          (550, 'MixStyle · CosFace m=0.25 · trunk at 0.1× LR', 9.5, 400, False),
                          (563, 'weight averaging + BN recalibration', 9.5, 400, False),
                          (578, 'BN recal. sees whole, unedited frames', 9.5, 400, True)]:
    out.append(t(ox, y, txt, sz, O, w, it=it))
out.append(t(972, 612, 'only the training views changed · graph, size (31.7 MB), decode and gate identical to exp 20', 9, '#9b998c', anchor='end', it=True))
svg = s + ''.join(out) + '</svg>'
open('fig.svg', 'w').write(svg)
json.dump({'architecture_svg': svg}, open('figure.json', 'w'), ensure_ascii=False)
