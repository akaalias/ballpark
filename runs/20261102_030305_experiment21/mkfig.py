import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
s = open('prev.svg').read()
# drop exp-20's red SWA additions (appended after the original elements)
cut = s.index("<ellipse cx='672' cy='552'")
s = s[:cut] + '</svg>'
F = "font-family='Palatino,Georgia,serif'"


def rep(old, new):
    global s
    assert old in s, old
    s = s.replace(old, new)


rep('8 epochs · MixStyle is identity at export', 'weights averaged over epochs 5–8 · MixStyle off at export')
rep('only which weights fly changed · graph, size (31.7 MB), decode and gate identical to exp 19',
    'only the training schedule changed · graph, size (31.7 MB), decode and gate identical to exp 20')
R, A = '#8c2f1f', '#8a6a1e'
g = []
x0, y0, cw, ch = 616, 538, 12, 14
for i in range(9):
    x = x0 + i * cw
    if i == 0:
        g.append(f"<rect x='{x}' y='{y0}' width='{cw}' height='{ch}' fill='{R}' fill-opacity='.12' stroke='{R}' stroke-width='1.2'/>")
    else:
        op = ".12" if i >= 5 else "0"
        g.append(f"<rect x='{x}' y='{y0}' width='{cw}' height='{ch}' fill='{A}' fill-opacity='{op}' stroke='{A}' stroke-width='.9'/>")
    col = R if i == 0 else A
    g.append(f"<text x='{x + cw / 2}' y='{y0 + ch + 11}' {F} font-size='8' fill='{col}' text-anchor='middle'>{i}</text>")
mid = x0 + 54
cx = x0 + cw / 2
g.append(f"<text x='{mid}' y='{y0 - 6}' {F} font-size='8.5' fill='{A}' font-style='italic' text-anchor='middle'>epochs</text>")
g.append(f"<text x='{mid}' y='{y0 + ch + 26}' {F} font-size='8.5' fill='{A}' font-style='italic' text-anchor='middle'>fine-tune 1–8 as before</text>")
g.append(f"<text x='{mid}' y='{y0 + ch + 38}' {F} font-size='8.5' fill='{R}' font-style='italic' text-anchor='middle'>0 = head-only probe</text>")
g.append(f"<path d='M {cx},{y0} V 505 H 462 V 492' fill='none' stroke='{R}' stroke-width='1' stroke-dasharray='2 4'/>")
tx = 736
rows = [(534, 10, 600, False, 'NEW — probe first, then fine-tune (LP-FT)'),
        (547, 9.5, 400, False, 'extra epoch 0: trunks A, B held still —'),
        (560, 9.5, 400, False, 'weights and BN stats stay ImageNet’s;'),
        (573, 9.5, 400, False, 'only lifts, prototypes, offset head train'),
        (588, 9.5, 400, True, 'the random head cannot distort borrowed eyes')]
for y, sz, w, it, t in rows:
    st = " font-style='italic'" if it else ''
    g.append(f"<text x='{tx}' y='{y}' {F} font-size='{sz}' fill='{R}' font-weight='{w}' text-anchor='start'{st}>{t}</text>")
s = s.replace('</svg>', ''.join(g) + '</svg>')
open('fig.svg', 'w').write(s)
json.dump({'architecture_svg': s}, open('figure.json', 'w'))
