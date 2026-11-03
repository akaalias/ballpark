import json, re
R = '#8c2f1f'
s = open('ref27.svg').read()
head = s[:s.index('>') + 1]
body = s[len(head):s.rindex('</svg>')]

# 1. drop exp 27's patchwork training glyphs, texts and red leader
body = body[:31319] + body[35895:]

# 2. resize stage: replace the 224 grid (rect + lines + fan) with a larger red 288 grid
start = body.index("<rect x='166.0' y='282.0' width='56'")
end = body.index("<text x='133.0' y='289.0'")
x0, y0, w = 160.0, 277.0, 66.0
g = [f"<rect x='{x0}' y='{y0}' width='{w}' height='{w}' fill='{R}' fill-opacity='.04' stroke='{R}' stroke-width='1.2'/>"]
n = 18
for i in range(1, n):
    d = round(w * i / n, 2)
    g.append(f"<line x1='{x0+d}' y1='{y0}' x2='{x0+d}' y2='{y0+w}' stroke='{R}' stroke-width='0.45' opacity='0.35'/>")
    g.append(f"<line x1='{x0}' y1='{y0+d}' x2='{x0+w}' y2='{y0+d}' stroke='{R}' stroke-width='0.45' opacity='0.35'/>")
g.append(f"<line x1='148.0' y1='295.0' x2='{x0}' y2='{y0}' stroke='{R}' stroke-width='0.55' opacity='0.5'/>")
g.append(f"<line x1='148.0' y1='325.0' x2='{x0}' y2='{y0+w}' stroke='{R}' stroke-width='0.55' opacity='0.5'/>")
body = body[:start] + ''.join(g) + body[end:]
body = body.replace("<line x1='222.0' y1='310.0' x2='240.0'", "<line x1='226.0' y1='310.0' x2='240.0'")

def retext(old, new, color=None):
    global body
    m = re.search(r"<text([^>]*)>" + re.escape(old) + "</text>", body)
    assert m, old
    attrs = m.group(1)
    if color:
        attrs = re.sub(r"fill='[^']*'", f"fill='{color}'", attrs)
    body = body[:m.start()] + f"<text{attrs}>{new}</text>" + body[m.end():]

retext('224²×3', '288²×3', R)
body = body.replace("<text x='194.0' y='276.0'", "<text x='193.0' y='271.0'")
retext('resize to 224²', 'resize to 288²', R)
retext("0 params · ImageNet's scale", '0 params · was 224²', R)

for _ in range(2):
    retext('24×56²', '24×72²', R)
    retext('40×28²', '40×36²', R)
    retext('112×14²', '112×18²', R)
    retext('320×7²', '320×9²', R)
    retext('7 MBConv+SE stages · GAP of 112×14², 320×7²', '7 MBConv+SE stages · GAP of 112×18², 320×9²', R)

# 3. training lane: all unchanged from exp 20
retext('MixStyle · CosFace m=0.25 · trunk at 0.1× LR', 'MixStyle · CosFace m=0.25 · trunk at 0.1× LR')
retext('weight averaging + BN recalibration', 'paired three-survey batches · re-photographing aug.')
retext('BN recal. sees whole, unedited frames', '8 epochs · weight averaging from ep. 5 + BN recal.')
body = body.replace("y='578' font-family='Palatino,Georgia,serif' font-size='9.5' fill='#8a6a1e' font-weight='400' text-anchor='start' font-style='italic'>8 epochs",
                    "y='576' font-family='Palatino,Georgia,serif' font-size='9.5' fill='#8a6a1e' font-weight='400' text-anchor='start'>8 epochs")
retext('only the training views changed · graph, size (31.7 MB), decode and gate identical to exp 20',
       'only the in-graph enlargement changed (224² → 288²) · parameters, size (31.7 MB), decode and gate identical to exp 20')

svg = head + body + '</svg>'
open('fig.svg', 'w').write(svg)
json.dump({'architecture_svg': svg}, open('figure.json', 'w'), ensure_ascii=False)
print('ok', len(svg))
