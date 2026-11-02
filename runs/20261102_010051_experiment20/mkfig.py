import json
s = open('prev.svg').read()
s = s.replace('#8c2f1f', '#111111').replace('NEW — EfficientNet', 'EfficientNet')
F = "font-family='Palatino,Georgia,serif'"


def rep(old, new):
    global s
    assert old in s, old
    s = s.replace(old, new)


rep('MixStyle after stages 2, 3, 4 (same strides) · CosFace m=0.25 · trunk at 0.1× LR — unchanged from exp 16',
    'MixStyle · CosFace m=0.25 · trunk at 0.1× LR — unchanged')
rep('paired three-survey batches, re-photographing augmentation, offset loss; MixStyle is identity at export',
    'paired three-survey batches, re-photographing aug., offset loss')
rep('training views pass the same in-graph 128→224 resize · bf16 autocast on CUDA only (infrastructure)',
    '8 epochs · MixStyle is identity at export')
rep('only the trunk changed: ImageNet EfficientNet-B0 (Apache-2.0) · decode, gate, losses unchanged',
    'only which weights fly changed · graph, size (31.7 MB), decode and gate identical to exp 19')
R = '#8c2f1f'
CX, CY = 672, 552
g = []
g.append(f"<ellipse cx='{CX}' cy='{CY}' rx='46' ry='24' fill='none' stroke='{R}' stroke-width='.8' opacity='.35'/>")
g.append(f"<ellipse cx='{CX}' cy='{CY}' rx='28' ry='14' fill='none' stroke='{R}' stroke-width='.8' opacity='.5'/>")
pts = [(632, 538), (646, 566), (660, 537), (676, 568), (690, 539), (702, 563), (712, 549)]
g.append("<polyline points='" + ' '.join(f'{x},{y}' for x, y in pts) +
         f"' fill='none' stroke='{R}' stroke-width='.8' stroke-dasharray='2 2'/>")
for x, y in pts:
    g.append(f"<line x1='{x}' y1='{y}' x2='{CX}' y2='{CY}' stroke='{R}' stroke-width='.5' opacity='.45'/>")
for x, y in pts:
    g.append(f"<circle cx='{x}' cy='{y}' r='1.8' fill='{R}' opacity='.6'/>")
g.append(f"<circle cx='{CX}' cy='{CY}' r='4' fill='none' stroke='{R}' stroke-width='1.4'/>"
         f"<circle cx='{CX}' cy='{CY}' r='1.4' fill='{R}'/>")
g.append(f"<text x='{CX}' y='592' {F} font-size='9' fill='{R}' font-style='italic' text-anchor='middle'>θ̄ = mean of visited θ</text>")
g.append(f"<path d='M {CX},526 V 505 H 462 V 492' fill='none' stroke='{R}' stroke-width='1' stroke-dasharray='2 4'/>")
tx = 732
rows = [(534, 10, 600, False, 'NEW — weight averaging (SWA/SWAD)'),
        (547, 9.5, 400, False, 'uniform mean, epochs 5–8, every 4 steps'),
        (560, 9.5, 400, False, '~3.4k snapshots, both members A and B'),
        (573, 9.5, 400, False, 'BN stats re-measured on 400 batches'),
        (588, 9.5, 400, True, 'the average flies, not the last step')]
for y, sz, w, it, t in rows:
    st = " font-style='italic'" if it else ''
    g.append(f"<text x='{tx}' y='{y}' {F} font-size='{sz}' fill='{R}' font-weight='{w}' text-anchor='start'{st}>{t}</text>")
s = s.replace('</svg>', ''.join(g) + '</svg>')
open('fig.svg', 'w').write(s)
json.dump({'architecture_svg': s}, open('figure.json', 'w'))
