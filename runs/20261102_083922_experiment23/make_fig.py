import json, os
here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
# exp 20 (current champion) figure is the base; only the training lane changes.
s = open('../20261102_045316_experiment22/prev20.svg').read()
s = s[:s.index("<ellipse cx='672' cy='552'")]
s = s.replace("<text x='304.0' y='566.0' font-family='Palatino,Georgia,serif' font-size='9.5' fill='#8a6a1e' font-weight='400' text-anchor='start' font-style='italic'>8 epochs · MixStyle is identity at export</text>",
 "<text x='304.0' y='560.0' font-family='Palatino,Georgia,serif' font-size='9.5' fill='#8a6a1e' font-weight='400' text-anchor='start'>weight averaging epochs 5–8, BN stats re-measured</text>"
 "<text x='304.0' y='578.0' font-family='Palatino,Georgia,serif' font-size='9.5' fill='#8a6a1e' font-weight='400' text-anchor='start' font-style='italic'>8 epochs · MixStyle is identity at export</text>")
s = s.replace("only which weights fly changed · graph, size (31.7 MB), decode and gate identical to exp 19",
              "only the training loss and batching changed · graph, size (31.7 MB), decode and gate identical to exp 20")
R = '#8c2f1f'
o = []

def grid(x0, y0, shade, c=5, n=4, sw='.6'):
    for (i, j), a in shade.items():
        o.append(f"<rect x='{x0+i*c}' y='{y0+j*c}' width='{c}' height='{c}' fill='{R}' fill-opacity='{a}'/>")
    for k in range(n+1):
        o.append(f"<line x1='{x0+k*c}' y1='{y0}' x2='{x0+k*c}' y2='{y0+n*c}' stroke='{R}' stroke-width='{sw}'/>")
        o.append(f"<line x1='{x0}' y1='{y0+k*c}' x2='{x0+n*c}' y2='{y0+k*c}' stroke='{R}' stroke-width='{sw}'/>")

tops = 512
xs = (604, 630, 656)
shades = [{(2, 1): .12, (1, 1): .05},
          {(2, 1): .12, (2, 2): .05},
          {(2, 1): .06, (0, 3): .07, (3, 0): .05, (1, 2): .04}]
for x, sh in zip(xs, shades):
    grid(x, tops, sh)
mx, my = 630, 556
grid(mx, my, {(2, 1): .12, (1, 1): .03, (2, 2): .03}, sw='.9')
# converge lines: each survey's peak cell -> mixture's peak cell
tx, ty = mx + 12.5, my
for x in xs:
    o.append(f"<line x1='{x+12.5}' y1='{tops+20}' x2='{tx}' y2='{ty}' stroke='{R}' stroke-width='.8' stroke-dasharray='1.5 1.5'/>")
o.append(f"<text x='640' y='594' font-family='Palatino,Georgia,serif' font-size='9' fill='{R}' font-style='italic' text-anchor='middle'>p₁ p₂ p₃ of one place → mixture p̄</text>")
# leader from the glyph up to the cosine-tile-match belief of trunk B
o.append(f"<path d='M 640,508 V 500 H 538 V 492' fill='none' stroke='{R}' stroke-width='1' stroke-dasharray='2 4'/>")

def T(y, t, w='400', it=False, fs='9.5'):
    st_ = " font-style='italic'" if it else ''
    return f"<text x='716' y='{y}' font-family='Palatino,Georgia,serif' font-size='{fs}' fill='{R}' font-weight='{w}' text-anchor='start'{st_}>{t}</text>"

o += [T(522, 'NEW — same place, same belief (JSD, λ = 6)', '600', fs='10'),
      T(536, 'batches of 21 places × all 3 surveys (63 views)'),
      T(549, 'per member: tile-belief maps of one place in the'),
      T(562, "three surveys pulled to their mean, ⅓ Σ KL(p<tspan font-size='7' dy='2'>k</tspan><tspan dy='-2'> ‖ p̄)</tspan>"),
      T(577, 'deployed margin-free softmax · no cross-member term', it=True)]
s += ''.join(o) + '</svg>'
open('fig.svg', 'w').write(s)
json.dump({'architecture_svg': s}, open('figure.json', 'w'), ensure_ascii=False)
