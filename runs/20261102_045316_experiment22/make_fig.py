import json, random, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
s = open('prev20.svg').read()
s = s[:s.index("<ellipse cx='672' cy='552'")]
s = s.replace("<text x='304.0' y='566.0' font-family='Palatino,Georgia,serif' font-size='9.5' fill='#8a6a1e' font-weight='400' text-anchor='start' font-style='italic'>8 epochs · MixStyle is identity at export</text>",
 "<text x='304.0' y='560.0' font-family='Palatino,Georgia,serif' font-size='9.5' fill='#8a6a1e' font-weight='400' text-anchor='start'>weight averaging epochs 5–8, BN stats re-measured</text>"
 "<text x='304.0' y='578.0' font-family='Palatino,Georgia,serif' font-size='9.5' fill='#8a6a1e' font-weight='400' text-anchor='start' font-style='italic'>8 epochs · MixStyle is identity at export</text>")
s = s.replace("only which weights fly changed · graph, size (31.7 MB), decode and gate identical to exp 19",
              "only where training views stand changed · graph, size (31.7 MB), decode and gate identical to exp 20")
R = '#8c2f1f'; G = '#9b998c'
o = []
cx0, cy0, st = 665, 544, 24
for i in (-1, 0, 1):
    for j in (-1, 0, 1):
        x = cx0 + i*st; y = cy0 + j*st
        o.append(f"<rect x='{x-12}' y='{y-12}' width='24' height='24' fill='none' stroke='{R}' stroke-width='.6' stroke-dasharray='2 2' opacity='.5'/>")
        o.append(f"<circle cx='{x}' cy='{y}' r='1.6' fill='{G}'/>")
random.seed(3)
for i in (-1, 0, 1):
    for j in (-1, 0, 1):
        for k in range(2):
            x = cx0 + i*st + random.uniform(-10, 10); y = cy0 + j*st + random.uniform(-10, 10)
            o.append(f"<path d='M {x-2:.1f},{y-2:.1f} L {x+2:.1f},{y+2:.1f} M {x-2:.1f},{y+2:.1f} L {x+2:.1f},{y-2:.1f}' stroke='{R}' stroke-width='.9'/>")
o.append(f"<line x1='{cx0}' y1='{cy0}' x2='673' y2='537' stroke='{R}' stroke-width='1'/>")
o.append(f"<circle cx='673' cy='537' r='2' fill='{R}'/>")
o.append(f"<rect x='655' y='519' width='36' height='36' fill='{R}' fill-opacity='.06' stroke='{R}' stroke-width='1.2'/>")
o.append(f"<text x='665' y='594' font-family='Palatino,Georgia,serif' font-size='9' fill='{R}' font-style='italic' text-anchor='middle'>● lattice · × jittered vantages</text>")
o.append(f"<path d='M 665,511 V 502 H 462 V 492' fill='none' stroke='{R}' stroke-width='1' stroke-dasharray='2 4'/>")

def T(y, t, w='400', it=False, fs='9.5'):
    st_ = " font-style='italic'" if it else ''
    return f"<text x='716' y='{y}' font-family='Palatino,Georgia,serif' font-size='{fs}' fill='{R}' font-weight='{w}' text-anchor='start'{st_}>{t}</text>"

o += [T(526, 'NEW — continuous vantages, ±12 px jitter', '600', fs='10'),
      T(540, 'fresh (dx, dy) ∈ [−12, 12]² px per use of a spot'),
      T(553, 'one shift shared by all three surveys, like heading'),
      T(566, 'tile and offset targets follow the shifted centre'),
      T(581, 'views tile the ground, not the fixed 24 m lattice', it=True)]
s += ''.join(o) + '</svg>'
open('fig.svg', 'w').write(s)
json.dump({'architecture_svg': s}, open('figure.json', 'w'), ensure_ascii=False)
