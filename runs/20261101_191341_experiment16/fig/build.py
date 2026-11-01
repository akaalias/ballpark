import json, re, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
s = open('prev.svg').read()
R = '#8c2f1f'
# resize stage is unchanged this time: back to ink / caption colours
a = s.index("<rect x='118.0' y='295.0'"); b = s.index("<line x1='222.0' y1='310.0' x2='240.0'")
seg = s[a:b].replace(R, '#111111')
for t in ('128²<', '224²×3<', 'bilinear', '0 params'):
    seg = re.sub(r"fill='#111111'( font-weight='400' text-anchor='middle'( font-style='italic')?>%s)" % re.escape(t), r"fill='#6b6a60'\1", seg)
seg = seg.replace("NEW — resize to 224²", "resize to 224²")
s = s[:a] + seg + s[b:]

def redden(m):
    return m.group(0).replace("stroke='#111111'", "stroke='%s'" % R)
s = re.sub(r"<polygon points='(2[6-9]\d|3\d\d|4[0-4]\d)\.0,[^']*'[^>]*/>", redden, s)
s = re.sub(r"<rect x='(262|318|368|414)\.0' y='\d+\.0' width='(30|24|18|12)'[^>]*/>", redden, s)
s = re.sub(r"<line x1='[\d.]+' y1='[\d.]+' x2='(321\.5|371\.5|417\.5)'[^>]*/>", redden, s)
s = re.sub(r"<path d='M 377,\d+ V \d+ H 456 V [\d.]+'[^>]*/>", redden, s)
s = re.sub(r"<line x1='427\.0' y1='\d+\.0' x2='454\.0'[^>]*/>", redden, s)
for cy in ('170', '430'):
    s = s.replace("<circle cx='456' cy='%s' r='2.2' fill='#111111'/>" % cy, "<circle cx='456' cy='%s' r='2.2' fill='%s'/>" % (cy, R))
for o, n in (('16×112²', '24×56²'), ('24×28²', '32×28²'), ('48×14²', '96×14²'), ('96×7²', '320×7²')):
    s = s.replace('>%s<' % o, '>%s<' % n)
for m in 'AB':
    s = s.replace("trunk %s (pretrained), read at 2 depths" % m, "NEW — MobileNetV2 trunk %s" % m)
s = re.sub(r"fill='#111111' (font-weight='600' text-anchor='middle'>NEW — MobileNetV2)", r"fill='%s' \1" % R, s)
s = s.replace("11 MobileNetV3-S blocks · GAP of 14², 7²", "17 blocks · GAP of 96×14², 320×7²")
s = re.sub(r"fill='#6b6a60' (font-weight='400' text-anchor='middle'>17 blocks)", r"fill='%s' \1" % R, s)
s = s.replace("MixStyle after blocks 1, 3, 6 · CosFace margin m=0.25 · trunk at 0.1× LR — unchanged from exp 14",
              "MixStyle after blocks 3, 6, 10 (same strides) · CosFace m=0.25 · trunk at 0.1× LR — unchanged from exp 15")
a = s.index("<text x='304.0' y='566.0'"); b = s.index("</text>", a) + 7
s = s[:a] + ("<text x='304.0' y='566.0' font-family='Palatino,Georgia,serif' font-size='9.5' fill='#8a6a1e' "
             "font-weight='400' text-anchor='start' font-style='italic'>training views pass the same in-graph "
             "128→224 resize · bf16 autocast on CUDA only (infrastructure)</text>") + s[b:]
s = s.replace("trunk weights, decode, gate, losses unchanged · ONNX I/O still 1×3×128²",
              "only the trunk changed: ImageNet MobileNetV2 (BSD-3) · decode, gate, losses unchanged")
open('fig.svg', 'w').write(s)
json.dump({'architecture_svg': s}, open('../figure.json', 'w'), ensure_ascii=False)
