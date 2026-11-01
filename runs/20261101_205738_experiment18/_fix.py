import json
d = 'runs/20261101_205738_experiment18/'
s = open(d + 'fig17.svg').read()
R = [("<text x='8.0' y='612.0'", "<text x='8.0' y='538.0'"),
     ("M 262,438 H 247 V 536 H 270", "M 262,438 H 247 V 562 H 270"),
     ("M 270,536 A 12 12 0 0 1 294,536", "M 270,562 A 12 12 0 0 1 294,562"),
     ("cx='282' cy='536'", "cx='282' cy='562'"),
     ("<text x='304.0' y='566.0'", "<text x='304.0' y='592.0'"),
     ("<text x='304.0' y='547.0'", "<text x='304.0' y='573.0'"),
     ("<text x='304.0' y='534.0'", "<text x='304.0' y='560.0'")]
for a, b in R:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(d + 'fig18.svg', 'w').write(s)
json.dump({'architecture_svg': s}, open(d + 'figure.json', 'w'))
