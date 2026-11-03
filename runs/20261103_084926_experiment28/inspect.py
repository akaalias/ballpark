import re
s = open('ref27.svg').read()
body = s[s.index('>') + 1:s.rindex('</svg>')]
for m in re.finditer(r'<(g|text)\b[^>]*>.*?</\1>|<\w+[^>]*/>', body, re.S):
    t = m.group(0)
    if (2000 < m.start() < 9000) or (29000 < m.start() < 36000):
        print(m.start(), t[:150])
