s = open('runs/20261101_180233_experiment14/scratch/exp13.svg').read()
import re
idxs = [m.start() for m in re.finditer('8c2f1f', s)]
print(len(idxs))
# print ranges of text between first and last occurrence region, find where NEW word appears
for m in re.finditer('NEW', s):
    print(m.start(), s[max(0,m.start()-20):m.start()+40])
