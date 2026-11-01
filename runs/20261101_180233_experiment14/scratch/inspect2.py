import re
s = open('runs/20261101_180233_experiment14/scratch/exp13.svg').read()

for coord in ["177.2", "214.6", "248.6"]:
    idxs = [m.start() for m in re.finditer(re.escape(f"x='{coord}'"), s)]
    print(coord, idxs)
