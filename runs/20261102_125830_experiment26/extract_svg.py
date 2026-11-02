import json, pathlib
d = pathlib.Path(__file__).parent
(d / "figure.svg").write_text(json.loads((d / "figure.json").read_text())["architecture_svg"])
