"""assets/tiles/versions/: each of the overview's eight example tiles as it
appears in each of the three TRAINING photographs (Brandenburg DOP20, Berlin
TrueDOP 2022 and 2023), and the "versions" list in assets/tiles/tiles.json.

The model learns a place from all three at once and is tested on a fourth
(TrueDOP 2024) it never sees, which is deliberately not shown here: this
grid is the training data. A tile's lat/lon in tiles.json is its centre;
the crop is the 128 m square around it, which reproduces the original
single-photograph tiles (correlation 0.99 against bb_dop20). Needs data/
(local only); CI renders from the committed images.

    .venv/bin/python infra/make_tile_versions.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.warp import transform

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline.common import buckets, load_meta  # noqa: E402

TILES = REPO / "assets" / "tiles" / "tiles.json"
OUT = REPO / "assets" / "tiles" / "versions"
LABEL = {"bb_dop20": "Brandenburg DOP20", "truedop_2022": "TrueDOP 2022",
         "truedop_2023": "TrueDOP 2023"}


def main():
    Image.MAX_IMAGE_PIXELS = None
    meta = load_meta("berlin")
    train = list(buckets(meta, "train"))          # never the eval capture
    assert set(train) == set(LABEL), train
    d = json.loads(TILES.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rasters = {c: np.asarray(Image.open(REPO / "data" / "berlin" / "relight" / f"{c}.png").convert("RGB"))
               for c in train}
    for t in d["tiles"]:
        xs, ys = transform("EPSG:4326", f"EPSG:{meta['epsg']}", [t["lon"]], [t["lat"]])
        x0 = int(round(xs[0] - meta["origin_xy"][0])) - 64
        y0 = int(round(meta["origin_xy"][1] - ys[0])) - 64
        t["versions"] = []
        for c in ("bb_dop20", "truedop_2022", "truedop_2023"):
            name = f"{Path(t['file']).stem}_{c}.jpg"
            crop = Image.fromarray(rasters[c][y0:y0 + 128, x0:x0 + 128])
            crop.resize((256, 256), Image.LANCZOS).save(OUT / name, quality=84, optimize=True)
            t["versions"].append({"file": f"versions/{name}", "label": LABEL[c]})
    TILES.write_text(json.dumps(d))
    print(f"wrote {3 * len(d['tiles'])} crops to {OUT}")


if __name__ == "__main__":
    main()
