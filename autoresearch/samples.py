"""Copy representative synthetic low-light sample crops into a run's
artifacts dir for the gallery (§7). Not frozen.

Usage: python -m autoresearch.samples --areas berlin,prignitz --out runs/X/samples
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.common import DATA_DIR, area_dir, buckets, load_meta

SAMPLE_PX = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for area in args.areas.split(","):
        for bucket in buckets(load_meta(area, data_dir)):
            img = np.asarray(Image.open(area_dir(area, data_dir) / "relight" / f"{bucket}.png"))
            cy, cx = img.shape[0] // 2, img.shape[1] // 2
            h = SAMPLE_PX // 2
            crop = img[cy - h:cy + h, cx - h:cx + h]
            Image.fromarray(crop).save(out / f"{area}_{bucket}.png")
    print(f"samples -> {out}")


if __name__ == "__main__":
    main()
