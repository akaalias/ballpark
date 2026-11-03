"""assets/sim/berlin-wide.jpg: a display-only basemap wider than the model's box.

The replay map is drawn wider than Berlin's model box (6.9 x 6.8 km), so it
needs imagery beyond the box. This fetches Berlin's TrueDOP 2024 (the same
photograph the camera sees, Geoportal Berlin, dl-de/zero-2-0) over the box
plus a margin, saved at 6 m/px, and fades it the way build_basemap() fades the
in-box map. Display only: nothing here reaches training, scoring or the
model.

The service offers ETRS89 / UTM 33N (EPSG:25833); the model's grid is
WGS 84 / UTM 33N (EPSG:32633). The two differ by under a metre here, a
quarter of one pixel of this image.

    .venv/bin/python -m sim.fetch_wide_basemap
"""

import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageEnhance

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "assets" / "sim" / "berlin-wide.jpg"
META_OUT = REPO / "assets" / "sim" / "berlin-wide.json"
WMS = "https://gdi.berlin.de/services/wms/truedop_2024"
ORIGIN = (385883.0, 5821316.0)       # UTM 33N of the box's top-left (data/berlin/meta.json)
SIZE = (6939, 6828)                  # box size in m (1 px = 1 m on the model's grid)
MARGIN = (4000, 2500)                # m added east/west and north/south
GSD = 4.0                            # m per pixel fetched
SAVE_GSD = 6.0                       # m per pixel saved: the map never shows finer
                                     # than ~7.7 m per screen pixel, and 4 m/px
                                     # cost 3.2 MB of first load (1.2 MB at 6 m/px)
TILE = 1024                          # px per WMS request, to stay under server limits


def get_tile(minx, miny, maxx, maxy, w, h):
    q = urllib.parse.urlencode({
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": "truedop_2024", "STYLES": "", "CRS": "EPSG:25833",
        "BBOX": f"{minx},{miny},{maxx},{maxy}", "WIDTH": w, "HEIGHT": h,
        "FORMAT": "image/jpeg"})
    req = urllib.request.Request(f"{WMS}?{q}", headers={"User-Agent": "ballpark-basemap"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def main():
    x0 = ORIGIN[0] - MARGIN[0]
    y_top = ORIGIN[1] + MARGIN[1]
    w_m = SIZE[0] + 2 * MARGIN[0]
    h_m = SIZE[1] + 2 * MARGIN[1]
    W, H = round(w_m / GSD), round(h_m / GSD)
    img = Image.new("RGB", (W, H))
    for ty in range(0, H, TILE):
        for tx in range(0, W, TILE):
            tw, th = min(TILE, W - tx), min(TILE, H - ty)
            minx, maxx = x0 + tx * GSD, x0 + (tx + tw) * GSD
            maxy, miny = y_top - ty * GSD, y_top - (ty + th) * GSD
            img.paste(get_tile(minx, miny, maxx, maxy, tw, th), (tx, ty))
            print(f"tile {tx},{ty} ok", flush=True)
    # the same treatment as build_basemap(): faded so the overlays carry it
    img = ImageEnhance.Color(img).enhance(0.45)
    img = ImageEnhance.Brightness(img).enhance(1.12)
    img = Image.blend(img, Image.new("RGB", img.size, (255, 255, 248)), 0.22)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img = img.resize((round(w_m / SAVE_GSD), round(h_m / SAVE_GSD)), Image.LANCZOS)
    img.save(OUT, quality=72, optimize=True, progressive=True)
    # where it sits in the replay's coordinates (model-grid px == m, origin at
    # the box's top-left), so the page can place it without re-deriving
    META_OUT.write_text(json.dumps({"x": -MARGIN[0], "y": -MARGIN[1],
                                    "w": w_m, "h": h_m, "gsd_m": SAVE_GSD,
                                    "source": "Geoportal Berlin TrueDOP 2024, dl-de/zero-2-0"}))
    print(f"wrote {OUT} ({img.size[0]}x{img.size[1]}, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
