"""assets/social/card.jpg: the 1200x630 sharing card every page points at.

A band of real Berlin tiles above and below the question, with two lines
under it. The numbers on it come from the same places the overview reads:
the price from gallery.FLIGHT_COMPUTER_USD, the usable rate from the best
row of the re-scored history (today's ruler). Re-run after either changes,
then commit the image; CI has no raster, so it never regenerates the card.

The card used to be made once by hand (af4faec) and said "a $4 flight
computer" and "one city in 3.1 MB" long after both had stopped being true.

    .venv/bin/python infra/make_social_card.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autoresearch.gallery import FAIL, FLIGHT_COMPUTER_USD, load_history  # noqa: E402

OUT = REPO / "assets" / "social" / "card.jpg"
# A training photograph, never the held-out one: the card is a picture of the
# city, and the eval capture stays out of anything that is not evaluation.
RASTER = REPO / "data" / "berlin" / "relight" / "bb_dop20.png"
W, H = 1200, 630
TILE_W, TILE_H, COLS = 200, 97, 6
BAND = 2 * TILE_H + 1                      # two tile rows, 1 px gap
PAPER, INK, MUTED, ACCENT = (255, 255, 248), (25, 24, 20), (110, 108, 98), (140, 47, 31)
FONT = "/System/Library/Fonts/Palatino.ttc"   # regular 0, italic 1


def font(size, italic=False):
    return ImageFont.truetype(FONT, size, index=1 if italic else 0)


def tiles(img, n, rng):
    """n crops of whole 128 m frames, squashed to the band's tile shape."""
    h, w = img.shape[:2]
    out = []
    for _ in range(n):
        x, y = rng.integers(300, w - 300 - 128), rng.integers(300, h - 300 - 128)
        crop = Image.fromarray(img[y:y + 128, x:x + 128]).resize((TILE_W, TILE_W), Image.LANCZOS)
        top = (TILE_W - TILE_H) // 2
        out.append(crop.crop((0, top, TILE_W, top + TILE_H)))
    return out


def main():
    Image.MAX_IMAGE_PIXELS = None
    hist, _ = load_history()
    scored = [r for r in hist if r["kind"] != "holdout_check"
              and r["mission_score"] is not None and r["mission_score"] < FAIL]
    best = min(scored, key=lambda r: r["mission_score"])
    usable = 100 * best["usable_fix_rate"]

    img = np.asarray(Image.open(RASTER).convert("RGB"))
    rng = np.random.default_rng(7)
    card = Image.new("RGB", (W, H), "white")
    ts = tiles(img, 4 * COLS, rng)
    for band, y0 in enumerate((0, H - BAND)):
        for r in range(2):
            for c in range(COLS):
                card.paste(ts[band * 2 * COLS + r * COLS + c],
                           (c * (TILE_W + 1) - (1 if c else 0), y0 + r * (TILE_H + 1)))

    d = ImageDraw.Draw(card)
    d.rectangle((0, BAND, W, H - BAND), fill=PAPER)
    d.rectangle((0, BAND, W, BAND + 2), fill=ACCENT)
    d.rectangle((0, H - BAND - 3, W, H - BAND - 1), fill=ACCENT)
    mid = H // 2
    lines = [
        ("Can a UAV learn a city by heart?", font(60), INK, mid - 48),
        (f"no GPS · no map on board · a ${FLIGHT_COMPUTER_USD} flight computer",
         font(29), MUTED, mid + 22),
        (f"the map is the weights — usable on {usable:.0f}% of frames "
         f"from a photograph it never saw", font(26, italic=True), ACCENT, mid + 66),
    ]
    for text, f, col, y in lines:
        d.text((W / 2, y), text, font=f, fill=col, anchor="mm")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    card.save(OUT, quality=88, optimize=True)
    print(f"wrote {OUT} (best {best['mission_score']:.3f}, usable {usable:.1f}%, "
          f"${FLIGHT_COMPUTER_USD})")


if __name__ == "__main__":
    main()
