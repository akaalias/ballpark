"""FROZEN (see /FROZEN) — synthetic low-light relighting, §4 step 3.

v0 model, ported in spirit from the human's earlier HTML/canvas ambient
simulator: split the daytime reference into
  (a) reflected ambient light — the whole scene, scaled by the bucket's
      natural ambient level and cooled toward moonlight at night, and
  (b) active artificial lighting — a heuristic built-up-area light map
      (bright, low-vegetation pixels, thinned to discrete lamp-like points)
      that stays lit regardless of ambient level, rendered as warm glow,
then apply a starlight-sensor response: auto-gain toward a target exposure,
shot + read noise growing with gain, and clipping.

All spatial parameters are in METERS and scaled by the area's GSD from
meta.json, so the same physics applies at 0.2 m DOP and 10 m Sentinel data.

NOTE per CLAUDE.md §4: this frozen file defines the EVAL data forever, but the
relighting *method* is explicitly a research target — the loop may build a
better trainable relighting for TRAINING data in model/, it just can't change
what the eval set looks like.

Usage: python -m pipeline.relight --area berlin
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageFilter

from pipeline.common import (DATA_DIR, LIGHTING_BUCKETS, area_dir, buckets,
                             load_meta, stable_hash)

# Scene model constants (bbox-independent; spatial units in meters).
AMBIENT_DAY_COLOR = np.array([1.00, 1.00, 1.00], dtype=np.float32)
AMBIENT_NIGHT_COLOR = np.array([0.80, 0.92, 1.12], dtype=np.float32)  # moonlight
LAMP_COLOR = np.array([1.00, 0.72, 0.42], dtype=np.float32)           # warm lamps
# ~1 lamp per 2500 m^2 of built-up area (streetlights + lit windows); low
# enough that 16 m glow pools stay mostly discrete instead of merging into
# saturated lakes.
LAMP_DENSITY_PER_M2 = 0.0004
GLOW_SIGMA_M = 16.0            # light-pool radius on the ground
TARGET_MEAN = 0.35             # auto-exposure target at full daylight (linear)
# Real auto-exposure deliberately under-exposes as light falls (protecting
# highlights / limiting gain): the effective target scales with ambient, so
# dusk renders visibly dimmer than noon even before the gain cap binds.
AE_FLOOR, AE_SLOPE, AE_GAMMA = 0.25, 0.75, 0.7
# Golden-hour warmth: strongest when the sun is low (mid-low ambient),
# negligible at noon; below that the ambient blend hands over to blue hour
# and then cool moonlight (AMBIENT_NIGHT_COLOR).
GOLDEN_TINT = np.array([0.12, -0.02, -0.18], dtype=np.float32)
GOLDEN_MU, GOLDEN_SIGMA = 0.5, 0.22
MAX_GAIN = 32.0
READ_NOISE = 0.012
SHOT_NOISE_K = 0.030


def artificial_light_map(ref: np.ndarray, gsd: float, seed: int) -> np.ndarray:
    """Heuristic map of active artificial lighting, in [0,1].

    Bright + non-vegetated daytime pixels are treated as built-up and likely
    to emit/reflect artificial light at night; thinned with a seeded random
    mask (density per m^2, so lamp spacing is GSD-independent) so lights read
    as discrete sources rather than uniform glow.
    """
    lin = (ref.astype(np.float32) / 255.0) ** 2.2
    lum = lin.mean(axis=2)
    greenness = lin[..., 1] - 0.5 * (lin[..., 0] + lin[..., 2])
    built = np.clip((lum - 0.06) * 6.0, 0, 1) * np.clip(1.0 - greenness * 25.0, 0, 1)
    rng = np.random.default_rng(seed)
    p_lamp = min(1.0, LAMP_DENSITY_PER_M2 * gsd * gsd)
    lamps = built * (rng.random(built.shape, dtype=np.float32) < p_lamp)
    img = Image.fromarray((lamps * 255).astype(np.uint8))
    sigma_px = max(GLOW_SIGMA_M / gsd, 0.8)
    glow = np.asarray(img.filter(ImageFilter.GaussianBlur(sigma_px)),
                      dtype=np.float32) / 255.0
    # sigma^2 normalization keeps both per-lamp peak brightness and total
    # emitted energy per m^2 GSD-independent (lamp count ~ gsd^2, blur area
    # ~ sigma_px^2).
    return np.clip(glow * 2.2 * (sigma_px / 1.6) ** 2, 0, 1)


def relight(ref: np.ndarray, ambient: float, gsd: float, seed: int) -> np.ndarray:
    """Render one lighting bucket. ref: HxWx3 uint8 daytime. Returns uint8."""
    rng = np.random.default_rng(seed)
    lin = (ref.astype(np.float32) / 255.0) ** 2.2

    night_w = np.float32(1.0 - ambient)
    warm = float(np.exp(-((ambient - GOLDEN_MU) ** 2) / (2 * GOLDEN_SIGMA ** 2)))
    amb_color = (AMBIENT_DAY_COLOR * ambient + AMBIENT_NIGHT_COLOR * night_w) \
        * (1.0 + warm * GOLDEN_TINT)
    scene = lin * np.float32(ambient) * amb_color[None, None, :]
    del lin

    # Artificial lights: constant emission, visually dominant only when dark.
    lamp = artificial_light_map(ref, gsd, seed) * (night_w ** 2)
    scene += lamp[..., None] * LAMP_COLOR[None, None, :] * np.float32(0.55)
    del lamp

    # Sensor: auto-gain toward an ambient-compensated target exposure, capped
    # (starlight sensors are good, not magic); noise grows with applied gain.
    ae_target = TARGET_MEAN * (AE_FLOOR + AE_SLOPE * float(ambient) ** AE_GAMMA)
    gain = float(np.clip(ae_target / max(float(scene.mean()), 1e-6), 1.0, MAX_GAIN))
    scene *= np.float32(gain)
    noise_scale = np.float32(np.sqrt(gain / MAX_GAIN))
    shot = rng.standard_normal(scene.shape, dtype=np.float32)
    shot *= np.sqrt(np.clip(scene, 0, 1) * np.float32(SHOT_NOISE_K)) * noise_scale
    scene += shot
    del shot
    read = rng.standard_normal(scene.shape, dtype=np.float32)
    read *= np.float32(READ_NOISE) * (np.float32(0.3) + noise_scale)
    scene += read
    del read
    np.clip(scene, 0.0, 1.0, out=scene)
    return (scene ** np.float32(1 / 2.2) * 255.0).astype(np.uint8)


def _read_capture(d: Path, capture: str) -> np.ndarray:
    """captures/<name>.tif (pipeline_data_version 3), falling back to the
    single reference.tif of areas fetched before captures existed."""
    path = d / "captures" / f"{capture}.tif"
    if not path.exists():
        path = d / "reference.tif"
    with rasterio.open(path) as src:
        return src.read().transpose(1, 2, 0)  # HxWx3


def relight_area(area: str, data_dir: Path | None = None):
    d = area_dir(area, data_dir)
    meta = load_meta(area, data_dir)
    out_dir = d / "relight"
    out_dir.mkdir(exist_ok=True)
    # One PNG per bucket = one capture (photograph) x one lighting condition.
    by_capture: dict[str, list[tuple[str, str]]] = {}
    for bname, b in buckets(meta).items():
        by_capture.setdefault(b["capture"], []).append((bname, b["lighting"]))
    for capture, blist in by_capture.items():
        ref = _read_capture(d, capture)
        for bname, lighting in blist:
            ambient = LIGHTING_BUCKETS[lighting]
            if lighting == "asis":
                # berlin-slim branch override (see CLAUDE.md "BRANCH OVERRIDE"):
                # pass the daytime photograph through unmodified — no ambient
                # dimming, no artificial-light sim, no sensor noise curve.
                img = ref.astype(np.uint8)
            else:
                seed = stable_hash(f"{area}:{capture}:{lighting}")
                img = relight(ref, ambient, meta["gsd_m"], seed)
            Image.fromarray(img).save(out_dir / f"{bname}.png")
            print(f"  {bname} (capture={capture}, ambient={ambient}) -> "
                  f"{out_dir / (bname + '.png')}", flush=True)
        del ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", required=True)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    n = len(buckets(load_meta(args.area, data_dir)))
    print(f"Relighting {args.area} into {n} bucket(s) "
          f"({len(LIGHTING_BUCKETS)} lighting condition(s) per capture)")
    relight_area(args.area, data_dir)


if __name__ == "__main__":
    main()
