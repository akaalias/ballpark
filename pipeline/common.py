"""FROZEN (see /FROZEN) — shared helpers for the data pipeline.

Everything here is bbox-parameterized; no area-specific constants allowed.
"""

import json
import zlib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# berlin-slim branch override (see CLAUDE.md "BRANCH OVERRIDE"): the six
# lighting buckets of §4 collapse to one pass-through entry — raw daytime
# imagery as fetched, no synthetic ambient/gain/noise simulation. relight.py
# treats this bucket as an identity crop, not a relighting transform.
LIGHTING_BUCKETS = {
    "asis": 1.00,
}

# Ground sample distance is per-area (meta.json "gsd_m") — set by the imagery
# source registry (pipeline/sources.yaml), 1 m/px for DOP-covered regions.
CROP_PX = 128  # model input crop size (128 m footprint at 1 m/px ≈ UAV @ 100 m AGL)

# --- captures: several photographs of the same box (capture-holdout era) ----
#
# 2026-09-28. The 0.040 Berlin champion, scored on ANY other photograph of the
# same ground warped onto the same grid — Berlin's open TrueDOP 2022/2023/2024
# or (as a local diagnostic only) Google's mosaic — scored 1.78-1.97: confident
# and wrong on ~90% of frames, median miss ~3.5 km, i.e. a random guess. Adding
# +20 to every pixel of its OWN training raster scored 1.453. It had memorised
# one photograph's pixel values, not the ground. The same-photograph eval could
# never see this, and the loop only fixes what its ruler measures.
#
# So an area may now carry several CAPTURES — independent photographs of the
# same bbox (different survey year, season, sun angle) — each fetched onto the
# same grid. `role: train` captures are what the model is shown; a `role: eval`
# capture is a photograph it has NEVER seen, and the §6 score is computed on
# that one. §4's split still holds out viewpoints; this adds the axis "the
# photograph is new" on top of "the view is new". Areas without a captures list
# keep the old single-photograph behaviour (eval on the training photograph).
#
# A BUCKET is one capture rendered under one lighting condition. With the
# lighting axis collapsed to one entry (berlin-slim) the bucket name is simply
# the capture name; with several it is "<capture>__<lighting>".
DEFAULT_CAPTURE = "asis"


def area_captures(area_cfg: dict | None) -> list[dict]:
    """Capture list for an area from areas.yaml, or the single default."""
    caps = (area_cfg or {}).get("captures")
    if not caps:
        return [{"name": DEFAULT_CAPTURE, "source": None, "role": "train"}]
    for c in caps:
        if c.get("role") not in ("train", "eval"):
            raise ValueError(f"capture {c.get('name')!r}: role must be train|eval")
    return caps


def buckets(meta: dict, role: str | None = None) -> dict[str, dict]:
    """Bucket name -> {capture, lighting, role} for one area.

    role: 'train' | 'eval' | None (all). If no capture has role 'eval' the
    eval set falls back to the train captures — the single-photograph case.
    """
    caps = meta.get("captures") or {DEFAULT_CAPTURE: {"role": "train"}}
    if role == "eval" and not any(c.get("role") == "eval" for c in caps.values()):
        role = "train"
    out = {}
    for cname, c in caps.items():
        if role and c.get("role", "train") != role:
            continue
        for lname in LIGHTING_BUCKETS:
            bname = cname if len(LIGHTING_BUCKETS) == 1 else f"{cname}__{lname}"
            out[bname] = {"capture": cname, "lighting": lname,
                          "role": c.get("role", "train")}
    return out


def stable_hash(s: str) -> int:
    """Deterministic across runs/machines (unlike Python's hash())."""
    return zlib.crc32(s.encode("utf-8"))


def load_areas(path: Path | None = None) -> dict:
    p = path or (REPO_ROOT / "areas.yaml")
    with open(p) as f:
        return yaml.safe_load(f)["areas"]


def parse_bbox(s: str) -> list[float]:
    """Parse 'west,south,east,north' degrees."""
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be west,south,east,north")
    w, s_, e, n = parts
    if not (w < e and s_ < n and -180 <= w <= 180 and -90 <= s_ <= 90):
        raise ValueError(f"invalid bbox: {parts}")
    return parts


def utm_epsg_for(bbox: list[float]) -> int:
    """UTM zone EPSG for the bbox center — meter-true local grid for any bbox."""
    lon = (bbox[0] + bbox[2]) / 2.0
    lat = (bbox[1] + bbox[3]) / 2.0
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def area_dir(area: str, data_dir: Path | None = None) -> Path:
    return (data_dir or DATA_DIR) / area


def load_meta(area: str, data_dir: Path | None = None) -> dict:
    with open(area_dir(area, data_dir) / "meta.json") as f:
        return json.load(f)


def save_meta(area: str, meta: dict, data_dir: Path | None = None) -> None:
    d = area_dir(area, data_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def px_to_lonlat(meta: dict, px: float, py: float) -> tuple[float, float]:
    """Pixel coords in the reference raster -> (lon, lat)."""
    from rasterio.warp import transform as rio_transform

    x0, y0 = meta["origin_xy"]  # UTM coords of raster origin (top-left)
    x = x0 + px * meta["gsd_m"]
    y = y0 - py * meta["gsd_m"]
    lons, lats = rio_transform(f"EPSG:{meta['epsg']}", "EPSG:4326", [x], [y])
    return lons[0], lats[0]
