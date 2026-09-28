"""FROZEN (see /FROZEN) — reference imagery fetch, §4 step 2.

Fetches open-licensed daytime reference imagery for an arbitrary bbox onto a
north-up grid in the bbox's local UTM zone at the registry's target GSD
(default 1 m/px — matched to a UAV camera footprint at ~100 m AGL).

Sources come from pipeline/sources.yaml (a data file, keeping this code
bbox-generic): regional open-data DOP orthophoto WMS services where coverage
exists (20-40 cm native, server-resampled to the target GSD), with global
Sentinel-2 L2A (10 m/px, credential-free AWS COGs via Earth Search STAC) as
fallback. No Google/Bing tiles anywhere, per spec.

CAPTURES (2026-09-28, see pipeline/common.py). An area may list several
captures — independent photographs of the same bbox (different survey year,
season, sun angle), each fetched from a NAMED registry source onto the SAME
grid, so a crop at (cx, cy) shows the same ground in every capture. They land
in captures/<name>.tif; a capture whose file already exists is reused, never
re-downloaded (the training raster must stay byte-identical across the era).
Without a captures list an area gets one capture from the auto-picked source,
exactly as before.

Usage:
  python -m pipeline.fetch --area berlin
  python -m pipeline.fetch --area berlin --captures truedop_2024 --force
  python -m pipeline.fetch --name mytown --bbox 9.10,48.70,9.20,48.76
"""

import argparse
import datetime
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from pipeline.common import (DATA_DIR, REPO_ROOT, area_captures, area_dir,
                             load_areas, parse_bbox, save_meta, utm_epsg_for)

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
# Fixed default acquisition window so Sentinel fetches are reproducible.
DEFAULT_DATERANGE = "2025-04-01T00:00:00Z/2025-09-30T23:59:59Z"
MAX_CLOUD = 8.0
TILE_PX = 2048           # target-grid tile size per WMS request
WMS_OVERSAMPLE = 1.15    # request slightly finer than target to protect detail
WMS_RETRIES = 3


def load_sources():
    with open(REPO_ROOT / "pipeline" / "sources.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg["target_gsd_m"], cfg["sources"]


def _covers(src, bbox):
    cov = src["coverage"]
    return (cov[0] <= bbox[0] and cov[1] <= bbox[1]
            and cov[2] >= bbox[2] and cov[3] >= bbox[3])


def pick_source(bbox, sources):
    """Finest auto-pickable source covering the bbox (named-only sources are
    skipped — they exist for explicit captures)."""
    candidates = [s for s in sources
                  if s.get("auto_pick", True) and _covers(s, bbox)]
    if not candidates:
        raise SystemExit(f"no imagery source covers bbox {bbox}")
    return min(candidates, key=lambda s: s["native_gsd_m"])


def named_source(name, bbox, sources):
    src = next((s for s in sources if s["name"] == name), None)
    if src is None:
        raise SystemExit(f"unknown imagery source {name!r} in areas.yaml captures")
    if not _covers(src, bbox):
        raise SystemExit(f"source {name!r} does not cover bbox {bbox}")
    return src


def make_target_grid(bbox, gsd):
    epsg = utm_epsg_for(bbox)
    dst_crs = f"EPSG:{epsg}"
    left, bottom, right, top = transform_bounds("EPSG:4326", dst_crs, *bbox)
    left, top = np.floor(left / gsd) * gsd, np.ceil(top / gsd) * gsd
    width = int(np.ceil((right - left) / gsd))
    height = int(np.ceil((top - bottom) / gsd))
    return epsg, dst_crs, from_origin(left, top, gsd, gsd), width, height, (left, top)


# --- WMS path -------------------------------------------------------------

def wms_getmap(src, bbox4326, px_w, px_h):
    """One GetMap request in EPSG:4326, version-aware axis order."""
    w, s, e, n = bbox4326
    v = src["wms_version"]
    params = {
        "SERVICE": "WMS", "VERSION": v, "REQUEST": "GetMap",
        "LAYERS": src["layer"], "STYLES": "",
        "WIDTH": str(px_w), "HEIGHT": str(px_h), "FORMAT": "image/png",
    }
    if v == "1.3.0":
        params["CRS"] = "EPSG:4326"
        params["BBOX"] = f"{s},{w},{n},{e}"   # 1.3.0: lat,lon order
    else:
        params["SRS"] = "EPSG:4326"
        params["BBOX"] = f"{w},{s},{e},{n}"
    url = src["url"] + ("&" if "?" in src["url"] else "?") + urllib.parse.urlencode(params)
    last_err = None
    for attempt in range(WMS_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                data = r.read()
            if not data.startswith(b"\x89PNG"):
                raise IOError(f"non-PNG response ({data[:80]!r})")
            return data
        except Exception as e:  # noqa: BLE001 — retry any transport error
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise IOError(f"WMS GetMap failed after {WMS_RETRIES} tries: {last_err}")


def fetch_wms(src, dst_crs, dst_transform, width, height, gsd):
    """Tile over the target grid; each tile: WMS 4326 image -> warp to UTM."""
    mosaic = np.zeros((3, height, width), dtype=np.uint8)
    n_tiles_x = int(np.ceil(width / TILE_PX))
    n_tiles_y = int(np.ceil(height / TILE_PX))
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            x0, y0 = tx * TILE_PX, ty * TILE_PX
            tw, th = min(TILE_PX, width - x0), min(TILE_PX, height - y0)
            # Tile bounds in target CRS, then in EPSG:4326.
            left = dst_transform.c + x0 * gsd
            top = dst_transform.f - y0 * gsd
            tile_bounds = (left, top - th * gsd, left + tw * gsd, top)
            b4326 = transform_bounds(dst_crs, "EPSG:4326", *tile_bounds)
            req_w = int(tw * WMS_OVERSAMPLE)
            req_h = int(th * WMS_OVERSAMPLE)
            png = wms_getmap(src, b4326, req_w, req_h)
            print(f"  tile {tx},{ty} ({tw}x{th}px) fetched", flush=True)
            # Wrap the PNG with its known 4326 georeferencing, warp into place.
            with MemoryFile(png) as mem, mem.open() as ds:
                arr = ds.read()
                if arr.shape[0] == 1:
                    arr = np.repeat(arr, 3, axis=0)
                t4326 = rasterio.transform.from_bounds(
                    b4326[0], b4326[1], b4326[2], b4326[3], ds.width, ds.height)
                profile = {"driver": "GTiff", "count": 3, "dtype": "uint8",
                           "width": ds.width, "height": ds.height,
                           "crs": "EPSG:4326", "transform": t4326}
                with MemoryFile() as gmem:
                    with gmem.open(**profile) as gds:
                        gds.write(arr[:3])
                    with gmem.open() as gds, WarpedVRT(
                            gds, crs=dst_crs,
                            transform=rasterio.transform.from_origin(left, top, gsd, gsd),
                            width=tw, height=th) as vrt:
                        mosaic[:, y0:y0 + th, x0:x0 + tw] = vrt.read()
    return mosaic, {"tiles": n_tiles_x * n_tiles_y}


# --- Sentinel-2 fallback path --------------------------------------------

def stac_search(bbox, daterange, limit=30):
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": daterange,
        "query": {"eo:cloud_cover": {"lt": MAX_CLOUD}},
        "limit": limit,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }
    req = urllib.request.Request(
        STAC_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["features"]


def fetch_sentinel(bbox, daterange, dst_crs, dst_transform, width, height):
    items = stac_search(bbox, daterange)
    if not items:
        raise SystemExit(f"No Sentinel-2 scenes for bbox {bbox} in {daterange}")
    mosaic = np.zeros((3, height, width), dtype=np.uint8)
    used = []
    for item in items:
        if not (mosaic == 0).any():
            break
        href = item["assets"]["visual"]["href"]
        print(f"  reading {item['id']} (cloud {item['properties']['eo:cloud_cover']:.2f}%)")
        try:
            with rasterio.open(href) as s2, WarpedVRT(
                    s2, crs=dst_crs, transform=dst_transform,
                    width=width, height=height) as vrt:
                arr = vrt.read()
        except rasterio.errors.RasterioIOError as e:
            print(f"    skipped (read error: {e})")
            continue
        empty = (mosaic == 0).all(axis=0)
        has_data = (arr != 0).any(axis=0)
        fill = empty & has_data
        if fill.any():
            mosaic[:, fill] = arr[:, fill]
            used.append(item["id"])
    return mosaic, {"stac_items": used, "daterange": daterange}


# --- entry point ----------------------------------------------------------

def fetch_capture(src, bbox, dst_crs, dst_transform, width, height, gsd,
                  daterange):
    if src["kind"] == "wms":
        return fetch_wms(src, dst_crs, dst_transform, width, height, gsd)
    if src["kind"] == "sentinel2_stac":
        return fetch_sentinel(bbox, daterange, dst_crs, dst_transform,
                              width, height)
    raise SystemExit(f"unknown source kind {src['kind']}")


def fetch_area(name: str, bbox: list[float], data_dir=None,
               daterange=DEFAULT_DATERANGE, captures=None, only=None,
               force=False):
    """Fetch every capture of an area onto one shared grid.

    captures: list of {name, source, role} (areas.yaml); None = the single
    default capture from the auto-picked source. only: restrict to these
    capture names. force: re-download captures whose .tif already exists.
    """
    target_gsd, sources = load_sources()
    captures = captures or area_captures(None)
    resolved = []
    for c in captures:
        src = (named_source(c["source"], bbox, sources) if c.get("source")
               else pick_source(bbox, sources))
        resolved.append((c, src))

    # One grid for all captures: the GSD is set by the coarsest source, so a
    # crop at (cx, cy) shows the same ground in every photograph.
    gsd = max([target_gsd] + [src["native_gsd_m"] for _, src in resolved])
    epsg, dst_crs, dst_transform, width, height, (left, top) = \
        make_target_grid(bbox, gsd)
    print(f"  gsd={gsd} m/px grid={width}x{height}px "
          f"captures={[c['name'] for c, _ in resolved]}")

    d = area_dir(name, data_dir)
    cap_dir = d / "captures"
    cap_dir.mkdir(parents=True, exist_ok=True)
    meta_path = d / "meta.json"
    old_caps = {}
    if meta_path.exists():
        with open(meta_path) as f:
            old_caps = json.load(f).get("captures") or {}

    cap_meta = {}
    for c, src in resolved:
        out = cap_dir / f"{c['name']}.tif"
        record = dict(old_caps.get(c["name"]) or {})
        if only and c["name"] not in only:
            print(f"  [{c['name']}] not requested, keeping as is")
        elif out.exists() and not force:
            print(f"  [{c['name']}] reusing existing {out}")
        else:
            print(f"  [{c['name']}] source={src['name']}")
            mosaic, extra = fetch_capture(src, bbox, dst_crs, dst_transform,
                                          width, height, gsd, daterange)
            coverage = float(((mosaic != 0).any(axis=0)).mean())
            if coverage < 0.995:
                print(f"WARNING: [{c['name']}] mosaic only {coverage*100:.1f}% covered")
            with rasterio.open(
                    out, "w", driver="GTiff", width=width, height=height,
                    count=3, dtype="uint8", crs=dst_crs,
                    transform=dst_transform, compress="lzw", tiled=True) as dst:
                dst.write(mosaic)
            record = {
                "source": src["name"], "attribution": src["attribution"],
                "coverage": coverage,
                "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **extra,
            }
            print(f"  wrote {out} ({coverage*100:.1f}% covered)")
        if out.exists():
            # Every capture must sit on THE grid — refuse a stray raster.
            with rasterio.open(out) as chk:
                if (chk.width, chk.height) != (width, height) or \
                        chk.transform != dst_transform:
                    raise SystemExit(f"{out} is not on the area grid "
                                     f"({chk.width}x{chk.height}); delete it "
                                     f"or re-run with --force")
        record.setdefault("source", src["name"])
        record.setdefault("attribution", src["attribution"])
        record["role"] = c["role"]
        cap_meta[c["name"]] = record

    first = next(iter(cap_meta.values()))
    save_meta(name, {
        "area": name,
        "pipeline_data_version": 3,
        "bbox": bbox,
        "epsg": epsg,
        "origin_xy": [left, top],
        "gsd_m": gsd,
        "width": width,
        "height": height,
        # Top-level source/attribution/coverage describe the FIRST capture
        # (compat with readers that predate captures); the full record is
        # under "captures".
        "source": first.get("source"),
        "coverage": first.get("coverage"),
        "fetched_at": first.get("fetched_at"),
        "attribution": first.get("attribution"),
        "captures": cap_meta,
    }, data_dir)
    print(f"  meta: {len(cap_meta)} capture(s), "
          f"eval={[k for k, v in cap_meta.items() if v['role'] == 'eval'] or 'same photograph as training'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", help="named area from areas.yaml")
    ap.add_argument("--name", help="name for an ad-hoc bbox area")
    ap.add_argument("--bbox", help="west,south,east,north (EPSG:4326)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--daterange", default=DEFAULT_DATERANGE,
                    help="sentinel2 fallback only")
    ap.add_argument("--captures", default=None,
                    help="comma-separated capture names to (re)fetch; default all")
    ap.add_argument("--force", action="store_true",
                    help="re-download captures whose .tif already exists")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    captures = None
    if args.area:
        areas = load_areas()
        if args.area not in areas:
            raise SystemExit(f"unknown area {args.area}; known: {list(areas)}")
        name, bbox = args.area, areas[args.area]["bbox"]
        captures = area_captures(areas[args.area])
    elif args.name and args.bbox:
        name, bbox = args.name, parse_bbox(args.bbox)
    else:
        raise SystemExit("need --area OR (--name AND --bbox)")

    print(f"Fetching {name} bbox={bbox}")
    fetch_area(name, bbox, data_dir, args.daterange, captures=captures,
               only=args.captures.split(",") if args.captures else None,
               force=args.force)


if __name__ == "__main__":
    main()
