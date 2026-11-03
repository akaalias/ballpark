"""Mission types for the flight simulator (issue #21).

sim/flightsim.py flies one shape of mission: a random A->B leg. This module
builds six more, each asking the champion something A->B does not, and flies
them with the same airframe, disturbances and navigator (fly_mission):

  a_to_b       the original, for reference
  return_home  out to B, back to the launch point (re-find home from the
               opposite heading — the lost-link failsafe)
  tour         four spots in a row: does error build up across legs?
  loiter       orbit one point (r = 150 m) for 5 min: many views of the same
               ground from every heading; one false fix pulls it off station
  survey       lawnmower grid, 6 lines x 1.2 km, 200 m apart, random
               orientation, 150 m run-in/out: cross-track error on the lines
  cold_start   A->B, but the navigator starts believing it is at the centre
               of the box: time until the estimate is within 100 m
  spree        follow the Spree ~7.5 km through the centre (traced from a
               water mask of the 2024 photograph): cross-track error over
               repetitive river-and-embankment texture

Success, per flight, at the same 100 m the mission score uses:
  a_to_b / cold_start   arrived, final miss <= 100 m
  return_home           B and home both reached within 100 m
  tour                  every spot reached within 100 m
  loiter                completed, >= 95% of the orbit within r + 100 m
  survey / spree        completed, >= 95% of the planned lines within 100 m

The camera is the held-out eval capture, as in flightsim. Mission geometry
comes from its own seeded RNG and the flight's hidden truth (wind, gyro bias)
from another, so --no-vision flies the very same missions in the very same
weather.

Usage:
  .venv/bin/python -m sim.missions --mission loiter --flights 40 --seed 1
  .venv/bin/python -m sim.missions --mission all --flights 40 --seed 1
  .venv/bin/python -m sim.missions --mission all --flights 40 --seed 1 --no-vision
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.common import load_meta
from pipeline.score import CONF_THRESHOLD, TARGET_M, load_session
from sim.flightsim import CHAMPION_ONNX, MARGIN_PX, fly_mission

INSET = MARGIN_PX + 400        # keep mission geometry well inside the mapped box

# Spree centreline, east -> west, raster px (1 m). Traced from a water mask of
# truedop_2024 (dark, blue-leaning 8 m cells), checked by overlay.
SPREE_PX = [(6264, 1884), (6024, 1724), (5864, 1564), (5704, 1284), (5544, 1084),
            (5304, 980), (4984, 964), (4744, 988), (4584, 1068), (4424, 1204),
            (4224, 1284), (4024, 1244), (3904, 1084), (3784, 924), (3664, 860),
            (3544, 900), (3384, 1004), (3264, 1124), (3064, 1228), (2904, 1324),
            (2744, 1428), (2584, 1460), (2424, 1404), (2264, 1284), (2104, 1124),
            (1944, 948), (1744, 844), (1544, 844), (1424, 964), (1344, 1124),
            (1264, 1244), (1104, 1444), (944, 1484), (824, 1364), (768, 1164),
            (744, 964), (664, 764)]

MISSIONS = ["a_to_b", "return_home", "tour", "loiter", "survey", "cold_start", "spree"]


def _pt(meta, rng, inset=INSET):
    return rng.uniform([inset, inset], [meta["width"] - inset, meta["height"] - inset])


def _far_pt(meta, rng, frm, min_m, max_m=None):
    while True:
        p = _pt(meta, rng)
        d = float(np.linalg.norm(p - frm)) * meta["gsd_m"]
        if d >= min_m and (max_m is None or d <= max_m):
            return p


def build(kind, meta, rng, i):
    """-> (start_px, waypoints, kwargs for fly_mission)."""
    if kind in ("a_to_b", "cold_start"):
        a = _pt(meta, rng)
        b = _far_pt(meta, rng, a, 2000)
        return a, [b], {"cold_start": kind == "cold_start"}
    if kind == "return_home":
        a = _pt(meta, rng)
        b = _far_pt(meta, rng, a, 2000)
        return a, [b, a.copy()], {}
    if kind == "tour":
        a = _pt(meta, rng)
        pts, prev = [], a
        for _ in range(4):
            prev = _far_pt(meta, rng, prev, 1000, 3000)
            pts.append(prev)
        return a, pts, {}
    if kind == "loiter":
        a = _pt(meta, rng)
        c = _far_pt(meta, rng, a, 1500, 3000)
        return a, [], {"loiter": {"center": [float(c[0]), float(c[1])],
                                  "radius_m": 150.0, "duration_s": 300.0}}
    if kind == "survey":
        n_lines, length, spacing, run_in = 6, 1200.0, 200.0, 150.0
        half = math.hypot(length + 2 * run_in, (n_lines - 1) * spacing) / 2 / meta["gsd_m"]
        c = _pt(meta, rng, inset=INSET + int(half))
        th = rng.uniform(0, math.pi)
        u = np.array([math.cos(th), math.sin(th)])          # along-line
        v = np.array([-u[1], u[0]])                          # across-line
        wps, legs = [], []
        for k in range(n_lines):
            off = (k - (n_lines - 1) / 2) * spacing / meta["gsd_m"]
            # each line is flown with a run-in/run-out beyond the grid, as a
            # survey planner would; cross-track is scored on the grid only
            ends = [c + v * off - u * (length / 2 + run_in) / meta["gsd_m"],
                    c + v * off + u * (length / 2 + run_in) / meta["gsd_m"]]
            if k % 2:
                ends.reverse()
            wps += ends
            legs.append(len(wps) - 1)                        # the leg INTO the line's end
        a = _far_pt(meta, rng, wps[0], 600, 1200)
        return a, wps, {"track_legs": tuple(legs), "xtrack_trim_m": run_in}
    if kind == "spree":
        pts = [np.array(p, float) for p in SPREE_PX]
        if i % 2:
            pts.reverse()                                    # alternate direction
        return pts[0], pts[1:], {"track_legs": tuple(range(len(pts) - 1))}
    raise ValueError(kind)


def success(kind, f):
    wp = f.get("waypoints", [])
    if kind in ("a_to_b", "cold_start"):
        return bool(f["arrived_within_100m"])
    if kind in ("return_home", "tour"):
        return f["result"] == "arrived" and all(w["true_miss_m"] <= TARGET_M for w in wp)
    if kind == "loiter":
        return f["result"] == "completed" and f["loiter"]["on_station"] >= 0.95
    if kind in ("survey", "spree"):
        return f["result"] == "arrived" and f["xtrack"]["within_100m"] >= 0.95
    raise ValueError(kind)


def _med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def summarize(kind, flights):
    n = len(flights)
    s = {"mission": kind, "n_flights": n,
         "success_rate": round(sum(f["success"] for f in flights) / n, 4),
         "result_counts": {r: sum(f["result"] == r for f in flights)
                           for r in sorted({f["result"] for f in flights})},
         "n_flights_with_false_fix": sum(1 for f in flights if f["fix_counts"]["false_fix"]),
         "total_fixes": {k: sum(f["fix_counts"][k] for f in flights)
                         for k in ("usable", "abstain", "false_fix", "off_map")},
         "est_err_max_median_m": _med([f["est_err_max_m"] for f in flights]),
         "flight_s_median": _med([f["flight_s"] for f in flights])}
    if kind in ("a_to_b", "cold_start"):
        s["final_miss_median_m"] = _med([f["final_true_dist_m"] for f in flights])
        s["final_miss_worst_m"] = max(f["final_true_dist_m"] for f in flights)
    if kind in ("return_home", "tour"):
        per = list(zip(*[[w["true_miss_m"] for w in f["waypoints"]]
                         for f in flights if len(f["waypoints"]) == len(f["waypoints_px"])]))
        s["waypoint_miss_median_m"] = [_med(p) for p in per]
        s["waypoint_miss_worst_m"] = [max(p) for p in per]
        s["waypoints_within_100m"] = round(
            sum(w["true_miss_m"] <= TARGET_M for f in flights for w in f["waypoints"])
            / sum(len(f["waypoints_px"]) for f in flights), 4)
    if kind == "loiter":
        s["on_station_median"] = _med([f["loiter"]["on_station"] for f in flights])
        s["on_station_worst"] = min(f["loiter"]["on_station"] for f in flights)
        s["max_dist_median_m"] = _med([f["loiter"]["max_dist_m"] for f in flights])
        s["max_dist_worst_m"] = max(f["loiter"]["max_dist_m"] for f in flights)
    if kind in ("survey", "spree"):
        s["xtrack_mean_median_m"] = _med([f["xtrack"]["mean_m"] for f in flights])
        s["xtrack_p95_median_m"] = _med([f["xtrack"]["p95_m"] for f in flights])
        s["xtrack_max_worst_m"] = max(f["xtrack"]["max_m"] for f in flights)
        s["xtrack_within_100m_median"] = _med([f["xtrack"]["within_100m"] for f in flights])
    if kind == "cold_start":
        s["localized_rate"] = round(sum(f["localized_s"] is not None for f in flights) / n, 4)
        s["localized_s_median"] = _med([f["localized_s"] for f in flights])
        s["first_confident_fix_false"] = sum(
            1 for f in flights if (f["first_confident_fix"] or {}).get("outcome") == "false_fix")
    return s


def run(kind, args, meta, img, sess):
    mrng = np.random.default_rng([args.seed, MISSIONS.index(kind)])
    frng = np.random.default_rng([args.seed, MISSIONS.index(kind), 1])
    flights = []
    for i in range(args.flights):
        start, wps, kw = build(kind, meta, mrng, i)
        f = fly_mission(sess, img, meta, [float(x) for x in start],
                        [[float(x) for x in w] for w in wps], frng,
                        vision=not args.no_vision, **kw)
        f["flight_id"] = i
        f["success"] = success(kind, f)
        if i >= args.keep_tracks:          # keep the file small; tracks only for the first few
            f.pop("track")
        flights.append(f)
        print(f"[{kind} {i:3d}] {f['result']:9s} {'OK ' if f['success'] else 'FAIL'} "
              f"fixes u/a/f/o {f['fix_counts']['usable']}/{f['fix_counts']['abstain']}/"
              f"{f['fix_counts']['false_fix']}/{f['fix_counts']['off_map']}", flush=True)
    summary = summarize(kind, flights)
    summary.update({"vision": not args.no_vision, "onnx": str(args.onnx),
                    "capture": args.capture, "seed": args.seed,
                    "conf_threshold": CONF_THRESHOLD, "target_m": TARGET_M})
    out = args.out_dir / f"{kind}{'_dr' if args.no_vision else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"summary": summary, "flights": flights}, fh)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="all", choices=MISSIONS + ["all"])
    ap.add_argument("--area", default="berlin")
    ap.add_argument("--onnx", type=Path, default=CHAMPION_ONNX)
    ap.add_argument("--capture", default="truedop_2024")
    ap.add_argument("--flights", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--keep-tracks", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=Path("sim/out/missions"))
    args = ap.parse_args()

    meta = load_meta(args.area)
    img = np.asarray(Image.open(Path("data") / args.area / "relight" / f"{args.capture}.png"))
    sess = load_session(args.onnx)
    for kind in (MISSIONS if args.mission == "all" else [args.mission]):
        run(kind, args, meta, img, sess)


if __name__ == "__main__":
    main()
