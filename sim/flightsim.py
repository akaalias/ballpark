"""Virtual UAV flight simulator — puts the champion to the a-to-b test.

NOT part of the research loop and NOT frozen. Reads the frozen pipeline
(extract_crop, predict, the score thresholds) and never modifies it. The
question it answers is the product question the mission score was derived
from: with vision fixes as the ONLY drift correction (CLAUDE.md §2), can
the champion actually guide a fixed-wing UAV from a to b over Berlin?

Honesty rules, fixed up front (2026-08-28, per Alexis):
  - The navigator sees only what a real FC would see: dead reckoning from
    commanded heading + nominal airspeed (wind and gyro bias unknown to it),
    corrected only by fixes the model is confident about. False fixes are
    accepted like any other confident fix — no oracle filtering, no
    innovation gate in v1.
  - Arrival is declared on the ESTIMATE (all a real UAV can do); the
    reported miss is the TRUE distance to target at that moment. If the
    flight times out, that is reported as no arrival — not retried away.
  - Fix outcomes are classified with the SAME thresholds as score.py
    (CONF_THRESHOLD, TARGET_M=100), imported, not re-typed.
  - The camera renders from the area's EVAL capture (--capture, default
    truedop_2024 for Berlin since 2026-11-02, issue #15): a photograph the
    model never saw in training, exactly as score.py asks. Eval-time use
    only. Until then it rendered from the raster the model memorized.
  - Known best-case caveats, stated rather than hidden: altitude is fixed at the trained
    1 m/px scale; true pose is rounded to the nearest pixel (<=0.7 m,
    below the GSD).

Usage:
  .venv/bin/python -m sim.flightsim --seed 7 --out sim/out/showcase.json
  .venv/bin/python -m sim.flightsim --flights 100 --seed 1 --out sim/out/mc.json
  .venv/bin/python -m sim.flightsim --flights 100 --seed 1 --no-vision --out sim/out/mc_dr.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.common import CROP_PX, load_meta, px_to_lonlat
from pipeline.dataset import WINDOW_PX, extract_crop, norm_to_px
from pipeline.score import CONF_THRESHOLD, TARGET_M, load_session, predict

# capture-holdout experiment 20 (mission 0.0777 on the held-out photograph).
# Until 2026-11-02: runs/20260731_112048_experiment5 (berlin-slim, 0.040 on
# its own training photograph).
CHAMPION_ONNX = Path("runs/20261102_010051_experiment20/models/berlin.onnx")

# ---- airframe / dynamics (TBS Source One V6-class fixed-wing conversion) ----
AIRSPEED = 16.0          # m/s nominal cruise
MAX_TURN_RATE = 25.0     # deg/s (~35 deg bank at 16 m/s)
DT = 0.1                 # s integration step
GUIDANCE_GAIN = 1.2      # heading-error -> turn-rate command (1/s)

# ---- disturbances the navigator does NOT know about ----
WIND_SPEED_RANGE = (2.0, 6.0)   # m/s, constant per flight, random direction
GUST_SIGMA = 1.0                # m/s, OU process per axis
GUST_TAU = 5.0                  # s
GYRO_BIAS_SIGMA = 0.3           # deg/s, constant per flight
AIRSPEED_SCALE_SIGMA = 0.03     # true airspeed = nominal * N(1, sigma)

# ---- magnetometer (standard FC hardware — yaw aiding, NOT position aiding).
# Without it the gyro bias integrates unbounded and the believed heading is
# driven to the target while the true heading walks off by 100+ degrees —
# an aircraft nobody would fly. Vision remains the ONLY position correction.
MAG_NOISE_DEG = 3.0             # per-reading noise
MAG_OFFSET_SIGMA = 2.0          # deg, constant per flight (hard-iron/declination)
MAG_GAIN = 0.5                  # 1/s complementary-filter pull toward mag yaw

# ---- navigator (what the FC believes) ----
FIX_INTERVAL_CRUISE = 6.0       # s (§2: 5-10 s cruise)
FIX_INTERVAL_APPROACH = 2.0     # s (§2: 1-2 s final approach)
APPROACH_DIST = 300.0           # m estimated distance that switches schedule
EST_VEL_SIGMA = 5.0             # m/s assumed unmodeled velocity error (drift growth)
EST_BASE_SIGMA = 10.0           # m floor on estimate uncertainty
FIX_SIGMA = 30.0                # m assumed fix measurement noise (~champion median 27 m)
ARRIVE_RADIUS = 30.0            # m estimated distance at which arrival is declared

MARGIN_PX = WINDOW_PX // 2 + 1  # extract_crop needs this much raster around the pose
TRACK_EVERY = 0.5               # s between recorded track points


def heading_vec(psi_deg: float) -> np.ndarray:
    """Compass heading -> raster-frame unit vector (x=east=+col, y=+row=south)."""
    r = math.radians(psi_deg)
    return np.array([math.sin(r), -math.cos(r)])


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def simulate_flight(sess, img, meta, start_px, target_px, rng, *,
                    vision=True, save_crops_dir=None):
    """One flight. Coordinates are raster pixels; 1 px = gsd_m meters (1.0)."""
    gsd = meta["gsd_m"]
    width, height = meta["width"], meta["height"]

    # -- per-flight hidden truth (unknown to the navigator) --
    wind_dir = rng.uniform(0, 360)
    wind_speed = rng.uniform(*WIND_SPEED_RANGE)
    wind = heading_vec(wind_dir) * wind_speed
    gust = np.zeros(2)
    gyro_bias = rng.normal(0.0, GYRO_BIAS_SIGMA)
    airspeed_true = AIRSPEED * rng.normal(1.0, AIRSPEED_SCALE_SIGMA)
    mag_offset = rng.normal(0.0, MAG_OFFSET_SIGMA)

    # -- state --
    pos = np.array(start_px, dtype=float)          # true (px == m)
    est = np.array(start_px, dtype=float)          # navigator's belief
    psi_true = rng.uniform(0, 360)                 # true heading
    psi_est = psi_true                             # believed heading (gyro-integrated)
    t = 0.0
    t_last_fix_attempt = -1e9
    t_last_accepted = 0.0
    dist_flown = 0.0

    direct_m = float(np.linalg.norm(np.array(target_px) - np.array(start_px))) * gsd
    timeout = 3.0 * direct_m / AIRSPEED + 60.0

    track, fixes = [], []
    n_no_image = 0
    closest_true = float("inf")
    next_track_t = 0.0
    result = None

    while t < timeout:
        # ---- guidance (uses ONLY the navigator's belief) ----
        to_target = np.array(target_px) - est
        est_dist = float(np.linalg.norm(to_target)) * gsd
        if est_dist < ARRIVE_RADIUS:
            result = "arrived"
            break
        desired_psi = math.degrees(math.atan2(to_target[0], -to_target[1]))
        r_cmd = float(np.clip(GUIDANCE_GAIN * wrap180(desired_psi - psi_est),
                              -MAX_TURN_RATE, MAX_TURN_RATE))

        # ---- true dynamics ----
        r_true = r_cmd + rng.normal(0.0, 0.5)      # actuation/turbulence jitter
        psi_true = (psi_true + r_true * DT) % 360.0
        gust += (-gust / GUST_TAU) * DT + GUST_SIGMA * math.sqrt(2 * DT / GUST_TAU) * rng.standard_normal(2)
        vel_true = heading_vec(psi_true) * airspeed_true + wind + gust
        pos += vel_true * DT / gsd
        dist_flown += float(np.linalg.norm(vel_true)) * DT

        # ---- navigator propagation (gyro has bias; wind/gusts unknown) ----
        psi_est = (psi_est + (r_true + gyro_bias) * DT) % 360.0
        psi_mag = psi_true + mag_offset + rng.normal(0.0, MAG_NOISE_DEG)
        psi_est = (psi_est + MAG_GAIN * wrap180(psi_mag - psi_est) * DT) % 360.0
        est += heading_vec(psi_est) * AIRSPEED * DT / gsd

        true_dist = float(np.linalg.norm(np.array(target_px) - pos)) * gsd
        closest_true = min(closest_true, true_dist)

        # ---- vision fix ----
        interval = FIX_INTERVAL_APPROACH if est_dist < APPROACH_DIST else FIX_INTERVAL_CRUISE
        if vision and t - t_last_fix_attempt >= interval:
            t_last_fix_attempt = t
            cx, cy = int(round(pos[0])), int(round(pos[1]))
            if not (MARGIN_PX <= cx < width - MARGIN_PX and MARGIN_PX <= cy < height - MARGIN_PX):
                n_no_image += 1
                fixes.append({"t": round(t, 1), "outcome": "off_map",
                              "true": [round(pos[0], 1), round(pos[1], 1)]})
            else:
                crop = extract_crop(img, cx, cy, psi_true % 360.0)
                u, v, conf = predict(sess, crop)
                zx, zy = norm_to_px(meta, u, v)
                err_m = float(np.hypot(zx - pos[0], zy - pos[1])) * gsd
                if conf < CONF_THRESHOLD:
                    outcome = "abstain"
                elif err_m <= TARGET_M:
                    outcome = "usable"
                else:
                    outcome = "false_fix"
                rec = {"t": round(t, 1), "outcome": outcome, "conf": round(conf, 3),
                       "true": [round(pos[0], 1), round(pos[1], 1)],
                       "pred": [round(zx, 1), round(zy, 1)],
                       "est_before": [round(est[0], 1), round(est[1], 1)],
                       "err_m": round(err_m, 1)}
                if conf >= CONF_THRESHOLD:
                    # accept: blend by staleness of the estimate vs fix noise
                    sigma_est = EST_BASE_SIGMA + EST_VEL_SIGMA * (t - t_last_accepted)
                    k = sigma_est**2 / (sigma_est**2 + FIX_SIGMA**2)
                    est = est + k * (np.array([zx, zy]) - est)
                    t_last_accepted = t
                    rec["gain"] = round(k, 3)
                    rec["est_after"] = [round(est[0], 1), round(est[1], 1)]
                if save_crops_dir is not None:
                    p = save_crops_dir / f"fix_{len(fixes):03d}_{outcome}.png"
                    Image.fromarray(crop).save(p)
                    rec["crop"] = p.name
                fixes.append(rec)

        if t >= next_track_t:
            track.append([round(t, 1), round(pos[0], 1), round(pos[1], 1),
                          round(est[0], 1), round(est[1], 1)])
            next_track_t += TRACK_EVERY
        t += DT

    true_miss = float(np.linalg.norm(np.array(target_px) - pos)) * gsd
    n = {o: sum(1 for f in fixes if f["outcome"] == o)
         for o in ("usable", "abstain", "false_fix", "off_map")}
    return {
        "start_px": list(start_px), "target_px": list(target_px),
        "start_lonlat": px_to_lonlat(meta, *start_px),
        "target_lonlat": px_to_lonlat(meta, *target_px),
        "direct_m": round(direct_m, 1),
        "wind": {"speed_ms": round(wind_speed, 2), "dir_deg": round(wind_dir, 1)},
        "gyro_bias_dps": round(gyro_bias, 3),
        "result": result or "timeout",
        "flight_s": round(t, 1),
        "dist_flown_m": round(dist_flown, 1),
        "declared_miss_m": round(true_miss, 1) if result == "arrived" else None,
        "final_true_dist_m": round(true_miss, 1),
        "closest_true_m": round(closest_true, 1),
        "arrived_within_100m": result == "arrived" and true_miss <= 100.0,
        "fix_counts": n,
        "fixes": fixes,
        "track": track,
    }


def sample_endpoints(meta, rng, min_sep_m=2000.0):
    """Random start/target inside the raster with margin, min separation."""
    inset = MARGIN_PX + 400  # keep endpoints well inside the mapped box
    while True:
        pts = rng.uniform([inset, inset],
                          [meta["width"] - inset, meta["height"] - inset], (2, 2))
        if np.linalg.norm(pts[1] - pts[0]) * meta["gsd_m"] >= min_sep_m:
            return [float(pts[0][0]), float(pts[0][1])], [float(pts[1][0]), float(pts[1][1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="berlin")
    ap.add_argument("--onnx", type=Path, default=CHAMPION_ONNX)
    ap.add_argument("--capture", default="truedop_2024",
                    help="which photograph the camera sees (data/<area>/relight/<capture>.png); "
                         "default is Berlin's held-out eval capture")
    ap.add_argument("--flights", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-vision", action="store_true",
                    help="dead-reckoning-only control: no fixes taken at all")
    ap.add_argument("--start", help="fixed start 'x,y' raster px (default: random)")
    ap.add_argument("--target", help="fixed target 'x,y' raster px (default: random)")
    ap.add_argument("--flight-dir", type=Path,
                    help="also write each flight's FULL record (track included) "
                         "as <dir>/f<id>.json before the compact multi-flight "
                         "stripping is applied")
    ap.add_argument("--save-crops", action="store_true",
                    help="save each fix's camera crop next to the output JSON")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    meta = load_meta(args.area)
    img = np.asarray(Image.open(Path("data") / args.area / "relight" / f"{args.capture}.png"))
    sess = load_session(args.onnx)
    rng = np.random.default_rng(args.seed)

    crops_dir = None
    if args.save_crops:
        crops_dir = args.out.parent / (args.out.stem + "_crops")
        crops_dir.mkdir(parents=True, exist_ok=True)

    flights = []
    for i in range(args.flights):
        start, target = sample_endpoints(meta, rng)
        if args.start:
            start = [float(v) for v in args.start.split(",")]
        if args.target:
            target = [float(v) for v in args.target.split(",")]
        f = simulate_flight(sess, img, meta, start, target, rng,
                            vision=not args.no_vision, save_crops_dir=crops_dir)
        f["flight_id"] = i
        if args.flight_dir:
            args.flight_dir.mkdir(parents=True, exist_ok=True)
            with open(args.flight_dir / f"f{i}.json", "w") as fh:
                json.dump(f, fh)
        if args.flights > 1:      # keep Monte Carlo output small
            f.pop("track")
            f["fixes"] = [{k: v for k, v in fx.items() if k != "crop"}
                          for fx in f["fixes"]]
        flights.append(f)
        print(f"[{i:3d}] {f['result']:8s} direct {f['direct_m']:7.0f} m  "
              f"miss {f['final_true_dist_m']:7.1f} m  closest {f['closest_true_m']:7.1f} m  "
              f"fixes u/a/f/o {f['fix_counts']['usable']}/{f['fix_counts']['abstain']}/"
              f"{f['fix_counts']['false_fix']}/{f['fix_counts']['off_map']}")

    arrived = [f for f in flights if f["result"] == "arrived"]
    ok100 = [f for f in flights if f["arrived_within_100m"]]
    misses = sorted(f["declared_miss_m"] for f in arrived)
    summary = {
        "n_flights": len(flights),
        "vision": not args.no_vision,
        "onnx": str(args.onnx),
        "capture": args.capture,
        "conf_threshold": CONF_THRESHOLD, "target_m": TARGET_M,
        "params": {"airspeed_ms": AIRSPEED, "max_turn_dps": MAX_TURN_RATE,
                   "fix_s_cruise": FIX_INTERVAL_CRUISE,
                   "fix_s_approach": FIX_INTERVAL_APPROACH,
                   "wind_ms": list(WIND_SPEED_RANGE),
                   "gyro_bias_sigma_dps": GYRO_BIAS_SIGMA,
                   "mag_noise_deg": MAG_NOISE_DEG, "mag_offset_sigma_deg": MAG_OFFSET_SIGMA,
                   "fix_sigma_m": FIX_SIGMA, "arrive_radius_m": ARRIVE_RADIUS,
                   "seed": args.seed},
        "n_arrived": len(arrived),
        "n_arrived_within_100m": len(ok100),
        "arrival_rate_100m": round(len(ok100) / len(flights), 4),
        "declared_miss_median_m": misses[len(misses) // 2] if misses else None,
        "declared_miss_worst_m": misses[-1] if misses else None,
        "n_flights_with_false_fix": sum(1 for f in flights if f["fix_counts"]["false_fix"]),
        "total_fixes": {k: sum(f["fix_counts"][k] for f in flights)
                        for k in ("usable", "abstain", "false_fix", "off_map")},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "flights": flights}, f)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
