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
L1_LOOKAHEAD = 80.0             # m, line-following lookahead on tracked legs (fly_mission)

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
    """One A->B flight (the original showcase mission): a single waypoint."""
    return fly_mission(sess, img, meta, start_px, [target_px], rng,
                       vision=vision, save_crops_dir=save_crops_dir)


def _seg_dist(p, a, b):
    """Distance from point p to segment a-b (all in px)."""
    a, b, p = np.asarray(a, float), np.asarray(b, float), np.asarray(p, float)
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def fly_mission(sess, img, meta, start_px, waypoints, rng, *,
                vision=True, save_crops_dir=None, loiter=None,
                cold_start=False, track_legs=(), xtrack_trim_m=0.0):
    """One mission. Coordinates are raster pixels; 1 px = gsd_m meters (1.0).

    waypoints   visited in order. An intermediate waypoint counts as reached
                when the ESTIMATE is within ARRIVE_RADIUS or has passed the
                line through the waypoint perpendicular to the leg (ArduPlane's
                rule; a fixed-wing with a 37 m turn radius can otherwise orbit
                a point it overshot forever). The final waypoint needs the
                radius, as in the original A->B mission, and gets the
                approach fix schedule.
    loiter      {"center": [x, y], "radius_m": R, "duration_s": T}: after the
                waypoints, fly to the centre and orbit it clockwise for T s
                (cruise fix schedule throughout). Ends "completed".
    cold_start  the navigator does not know where it starts: its belief is
                the centre of the box with unbounded uncertainty, so the first
                confident fix (right or wrong) is taken almost whole. Until
                that first fix it circles in place rather than navigate on
                a guess.
    track_legs  indices i of legs (waypoint i-1 -> waypoint i; leg 0 starts
                at start_px) on which the TRUE cross-track distance to the
                planned line is recorded (survey lines, corridor). On these
                legs guidance follows the LINE (L1-style lookahead) instead
                of pursuing the end point, as a real autopilot would.
    xtrack_trim_m  ignore cross-track over the first/last this-many metres
                of each tracked leg (survey run-in/run-out, where the
                aircraft is still turning onto the line).
    """
    gsd = meta["gsd_m"]
    width, height = meta["width"], meta["height"]
    waypoints = [list(map(float, w)) for w in waypoints]

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
    if cold_start:
        est = np.array([width / 2.0, height / 2.0])
        t_last_accepted = -1e6                     # sigma_est huge -> k ~ 1
    else:
        est = np.array(start_px, dtype=float)      # navigator's belief
        t_last_accepted = 0.0
    psi_true = rng.uniform(0, 360)                 # true heading
    psi_est = psi_true                             # believed heading (gyro-integrated)
    t = 0.0
    t_last_fix_attempt = -1e9
    dist_flown = 0.0

    legs = [start_px] + waypoints
    path_m = sum(float(np.linalg.norm(np.subtract(legs[i + 1], legs[i])))
                 for i in range(len(legs) - 1)) * gsd
    if loiter is not None:
        path_m += float(np.linalg.norm(np.subtract(loiter["center"], legs[-1]))) * gsd
    direct_m = path_m
    timeout = 3.0 * path_m / AIRSPEED + 60.0
    if loiter is not None:
        timeout += loiter["duration_s"] + 120.0

    track, fixes, wp_log, xtrack = [], [], [], []
    n_no_image = 0
    closest_true = float("inf")
    next_track_t = 0.0
    result = None
    wi = 0                                          # current waypoint index
    final_target = waypoints[-1] if waypoints else loiter["center"]
    loiter_t0 = None
    loiter_d = []                                   # true dist to centre while orbiting
    localized_s = 0.0 if not cold_start else None   # first time |est - true| <= TARGET_M
    est_err_max = 0.0

    while t < timeout:
        # ---- guidance (uses ONLY the navigator's belief) ----
        if cold_start and t_last_accepted < 0:
            # no position yet: hold a gentle circle until the first confident
            # fix, instead of "arriving" somewhere on a guess
            desired_psi = psi_est + 0.5 * MAX_TURN_RATE / GUIDANCE_GAIN
            est_dist = float("inf")
            approach = False
        elif wi < len(waypoints):
            tgt = np.array(waypoints[wi])
            to_target = tgt - est
            est_dist = float(np.linalg.norm(to_target)) * gsd
            last = wi == len(waypoints) - 1 and loiter is None
            reached = est_dist < ARRIVE_RADIUS
            if not reached and not last:
                prev = np.array(legs[wi])
                leg = tgt - prev
                reached = float(np.dot(est - tgt, leg)) > 0.0   # passed the perpendicular
            if reached:
                wp_log.append({"wp": wi, "t": round(t, 1),
                               "true_miss_m": round(float(np.linalg.norm(tgt - pos)) * gsd, 1),
                               "how": "radius" if est_dist < ARRIVE_RADIUS else "passed"})
                wi += 1
                if wi == len(waypoints) and loiter is None:
                    result = "arrived"
                    break
                continue                            # re-plan on the next waypoint, same t
            aim = to_target
            if wi in track_legs:
                # line following (ArduPlane L1-style): aim at a point
                # L1_LOOKAHEAD ahead of our projection onto the planned leg
                a_ = np.array(legs[wi])
                ab = tgt - a_
                L = float(np.linalg.norm(ab))
                s_ = float(np.dot(est - a_, ab)) / max(L, 1e-9) + L1_LOOKAHEAD / gsd
                aim = (a_ + ab * min(s_ / max(L, 1e-9), 1.0)) - est
            desired_psi = math.degrees(math.atan2(aim[0], -aim[1]))
            approach = last
        else:                                       # loiter
            c = np.array(loiter["center"])
            to_c = c - est
            d_est = float(np.linalg.norm(to_c)) * gsd
            R = loiter["radius_m"]
            if loiter_t0 is None and d_est < 1.5 * R:
                loiter_t0 = t
            if loiter_t0 is not None and t - loiter_t0 >= loiter["duration_s"]:
                result = "completed"
                break
            bearing = math.degrees(math.atan2(to_c[0], -to_c[1]))
            desired_psi = bearing - 90.0 + math.degrees(math.atan((d_est - R) / 50.0))
            est_dist = d_est
            approach = False
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

        true_dist = float(np.linalg.norm(np.array(final_target) - pos)) * gsd
        closest_true = min(closest_true, true_dist)
        est_err = float(np.linalg.norm(est - pos)) * gsd
        if localized_s is None and est_err <= TARGET_M:
            localized_s = round(t, 1)
        if localized_s is not None:
            est_err_max = max(est_err_max, est_err)
        if wi < len(waypoints) and wi in track_legs:
            a_, b_ = np.array(legs[wi], float), np.array(waypoints[wi], float)
            ab = b_ - a_
            L = float(np.linalg.norm(ab)) * gsd
            along = float(np.dot(pos - a_, ab)) / max(L / gsd, 1e-9) * gsd
            if xtrack_trim_m <= along <= L - xtrack_trim_m:
                xtrack.append(_seg_dist(pos, a_, b_) * gsd)
        if loiter_t0 is not None:
            loiter_d.append(float(np.linalg.norm(np.array(loiter["center"]) - pos)) * gsd)

        # ---- vision fix ----
        interval = FIX_INTERVAL_APPROACH if (approach and est_dist < APPROACH_DIST) else FIX_INTERVAL_CRUISE
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

    true_miss = float(np.linalg.norm(np.array(final_target) - pos)) * gsd
    n = {o: sum(1 for f in fixes if f["outcome"] == o)
         for o in ("usable", "abstain", "false_fix", "off_map")}
    out = {
        "start_px": list(start_px), "target_px": list(final_target),
        "start_lonlat": px_to_lonlat(meta, *start_px),
        "target_lonlat": px_to_lonlat(meta, *final_target),
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
    if len(waypoints) > 1 or loiter is not None:
        out["waypoints_px"] = waypoints
        out["waypoints"] = wp_log
    if cold_start:
        out["cold_start"] = True
        out["localized_s"] = localized_s
        first = next((f for f in fixes if f["outcome"] in ("usable", "false_fix")), None)
        out["first_confident_fix"] = (None if first is None
                                      else {"t": first["t"], "outcome": first["outcome"]})
    out["est_err_max_m"] = round(est_err_max, 1)
    if track_legs:
        xa = np.array(xtrack) if xtrack else np.zeros(1)
        out["xtrack"] = {"mean_m": round(float(xa.mean()), 1),
                         "p95_m": round(float(np.percentile(xa, 95)), 1),
                         "max_m": round(float(xa.max()), 1),
                         "within_100m": round(float((xa <= TARGET_M).mean()), 4)}
    if loiter is not None:
        out["loiter"] = dict(loiter)
        la = np.array(loiter_d) if loiter_d else np.array([np.inf])
        out["loiter"].update({
            "started_s": None if loiter_t0 is None else round(loiter_t0, 1),
            "on_station": round(float((la <= loiter["radius_m"] + TARGET_M).mean()), 4),
            "max_dist_m": round(float(la.max()), 1),
            "mean_abs_radial_err_m": round(float(np.abs(la - loiter["radius_m"]).mean()), 1)})
    return out


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
