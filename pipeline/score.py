"""FROZEN (see /FROZEN) — scoring, §6.

PRIMARY METRIC = THE PRODUCT REQUIREMENT, not a statistic about errors.

The aircraft takes a vision fix every 5-10 s and it is its ONLY drift
correction (§2). Per frame, exactly three things can happen:

    confident AND within TARGET_M   -> USABLE FIX   (the product)
    not confident                   -> abstains     (safe; waits for the next)
    confident AND outside TARGET_M  -> FALSE FIX    (dangerous; injects a
                                       wrong position into navigation)

    mission_score = (1 - usable_fix_rate) + false_fix_rate   [minimized]

0.0 is perfect; abstaining everywhere is 1.0; being confidently wrong
everywhere is 2.0 — strictly worse than silence, which is what a drone needs.
The loop optimizes the worst such score across buckets x development areas.

WHY NOT AN ERROR STATISTIC. Median, geometric mean and percentiles were each
tried as the primary and each rewarded something the product does not want.
The median (to 2026-07-31) rewarded a model that guesses the map centre — a
tight mediocre distribution beats a bimodal one — and was caught reverting the
only experiment that had begun to memorize anything. The geometric mean
(2026-07-31, hours) fixed that but was still a research proxy chosen to make
progress visible rather than to state what the aircraft needs, which is the
same category of error. All of them stay LOGGED as diagnostics; none is
optimized. If a future change makes the score improve while usable_fix_rate
does not, the metric is wrong again — that is the check to run.

A tiny tie-break term (TIE_BREAK_W, bounded strictly below 1/n_eval) separates
candidates the product metric scores identically, so the loop still has signal
at 0% usable. It can never outrank a real difference in usable/false rates.

Gates folded into the scalar (§6): a per-area exported ONNX must fit the
ESP32-P4 envelope — file size <= MODEL_MAX_BYTES and single-thread host CPU
latency <= LATENCY_MAX_MS (a documented *proxy* for on-target latency, not a
measurement of it). Coverage rule: predictions with confidence < CONF_THRESHOLD
are abstentions; a (area, bucket) cell with coverage < MIN_COVERAGE scores
FAIL_SCORE — a model cannot pass by abstaining its way out.

Holdout (§5): scoring hamburg requires --holdout, writes to a separate output,
and must never feed the keep/revert decision (enforced in loop.sh).

CAPTURES (2026-09-28, see pipeline/common.py). The scored buckets are the
area's EVAL captures — photographs the model never trained on — under each
lighting condition; the primary is the worst of them. The training captures
are scored too and written under "train_capture_diagnostics": LOGGED ONLY,
never part of the primary, so the memorisation/generalisation gap of every
experiment is on the record without being able to drive keep/revert. Set
SCORE_DIAGNOSTICS=0 to skip them (bulk re-scoring of history). For an area
without an eval capture the eval buckets ARE the training buckets — the
pre-era behaviour — and no diagnostics block is written.

Usage:
  python -m pipeline.score --areas berlin,prignitz --model-dir runs/X/models --out runs/X/metrics.json
  python -m pipeline.score --areas hamburg --holdout --model-dir ... --out runs/X/holdout.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pipeline.common import (CROP_PX, DATA_DIR, area_dir, buckets, load_areas,
                             load_meta)
from pipeline.dataset import error_meters, extract_crop, list_crops

MODEL_MAX_BYTES = 4 * 1024 * 1024   # ESP32-P4 flash/PSRAM envelope per model
LATENCY_MAX_MS = 250.0              # single-thread host-CPU proxy budget
CONF_THRESHOLD = 0.3
MIN_COVERAGE = 0.2
FAIL_SCORE = 1e9
MAX_EVAL_CROPS_PER_BUCKET = 400     # deterministic subsample cap for speed
TARGET_M = 100.0                    # a fix within this radius is USABLE (main branch: 20 m)
TIE_BREAK_W = 0.001                 # max influence of the log-error tie-break; < 1/n_eval


def load_session(onnx_path: Path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), sess_options=opts,
                                providers=["CPUExecutionProvider"])


def predict(sess, crop_u8: np.ndarray) -> tuple[float, float, float]:
    x = crop_u8.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    u, v, conf = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    return float(u), float(v), float(conf)


def measure_latency_ms(sess) -> float:
    x = np.zeros((CROP_PX, CROP_PX, 3), dtype=np.uint8)
    for _ in range(3):
        predict(sess, x)
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        predict(sess, x)
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def score_area(area: str, model_dir: Path, data_dir: Path, heatmap_dir: Path | None):
    meta = load_meta(area, data_dir)
    onnx_path = model_dir / f"{area}.onnx"
    result = {"area": area, "buckets": {}, "gates": {}}

    size = onnx_path.stat().st_size if onnx_path.exists() else None
    result["gates"]["model_bytes"] = size
    if size is None or size > MODEL_MAX_BYTES:
        result["gates"]["failed"] = "missing model" if size is None else "model too large"
        return result

    sess = load_session(onnx_path)
    lat = measure_latency_ms(sess)
    result["gates"]["latency_ms_host_proxy"] = round(lat, 2)
    if lat > LATENCY_MAX_MS:
        result["gates"]["failed"] = "latency over proxy budget"
        return result

    crops = list_crops(area, meta["width"], meta["height"], "eval")
    if len(crops) > MAX_EVAL_CROPS_PER_BUCKET:
        idx = np.linspace(0, len(crops) - 1, MAX_EVAL_CROPS_PER_BUCKET).astype(int)
        crops = [crops[i] for i in idx]

    eval_b = buckets(meta, "eval")
    train_b = {k: v for k, v in buckets(meta, "train").items() if k not in eval_b}
    heat_points = []

    def score_bucket(bucket):
        img = np.asarray(Image.open(area_dir(area, data_dir) / "relight" / f"{bucket}.png"))
        errs, n_conf = [], 0
        for c in crops:
            crop = extract_crop(img, c["cx"], c["cy"], c["angle"])
            u, v, conf = predict(sess, crop)
            if conf >= CONF_THRESHOLD:
                n_conf += 1
                e = error_meters(meta, u, v, c["cx"], c["cy"])
                errs.append(e)
                heat_points.append((c["cx"], c["cy"], e))
        # ---- THE PRODUCT REQUIREMENT (CLAUDE.md §2/§6) ----
        # The aircraft takes a vision fix every 5-10 s and it is the ONLY drift
        # correction available. For each frame exactly one of three things
        # happens, and only these three things matter:
        #
        #   confident AND within target  -> USABLE FIX      (what we want)
        #   not confident                -> abstains        (safe: waits)
        #   confident AND outside target -> FALSE FIX       (dangerous: it
        #                                   injects a wrong position into nav)
        #
        # So the score is a mission-failure rate, minimized:
        #
        #   score = (1 - usable_fix_rate) + false_fix_rate  [+ tiny tie-break]
        #
        # Abstaining everywhere gives 1.0. Being confidently wrong everywhere
        # gives 2.0 -- strictly worse than silence, which is correct for a
        # drone. Perfect gives 0.0. This directly encodes §6's rule that "a
        # model that honestly abstains on bad frames is more useful than one
        # that confidently guesses wrong".
        #
        # Deliberately NOT a statistic about the error distribution. Median,
        # geometric mean and percentiles were each tried as the primary and
        # each rewarded something the product does not want (2026-07-31: the
        # median rewarded guessing the map centre; the geometric mean was a
        # research proxy chosen to make progress visible, which is the same
        # category of mistake). They remain logged as diagnostics ONLY.
        #
        # TIE-BREAK: usable/false rates are quantized to 1/n_eval (0.0025 at
        # n=400). The log-error term below can shift the score by at most
        # TIE_BREAK_W (0.001), so it can NEVER outrank a genuine difference in
        # the product metric -- it only separates candidates the product
        # metric scores identically, so the loop is not blind at 0% usable.
        coverage = n_conf / max(len(crops), 1)
        ea = np.asarray(errs, dtype=float) if errs else np.array([])
        n_all = max(len(crops), 1)
        n_usable = int((ea <= TARGET_M).sum()) if errs else 0
        n_false = int((ea > TARGET_M).sum()) if errs else 0
        usable_rate = n_usable / n_all
        false_rate = n_false / n_all
        geo = float(np.exp(np.mean(np.log(np.clip(ea, 1.0, None))))) if errs else None
        tie = 0.0
        if geo is not None:
            tie = TIE_BREAK_W * min(max((np.log10(geo) - 1.0) / 3.0, 0.0), 1.0)
        mission = (1.0 - usable_rate) + false_rate + tie
        cell = {
            "n_eval": len(crops),
            "coverage": round(coverage, 4),
            # --- the product requirement ---
            "usable_fix_rate": round(usable_rate, 4),
            "false_fix_rate": round(false_rate, 4),
            "abstain_rate": round(1.0 - coverage, 4),
            "mission_score": round(mission, 6),
            # --- diagnostics only, never optimized ---
            "geomean_error_m": round(geo, 2) if geo is not None else None,
            "median_error_m": round(float(np.median(ea)), 2) if errs else None,
            "mean_error_m": round(float(np.mean(ea)), 2) if errs else None,
            "p10_error_m": round(float(np.percentile(ea, 10)), 2) if errs else None,
            "p25_error_m": round(float(np.percentile(ea, 25)), 2) if errs else None,
        }
        cell["score"] = FAIL_SCORE if coverage < MIN_COVERAGE else cell["mission_score"]
        return cell

    for bucket, b in eval_b.items():
        cell = score_bucket(bucket)
        cell["capture"] = b["capture"]
        cell["lighting"] = b["lighting"]
        result["buckets"][bucket] = cell

    # Training-capture diagnostics: how the model does on the photographs it
    # WAS shown, from held-out viewpoints. The gap between this and the eval
    # capture is the memorisation-vs-ground question this era exists to ask.
    # Logged only; never enters the primary.
    if train_b and os.environ.get("SCORE_DIAGNOSTICS", "1") != "0":
        heat_keep = list(heat_points)
        diag = {}
        for bucket, b in train_b.items():
            cell = score_bucket(bucket)
            cell["capture"] = b["capture"]
            cell["lighting"] = b["lighting"]
            cell.pop("score", None)
            cell["note"] = "training photograph, held-out viewpoints; logged only, never scored"
            diag[bucket] = cell
        result["train_capture_diagnostics"] = diag
        heat_points[:] = heat_keep  # heatmap shows the EVAL capture only

    # Region-holdout diagnostic (dataset.py v2): a small 1-in-32-block region
    # genuinely excluded from training. LOGGED ONLY — deliberately not folded
    # into any bucket "score", so it can never drive keep/revert. Its job is to
    # reveal whether the model has spatial structure or is a pure lookup table
    # over memorized vantages. Expect it to be much worse than the primary
    # metric; that is not a failure, it is the point of the measurement.
    hold = list_crops(area, meta["width"], meta["height"], "holdout_region")
    if hold:
        if len(hold) > MAX_EVAL_CROPS_PER_BUCKET:
            idx = np.linspace(0, len(hold) - 1, MAX_EVAL_CROPS_PER_BUCKET).astype(int)
            hold = [hold[i] for i in idx]
        bucket0 = next(iter(eval_b))
        img = np.asarray(Image.open(area_dir(area, data_dir) / "relight" / f"{bucket0}.png"))
        errs, n_conf = [], 0
        for c in hold:
            u, v, conf = predict(sess, extract_crop(img, c["cx"], c["cy"], c["angle"]))
            if conf >= CONF_THRESHOLD:
                n_conf += 1
                errs.append(error_meters(meta, u, v, c["cx"], c["cy"]))
        eh = np.asarray(errs, dtype=float) if errs else np.array([])
        nh = max(len(hold), 1)
        result["region_holdout"] = {
            "n_eval": len(hold),
            "bucket": bucket0,
            "coverage": round(n_conf / nh, 4),
            "usable_fix_rate": round(float((eh <= TARGET_M).sum()) / nh, 4) if errs else 0.0,
            "false_fix_rate": round(float((eh > TARGET_M).sum()) / nh, 4) if errs else 0.0,
            "median_error_m": round(float(np.median(eh)), 2) if errs else None,
            "mean_error_m": round(float(np.mean(eh)), 2) if errs else None,
            "note": "unseen-region diagnostic; logged only, never scored",
        }

    if heatmap_dir:
        render_heatmap(area, data_dir, heat_points, heatmap_dir / f"heatmap_{area}.png",
                       bg_bucket=next(iter(eval_b)))
    return result


HEATMAP_MAX_PX = 1600


def render_heatmap(area, data_dir, points, out_path: Path, bg_bucket: str):
    # Background: the eval capture the dots were scored on, not a hardcoded
    # name — the photograph behind the dots is the one that was asked about.
    base = Image.open(area_dir(area, data_dir) / "relight" / f"{bg_bucket}.png").convert("RGB")
    scale = min(1.0, HEATMAP_MAX_PX / max(base.size))
    if scale < 1.0:
        base = base.resize((int(base.width * scale), int(base.height * scale)),
                           Image.BILINEAR)
    base = Image.eval(base, lambda p: p * 2 // 3)  # dim slightly for contrast
    draw = ImageDraw.Draw(base)
    r = 5
    for cx, cy, e in points:
        # Color bands scaled to this branch's 100 m milestone (main branch's
        # bands are 20/50 m, keyed to its 20 m target — same ratio, 5x scale).
        color = (60, 220, 60) if e < 100 else (240, 200, 40) if e < 250 else (230, 60, 60)
        x, y = cx * scale, cy * scale
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", required=True, help="comma-separated area names")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--heatmap-dir", default=None)
    ap.add_argument("--holdout", action="store_true",
                    help="required to score a holdout-role area")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    areas_cfg = load_areas()
    names = args.areas.split(",")
    for n in names:
        role = areas_cfg.get(n, {}).get("role")
        if role == "holdout" and not args.holdout:
            raise SystemExit(f"{n} is the blind holdout (§5); scoring it requires "
                             f"--holdout and its result must not drive keep/revert.")
        if role == "development" and args.holdout:
            raise SystemExit(f"--holdout runs must only contain holdout areas, got {n}")

    heat = Path(args.heatmap_dir) if args.heatmap_dir else None
    per_area = [score_area(n, Path(args.model_dir), data_dir, heat) for n in names]

    cell_scores = []
    for a in per_area:
        if a["gates"].get("failed"):
            cell_scores.append(FAIL_SCORE)
        else:
            cell_scores.extend(c["score"] for c in a["buckets"].values())
    primary = float(max(cell_scores)) if cell_scores else FAIL_SCORE

    out = {
        "kind": "holdout_check" if args.holdout else "development",
        "areas": per_area,
        "primary_mission_score": primary,
        "target_m": TARGET_M,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    label = "HOLDOUT" if args.holdout else "PRIMARY"
    print(f"{label} worst-case mission score: {primary:.4f} (0=every frame a usable fix, 1=abstains everywhere, 2=confidently wrong everywhere)")


if __name__ == "__main__":
    main()
