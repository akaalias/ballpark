"""Re-score every archived experiment on TODAY's ruler.

WHY THIS EXISTS
---------------
The project ran ~76 experiments across five eval/metric eras. Three of them
were scored against a broken evaluation (region holdout) and/or an error
statistic (median, then geometric mean) rather than the product requirement.
CLAUDE.md correctly voids those *verdicts*. But the work itself is real, and a
published progress chart that starts at experiment 1 of the current era makes
a two-week grind look like a four-experiment sprint.

This script does NOT "convert" old numbers with a fudge factor. It re-runs the
frozen, current `pipeline/score.py` against each old experiment's EXPORTED
`berlin.onnx` on the CURRENT eval set. Every point on the resulting chart is
therefore measured with the same ruler, because it literally is the same ruler.

WHAT IS AND IS NOT COMPARABLE
-----------------------------
Re-scored honestly means: same eval crops, same 100 m target, same 0.3
confidence threshold, same mission-score formula. It does NOT mean the old
models were *aiming* at this target — they were optimized against a different
metric on a different eval set, and eras 1-2 also trained on a region-holdout
split that excluded ~28% of the box, so they are being asked about ground they
were never shown. That is a real handicap and it is stated on the page rather
than corrected for: the point of the chart is "what would this model have been
worth to the aircraft", and the answer for a model that never saw a third of
the map is legitimately "not much".

RECOVERABILITY
--------------
  exact (rescored)  — an exported berlin.onnx still exists on disk
  exact (gated)     — the experiment failed a deployment gate or never trained;
                      it had no usable model then and has none now, so it plots
                      as a gated fail exactly as it always did
  derived           — pre-mission-metric era: the run dirs were deleted, but
                      that era's score.py logged `coverage` and
                      `hit_rate_at_target` over the confident subset, which
                      determines usable/false rates algebraically
  unrecoverable     — model gone and no rate data logged; plotted as a gap

Usage:
  .venv/bin/python -m autoresearch.rescore_history            # all eras
  .venv/bin/python -m autoresearch.rescore_history --dry-run  # plan only
"""

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DB = REPO_ROOT / "lineage_history.sqlite"
CACHE = REPO_ROOT / "state" / "rescore_cache"

FAIL = 1e9

# The five eras, oldest first. `db` is the archived lineage; `ruler` is what
# the loop was optimizing at the time (shown as era context on the chart);
# `eval_set` is what the questions were asked over.
ERAS = [
    dict(
        key="bootstrap",
        label="Bootstrap",
        db="archive/bootstrap-experiments.sqlite",
        ruler="worst-case median error (m)",
        eval_set="region holdout, 6 lighting buckets, 4 areas",
        note="Harness proving runs. The imagery itself changed mid-era: "
             "experiments 1-2 ran on 10 m/px Sentinel-class tiles, and from "
             "experiment 3 on it was 1 m/px orthophoto (commit c0a1bdf, "
             "2026-07-20 18:02 UTC). A 10 m/px model fed 1 m/px crops is not "
             "being asked the same question at all, so the first two are the "
             "only experiments in the whole project that cannot be placed on "
             "today's ruler.",
        rescorable=True,
        rescore_from_ts="2026-07-20T18:02",
    ),
    dict(
        key="main",
        label="Four areas, six lighting buckets",
        db="archive/pre-berlin-slim-wipe-experiments.sqlite",
        ruler="worst-case median error (m), target 20 m",
        eval_set="region holdout, 6 lighting buckets, 4 areas",
        note="The long grind: 62 experiments over ten days against a "
             "general-purpose, four-area, six-lighting-bucket problem. Scored "
             "on a region holdout that asked a memorization model to locate "
             "ground it had never been shown.",
        rescorable=True,
    ),
    dict(
        key="slim_v1",
        label="Berlin only",
        db="archive/pre-eval-v2-experiments.sqlite",
        ruler="worst-case median error (m), target 100 m",
        eval_set="region holdout, daytime only, Berlin only",
        note="Scope cut to one city and one lighting condition for iteration "
             "speed. Same broken region-holdout eval.",
        rescorable=True,
    ),
    dict(
        key="eval_v2",
        label="Viewpoint holdout",
        db="archive/pre-mission-metric-experiments.sqlite",
        ruler="worst-case geometric-mean error (m)",
        eval_set="viewpoint holdout, daytime only, Berlin only",
        note="The evaluation was fixed here: held-out viewpoints over mapped "
             "ground instead of held-out regions. The metric was still an "
             "error statistic. Run artifacts were not kept.",
        rescorable=False,  # run dirs deleted; rates derived from stored JSON
    ),
    dict(
        key="mission",
        label="Mission score",
        db="archive/pre-capture-holdout-experiments.sqlite",
        ruler="mission score — (1 - usable) + false, same photograph",
        eval_set="viewpoint holdout, daytime only, Berlin only, eval on the "
                 "training photograph",
        note="The metric became the product requirement itself and the "
             "approach worked: 0.040, 96.5% usable. But the eval crops came "
             "from the very raster the model trained on. On 2026-09-28 the "
             "champion was scored on other open photographs of the same "
             "ground (Berlin TrueDOP 2022/2023/2024) and on +20 brightness "
             "of its own raster: 1.78-1.97 and 1.45. It had memorised one "
             "photograph's pixels, not the city. Re-scored here on the "
             "held-out photograph like every earlier era.",
        rescorable=True,
    ),
    dict(
        key="capture_holdout",
        label="Held-out photograph",
        db="experiments.sqlite",
        ruler="mission score — (1 - usable) + false, unseen photograph",
        eval_set="viewpoint holdout on a photograph never used in training "
                 "(Berlin TrueDOP 2024); training on three other captures",
        note="The eval capture is a photograph the model has never seen. "
             "Same viewpoint holdout, same product metric; the new axis is "
             "'the photograph is new' on top of 'the view is new'. Scores "
             "from here on are native.",
        rescorable=True,
        native=True,
    ),
]


def rescore_one(models_dir: Path, out_json: Path) -> dict | None:
    """Run the frozen scorer against one exported model directory."""
    if out_json.exists():                     # cached from an earlier pass
        try:
            return json.loads(out_json.read_text())
        except json.JSONDecodeError:
            out_json.unlink()
    if not (models_dir / "berlin.onnx").exists():
        return None
    out_json.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(REPO_ROOT / ".venv" / "bin" / "python"), "-m", "pipeline.score",
         "--areas", "berlin", "--model-dir", str(models_dir),
         "--out", str(out_json)],
        cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0 or not out_json.exists():
        print(f"    scorer failed: {proc.stderr.strip()[-200:]}", file=sys.stderr)
        return None
    return json.loads(out_json.read_text())


# --- what each experiment cost -------------------------------------------
#
# Claude Code writes its own accounting into every runs/<id>/agent_*.json:
# `total_cost_usd` for the invocation and `modelUsage[model].costUSD` broken
# down per model. So this is READ, not reverse-engineered from a price list —
# and it stays correct as prices change, because it is what was recorded.
#
# IMPORTANT: the agents ran on the user's claude.ai Max subscription, not on
# API billing. These figures are therefore the *equivalent* API cost of the
# tokens — what this research would have cost billed per token — not money
# that left the account. Every surface that shows them must say so.
AGENT_STAGES = ("design", "impl", "figure", "result")

# The rented-GPU window. Billed by wall clock regardless of phase, so an
# experiment's compute cost is its whole duration at the hourly rate. Before
# and after this window the loop ran on hardware with ~$0 marginal cost (a
# laptop, then Modal credits that were a fixed prepaid grant rather than a
# per-experiment charge).
POD_USD_PER_HR = 0.69
POD_ERA_START = "2026-07-21"
POD_ERA_END = "2026-07-23T12:00:00"


def agent_costs(artifacts_dir: str | None) -> dict:
    """Per-stage and per-model agent cost for one experiment's run directory."""
    out = {"by_stage": {}, "by_model": {}, "total": 0.0, "stages_found": 0}
    if not artifacts_dir:
        return out
    for stage in AGENT_STAGES:
        f = REPO_ROOT / artifacts_dir / f"agent_{stage}.json"
        if not f.exists():
            continue
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        c = d.get("total_cost_usd")
        if c is None:
            continue
        out["by_stage"][stage] = round(float(c), 4)
        out["total"] += float(c)
        out["stages_found"] += 1
        for model, mu in (d.get("modelUsage") or {}).items():
            out["by_model"][model] = round(
                out["by_model"].get(model, 0.0) + (mu.get("costUSD") or 0.0), 4)
    out["total"] = round(out["total"], 4)
    return out


def gpu_cost(ts: str | None, duration_s: float | None, kind: str) -> float:
    if kind == "holdout_check" or not duration_s or not ts:
        return 0.0
    if ts < POD_ERA_START or ts >= POD_ERA_END:
        return 0.0
    return round(duration_s / 3600 * POD_USD_PER_HR, 4)


def berlin_cell(metrics: dict) -> dict | None:
    for a in metrics.get("areas", []):
        if a.get("area") == "berlin":
            b = a.get("buckets") or {}
            return next(iter(b.values()), None)
    return None


def derive_rates(old_metrics: dict) -> dict | None:
    """Reconstruct usable/false rates from an eval_v2-era record.

    That era's score.py logged, over the CONFIDENT subset only:
        hit_rate_at_target = (errors < TARGET_M).mean()
    and separately `coverage` over all frames. So, over all frames:
        usable = coverage * hit_rate ;  false = coverage - usable
    which is exactly what the mission score is built from. The eval set and
    target are identical to today's, so this is a derivation, not an estimate.
    """
    cell = berlin_cell(old_metrics)
    if not cell:
        return None
    cov, hit = cell.get("coverage"), cell.get("hit_rate_at_target")
    if cov is None or hit is None:
        return None
    usable = cov * hit
    false = cov - usable
    return {
        "coverage": round(cov, 4),
        "usable_fix_rate": round(usable, 4),
        "false_fix_rate": round(false, 4),
        "abstain_rate": round(1.0 - cov, 4),
        "mission_score": round((1.0 - usable) + false, 6),
        "median_error_m": cell.get("median_error_m"),
        "geomean_error_m": cell.get("geomean_error_m"),
        "mean_error_m": cell.get("mean_error_m"),
        "p10_error_m": cell.get("p10_error_m"),
        "n_eval": cell.get("n_eval"),
    }


SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    seq            INTEGER PRIMARY KEY,  -- global order across all eras
    era            TEXT NOT NULL,        -- era key
    era_label      TEXT NOT NULL,
    era_index      INTEGER NOT NULL,     -- 0-based era ordinal
    src_id         INTEGER NOT NULL,     -- id within that era's DB
    ts             TEXT,
    kind           TEXT,
    kept           INTEGER,              -- keep/revert AS DECIDED AT THE TIME
    title          TEXT,
    category       TEXT,
    hypothesis     TEXT,
    method         TEXT,
    conclusion     TEXT,
    eli5           TEXT,
    arch_svg       TEXT,                  -- the agent-drawn architecture figure
    arch_json      TEXT,                  -- pre-registered {"stages":[...]}
    init_strategy  TEXT,
    artifacts_dir  TEXT,
    git_commit     TEXT,
    duration_s     REAL,
    -- cost: agent spend is the EQUIVALENT API cost of tokens actually used
    -- (recorded by Claude Code); the work ran on a Max subscription.
    cost_agent     REAL,
    cost_gpu       REAL,
    cost_total     REAL,
    cost_by_stage  TEXT,                  -- {"design":2.81,"impl":0.9,...}
    cost_by_model  TEXT,                  -- {"claude-fable-5":2.81,...}
    cost_stages_n  INTEGER,               -- how many agent records survived
    -- what the loop optimized at the time, in that era's own units
    era_metric     REAL,
    era_metric_kind TEXT,
    -- TODAY's ruler
    mission_score  REAL,
    usable_fix_rate REAL,
    false_fix_rate REAL,
    abstain_rate   REAL,
    coverage       REAL,
    median_error_m REAL,
    geomean_error_m REAL,
    p10_error_m    REAL,
    provenance     TEXT NOT NULL,        -- rescored|native|derived|gated|unrecoverable
    rescored_json  TEXT
);
CREATE TABLE IF NOT EXISTS eras (
    era_index INTEGER PRIMARY KEY,
    key       TEXT NOT NULL,
    label     TEXT NOT NULL,
    ruler     TEXT,
    eval_set  TEXT,
    note      TEXT,
    seq_start INTEGER,
    seq_end   INTEGER,
    ts_start  TEXT,
    ts_end    TEXT
);
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(OUT_DB))
    args = ap.parse_args()

    rows, seq = [], 0
    era_bounds = []

    for era_index, era in enumerate(ERAS):
        db_path = REPO_ROOT / era["db"]
        if not db_path.exists():
            print(f"!! missing {era['db']}", file=sys.stderr)
            continue
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        src = conn.execute("SELECT * FROM experiments ORDER BY id ASC").fetchall()
        print(f"\n== era {era_index} {era['label']}: {len(src)} experiments")
        seq_start = seq + 1

        for r in src:
            seq += 1
            d = dict(r)
            try:
                old_metrics = json.loads(d.get("metrics_json") or "{}")
            except json.JSONDecodeError:
                old_metrics = {}
            era_metric = d.get("primary_metric")

            row = dict(
                seq=seq, era=era["key"], era_label=era["label"],
                era_index=era_index, src_id=d["id"], ts=d.get("ts"),
                kind=d.get("kind"), kept=d.get("kept"), title=d.get("title"),
                category=d.get("category"), hypothesis=d.get("hypothesis"),
                method=d.get("method"), conclusion=d.get("conclusion"),
                eli5=d.get("eli5"), arch_svg=d.get("arch_svg"),
                arch_json=d.get("arch_json"),
                init_strategy=d.get("init_strategy"),
                artifacts_dir=d.get("artifacts_dir"),
                git_commit=d.get("git_commit"), duration_s=d.get("duration_s"),
                era_metric=era_metric, era_metric_kind=era["ruler"],
                cost_agent=None, cost_gpu=None, cost_total=None,
                cost_by_stage=None, cost_by_model=None, cost_stages_n=None,
                mission_score=None, usable_fix_rate=None, false_fix_rate=None,
                abstain_rate=None, coverage=None, median_error_m=None,
                geomean_error_m=None, p10_error_m=None,
                provenance="unrecoverable", rescored_json=None,
            )

            ac = agent_costs(d.get("artifacts_dir"))
            gc = gpu_cost(d.get("ts"), d.get("duration_s"), d.get("kind") or "")
            row.update(
                cost_agent=ac["total"], cost_gpu=gc,
                cost_total=round(ac["total"] + gc, 4),
                cost_by_stage=json.dumps(ac["by_stage"]) if ac["by_stage"] else None,
                cost_by_model=json.dumps(ac["by_model"]) if ac["by_model"] else None,
                cost_stages_n=ac["stages_found"])

            # --- the current era needs no re-scoring: it IS the ruler ---
            if era.get("native"):
                cell = berlin_cell(old_metrics)
                row["provenance"] = "native"
                if era_metric is not None and era_metric >= FAIL:
                    row["provenance"] = "gated"
                elif cell:
                    row.update(
                        mission_score=cell.get("mission_score"),
                        usable_fix_rate=cell.get("usable_fix_rate"),
                        false_fix_rate=cell.get("false_fix_rate"),
                        abstain_rate=cell.get("abstain_rate"),
                        coverage=cell.get("coverage"),
                        median_error_m=cell.get("median_error_m"),
                        geomean_error_m=cell.get("geomean_error_m"),
                        p10_error_m=cell.get("p10_error_m"))
                rows.append(row)
                continue

            # --- a periodic blind holdout check measured HAMBURG (§5), not
            # Berlin. There is no Berlin mission score for it and inventing
            # one would be worse than leaving the gap, so it keeps its place
            # in the sequence and carries no score.
            if d.get("kind") == "holdout_check":
                row["provenance"] = "holdout"
                rows.append(row)
                continue

            # --- a gated fail had no deployable model then and has none now ---
            if era_metric is not None and era_metric >= FAIL:
                row["provenance"] = "gated"
                rows.append(row)
                continue

            # An era can be only PARTLY rescorable: the bootstrap era spans
            # the 10 m/px -> 1 m/px imagery change, and a model trained at one
            # ground sample distance is not answerable on crops at the other.
            from_ts = era.get("rescore_from_ts")
            in_window = not from_ts or (d.get("ts") or "") >= from_ts

            models = REPO_ROOT / (d.get("artifacts_dir") or "") / "models"
            if era["rescorable"] and in_window and (models / "berlin.onnx").exists():
                if args.dry_run:
                    row["provenance"] = "rescored"
                    rows.append(row)
                    print(f"   #{d['id']:>3}  would rescore  {models}")
                    continue
                out_json = CACHE / f"{era['key']}_{d['id']}.json"
                m = rescore_one(models, out_json)
                cell = berlin_cell(m) if m else None
                if cell:
                    row.update(
                        provenance="rescored",
                        mission_score=cell.get("mission_score"),
                        usable_fix_rate=cell.get("usable_fix_rate"),
                        false_fix_rate=cell.get("false_fix_rate"),
                        abstain_rate=cell.get("abstain_rate"),
                        coverage=cell.get("coverage"),
                        median_error_m=cell.get("median_error_m"),
                        geomean_error_m=cell.get("geomean_error_m"),
                        p10_error_m=cell.get("p10_error_m"),
                        rescored_json=json.dumps(m))
                    print(f"   #{d['id']:>3}  {row['mission_score']:.4f}  "
                          f"usable={row['usable_fix_rate']:.3f} "
                          f"false={row['false_fix_rate']:.3f}  {d['title'][:48]}")
                rows.append(row)
                continue

            # --- artifacts gone: derive the rates from the stored record ---
            derived = derive_rates(old_metrics)
            if derived:
                row.update(provenance="derived", **{
                    k: v for k, v in derived.items() if k in row})
                print(f"   #{d['id']:>3}  {row['mission_score']:.4f}  (derived)"
                      f"  {d['title'][:48]}")
            elif not in_window:
                row["provenance"] = "incomparable"
                print(f"   #{d['id']:>3}  INCOMPARABLE (pre-1 m/px imagery)"
                      f"  {d['title'][:40]}")
            else:
                print(f"   #{d['id']:>3}  UNRECOVERABLE  {d['title'][:48]}")
            rows.append(row)

        era_bounds.append(dict(
            era_index=era_index, key=era["key"], label=era["label"],
            ruler=era["ruler"], eval_set=era["eval_set"], note=era["note"],
            seq_start=seq_start, seq_end=seq,
            ts_start=src[0]["ts"] if src else None,
            ts_end=src[-1]["ts"] if src else None))
        conn.close()

    if args.dry_run:
        print(f"\ndry run: {len(rows)} rows, no DB written")
        return

    out = Path(args.out)
    out.unlink(missing_ok=True)
    conn = sqlite3.connect(out)
    conn.executescript(SCHEMA)
    cols = list(rows[0])
    conn.executemany(
        f"INSERT INTO history ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        [[r[c] for c in cols] for r in rows])
    ecols = list(era_bounds[0])
    conn.executemany(
        f"INSERT INTO eras ({','.join(ecols)}) VALUES ({','.join('?' * len(ecols))})",
        [[e[c] for c in ecols] for e in era_bounds])
    conn.commit()

    n = {}
    for r in rows:
        n[r["provenance"]] = n.get(r["provenance"], 0) + 1
    print(f"\nwrote {out} — {len(rows)} experiments across {len(era_bounds)} eras")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(n.items())))


if __name__ == "__main__":
    main()
