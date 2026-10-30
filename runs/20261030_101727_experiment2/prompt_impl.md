# Autoresearch experiment — implementation stage

You are the implementation stage of an autonomous research loop that builds a
**visual memory of one bounding box** (Berlin, daytime imagery only — see
CLAUDE.md "BRANCH OVERRIDE"). Training covers all of Berlin; evaluation holds
out VIEWPOINTS (frames 11-17 m off the nearest training vantage), not regions.
The optimized score is the **mission score** = (1 - usable_fix_rate) +
false_fix_rate, minimized: a frame is a USABLE FIX when the model is confident
AND within 100 m, an abstention is safe, and a confident-but-wrong frame is a
FALSE FIX that is strictly worse than silence. Preserve the confidence output's
meaning — honest abstention is part of the product, not a failure mode. A design agent has pre-registered ONE experiment
in `runs/pending_experiment.json`; the harness will train, score, log, and
keep/revert after you exit. Your only job is a faithful implementation.

## Your job, in order

1. Read `runs/pending_experiment.json` — especially `method` and
   `implementation_brief` — then the current `model/model.py` and
   `model/train.py`.
2. Implement EXACTLY the described change in `model/` — nothing more. If
   the brief conflicts with the code it describes, follow the brief's
   intent with the smallest faithful adaptation; do not invent
   improvements, refactors, or extra changes of your own.
3. Preserve the fixed contracts: `train.py`'s CLI
   (`--area --out-dir --data-dir --epochs --max-crops-per-bucket --seed`),
   the data contract — training reads ONLY the buckets returned by
   `pipeline.common.buckets(meta, "train")` (`relight/<bucket>.png`); the
   eval capture (`buckets(meta, "eval")`) must never be opened by anything
   under `model/`, in any form, or the experiment is void —
   the ONNX export contract in `model/model.py`'s docstring, exported ONNX
   ≤ 4 MiB per area, host latency proxy ≤ 250 ms (see pipeline/score.py).
4. Syntax check when done:
   `.venv/bin/python -c "import model.train, model.model"`

## Hard rules

- Edit ONLY files under `model/`. Never edit `runs/pending_experiment.json`,
  anything listed in `/FROZEN`, or anything else.
- Do not run training yourself; the harness does that.
- Python on this host is 3.11 — no 3.12-only syntax (e.g. nested
  same-quote f-strings).
- If the design's `init_strategy` is `pretrained:<name>`, load the weights
  from a permissively-licensed source at TRAIN time — e.g. a torchvision
  `weights=` backbone (BSD). The GPU trainer has network, so the download
  happens once during training and the weights get baked into the exported
  ONNX; do NOT rely on a local weight file under `model/pretrained/` (the
  remote trainer receives only `model.py` + `train.py`, not vendored
  binaries), and do NOT add any network fetch to the inference/export path —
  on-device inference must stay fully offline (§2).
