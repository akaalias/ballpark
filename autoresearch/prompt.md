# Autoresearch experiment — the design stage

You are one experiment of an autonomous research loop for UAV geolocalization.
Read `CLAUDE.md`'s "CAPTURE-HOLDOUT ERA" section first, then "SCOPE OVERRIDE",
then §3, §4 and §6. The harness (loop.sh) will train, score, log, and
keep/revert AFTER you exit — you only design the experiment.

## THE GOAL: MEMORIZE ONE BOUNDING BOX — THE GROUND, NOT ONE PHOTOGRAPH OF IT

This branch builds a **visual memory of Berlin**, not a model that reasons
about aerial imagery in general. One model, one bbox, deployed only over that
bbox. It is *supposed* to know this specific city by heart. You are not trying
to generalize to other places or to unmapped ground.

**But it must know the CITY, not one picture of the city.** This era exists
because the previous champion did not. Measured on 2026-09-28, same eval
crops, same grid, only the pixels changed:

| what the 0.040 champion was scored on | mission | usable | false fix | median |
|---|---|---|---|---|
| the photograph it trained on (Brandenburg DOP20) | 0.040 | 96.5% | 0.5% | 27 m |
| the same raster, +20 added to every pixel value | 1.453 | 1.5% | 46.8% | 3.1 km |
| the same raster, 1 px Gaussian blur | 1.271 | 14.2% | 41.2% | 1.3 km |
| the same raster, shifted 3 px | 0.040 | 96.8% | 0.8% | 27 m |
| Berlin TrueDOP 2022 / 2023 / 2024 (other open surveys, same ground) | 1.87 / 1.78 / 1.87 | ≤1.2% | 79–87% | 3.3–3.9 km |

A random guess over the box misses by 3.6 km. The model had memorised one
photograph's pixel intensities and fine texture (rotation was its only
augmentation); any other exposure of the same ground was a foreign image, and
its confidence head — trained on the same photograph — stayed 97% confident
while wrong. 390 confident predictions on a foreign photograph collapsed onto
40 map cells, two of them absorbing 225. Colour-matching the foreign
photograph to the training one recovered abstention (coverage 97% → 61%) but
not accuracy (median still 3.1 km): intensity is what the confidence keyed on;
the fix itself rode texture and shadow, which change between surveys.

**So the eval now asks about a photograph the model has never seen.** Berlin
has four independent open captures on one grid (`areas.yaml`): the model
trains on three (Brandenburg DOP20 summer canopy, TrueDOP 2022 leaf-off,
TrueDOP 2023 early spring) and is scored on the fourth (TrueDOP 2024), which
it never sees in any form. Season, sun angle, shadows, cars, construction and
colour balance all differ between captures. The three training captures are
also scored, as `train_capture_diagnostics` in metrics.json — LOGGED ONLY,
never part of the primary — so every experiment records its
memorisation-vs-ground gap. Read them: a design that scores 0.05 on the
training photographs and 1.9 on the held-out one has learned the pictures,
not the place.

What this implies for design, without prescribing the answer: whatever
survives a change of photograph must be learned in TRAINING (the aircraft's
camera is a fifth photograph nobody has). Decode-time tricks cannot reach it.
Photometric robustness alone is not enough — colour matching did not recover
accuracy — but it is clearly part of it. Capacity spent on pixel-exact
fingerprints of three photographs is capacity wasted; capacity spent on the
structure they share is the point.

**How the split works, because it determines what a good design looks like:**

* **Training covers ALL of Berlin** — every lattice position, whole raster.
  The model is shown the entire area it must memorize.
* **Eval holds out VIEWPOINTS, not REGIONS.** Every eval frame stands on
  ground that WAS in training, but is framed 11–17 m off the nearest training
  vantage and at its own rotation. Mapped ground, novel view — exactly what an
  aircraft over a mapped city faces.
* So the task is: **recall a memorized place from a viewpoint you have not
  seen.** It is NOT: infer the location of ground you were never shown.
* A small 1-in-32-block region IS genuinely untrained, scored separately and
  logged only. It can never affect keep/revert. Ignore it when designing;
  it exists to reveal whether the model has spatial structure at all.

This matters because the split used to hold out regions, which left 28% of
Berlin untrained and put 100% of eval questions on never-seen ground — an
unanswerable task for a memorization model, and the likely reason ~60 earlier
experiments plateaued. Do not design as if that were still true.

**Scope:** ONE locale (Berlin), ONE lighting condition (raw daytime imagery,
no synthetic relighting — the relighting machinery is disabled on this branch),
FOUR photographs of it (three for training, one held out — see above). There
is no synthetic-lighting robustness to reason about and no other area's
texture to generalize to; there IS cross-photograph robustness, and it is the
whole score.

## THE METRIC IS THE PRODUCT REQUIREMENT, NOT AN ERROR STATISTIC

The aircraft takes a vision fix every 5–10 s and it is its ONLY drift
correction. Per frame exactly three things can happen:

| outcome | meaning |
|---|---|
| confident **and** within 100 m | **USABLE FIX** — this is the product |
| not confident | abstains — safe, it waits for the next frame |
| confident **but** outside 100 m | **FALSE FIX** — dangerous, it injects a wrong position into navigation |

    mission_score = (1 - usable_fix_rate) + false_fix_rate      [MINIMIZED]

* **0.0** = every frame a usable fix. **1.0** = abstains everywhere.
  **2.0** = confidently wrong everywhere — *strictly worse than silence*,
  because a confidently wrong fix corrupts navigation while an abstention
  merely costs time.
* Therefore: **making the model honest is as valuable as making it
  accurate.** Converting a confident-but-wrong frame into an abstention
  improves the score. Converting an abstention into a correct confident fix
  improves it twice as much. Both are real progress; design for either.
* A cell whose coverage falls below 0.2 scores FAIL — you cannot pass by
  abstaining on everything.

**Do not optimize an error statistic.** Median, geometric mean, p10, p25 are
all still logged, and you may read them to understand *why* something worked,
but none of them is the target. Each was tried as the primary and each
rewarded something the aircraft does not want — the median rewarded a model
that guesses the map centre, and it reverted the first experiment that had
actually begun memorizing. If your design improves the geometric mean while
`usable_fix_rate` stays flat, it has not helped the product.

## Your job, in order

0. **Skim the library (optional input).** `autoresearch/library.md` holds
   the human researcher's inspiration notes. They do not fix your answer
   and you are free to ignore them — pick an entry up only when it
   genuinely fits your read of the history, and if you build on one, say
   so in your hypothesis.

1. **Review the research history.** Two sources, treated differently.

   (a) THIS era — verdicts are valid, on this ruler:
   `sqlite3 experiments.sqlite "SELECT id, title, category, hypothesis, expected_outcome, result, conclusion, primary_metric, kept FROM experiments ORDER BY id DESC LIMIT 15;"`
   Note which hypotheses were supported/refuted. Do not repeat a refuted
   experiment without a materially new angle. Also read each row's
   `metrics_json` -> `train_capture_diagnostics`: the gap between the
   training photographs and the held-out one is the thing to close.

   (b) EARLIER eras — what was tried, with the verdicts VOID. 83 experiments
   over five earlier evaluation regimes have been re-scored on today's ruler:
   `sqlite3 lineage_history.sqlite "SELECT era_label, src_id, title, category, hypothesis, init_strategy, mission_score, usable_fix_rate, false_fix_rate, abstain_rate, provenance FROM history WHERE kind != 'holdout_check' ORDER BY mission_score ASC;"`
   Read this as a MAP OF THE SEARCH SPACE, not as a list of refutations:
   every one of those experiments was kept or reverted under an evaluation
   that asked about the training photograph (or about never-seen ground),
   and its stored `conclusion` is about that question, not this one — which
   is why the query above deliberately omits it. Their `mission_score` here
   is the re-measurement on the held-out photograph, and none is good: the
   best is 1.23, reached by abstaining on 77% of frames, and the 0.040
   same-photograph champion sits at 1.87. So an idea from an earlier era is
   neither validated nor refuted for THIS question — contrastive
   pretraining, learned relighting, pretrained trunks, dense voting and the
   rest are all open again. A row's `provenance` tells you how it was
   measured (`rescored` = its exported model re-run today; `gated` = never
   had a working model; `incomparable` = cannot be placed on this ruler).

   **Plateau rule (advisory only on this branch):** the harness's automatic
   pivot enforcement (mandatory-pivot preamble, backbone-carry rejection) is
   disabled on berlin-slim — see CLAUDE.md "BRANCH OVERRIDE". Nothing will
   force or reject a pivot below. That said, the underlying discipline still
   applies by judgment: if three or more consecutive experiments were
   reverted, don't attempt another variation of the last refuted mechanism.
   Either pick a design family absent from the history (pretrained init,
   training-scale/coverage, a different coordinate parameterization, capacity
   allocation, decode resolution, …) or attack the bottleneck the refuted hypotheses
   jointly point at — checking, via `arch_json`
   (`SELECT arch_json FROM experiments WHERE kind='development' ORDER BY id
   DESC LIMIT 10;`), which stage names never carry `"changed": true`, since a
   losing streak is usually one unquestioned stage (trunk / descriptor /
   decode) with everything else churning around it, not a missing family.

2. **Design ONE focused experiment** — proper experiment design, pre-registered
   before you touch code. Write it to `runs/pending_experiment.json`:
   ```json
   {
     "title": "one-line name",
     "category": "architecture|loss|augmentation|relighting|training|quantization|other",
     "hypothesis": "what you believe is limiting the metric and why this change addresses it",
     "method": "the ONE focused change, concretely (files, mechanism)",
     "expected_outcome": "predicted effect on mission_score, and on usable_fix_rate / false_fix_rate specifically, quantified if possible",
     "init_strategy": "from-scratch | pretrained:<name>",
     "eli5": "2-4 sentences for a smart non-ML reader: what you changed and why it might help, in everyday language — analogies welcome, zero jargon",
     "architecture": {"stages": [
       {"name": "Camera frame", "detail": "128×128 px daytime crop, Berlin only; trained on three survey photographs, scored on a fourth it never saw", "changed": false},
       {"name": "Feature extractor", "detail": "plain-language description", "changed": false},
       {"name": "…", "detail": "…", "changed": true}
     ]}
   }
   ```

   Do NOT draw the architecture figure here — that is a separate agent's
   job, run later, only for experiments that actually reach training (see
   `autoresearch/prompt_figure.md`). Spending time perfecting an SVG for a
   design that might get rejected before it ever runs was wasted work; this
   stage's only output is the decision itself.

   `eli5` and `architecture` feed the human-facing gallery. `architecture.stages`
   is the model's inference path left-to-right, camera frame → (lat, lon,
   confidence) output — one box per stage, plain-language `detail`,
   `"changed": true` ONLY on the stages this experiment touches. Reuse the
   previous experiment's stage names verbatim wherever a stage is unchanged
   (check `SELECT arch_json FROM experiments WHERE arch_json IS NOT NULL ORDER BY id DESC LIMIT 1;`) —
   stage names must stay consistent across experiments since the later
   figure-drawing stage and plateaucheck both compare/track them by name.
   A change that only affects training (loss, augmentation, schedule) keeps
   the inference stages unchanged and adds one final stage with
   `"train_only": true` describing the training signal.

3. **Write the implementation brief.** You do NOT edit `model/` yourself —
   a separate implementation agent applies your design, seeing only the
   current code plus what you pre-registered. Add one more field to
   `runs/pending_experiment.json`:
   ```
   "implementation_brief": "exact file-level instructions: which functions/
   blocks in model/model.py and model/train.py change and how, what stays
   untouched, and every contract to preserve"
   ```
   Be precise enough that a competent engineer with no other context
   implements it in one pass. Always restate the fixed contracts:
   `train.py`'s CLI and the ONNX export contract in `model/model.py`'s
   docstring — the frozen scorer depends on them.

## Hard rules

- Edit ONLY `runs/pending_experiment.json`. You never edit `model/` (the
  implementation stage does) and files listed in `/FROZEN` are off-limits;
  the harness hard-reverts any change to them.
- ONE focused change per experiment — if you can't describe it in one sentence,
  it's too big. Prefer architectural/procedural novelty over hyperparameter
  nudges (§3): changing a learning rate is a weak experiment; changing the
  coordinate parameterization, loss family, capacity allocation, decode
  resolution, or model topology is a strong one.
- Do not run training yourself; the harness does that.
- Stay within the deployment gates: exported ONNX ≤ 4 MiB per area, host
  latency proxy ≤ 250 ms (see pipeline/score.py).
- Keep one experiment tractable to train. This branch trains Berlin only, on
  a laptop GPU, and every epoch now covers THREE photographs (the training
  loop in `model/train.py` iterates `buckets(meta, "train")`, so an epoch is
  ~3x the old one); an inherently expensive per-sample mechanism (e.g. many-round
  iterative solves over thousands of votes per crop) can still push a round
  into hours. Budget the per-crop cost so a round finishes in a sensible
  wall-time — an idea that can't be evaluated in a round can't be kept.
