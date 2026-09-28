# UAV Low-Light Geolocalization — Autoresearch Bootstrap Spec

## THE CAPTURE-HOLDOUT ERA — branch `capture-holdout`, opened 2026-09-28

**This is a testing branch. `main` is untouched and still publishes the
berlin-slim result below.** Nothing here merges until the era has produced
something better than its own baseline on its own ruler; only then is the
question of `main` and the published report reopened. The state block below
("STATE AS OF 2026-07-31") describes `main`, and stays true for `main`.

### Why the era exists — the founding measurement

The 0.040 champion was asked to locate frames from **other photographs of the
same ground**, warped onto the exact grid it trained on (alignment ≤ 1 px):

| scored on | mission | usable | false fix | median |
|---|---|---|---|---|
| its training raster (Brandenburg DOP20) | 0.040 | 96.5% | 0.5% | 27 m |
| Berlin TrueDOP 2024 (open, dl-de/zero) | 1.873 | 0.0% | 87.2% | 3,315 m |
| Berlin TrueDOP 2023 | 1.776 | 1.2% | 78.8% | 3,510 m |
| Berlin TrueDOP 2022, leaf-off | 1.866 | 0.2% | 86.8% | 3,936 m |
| Google satellite (local diagnostic only, never in the repo) | 1.971 | 0.2% | 97.2% | 3,551 m |
| its own raster, **+20 on every pixel** | 1.453 | 1.5% | 46.8% | 3,076 m |
| its own raster, 1 px Gaussian blur | 1.271 | 14.2% | 41.2% | 1,326 m |
| its own raster, shifted 3 px | 0.040 | 96.8% | 0.8% | 27 m |

A random guess over the box misses by 3.6 km. The champion memorised one
photograph's pixel values (rotation was the only augmentation), and its
confidence head, trained on the same photograph, stayed ~97% confident while
wrong; 390 confident predictions on a foreign photograph collapsed onto 40 map
cells. Colour-matching the foreign photograph recovered abstention (coverage
97% → 61%) but not accuracy (median still 3.1 km). **The same-photograph eval
could never see any of this, and the loop only fixes what its ruler
measures** — a photometric-augmentation experiment scored on the old eval
would come in a hair above 0.040 and be reverted. Hence a new era, not an
experiment. Full record: memory `google-tiles-domain-gap-probe`,
`berlin-truedop-years`; scratch artifacts in `~/scratch/gtest/` are working
files, not the research record.

### What the era changes (all frozen, all committed in the opening commit)

- **Captures** (`pipeline/common.py`): an area may carry several independent
  photographs of the same bbox on one grid. `areas.yaml` gives Berlin four:
  `bb_dop20` (the champion's raster, registered unchanged), `truedop_2022`,
  `truedop_2023` (all `role: train`) and **`truedop_2024`, `role: eval` — held
  out of training entirely.** Sources: `pipeline/sources.yaml`, three Berlin
  TrueDOP WMS entries (`auto_pick: false`, so the generic path is unchanged).
- **Layout**: `data/<area>/captures/<name>.tif` (fetch), `relight/<bucket>.png`
  per capture × lighting bucket; with one lighting bucket the bucket IS the
  capture name. `relight/asis.png` and the bucket name `asis` are gone for
  Berlin (areas without a captures list still get a single `asis`).
  `reference.tif` is kept on disk but nothing reads it when captures exist.
- **Score** (`pipeline/score.py`): the primary is the worst over the **eval**
  buckets — for Berlin, exactly one: `truedop_2024`. The training captures
  are scored too under `train_capture_diagnostics`, logged only, so every
  experiment records its memorisation-vs-ground gap. `SCORE_DIAGNOSTICS=0`
  skips them (bulk re-scoring). The heatmap and the region-holdout diagnostic
  use the eval capture.
- **Training contract** (`model/train.py`, agent-editable, and
  `autoresearch/prompt_impl.md`): training reads only
  `buckets(meta, "train")`. **Nothing under `model/` may open the eval
  capture in any form** — that would be leakage, and the experiment is void.
  An epoch now covers three photographs, so training is ~3× the old
  wall-clock (~45 min on the M1 at default `EPOCHS`).
- **Lineage**: `experiments.sqlite` wiped; the mission era is archived as
  `archive/pre-capture-holdout-experiments.sqlite` and re-scored on the new
  ruler by `rescore_history.py` (era key `mission`, no longer native; new era
  `capture_holdout` is native). `state/rescore_cache/` was cleared because
  the ruler changed. The overview fix from `f7ce820` (best row across all
  eras, from `lineage_history.sqlite`) is ported, so the front page cannot
  reset to this era's baseline.
- **Design brief** (`autoresearch/prompt.md`): carries the founding table and
  the split; it deliberately does not prescribe the fix.

### Rules for this branch

- **Never put Google imagery anywhere in the repo or the pipeline** (§4). It
  was a one-off local diagnostic; the legitimate second photographs are the
  open TrueDOP years.
- Eval capture = `truedop_2024`, fixed for the era. Do not rotate it, do not
  add it to training "just to see". Changing it is a new era.
- The loop pushes to `origin/capture-holdout`. `state/research_status` is
  `live` here; `main`'s is `finished`.
- Before merging: re-run `rescore_history`, re-render, then the
  `freeze_site_assets` routine — the heatmaps now show the 2024 photograph.

### Expected shape of the era

The baseline seed (experiment 1, `SKIP_AGENT=1`) trains the champion's
architecture on three captures and scores on the fourth. Expect it to be bad
on the held-out photograph and near 0.04 on the training ones; the gap
between those two numbers is what the loop is now asked to close.

---

## THE PRIGNITZ DETOUR — opened and rewound, 2026-07-31 → 08-01

On the evening of 31 July a rural era was opened on Prignitz (`AREAS=prignitz`).
It ran three experiments in four hours. **On 1 August the repository was rewound
to `db470da`, the commit before the era opened**, and its rows were removed from
`experiments.sqlite`. Everything below this section — the berlin-slim
configuration, the 0.040 champion, `AREAS=berlin` — is therefore current again,
exactly as it was. Do not read the state block below as stale; it is the state.

**Do not re-open the rural era as though it were a fresh idea, and do not
re-derive what it already measured.** What it established, in order:

- **The Berlin champion applied unchanged to Prignitz scores 0.113** (Berlin
  0.040). The loss is almost entirely **false fixes**: 3.75% vs 0.50%, a 7.5x
  rise, while usable (92.5% vs 96.5%), abstention (3.8% vs 3.0%) and median
  error (29.5 m vs 27.1 m) barely move. Precision is intact; calibration is
  not. Belief pooling defends against mass split across *adjacent* cells and
  against mass scattered across *distant* ones, but not against mass landing
  confidently on one wrong field that looks exactly like the right one.
- **The rural confusions are rotation-stable — refuted, and this is the useful
  part.** Experiment 2 averaged the belief map across the frame's four 90°
  rotations before decoding, on the theory that a look-alike match rides
  view-specific micro-texture. False fixes went *up* (3.75% → 4.25%), usable
  down (92.5% → 92.0%), mission 0.113 → 0.123. Two fields that fool the model
  fool it from every heading, so this is not a viewing-angle artefact and no
  decode-time consensus can reach it. **Whatever fixes it must act during
  training.** Do not re-run a decode-side consensus variant.
- **Untried, and the natural next move if it is reopened:** continuous-vantage
  training — jitter each training framing ±12 px so vantages tile the ground
  instead of standing on the 24 m lattice, since every eval view stands
  8–16 px off-lattice. Designed and implemented; never trained.
- **One model over Berlin + Prignitz** was proposed and dropped the same
  evening before any experiment ran. §1's one-model-per-bbox rule stands
  unamended. The union of the two boxes is 88 x 71 km of which only 1.6% is
  imaged; a grid over just the two imaged boxes is 5,995 cells. Size was never
  the obstacle (see the open finding below) — it was dropped because unifying
  is a different question from the one Prignitz poses.

**Nothing about the era failed on its merits.** It was closed to keep pushing
one question rather than run two. The record lives in three places and nowhere
else: the lab-notebook entries for 31 July evening and 1 August, the closed
detour branch on the evolution graph, and the git tag
**`prignitz-detour-record`** (= old `463eeb2`), which holds the three run
directories, exported models and agent transcripts. Do not delete that tag.

### OPEN FINDING: the size gate measures the wrong artifact

Found during the detour, independent of it, and **still open**.

`MODEL_MAX_BYTES = 4 * 1024 * 1024` in `pipeline/score.py` was set in the
**bootstrap commit** (`15b43a2`) and never revisited — `git log -S` returns
that one commit. There is no derivation behind it, and §9 still lists
ESP32-P4 validation as open.

It also weighs the **fp32** ONNX, while the README's own documented path is
esp-video → esp-dl, and esp-dl ingests the ONNX for **int8** quantization.
Measured: the Berlin champion is 2.94 MB fp32 and **0.78 MB int8** — the gate
overstates every model by ~3.8x. Nothing in `pipeline/` or `model/` quantizes
anything, so §3's "quantization-aware training approach" is a line in the spec
no experiment can currently act on.

**Not changed yet, deliberately.** It binds on nothing today (the Berlin
champion passes either way), and changing a frozen gate changes the ruler for
every archived era at once — `rescore_history` re-scores old models with the
*current* scorer, so some of the 17 gated fails could flip. Worth doing as its
own deliberate step, with that effect measured first, not as a side-effect of
something else.

### KNOWN HAZARD: a lineage wipe resets the overview to a baseline

`render_overview()` takes its headline figures (median error, model size,
usable rate, the champion architecture figure) from **the last kept experiment
in `experiments.sqlite`** — i.e. the current era's champion. That is right only
while one era is the whole story. The moment the lineage is wiped for a new
era, "champion" becomes that era's day-one baseline and the front page
advertises a *weaker* result than the project has already achieved, under
prose that still names Berlin. This happened on 31 July: the overview briefly
showed 92.5% / 29 m rural numbers under a Berlin sentence, with nothing
warning, because the figures were correct and only the noun was wrong.

A fix exists in the rewound commit `f7ce820` — read the best row across all
eras out of `lineage_history.sqlite` instead, and derive the area name from
its own `metrics.json`. It was not carried across the rewind because it is a
no-op today (the Berlin champion *is* the best across all eras). **Port it
before the next lineage wipe, not after.**

---

## STATE AS OF 2026-07-31 — read this before doing anything

**Branch `berlin-slim`, pushed to origin. Loop stopped, tree clean.**

Best: **experiment 5, mission penalty 0.040** — 96.5% of held-out frames give
a usable fix (confident and within 100 m), 3.0% honest abstention, 0.5%
confidently wrong, median miss 27 m, p10 10 m. The region-holdout diagnostic
reads 0% usable / 756 m, which is **correct and expected**: this is
memorisation of one bounding box, not generalisation. That is the point of
the branch.

### Three measurement bugs were found and fixed on 31 July. Do not undo them.

1. **The evaluation held out REGIONS.** That left 28.2% of Berlin in no
   training crop and put 100% of eval questions on ground the model had never
   seen — unanswerable for a memorisation system. Now it holds out
   **viewpoints** (§4 step 4). Never reintroduce a region-based holdout as the
   primary metric.
2. **The score was an error statistic.** Median rewarded guessing the map
   centre over genuinely memorising, and reverted the first experiment that
   worked; the geometric mean that briefly replaced it was still a proxy. The
   score is now **the product requirement** (§6): `(1 - usable_fix_rate) +
   false_fix_rate`. Standing check: *if the score improves while
   `usable_fix_rate` does not, the metric is wrong again.*
3. **The harness deleted reverted experiments' source.** `snapshot_model_src()`
   in `loop.sh` now guards all four code-discarding paths.

**Every verdict before 31 July is void** — 76 experiments scored on a broken
evaluation with a metric that preferred guessing. Their verdicts do not
transfer, and the user has decided the design agent must **not** be given the
old conclusions (importing invalid refutations would steer it away from ideas
that were never actually refuted). Old DBs are in `archive/`.

**The experiments themselves are not void, and are back on the record.**
`autoresearch/rescore_history.py` re-runs the *current* frozen `score.py`
against each archived experiment's exported `berlin.onnx` on the *current*
eval set, and writes `lineage_history.sqlite` (83 rows, 5 eras). This is
measurement, not conversion: 53 re-scored from surviving models, 4 native, 4
with rates recovered arithmetically from the logged record, 17 gated fails
that had no working model then or now, 2 that predate the 10 m/px → 1 m/px
imagery switch and so cannot be asked the same question, 1 lost outright.
Every one of those states is drawn distinctly on the site; none is faked.
Rebuild with `.venv/bin/python -m autoresearch.rescore_history` (cached in
`state/rescore_cache/`, so re-runs are seconds) whenever an archived DB
changes — the gallery reads the generated DB, never the archives directly.

The headline result of that exercise: the ten-day champion re-scores at
**1.588** — 3.0% usable, 61.8% confidently wrong, i.e. **worse than saying
nothing**. The best of all 76 is 1.311. That is what makes 0.040 legible as
an achievement rather than a gift, and it is why every published page shows
all 81 development experiments rather than the current era's five.

### How to run it

```bash
./autoresearch/loop.sh 12        # runs until the DB holds 12 experiments
touch state/stop                 # graceful stop after the current one
```

Training is **local on the M1**. Modal credits are exhausted; RunPod was
abandoned. One git writer, always — the laptop commits, a remote may only
train.

### Operational hazards that have already bitten

- **Commit frozen-file edits immediately.** The running loop hard-reverts any
  uncommitted change to a `/FROZEN` file — it cannot tell an assistant's edit
  from an agent trying to cheat.
- **A running `loop.sh` ignores edits to itself** until the next launch.
- **Never kill the loop during `train`.** By then the design and
  implementation agents have already finished (~6 min of work) but nothing is
  written until the experiment completes. Stop during `design`, or use
  `state/stop`.
- **After any significant change, sweep the whole project**, not just the file
  you were asked to touch. The 20 m → 100 m change missed `gallery.py`'s own
  `TARGET_M` and shipped stale numbers to the live site.

### Pages

`gallery/index.html` research log · `gallery/inference-paths.html` model
designs · `gallery/research-lineage.html` experiment lineage ·
`gallery/research-evolution.html` how the research process itself branched
(reconstructed from conversation transcripts) · `gallery/lab-notebook.html`
dated narrative · `gallery/flight-path.html` the flight-simulator test
(rendered by `sim/render_flightpath.py` from `sim/out/`, not from the
experiment DBs; its images live in `assets/sim/`, plain git, no LFS) ·
`index.html` overview. `./infra/build_site.sh` refreshes the
`_site/` preview copy, which does **not** update on its own.

**The project is called Ballpark and the repo is `akaalias/ballpark`**, site at
`https://alexisrondeau.me/ballpark/` (2026-09-17). It was
`low-light-geolocalization-autoresearch` until that morning and `beeline` for a
few hours in between (dropped: Beeline is a Russian telco and a UK
bike-navigation brand). github.com links and git remotes redirect on their own
through both renames; Pages URLs do **not**, so the old
`alexisrondeau.me/low-light-geolocalization-autoresearch/…` links (posted in
several places) and the short-lived `/beeline/…` ones are kept alive by
redirect stubs — seven per old name — living in the **personal-site repo**
`akaalias/akaalias.github.io`, under folders of the old names. Two
consequences: never create a new repo called
`low-light-geolocalization-autoresearch` or `beeline` (it would shadow the
stubs *and* kill GitHub's repo redirect), and **a newly published page needs
no stub, but a renamed or removed one does** — add it there. The name is
applied narrowly — nav wordmark, `<title>`/og:title, README heading — and the
two tagline sentences stay as prose. The local checkout directory keeps the
original long name on purpose (Claude's project memory is keyed to the path).

All five pages now span **every era**, driven by `lineage_history.sqlite`, and
share one visual system: a background band per evaluation era, same tints and
same captions on the chart, the lineage arcs, the evolution graph and the
model-designs headings. Two rules hold across all of them and are easy to
break by accident:

- **No arc, chain or running-best line crosses an era boundary as ancestry.**
  Each wipe restarted the search from a fresh baseline, so a link across a
  boundary asserts a descent that never happened. The one thing that *is*
  continuous is the re-scored mission score, because the ruler no longer
  changes at the boundaries.
- **`kept`/`discarded` is the decision made AT THE TIME**, under whatever
  metric was then in force — which is why black dots sit high in the early
  bands. Never re-derive those flags from the re-scored number; say the two
  disagree instead.

`state/research_status` holds `finished`, which retires the pinned live banner
(it becomes a footer line) and the pulsing in-progress table row. Set it back
to `live` before restarting the loop.

### Publishing does NOT use Git LFS — keep it that way

On 2026-07-31 the account's LFS budget ran out and every Pages deploy died at
`git lfs pull`, with nothing wrong with the site. The build was hydrating
~1.8 GB of experiment binaries to publish a page whose entire image payload is
about 9 MB, because the pages reference **20 images** and the build rsynced
everything.

So the images the pages reference are frozen into `site_assets/` as ordinary
git objects (`.gitattributes` only routes `runs/**`, so nothing there is LFS),
and `infra/build_site.sh` prefers them. CI checks out with `lfs: false`.

**After any change that alters which images the pages link** — a new
experiment, a new heatmap on the overview, a layout change that adds a figure
— re-run it and commit the result, or the site will publish stale or missing
images:

```bash
.venv/bin/python -m autoresearch.gallery      # render first: it reads the HTML
.venv/bin/python infra/freeze_site_assets.py  # needs LFS hydrated locally
```

It refuses rather than publishing a broken image if any referenced file is
missing or is still an unhydrated pointer. The full-resolution originals in
`runs/` remain the research record and are untouched; `site_assets/` is a
derived artifact, reproducible byte-for-byte by re-running the script.

---

## SCOPE OVERRIDE — the `berlin-slim` configuration (2026-07-31)

**Merged into `main` on 2026-07-31 and now the project's default.** This began
as a scoped fork; `main` was fast-forwarded to it so the published site would
carry the result, and local work is back on `main`. The `berlin-slim` branch
still exists at the same commit but is no longer where work happens — if you
find yourself on it, you are on a leftover.

So read every "this branch" below as "the current configuration". It is a
deliberate, scoped narrowing of the spec that follows, traded for much faster
iteration, and it supersedes specific sections; everything else in this file
still holds. **Do not treat this section as evidence the original spec was
wrong** — it is a scoped experiment, not a reversal of the project's direction
(see memory: `berlin-slim-branch-pivot`). Widening back out to the full
four-area, six-lighting-bucket problem means reversing the items below, not
undoing the merge.

- **§0/§1 (general-purpose, one-model-per-bbox, four independent runs):**
  superseded. This branch optimizes for **one locale only — Berlin.** The
  loop is explicitly free to find a technique that only works for Berlin;
  it does not need to generalize to Prignitz/Munich/Frankfurt/Hamburg on
  this branch. "Overfitting" to Berlin's texture and lighting is not a
  failure mode here — it's the point.
- **§4 (six-bucket synthetic relighting):** disabled. Training and eval use
  the raw daytime reference imagery as fetched, unmodified — no ambient
  dimming, no artificial-light simulation, no sensor noise curve.
  `pipeline/common.py`'s `LIGHTING_BUCKETS` collapses to one pass-through
  entry. Low-light is out of scope for this branch entirely.
- **§5 (four dev areas + Hamburg holdout):** not run. `AREAS=berlin` only;
  the periodic Hamburg holdout check is disabled (gated off, not deleted —
  see `HOLDOUT_ENABLED` in `loop.sh`).
- **§6 (target ≤ 20 m):** on this branch a fix counts as **usable within
  100 m** rather than 20 m, Berlin daytime-only. The statistic itself is
  the §6 mission score (product requirement), not an error percentile. This was
  derived from data, not chosen arbitrarily: it was set from what the
  then-champion actually achieved on Berlin daytime imagery. (Those
  pre-31-July numbers are now void — see the state block at the top — but the
  100 m figure stands on its own as the radius inside which a fix is useful.)
  20 m remains the real deployment-relevant number and stays logged, not
  deleted — "milestone, not a stop condition": hitting 100 m does not end the
  loop, it just marks a report-worthy checkpoint. As of experiment 5 the
  median miss is 27 m, so the interesting question is now how far below the
  milestone this can go, not whether it can be reached.
- **§7 (pivot-gate enforcement, `PIVOT_DEMANDED`/`backbonecheck.py`):**
  disabled on this branch (forced to `PIVOT_DEMANDED=0` in `loop.sh`).
  Reintroduce once a plateau shows up worth forcing a rethink over — this
  is a "not needed yet," not a "this was broken."
- **§7 (self-refreshing HTML gallery):** kept — the gallery, per-area
  heatmap, and sample-crop rendering all stay on. Only the agent-drawn,
  `figcheck`-validated architecture SVG (`draw_figure()` in `loop.sh`) is
  cut — it was the single most expensive non-training phase per iteration
  (~15-20 min) and isn't load-bearing for the actual research question.
- **§9 (from-scratch vs. pretrained, unified vs. dispatcher):** still the
  loop's call, unchanged.
- **Model assignment:** design runs on **Fable**, implementation on
  **Opus 5** — a deliberate, branch-scoped reversal of the main branch's
  "Haiku 4.5 + Sonnet 5, never Opus" policy (see memory:
  `model-selection-policy`). Flagged, not silently applied — both models
  were previously pulled from rotation for real problems (Fable hit its
  weekly cap and stalled the loop; Opus was banned after prior issues).
  Watch for a repeat of either on this branch specifically.

Everything not listed above — §2 deployment constraints, §3 modeling
approach, §7's lineage-tracking (git commit + SQLite row + gallery)
requirement, §8-10 — still applies as written.

## Working agreement: sweep after significant changes

When a significant change lands here — a target/metric number, a scope
change (areas, lighting buckets), a frozen-file behavior change, a model
assignment — **grep the rest of the project for anything that now
references the old value, not just the file you were asked to change.**
This bit the project directly on 2026-07-31: the §6 target moved from
20 m to 100 m in `pipeline/score.py`, but `autoresearch/gallery.py` had
its *own* separate `TARGET_M = 20.0` constant plus several standalone
hardcoded "20 m" strings, all missed in the first pass and shipped to the
live gallery. Treat "I changed the source of truth" and "I changed every
place that reads it" as two separate, both-required steps — the second
one does not happen automatically, and a partial sweep (fixing the one
place you happened to be looking at) is worse than not sweeping, because
it looks done.

---

You are Claude Code Instance #1. Your job in this session is to **bootstrap**
this repository — scaffold the pipeline, prove the harness works end-to-end on
one small baseline run, and set up the autoresearch loop as a standalone bash
script. You are **not** the autoresearch loop itself. Once the harness is
proven, stop and hand off; a separate bash script will invoke headless Claude
Code instances repeatedly to do the actual research.

---

## 0. Mission

Build a from-scratch, from-scratch-licensed pipeline that, given a geographic
bounding box, trains a compact model that takes a single live camera frame
from a UAV's low-light sensor and returns an estimated GPS coordinate — with
no reference imagery shipped on the aircraft, no internet connectivity
required at inference time, and no external tool/matcher dependencies with
unclear licensing. The end deliverable per bounding box is a single weights
file plus one inference function: `estimate_position(frame) -> (lat, lon, confidence)`.

The system that produces that deliverable is itself the thing we're building:
an autonomous research loop that repeatedly trains, evaluates, and rewrites
its own approach until it hits the target metric defined below — not a
hyperparameter search over a fixed design, a coding agent free to change
architecture, loss, data augmentation, and training procedure entirely.

**Hard requirement: the pipeline must be general-purpose, not built for
Berlin+Prignitz specifically.** Every stage — fetch, relight, train, export
— must accept an arbitrary bounding box as a parameter, with no area-specific
constants, coordinate-range assumptions, or hardcoded logic anywhere in the
frozen pipeline or the trainable code. The five areas in §5 exist to exercise
and verify this genericity; they are not the limit of what the pipeline
should support. See §5 for how this is actually tested, since "accepts any
bbox in principle" and "actually verified not to overfit to the known areas"
are different claims.

---

## 1. Critical framing — read this twice

**One trained model = one bounding box.** The model memorizes its specific
training area directly into its weights (scene-coordinate regression /
absolute pose regression, not retrieval or matching against shipped
reference images). It will not — and is not expected to — generalize to a
different city.

The four evaluation areas defined in §6 are **four independent pipeline
runs**, each producing its own model. They exist to prove that the
*training method* (architecture search, relighting approach, augmentation
strategy — whatever the autoresearch loop lands on) generalizes across
structurally different area types (dense urban vs. sparse rural vs. two
other major metros), not to produce one model that covers all four. Do not
build a multi-area or multi-tenant model. Do not average performance across
areas in a way that hides a bad result in one of them — see §7.

---

## 2. Deployment target (hard constraints)

These come from a real, already-chosen airframe and are non-negotiable
unless explicitly revisited by the human:

- **Airframe:** TBS Source One V6, 5" open-source freestyle frame, paired
  with a minimal/low-cost FC+ESC stack and motors. This is a freestyle
  platform, not a survey platform — spare payload and power margin are
  small.
- **Payload budget:** camera + companion compute combined should target
  well under 50 g.
- **Power budget:** hard ceiling ≈ 6–9 W average (derived from ~5% of a
  typical 20–30 Wh 4S–6S pack over a ~10 minute flight). Target well under
  that — ideally under 2 W — since the whole point of this project is
  "little expense to battery."
- **Companion board:** **ESP32-P4**, chosen over ESP32-S3 specifically for
  its native MIPI-CSI + integrated ISP (so a real low-light MIPI camera
  module can attach directly) and its hardware CNN accelerator / vector
  instructions. Bare chip ~$3–4 in quantity; dev boards with camera support
  ~$25–50. **This is a recommendation, not yet hardware-validated** — see
  §10, first bullet.
- **Sensor:** low-light "starlight"-class CMOS (STARVIS2/IMX585 class or
  similar), MIPI interface to match the P4. Sensor cost is explicitly
  **not** constrained — only weight and power are.
- **Deployment artifact:** a single weights file (ONNX, converted to
  whatever runtime the P4 toolchain needs) plus one function,
  `estimate_position(frame) -> (lat, lon, confidence)`. **No reference
  imagery is shipped on-device** — this is the constraint that drives the
  scene-coordinate-regression architecture choice in §4.
- **Fix schedule:** adaptive, not continuous — roughly every 5–10 s during
  cruise, stepping up to every 1–2 s during final approach/landing (below
  ~30–50 m altitude). No optical-flow or rangefinder aiding is assumed;
  the vision fix is the only drift correction available, which is why the
  coarse-phase interval should stay toward the tighter end of that range.
- **Cost outside the sensor:** prefer the smallest/cheapest board and
  components that satisfy the above, but this is a soft preference, not a
  hard gate.

---

## 3. Modeling approach

**Scene coordinate regression / absolute pose regression, trained from
scratch per bounding box.** The model encodes "what does this specific area
look like, from which positions, under which lighting" directly into its
weights and regresses a coordinate at inference — no retrieval, no stored
reference images, no external matcher network in the deployed path. (This
family includes methods like DSAC/DSAC++ and ACE — "Accelerated Coordinate
Encoding" — which are specifically designed to train a compact,
scene-specific network quickly on a single GPU; useful prior art, not a
required dependency.)

**Search space explicitly open to the autoresearch loop, not fixed by this
spec:**
- Single unified model vs. several small lighting-condition-specific models
  behind a lightweight dispatcher (the human's own instinct leans toward
  the latter, but the loop should be free to experiment and decide based
  on the metric — including how well each option fits the P4's memory
  budget).
- Backbone/architecture, loss function, augmentation strategy, and
  quantization-aware training approach.
- From-scratch weight init vs. a permissively-licensed (MIT/BSD/Apache)
  pretrained backbone used only for initialization — log which was tried
  and why; see §10.

**Explicit goal for the loop:** favor real architectural/procedural
novelty over parameter tuning. If an experiment only changes a
hyperparameter within a fixed design, it is not using the agent's actual
advantage over a hyperparameter sweep.

---

## 4. Data pipeline (frozen — the loop must not rewrite this)

This part is fixed by the bootstrap instance (you) and must not be modified
by later autoresearch iterations, so that improvements are measured against
a stable, trustworthy eval set.

1. **Input:** a geographic bounding box.
2. **Reference imagery fetch:** daytime satellite/aerial imagery for that
   box from an **open-licensed source only** — Sentinel-2 (Copernicus, free
   for commercial and non-commercial use with attribution) and/or open
   orthophoto/DOP programs (e.g. Berlin/Brandenburg's open geoportal,
   equivalents for Bavaria/Hesse for the Munich/Frankfurt areas). **Do not
   use Google Maps or Bing tiles** — their terms prohibit exactly this kind
   of caching/derivative-model use, and the human wants this repo
   open-sourceable.
3. **Synthetic low-light relighting:** transform the daytime reference into
   six lighting-condition variants — morning, midday, afternoon, early
   evening, evening, night. Starting point: separate **reflected ambient
   light** (terrain/buildings — dims with time-of-day/ambient level) from
   **active artificial lighting** (streetlamps, windows — stays lit
   regardless of natural ambient level or cloud cover), then apply a
   sensor gain/noise curve approximating the chosen low-light sensor's
   response. This mirrors a rough prototype already built by the human in
   an earlier exploration (an HTML/canvas ambient-illumination simulator) —
   reuse that logic as a v0 starting point, not as a fixed final answer.
   **This relighting method is one of the areas most open to genuine
   improvement by the loop** — if it can find a better sim-to-real
   approach (e.g. learned relighting instead of hand-tuned curves), that's
   exactly the kind of novel result this project is for.
4. **Train/eval split:** held-out **viewpoints**, not held-out regions.
   Training covers every lattice position of the whole bounding box — the
   model is meant to memorize its one box, so it is shown all of it. Eval
   crops are off-lattice viewpoints over that same mapped ground, 11–17 m
   from the nearest training framing and carrying a deterministic rotation:
   the ground is mapped, the view is new, which is exactly the deployment
   condition. A small 1-in-32-block region stays genuinely out of training
   as a logged-only diagnostic (never scored) against a structureless
   lookup table. **Do not reintroduce a region-based holdout as the primary
   metric** — that was the v1 design, and it left 28.2% of Berlin in no
   training crop and 100% of eval questions on never-seen ground, asking a
   memorization model to locate places it had never been shown. See the
   docstring of `pipeline/dataset.py` for the full measurement.

---

## 5. Areas (see §1 — five separate pipeline runs, not one combined model)

Four **development areas** the loop optimizes against, plus one **blind
holdout** it never sees during optimization:

| Area | Role |
|---|---|
| Berlin | Primary / dense urban reference case |
| Prignitz, Brandenburg | Rural extreme — among the lowest population density in Germany, minimal artificial lighting |
| Munich metro | Second major-metro replicate |
| Frankfurt metro | Third major-metro replicate |
| **Hamburg metro** | **Blind generalization holdout.** Structurally distinct from the other four (port city, Elbe river, more spread-out geometry) so it can't be passed by coincidence. |

The rural/urban spread across the four development areas exists to catch a
pipeline that quietly overfits its design choices (relighting curve,
augmentation, model capacity) to one area's texture and lighting density.
The fifth area exists for a different, stricter reason: to catch the
pipeline *itself* being overfit — hardcoded assumptions, quirks the loop
learned to exploit across all four known areas at once — which a worst-case
metric computed only over those four cannot detect, since the loop is free
to adapt to all of them simultaneously.

**Rule: the holdout area's data is never touched by the autoresearch loop's
optimization step.** It is evaluated only as a final, periodic, read-only
check (e.g. once per N kept improvements, or once at the end of a work
session) and its result is logged but must **not** feed back into which
experiments get kept or reverted — see §6 and §7. If the holdout score
diverges badly from the four development areas' worst-case score, that's a
signal the pipeline has a genericity problem worth investigating before
continuing, not something to quietly average away.

---

## 6. Target metric — the single scalar the loop optimizes

**Primary (optimized by the loop) — THE PRODUCT REQUIREMENT, not an error
statistic.** The aircraft takes a vision fix every 5–10 s and it is its only
drift correction (§2). Per frame, exactly three things can happen:

| outcome | meaning |
|---|---|
| confident **and** within target | **usable fix** — the product |
| not confident | abstains — safe, waits for the next frame |
| confident **but** outside target | **false fix** — dangerous, injects a wrong position into navigation |

    mission_score = (1 - usable_fix_rate) + false_fix_rate     [minimized]

0.0 = every frame usable; 1.0 = abstains everywhere; 2.0 = confidently wrong
everywhere, i.e. **strictly worse than silence** — which is what this section
always asserted ("a model that honestly abstains on bad frames is more useful
than one that confidently guesses wrong") and what no earlier metric actually
encoded. The loop optimizes the worst such score across lighting buckets ×
development areas. A fix counts as usable within **20 m** (**100 m** on the
berlin-slim branch).

**Do not replace this with an error statistic.** Median, geometric mean, p10
and p25 were each tried as the primary and each rewarded something the product
does not want. The median rewarded a model guessing the map centre and was
caught reverting the only experiment that had begun to memorize; the geometric
mean that replaced it was a research proxy chosen to make progress *visible*
rather than to state what the aircraft *needs* — the same category of error.
All remain logged as diagnostics. **The standing check: if the score improves
while `usable_fix_rate` does not, the metric is wrong again.**

Use worst-case rather than mean deliberately — an average can hide a bad
rural or night-time failure behind a good Berlin-daytime score, which is
exactly the failure mode §5's rural/urban spread is designed to catch.

**Reported separately, not optimized (the §5 holdout check):** the same
worst-case-across-lighting-buckets metric, computed on Hamburg, logged
alongside every periodic holdout check. This number must never influence
which experiments the loop keeps or reverts — its only job is to reveal
whether the pipeline generalizes to a bounding box it has never
specifically been tuned against.

**Gating (folds hard constraints into the same scalar):** if a candidate
model's exported artifact doesn't fit the ESP32-P4 deployment envelope
(memory footprint, inference latency within the adaptive-schedule budget),
record the score as failed/worst-possible regardless of accuracy. This
keeps the loop optimizing one number while still enforcing the deployment
constraints from §2.

**Secondary, logged but not directly optimized:** mean error, per-area and
per-lighting-bucket breakdown, exported model size, single-inference
latency and estimated power draw on the target board, and a coverage
metric (% of held-out frames the model returns a confident estimate for at
all — a model that honestly abstains on bad frames is more useful than one
that confidently guesses wrong).

---

## 7. Autoresearch loop architecture

Modeled on Karpathy's `autoresearch` pattern (github.com/karpathy/autoresearch),
**not** an evolutionary/genome search — the point is that the agent can
rewrite actual code (architecture, loss, training loop), not just search a
predefined parameter space.

**Structure:**
- A small set of **frozen files** the loop must never modify: the data
  pipeline (§4), the four areas' held-out eval sets, and the scoring script
  that computes §6's metric.
- One (or a small set of) **agent-editable file(s)**: model architecture
  and training procedure.
- **Loop mechanics per experiment:** branch from current best, make one
  focused change, run a short training job, score against §6, and either
  advance (keep the commit) if the score improved or revert if it didn't —
  same keep/revert discipline as Karpathy's design.
- **Compute:** one GPU per experiment run. The human has RunPod.io credits
  available — the bootstrap should make it straightforward to point a run
  at a RunPod instance, but this doesn't need to be fully automated
  (spin-up/tear-down) in v1; a documented manual step is fine to start.

**Full lineage tracking (hard requirement, not optional):**
Every experiment round must be fully reconstructable after the fact —
not just a metric number in a CSV row. Concretely:
- **Git**: one commit per kept improvement (Karpathy-style), so the code
  history *is* the research trail.
- **SQLite**: one row per experiment — timestamp, git commit hash, area,
  a description of what changed and why, every metric from §6 (primary,
  secondary, per-area/per-bucket breakdown), and paths to any generated
  artifact files for that run.
- **Self-refreshing HTML gallery**: rendered from the SQLite log, showing
  per experiment the synthetic low-light training samples used and a
  per-area error heatmap image — so a human can visually sanity-check
  results, not just read numbers. This combination (SQLite + self-refreshing
  HTML gallery + lineage tracking) mirrors a pattern the human has used
  before in a separate project; if their actual implementation differs
  from what you scaffold here, expect it to be revised after the fact —
  build a clean, reasonable version now rather than blocking on it.

---

## 8. Your job in this session (Phase 1 — bootstrap only)

1. Scaffold the repo: frozen data pipeline (§4) built as a **general
   bbox-parameterized system** (§0), proven by running it unmodified
   against at least Berlin and one other area — not just implemented once
   for Berlin and assumed to generalize. Also scaffold a first
   trivial/naive baseline model + training script, the scoring script
   (§6, including the separate holdout-check path for Hamburg per §5),
   the SQLite schema, the HTML gallery template, and the bash loop script
   described in §7 (not yet running it repeatedly).
2. Run one baseline experiment end-to-end to prove the harness works:
   fetch → relight → train (even badly) → score → log to SQLite → render
   gallery → commit.
3. Write a README explaining how to run the bash loop separately (Phase 2,
   run by the human afterward, not by you in this session).
4. Do **not** attempt to hit the §6 target yourself, and do not run more
   than the one proving experiment — that's the autoresearch loop's job,
   invoked separately.

---

## 9. Open items — resolve during bootstrap, or flag clearly if you can't

- **ESP32-P4 hardware validation**: confirm a specific low-light MIPI
  camera module is actually available and documented-compatible with the
  P4's MIPI-CSI before assuming this hardware choice is final. If you
  can't verify this with confidence, flag it explicitly in the README
  rather than silently proceeding as if it's settled.
- **From-scratch vs. permissively-licensed pretrained init** (§3) — the
  loop's call; log which was tried in each experiment's SQLite row.
- **Unified vs. dispatcher+specialist models** (§3) — the loop's call.

---

## 10. Explicitly out of scope for this project

These were explored and deliberately set aside — don't let the loop wander
into them:
- Optical-flow / rangefinder sensor fusion (§2 assumes vision-only fixes).
- Star-tracker / celestial navigation.
- Thermal or SWIR imaging as the primary sensor.
- Streetlight-constellation matching as a standalone fallback method
  (interesting, but a different system from this one).
