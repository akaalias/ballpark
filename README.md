# Ballpark

**Can a UAV learn a city by heart?** Given a geographic bounding box, this
repo trains a compact per-area model that takes a single downward camera
frame from a UAV and returns
`estimate_position(frame) -> (lat, lon, confidence)` — no reference imagery
on the aircraft, no connectivity, targeting an ESP32-P4 companion board.
The interesting part: the model-side code is improved by an **autonomous
research loop** (Karpathy `autoresearch`-style) that designs experiments,
rewrites architecture/loss/training, and keeps or reverts by a single frozen
metric. Full context: `CLAUDE.md`.

## Repo layout

| Path | Status | What |
|---|---|---|
| `pipeline/` | **FROZEN** | bbox-generic fetch of one or several **captures** per area (independent open-licensed photographs of the same ground on one grid — see `pipeline/common.py`), 6-bucket synthetic low-light relighting (collapsed to a daytime pass-through in the current berlin-slim configuration), deterministic split holding out **viewpoints, not regions** (training covers the whole bbox; eval frames sit 11-17 m off the nearest training vantage — see `pipeline/dataset.py` for why a region-based holdout was wrong) **and, since 2026-09-28, the photograph itself** (the eval capture is never trained on), §6 scoring — the mission score, (1 - usable_fix_rate) + false_fix_rate, which is the product requirement rather than an error statistic — with ESP32-P4 deployment gates |
| `areas.yaml` | **FROZEN** | 4 development areas + hamburg blind holdout; per-area capture lists (Berlin: 3 training photographs + 1 held out) |
| `model/` | agent-editable | model architecture + training procedure (the loop's playground) |
| `autoresearch/` | mostly frozen | `loop.sh` harness, SQLite schema + logger, gallery renderer, per-iteration agent prompt |
| `FROZEN` | — | the exact list of files the loop may never touch (enforced by `loop.sh` via git revert) |
| `runs/` | generated | per-experiment artifacts (models, metrics, heatmaps, samples) |
| `experiments.sqlite` | generated | full experiment lineage: hypothesis, method, expected outcome, result, conclusion, all metrics |
| `gallery/` | generated | self-refreshing pages: research log (`index.html`, samples + error heatmaps) and proposed inference paths (`inference-paths.html`, every experiment's pre-registered architecture figure) |
| `sim/` | tool, not frozen | virtual UAV flight simulator (`flightsim.py`) — flies the exported champion from a to b with its fixes as the only position source — and the flight-path page renderer (`render_flightpath.py`). Reads the frozen pipeline, never modifies it |

## Quickstart (already proven end-to-end by the bootstrap run)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy pillow rasterio pystac-client torch onnx onnxruntime pyyaml

# Frozen pipeline — any bbox works, named areas are just presets:
.venv/bin/python -m pipeline.fetch --area berlin          # or: --name x --bbox w,s,e,n
                                                          # fetches every capture in areas.yaml; existing ones are reused
.venv/bin/python -m pipeline.relight --area berlin        # one relight/<bucket>.png per capture x lighting condition
.venv/bin/python -m model.train --area berlin --out-dir runs/test --epochs 3
.venv/bin/python -m pipeline.score --areas berlin --model-dir runs/test/models --out runs/test/metrics.json
.venv/bin/python -m autoresearch.gallery                  # open gallery/index.html
```

Bootstrap proving runs: experiment #1 (commit `15b43a2`) ran berlin +
prignitz on 10 m Sentinel-2 data → 2926 m. Experiment #3 (commit `c0a1bdf`,
pipeline **data v2**) re-proved the harness on 1 m/px DOP imagery across all
four development areas → **2765.67 m** worst-case median error, coverage 1.0
(historical: that bootstrap run predates the current metric — the primary
score is now the §6 **mission score**, not a median; see `pipeline/score.py`)
in all 24 area×bucket cells (target ≤ 20 m; a center-guess on a ~7 km box is
~2.8 km — the baseline is deliberately naive). All deployment gates passed.
The §6 anti-abstention gate was exercised for real en route: two confidence
calibrations collapsed dark-bucket coverage and were correctly scored FAIL.

## Phase 2 — running the autoresearch loop (run this yourself, separately)

> **Current configuration: `berlin-slim`** (merged into `main` 2026-07-31),
> **capture-holdout era** (branch `capture-holdout`, opened 2026-09-28).
> Berlin only, raw daytime imagery, no synthetic relighting, figures off,
> pivot gate off. The model trains on three open survey photographs of
> Berlin and is scored on a fourth it never sees — because the 0.040
> champion, scored on any other photograph of the same ground, scored
> 1.8-2.0 (random, and confident). See CLAUDE.md's "CAPTURE-HOLDOUT ERA" and
> "SCOPE OVERRIDE" for what each supersedes and why.

```bash
./autoresearch/loop.sh 12               # run until the DB holds 12 experiments
SKIP_AGENT=1 ./autoresearch/loop.sh 1   # smoke-test the harness without an agent
touch state/stop                        # graceful stop after the current experiment
```

Data for berlin is already fetched and relit; the other areas are only
needed if you restore the full four-area scope:

```bash
for a in prignitz munich frankfurt hamburg; do
  .venv/bin/python -m pipeline.fetch --area $a
  .venv/bin/python -m pipeline.relight --area $a
done
```

Per experiment the design agent pre-registers `title / category / hypothesis
/ method / expected_outcome` (written to the run's `experiment.json`); the
harness appends the measured `result` and a kept/reverted `conclusion`, logs
to SQLite, snapshots `model/*.py` into `runs/<id>/model_src/` **whatever the
outcome**, and commits kept improvements so git history is the research trail.

Watch progress: `open gallery/index.html` (re-rendered when a design lands
*and* again when the experiment finishes), or
`sqlite3 experiments.sqlite "SELECT id,title,primary_metric,kept FROM experiments ORDER BY id DESC LIMIT 10;"`

### Compute

Training runs **locally on the M1** (`model/train.py` auto-selects
cuda > mps > cpu). Viable on a laptop again only because this branch trains
one area instead of four.

Two remote paths exist and are dormant:

| | status |
|---|---|
| **Modal** (`REMOTE_TRAIN=1 REMOTE_BACKEND=modal`) | stateless per-area trainer, never touches git. Used 23–30 Jul; **credits exhausted**, so unusable until topped up. |
| **RunPod** (`infra/runpod.sh`, `REMOTE_BACKEND=runpod`) | **abandoned 2026-07-23** after three days of stuck commits and merge conflicts. Scripts kept for reference; do not restart it without reading `docs/infra.md` first. |

The rule that came out of that period and still holds: **one git writer,
always.** The laptop commits; a remote may only train and hand back
artifacts. Two writers on one branch was the root cause of the whole
mess, not the provider.

## Flight test (`sim/`) — does the champion actually get a UAV from a to b?

A virtual-UAV flight simulator closes the loop on the mission score's own
premise (§2: the vision fix is the only drift correction). A point-mass
fixed-wing flies through wind, gusts and gyro bias its navigator cannot see;
every fix interval the sim crops the reference raster at the aircraft's
**true** pose, runs the exported champion ONNX, and hands the answer to a
dead-reckoning navigator that accepts false fixes like any other. Arrival is
declared on the navigator's estimate and scored against the truth. Honesty
rules are documented at the top of `sim/flightsim.py`; the known best-case
caveats (camera = the memorized reference imagery; fixed 1 m/px altitude) are
stated on the results page rather than hidden.

Headline (100 flights, seed 1): **91/100 arrive within 100 m**, median
declared miss 45 m; the dead-reckoning control arrives 0/100. All three
catastrophic failures had targets on the never-trained region-holdout
diagnostic ground — an artifact of simulating over the research
configuration, reported as flown. Results are rendered as the
`gallery/flight-path.html` page (replay, estimation-error sawtooth,
100-flight record).

```bash
.venv/bin/python -m sim.flightsim --seed 6 --start 350,350 --target 6800,6600 \
    --save-crops --out sim/out/showcase.json      # the page's showcase flight
.venv/bin/python -m sim.flightsim --flights 100 --seed 1 --out sim/out/mc.json
.venv/bin/python -m sim.flightsim --flights 100 --seed 1 --no-vision --out sim/out/mc_dr.json
.venv/bin/python -m sim.render_flightpath         # rebuild the page from sim/out/
```

## Reference imagery (pipeline data v3) & licensing

Fetching is driven by a **source registry** (`pipeline/sources.yaml`,
frozen): for any bbox, the finest open-licensed auto-pickable source
covering it wins, falling back to global Sentinel-2. An area may also list
**captures** in `areas.yaml` — several independent photographs of the same
bbox, each from a *named* source, all warped onto one shared grid
(`data/<area>/captures/<name>.tif`) so a crop at the same pixel shows the
same ground in every one. `role: train` captures are what the model sees;
the `role: eval` capture is a photograph it never sees and the §6 score is
computed on it. The pipeline code stays fully bbox-generic — regional
knowledge is config, not code.

| Source | Covers | Native | License / attribution |
|---|---|---|---|
| [BB-BE DOP20 WMS](https://isk.geobasis-bb.de/mapproxy/dop20c/service/wms?REQUEST=GetCapabilities&SERVICE=WMS) | Brandenburg **+ Berlin** | 20 cm | dl-de/by-2-0 — © GeoBasis-DE/LGB |
| [Berlin TrueDOP 2022 / 2023 / 2024 WMS](https://gdi.berlin.de/services/wms/truedop_2024?SERVICE=WMS&REQUEST=GetCapabilities) (named captures only) | Berlin | 20 cm | dl-de/zero-2-0 — Geoportal Berlin |
| [Bavaria DOP40 WMS](https://geoservices.bayern.de/od/wms/dop/v1/dop40?SERVICE=WMS&REQUEST=GetCapabilities) | Bavaria (Munich) | 40 cm | CC BY 4.0 — © Bayerische Vermessungsverwaltung |
| [Hesse DOP20 WMS](https://www.gds-srv.hessen.de/cgi-bin/lika-services/ogc-free-images.ows?SERVICE=WMS&REQUEST=GetCapabilities) | Hesse (Frankfurt) | 20 cm | dl-de/by-2-0 — © HVBG Hessen |
| [Hamburg DOP WMS](https://geodienste.hamburg.de/wms_dop_zeitreihe_unbelaubt?Service=WMS&Request=GetCapabilities) | Hamburg | 20 cm | dl-de/by-2-0 — © FHH LGV |
| Sentinel-2 L2A (Earth Search STAC → AWS COGs) | global fallback | 10 m | Copernicus — "Contains modified Copernicus Sentinel data" |

Everything is server-resampled to a **1 m/px** reference grid (`target_gsd_m`
in the registry): a 128 px model crop = 128 m ground footprint, matching a
typical UAV camera at ~100 m AGL — so training imagery scale ≈ deployment
imagery scale. No Google/Bing tiles anywhere. Expect roughly 0.5–3 GB of
imagery per area on disk after relighting.

## Validated hardware setup (§9 resolved for bring-up; stretch sensor still open)

**Buy-now, documented-working stack:**

- **Board:** [ESP32-P4-Function-EV-Board](https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32p4/esp32-p4-function-ev-board/user_guide.html)
  — Espressif's official P4 dev kit; **ships with a 2 MP MIPI-CSI camera
  module (SC2336)** in the box.
- **Sensor:** SmartSens **SC2336** (2 MP, 1/3", RAW out over MIPI CSI-2;
  SmartSens' "SmartClarity-2" low-light line). Driver ships in Espressif's
  [esp_cam_sensor](https://components.espressif.com/components/espressif/esp_cam_sensor/versions/2.3.0/readme)
  component; the P4's on-chip ISP handles debayer/AWB/denoise
  ([camera driver docs](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-reference/peripherals/camera_driver.html)),
  and a [community P4+SC2336 example](https://github.com/jeff-cn/esp32-p4-cam) exists.
- **Software path:** esp-video (V4L2-style capture) → esp-dl (quantized CNN
  inference on the P4's PPA/vector unit). Model export here is ONNX, which
  esp-dl's toolchain ingests for int8 quantization.
- For the flight article (vs. dev kit), the same chip + SC2336 module on a
  minimal carrier lands well under the 50 g / 2 W targets of §2.

**Still open (stretch):** a true STARVIS2/IMX585-class starlight sensor on
the P4 has no documented pairing — IMX585 typically wants 4-lane MIPI (P4
has 2) and a custom driver. Path: prove the full system on SC2336 first;
invest in starlight bring-up only if night accuracy turns out sensor-limited.
(The [ams MIRA220 NIR example](https://github.com/ams-OSRAM/esp32_p4_MIPI_DSI_CSI_mira220)
shows third-party sensor bring-up on the P4 is feasible.)

## Adding a new deployment area

`.venv/bin/python -m pipeline.fetch --name myarea --bbox 9.10,48.70,9.20,48.76`
then relight/train/score as above — nothing in the pipeline knows about
specific areas. Only add entries to `areas.yaml` if it should become part of
the frozen evaluation set (that's a human decision, not the loop's).

## License

**GPL-3.0-or-later** (see `LICENSE`). The choice is deliberate: the point of
this project is that anyone can generate their own tiny, self-contained,
geo-boxed localization model — so the pipeline that makes that possible must
stay free. Copyleft guarantees it: you may use, study, modify and
redistribute everything here, but a distributed fork — improved or otherwise
— must remain under the GPL. Freedom is the priority here, not adoption.

Two boundaries, spelled out in `NOTICE`:

- **Model weights you train are yours.** They are your output, not a
  derivative of this code in the license's sense; this repository claims no
  rights over them.
- **Imagery-derived figures** (under `runs/` and `site_assets/`) carry their
  geodata sources' attribution terms (dl-de/by-2-0, CC BY 4.0, Copernicus)
  independent of the code license.

The repository was MIT-licensed before 2026-08-01; the relicense was made by
the sole copyright holder.
