"""gallery/flight-path.html — the champion flying a virtual UAV from a to b.

Renders the flight-simulator results (sim/flightsim.py) as a site page in the
shared visual system: the showcase flight replayed over the Berlin orthophoto,
the estimation-error sawtooth, the 100-flight Monte Carlo, and an explicit
what-is-real / what-is-simulated accounting. Reads sim/out/*.json; never
touches the frozen pipeline.

Assets: writes a desaturated, downsampled basemap and the showcase camera
crops to assets/sim/ (plain git, copied by build_site.sh — deliberately NOT
runs/, so nothing here goes near LFS).

Usage:  .venv/bin/python -m sim.render_flightpath
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from autoresearch.gallery import (CSS, compute_banner, credits_html, rsec,
                                  social_meta, topnav)
from pipeline.dataset import BLOCK_PX, WINDOW_PX, _window_hits_holdout, block_role

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "gallery" / "flight-path.html"
ASSETS = REPO_ROOT / "assets" / "sim"
SHOWCASE = REPO_ROOT / "sim" / "out" / "showcase.json"
MC = REPO_ROOT / "sim" / "out" / "mc.json"
MC_DR = REPO_ROOT / "sim" / "out" / "mc_dr.json"
CROPS_SRC = REPO_ROOT / "sim" / "out" / "showcase_crops"

BASEMAP_W = 1600

# Styles for the shared flight-replay section (embedded on both the
# flight-path page and the root overview via flight_section()).
FP_CSS = """
.fp-legend{display:flex;gap:22px;justify-content:center;flex-wrap:wrap;
  font:13px var(--serif);color:var(--muted);margin:10px 0 4px}
.fp-legend span{display:inline-flex;align-items:center;gap:7px}
.fp-legend i{font-style:normal;display:inline-block}
.k-true{width:22px;height:0;border-top:2.5px solid var(--ink)}
.k-est{width:22px;height:0;border-top:2.5px dashed var(--ochre)}
.k-usable{width:9px;height:9px;border-radius:50%;background:var(--ink)}
.k-abstain{width:9px;height:9px;border-radius:50%;border:1.6px solid var(--faint);background:transparent}
.k-false{width:9px;height:9px;border-radius:50%;border:2px solid var(--accent);background:transparent}
.k-offmap{width:9px;height:9px;border:1.6px dashed var(--faint);background:transparent}
.k-box{width:16px;height:10px;border:1.2px dashed var(--muted);background:transparent}
/* the replay breaks out of the text column: up to 1440 px of the window,
   24 px clear of each edge, centred on the column it sits in */
.fp-wide{width:min(1440px,calc(100vw - 48px));position:relative;left:50%;
  transform:translateX(-50%)}
.fp-stage{display:flex;gap:22px;align-items:flex-start;margin-top:12px}
.fp-map{flex:1 1 640px;min-width:0;position:relative}
.fp-map svg{display:block;width:100%;height:auto;border:1px solid var(--rule)}
.fp-side{flex:0 0 190px;position:sticky;top:14px}
.fp-cam{border:1px solid var(--rule);background:#f7f4e6;padding:10px}
/* shown at (near-)native size — stretching the 128 px crops, and especially
   the 64 px reconstructions, to panel width just reads as blur */
.fp-cam img,.fp-cam canvas{width:100%;max-width:132px;height:auto;
  aspect-ratio:1/1;image-rendering:auto;display:block;margin:0 auto;
  border:1px solid var(--rule-soft)}
.fp-cam .cam-h{font:600 10.5px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin:0 0 7px}
/* three fixed lines (time, outcome, how far off) that never wrap, so the
   panel keeps one height for the whole flight */
.fp-cam .cam-out{margin:8px 0 0;font:13px/1.5 var(--serif);height:4.5em}
.fp-cam .cam-out span{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.fp-cam .cam-out b.usable{color:var(--ink)}
.fp-cam .cam-out b.abstain{color:var(--faint)}
.fp-cam .cam-out b.false_fix{color:var(--accent)}
.fp-cam .cam-sub{margin:2px 0 0;font:12.5px/1.5 var(--serif);color:var(--muted)}
.fp-read{margin-top:12px;font:13px/1.7 var(--serif);color:var(--muted)}
.fp-read b{color:var(--ink);font-variant-numeric:lining-nums tabular-nums}
/* ONE row under the map: mission, flight, play, scrub, time, speed. The
   menus size to their content; the scrubber takes what is left; on a
   narrow screen the row wraps rather than overflowing. */
.fp-controls select{font:13px var(--serif);color:var(--ink);
  background:var(--paper);border:1px solid var(--rule);height:28px;padding:0 6px;
  min-width:0;max-width:100%}
.fp-controls{display:flex;gap:8px;align-items:center;margin:12px 0 0;
  font:13px var(--serif);color:var(--muted);flex-wrap:wrap}
.fp-controls button{font:600 12px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.07em;color:var(--paper);
  background:var(--ink);border:none;height:28px;padding:0 16px;cursor:pointer}
.fp-controls button:hover{background:var(--accent)}
.fp-controls input[type=range]{flex:1 1 120px;min-width:80px;accent-color:#8c2f1f}
.fp-controls .t-read{font-variant-numeric:lining-nums tabular-nums;min-width:52px;text-align:right}
/* the readout: a two-line heading that never wraps, then five fixed rows,
   so its height never changes while the numbers do */
.fp-read .rd-h{color:var(--ink);font-weight:600;line-height:1.45;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.fp-read .lab-sub{font-weight:400;color:var(--muted)}
.fp-read dl.rd{display:grid;grid-template-columns:1fr auto;gap:1px 8px;margin:8px 0 0;
  font:12.5px/1.6 var(--serif)}
.fp-read dl.rd dt{color:var(--muted);white-space:nowrap}
.fp-read dl.rd dd{margin:0;text-align:right;color:var(--ink);font-weight:600;
  font-variant-numeric:lining-nums tabular-nums;white-space:nowrap}
.fp-note{max-width:780px;margin:14px auto 0;font:14.5px/1.7 var(--serif);color:#4a473e}
.fp-note p{margin:0 0 10px}
.fp-note b{color:var(--ink)}
.fp-tip{position:fixed;z-index:60;pointer-events:none;background:var(--paper);
  border:1px solid var(--rule);box-shadow:0 2px 10px rgba(0,0,0,.09);
  font:12.5px/1.5 var(--serif);color:var(--ink);padding:7px 10px;display:none;max-width:250px}
.fp-tip .tt-h{font-weight:600}
.fp-tip .tt-m{color:var(--muted)}
/* Slim viewports: the side panel would land far below the map, so it is
   replaced by a compact camera preview overlaid on the map itself. */
.fp-cam-mini{display:none;position:absolute;top:8px;right:8px;width:92px;
  background:var(--paper);border:1px solid var(--rule);padding:4px;
  box-shadow:0 1px 6px rgba(0,0,0,.18)}
.fp-cam-mini img,.fp-cam-mini canvas{display:block;width:100%;height:auto;
  aspect-ratio:1/1;border:1px solid var(--rule-soft)}
.fp-cam-mini .mini-t{display:block;font:10.5px var(--serif);color:var(--muted);
  text-align:center;margin-top:3px;font-variant-numeric:lining-nums tabular-nums}
@media(max-width:820px){
  table.fp-fail{display:block;max-width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch}
  .fp-stage{flex-direction:column}
  .fp-side{display:none}
  .fp-cam-mini{display:block}
}
"""

# Styles used only by the flight-path page's own sections (charts, failure
# table, honesty grid).
FP_PAGE_CSS = """
.fp-wrap{max-width:1020px;margin:0 auto;padding:14px 28px 96px}
.fp-chart{margin-top:8px}
.fp-chart svg{display:block;width:100%;height:auto}
table.fp-fail{margin:18px auto 0;border-collapse:collapse;font:14px/1.55 var(--serif)}
.fp-cmp .sub{font-size:12px;color:var(--muted);font-weight:400}
.fp-cmp th{vertical-align:bottom}
table.fp-fail th{font:600 10.5px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
  border-bottom:1px solid var(--rule);padding:4px 14px;text-align:left}
table.fp-fail td{border-bottom:1px solid var(--rule-soft);padding:5px 14px}
table.fp-fail td.num{font-variant-numeric:lining-nums tabular-nums}
table.fp-fail .hold{color:var(--ochre);font-size:12.5px}
table.fp-mis{max-width:100%}
table.fp-mis td{vertical-align:top}
.fp-mis .mis-asks{font-size:13px;color:#4a473e;max-width:330px;margin-top:2px}
.fp-mis .mis-ok{font-size:12px;color:var(--muted);font-style:italic;margin-top:2px}
.fp-mis td.rp{white-space:nowrap}
.fp-rp{font:600 11px var(--serif);font-feature-settings:"smcp" 1;text-transform:uppercase;
  letter-spacing:.06em;color:var(--paper);background:var(--ink);border:none;
  padding:5px 10px;cursor:pointer;margin:0 0 4px}
.fp-rp.fail{background:var(--accent)}
.fp-rp:hover{opacity:.85}
@media(max-width:700px){table.fp-mis td,table.fp-mis th{padding:5px 6px}
  .fp-mis td.rp{white-space:normal}}
.fp-honest{max-width:780px;margin:16px auto 0;display:grid;
  grid-template-columns:1fr 1fr;gap:6px 34px;font:14.5px/1.65 var(--serif)}
.fp-honest h3{grid-column:span 1;font:600 11px var(--serif);
  font-feature-settings:"smcp" 1;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);margin:14px 0 2px;border-bottom:1px solid var(--rule-soft);padding-bottom:4px}
.fp-honest ul{margin:4px 0 0;padding-left:20px}
.fp-honest li{margin:0 0 7px;color:#33312b}
.fp-honest li b{color:var(--ink)}
@media(max-width:820px){
  .fp-honest{grid-template-columns:1fr}
}
"""


def build_basemap(meta_w, meta_h):
    """Downsampled, desaturated orthophoto (Berlin TrueDOP 2024, the eval
    capture) so the overlays carry the page.

    The source raster (data/, gitignored) only exists on a research machine;
    the derived basemap is committed in assets/sim/, so CI — which re-renders
    every page via build_site.sh — reuses it rather than regenerating."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "berlin-base.jpg"
    # The held-out 2024 photograph (issue #15): the one the camera sees, and
    # one the model never trained on.
    src = REPO_ROOT / "data" / "berlin" / "relight" / "truedop_2024.png"
    if not src.exists():
        if not out.exists():
            raise SystemExit(f"neither {src} nor committed {out} exists")
        return out
    img = Image.open(src).convert("RGB")
    # Full-color 2 m/px copy for the client-side camera-view reconstruction
    # (lazy-loaded only when a non-showcase flight is selected).
    zoom = img.resize((meta_w // 2, meta_h // 2), Image.LANCZOS)
    zoom.save(ASSETS / "berlin-zoom.jpg", quality=58, optimize=True)
    img = img.resize((BASEMAP_W, round(meta_h * BASEMAP_W / meta_w)), Image.LANCZOS)
    img = ImageEnhance.Color(img).enhance(0.45)
    img = ImageEnhance.Brightness(img).enhance(1.12)
    paper = Image.new("RGB", img.size, (255, 255, 248))
    img = Image.blend(img, paper, 0.22)
    img.save(out, quality=80, optimize=True)
    return out


def build_crops(fixes):
    """Camera crops -> small JPEGs; returns fix-index -> asset filename."""
    d = ASSETS / "crops"
    d.mkdir(parents=True, exist_ok=True)
    names = {}
    stale = {f.name for f in d.glob("*.jpg")}
    for i, fx in enumerate(fixes):
        src = fx.get("crop")
        if not src:
            continue
        jpg = Path(src).stem + ".jpg"
        if (CROPS_SRC / src).exists():
            Image.open(CROPS_SRC / src).convert("RGB").save(d / jpg, quality=82, optimize=True)
        elif not (d / jpg).exists():
            raise SystemExit(f"crop source {src} and committed {d / jpg} both missing")
        names[i] = jpg
        stale.discard(jpg)
    for name in stale:  # crops from a previous showcase flight
        (d / name).unlink()
    return names


MC_FLIGHTS_SRC = REPO_ROOT / "sim" / "out" / "mc_flights"
SHOWCASE_MISS = json.loads(SHOWCASE.read_text())["flights"][0]["declared_miss_m"]

# The previous record, kept for the comparison block: berlin-slim experiment 5
# (0.040), flown 2026-08-28 over the raster it had memorized. Same routes,
# seeds and wind as today's runs.
PREVIOUS = {"within": 91, "median": 45.4, "worst": 7362, "ff": 207, "ff_flights": 50,
            "off_map": 77, "showcase_miss": 25.5, "showcase_ff": 7, "showcase_peak": 5510}


def build_flight_assets():
    """Slim the full per-flight Monte-Carlo records (sim/out/mc_flights/,
    local-only — reproducible from seed 1, so not committed) into the
    committed assets/sim/flights/ the page fetches on flight selection.
    Track thinned to whole seconds, everything else kept verbatim. CI-safe:
    without the source dir the committed assets are reused, same pattern as
    the basemap."""
    dst = ASSETS / "flights"
    dst.mkdir(parents=True, exist_ok=True)
    if not MC_FLIGHTS_SRC.exists():
        if not any(dst.glob("f*.json")):
            raise SystemExit(f"neither {MC_FLIGHTS_SRC} nor committed {dst}/f*.json exists")
        return
    for p in sorted(MC_FLIGHTS_SRC.glob("f*.json")):
        f = json.loads(p.read_text())
        slim = {k: f[k] for k in
                ("flight_id", "start_px", "target_px", "direct_m", "flight_s",
                 "wind", "result", "declared_miss_m", "final_true_dist_m",
                 "fix_counts", "fixes")}
        # Thin to ~1 s by INDEX (recorded cadence is 0.5 s), never by matching
        # timestamps: the sim clock accumulates 0.1 s floats, so sample times
        # phase-shift (x.0 -> x.1) mid-flight and a whole-second filter would
        # silently truncate the track there. Always keep the final point.
        slim["track"] = f["track"][::2]
        if slim["track"][-1] is not f["track"][-1]:
            slim["track"].append(f["track"][-1])
        (dst / p.name).write_text(json.dumps(slim, separators=(",", ":")))


MISSIONS_OUT = REPO_ROOT / "sim" / "out" / "missions"
# The replay opens on this flight (Alexis, 2026-11-03): Spree flight 2 of 40,
# east to west, a success whose five false fixes sit exactly where the
# river's false fixes cluster. Its camera frames were saved by re-flying it
# (sim/missions.py --save-crops-for 2, bit-identical to the 40-flight run).
SHOW_KIND, SHOW_ID = "spree", 2
SHOW_TITLE = "Berlin Spree corridor mission, camera only, GPS-denied"   # the section's heading
SHOW_CROPS_SRC = REPO_ROOT / "sim" / "out" / "mission_crops" / f"{SHOW_KIND}_f{SHOW_ID}"
SHOW_CROPS_REL = f"assets/sim/missions/{SHOW_KIND}/crops_f{SHOW_ID}/"
MISSION_FLIGHTS_SRC = REPO_ROOT / "sim" / "out" / "mission_flights"
# (key, picker name, what it asks) — order of the picker and the table
MISSION_KINDS = [
    ("return_home", "Return home",
     "Fly 2+ km out to B, then back to the launch point: the lost-link failsafe. Home has to be found again from the opposite heading."),
    ("tour", "Waypoint tour",
     "Four spots in a row, 1&ndash;3 km apart. Does the error build up from leg to leg?"),
    ("loiter", "Loiter",
     "Orbit one spot at 150 m radius for five minutes: the same ground from every heading, where one false fix pulls the aircraft off station."),
    ("survey", "Survey grid",
     "Six 1.2 km lines, 200 m apart, at a random angle: the aircraft has to hold each line, not just reach its end."),
    ("cold_start", "Cold start",
     "The aircraft does not know where it is: it believes it sits at the centre of the map and circles until the first confident fix."),
    ("spree", "Spree corridor",
     "7.5 km along the river through the centre, both directions: long stretches of water and embankment that look alike."),
]
MISSION_SUCCESS = {
    "return_home": "B and home both within 100 m",
    "tour": "all four spots within 100 m",
    "loiter": "&ge;95% of the orbit within 100 m of the circle",
    "survey": "&ge;95% of the grid lines flown within 100 m",
    "cold_start": "arrives within 100 m",
    "spree": "&ge;95% of the river flown within 100 m",
}


def build_mission_assets():
    """Slim the full mission flight records (sim/out/mission_flights/, local-
    only, reproducible from seed 1 via sim/missions.py --flight-dir) into the
    committed assets/sim/missions/<kind>/f<i>.json the replay fetches. Same
    CI-safe pattern as build_flight_assets."""
    for kind, _, _ in MISSION_KINDS:
        src, dst = MISSION_FLIGHTS_SRC / kind, ASSETS / "missions" / kind
        dst.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            if not any(dst.glob("f*.json")):
                raise SystemExit(f"neither {src} nor committed {dst}/f*.json exists")
            continue
        for q in sorted(src.glob("f*.json")):
            f = json.loads(q.read_text())
            slim = {k: f[k] for k in
                    ("flight_id", "mission", "success", "start_px", "target_px",
                     "direct_m", "flight_s", "wind", "result", "declared_miss_m",
                     "final_true_dist_m", "fix_counts", "fixes",
                     "waypoints_px", "waypoints", "loiter", "cold_start",
                     "localized_s", "xtrack") if k in f}
            if kind == SHOW_KIND and f["flight_id"] == SHOW_ID and any("crop" in x for x in f["fixes"]):
                cdir = REPO_ROOT / SHOW_CROPS_REL
                cdir.mkdir(parents=True, exist_ok=True)
                for x in slim["fixes"]:
                    if "crop" in x:
                        jpg = Path(x["crop"]).stem + ".jpg"
                        src = SHOW_CROPS_SRC / x["crop"]
                        if src.exists():
                            Image.open(src).convert("RGB").save(cdir / jpg, quality=82, optimize=True)
                        elif not (cdir / jpg).exists():
                            raise SystemExit(f"showcase crop {src} and committed {cdir / jpg} both missing")
                        x["crop"] = jpg
                slim["crop_base"] = SHOW_CROPS_REL
            slim["track"] = f["track"][::2]          # ~1 s, by index (see build_flight_assets)
            if slim["track"][-1] is not f["track"][-1]:
                slim["track"].append(f["track"][-1])
            (dst / q.name).write_text(json.dumps(slim, separators=(",", ":")))


def _mission_label(kind, f):
    """Short per-flight label for the picker: which flight, how many false
    fixes. Success or failure is the optgroup it sits in."""
    n = f["fix_counts"]["false_fix"]
    return f"flight {f['flight_id']} &middot; {n} false fix{'' if n == 1 else 'es'}"


def mission_data():
    """Everything the page needs about the six missions, from the committed
    sim/out/missions/*.json: picker rows, table numbers, and the Spree
    false-fix profile along the river."""
    out = {}
    for kind, name, asks in MISSION_KINDS:
        d = json.loads((MISSIONS_OUT / f"{kind}.json").read_text())
        dr = json.loads((MISSIONS_OUT / f"{kind}_dr.json").read_text())
        fl = d["flights"]
        t = d["summary"]["total_fixes"]
        ok = [f["flight_id"] for f in fl if f["success"]]
        bad = [f["flight_id"] for f in fl if not f["success"]]
        out[kind] = {
            "name": name, "asks": asks, "n": len(fl),
            "rows": [{"id": f["flight_id"], "ok": bool(f["success"]),
                      "lab": _mission_label(kind, f)} for f in fl],
            "example": ok[0] if ok else fl[0]["flight_id"],
            "fail_example": bad[0] if bad else None,
            "success": d["summary"]["success_rate"],
            "success_dr": dr["summary"]["success_rate"],
            "ff_rate": t["false_fix"] / max(1, sum(t[k] for k in ("usable", "abstain", "false_fix"))),
            "ff_flights": d["summary"]["n_flights_with_false_fix"],
            "summary": d["summary"],
            "flights": fl,
        }
    return out


def _route_km(route, p):
    """Distance along a polyline (px == m) of the point nearest p, in km."""
    route = np.asarray(route, float)
    seg = np.diff(route, axis=0)
    cum = np.r_[0, np.cumsum(np.linalg.norm(seg, axis=1))]
    best = (1e18, 0.0)
    for i, ab in enumerate(seg):
        a = route[i]
        t = float(np.clip(np.dot(np.asarray(p) - a, ab) / ab.dot(ab), 0, 1))
        dd = float(np.linalg.norm(np.asarray(p) - (a + t * ab)))
        if dd < best[0]:
            best = (dd, cum[i] + t * float(np.linalg.norm(ab)))
    return best[1] / 1000


def show_flight():
    """The default replay flight, read from its committed slim record."""
    return json.loads((ASSETS / "missions" / SHOW_KIND / f"f{SHOW_ID}.json").read_text())


def show_story(f):
    """Plain-language facts of the showcase flight, all from its record:
    where on the river each false fix happened, how far the belief strayed,
    how it finished. Grouped by place so the prose can say 'three at the
    Spreebogen' rather than list timestamps."""
    route = [f["start_px"]] + f["waypoints_px"]
    # km counts from the EAST end whatever the direction flown, so the place
    # names match the section-4 strip
    east = route if route[0][0] > route[-1][0] else route[::-1]
    ff = [x for x in f["fixes"] if x["outcome"] == "false_fix"]
    groups = []
    for x in ff:
        km = _route_km(east, x["true"])
        place = SPREE_PLACES.get(int(km), f"km {km:.1f}")
        if groups and groups[-1]["place"] == place:
            groups[-1]["fx"].append(x)
        else:
            groups.append({"place": place, "fx": [x]})
    T = f["track"]
    est_max = max(((p[1] - p[3]) ** 2 + (p[2] - p[4]) ** 2) ** 0.5 for p in T)
    return {"ff": ff, "groups": groups, "est_max": est_max,
            "km": sum(np.linalg.norm(np.subtract(route[i + 1], route[i]))
                      for i in range(len(route) - 1)) / 1000,
            "dur_min": f["flight_s"] / 60, "end_m": f["final_true_dist_m"],
            "within": f["xtrack"]["within_100m"], "fc": f["fix_counts"]}


def _groups_html(st):
    parts = []
    for g in st["groups"]:
        n = len(g["fx"])
        if n == 1:
            parts.append(f"at {g['place']} it takes <b>one false fix</b>")
        else:
            parts.append(f"at {g['place']} it takes <b>{n} false fixes</b> in "
                         f"{g['fx'][-1]['t'] - g['fx'][0]['t']:.0f}&nbsp;s")
    txt = "; ".join(parts)
    return txt[0].upper() + txt[1:]


def spree_profile(fl):
    """False-fix share per kilometre of river, measured along the route."""
    route = np.array([fl[0]["start_px"]] + fl[0]["waypoints_px"], float)
    seg = np.diff(route, axis=0)
    cum = np.r_[0, np.cumsum(np.linalg.norm(seg, axis=1))]
    def along(p):
        best = (1e18, 0.0)
        for i, ab in enumerate(seg):
            a = route[i]
            t = float(np.clip(np.dot(p - a, ab) / ab.dot(ab), 0, 1))
            dd = float(np.linalg.norm(p - (a + t * ab)))
            if dd < best[0]:
                best = (dd, cum[i] + t * float(np.linalg.norm(ab)))
        return best[1]
    nb = int(np.ceil(cum[-1] / 1000))
    tot, ff = np.zeros(nb), np.zeros(nb)
    for f in fl:
        for x in f["fixes"]:
            if x["outcome"] == "off_map":
                continue
            k = min(nb - 1, int(along(np.array(x["true"])) // 1000))
            tot[k] += 1
            ff[k] += x["outcome"] == "false_fix"
    return {"km": round(cum[-1] / 1000, 1), "tot": tot.tolist(), "ff": ff.tolist()}


SPREE_PLACES = {3: "the Spreebogen beside the Chancellery", 6: "the Charlottenburg bend"}


def spree_svg(prof):
    """Bar strip: share of fixes that were false, per kilometre of river."""
    W, H, ML, MB, MT = 900, 150, 40, 30, 22
    n = len(prof["tot"])
    rates = [f / t if t else 0.0 for f, t in zip(prof["ff"], prof["tot"])]
    top = max(0.2, max(rates))
    bw = (W - ML - 10) / n
    out = [f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='share of false fixes per kilometre of the Spree route'>"]
    for k, r in enumerate(rates):
        x = ML + k * bw
        h = (H - MB - MT) * r / top
        col = "var(--accent)" if r >= 0.02 else "var(--ink)"
        out.append(f"<rect x='{x + 6:.1f}' y='{H - MB - max(h, 1.2):.1f}' width='{bw - 12:.1f}' "
                   f"height='{max(h, 1.2):.1f}' style='fill:{col}'/>")
        out.append(f"<text x='{x + bw / 2:.1f}' y='{H - MB - max(h, 1.2) - 6:.1f}' text-anchor='middle' "
                   f"style='font:12px var(--serif);fill:{col}'>{r:.1%}</text>")
        out.append(f"<text x='{x + bw / 2:.1f}' y='{H - 10}' text-anchor='middle' "
                   f"style='font:11.5px var(--serif);fill:var(--muted)'>km {k}&ndash;{k + 1}</text>")
    out.append(f"<line x1='{ML}' x2='{W - 10}' y1='{H - MB}' y2='{H - MB}' style='stroke:var(--rule)'/>")
    out.append(f"<text x='{ML}' y='12' style='font:italic 11.5px var(--serif);fill:var(--muted)'>"
               "share of fixes that were false, by kilometre along the river (east &rarr; west)</text>")
    out.append("</svg>")
    return "".join(out)


def missions_section(n):
    """Section: the six other mission types — table, Spree profile, replay links."""
    md = mission_data()
    def pct(x):
        return f"{x:.0%}"
    def detail(kind, m):
        sm = m["summary"]
        if kind == "return_home":
            return (f"home reached within 100 m on {sum(1 for f in m['flights'] if len(f['waypoints']) == 2 and f['waypoints'][1]['true_miss_m'] <= 100)}"
                    f" of {m['n']} flights, median {sm['waypoint_miss_median_m'][1]:.0f} m")
        if kind == "tour":
            return ("median miss per spot " + " / ".join(f"{v:.0f}" for v in sm["waypoint_miss_median_m"])
                    + " m: no build-up")
        if kind == "loiter":
            return (f"median {sm['on_station_median']:.0%} of the orbit on station; worst flight "
                    f"{sm['on_station_worst']:.0%}")
        if kind in ("survey", "spree"):
            return f"median cross-track {sm['xtrack_mean_median_m']:.0f} m"
        if kind == "cold_start":
            first = ("located itself on its first frame (median flight); " if sm['localized_s_median'] < 1
                     else f"located itself after {sm['localized_s_median']:.0f} s (median); ")
            return first + f"{sm['first_confident_fix_false']} flights began from a false fix and recovered"
        return ""
    rows = []
    for kind, m in md.items():
        rp = (f"<button class='fp-rp' onclick=\"FP.loadFlight('{kind}:{m['example']}');"
              f"document.getElementById('map').scrollIntoView({{behavior:'smooth',block:'center'}})\">success</button>")
        if m["fail_example"] is not None:
            rp += (f" <button class='fp-rp fail' onclick=\"FP.loadFlight('{kind}:{m['fail_example']}');"
                   f"document.getElementById('map').scrollIntoView({{behavior:'smooth',block:'center'}})\">failure</button>")
        rows.append(
            f"<tr><td><b>{m['name']}</b><div class='mis-asks'>{m['asks']}</div>"
            f"<div class='mis-ok'>success = {MISSION_SUCCESS[kind]}</div></td>"
            f"<td class='num'><b>{pct(m['success'])}</b></td>"
            f"<td class='num'>{pct(m['success_dr'])}</td>"
            f"<td class='num'>{m['ff_rate']:.1%}</td>"
            f"<td>{detail(kind, m)}</td><td class='rp'>{rp}</td></tr>")

    prof = spree_profile(md["spree"]["flights"])
    ff_all = sum(prof["ff"])
    order = sorted(range(len(prof["ff"])), key=lambda k: -prof["ff"][k])[:2]
    hot = sum(prof["ff"][k] for k in order)
    def place(k):
        return SPREE_PLACES.get(k, f"km {k}&ndash;{k + 1}")
    hot_txt = " and ".join(
        f"{place(k)} ({prof['ff'][k] / prof['tot'][k]:.0%} of fixes there false)" for k in order)
    others = [prof["ff"][k] / prof["tot"][k] for k in range(len(prof["ff"])) if k not in order and prof["tot"][k]]
    tour, spree = md["tour"], md["spree"]
    failed = [f for m in md.values() for f in m["flights"] if not f["success"]]
    failed_ff = sum(1 for f in failed if f["fix_counts"]["false_fix"])
    def worst_miss(f):
        if "xtrack" in f:
            return f["xtrack"]["max_m"]
        return max([w["true_miss_m"] for w in f.get("waypoints", [])] + [f["final_true_dist_m"]])
    near = [worst_miss(f) for f in failed if not f["fix_counts"]["false_fix"]]
    return f"""
{rsec(n, "Six more missions",
      "Return home, a tour, a loiter, a survey, a cold start and the river: the same aircraft and model, asked different questions.")}
<div class="fp-note">
<p>A&nbsp;&rarr;&nbsp;B asks one thing. Real flights ask more: come back,
hold a spot, sweep an area, start without knowing where you are. Each
mission below was flown {md['spree']['n']} times with the same airframe,
wind, navigator and model as above, over the same 2024 photograph, and
once more <b>without the camera</b> in the very same weather. Pick one in
the replay&rsquo;s first menu, or use the buttons to watch a success and a
failure.</p></div>
<table class="fp-fail fp-mis">
<thead><tr><th>mission</th><th>succeeds</th><th>without<br>camera</th>
<th>fixes<br>false</th><th>measured</th><th>replay</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<div class="fp-note">
<p><b>The error does not build up.</b> On the tour the median miss is the
same at the fourth spot as at the first; every usable fix resets the
estimate. Of the {len(failed)} failed mission flights, {failed_ff} took at
least one confident wrong answer that dragged the believed position away;
the other {len(near)} had none and missed narrowly
({min(near):.0f}&ndash;{max(near):.0f} m against the 100 m limit). A longer
mission simply draws more wrong answers, which is why the four-spot tour succeeds on
{pct(tour['success'])} of flights while each of its spots is reached on
{pct(tour['summary']['waypoints_within_100m'])} of visits.</p>
<p><b>False fixes belong to places, not to chance.</b> Along the Spree
{spree['ff_rate']:.1%} of fixes are false, against
{min(v['ff_rate'] for k, v in md.items() if k != 'spree'):.1%}&ndash;{max(v['ff_rate'] for k, v in md.items() if k != 'spree'):.1%}
on the other missions, and every one of the {spree['n']} flights had at least one.
But {hot:.0f} of the {ff_all:.0f} sit in two kilometres of river:
{hot_txt}. The other kilometres run at {min(others):.1%}&ndash;{max(others):.1%}.
The same model flying the same river fails at the same spots, so this is a
property of the map, measurable before anyone flies, and the kind of
weakness a training experiment can target.</p></div>
<div class="fp-chart">{spree_svg(prof)}</div>"""


def mission_picker_html():
    opts = "".join(f"<option value='{k}'>{n}</option>" for k, n, _ in MISSION_KINDS)
    return ("<select id='missionsel' aria-label='choose a mission type'>"
            "<option value='a_to_b' selected>A &rarr; B</option>" + opts + "</select>")


def flight_selector_html():
    """<select> over the showcase + all 100 Monte-Carlo flights, grouped by
    outcome, labels carrying the true miss so failures are browsable too."""
    rows = mc_compact(json.loads(MC.read_text()))
    def opt(r):
        return (f"<option value='{r['id']}'>flight {r['id']} &middot; "
                f"{'timed out' if r['result'] == 'timeout' else 'lands'} "
                f"{r['miss']:,.0f} m out</option>")
    groups = (
        ("arrived within 100 m", [r for r in rows if r["within"]]),
        ("missed beyond 100 m", [r for r in rows if not r["within"] and r["result"] != "timeout"]),
        ("timed out", [r for r in rows if r["result"] == "timeout"]),
    )
    body = "".join(
        f"<optgroup label='{label} ({len(rs)})'>" + "".join(opt(r) for r in rs) + "</optgroup>"
        for label, rs in groups if rs)
    return ("<select id='flightsel' aria-label='choose a flight to replay'>"
            "<option value='showcase' selected>showcase &middot; corner to corner "
            f"&middot; lands {SHOWCASE_MISS:.1f} m out</option>" + body + "</select>")


def mc_compact(mc):
    """Per-flight rows for the dot strip + failure table, holdout attribution."""
    half = WINDOW_PX // 2 + 1
    rows = []
    for f in mc["flights"]:
        ff = [fx for fx in f["fixes"] if fx["outcome"] == "false_fix"]
        n_hold = sum(1 for fx in ff if _window_hits_holdout(
            "berlin", int(round(fx["true"][0])), int(round(fx["true"][1])), half))
        tx, ty = (int(round(v)) for v in f["target_px"])
        rows.append({
            "id": f["flight_id"], "result": f["result"],
            "miss": f["declared_miss_m"] if f["result"] == "arrived" else f["final_true_dist_m"],
            "within": bool(f["arrived_within_100m"]),
            "direct": f["direct_m"],
            "u": f["fix_counts"]["usable"], "a": f["fix_counts"]["abstain"],
            "ff": len(ff), "ff_hold": n_hold,
            "target_hold": _window_hits_holdout("berlin", tx, ty, half)
                           or block_role("berlin", tx // BLOCK_PX, ty // BLOCK_PX) == "holdout",
        })
    return rows


def flight_section(rel, record_note, n=1):
    """The complete flight-replay section (heading SHOW_TITLE) — heading, narrative,
    legend, replay stage and its script — shared by the flight-path page
    (rel='../') and the root overview (rel=''). Reads only committed inputs
    (sim/out/showcase.json + assets/sim/), so any page can embed it without
    the research machine's data/ or a prior asset build.

    record_note: the closing sentence of the disclosure footnote — the two
    pages point at the 100-flight record differently ("below" vs a link).
    """
    showcase = json.loads(SHOWCASE.read_text())["flights"][0]
    for fx in showcase["fixes"]:
        if "crop" in fx:
            fx["crop"] = Path(fx["crop"]).stem + ".jpg"
    show = show_flight()
    st = show_story(show)
    md = mission_data()
    row = next(r for r in md[SHOW_KIND]["rows"] if r["id"] == SHOW_ID)
    first_crop = next((fx["crop"] for fx in show["fixes"] if "crop" in fx), "")
    EXT = json.loads((ASSETS / "berlin-wide.json").read_text())
    meta_w, meta_h = 6939, 6828  # berlin raster (data/berlin/meta.json)
    data = {
        "w": meta_w, "h": meta_h, "ext": EXT,
        "label": (f"{md[SHOW_KIND]['name']} &middot; flight {SHOW_ID}<br><span class='lab-sub'>"
                  f"{'mission succeeded' if row['ok'] else 'mission failed'}</span>"),
        "flight": show,
        "show": f"{SHOW_KIND}:{SHOW_ID}",
        "ab_label": (f"A &rarr; B &middot; showcase<br><span class='lab-sub'>lands "
                     f"{showcase['declared_miss_m']:.0f} m from target</span>"),
        "ab": {k: showcase[k] for k in
               ("start_px", "target_px", "direct_m", "flight_s", "wind",
                "declared_miss_m", "fix_counts", "fixes", "track")},
        "missions": {k: {"name": v["name"], "example": v["example"],
                         "rows": v["rows"]} for k, v in md.items()},
    }
    return f"""
{rsec(n, SHOW_TITLE,
      f"Along the Spree over a photograph the model never saw — {st['km']:.1f} km in {st['dur_min']:.0f} minutes, {len(st['ff'])} false fixes at {len(st['groups'])} places, finished {st['end_m']:.0f} m from the end.")}
<div class="fp-wide">
<div class="fp-legend">
<span><i class="k-true"></i>true path</span>
<span><i class="k-est"></i>believed path</span>
<span><i class="k-usable"></i>usable fix</span>
<span><i class="k-abstain"></i>abstention</span>
<span><i class="k-false"></i>false fix</span>
<span><i style="color:var(--accent);font-weight:700">&times;</i>where the false fix claimed to be</span>
<span><i class="k-offmap"></i>off the mapped box &mdash; no imagery, no fix</span>
</div>

<div class="fp-stage">
<div class="fp-map">
<svg id="map" viewBox="0 0 {meta_w} {meta_h}" role="img"
     aria-label="Berlin orthophoto with the simulated flight path from start to target">
<image href="{rel}assets/sim/berlin-wide.jpg" x="{EXT['x']}" y="{EXT['y']}" width="{EXT['w']}" height="{EXT['h']}" preserveAspectRatio="none"/>
<g id="mapg"></g>
</svg>
<div class="fp-cam-mini" id="cammini">
<img id="miniimg" src="{rel}{SHOW_CROPS_REL}{first_crop}" alt="camera crop at the latest fix" width="128" height="128">
<canvas id="minicv" width="128" height="128" style="display:none"></canvas>
<span class="mini-t" id="minit">&mdash;</span>
</div>
<div class="fp-controls">
{mission_picker_html()}
{flight_selector_html()}
<button id="play">Play</button>
<input id="scrub" type="range" min="0" max="1000" value="0" aria-label="flight time">
<span class="t-read num" id="tread">0.0 s</span>
<select id="speed" aria-label="replay speed">
<option value="4">4&times;</option><option value="10" selected>10&times;</option>
<option value="25">25&times;</option><option value="50">50&times;</option></select>
</div>
</div>
<div class="fp-side">
<div class="fp-cam">
<p class="cam-h">Bottom camera &mdash; what the model saw</p>
<img id="cam" src="{rel}{SHOW_CROPS_REL}{first_crop}" alt="camera crop at the latest fix" width="128" height="128">
<canvas id="camcv" width="128" height="128" style="display:none"></canvas>
<p class="cam-out" id="camout">&mdash;</p>
<p class="cam-sub" id="camsub">128&thinsp;&times;&thinsp;128 px &middot; 1 m/px &middot; ~100 m AGL</p>
</div>
<div class="fp-read" id="read"></div>
</div>
</div>
</div>
<div class="fp-tip" id="tip"></div>
<script>{replay_js(json.dumps(data, separators=(',', ':')), json.dumps(rel))}</script>"""


def main():
    showcase = json.loads(SHOWCASE.read_text())["flights"][0]
    mc = json.loads(MC.read_text())
    mc_dr = json.loads(MC_DR.read_text())
    meta_w, meta_h = 6939, 6828  # berlin raster (data/berlin/meta.json)

    build_basemap(meta_w, meta_h)
    build_crops(showcase["fixes"])
    build_flight_assets()
    build_mission_assets()

    rows = mc_compact(mc)
    sst = show_story(show_flight())
    s = mc["summary"]
    n_within = s["n_arrived_within_100m"]
    failures = sorted((r for r in rows if not r["within"]), key=lambda r: r["miss"])
    ff_total = s["total_fixes"]["false_fix"]
    ff_hold = sum(r["ff_hold"] for r in rows)
    n_conf_or_abst = sum(s["total_fixes"][k] for k in ("usable", "abstain", "false_fix"))
    ff_rate_nonhold = 100.0 * (ff_total - ff_hold) / n_conf_or_abst
    misses = sorted(r["miss"] for r in rows if r["result"] == "arrived")
    p10, p90 = misses[9], misses[89] if len(misses) > 89 else misses[-1]
    peak_km = max(((a[1] - a[3]) ** 2 + (a[2] - a[4]) ** 2) ** 0.5
                  for a in showcase["track"]) / 1000

    failure_rows = "\n".join(
        f"<tr><td class='num'>{r['id']}</td>"
        f"<td>{'timed out' if r['result'] == 'timeout' else 'declared arrival'}</td>"
        f"<td class='num'>{r['miss']:,.0f} m</td>"
        f"<td class='num'>{r['ff']}{f' <span class=hold>({r['ff_hold']} on never-trained ground)</span>' if r['ff_hold'] else ''}</td>"
        f"<td>{'yes' if r['target_hold'] else ''}</td></tr>"
        for r in failures)

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{social_meta('gallery/flight-path.html', 'Flight path — Ballpark',
             f'A virtual fixed-wing UAV flies Berlin from a to b with the champion model as its only position source, over a photograph it never trained on. {n_within} of 100 flights arrive within 100 m; the failures are shown, not averaged away. Six more missions follow: return home, tour, loiter, survey, cold start and the Spree.')}
<title>Flight path &mdash; Ballpark</title>
<style>{CSS}
{FP_CSS}
{FP_PAGE_CSS}</style></head><body>
{topnav('flight')}
{compute_banner()}

<header class='page-head'>
<h1>Flight path</h1>
<p class='page-sub'>The champion put to the product&rsquo;s own test: a
virtual fixed-wing UAV that must fly from <b>a</b> to <b>b</b> over Berlin
with the model&rsquo;s vision fixes as its <b>only</b> position source
&mdash; dead reckoning in between, wind and gyro bias it cannot see, false
fixes accepted like any other. Arrival is declared on the aircraft&rsquo;s
own estimate and scored against the truth. Then six other kinds of
mission: return home, a tour, a loiter, a survey grid, a cold start and
the river.</p></header>

<div class="fp-wrap">

<div class="stat-hero"><b class="num">{n_within}<span style="color:var(--faint)">/100</span></b>
<span>flights arrive within 100 m</span></div>
<div class="stats">
<div class="stat"><b class="num">{s['declared_miss_median_m']:.0f} m</b>
<span>median miss</span></div>
<div class="stat"><b class="num">0<span style="color:var(--faint)">/100</span></b>
<span>without camera</span></div>
<div class="stat"><b class="num">{len(failures)}</b>
<span>miss &mdash; all shown below</span></div>
</div>
<p class="hero-sub">100 simulated flights, random start and target &ge;2 km
apart, random 2&ndash;6 m/s wind. Each vision fix is the real exported
champion (<span class="mono">berlin.onnx</span>) run on a crop of Berlin&rsquo;s
<b>2024 aerial survey</b> at the aircraft&rsquo;s true position and heading
&mdash; a photograph it never saw in training &mdash; classified with the
frozen scorer&rsquo;s own thresholds.</p>

<p class="bl-h">Then and now</p>
<table class="fp-fail fp-cmp">
<thead><tr><th></th><th>previous champion<br><span class="sub">its own training photograph</span></th>
<th>current champion<br><span class="sub">a photograph it never saw</span></th></tr></thead>
<tbody>
<tr><td>flights within 100 m</td><td class="num">{PREVIOUS['within']}/100</td><td class="num"><b>{n_within}/100</b></td></tr>
<tr><td>median miss at arrival</td><td class="num">{PREVIOUS['median']:.0f} m</td><td class="num">{s['declared_miss_median_m']:.0f} m</td></tr>
<tr><td>worst flight</td><td class="num">{PREVIOUS['worst']:,} m</td><td class="num">{s['declared_miss_worst_m']:,.0f} m</td></tr>
<tr><td>false fixes, all 100 flights</td><td class="num">{PREVIOUS['ff']} <span class="sub">in {PREVIOUS['ff_flights']} flights</span></td><td class="num"><b>{ff_total}</b> <span class="sub">in {s['n_flights_with_false_fix']} flights</span></td></tr>
<tr><td>fixes lost off the mapped box</td><td class="num">{PREVIOUS['off_map']}</td><td class="num">{s['total_fixes']['off_map']}</td></tr>
<tr><td>corner-to-corner showcase: landed / false fixes / worst belief error</td><td class="num">{PREVIOUS['showcase_miss']} m / {PREVIOUS['showcase_ff']} / {PREVIOUS['showcase_peak'] / 1000:.1f} km</td><td class="num">{showcase['declared_miss_m']:.0f} m / {showcase['fix_counts']['false_fix']} / {peak_km:.1f} km</td></tr>
</tbody></table>
<p class="hero-sub">Same 100 routes, seeds and wind. The previous champion
(berlin-slim experiment 5, 0.040) was flown over the very photograph it had
memorized; on any other photograph of Berlin it scores 1.87, as good as
guessing. The current champion (capture-holdout experiment 20, 0.078) is
flown over the held-out 2024 survey, the harder test. It lands about as
close (median 47 m against 45 m), but far more flights arrive, nothing is
lost off the map, and it makes under a third of the false fixes.</p>

{flight_section("../", "The 100-flight record below uses random routes, failures included.")}

{rsec(2, "The estimate against the truth",
      "How far the navigator's belief is from the aircraft's true position, second by second.")}
<div class="fp-note"><p>Between fixes the error grows &mdash; wind the
navigator cannot see. Each <b>usable fix</b> snaps it back down. The
spikes are the <b style="color:var(--accent)">false fixes</b>:
{"; ".join(f"t&thinsp;=&thinsp;{g['fx'][0]['t']:.0f}{('&ndash;' + format(g['fx'][-1]['t'], '.0f')) if len(g['fx']) > 1 else ''}&thinsp;s at {g['place']}" for g in sst['groups'])}.
The navigator moves most of the way toward every confident answer, so a
false fix drags the belief in proportion to how wrong it is. The
river&rsquo;s are {min(x['err_m'] for x in sst['ff']):.0f}&ndash;{max(x['err_m'] for x in sst['ff']):.0f}&nbsp;m
off, so the belief never strays more than {sst['est_max']:.0f}&nbsp;m, and the
next usable fix pulls it back. On the corner-to-corner A&nbsp;&rarr;&nbsp;B
showcase (first menu) a single barely-confident answer 2.3&nbsp;km off
pulled the belief {peak_km:.1f}&nbsp;km away: that is why the mission score
prices false fixes so high. Square-root vertical scale, so the cruise
detail and the spikes fit one axis; the dashed line is the 100&nbsp;m usable
radius. The chart follows whichever flight the replay is showing.</p></div>
<div class="fp-chart" id="chartbox"></div>

{rsec(3, "A hundred flights",
      "Every Monte-Carlo flight's miss distance — arrivals and near-misses, no catastrophes.")}
<div class="fp-note">
<p>Each dot is one flight&rsquo;s true distance from target at the moment it
declared arrival (hollow&nbsp;= timed out, plotted at final distance).
{n_within} land inside the 100 m line. The other {len(failures)} miss
moderately ({failures[0]['miss']:.0f}&ndash;{failures[-1]['miss']:.0f} m),
and none is lost: no flight left the mapped box, none timed out. (The
previous champion had three catastrophes of 2&ndash;7 km.) {ff_hold} of the
{ff_total} false fixes stand on the never-trained holdout ground, the ~3% of
Berlin the evaluation harness deliberately withholds from training as a
diagnostic. Excluding it, the in-flight false-fix rate is
{ff_rate_nonhold:.2f}%, consistent with the 0.75% the frozen eval measures
on the same photograph. A deployed model would train on 100% of its box,
so the holdout share is an artifact of simulating over the research
configuration &mdash; reported as flown, not filtered out.</p></div>
<div class="fp-chart" id="mcbox"></div>
<table class="fp-fail">
<thead><tr><th>flight</th><th>ended</th><th>true miss</th>
<th>false fixes</th><th>target on never-trained ground</th></tr></thead>
<tbody>{failure_rows}</tbody></table>

{missions_section(4)}

{rsec(5, "What is real here, and what is simulated",
      "The test is honest only if its seams are visible.")}
<div class="fp-honest">
<div><h3>Real</h3><ul>
<li><b>The model:</b> the exported champion <span class="mono">berlin.onnx</span>
(capture-holdout experiment 20, mission score 0.078), unmodified.</li>
<li><b>The camera:</b> a 128&thinsp;&times;&thinsp;128 m crop of Berlin&rsquo;s
2024 aerial survey at the aircraft&rsquo;s true position and true heading
&mdash; a photograph the model never saw in training, the same held-out
condition the frozen eval measures.</li>
<li><b>The thresholds:</b> confidence gate and the 100 m usable radius
imported from the frozen scorer, not re-chosen.</li>
<li><b>The fix schedule:</b> every 6 s in cruise, 2 s on approach &mdash;
the deployment spec&rsquo;s own numbers.</li>
</ul></div>
<div><h3>Simulated</h3><ul>
<li><b>The airframe:</b> point-mass fixed-wing, 16 m/s, bank-limited turns
&mdash; kinematics, not aerodynamics.</li>
<li><b>The disturbances:</b> constant wind 2&ndash;6 m/s + gusts, gyro bias,
airspeed error &mdash; all invisible to the navigator.</li>
<li><b>The navigator:</b> dead reckoning + magnetometer heading (standard FC
hardware; vision stays the only <i>position</i> source), fixes blended by
staleness. False fixes are accepted like any other.</li>
<li><b>The mission logic:</b> written for this simulator in the manner of
ArduPlane, not ArduPilot&rsquo;s code: a waypoint counts as passed once
the aircraft crosses the line through it; survey lines and the river are
followed as lines; the orbit is a guidance field; a cold-started aircraft
circles until its first confident fix.</li>
</ul></div>
<div><h3>Best-case caveats</h3><ul>
<li>The camera is a <b>different aerial survey</b> than the training ones,
not a real drone camera: no lens distortion, motion blur, night or sensor
noise. It tests a photograph the model never saw, not sim-to-real
transfer.</li>
<li>Altitude is fixed at the trained 1 m/px scale; tilt and altitude
robustness are unmeasured, here as in the eval.</li>
</ul></div>
<div><h3>The control</h3><ul>
<li>Same 100 flights with vision disabled: <b>0 arrive within
100 m</b>, median miss {mc_dr['summary']['declared_miss_median_m']:,.0f} m.
Whatever navigation is happening here, it is the model doing it.</li>
<li>The six other missions, flown again without the camera, succeed on
{', '.join(f"{v['success_dr']:.0%}" for v in mission_data().values())} of
flights ({', '.join(v['name'].lower() for v in mission_data().values())}).</li>
</ul></div>
</div>

<p class="bl-h">The bottom line</p>
<p class="bottom-line"><span class="hl">Asked to fly over a photograph it
never saw, the champion delivers {n_within} flights in 100 to within
100 m</span> &mdash; with under a third of the false fixes the previous champion
made on its own training photograph, and every miss on the page. Across
six other missions the error never builds up; what fails a mission is
nearly always one confident wrong answer, and those cluster at a few places
on the map.</p>

</div>
<script>{charts_js(json.dumps(rows, separators=(',', ':')))}</script>
{credits_html()}</body></html>"""

    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html) / 1024:.0f} KB)")


def replay_js(data_json, rel_json):
    """The shared replay script (map, markers, controls, tooltip, autostart),
    wrapped in an IIFE so it declares NO page globals — the overview's own
    scripts already use names like `playing`, and a top-level collision is a
    SyntaxError that kills the whole script. The one deliberate export is
    window.FP, the handle charts_js() uses to follow the replay clock."""
    return "(()=>{const D=" + data_json + ";const REL=" + rel_json + ";" + r"""
const NS='http://www.w3.org/2000/svg';
const el=(t,a,p)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e};
const INK='#111111', OCHRE='#8a6a1e', ACCENT='#8c2f1f', FAINT='#9b998c', PAPER='#fffff8', MUTED='#6b6a60', RULE='#d9d5c3';

/* ---------- map (rebuilt per flight by initFlight) ---------- */
const g=document.getElementById('mapg');
const halo=w=>el('polyline',{fill:'none',stroke:PAPER,'stroke-width':w,'stroke-opacity':.75,'vector-effect':'non-scaling-stroke','stroke-linejoin':'round'},g);
let F,T,dur,truePts,estCmds,trueLine,trueHalo,estLine,estHalo,fixG,ac,hasCrops,flightLabel;
const flightHooks=[];
// start / target markers — must be findable at a glance
// S = map units per screen unit relative to the whole-box view: mission
// replays zoom the map to their own area, and every marker scales by S so it
// keeps its on-screen size.
let S=1, VB=null;   // VB = the zoomed view [x, y, w, h], null for the whole box
const lab=(x,y,t,size,italic)=>{const e=el('text',{x:x,y:y,'font-size':size,
  'font-family':'Palatino,Georgia,serif','font-style':italic?'italic':'normal',
  'font-weight':italic?'400':'600',fill:INK,'text-anchor':'middle',
  style:`paint-order:stroke;stroke:#fffff8;stroke-width:${20*S}px;stroke-linejoin:round`+(italic?'':';letter-spacing:.08em')},g);e.textContent=t;return e};
const endpoint=(x,y,color,word,dx,dy)=>{
  el('circle',{cx:x,cy:y,r:320*S,fill:PAPER,'fill-opacity':.3},g);   // spotlight
  const pulse=el('circle',{cx:x,cy:y,r:140*S,fill:'none',stroke:color,'stroke-width':18*S,opacity:.75},g);
  el('animate',{attributeName:'r',values:`${140*S};${330*S}`,dur:'2.6s',repeatCount:'indefinite'},pulse);
  el('animate',{attributeName:'opacity',values:'.75;0',dur:'2.6s',repeatCount:'indefinite'},pulse);
  el('circle',{cx:x,cy:y,r:150*S,fill:'none',stroke:color,'stroke-width':30*S},g);
  el('circle',{cx:x,cy:y,r:64*S,fill:color,stroke:PAPER,'stroke-width':26*S},g);
  if(VB){   // in a zoomed view, keep the word on the visible side of its marker
    const half=(word.length*70+60)*S;
    if(x+dx*S+half>VB[0]+VB[2]) dx=-Math.abs(dx);
    if(x+dx*S-half<VB[0]) dx=Math.abs(dx);
    if(y+dy*S-200*S<VB[1]) dy=Math.abs(dy)+160;
    if(y+dy*S+40*S>VB[1]+VB[3]) dy=-Math.abs(dy);
  }
  lab(x+dx*S,y+dy*S,word,200*S,false);
};

function initFlight(f,label){
  F=f;T=f.track;dur=T[T.length-1][0];
  flightLabel=label;
  hasCrops=F.fixes.some(x=>x.crop);
  playing=false;playBtn.textContent='Play';cur=0;lastTs=0;
  g.replaceChildren();
  {const svg=document.getElementById('map');
   // Every flight is framed at one ground scale (M_PER_PX metres per screen
   // pixel, the old whole-box look), so a wider window shows MORE of Berlin
   // rather than a bigger copy of the same ground. The basemap extends past
   // the model's box (sim/fetch_wide_basemap.py); the view stays inside it.
   const M_PER_PX=7.7, Wpx=Math.max(240, svg.parentElement.clientWidth||840), EX=D.ext;
   const xs=[],ys=[];const add=q=>{xs.push(q[0]);ys.push(q[1]);};
   T.forEach(p=>add([p[1],p[2]]));add(F.start_px);add(F.target_px);(F.waypoints_px||[]).forEach(add);
   if(F.loiter){const [cx,cy]=F.loiter.center,R=F.loiter.radius_m+100;add([cx-R,cy-R]);add([cx+R,cy+R]);}
   if(F.cold_start)add([D.w/2,D.h/2]);
   const pad=450,x0=Math.min(...xs)-pad,x1=Math.max(...xs)+pad,y0=Math.min(...ys)-pad,y1=Math.max(...ys)+pad;
   const vw=Math.min(EX.w,Math.max(x1-x0,Wpx*M_PER_PX)), vh=Math.min(EX.h,Math.max(y1-y0,vw*0.5));
   const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
   const vx=clamp((x0+x1-vw)/2,EX.x,EX.x+EX.w-vw), vy=clamp((y0+y1-vh)/2,EX.y,EX.y+EX.h-vh);
   svg.setAttribute('viewBox',`${vx} ${vy} ${vw} ${vh}`);VB=[vx,vy,vw,vh];
   // markers keep their on-screen size: S is map metres per screen pixel
   // relative to the original whole-box view (6,939 m across 840 px)
   S=(vw/Wpx)/(D.w/840);}
  const MK=F.mission||'a_to_b';
  if(MK!=='loiter')el('circle',{cx:F.target_px[0],cy:F.target_px[1],r:100,fill:'none',stroke:INK,'stroke-dasharray':'4 4','vector-effect':'non-scaling-stroke','stroke-width':1.3},g);
  // the mission's plan, faint and under everything: route, orbit, believed start
  const planLine=(pts)=>{const a=pts.map(q=>q[0]+','+q[1]).join(' ');
    el('polyline',{points:a,fill:'none',stroke:PAPER,'stroke-width':4,'stroke-opacity':.7,'vector-effect':'non-scaling-stroke','stroke-linejoin':'round'},g);
    el('polyline',{points:a,fill:'none',stroke:MUTED,'stroke-width':1.6,'stroke-dasharray':'8 6','vector-effect':'non-scaling-stroke','stroke-linejoin':'round'},g);};
  if(F.waypoints_px)planLine([F.start_px].concat(F.waypoints_px));
  if(F.loiter){const [cx,cy]=F.loiter.center,R=F.loiter.radius_m;
    el('circle',{cx,cy,r:R,fill:'none',stroke:PAPER,'stroke-width':4,'stroke-opacity':.7,'vector-effect':'non-scaling-stroke'},g);
    el('circle',{cx,cy,r:R,fill:'none',stroke:MUTED,'stroke-width':1.6,'stroke-dasharray':'8 6','vector-effect':'non-scaling-stroke'},g);
    el('circle',{cx,cy,r:R+100,fill:'none',stroke:FAINT,'stroke-width':1.1,'stroke-dasharray':'2 5','vector-effect':'non-scaling-stroke'},g);}
  if(F.cold_start){const bx=D.w/2,by=D.h/2;
    // ochre = the aircraft's belief, as everywhere on this page
    el('line',{x1:bx,y1:by,x2:F.start_px[0],y2:F.start_px[1],stroke:PAPER,'stroke-width':4,'stroke-opacity':.7,'vector-effect':'non-scaling-stroke'},g);
    el('line',{x1:bx,y1:by,x2:F.start_px[0],y2:F.start_px[1],stroke:OCHRE,'stroke-width':2,'stroke-dasharray':'6 5','vector-effect':'non-scaling-stroke'},g);
    el('circle',{cx:bx,cy:by,r:90*S,fill:PAPER,'fill-opacity':.7,stroke:OCHRE,'stroke-width':26*S},g);
    lab(bx,by-170*S,'believed start',170*S,true);}
  truePts=T.map(p=>p[1]+','+p[2]);
  // The believed path is drawn only while the estimate is within 300 m of
  // the truth. A false fix teleports the BELIEF across the city — the
  // aircraft never flies that excursion — so drawing it as a continuous
  // line reads as physical flight. The break itself is the signal; the red
  // claim-line and x mark what caused it.
  const vis=T.map(p=>Math.hypot(p[1]-p[3],p[2]-p[4])<=300);
  estCmds=T.map((p,i)=>vis[i]?((i>0&&vis[i-1])?'L':'M')+p[3]+' '+p[4]:'');
  estHalo=el('path',{fill:'none',stroke:PAPER,'stroke-width':4.5,'stroke-opacity':.75,'vector-effect':'non-scaling-stroke','stroke-linejoin':'round'},g);
  estLine=el('path',{fill:'none',stroke:OCHRE,'stroke-width':2.2,'stroke-dasharray':'6 5','vector-effect':'non-scaling-stroke','stroke-linejoin':'round'},g);
  trueHalo=halo(5); trueLine=el('polyline',{fill:'none',stroke:INK,'stroke-width':2.4,'vector-effect':'non-scaling-stroke','stroke-linejoin':'round'},g);
  fixG=el('g',{},g);
  const WORDS={return_home:['HOME',null],loiter:['START','ORBIT'],survey:['START','END'],spree:['START','END']}[MK]||['START','TARGET'];
  if(MK==='tour'||MK==='return_home'){   // the intermediate stops, numbered
    F.waypoints_px.slice(0,-1).forEach((w,k)=>{
      if(MK==='return_home'){endpoint(w[0],w[1],ACCENT,'B',-300,-240);return;}
      el('rect',{x:w[0]-70*S,y:w[1]-70*S,width:140*S,height:140*S,fill:PAPER,stroke:INK,'stroke-width':24*S},g);
      lab(w[0],w[1]-140*S,String(k+1),190*S,false);});
  }
  endpoint(F.start_px[0],F.start_px[1],INK,WORDS[0],560,80);
  if(MK==='loiter'){   // a small centre mark: the big target rings would hide the orbit itself
    el('circle',{cx:F.target_px[0],cy:F.target_px[1],r:34*S,fill:ACCENT,stroke:PAPER,'stroke-width':14*S},g);
    lab(F.target_px[0],F.target_px[1]-F.loiter.radius_m-130*S,'ORBIT',200*S,false);
  } else if(WORDS[1])endpoint(F.target_px[0],F.target_px[1],ACCENT,MK==='tour'?'4':WORDS[1],-560,-240);
  ac=el('polygon',{points:'0,-160 95,130 0,70 -95,130',fill:INK,stroke:PAPER,'stroke-width':34},g);
  cam.style.display=hasCrops?'':'none';
  camcv.style.display=hasCrops?'none':'';
  miniImg.style.display=hasCrops?'':'none';
  miniCv.style.display=hasCrops?'none':'';
  miniT.innerHTML='&mdash;';
  camsub.innerHTML=hasCrops?'128&thinsp;&times;&thinsp;128 px &middot; 1 m/px &middot; ~100 m AGL'
    :'reconstructed view &middot; 2 m/px (frames saved only for the showcase)';
  if(!hasCrops&&!zoomImg.src)zoomImg.src=REL+'assets/sim/berlin-zoom.jpg';
  flightHooks.forEach(h=>h(F));
  render(0);
}

async function loadFlight(val){
  val=String(val);
  const isM=val.includes(':'), kind=isM?val.split(':')[0]:'a_to_b';
  setMission(kind);
  if(flightSel)flightSel.value=val;
  if(val==='showcase'){initFlight(D.ab,D.ab_label);startPlay();return;}
  if(val===D.show){initFlight(D.flight,D.label);startPlay();return;}
  if(isM){
    const id=val.split(':')[1], M=D.missions[kind];
    const f=await fetch(REL+'assets/sim/missions/'+kind+'/f'+id+'.json').then(r=>r.json());
    const row=M.rows.find(r=>String(r.id)===id);
    initFlight(f,M.name+' &middot; flight '+id+'<br>'+(f.success
      ?'<span class="lab-sub">mission succeeded</span>'
      :'<span class="lab-sub" style="color:'+ACCENT+'">mission failed</span>'));
    startPlay();return;
  }
  const f=await fetch(REL+'assets/sim/flights/f'+val+'.json').then(r=>r.json());
  const miss=f.result==='arrived'?f.declared_miss_m:f.final_true_dist_m;
  initFlight(f,'A &rarr; B &middot; flight '+f.flight_id+'<br><span class="lab-sub">'+
    (f.result==='timeout'?'timed out, ':'lands ')+fmt(miss)+' from target</span>');
  startPlay();
}
function startPlay(){playing=true;lastTs=0;playBtn.textContent='Pause';requestAnimationFrame(tick);}

const fixMark=(fx,i)=>{
  const [x,y]=fx.true;let m;
  if(fx.outcome==='usable') m=el('circle',{cx:x,cy:y,r:28*S,fill:INK,stroke:PAPER,'stroke-width':10*S},fixG);
  else if(fx.outcome==='abstain') m=el('circle',{cx:x,cy:y,r:30*S,fill:PAPER,'fill-opacity':.5,stroke:FAINT,'stroke-width':14*S},fixG);
  else if(fx.outcome==='off_map') m=el('rect',{x:x-30*S,y:y-30*S,width:60*S,height:60*S,fill:'none',stroke:FAINT,'stroke-width':12*S,'stroke-dasharray':`${22*S} ${14*S}`},fixG);
  else{ // false fix: ring at true pos + thin line to an x at the claimed position
    const px=fx.pred[0],py=fx.pred[1],s=52*S;
    el('line',{x1:x,y1:y,x2:px,y2:py,stroke:ACCENT,'stroke-width':1.2,'stroke-dasharray':'3 3','vector-effect':'non-scaling-stroke'},fixG);
    el('line',{x1:px-s,y1:py-s,x2:px+s,y2:py+s,stroke:PAPER,'stroke-width':46*S},fixG);
    el('line',{x1:px-s,y1:py+s,x2:px+s,y2:py-s,stroke:PAPER,'stroke-width':46*S},fixG);
    el('line',{x1:px-s,y1:py-s,x2:px+s,y2:py+s,stroke:ACCENT,'stroke-width':22*S},fixG);
    el('line',{x1:px-s,y1:py+s,x2:px+s,y2:py-s,stroke:ACCENT,'stroke-width':22*S},fixG);
    m=el('circle',{cx:x,cy:y,r:44*S,fill:'none',stroke:ACCENT,'stroke-width':20*S},fixG);
  }
  const hit=el('circle',{cx:x,cy:y,r:120*S,fill:'transparent'},fixG);
  hit.style.cursor='pointer';
  hit.addEventListener('mousemove',e=>tip(e,fixTipHtml(fx,i)));
  hit.addEventListener('mouseleave',hideTip);
  hit.addEventListener('click',()=>{seek(fx.t);});
  return m;
};
const fixTipHtml=(fx,i)=>{
  const o={usable:'usable fix',abstain:'abstained',false_fix:'FALSE FIX',off_map:'off the map'}[fx.outcome];
  let h=`<div class="tt-h">t = ${fx.t.toFixed(0)} s — ${o}</div>`;
  if(fx.conf!==undefined)h+=`<div class="tt-m">confidence ${fx.conf.toFixed(2)}`+(fx.err_m!==undefined?` · answer ${fmt(fx.err_m)} off`:'')+`</div>`;
  return h;
};
const fmt=m=>m>=1000?(m/1000).toFixed(1)+' km':m.toFixed(0)+' m';

/* ---------- replay ---------- */
let cur=0, playing=false, lastTs=0;
const scrub=document.getElementById('scrub'), tread=document.getElementById('tread'),
      playBtn=document.getElementById('play'), speedSel=document.getElementById('speed'),
      cam=document.getElementById('cam'), camout=document.getElementById('camout'),
      camsub=document.getElementById('camsub'), read=document.getElementById('read'),
      flightSel=document.getElementById('flightsel'), camcv=document.getElementById('camcv'),
      miniImg=document.getElementById('miniimg'), miniCv=document.getElementById('minicv'),
      miniT=document.getElementById('minit'), missionSel=document.getElementById('missionsel');
// the flight list follows the mission picker; A -> B restores the original list
const AB_HTML=flightSel?flightSel.innerHTML:'';
if(flightSel)flightSel.dataset.kind='a_to_b';
function setMission(kind){
  if(!flightSel||!missionSel||!D.missions)return;
  missionSel.value=kind;
  if(flightSel.dataset.kind===kind)return;
  flightSel.dataset.kind=kind;
  if(kind==='a_to_b'){flightSel.innerHTML=AB_HTML;return;}
  const M=D.missions[kind], ok=M.rows.filter(r=>r.ok), bad=M.rows.filter(r=>!r.ok);
  const grp=(l,rs)=>rs.length?`<optgroup label="${l} (${rs.length})">`+
    rs.map(r=>`<option value="${kind}:${r.id}">${r.lab}</option>`).join('')+'</optgroup>':'';
  flightSel.innerHTML=grp('mission succeeded',ok)+grp('mission failed',bad);
}
if(missionSel)missionSel.addEventListener('change',e=>{const k=e.target.value;
  loadFlight(k==='a_to_b'?'showcase':k+':'+D.missions[k].example);});
// Camera-view reconstruction for flights whose frames weren't archived:
// crop the 2 m/px reference raster at the fix's true position, rotated so
// the aircraft's direction of travel points up — same convention as the
// real crops (heading approximated from the track, so it omits wind crab).
const zoomImg=new Image(); let zoomReady=false;
zoomImg.onload=()=>{zoomReady=true;render(cur);};
function drawCam(fx){
  if(!zoomReady){if(!zoomImg.src)zoomImg.src=REL+'assets/sim/berlin-zoom.jpg';return;}
  const i=idxAt(fx.t), p=T[i], q=T[Math.max(0,i-1)];
  const psi=Math.atan2(p[1]-q[1],-(p[2]-q[2]));
  const zs=zoomImg.naturalWidth/D.w, ctx=camcv.getContext('2d');
  ctx.clearRect(0,0,128,128);ctx.save();
  ctx.translate(64,64);ctx.rotate(-psi);
  ctx.drawImage(zoomImg,(fx.true[0]-91)*zs,(fx.true[1]-91)*zs,182*zs,182*zs,-91,-91,182,182);
  ctx.restore();
  const mctx=miniCv.getContext('2d');
  mctx.clearRect(0,0,128,128);mctx.drawImage(camcv,0,0);
}
const idxAt=t=>{let lo=0,hi=T.length-1;while(lo<hi){const m=(lo+hi+1>>1);if(T[m][0]<=t)lo=m;else hi=m-1;}return lo;};
function render(t){
  cur=Math.max(0,Math.min(dur,t));
  const i=idxAt(cur);
  trueLine.setAttribute('points',truePts.slice(0,i+1).join(' '));
  trueHalo.setAttribute('points',truePts.slice(0,i+1).join(' '));
  const dEst=estCmds.slice(0,i+1).join(' ');
  estLine.setAttribute('d',dEst);
  estHalo.setAttribute('d',dEst);
  const p=T[i], q=T[Math.max(0,i-1)];
  const angle=Math.atan2(p[1]-q[1],-(p[2]-q[2]))*180/Math.PI;
  ac.setAttribute('transform',`translate(${p[1]},${p[2]}) rotate(${angle}) scale(${0.95*S})`);
  // fixes reached so far
  fixG.replaceChildren();
  let last=null,counts={usable:0,abstain:0,false_fix:0};
  F.fixes.forEach((fx,j)=>{if(fx.t<=cur){fixMark(fx,j);last=[fx,j];if(counts[fx.outcome]!==undefined)counts[fx.outcome]++;}});
  if(last){
    const [fx,j]=last;
    if(fx.crop){cam.src=REL+(F.crop_base||'assets/sim/crops/')+fx.crop;miniImg.src=cam.src;}
    if(!hasCrops&&fx.outcome!=='off_map')drawCam(fx);
    miniT.textContent=fx.t.toFixed(0)+' s · '+({usable:'usable',abstain:'abstained',false_fix:'false fix',off_map:'off map'}[fx.outcome]);
    miniT.style.color=fx.outcome==='false_fix'?ACCENT:(fx.outcome==='usable'?INK:MUTED);
    const oCls=fx.outcome, oTxt={usable:'usable fix',abstain:'abstained',false_fix:'false fix',off_map:'off the map'}[oCls];
    camout.innerHTML=`<span>t = ${fx.t.toFixed(0)} s</span><span><b class="${oCls}">${oTxt}</b></span>`+
      `<span style="color:${MUTED}">${fx.err_m!==undefined&&fx.outcome!=='abstain'?'answer '+fmt(fx.err_m)+' off':'&nbsp;'}</span>`;
  } else { camout.innerHTML='<span>&mdash; no fix yet</span><span>&nbsp;</span><span>&nbsp;</span>'; }
  const err=Math.hypot(p[1]-p[3],p[2]-p[4]);
  const dist=Math.hypot(F.target_px[0]-p[1],F.target_px[1]-p[2]);
  const toWord={loiter:'to orbit centre',return_home:'to home',survey:'to end of grid',spree:'to end of route',tour:'to last spot'}[F.mission]||'to target';
  read.innerHTML=`<div class="rd-h">${flightLabel}</div><dl class="rd">`+
    `<dt>estimate off by</dt><dd>${fmt(err)}</dd>`+
    `<dt>${toWord}</dt><dd>${fmt(dist)}</dd>`+
    `<dt>usable fixes</dt><dd>${counts.usable}</dd>`+
    `<dt>abstained</dt><dd>${counts.abstain}</dd>`+
    `<dt>false fixes</dt><dd>${counts.false_fix}</dd></dl>`;
  scrub.value=Math.round(1000*cur/dur);
  tread.textContent=cur.toFixed(1)+' s';
  if(window.FP&&window.FP.onTick)window.FP.onTick(cur);
}
function tick(ts){
  if(!playing)return;
  if(lastTs)render(cur+(ts-lastTs)/1000*(+speedSel.value));
  lastTs=ts;
  if(cur>=dur){playing=false;playBtn.textContent='Replay';return;}
  requestAnimationFrame(tick);
}
playBtn.addEventListener('click',()=>{
  if(playing){playing=false;playBtn.textContent='Play';return;}
  if(cur>=dur)cur=0;
  startPlay();
});
if(flightSel)flightSel.addEventListener('change',e=>loadFlight(e.target.value));
scrub.addEventListener('input',()=>{playing=false;playBtn.textContent='Play';render(+scrub.value/1000*dur);});
const seek=t=>{playing=false;playBtn.textContent='Play';render(t);};

/* ---------- tooltip ---------- */
const tipEl=document.getElementById('tip');
function tip(e,html){tipEl.innerHTML=html;tipEl.style.display='block';
  const w=tipEl.offsetWidth;
  tipEl.style.left=Math.min(window.innerWidth-w-8,e.clientX+14)+'px';
  tipEl.style.top=(e.clientY+16)+'px';}
function hideTip(){tipEl.style.display='none';}

window.FP={el,fmt,tip,hideTip,seek,loadFlight,flightHooks,flight:()=>F,onTick:null,
           INK,OCHRE,ACCENT,FAINT,PAPER,MUTED,RULE};
initFlight(D.flight,D.label);
setMission(D.show.split(':')[0]); if(flightSel)flightSel.value=D.show;
// autostart the replay the first time the map comes into view
const io=new IntersectionObserver(es=>{
  if(es.some(e=>e.isIntersecting)&&!playing&&cur===0){
    startPlay();
    io.disconnect();
  }
},{threshold:0.45});
io.observe(document.getElementById('map'));
})();"""


def charts_js(rows_json):
    """The flight-path page's own charts (sawtooth + Monte-Carlo dot strip).
    IIFE-wrapped like replay_js; everything shared comes in through the
    window.FP handle that script exports."""
    return ("(()=>{const MCROWS=" + rows_json + ";"
            "const {el,fmt,tip,hideTip,seek,loadFlight,flightHooks,flight,"
            "INK,OCHRE,ACCENT,FAINT,PAPER,MUTED,RULE}=window.FP;" + r"""
/* ---------- sawtooth: estimation error over time (rebuilt per flight) ---------- */
function buildSaw(F){
const T=F.track, dur=T[T.length-1][0];
const idxAt=t=>{let lo=0,hi=T.length-1;while(lo<hi){const m=(lo+hi+1>>1);if(T[m][0]<=t)lo=m;else hi=m-1;}return lo;};
const CW=900, CH=290, ML=52, MR=14, MT=26, MB=34;
const errAt=i=>Math.hypot(T[i][1]-T[i][3],T[i][2]-T[i][4]);
const EMAX=Math.max(...T.map((_,i)=>errAt(i)))*1.04;
const sq=v=>Math.sqrt(Math.min(v,EMAX)/EMAX);
const X=t=>ML+(CW-ML-MR)*t/dur, Y=v=>MT+(CH-MT-MB)*(1-sq(v));
const box=document.getElementById('chartbox'); box.replaceChildren();
const chart=el('svg',{viewBox:`0 0 ${CW} ${CH}`,role:'img','aria-label':'estimation error over time'},box);
const gl=[[10,'10'],[50,'50'],[100,'100 m'],[500,'500'],[2000,'2 km'],[5000,'5 km']]
  .filter(([v])=>v<=EMAX);
gl.forEach(([v,l])=>{
  const dash=v===100?'5 4':'', col=v===100?ACCENT:RULE;
  el('line',{x1:ML,x2:CW-MR,y1:Y(v),y2:Y(v),stroke:col,'stroke-width':v===100?1.2:.8,'stroke-dasharray':dash},chart);
  const tx=el('text',{x:ML-7,y:Y(v)+4,'text-anchor':'end','font-size':11.5,fill:v===100?ACCENT:MUTED,'font-family':'Palatino,Georgia,serif'},chart);tx.textContent=l;
});
const tstep=dur>400?120:60;
Array.from({length:Math.floor(dur/tstep)+1},(_,k)=>k*tstep).forEach(tv=>{
  el('line',{x1:X(tv),x2:X(tv),y1:CH-MB,y2:CH-MB+4,stroke:MUTED,'stroke-width':1},chart);
  const tx=el('text',{x:X(tv),y:CH-MB+17,'text-anchor':'middle','font-size':11.5,fill:MUTED,'font-family':'Palatino,Georgia,serif'},chart);tx.textContent=tv+' s';
});
const errPts=T.map((p,i)=>X(p[0])+','+Y(errAt(i))).join(' ');
el('polyline',{points:errPts,fill:'none',stroke:INK,'stroke-width':1.8,'stroke-linejoin':'round'},chart);
// fix ticks along the top
F.fixes.forEach(fx=>{
  const x=X(fx.t);
  if(fx.outcome==='usable')el('line',{x1:x,x2:x,y1:MT-14,y2:MT-4,stroke:INK,'stroke-width':1.6},chart);
  else if(fx.outcome==='abstain')el('circle',{cx:x,cy:MT-9,r:3,fill:'none',stroke:FAINT,'stroke-width':1.4},chart);
  else if(fx.outcome==='false_fix')el('circle',{cx:x,cy:MT-9,r:4,fill:'none',stroke:ACCENT,'stroke-width':2},chart);
});
const tl=el('text',{x:ML,y:12,'font-size':11,fill:MUTED,'font-family':'Palatino,Georgia,serif'},chart);
tl.textContent='fixes:  | usable   ○ abstained   ○ false';
const cursor=el('line',{x1:X(0),x2:X(0),y1:MT,y2:CH-MB,stroke:ACCENT,'stroke-width':1,'stroke-opacity':.7},chart);
window.FP.onTick=t=>{cursor.setAttribute('x1',X(t));cursor.setAttribute('x2',X(t));};
// hover: nearest point + seek on click
const hitRect=el('rect',{x:ML,y:MT,width:CW-ML-MR,height:CH-MT-MB,fill:'transparent'},chart);
hitRect.style.cursor='crosshair';
hitRect.addEventListener('mousemove',e=>{
  const r=chart.getBoundingClientRect(), t=(e.clientX-r.left)/r.width*CW;
  const tt=Math.max(0,Math.min(dur,(t-ML)/(CW-ML-MR)*dur)), i=idxAt(tt);
  tip(e,`<div class="tt-h">t = ${T[i][0].toFixed(0)} s</div><div class="tt-m">estimate ${fmt(errAt(i))} from the truth</div>`);
});
hitRect.addEventListener('mouseleave',hideTip);
hitRect.addEventListener('click',e=>{
  const r=chart.getBoundingClientRect(), t=(e.clientX-r.left)/r.width*CW;
  seek(Math.max(0,Math.min(dur,(t-ML)/(CW-ML-MR)*dur)));
});
}
buildSaw(flight());
flightHooks.push(buildSaw);

/* ---------- Monte Carlo dot strip ---------- */
const MW=900, MH=210, MML=16, MMR=16, MMT=40, MMB=40;
const LMIN=Math.log10(5), LMAX=Math.log10(12000);
const MX=m=>MML+(MW-MML-MMR)*(Math.log10(Math.max(m,5))-LMIN)/(LMAX-LMIN);
const mc=el('svg',{viewBox:`0 0 ${MW} ${MH}`,role:'img','aria-label':'true miss distance of all 100 flights, log scale'},document.getElementById('mcbox'));
[[10,'10 m'],[30,'30'],[100,'100 m'],[300,'300'],[1000,'1 km'],[3000,'3'],[10000,'10 km']].forEach(([v,l])=>{
  const hot=v===100;
  el('line',{x1:MX(v),x2:MX(v),y1:MMT-6,y2:MH-MMB+6,stroke:hot?ACCENT:RULE,'stroke-width':hot?1.3:.8,'stroke-dasharray':hot?'5 4':''},mc);
  const tx=el('text',{x:MX(v),y:MH-MMB+22,'text-anchor':'middle','font-size':11.5,fill:hot?ACCENT:MUTED,'font-family':'Palatino,Georgia,serif'},mc);tx.textContent=l;
});
const t1=el('text',{x:MX(9),y:MMT-16,'font-size':11.5,'font-style':'italic',fill:MUTED,'font-family':'Palatino,Georgia,serif'},mc);t1.textContent='true miss at declared arrival →';
// deterministic vertical spread so dots don't stack
const seen=[];
MCROWS.slice().sort((a,b)=>a.miss-b.miss).forEach(r=>{
  const x=MX(r.miss);
  let lane=0;while(seen.some(s=>s.lane===lane&&Math.abs(s.x-x)<11))lane++;
  seen.push({x,lane});
  const y=MH-MMB-14-lane*13;
  const a={cx:x,cy:y,r:4.6};
  let m;
  if(r.result==='timeout')m=el('rect',{x:x-4,y:y-4,width:8,height:8,fill:'none',stroke:ACCENT,'stroke-width':1.6},mc);
  else if(r.within)m=el('circle',{...a,fill:INK,stroke:PAPER,'stroke-width':1},mc);
  else m=el('circle',{...a,fill:'none',stroke:ACCENT,'stroke-width':2},mc);
  const hit=el('circle',{cx:x,cy:y,r:9,fill:'transparent'},mc);
  hit.style.cursor='pointer';
  hit.addEventListener('mousemove',e=>tip(e,
    `<div class="tt-h">flight ${r.id} — ${r.result==='timeout'?'timed out':'declared arrival'}</div>`+
    `<div class="tt-m">true miss ${fmt(r.miss)} · ${r.u} usable · ${r.a} abstained · ${r.ff} false`+
    (r.ff_hold?` (${r.ff_hold} on never-trained ground)`:'')+
    `</div><div class="tt-m" style="font-style:italic">click to replay this flight</div>`));
  hit.addEventListener('mouseleave',hideTip);
  hit.addEventListener('click',()=>{hideTip();loadFlight(String(r.id));
    document.getElementById('map').scrollIntoView({behavior:'smooth',block:'center'});});
});
})();""")


if __name__ == "__main__":
    main()
