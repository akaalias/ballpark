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

# Styles for the shared "One flight, replayed" section (embedded on both the
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
.fp-stage{display:flex;gap:22px;align-items:flex-start;margin-top:12px}
.fp-map{flex:1 1 640px;min-width:0;position:relative}
.fp-map svg{display:block;width:100%;height:auto;border:1px solid var(--rule)}
.fp-side{flex:0 0 158px;position:sticky;top:14px}
.fp-cam{border:1px solid var(--rule);background:#f7f4e6;padding:10px}
/* shown at (near-)native size — stretching the 128 px crops, and especially
   the 64 px reconstructions, to panel width just reads as blur */
.fp-cam img,.fp-cam canvas{width:100%;max-width:132px;height:auto;
  aspect-ratio:1/1;image-rendering:auto;display:block;margin:0 auto;
  border:1px solid var(--rule-soft)}
.fp-cam .cam-h{font:600 10.5px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin:0 0 7px}
.fp-cam .cam-out{margin:8px 0 0;font:13px/1.5 var(--serif)}
.fp-cam .cam-out b.usable{color:var(--ink)}
.fp-cam .cam-out b.abstain{color:var(--faint)}
.fp-cam .cam-out b.false_fix{color:var(--accent)}
.fp-cam .cam-sub{margin:2px 0 0;font:12.5px/1.5 var(--serif);color:var(--muted)}
.fp-read{margin-top:12px;font:13px/1.7 var(--serif);color:var(--muted)}
.fp-read b{color:var(--ink);font-variant-numeric:lining-nums tabular-nums}
.fp-controls{display:flex;gap:12px;align-items:center;margin:12px 0 0;
  font:13px var(--serif);color:var(--muted);flex-wrap:wrap}
.fp-controls #flightsel{max-width:46%}
.fp-controls button{font:600 12px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.07em;color:var(--paper);
  background:var(--ink);border:none;padding:7px 16px;cursor:pointer}
.fp-controls button:hover{background:var(--accent)}
.fp-controls input[type=range]{flex:1;accent-color:#8c2f1f}
.fp-controls select{font:13px var(--serif);color:var(--ink);background:var(--paper);
  border:1px solid var(--rule);padding:3px 6px}
.fp-controls .t-read{font-variant-numeric:lining-nums tabular-nums;min-width:88px;text-align:right}
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
table.fp-fail th{font:600 10.5px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
  border-bottom:1px solid var(--rule);padding:4px 14px;text-align:left}
table.fp-fail td{border-bottom:1px solid var(--rule-soft);padding:5px 14px}
table.fp-fail td.num{font-variant-numeric:lining-nums tabular-nums}
table.fp-fail .hold{color:var(--ochre);font-size:12.5px}
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
    """Downsampled, desaturated orthophoto so the overlays carry the page.

    The source raster (data/, gitignored) only exists on a research machine;
    the derived basemap is committed in assets/sim/, so CI — which re-renders
    every page via build_site.sh — reuses it rather than regenerating."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "berlin-base.jpg"
    src = REPO_ROOT / "data" / "berlin" / "relight" / "asis.png"
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


def flight_selector_html():
    """<select> over the showcase + all 100 Monte-Carlo flights, grouped by
    outcome, labels carrying the true miss so failures are browsable too."""
    rows = mc_compact(json.loads(MC.read_text()))
    def opt(r):
        return (f"<option value='{r['id']}'>flight {r['id']} &middot; "
                f"{'timed out' if r['result'] == 'timeout' else 'lands'} "
                f"{r['miss']:,.0f} m out &middot; {r['ff']} false</option>")
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
            "&middot; lands 25.5 m out</option>" + body + "</select>")


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
    """The complete "One flight, replayed" section — heading, narrative,
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
    first_crop = next((fx["crop"] for fx in showcase["fixes"] if "crop" in fx), "")
    meta_w, meta_h = 6939, 6828  # berlin raster (data/berlin/meta.json)
    data = {
        "w": meta_w, "h": meta_h,
        "label": "showcase &middot; lands 25.5 m out",
        "flight": {k: showcase[k] for k in
                   ("start_px", "target_px", "direct_m", "flight_s", "wind",
                    "declared_miss_m", "fix_counts", "fixes", "track")},
    }
    return f"""
{rsec(n, "One flight, replayed",
      "Corner to corner — 9.2 km in ten minutes, seven false fixes survived, arrival at 25.5 m.")}
<div class="fp-note">
<p>North-west corner to south-east corner, 9 km. At
<b>t&thinsp;=&thinsp;85&thinsp;s</b> a
<b style="color:var(--accent)">false fix</b> &mdash; barely confident,
<b>6.5 km wrong</b> (the red line points to its claim, marked
<span style="color:var(--accent)">&times;</span>) &mdash; hijacks the
navigator&rsquo;s belief. The aircraft never flies to the claimed spot:
only its <i>estimate</i> teleports there, so the dashed believed path
simply breaks off and rejoins once usable fixes haul the belief back
within 300 m of the truth. A late burst of false fixes on final approach
does little harm: by then the estimate is fresh, so wrong answers get
little weight. The flight lands <b>25.5 m</b> from the target.</p>
<p style="font-size:13px;color:var(--muted)">Seed 6, disclosed: three of
six seeds on this route overshot the corner out of the mapped box and were
lost &mdash; beyond the edge there is no imagery and no way back.
{record_note}</p></div>

<div class="fp-legend">
<span><i class="k-true"></i>true path</span>
<span><i class="k-est"></i>believed path &mdash; drawn only while the estimate is within 300 m of the truth</span>
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
<image href="{rel}assets/sim/berlin-base.jpg" x="0" y="0" width="{meta_w}" height="{meta_h}"/>
<g id="mapg"></g>
</svg>
<div class="fp-cam-mini" id="cammini">
<img id="miniimg" src="{rel}assets/sim/crops/{first_crop}" alt="camera crop at the latest fix" width="128" height="128">
<canvas id="minicv" width="128" height="128" style="display:none"></canvas>
<span class="mini-t" id="minit">&mdash;</span>
</div>
<div class="fp-controls">
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
<img id="cam" src="{rel}assets/sim/crops/{first_crop}" alt="camera crop at the latest fix" width="128" height="128">
<canvas id="camcv" width="128" height="128" style="display:none"></canvas>
<p class="cam-out" id="camout">&mdash;</p>
<p class="cam-sub" id="camsub">128&thinsp;&times;&thinsp;128 px &middot; 1 m/px &middot; ~100 m AGL</p>
</div>
<div class="fp-read" id="read"></div>
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

    rows = mc_compact(mc)
    s = mc["summary"]
    n_within = s["n_arrived_within_100m"]
    failures = sorted((r for r in rows if not r["within"]), key=lambda r: r["miss"])
    ff_total = s["total_fixes"]["false_fix"]
    ff_hold = sum(r["ff_hold"] for r in rows)
    n_conf_or_abst = sum(s["total_fixes"][k] for k in ("usable", "abstain", "false_fix"))
    ff_rate_nonhold = 100.0 * (ff_total - ff_hold) / n_conf_or_abst
    misses = sorted(r["miss"] for r in rows if r["result"] == "arrived")
    p10, p90 = misses[9], misses[89] if len(misses) > 89 else misses[-1]

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
             'A virtual fixed-wing UAV flies Berlin from a to b with the champion model as its only position source. 91 of 100 flights arrive within 100 m; the failures are shown, not averaged away.')}
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
own estimate and scored against the truth.</p></header>

<div class="fp-wrap">

<div class="stat-hero"><b class="num">{n_within}<span style="color:var(--faint)">/100</span></b>
<span>flights arrive within 100 m of the target</span></div>
<div class="stats">
<div class="stat"><b class="num">{s['declared_miss_median_m']:.0f} m</b>
<span>median miss at declared arrival<br>(p10 {p10:.0f} m &middot; p90 {p90:.0f} m)</span></div>
<div class="stat"><b class="num">0<span style="color:var(--faint)">/100</span></b>
<span>arrive without vision fixes<br>(dead-reckoning control, median miss {mc_dr['summary']['declared_miss_median_m']:,.0f} m)</span></div>
<div class="stat"><b class="num">{len(failures)}</b>
<span>flights miss beyond 100 m &mdash;<br>every one shown below</span></div>
</div>
<p class="hero-sub">100 simulated flights, random start and target &ge;2 km
apart, random 2&ndash;6 m/s wind. Each vision fix is the real exported
champion (<span class="mono">berlin.onnx</span>) run on a real orthophoto
crop at the aircraft&rsquo;s true position and heading, classified with the
frozen scorer&rsquo;s own thresholds.</p>

{flight_section("../", "The 100-flight record below uses random routes, failures included.")}

{rsec(2, "The estimate against the truth",
      "How far the navigator's belief is from the aircraft's true position, second by second.")}
<div class="fp-note"><p>Between fixes the error grows &mdash; wind the
navigator cannot see. Each <b>usable fix</b> snaps it back down. The
<b style="color:var(--accent)">false fix</b> at t&thinsp;=&thinsp;85 s is
the spike: one confident wrong answer worth over 5 km of injected error
&mdash; the reason the mission score prices false fixes so high. The late
cluster of false fixes barely registers: a fresh estimate gives a wrong
answer little weight. Square-root vertical scale, so the 20&ndash;60 m
cruise detail and the kilometre spike fit one honest axis; the dashed line
is the 100 m usable radius.</p></div>
<div class="fp-chart" id="chartbox"></div>

{rsec(3, "A hundred flights",
      "Every Monte-Carlo flight's miss distance — arrivals, overshoots, and the three catastrophes.")}
<div class="fp-note">
<p>Each dot is one flight&rsquo;s true distance from target at the moment it
declared arrival (hollow&nbsp;= timed out, plotted at final distance).
{n_within} land inside the 100 m line. Six miss moderately
(120&ndash;314 m). Three fail catastrophically &mdash; and all three had
their <b>target on or beside the never-trained holdout ground</b>: the
~3% of Berlin that the evaluation harness deliberately withholds from
training as a diagnostic. Over that ground the model hallucinates
confidently &mdash; {ff_hold} of {ff_total} false fixes across all
100 flights stand on it. Excluding it, the in-flight false-fix rate is
{ff_rate_nonhold:.2f}%, consistent with the 0.5% measured by the frozen
eval. A deployed model would train on 100% of its box, so this exact
failure is an artifact of simulating over the research configuration
&mdash; reported as flown, not filtered out.</p></div>
<div class="fp-chart" id="mcbox"></div>
<table class="fp-fail">
<thead><tr><th>flight</th><th>ended</th><th>true miss</th>
<th>false fixes</th><th>target on never-trained ground</th></tr></thead>
<tbody>{failure_rows}</tbody></table>

{rsec(4, "What is real here, and what is simulated",
      "The test is honest only if its seams are visible.")}
<div class="fp-honest">
<div><h3>Real</h3><ul>
<li><b>The model:</b> the exported champion <span class="mono">berlin.onnx</span>
(experiment 5, mission score 0.040), unmodified.</li>
<li><b>The camera:</b> a 128&thinsp;&times;&thinsp;128 m orthophoto crop at the
aircraft&rsquo;s true position and true heading &mdash; the same off-lattice,
rotated condition the frozen eval measures.</li>
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
</ul></div>
<div><h3>Best-case caveats</h3><ul>
<li>The camera renders from the <b>same reference imagery the model
memorized</b> &mdash; this validates the navigation loop, not
sim-to-real transfer.</li>
<li>Altitude is fixed at the trained 1 m/px scale; tilt and altitude
robustness are unmeasured, here as in the eval.</li>
</ul></div>
<div><h3>The control</h3><ul>
<li>Same 100 flights with vision disabled: <b>0 arrive within
100 m</b>, median miss {mc_dr['summary']['declared_miss_median_m']:,.0f} m.
Whatever navigation is happening here, it is the model doing it.</li>
</ul></div>
</div>

<p class="bl-h">The bottom line</p>
<p class="bottom-line"><span class="hl">Asked to actually fly, the champion
delivers 91 flights in 100 to within 100 m</span> &mdash; and every failure
is on the page, three of them explained by ground the research harness
deliberately never taught it.</p>

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
const HALO_STYLE='paint-order:stroke;stroke:#fffff8;stroke-width:20px;stroke-linejoin:round';
const lab=(x,y,t,size,italic)=>{const e=el('text',{x:x,y:y,'font-size':size,
  'font-family':'Palatino,Georgia,serif','font-style':italic?'italic':'normal',
  'font-weight':italic?'400':'600',fill:INK,'text-anchor':'middle',
  style:HALO_STYLE+(italic?'':';letter-spacing:.08em')},g);e.textContent=t;return e};
const endpoint=(x,y,color,word,dx,dy)=>{
  el('circle',{cx:x,cy:y,r:320,fill:PAPER,'fill-opacity':.3},g);   // spotlight
  const pulse=el('circle',{cx:x,cy:y,r:140,fill:'none',stroke:color,'stroke-width':18,opacity:.75},g);
  el('animate',{attributeName:'r',values:'140;330',dur:'2.6s',repeatCount:'indefinite'},pulse);
  el('animate',{attributeName:'opacity',values:'.75;0',dur:'2.6s',repeatCount:'indefinite'},pulse);
  el('circle',{cx:x,cy:y,r:150,fill:'none',stroke:color,'stroke-width':30},g);
  el('circle',{cx:x,cy:y,r:64,fill:color,stroke:PAPER,'stroke-width':26},g);
  lab(x+dx,y+dy,word,200,false);
};

function initFlight(f,label){
  F=f;T=f.track;dur=T[T.length-1][0];
  flightLabel=label;
  hasCrops=F.fixes.some(x=>x.crop);
  playing=false;playBtn.textContent='Play';cur=0;lastTs=0;
  g.replaceChildren();
  el('circle',{cx:F.target_px[0],cy:F.target_px[1],r:100,fill:'none',stroke:INK,'stroke-dasharray':'4 4','vector-effect':'non-scaling-stroke','stroke-width':1.3},g);
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
  endpoint(F.start_px[0],F.start_px[1],INK,'START',560,80);
  endpoint(F.target_px[0],F.target_px[1],ACCENT,'TARGET',-560,-240);
  ac=el('polygon',{points:'0,-160 95,130 0,70 -95,130',fill:INK,stroke:PAPER,'stroke-width':34},g);
  cam.style.display=hasCrops?'':'none';
  camcv.style.display=hasCrops?'none':'';
  miniImg.style.display=hasCrops?'':'none';
  miniCv.style.display=hasCrops?'none':'';
  miniT.innerHTML='&mdash;';
  camsub.innerHTML=hasCrops?'128&thinsp;&times;&thinsp;128 px &middot; 1 m/px &middot; ~100 m AGL'
    :'reconstructed view (2 m/px preview) &mdash; original frames are archived only for the showcase flight';
  if(!hasCrops&&!zoomImg.src)zoomImg.src=REL+'assets/sim/berlin-zoom.jpg';
  flightHooks.forEach(h=>h(F));
  render(0);
}

async function loadFlight(val){
  if(flightSel)flightSel.value=String(val);
  if(val==='showcase'){initFlight(D.flight,D.label);startPlay();return;}
  const f=await fetch(REL+'assets/sim/flights/f'+val+'.json').then(r=>r.json());
  const miss=f.result==='arrived'?f.declared_miss_m:f.final_true_dist_m;
  initFlight(f,'flight '+f.flight_id+' &middot; '+(f.result==='timeout'?'timed out':'declared arrival')+' '+fmt(miss)+' from target');
  startPlay();
}
function startPlay(){playing=true;lastTs=0;playBtn.textContent='Pause';requestAnimationFrame(tick);}

const fixMark=(fx,i)=>{
  const [x,y]=fx.true;let m;
  if(fx.outcome==='usable') m=el('circle',{cx:x,cy:y,r:28,fill:INK,stroke:PAPER,'stroke-width':10},fixG);
  else if(fx.outcome==='abstain') m=el('circle',{cx:x,cy:y,r:30,fill:PAPER,'fill-opacity':.5,stroke:FAINT,'stroke-width':14},fixG);
  else if(fx.outcome==='off_map') m=el('rect',{x:x-30,y:y-30,width:60,height:60,fill:'none',stroke:FAINT,'stroke-width':12,'stroke-dasharray':'22 14'},fixG);
  else{ // false fix: ring at true pos + thin line to an x at the claimed position
    const px=fx.pred[0],py=fx.pred[1],s=52;
    el('line',{x1:x,y1:y,x2:px,y2:py,stroke:ACCENT,'stroke-width':1.2,'stroke-dasharray':'3 3','vector-effect':'non-scaling-stroke'},fixG);
    el('line',{x1:px-s,y1:py-s,x2:px+s,y2:py+s,stroke:PAPER,'stroke-width':46},fixG);
    el('line',{x1:px-s,y1:py+s,x2:px+s,y2:py-s,stroke:PAPER,'stroke-width':46},fixG);
    el('line',{x1:px-s,y1:py-s,x2:px+s,y2:py+s,stroke:ACCENT,'stroke-width':22},fixG);
    el('line',{x1:px-s,y1:py+s,x2:px+s,y2:py-s,stroke:ACCENT,'stroke-width':22},fixG);
    m=el('circle',{cx:x,cy:y,r:44,fill:'none',stroke:ACCENT,'stroke-width':20},fixG);
  }
  const hit=el('circle',{cx:x,cy:y,r:120,fill:'transparent'},fixG);
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
      miniT=document.getElementById('minit');
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
  ac.setAttribute('transform',`translate(${p[1]},${p[2]}) rotate(${angle}) scale(0.95)`);
  // fixes reached so far
  fixG.replaceChildren();
  let last=null,counts={usable:0,abstain:0,false_fix:0};
  F.fixes.forEach((fx,j)=>{if(fx.t<=cur){fixMark(fx,j);last=[fx,j];if(counts[fx.outcome]!==undefined)counts[fx.outcome]++;}});
  if(last){
    const [fx,j]=last;
    if(fx.crop){cam.src=REL+'assets/sim/crops/'+fx.crop;miniImg.src=cam.src;}
    if(!hasCrops&&fx.outcome!=='off_map')drawCam(fx);
    miniT.textContent=fx.t.toFixed(0)+' s · '+({usable:'usable',abstain:'abstained',false_fix:'false fix',off_map:'off map'}[fx.outcome]);
    miniT.style.color=fx.outcome==='false_fix'?ACCENT:(fx.outcome==='usable'?INK:MUTED);
    const oCls=fx.outcome, oTxt={usable:'usable fix',abstain:'abstained — not confident',false_fix:'false fix — confidently wrong',off_map:'off the mapped box — no image'}[oCls];
    camout.innerHTML=`t = ${fx.t.toFixed(0)} s · <b class="${oCls}">${oTxt}</b>`+
      (fx.err_m!==undefined&&fx.outcome!=='abstain'?`<br><span style="color:${MUTED}">model's answer was ${fmt(fx.err_m)} from the truth</span>`:'');
  } else { camout.innerHTML='&mdash; no fix yet'; }
  const err=Math.hypot(p[1]-p[3],p[2]-p[4]);
  const dist=Math.hypot(F.target_px[0]-p[1],F.target_px[1]-p[2]);
  read.innerHTML=`<span style="color:#111111;font-weight:600">${flightLabel}</span><br>`+
    `estimate is <b>${fmt(err)}</b> from the truth<br>`+
    `<b>${fmt(dist)}</b> to the target (true)<br>`+
    `fixes so far: <b>${counts.usable}</b> usable · <b>${counts.abstain}</b> abstained · <b>${counts.false_fix}</b> false`;
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
