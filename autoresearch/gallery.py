"""Static HTML research dashboard rendered from the SQLite lineage
log (§7), styled after the human's prior research dashboards (Tufte cream /
ink / Palatino vocabulary: airloom log, Heuristic Kitchen dashboard,
FMDiscovery autoresearch dashboard) — including their UX patterns:
plain-language framing, column tooltips + help disclosure, and an
interlinked chart <-> table (hover highlights, click opens the detail row).

Not frozen (presentation only — the data it renders comes solely from
experiments.sqlite and runs/ artifacts).

Renders two pages sharing the airloom top-navigation pattern:
  gallery/index.html            — the research log (chart + lineage table)
  gallery/inference-paths.html  — "Proposed Inference Paths": every
                                  pre-registered architecture figure
Planned but deliberately held out for now: a lineage page and a results
page (see the publishing roadmap).

Usage: python -m autoresearch.gallery   # writes both pages
"""

import datetime
import html
import io
import json
import keyword
import math
import re
import subprocess
import tokenize
from pathlib import Path

import numpy as np

from autoresearch import evofig, workedexample
from autoresearch.db import REPO_ROOT, connect

OUT = REPO_ROOT / "gallery" / "index.html"
TARGET_M = 100.0  # berlin-slim branch milestone (main-branch target is 20 m — see CLAUDE.md "BRANCH OVERRIDE")
FAIL = 1e9
PATIENCE = 4  # mirrors loop.sh's PATIENCE default; restated here since the
              # gallery only reads the DB, never the shell env the loop ran with

# The middle era of the project ran on a rented cloud GPU billed by wall
# clock regardless of phase (design/train/score), not per-GPU-second of
# training alone — so a pod-era experiment's compute cost is its full
# duration_s at the hourly rate. The bootstrap era (ids 1-10) and the later
# local era (2026-07-23 on) have ~$0 marginal compute cost instead, so only
# the pod window carries a Cost figure.
POD_USD_PER_HR = 0.69
POD_ERA_START = "2026-07-21"
POD_ERA_END = "2026-07-23T12:00:00"   # loop moved back to a local machine


def frame_caption():
    """What one camera frame IS, per the branch's lighting buckets.

    berlin-slim collapses LIGHTING_BUCKETS to a single raw-daytime
    pass-through; the main branch renders six, of which night is the hard one
    the project is named for."""
    try:
        from pipeline.common import LIGHTING_BUCKETS
        return ("one daytime frame" if list(LIGHTING_BUCKETS) == ["asis"]
                else "one night exposure")
    except Exception:
        return "one camera frame"


def cost_str(e):
    ts = e["ts"]
    if (e["kind"] == "holdout_check" or not e.get("duration_s")
            or ts < POD_ERA_START or ts >= POD_ERA_END):
        return "—"
    return f"${e['duration_s'] / 3600 * POD_USD_PER_HR:,.2f}"


# What an experiment cost. Two very different kinds of money, added together
# for the column but never conflated in the breakdown:
#
#   agents  — the EQUIVALENT API cost of the tokens the design, implementation,
#             figure and summary agents actually consumed, as recorded by
#             Claude Code itself in runs/<id>/agent_*.json. The work ran on a
#             claude.ai Max subscription, so this is what it WOULD have cost
#             billed per token; it is not money that left the account.
#   GPU     — real spend, and only during the rented-pod window; the laptop
#             and Modal-credit eras have ~$0 marginal cost per experiment.
#
# Rows whose run directory was deleted have no agent record and show an em
# dash rather than a zero: "we didn't keep the receipt" and "it was free" are
# not the same claim.
COST_STAGE_LABEL = {"design": "design agent", "impl": "implementation agent",
                    "figure": "figure agent", "result": "summary agent"}


def fmt_usd(v):
    if v is None:
        return "—"
    return f"${v:,.2f}" if v >= 0.005 else "$0.00"


def cost_cell(r):
    """The Cost column for one experiment. An em dash means unrecorded."""
    if not r or r.get("cost_total") is None:
        return "—"
    if not r.get("cost_stages_n") and not r.get("cost_gpu"):
        return "—"
    return fmt_usd(r["cost_total"])


def cost_block(r):
    """Per-stage and per-model breakdown for an expanded row."""
    if not r or r.get("cost_total") is None:
        return ""
    try:
        stages = json.loads(r.get("cost_by_stage") or "{}")
        models = json.loads(r.get("cost_by_model") or "{}")
    except json.JSONDecodeError:
        return ""
    gpu = r.get("cost_gpu") or 0.0
    if not stages and not gpu:
        return ("<div class='eb eb-exp'><div class='eb-h'>What it cost</div>"
                "<p>No agent accounting survives for this experiment — its run "
                "directory was deleted. That is a missing receipt, not a zero."
                "</p></div>")
    rows = "".join(
        f"<tr><td>{esc(COST_STAGE_LABEL.get(k, k))}</td>"
        f"<td class='num'>{fmt_usd(v)}</td></tr>"
        for k, v in sorted(stages.items(), key=lambda kv: -kv[1]))
    if gpu:
        rows += (f"<tr><td>rented GPU — {fmt_dur(r.get('duration_s'))} at "
                 f"${POD_USD_PER_HR:.2f}/hr</td>"
                 f"<td class='num'>{fmt_usd(gpu)}</td></tr>")
    rows += (f"<tr class='cost-tot'><td>total</td>"
             f"<td class='num'>{fmt_usd(r['cost_total'])}</td></tr>")
    by_model = ""
    if models:
        by_model = ("<div class='cost-models'>by model — " + " · ".join(
            f"{esc(m)} <b>{fmt_usd(v)}</b>"
            for m, v in sorted(models.items(), key=lambda kv: -kv[1])) + "</div>")
    return (f"<div class='eb eb-exp'><div class='eb-h'>What it cost</div>"
            f"<table class='cost-t'>{rows}</table>{by_model}"
            f"<p class='cost-note'>The agent lines are the <b>equivalent API "
            f"cost</b> of the tokens each agent used, as recorded by the "
            f"harness when it ran. The research ran on a claude.ai Max "
            f"subscription, so that is what it would have cost billed per "
            f"token — not money that left the account. The GPU line is real "
            f"spend.</p></div>")


def annotate_pivot(exps):
    """Mark each development experiment with whether the harness had already
    spent its patience and injected the mandatory "must pivot" directive into
    its design prompt (loop.sh: a running streak of consecutive non-kept dev
    experiments since the last kept one; streak >= PATIENCE forces the next
    design to come from a design family absent from the recent history)."""
    streak = 0
    for e in exps:
        if e["kind"] == "holdout_check":
            continue
        e["is_pivot"] = streak >= PATIENCE
        streak = 0 if e["kept"] else streak + 1

CSS = """
:root{
  --paper:#fffff8; --ink:#111111; --muted:#6b6a60; --faint:#9b998c;
  --rule:#d9d5c3; --rule-soft:#ece9da; --accent:#8c2f1f;
  --kept:#111111; --disc:#b9b6a6; --ochre:#8a6a1e;
  --serif:"Palatino","Palatino Linotype","Book Antiqua","URW Palladio L",Georgia,serif;
  --mono:ui-monospace,"SF Mono",Menlo,monospace;
}
*{box-sizing:border-box}
html{background:var(--paper)}
body{margin:0;background:var(--paper);color:var(--ink);
  font:16px/1.55 var(--serif);font-feature-settings:"onum" 1,"liga" 1;
  -webkit-font-smoothing:antialiased;overflow-x:auto;padding-bottom:40px}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid transparent}
a:hover{border-bottom-color:var(--accent)}
.smcp{font-feature-settings:"smcp" 1;text-transform:uppercase;letter-spacing:.05em}
.num{font-variant-numeric:lining-nums tabular-nums}
.mono{font-family:var(--mono);font-size:12px}

#gh-ribbon{position:fixed;right:-56px;bottom:36px;z-index:80;
  transform:rotate(-45deg);background:#111111;color:#fffff8;
  font:600 11px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.12em;white-space:nowrap;
  padding:7px 64px;text-decoration:none;border:none;
  outline:1px solid rgba(255,255,248,.4);outline-offset:-4px;
  box-shadow:0 1px 8px rgba(0,0,0,.3)}
#gh-ribbon:hover{background:#8c2f1f;color:#fffff8}
@media (max-width:640px){
  #gh-ribbon{right:-62px;bottom:22px;font-size:8.5px;letter-spacing:.09em;
    padding:5px 52px;outline-offset:-3px}
}

.topnav{display:flex;gap:28px;justify-content:center;align-items:baseline;
  flex-wrap:wrap;row-gap:8px;position:relative;
  border-bottom:1px solid var(--rule);padding:18px 0 12px;margin:0}
.topnav .brand{font:italic 14px var(--serif);color:var(--faint)}
/* Halo: mask any line passing behind a label — applies to every inline
   svg (agent figures, chart) without touching the drawings themselves. */
svg text{paint-order:stroke;stroke:var(--paper);stroke-width:2.8px;
  stroke-linejoin:round}
header.page-head{max-width:900px;margin:34px auto 4px;padding:0 16px;
  text-align:center;display:block}
header.page-head .eyebrow{text-align:center;margin:0}
header.page-head h1{font:400 34px/1.15 var(--serif);color:var(--ink);
  margin:10px 0 14px;letter-spacing:0}
header.page-head .page-sub{font:15.5px/1.6 var(--serif);color:var(--muted);
  font-style:normal;margin:0 auto;max-width:820px;text-align:center}
.page-head .page-sub b{color:var(--ink)}
.page-head .page-sub a{color:var(--accent)}
.paths-wrap>.eyebrow{margin:34px 0 0}
h1.home-h1{margin:28px auto 32px}
.live-row{background:rgba(140,47,31,.045);
  box-shadow:inset 0 0 0 1.5px var(--accent);
  animation:live-row-pulse 1.8s ease-in-out infinite}
.live-row td{color:var(--muted);font-style:italic}
.live-row .status-badge{font-style:normal}
@keyframes live-row-pulse{
  0%,100%{box-shadow:inset 0 0 0 1.5px rgba(140,47,31,.3);background:rgba(140,47,31,.03)}
  50%{box-shadow:inset 0 0 0 1.5px rgba(140,47,31,.85);background:rgba(140,47,31,.08)}
}
.compute-banner{position:fixed;left:0;right:0;bottom:0;z-index:70;
  padding:7px 18px;text-align:center;
  font:12.5px var(--serif);color:var(--muted);
  background:var(--paper);
  border-top:1px solid rgba(140,47,31,.12);
  box-shadow:0 -1px 8px rgba(0,0,0,.06)}
.compute-banner .status-badge{margin-right:10px;vertical-align:baseline}
.compute-banner a{color:#8c2f1f;text-decoration:none;font-weight:600}
.compute-banner a:hover{text-decoration:underline}
.status-badge{display:inline-flex;align-items:center;gap:6px;
  font:600 11px var(--serif);font-feature-settings:"smcp" 1;
  letter-spacing:.08em}
.status-badge .dot{width:7px;height:7px;border-radius:50%}
.status-badge.live{color:#8c2f1f}
.status-badge.live .dot{background:#8c2f1f;animation:livepulse 1.8s ease-out infinite}
.status-badge.finished{color:var(--muted)}
.status-badge.finished .dot{background:var(--muted)}
@keyframes livepulse{0%{box-shadow:0 0 0 0 rgba(140,47,31,.45)}
  70%{box-shadow:0 0 0 7px rgba(140,47,31,0)}100%{box-shadow:0 0 0 0 rgba(140,47,31,0)}}
.topnav a{font:600 12px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.09em;color:var(--muted);
  border-bottom:2px solid transparent;padding-bottom:3px}
.topnav a:hover{color:var(--ink);border-bottom-color:transparent}
.topnav a.on{color:var(--ink);border-bottom-color:var(--ink)}
/* the flight-path page is the site's demo — it gets the one CTA in the nav.
   Only pinned to the right edge when the centered links leave room for it;
   below that it sits inline at the end of the (wrapping) centered row. */
.topnav a.cta{color:#fffff8;background:var(--accent);padding:5px 12px 4px;
  border-bottom:none;border-radius:2px}
.topnav a.cta:hover{background:var(--ink);color:#fffff8}
.topnav a.cta.on{background:var(--ink);color:#fffff8}
@media(min-width:1330px){
  .topnav a.cta{position:absolute;right:24px;top:-4px}
}

.paths-wrap{max-width:1080px;margin:0 auto;padding:14px 28px 96px}
.paths-wrap h1.home-h1{font-weight:400;font-size:44px;line-height:1.15;
  letter-spacing:-.01em;text-align:center;margin:26px 0 14px}
p.psub{text-align:center;font-style:italic;color:var(--muted);
  font-size:15.5px;line-height:1.7;margin:0 auto 8px;max-width:820px}
p.psub.lead{font-size:19px;max-width:900px;margin-bottom:14px}
.pnote{max-width:780px;margin:20px auto 0;font-size:14.5px;line-height:1.7;
  color:#4a473e}
.pnote p{margin:0 0 10px}
.pnote b{color:var(--ink)}
/* Moved off the overview hero, where it competed with the headline number.
   Set as an aside so it reads as context for the figures, not as a preamble
   the reader has to get through first. */
.scope-note{border-left:2px solid var(--rule);padding:2px 0 2px 18px;
  margin-top:34px;font-size:13.5px;color:var(--muted)}
.scope-note p:last-child{margin-bottom:0}
/* The five steps that took the score from 2.001 to 0.040 — numbered because
   each one is a single change and the order is the argument. */
/* Report register, after the author's prior research reports: the research
   question as a large serif subhead, a grey one-line descriptor, then the
   finding stated up front with the claim itself highlighted. */
/* The quoted line stands alone, the research question carries the weight. */
.home-h1{font-style:italic}
.home-q{max-width:860px;margin:22px auto 0;text-align:center;
  font:400 32px/1.28 var(--serif);color:var(--ink)}
.home-desc{max-width:640px;margin:12px auto 0;text-align:center;
  font:16px/1.5 var(--serif);color:var(--faint)}
.bl-h{max-width:780px;margin:38px auto 0;font:700 11px var(--serif);
  font-feature-settings:"smcp" 1;text-transform:uppercase;letter-spacing:.1em;
  color:var(--accent);text-align:center}
.bottom-line{max-width:780px;margin:8px auto 0;text-align:center;
  font:italic 19px/1.62 var(--serif);color:#4a473e}
.bottom-line .hl{background:#f4ecd2;box-decoration-break:clone;
  -webkit-box-decoration-break:clone;padding:1px 2px;color:var(--ink)}
.bottom-line i{font-style:normal;font-weight:600;color:var(--ink)}
/* Numbered section headings: light numeral, title at reading size, and a
   one-line italic summary so the page can be skimmed by section. */
.rsec{max-width:780px;margin:56px auto 0;display:flex;align-items:baseline;gap:14px}
.rsec-n{font:italic 26px var(--serif);color:var(--rule);flex:none}
.rsec-t{margin:0;font:400 26px/1.2 var(--serif);color:var(--ink)}
.rsec-sub{max-width:780px;margin:6px auto 18px;
  font:italic 15px/1.55 var(--serif);color:var(--muted)}
.answer-steps{margin:14px 0 14px;padding-left:26px}
.answer-steps li{margin:0 0 11px;line-height:1.65}
.answer-steps li::marker{color:var(--faint);font-weight:700}
/* the optimized number gets its own centred row; the three below are the
   human-readable breakdown of it */
.stat-hero{max-width:300px;margin:32px auto 6px;text-align:center}
.stat-hero > b{display:block;font-size:52px;line-height:1.05;font-weight:600;
  font-variant-numeric:lining-nums tabular-nums;color:var(--ink)}
.stat-hero{max-width:560px}
.hero-sub{max-width:520px;margin:16px auto 0;font-size:12.5px;line-height:1.65;
  color:var(--muted);text-align:center}
.hero-sub b{color:var(--ink)}
.hero-sub .pen{font-variant-numeric:lining-nums tabular-nums;color:var(--accent);
  font-weight:600}
.stat-hero > span{display:block;font:600 11px var(--serif);
  font-feature-settings:"smcp" 1;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);margin-top:6px}
.stats{display:flex;flex-wrap:nowrap;gap:18px 30px;justify-content:center;
  align-items:flex-start;margin:22px auto 10px;text-align:center}
.stat{flex:0 1 190px;max-width:210px}
/* only the tile's OWN direct <b> is the headline number — a <b> inside the
   label used to inherit 34px and render "USABLE FIX" as a second headline */
.stat > b{display:block;font-size:34px;line-height:1.1;font-weight:600;
  font-variant-numeric:lining-nums tabular-nums}
.stat > span{display:block;font:600 10.5px var(--serif);
  font-feature-settings:"smcp" 1;text-transform:uppercase;letter-spacing:.06em;
  color:var(--muted);line-height:1.45;margin-top:5px}
.sec-h{font:600 12px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
  text-align:center;margin:44px 0 10px}
.status-callout{max-width:780px;margin:22px auto 0;padding:16px 22px 14px;
  border:1px solid var(--rule);border-left:3px solid var(--accent);
  border-radius:0 2px 2px 0;background:#fbf8ea}
.status-callout-h{font:600 11px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.08em;color:var(--accent);margin:0 0 7px}
.status-callout p{margin:0;font-size:15.5px;line-height:1.65;color:#33312b}
.status-callout .status-meta{margin-top:7px;font-size:12.5px;color:var(--muted)}
/* ---- mobile (airloom-style device-width + breakpoints) ------------------
   The pages now declare width=device-width, so small screens get REAL
   layout instead of a zoomed-out 1100px desktop. Images are lazy-loaded
   everywhere; these rules reflow the handful of side-by-side compositions
   that would otherwise overflow a phone. */
@media(max-width:640px){
  .paths-wrap{padding:10px 14px 60px}
  .paths-wrap h1.home-h1{font-size:28px}
  .home-q{font-size:20px;line-height:1.35}
  .topnav{flex-wrap:wrap;gap:6px 14px;row-gap:4px}
  .stats{flex-wrap:wrap;gap:14px 22px}
  .stat > b{font-size:26px}
  .rsec{margin-top:40px}
  .rsec-t{font-size:21px}
  /* contract-row mobile rules live AFTER its base rules below — an earlier
     override here silently loses to the base line-height:260px. */
  .wex-imgs{flex-direction:column}
  .wex-arr{display:none}
  .tile-grid{grid-template-columns:1fr 1fr;gap:10px}
  .train-grid{grid-template-columns:repeat(3,1fr);gap:6px}
  .train-grid figcaption{font-size:7px}
  pre.howto{font-size:10.5px;padding:10px 12px}
  .cta{flex-direction:column;align-items:center;gap:4px}
  .approach{gap:18px}
  .strategies{gap:22px}
}

/* ---- the download CTA under the bottom line: quiet, one rule, no button - */
.cta{max-width:780px;margin:18px auto 0;padding:12px 18px;
  border:1px solid var(--rule);border-left:3px solid var(--accent);
  border-radius:0 2px 2px 0;display:flex;flex-wrap:wrap;
  align-items:baseline;justify-content:center;gap:6px 18px}
.cta-link{font:600 15px var(--serif);color:var(--accent);
  text-decoration:none;border-bottom:1px solid transparent;white-space:nowrap}
.cta-link:hover{border-bottom-color:var(--accent)}
.cta-meta{font-size:12.5px;color:var(--muted);
  font-variant-numeric:lining-nums tabular-nums}

/* ---- the tile grid: eight real 128 m tiles, each with its coordinate ---- */
.tile-grid{max-width:860px;margin:26px auto 4px;padding:0 16px;
  display:grid;grid-template-columns:repeat(4,1fr);gap:16px 16px}
.tile-grid figure{margin:0}
.tile-grid img{width:100%;display:block;border:1px solid var(--rule)}
.tile-grid figcaption{margin-top:5px;font-size:11px;line-height:1.5;
  color:var(--muted);font-variant-numeric:lining-nums tabular-nums}
.tg-idx{display:block;font-weight:600;color:var(--ink)}
.tg-cap{max-width:780px;margin:14px auto 0;padding:0 16px;
  font:italic 14px/1.6 var(--serif);color:var(--muted);text-align:center}

/* ---- the how-to-use code block ------------------------------------------ */
pre.howto{max-width:780px;margin:18px auto 0;padding:16px 20px;
  font:12.5px/1.6 var(--mono);color:var(--ink);background:#fff;
  border:1px solid var(--rule-soft);border-radius:2px;overflow-x:auto}
.howto-note{font-size:13px;color:var(--muted)}

/* ---- the training-data grid: 100 tiles of the raster ---------------------
   80% of the BROWSER, not of the 1080px page column: width:80vw with the
   calc() margin breaks it out of .paths-wrap so the data gets the room. */
.train-grid{width:80vw;margin:26px 0 4px calc(50% - 40vw);
  display:grid;grid-template-columns:repeat(10,1fr);gap:9px 9px}
.train-grid figure{margin:0}
.train-grid img{width:100%;display:block;border:1px solid var(--rule)}
.train-grid figcaption{margin-top:3px;font-size:8.5px;line-height:1.4;
  color:var(--faint);font-variant-numeric:lining-nums tabular-nums;
  text-align:center;letter-spacing:.01em}
@media (max-width:900px){.train-grid{grid-template-columns:repeat(5,1fr)}}

/* ---- the model-designs mosaic -------------------------------------------
   A Pinterest-style masonry: a 6-column grid with grid-auto-flow:dense, so
   differently-SIZED tiles flow around each other. Tile width encodes the
   verdict (reverted 1 col, kept 2, champion 3); vertical packing comes from
   per-cell grid-row spans that MASONRY_JS measures at load/resize — CSS
   alone cannot span rows for content of unknown height. */
.design-grid{width:80vw;margin:26px 0 4px calc(50% - 40vw);
  display:grid;grid-template-columns:repeat(6,1fr);
  grid-auto-rows:8px;grid-auto-flow:dense;column-gap:18px;row-gap:0}
.design-grid figure{margin:0;min-width:0;padding-bottom:18px}
.design-grid .sz1{grid-column:span 1}
.design-grid .sz2{grid-column:span 2}
.design-grid .sz3{grid-column:span 3}
.design-grid .sz1 figcaption{white-space:normal;font-size:10px}
/* MASONRY_JS re-lays the grid as a true skyline packing: the container goes
   block/relative and every cell absolute at a JS-computed spot. */
.design-grid.packed{display:block;position:relative}
.design-grid.packed .dg-cell{position:absolute;padding-bottom:0}
/* Cells sit on the paper itself — no white card fill; a hairline is enough
   to bound each figure, and the champion carries the only strong colour. */
.design-grid .dg-fig{display:block;border:1px solid var(--rule);
  border-radius:2px;background:transparent;padding:8px;cursor:zoom-in}
.design-grid .dg-fig:hover{border-color:var(--ink)}
.design-grid .champ .dg-fig{border:1.5px solid var(--accent);
  box-shadow:0 0 0 4px rgba(140,47,31,.07)}
.design-grid .champ figcaption{color:var(--accent)}
.design-grid .champ figcaption b{color:var(--accent)}
.design-grid svg{width:100%;height:auto;display:block}
.design-grid figcaption{margin-top:5px;font-size:11px;line-height:1.5;
  color:var(--muted);overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap}
.design-grid figcaption b{color:var(--ink);
  font-variant-numeric:lining-nums tabular-nums}
.design-grid .reverted figcaption{color:var(--faint)}
@media (max-width:1000px){.design-grid{grid-template-columns:repeat(3,1fr)}
  .design-grid .sz3{grid-column:span 3}
  .design-grid .sz2{grid-column:span 2}}

/* ---- the autoresearch loop: prose left, the cycle right ----------------- */
.approach{max-width:980px;margin:26px auto 0;padding:0 16px;
  display:flex;gap:46px;align-items:center}
.approach-text{flex:1 1 0;min-width:0}
.approach-text p{font-size:14.5px;line-height:1.7;color:#33312b;margin:0 0 10px}
.approach-cycle{flex:0 0 46%;max-width:470px}
.cycle{display:block;width:100%;height:auto;margin:6px auto 4px}
.carrow{fill:none;stroke:#8a857a;stroke-width:1.6}
.cnode{fill:var(--paper);stroke:var(--ink);stroke-width:1.3}
.cnum{fill:var(--ink);font:600 14px/1 var(--serif)}
.clabel{fill:var(--ink);font:600 15px/1 var(--serif);letter-spacing:.05em}
.csub{fill:var(--muted);font:italic 13px/1 var(--serif)}
.ccenter{fill:var(--accent);font:600 12px/1 var(--serif);letter-spacing:.1em;
  text-anchor:middle}
.ccenter-sub{fill:var(--faint);font:italic 13px/1 var(--serif);text-anchor:middle}
.cycle-cap{font:italic 13.5px/1.55 var(--serif);color:var(--muted);
  text-align:center;margin-top:6px}
.carrow-h{fill:none;stroke:var(--accent);stroke-width:1.8}
.chuman{fill:var(--accent)}
.chuman-lbl{fill:var(--accent);font:600 14px/1 var(--serif);letter-spacing:.05em}
.chuman-sub{fill:var(--muted);font:italic 12.5px/1 var(--serif)}
.carrow-e{fill:none;stroke:var(--ochre);stroke-width:1.8}
.cexpert{fill:var(--ochre)}
.cexpert-lbl{fill:var(--ochre);font:600 14px/1 var(--serif);letter-spacing:.05em}
.cexpert-sub{fill:var(--muted);font:italic 12.5px/1 var(--serif)}
@media (max-width:900px){.approach{flex-direction:column}}

/* ---- the loop's research strategies, as a 2x2 grid ---------------------- */
.strategies{max-width:860px;margin:26px auto 0;padding:0 16px;
  display:grid;grid-template-columns:1fr 1fr;gap:30px 48px;align-items:start}
.strategy-h{font:400 19px/1.3 var(--serif);color:var(--ink);margin:0 0 8px}
.strategy p{font-size:14px;line-height:1.6;color:var(--muted);margin:0}
.strategy p b{color:var(--ink)}
@media (max-width:700px){.strategies{grid-template-columns:1fr}}

.explore{max-width:780px;margin:0 auto}
.explore a.card{display:block;border:1px solid var(--rule);border-radius:2px;
  padding:14px 18px;margin:0 0 12px;color:inherit}
.explore a.card:hover{border-color:var(--ink)}
.explore .card b{font-size:16px}
.explore .card span{display:block;font-size:13.5px;color:var(--muted);
  margin-top:3px;line-height:1.55}

.contract-fig{max-width:980px;margin:34px auto 4px}
.contract-fig svg{width:100%;height:auto;display:block}
.contract-fig.static>svg{cursor:default}
.contract-cap{max-width:760px;margin:10px auto 0;font-size:13px;
  line-height:1.65;color:var(--muted);text-align:center;font-style:italic}
.pkey{display:flex;flex-wrap:wrap;gap:8px 26px;justify-content:center;
  font-size:13.5px;color:var(--muted);margin:22px auto 4px}
.pkey .sw{display:inline-block;width:16px;height:0;border-top:3px solid;
  vertical-align:middle;margin-right:7px}
.fig-entry{margin:54px 0 0;border-left:2px solid var(--rule-soft);
  padding:4px 0 8px 26px}
.fig-entry.kept{border-left-color:var(--ink)}
.fig-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px;
  margin:0 0 2px}
.fig-no{font:600 12px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.08em;color:var(--faint);
  white-space:nowrap}
.fig-title{font-size:19px;font-weight:600;line-height:1.35}
.fig-status{font-size:13px;color:var(--faint);font-style:italic;
  white-space:nowrap}
.fig-status b{color:var(--ink)}
.fig-status .fail{color:var(--accent);font-weight:600}
.chip-cur{display:inline-block;font:600 10.5px var(--serif);font-style:normal;
  font-feature-settings:"smcp" 1;text-transform:uppercase;letter-spacing:.07em;
  color:var(--paper);background:var(--ink);padding:2px 8px 3px;
  border-radius:2px;vertical-align:1px}
.fig-svg{margin:10px 0 4px;overflow-x:auto;cursor:zoom-in}
.fig-svg svg{width:100%;min-width:720px;height:auto;display:block}
.contract-fig>svg{cursor:zoom-in}
.arch-svg[data-ovfig]{cursor:zoom-in}
.fig-svg{cursor:zoom-in}

#svgov{position:fixed;inset:0;background:var(--paper);z-index:60;
  display:none;flex-direction:column}
#svgov.on{display:flex}
.ov-bar{display:flex;align-items:baseline;gap:16px;padding:12px 26px;
  border-bottom:1px solid var(--rule);flex:none}
.ov-no{font:600 12px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.08em;color:var(--faint);
  white-space:nowrap}
.ov-title{font-size:15px;font-weight:600;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.ov-hint{margin-left:auto;font:600 10.5px var(--serif);
  font-feature-settings:"smcp" 1;text-transform:uppercase;
  letter-spacing:.06em;color:var(--faint);white-space:nowrap}
.ov-close{cursor:pointer;font:600 12px var(--serif);
  font-feature-settings:"smcp" 1;text-transform:uppercase;
  letter-spacing:.06em;color:var(--muted);background:none;
  border:1px solid var(--rule);border-radius:2px;padding:4px 12px}
.ov-close:hover{color:var(--ink);border-color:var(--ink)}
.ov-canvas{flex:1;overflow:hidden;cursor:grab;touch-action:none;
  user-select:none;-webkit-user-select:none}
.ov-canvas.dragging{cursor:grabbing}
.ov-inner{width:100%;height:100%;transform-origin:0 0;position:relative}
.ov-inner svg{width:100%;height:100%;display:block;position:absolute;
  inset:0;transition:opacity 1.1s ease}
.ov-replay{position:fixed;left:0;right:0;bottom:0;z-index:65;display:none;
  align-items:center;gap:14px;padding:10px 18px;background:var(--paper);
  border-top:1px solid var(--rule)}
#svgov.on .ov-replay.has-chain{display:flex}
.ov-play{font:14px var(--serif);border:1px solid var(--rule);
  background:none;color:var(--ink);width:34px;height:30px;cursor:pointer;
  border-radius:3px}
.ov-play:hover{border-color:var(--ink)}
.ov-steps{display:flex;gap:8px;flex-wrap:wrap}
.ov-step{font:600 11px var(--serif);font-feature-settings:"smcp" 1;
  letter-spacing:.06em;color:var(--muted);border:1px solid var(--rule);
  border-radius:3px;padding:4px 9px;cursor:pointer;background:none}
.ov-step.on{color:var(--ink);border-color:var(--ink)}
.ov-step:hover{border-color:var(--ink)}
.fig-cap{max-width:880px;font-size:14.5px;line-height:1.65;color:#4a473e}
.fig-cap p{margin:0 0 4px}
.fig-cap .fig-lead{font-weight:600;color:var(--ink)}
.fig-meta{font-size:12.5px;color:var(--faint);margin-top:2px}
.fig-meta .chg{color:var(--accent)}
.fig-meta a{white-space:nowrap}

.dash-head{max-width:92vw;margin:0 auto;padding:18px 0 6px}
.eyebrow{font:600 12px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.09em;color:var(--muted);margin-bottom:10px}
.dash-head .sub{margin:0 0 6px;color:var(--muted);max-width:960px}
.dash-head .sub b{color:var(--ink)}
.intro{max-width:960px;font-size:14.5px;line-height:1.6;color:#4a473e;margin:10px 0 16px}
.intro p{margin:0 0 8px}
.intro b{color:var(--ink)}
.dash-meta{display:flex;flex-wrap:wrap;align-items:baseline;gap:14px 26px;
  font-size:13.5px;color:var(--muted)}
.k{display:inline-flex;align-items:center;gap:7px;cursor:default}
.k[title]{cursor:help}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.dot.kept{background:var(--ink)} .dot.disc{background:var(--disc)}
.ring{width:9px;height:9px;border-radius:50%;display:inline-block;
  border:1.5px solid var(--ochre);background:transparent}
.bar{width:16px;height:0;border-top:2px solid var(--ink);display:inline-block}
.bar.dash{border-top-style:dashed;border-top-color:var(--accent)}
.vrule{width:0;height:12px;border-left:1px dashed var(--faint);display:inline-block}
.x{color:var(--accent);font-weight:700}
.tri{width:0;height:0;border-left:4.5px solid transparent;
  border-right:4.5px solid transparent;border-top:7px solid var(--ochre);
  display:inline-block}
#updated{color:var(--faint);font-style:italic;margin-left:auto}

.dash-wrap{max-width:92vw;margin:0 auto;padding:4px 0 64px}
.chart-card{border-top:1px solid var(--rule);padding:14px 0 4px;margin-top:18px}
.chart-title{font:600 12px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:2px 0 6px}
.chart-card svg{width:100%;height:auto;display:block}
.axis-lab{font:11px var(--serif);fill:var(--faint)}
.tick-line{stroke:var(--rule-soft);stroke-width:1}
/* Era bands: washes behind the data, never competing with it. The caption
   sits above its band, so identity is never colour-alone. */
.era-lab{font:600 10.5px var(--serif);font-feature-settings:"smcp" 1;
  letter-spacing:.05em}
rect.era{cursor:default}
rect.era:hover{filter:brightness(.985)}
.era-sw{width:9px;height:11px;display:inline-block;vertical-align:-1px;
  margin-right:1px;background:rgba(107,106,96,.09);
  outline:1px solid var(--rule-soft)}
.chart-note{max-width:74ch;margin:0 auto;font:14px/1.55 var(--serif);
  color:var(--muted)}
.chart-note b{color:var(--ink);font-weight:600}
.chart-foot{border-top:1px solid var(--rule);margin-top:26px;padding-top:16px}
.cost-t{border-collapse:collapse;margin:2px 0 8px;font:13px var(--serif)}
.cost-t td{padding:3px 16px 3px 0;border-bottom:1px solid var(--rule-soft)}
.cost-t td.num{text-align:right;padding-left:26px;
  font-variant-numeric:lining-nums tabular-nums}
.cost-t tr.cost-tot td{border-bottom:none;border-top:1px solid var(--rule);
  font-weight:700;color:var(--ink)}
.cost-models{font:12px var(--serif);color:var(--muted);margin-bottom:6px}
.cost-note{font:italic 12px/1.45 var(--serif);color:var(--faint);margin:0}
/* Era headings on the model-designs page — a hard visual break, because the
   designs either side of one were judged by different instruments. */
.era-head{background:var(--era-tint);border-top:2px solid var(--rule);
  margin:56px 0 26px;padding:14px 18px 15px;display:flex;flex-wrap:wrap;
  align-items:baseline;gap:6px 14px}
.era-head:first-of-type{margin-top:18px}
.era-head-n{font:700 11px var(--serif);font-feature-settings:"smcp" 1;
  letter-spacing:.09em;text-transform:uppercase}
.era-head-t{font:600 19px var(--serif);color:var(--ink)}
.era-head-s{font:italic 13px var(--serif);color:var(--muted);flex-basis:100%}
/* Every table row is washed with its era's band tint, so a row and its dot
   in the chart above are identifiable as the same era without a lookup. */
tr.row-main{background:var(--era-tint,transparent);scroll-margin-top:8px}
tr.hist-row td{color:var(--muted)}
tr.hist-row .title-cell b{font-weight:600;color:var(--ink)}
.pt{cursor:pointer}
circle.pt.big{r:7px}
text.pt.big{font-size:17px}

#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--paper);
  font:12.5px/1.45 var(--serif);padding:7px 10px;max-width:340px;opacity:0;
  transition:opacity .08s;z-index:10}
#tip b{color:#fffff8}
#tip .t-note{color:#cbc8ba;display:block;margin-top:3px}

.tbl-card{border-top:1.5px solid var(--ink);margin-top:26px}
table.main{width:100%;border-collapse:collapse;font-size:14px}
table.main th{font:600 11.5px/1.2 var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
  border-bottom:1.5px solid var(--ink);white-space:nowrap;text-align:left;
  padding:9px 12px 9px 0}
table.main th[title]{cursor:help}
table.main th[title]::after{content:"\\00b0";color:var(--faint)}
table.main td{border-bottom:1px solid var(--rule-soft);padding:9px 12px 9px 0;
  vertical-align:baseline;text-align:left}
tr.row-main{cursor:pointer}
tr.row-main:hover,tr.row-main.hl{background:#f7f3e3}
tr.kept-row{box-shadow:inset 2px 0 0 var(--ink)}
tr.kept-row td:first-child{padding-left:12px}
tr.row-flash td{animation:rowflash 1.8s ease-out}
@keyframes rowflash{0%,30%{background:#f0e2c4}100%{background:transparent}}
td.title-cell b{font-weight:600}
.caret{color:var(--faint);display:inline-block;width:12px;transition:transform .15s}
tr.open .caret{transform:rotate(90deg)}
.st-kept{color:var(--ink);font-weight:600}
.st-disc{color:var(--faint)}
.st-fail{color:var(--accent);font-weight:600}
.st-rej{color:var(--accent);font-weight:600;font-style:italic}
.st-hold{color:var(--ochre);font-weight:600}
.cat{color:var(--muted);font:600 11.5px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.05em}
.pivot-tag{display:inline-block;font:600 10px var(--serif);letter-spacing:.03em;
  color:var(--ochre);border:1px solid var(--ochre);border-radius:3px;
  padding:0 4px;margin-left:6px;vertical-align:1px;cursor:help}
.delta-up{color:var(--faint);font-size:12px}
.delta-dn{color:var(--ink);font-size:12px;font-weight:600}

tr.detail td{background:#fcfbf2;padding:0;border-bottom:1px solid var(--rule)}
.detail-inner{padding:18px 18px 22px 38px}
.detail-grid{display:grid;grid-template-columns:minmax(380px,1.1fr) minmax(420px,1fr);
  gap:16px 34px;align-items:start}
@media(max-width:1100px){.detail-grid{grid-template-columns:1fr}}
.explain{display:flex;flex-direction:column;gap:12px}
.eb{border-left:2px solid var(--rule);padding:7px 14px 8px 16px;background:#fff;
  box-shadow:0 1px 7px rgba(60,50,30,.08);border-radius:0 3px 3px 0}
.eb-h{font:800 11px/1 var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-bottom:5px}
.eb p{margin:0;font-size:13.5px;line-height:1.55;color:#33312b}
.eb-hyp{border-left-color:var(--muted)}
.eb-met{border-left-color:var(--ochre)} .eb-met .eb-h{color:var(--ochre)}
.eb-exp{border-left-color:var(--rule)}
.eb-res{border-left-color:var(--ink)} .eb-res .eb-h{color:var(--ink)}
.eb-con{border-left-color:var(--accent)} .eb-con .eb-h{color:var(--accent)}
.eb-eli{border-left-color:var(--ink);background:#fbf8ea}
/* "Why this status" — the headline plain-language reason, colour-keyed to
   the status chip (see status_reason()). Sits first in the detail column. */
.eb-why{border-left-width:3px}
.why-kept{border-left-color:var(--ink)} .why-kept .eb-h{color:var(--ink)}
.why-rej{border-left-color:var(--accent);background:#fdf6f4} .why-rej .eb-h{color:var(--accent)}
.why-fail{border-left-color:var(--accent);background:#fdf6f4} .why-fail .eb-h{color:var(--accent)}
.why-disc{border-left-color:var(--muted)} .why-disc .eb-h{color:var(--muted)}
.why-hold{border-left-color:var(--ochre);background:#fbf7ec} .why-hold .eb-h{color:var(--ochre)}
/* One-line reason under each log-row title, always visible (airloom-style). */
.row-why{font-size:11.5px;line-height:1.35;color:var(--faint);font-style:italic;margin-top:2px}
.row-why.why-rej,.row-why.why-fail{color:var(--accent);font-style:normal}
.row-why.why-kept{color:var(--muted);font-style:normal}
.row-why.why-hold{color:var(--ochre)}
.eb-eli p{font-size:14.5px}

/* The figure leads an expanded row; the two-column grid below needs real
   air under it rather than butting straight up against the drawing. */
.arch{margin:6px 0 34px;padding-bottom:26px;border-bottom:1px solid var(--rule-soft)}
.arch-h{font:600 12px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 10px}
.arch-h .chg{color:var(--accent)}
.wex-row{display:flex;flex-direction:column;gap:12px}
/* the two images must always hold ONE row inside the scoreboard column
   (minmax(420px,1fr)) — fluid flex:1 1 0 + min-width:0 makes them shrink to
   the available width instead of the old fixed 230px/340px, which wrapped */
.wex-imgs{display:flex;flex-wrap:nowrap;gap:10px;align-items:flex-start}
.wex-row figure{margin:0;flex:1 1 0;min-width:0}
.wex-row a{border-bottom:none}
.wex-frame img{width:100%;aspect-ratio:1;height:auto;display:block;
  border:1px solid var(--rule)}
.wex-map img{width:100%;height:auto;display:block;border:1px solid var(--rule)}
.wex-row figcaption{font-size:11.5px;color:var(--muted);line-height:1.5;
  margin-top:6px;max-width:100%}
.wex-row figcaption b{color:var(--ink);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.05em}
.wex-arr{color:var(--accent);font-size:30px;line-height:1;flex:none;
  align-self:center;padding-bottom:34px;opacity:.85}
.wex-stats{display:flex;flex-direction:row;gap:28px;padding:2px 0 0}
/* a bare "0.183" is meaningless without its scale — show where it sits */
.stat-hero .sc-wrap{display:block;margin-top:12px}
.sc-bar{position:relative;display:block;height:4px;border-radius:2px;
  background:linear-gradient(90deg,#8c2f1f 0%,#c9a227 50%,#3c9c3c 100%);
  opacity:.5}
.sc-dot{position:absolute;top:-3px;width:10px;height:10px;border-radius:50%;
  background:var(--ink);border:2px solid var(--paper);transform:translateX(-50%)}
.sc-ends{display:flex;justify-content:space-between;margin-top:5px;
  font-size:9.5px;letter-spacing:.04em;color:var(--faint);
  text-transform:none;font-feature-settings:normal}
/* the one-frame-in / one-position-out contract: a SMALL fixed thumbnail beside
   a compact readout — deliberately not the .wex-* map layout, whose fluid
   flex:1 1 0 blew the 256px frame up to ~430px and stretched the row */
/* Same composition as the two maps below: a pair of EQUAL boxes, centred,
   with a caption under each. Earlier versions put a small square beside a
   wider panel of a different height, left-aligned against the container edge,
   which read as two unrelated elements rather than one before/after pair. */
.contract-row{display:flex;justify-content:center;align-items:flex-start;
  gap:26px;margin:20px auto 8px;flex-wrap:wrap}
.contract-col{flex:none;width:260px;margin:0}
.contract-col img{width:260px;height:260px;display:block;
  border:1px solid var(--rule)}
.contract-col figcaption{font-size:11.5px;color:var(--muted);line-height:1.55;
  margin-top:9px;width:260px}
.contract-col figcaption b{color:var(--ink);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.05em;font-size:11px}
.contract-arrow{flex:none;color:var(--accent);font-size:30px;line-height:260px;
  opacity:.85}
/* the readout is a box of the SAME 260px square as the frame beside it */
.contract-out{width:260px;height:260px;box-sizing:border-box;
  border:1px solid var(--rule);background:var(--paper);padding:20px 20px;
  display:flex;flex-direction:column;justify-content:center;gap:14px}
.co-row{display:flex;justify-content:space-between;align-items:baseline;gap:16px;
  border-bottom:1px dotted var(--rule);padding-bottom:9px}
.co-row:last-child{border-bottom:none;padding-bottom:0}
.co-k{font:600 10px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.co-v{font-size:18px;color:var(--ink)}
.contract-note{max-width:780px;margin:14px auto 30px;font-size:13px;
  color:var(--muted);line-height:1.65;text-align:center}
.contract-note b{color:var(--ink)}
/* Stack the contract vertically once the three 260px cells no longer fit.
   Must come AFTER the base rules: same specificity, so source order decides,
   and the base .contract-arrow line-height:260px would otherwise win on
   mobile and open a huge blank gap around the arrow. */
@media(max-width:880px){
  .contract-row{flex-direction:column;align-items:center;gap:12px}
  .contract-arrow{line-height:1.2;transform:rotate(90deg)}
}
.wex-num{font-size:27px;line-height:1.1;color:var(--ink)}
.wex-lab{font:600 10.5px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-top:2px}
.wex-pipe{font-size:12.5px;color:var(--muted);line-height:1.7;margin-top:12px;
  max-width:1240px}
.wex-pipe b{color:var(--ink);font-weight:600;font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.05em;font-size:11.5px}
.wex-pipe b.chg{color:var(--accent)}
.wex-pipe .pd{color:#7a4438}
.wex-pipe .sep{color:var(--faint)}
.arch-svg{max-width:1120px;margin:2px 0 20px;overflow-x:auto}
.arch-svg svg{width:100%;min-width:760px;height:auto;display:block}

.score-head{font:600 12px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 4px}
.score-sub{font-size:12.5px;color:var(--faint);font-style:italic;margin:0 0 8px}
table.cells{border-collapse:collapse;font-size:12.5px;width:100%}
table.cells th{font:600 10.5px/1.2 var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
  border-bottom:1px solid var(--ink);padding:5px 12px 5px 0;text-align:left}
table.cells td{border-bottom:1px solid var(--rule-soft);padding:4px 12px 4px 0}
.cov{color:var(--faint);font-size:11px}
.cell-bad{color:var(--accent);font-weight:600}
.cell-good{color:var(--ink);font-weight:600}
.cell-worst{box-shadow:inset 0 -2px 0 var(--accent)}
.gates{font-size:13px;color:#4a473e;margin-top:12px;border-left:2px solid var(--rule);
  padding:4px 0 4px 14px}
.gates b{color:var(--ink)}
.gates .ok{color:var(--ink);font-weight:600}
.gates .bad{color:var(--accent);font-weight:600}
.provenance{font-size:11.5px;color:var(--faint);margin-top:14px;
  font-feature-settings:"smcp" 1;text-transform:uppercase;letter-spacing:.04em}
pre.prompt{font:12px/1.5 var(--mono);white-space:pre-wrap;background:#fff;
  border-left:2px solid var(--rule);padding:12px 16px;margin:8px 0 0;
  max-width:920px;max-height:420px;overflow:auto;color:#33312b;
  box-shadow:0 1px 7px rgba(60,50,30,.08)}
.figs-intro{font-size:13px;color:#4a473e;max-width:920px;margin:2px 0 8px}
.figs-intro i{color:var(--faint)}
#lightbox{position:fixed;inset:0;background:rgba(17,17,17,.92);z-index:50;
  display:none;align-items:center;justify-content:center;cursor:zoom-out;
  flex-direction:column;gap:10px}
#lightbox.on{display:flex}
#lightbox img{max-width:96vw;max-height:90vh;border:1px solid #444}
#lightbox .lb-cap{color:#cbc8ba;font:13.5px var(--serif);font-style:italic}
#lightbox .lb-hint{color:#77746a;font:11px var(--serif);
  font-feature-settings:"smcp" 1;text-transform:uppercase;letter-spacing:.06em}

.figs{margin-top:16px}
.figs-h{font:600 12px var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:14px 0 6px}
.thumbs{display:flex;flex-wrap:wrap;gap:12px}
.thumbs figure{margin:0}
.thumbs img{height:130px;display:block;border:1px solid var(--rule)}
.thumbs.maps{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;gap:14px}
.thumbs.maps figure{min-width:0}
.thumbs.maps a{display:block;aspect-ratio:1/1;overflow:hidden;
  border:1px solid var(--rule)}
.thumbs.maps img{width:100%;height:100%;object-fit:cover;border:none}
.thumbs.maps figcaption{margin-top:5px;font-size:11.5px;line-height:1.45}
.thumbs.maps figcaption b{color:var(--ink);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.05em;display:block}
.thumbs figcaption{font:11px var(--serif);color:var(--faint);margin-top:3px}
details.trywrap{margin:12px 0 0}
details.trywrap>summary{cursor:pointer;color:var(--muted);font-weight:600;
  font-size:12px;font-feature-settings:"smcp" 1;text-transform:uppercase;
  letter-spacing:.04em;list-style:none}
details.trywrap>summary::-webkit-details-marker{display:none}
details.trywrap>summary::before{content:"\\25b8  ";color:var(--faint)}
details.trywrap[open]>summary::before{content:"\\25be  "}

/* Inline agent-editable source. The search space is two files, so the code
   is primary evidence, not an appendix -- but it is long, so it stays folded
   until asked for. Muted syntax tints only: this is a printed-page palette,
   not an IDE theme, and the code should read as part of the document. */
.src-prov{font-size:12.5px;color:var(--muted);margin:10px 0 12px;max-width:70ch;
  line-height:1.5}
.src-prov code{font:11.5px var(--mono);color:var(--ink)}
a.src-link{color:var(--accent);border-bottom:1px solid var(--rule);
  text-decoration:none;white-space:nowrap}
a.src-link:hover{border-bottom-color:var(--accent)}
.src-file{margin:0 0 14px}
.src-h{font:600 11px var(--mono);color:var(--muted);text-transform:uppercase;
  letter-spacing:.06em;padding:0 0 5px;border-bottom:1px solid var(--rule);
  display:flex;justify-content:space-between;align-items:baseline;gap:12px}
.src-lines{font-weight:400;color:var(--faint);text-transform:none;
  letter-spacing:0;font-variant-numeric:tabular-nums}
pre.src{font:11.5px/1.62 var(--mono);background:#fff;border:1px solid var(--rule-soft);
  border-top:none;margin:0;padding:10px 12px 10px 0;overflow-x:auto;
  max-height:30em;overflow-y:auto;counter-reset:l;white-space:pre;tab-size:4}
pre.src .ln{counter-increment:l;display:inline-block;width:3.2em;
  padding-right:1em;text-align:right;color:#c9c6b6;
  font-variant-numeric:tabular-nums;user-select:none;-webkit-user-select:none}
pre.src .ln::before{content:counter(l)}
pre.src .k{color:#8c2f1f}                    /* keyword */
pre.src .s{color:#3f6b46}                    /* string  */
pre.src .c{color:#9b998c;font-style:italic}  /* comment */
pre.src .n{color:#2f5c8c}                    /* number  */
pre.src .b{color:#6b4a8c}                    /* builtin/self */
pre.src .o{color:#8a877a}                    /* operator */
@media(max-width:700px){pre.src{font-size:10.5px}
  pre.src .ln{width:2.4em;padding-right:.6em}}

.help{margin-top:16px;font-size:13.5px;color:#4a473e;max-width:960px}
.help>summary{cursor:pointer;color:var(--muted);font-weight:600;list-style:none;
  font-feature-settings:"smcp" 1;text-transform:uppercase;letter-spacing:.04em;
  font-size:12.5px}
.help>summary::-webkit-details-marker{display:none}
.help>summary::before{content:"\\25b8  ";color:var(--faint)}
.help[open]>summary::before{content:"\\25be  "}
.help-grid{margin:12px 0 2px;display:grid;grid-template-columns:max-content 1fr;
  gap:8px 18px;align-items:baseline}
.help-grid dt{font-weight:700;color:var(--ink);white-space:nowrap}
.help-grid dd{margin:0}
.help .foot{margin-top:10px;color:var(--muted);font-style:italic}

.nb-wrap{max-width:980px;margin:0 auto;padding:16px 28px 110px}
.nb-day{padding:38px 0 6px;border-top:1px solid var(--rule)}
.nb-day:first-child{border-top:none;padding-top:14px}
.nb-day h2{font:400 24px var(--serif);color:var(--ink);margin:0 0 2px}
.nb-day .daysub{font:italic 14.5px var(--serif);color:var(--muted);margin:0 0 26px}
.nb-row{display:grid;grid-template-columns:150px 1fr;column-gap:26px;
  margin:0 0 24px;align-items:baseline}
.nb-row .nb-time{min-width:0;font:600 11.5px var(--mono);color:var(--faint);
  letter-spacing:.02em;text-align:right;white-space:normal}
.nb-row .nb-text{min-width:0;font-size:16px;line-height:1.7;color:var(--ink)}
.nb-row .nb-text code{font:12.5px var(--mono);background:var(--rule-soft);
  padding:1px 4px;border-radius:2px}
.nb-pull{display:grid;grid-template-columns:150px 1fr;column-gap:26px;
  margin:8px 0 30px}
.nb-pull .nb-time{min-width:0;font:600 11px var(--mono);color:var(--ink);
  text-align:right;padding-top:3px}
.nb-pull .nb-quote{min-width:0;border-left:2px solid var(--ink);padding-left:22px}
.nb-pull .nb-quote p{font:italic 20px/1.5 var(--serif);color:var(--ink);margin:0}
.nb-text b,.nb-quote b{font-weight:600}
.nb-inline-spark{display:inline-block;vertical-align:-3px;margin:0 2px 0 6px}
.nb-source{font:13px var(--serif);color:var(--faint);font-style:italic;
  text-align:center;margin:40px 28px 0;padding-top:18px;border-top:1px solid var(--rule)}
@media (max-width:640px){
  .nb-row,.nb-pull{grid-template-columns:1fr}
  .nb-row .nb-time,.nb-pull .nb-time{text-align:left;padding-top:0}
}
"""

JS = """
var tip;
function toggle(id){
  var d=document.getElementById('d'+id), r=document.getElementById('r'+id);
  var open=d.style.display!=='none';
  d.style.display=open?'none':'table-row';
  r.classList.toggle('open',!open);
  try{
    var s=JSON.parse(sessionStorage.getItem('open')||'[]');
    if(open){s=s.filter(function(x){return x!==id})}else{s.push(id)}
    sessionStorage.setItem('open',JSON.stringify(s));
  }catch(e){}
}
function showTip(ev,el){
  tip.innerHTML=el.dataset.tip;
  tip.style.opacity=1;
  var x=Math.min(ev.clientX+14,window.innerWidth-tip.offsetWidth-8);
  tip.style.left=x+'px';
  tip.style.top=Math.min(ev.clientY+14,window.innerHeight-tip.offsetHeight-8)+'px';
}
window.addEventListener('load',function(){
  tip=document.getElementById('tip');
  // deep link from other pages: index.html#r<id> opens + flashes that row
  var hm=location.hash.match(/^#r([A-Za-z0-9_-]+)$/);
  if(hm){
    var hr=document.getElementById('r'+hm[1]),
        hd=document.getElementById('d'+hm[1]);
    if(hr){
      if(hd&&hd.style.display==='none')toggle(hm[1]);
      hr.scrollIntoView({behavior:'smooth',block:'start'});
      hr.classList.add('row-flash');
    }
  }
  try{JSON.parse(sessionStorage.getItem('open')||'[]').forEach(function(id){
    var d=document.getElementById('d'+id);
    if(d){d.style.display='table-row';
      document.getElementById('r'+id).classList.add('open')}
  })}catch(e){}
  // chart <-> table interlink
  document.querySelectorAll('.pt').forEach(function(el){
    var id=el.dataset.id, row=document.getElementById('r'+id);
    el.addEventListener('mousemove',function(ev){showTip(ev,el)});
    el.addEventListener('mouseenter',function(){if(row)row.classList.add('hl')});
    el.addEventListener('mouseleave',function(){tip.style.opacity=0;
      if(row)row.classList.remove('hl')});
    el.addEventListener('click',function(){
      if(!row)return;
      var d=document.getElementById('d'+id);
      if(d&&d.style.display==='none')toggle(id);
      row.scrollIntoView({behavior:'smooth',block:'start'});
      row.classList.remove('row-flash');void row.offsetWidth;
      row.classList.add('row-flash');
    });
  });
  document.querySelectorAll('tr.row-main').forEach(function(row){
    var id=row.id.slice(1);
    var mk=document.querySelector('.pt[data-id="'+id+'"]');
    if(!mk)return;
    row.addEventListener('mouseenter',function(){mk.classList.add('big')});
    row.addEventListener('mouseleave',function(){mk.classList.remove('big')});
  });
  // image lightbox: any figure image opens a full-screen modal
  var lb=document.getElementById('lightbox'),
      lbImg=lb.querySelector('img'), lbCap=lb.querySelector('.lb-cap');
  document.body.addEventListener('click',function(ev){
    var a=ev.target.closest('.thumbs a,.wex-row a');
    if(!a)return;
    ev.preventDefault();ev.stopPropagation();
    lbImg.src=a.getAttribute('href');
    var cap=a.parentElement.querySelector('figcaption');
    lbCap.textContent=cap?cap.textContent:'';
    lb.classList.add('on');
  });
  lb.addEventListener('click',function(){lb.classList.remove('on');lbImg.src=''});
  window.addEventListener('keydown',function(ev){
    if(ev.key==='Escape'&&lb.classList.contains('on')){
      lb.classList.remove('on');lbImg.src=''}
  });
});
"""


# Zoom/pan overlay for the inference-paths page: click any figure to open it
# full-screen (SVG, so zoom is lossless); wheel = zoom about the cursor,
# drag = pan, double-click = reset, Esc or the close button to exit.
OVERLAY_HTML = """<div id="svgov"><div class="ov-bar"><span class="ov-no"></span>
<span class="ov-title"></span>
<span class="ov-hint">scroll to zoom · drag to pan · double-click to reset</span>
<button class="ov-close">Esc · close</button></div>
<div class="ov-canvas"><div class="ov-inner"></div></div>
<div class="ov-replay" id="ov-replay"><button class="ov-play" title="replay the lineage: oldest ancestor first, fading forward to this design">▶</button><div class="ov-steps"></div></div></div>"""

PATHS_JS = """
var ov,ovIn,ovCv,sc=1,px=0,py=0,drag=null;
var figReg={},chain=[],step=-1,playing=null;
function ovApply(){ovIn.style.transform=
  'translate('+px+'px,'+py+'px) scale('+sc+')'}
function stopPlay(){if(playing){clearTimeout(playing);playing=null;
  var b=ov.querySelector('.ov-play');if(b)b.textContent='\u25B6';}}
function anchorTransform(f){
  // map this figure's frozen-input anchor onto the target figure's anchor,
  // so the frozen endpoints stay planted while the middle crossfades.
  var t=figReg[chain[chain.length-1]];
  if(!t||f.inY==null||t.inY==null)return '';
  var r=ovCv.getBoundingClientRect(),W=r.width,H=r.height;
  function geo(g){var k=Math.min(W/980,H/g.vbH);
    return {k:k,ox:(W-k*980)/2,oy:(H-k*g.vbH)/2};}
  var gf=geo(f),gt=geo(t),sc2=gt.k/gf.k;
  var dx=gt.ox-sc2*gf.ox;
  var dy=gt.oy+gt.k*(t.inY+38)-sc2*(gf.oy+gf.k*(f.inY+38));
  return 'translate('+dx.toFixed(1)+'px,'+dy.toFixed(1)+'px) scale('+sc2.toFixed(4)+')';
}
function showStep(i,fade){
  if(i<0||i>=chain.length)return;
  step=i;var f=figReg[chain[i]];if(!f)return;
  var nw=f.svg.cloneNode(true);
  nw.style.transformOrigin='0 0';
  var tf=anchorTransform(f);if(tf)nw.style.transform=tf;
  if(fade){nw.style.opacity='0';}
  var olds=[].slice.call(ovIn.children);
  ovIn.appendChild(nw);
  if(fade){void nw.getBoundingClientRect();nw.style.opacity='1';
    olds.forEach(function(o){o.style.opacity='0'});
    setTimeout(function(){olds.forEach(function(o){o.remove()})},1150);}
  else{olds.forEach(function(o){o.remove()});}
  document.querySelector('.ov-no').textContent=
    'step '+(i+1)+' of '+chain.length+' \u00B7 '+f.no;
  document.querySelector('.ov-title').textContent=f.title;
  ov.querySelectorAll('.ov-step').forEach(function(c,j){
    c.classList.toggle('on',j===i)});
}
function playFrom(i){
  showStep(i,true);
  if(i<chain.length-1){playing=setTimeout(function(){playFrom(i+1)},2400);}
  else{stopPlay();}
}
function buildReplay(){
  var bar=document.getElementById('ov-replay'),
      steps=bar.querySelector('.ov-steps');
  steps.innerHTML='';
  bar.classList.toggle('has-chain',chain.length>1);
  chain.forEach(function(id,i){
    var b=document.createElement('button');
    b.className='ov-step';b.textContent='#'+id;
    b.addEventListener('click',function(){stopPlay();showStep(i,true)});
    steps.appendChild(b);
  });
}
function ovOpen(svg,no,title,chainIds){
  stopPlay();
  chain=chainIds||[];buildReplay();
  ovIn.innerHTML='';ovIn.appendChild(svg.cloneNode(true));
  document.querySelector('.ov-no').textContent=no;
  document.querySelector('.ov-title').textContent=title;
  step=chain.length-1;
  ov.querySelectorAll('.ov-step').forEach(function(c,j){
    c.classList.toggle('on',j===step)});
  sc=1;px=0;py=0;ovApply();
  ov.classList.add('on');document.body.style.overflow='hidden';
}
function ovClose(){stopPlay();ov.classList.remove('on');
  document.body.style.overflow='';ovIn.innerHTML=''}
window.addEventListener('load',function(){
  ov=document.getElementById('svgov');
  ovIn=ov.querySelector('.ov-inner');
  ovCv=ov.querySelector('.ov-canvas');
  document.querySelectorAll('[data-ovfig]').forEach(function(holder){
    var svg=holder.querySelector('svg');
    if(!svg)return;
    var vb=(svg.getAttribute('viewBox')||'0 0 980 300').split(/\\s+/),
        fin=svg.querySelector("[id='frozen-input']"),
        finY=fin?parseFloat(fin.getAttribute('y')||'74'):null;
    if(holder.dataset.id)
      figReg[holder.dataset.id]={svg:svg,vbH:parseFloat(vb[3])||300,inY:finY,
        no:holder.dataset.no||'',title:holder.dataset.title||''};
    holder.addEventListener('click',function(){
      ovOpen(svg,holder.dataset.no||'',holder.dataset.title||'',
             (holder.dataset.chain||'').split(',').filter(function(x){
               return figReg[x]}))});
  });
  var pb=ov.querySelector('.ov-play');
  if(pb)pb.addEventListener('click',function(){
    if(playing){stopPlay();}
    else{pb.textContent='\u25A0';playFrom(0);}
  });

  ov.querySelector('.ov-close').addEventListener('click',ovClose);
  window.addEventListener('keydown',function(e){
    if(e.key==='Escape'&&ov.classList.contains('on'))ovClose()});
  ovCv.addEventListener('wheel',function(e){
    e.preventDefault();
    var r=ovCv.getBoundingClientRect(),
        mx=e.clientX-r.left,my=e.clientY-r.top,
        ns=Math.min(16,Math.max(0.5,sc*Math.exp(-e.deltaY*0.002))),
        k=ns/sc;
    px=mx-k*(mx-px);py=my-k*(my-py);sc=ns;ovApply();
  },{passive:false});
  ovCv.addEventListener('pointerdown',function(e){
    drag={x:e.clientX,y:e.clientY,px:px,py:py};
    ovCv.setPointerCapture(e.pointerId);
    ovCv.classList.add('dragging');
  });
  ovCv.addEventListener('pointermove',function(e){
    if(!drag)return;
    px=drag.px+e.clientX-drag.x;py=drag.py+e.clientY-drag.y;ovApply();
  });
  ovCv.addEventListener('pointerup',function(){
    drag=null;ovCv.classList.remove('dragging')});
  ovCv.addEventListener('dblclick',function(){sc=1;px=0;py=0;ovApply()});
});
"""


def esc(s):
    return html.escape(str(s if s is not None else ""))


# Shared top navigation, airloom pattern (centered, italic brand, smallcaps
# links, active page underlined in ink). The overview lives at the repo root
# (index.html) and the other pages under gallery/, so hrefs are resolved per
# page location. Lineage + results pages are planned but held out for now —
# add them here when they exist so every page's nav updates together.
NAV_PAGES = (("overview", "overview"),
             ("log", "research log"),
             ("paths", "model designs"),
             ("lineage", "experiment lineage"),
             ("evolution", "research evolution"),
             ("notebook", "lab notebook"),
             ("flight", "flight path"))


def research_status():
    """LIVE until a human calls the research finished (convergence, success,
    or budget) by writing 'finished[: reason]' to state/research_status and
    pushing. Missing file or any other content means the loop is still the
    story: LIVE."""
    p = REPO_ROOT / "state" / "research_status"
    try:
        txt = p.read_text().strip()
    except OSError:
        txt = ""
    if txt.lower().startswith("finished"):
        reason = txt.split(":", 1)[1].strip() if ":" in txt else ""
        return "finished", reason
    return "live", ""


def status_badge():
    state, reason = research_status()
    if state == "finished":
        title = f" title='{esc(reason)}'" if reason else ""
        return (f"<span class='status-badge finished'{title}>"
                f"<span class='dot'></span>finished</span>")
    return ("<span class='status-badge live' title='experiments are running "
            "and this page updates as each one lands'>"
            "<span class='dot'></span>live</span>")


# Compute credit banner — full-width strip under the topnav on every page,
# carrying the LIVE/FINISHED badge.
def compute_banner():
    """The pinned "live" strip along the bottom of every page.

    It earns a fixed overlay only while the loop is actually running, because
    then it is news: results land on the page without a reload. Once the
    research is called finished it is no longer news, and a permanent bar
    across the foot of the viewport just occludes whatever is under it. So
    when finished it renders nothing here — the same fact is stated once, in
    the page footer, in normal flow (see CREDITS/finished_note)."""
    state, _ = research_status()
    if state == "finished":
        return ""
    return (f"<div class='compute-banner'>{status_badge()}"
            f"experiments are running on a single local machine — every result "
            f"lands on this page automatically as the loop commits it</div>")


def finished_note():
    """The concluded-research statement, in normal flow in the footer."""
    state, reason = research_status()
    if state != "finished":
        return ""
    why = f" {esc(reason)}" if reason else ""
    return (f"<p style='margin:0 0 8px'>Experiments ran on a single local "
            f"machine. The research is concluded and the record on these pages "
            f"is complete.{why}</p>")


def live_row(next_id):
    """Pulsing in-progress row atop the research-log table. The page is
    rebuilt at each experiment's end — i.e. moments before the NEXT
    experiment starts — so 'elapsed since build' ≈ elapsed of the run in
    flight, and the current phase is estimated from the median phase
    durations of recent runs (runs/*/timings.json). Client JS ticks the
    clock; wording stays explicit that the phase is an estimate."""
    state, _ = research_status()
    if state == "finished":
        return ""
    med = {}
    files = sorted(REPO_ROOT.glob("runs/*/timings.json"))[-5:]
    if files:
        acc = {}
        for f in files:
            try:
                for k, v in json.loads(f.read_text()).items():
                    acc.setdefault(k, []).append(float(v))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
        med = {k: sorted(v)[len(v) // 2] for k, v in acc.items()}
    phases = [
        ("starting up", 90),
        ("designing the experiment", med.get("agent_design_s", 900)),
        ("implementing the design", med.get("agent_impl_s", 120)),
        ("training Berlin", med.get("train_wall_s", 600)),
        ("scoring against the frozen ruler", med.get("score_s", 240)),
        ("logging + publishing", med.get("samples_s", 60) + med.get("gallery_s", 60)),
    ]
    # Pre-fill from the LOCAL state/phase.json at render time. The JS below
    # refreshes this from the status branch, but that fetch hits
    # raw.githubusercontent.com, which returns 403/404 for a PRIVATE repo — so
    # for local viewing (and any private deployment) it never resolves and the
    # row was stuck on "Design not finished yet" even when the design existed.
    pre = {}
    try:
        pj = json.loads((REPO_ROOT / "state" / "phase.json").read_text())
        if int(pj.get("iter") or -1) == int(next_id):
            pre = pj.get("design") or {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pre = {}
    pre_title = esc(pre.get("title") or f"experiment #{next_id} in progress…")
    pre_cat = esc(pre.get("category") or "—")
    pre_blocks = ""
    for key, cls, label in (("eli5", "eb-eli", "In plain words"),
                            ("hypothesis", "eb-hyp", "Hypothesis"),
                            ("method", "eb-met", "Method"),
                            ("expected_outcome", "eb-exp", "Expected outcome")):
        if pre.get(key):
            pre_blocks += (f"<div class='eb {cls}'><div class='eb-h'>{label}</div>"
                           f"<p>{esc(pre[key])}</p></div>")
    if not pre_blocks:
        pre_blocks = ("<div class='eb eb-why'><div class='eb-h'>Status</div>"
                      "<p>Design not finished yet — check back shortly.</p></div>")
    phases_js = json.dumps([[n, round(s)] for n, s in phases])
    built_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    return f"""<tr class="row-main live-row" id="r{next_id}" onclick="toggle({next_id})">
<td><span class="caret">▸</span></td><td class="num">{next_id}</td>
<td class="title-cell"><b id="live-title">{pre_title}</b></td>
<td><span class="cat" id="live-cat">{pre_cat}</span></td>
<td class="num">…</td>
<td class="num" id="live-time">…</td>
<td class="num">—</td>
<td><span class="status-badge live"><span class="dot"></span>live</span></td></tr>
<tr class="detail" id="d{next_id}" style="display:none"><td colspan="8">
<div class="detail-inner"><div class="detail-grid">
<div class="explain" id="live-explain">{pre_blocks}</div>
<div><div class="score-head">Scoreboard — mission score per area × lighting</div>
<div class="score-sub">pending — lands once training and scoring finish</div></div>
</div></div></td></tr>
<script>(function(){{
  var built={built_ms}, phases={phases_js}, st=null;
  var NAMES={{design:'designing the experiment',
    implement:'implementing the design',
    train:'training Berlin',
    score:'scoring against the frozen ruler',
    publish:'logging + publishing the result'}};
  var elTime=document.getElementById('live-time'); if(!elTime) return;
  var elTitle=document.getElementById('live-title');
  var elCat=document.getElementById('live-cat');
  var elInit=document.getElementById('live-init');
  var elExplain=document.getElementById('live-explain');
  var shownDesign=null;
  var total=phases.reduce(function(a,p){{return a+p[1]}},0);
  function fmt(s){{s=Math.max(0,Math.floor(s));
    return s<60? s+' s' : Math.floor(s/60)+' m '+('0'+s%60).slice(-2)+' s';}}
  function esc(s){{var d=document.createElement('div');d.textContent=s;return d.innerHTML;}}
  // Live phase truth: the loop force-pushes state/phase.json to the repo's
  // 'status' branch at every phase transition; raw.githubusercontent serves
  // it with CORS. Elapsed counts from the experiment's true start. Once the
  // design agent has written a design (title/eli5/hypothesis/...), it rides
  // along in the same payload — populating this row's expandable detail
  // panel (same click-to-open behavior as every finished row) well before
  // training/scoring finish, not just a phase name in the collapsed row.
  var RAW='https://raw.githubusercontent.com/akaalias/ballpark/status/phase.json';
  function freshen(j){{
    // GitHub Pages pins Cache-Control to 10 min and headers are not
    // configurable — but the CDN caches per-URL, so redirecting to a
    // ?v=<experiment> URL it has never seen always fetches the fresh build.
    // Trigger whenever the running experiment began after this page was
    // built (i.e. a newer build exists), on first load or mid-session.
    var v=String(j.iter_started);
    var cur=new URLSearchParams(location.search).get('v');
    if(j.iter_started*1000>built+90000 && cur!==v){{
      location.replace(location.pathname+'?v='+v+location.hash);
    }}
  }}
  function showDesign(d){{
    if(!d || !d.title || shownDesign===d.title) return;
    shownDesign=d.title;
    elTitle.textContent=d.title;
    if(d.category){{elCat.textContent=d.category;}}
    if(d.init_strategy){{elInit.textContent=d.init_strategy;}}
    var parts=[];
    if(d.eli5) parts.push("<div class='eb eb-eli'><div class='eb-h'>In plain words</div><p>"+esc(d.eli5)+"</p></div>");
    if(d.hypothesis) parts.push("<div class='eb eb-hyp'><div class='eb-h'>Hypothesis</div><p>"+esc(d.hypothesis)+"</p></div>");
    if(d.method) parts.push("<div class='eb eb-met'><div class='eb-h'>Method</div><p>"+esc(d.method)+"</p></div>");
    if(d.expected_outcome) parts.push("<div class='eb eb-exp'><div class='eb-h'>Expected outcome</div><p>"+esc(d.expected_outcome)+"</p></div>");
    if(parts.length) elExplain.innerHTML=parts.join('');
  }}
  function refresh(){{
    fetch(RAW+'?t='+Date.now()).then(function(r){{return r.ok?r.json():null}})
      .then(function(j){{
        if(!(j&&j.phase))return;
        st=j;freshen(j);showDesign(j.design);
      }}).catch(function(){{}});
  }}
  refresh(); setInterval(refresh, 30000);
  // Backgrounded tabs get setInterval throttled by the browser (often to
  // minutes, not 30s) — so a tab left open in the background can sit on a
  // stale st.iter_started from an experiment that already finished, then
  // jump to the real value once it catches up. Force a fetch the instant
  // the tab becomes visible again instead of waiting for the throttled
  // interval, so returning to the tab never shows stale-then-jumping time.
  document.addEventListener('visibilitychange', function(){{
    if(document.visibilityState==='visible') refresh();
  }});
  function tick(){{
    var now=Date.now()/1000, msg;
    if(st && st.phase==='idle'){{
      msg=st.note || 'no experiment running right now — the last batch finished; the best result stands';
    }} else if(st && (st.phase==='waiting')){{
      msg='paused — waiting for agent capacity; resumes automatically';
    }} else if(st && now-st.phase_started < 5400){{
      msg=fmt(now-st.iter_started)+' · now '+(NAMES[st.phase]||st.phase);
    }} else if(st){{
      msg='stale ('+fmt(now-st.phase_started)+' since last report)';
    }} else {{
      var t=(Date.now()-built)/1000, acc=0, ph=null;
      for(var i=0;i<phases.length;i++){{acc+=phases[i][1];
        if(t<acc){{ph=phases[i][0];break;}}}}
      msg=ph ? '~'+fmt(t)+' · est. '+ph
             : '~'+fmt(t)+' · past usual '+fmt(total);
    }}
    elTime.textContent=msg;
  }}
  tick(); setInterval(tick,1000);
}})();</script>"""


GOAL_MAP = REPO_ROOT / "gallery" / "goal_map.png"


def ensure_goal_map():
    """Render the GOAL illustration: the exact same held-out viewpoints the
    real error map uses, drawn as if every one were a usable fix.

    This is deliberately a MOCK, not a measurement — no model produced it. It
    exists so the overview can show the target state next to the real one, and
    it is labelled as an illustration everywhere it appears. Reuses the frozen
    scorer's own renderer (feeding error=0 for every point) so the two images
    are pixel-for-pixel comparable rather than drawn by different code."""
    if GOAL_MAP.exists():
        return GOAL_MAP
    try:
        from pipeline.common import DATA_DIR, load_meta
        from pipeline.dataset import list_crops
        from pipeline.score import MAX_EVAL_CROPS_PER_BUCKET, render_heatmap
        meta = load_meta("berlin")
        crops = list_crops("berlin", meta["width"], meta["height"], "eval")
        if len(crops) > MAX_EVAL_CROPS_PER_BUCKET:
            idx = np.linspace(0, len(crops) - 1, MAX_EVAL_CROPS_PER_BUCKET).astype(int)
            crops = [crops[i] for i in idx]
        render_heatmap("berlin", DATA_DIR, [(c["cx"], c["cy"], 0.0) for c in crops], GOAL_MAP)
    except Exception:
        return None
    return GOAL_MAP


def _usable_of(e):
    try:
        b = json.loads(e["metrics_json"])["areas"][0]["buckets"]
        return max(c.get("usable_fix_rate") or 0.0 for c in b.values())
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None


def challenge_block(exps):
    """Overview hero: what one camera frame must become, then the starting
    baseline's error map beside the CURRENT model's — both real, both
    measured on the same test points by the same renderer.

    This used to put the baseline next to a synthetic picture of the goal,
    which was right while the baseline was the state of the art. It is not
    right now: the right-hand panel is an actual result, and showing an
    illustration in its place both understates the work and reads, to anyone
    skimming, as though nothing had been achieved."""
    dev = [e for e in exps if e["kind"] != "holdout_check"]
    base = next((e for e in dev if e.get("artifacts_dir")), None)
    champ = next((e for e in reversed(dev)
                  if e.get("kept") and e.get("artifacts_dir")
                  and e["primary_metric"] is not None
                  and e["primary_metric"] < FAIL), None)
    if not base:
        return ""
    real = REPO_ROOT / base["artifacts_dir"] / "heatmaps" / "heatmap_berlin.png"
    if not real.exists():
        return ""
    rel_real = Path(base["artifacts_dir"]) / "heatmaps" / "heatmap_berlin.png"

    # Prefer the current model's own map; fall back to the goal illustration
    # only while there is no kept result to show (i.e. a fresh lineage).
    now_map = (REPO_ROOT / champ["artifacts_dir"] / "heatmaps" / "heatmap_berlin.png"
               if champ else None)
    if champ and now_map.exists():
        rel_now = Path(champ["artifacts_dir"]) / "heatmaps" / "heatmap_berlin.png"
        now_usable = _usable_of(champ)
        right = (
            f"<figure class='wex-map'><a href='{rel_now}'>"
            f"<img src='{rel_now}' loading='lazy'></a>"
            "<figcaption><b>where we are — measured</b><br>mission score "
            f"<b class='num'>{fmt_score(champ['primary_metric'])}</b> · usable "
            f"fixes <b class='num'>"
            f"{f'{100*now_usable:.1f}%' if now_usable is not None else '—'}</b>"
            "<br>The identical test points, answered by the current model. "
            "Green is a fix inside 100 m — close enough to correct the "
            "aircraft's drift. Nothing about the test changed; only the "
            "answers did.</figcaption></figure>")
        lead = ("<b>This, on a map, is the goal.</b> Every dot is one held-out "
                "viewpoint: ground the model trained on, framed from a position "
                "and heading it has never seen. <b>Left is where this started. "
                "Right is where it is now.</b> Turning red into green — without "
                "ever being confidently wrong — is the whole project.")
    else:
        goal = ensure_goal_map()
        if not goal:
            return ""
        rel_goal = Path("gallery") / "goal_map.png"
        right = (
            f"<figure class='wex-map'><a href='{rel_goal}'>"
            f"<img src='{rel_goal}' loading='lazy'></a>"
            "<figcaption><b>the goal — an illustration, not a result</b><br>"
            "mission score <b class='num'>0.000</b> · usable fixes "
            "<b class='num'>100%</b><br>The identical test points drawn as if "
            "every one were a usable fix (within 100 m). <i>No model produced "
            "this image</i> — it is the target, not an achievement.</figcaption>"
            "</figure>")
        lead = ("<b>This, on a map, is the goal.</b> Every dot is one held-out "
                "viewpoint: ground the model trained on, framed from a position "
                "and heading it has never seen. <b>Left is where we are, "
                "measured. Right is what winning looks like.</b>")

    usable = _usable_of(base)
    usable_s = f"{100*usable:.0f}%" if usable is not None else "—"

    info = workedexample.ensure(base["artifacts_dir"])
    # The same frame, answered by the current model — the contract paragraph
    # otherwise leaves the reader on the baseline's 2 km miss.
    champ_answer = ""
    if champ and champ is not base:
        ci = workedexample.ensure(champ["artifacts_dir"])
        if ci and "miss_m" in ci:
            champ_answer = (f" The current model answers that same frame within "
                            f"<b>{fmt_m(ci['miss_m'])}</b>.")
    contract = ""
    if info and "lat_true" in info:
        fr = Path(base["artifacts_dir"]) / info["frame"]
        contract = (
            "<div class='contract-row'>"
            f"<figure class='contract-col'><a href='{fr}'>"
            f"<img src='{fr}' loading='lazy'></a>"
            "<figcaption><b>what the UAV sees</b> — one real 128 m frame. "
            "No map aboard, no internet, no GPS.</figcaption></figure>"
            "<div class='contract-arrow'>&rarr;</div>"
            "<figure class='contract-col'><div class='contract-out'>"
            "<div class='co-row'><span class='co-k'>lat</span>"
            f"<span class='co-v num'>{info['lat_true']:.5f}</span></div>"
            "<div class='co-row'><span class='co-k'>lon</span>"
            f"<span class='co-v num'>{info['lon_true']:.5f}</span></div>"
            "<div class='co-row'><span class='co-k'>confidence</span>"
            "<span class='co-v num'>high</span></div></div>"
            "<figcaption><b>what it must return</b> — the true answer for that "
            "frame, computed on board.</figcaption></figure>"
            "</div>"
            "<p class='contract-note'>The first baseline answered this frame "
            f"<b>{fmt_m(info['miss_m'])}</b> away at confidence "
            f"<span class='num'>{info['conf']:.2f}</span> — confident and "
            "wrong, the one answer a UAV must never get."
            + champ_answer + "</p>")

    return (
        # SPLIT MARKER — render_overview cuts here so the contract can sit in
        # "The question" and the before/after maps in "The evidence", which is
        # where each of them is actually an answer to something.
        contract + "<!--SPLIT-->"
        f"<div class='pnote'><p>{lead}</p></div>"
        "<div class='wex-row'><div class='wex-imgs'>"
        f"<figure class='wex-frame'><a href='{rel_real}'>"
        f"<img src='{rel_real}' loading='lazy'></a>"
        "<figcaption><b>where this started — measured</b><br>mission score "
        f"<b class='num'>{fmt_score(base['primary_metric'])}</b> · usable fixes "
        f"<b class='num'>{usable_s}</b><br>Red is a miss beyond 250 m. The "
        "baseline is confident on every frame and wrong on every frame — the "
        "worst value the metric allows, because for a UAV a confident wrong "
        "answer is worse than silence.</figcaption></figure>"
        "<div class='wex-arr'>→</div>"
        + right + "</div></div>")

def page_header(title, sub_html):
    """THE one header for every inner page (log, designs, lineage) — same
    markup, same classes, no per-page typography. Do not hand-roll page
    headers; call this."""
    return (f"<header class='page-head'>"
            f"<h1>{title}</h1>"
            f"<p class='page-sub'>{sub_html}</p></header>")


# Social sharing (Open Graph + Twitter cards). One shared card image — a
# mosaic of real geo-tiles with the headline — generated once, locally, into
# assets/social/card.jpg (CI has no raster; same scheme as the tile grids).
SITE_URL = "https://alexisrondeau.me/ballpark/"


def social_meta(path, title, desc):
    """OG/Twitter tags for one page. path is '' for the root overview or
    'gallery/<page>.html'; the card image is shared across all pages."""
    url = SITE_URL + path
    img = SITE_URL + "assets/social/card.jpg"
    return (
        f'<meta property="og:type" content="website">\n'
        f'<meta property="og:site_name" content="“Not all who wander are lost”">\n'
        f'<meta property="og:title" content="{esc(title)}">\n'
        f'<meta property="og:description" content="{esc(desc)}">\n'
        f'<meta property="og:url" content="{url}">\n'
        f'<meta property="og:image" content="{img}">\n'
        f'<meta property="og:image:width" content="1200">\n'
        f'<meta property="og:image:height" content="630">\n'
        f'<meta property="og:image:alt" content="A mosaic of aerial tiles of '
        f'Berlin around the question: Can a UAV learn a city by heart?">\n'
        f'<meta name="twitter:card" content="summary_large_image">\n'
        f'<meta name="twitter:title" content="{esc(title)}">\n'
        f'<meta name="twitter:description" content="{esc(desc)}">\n'
        f'<meta name="twitter:image" content="{img}">\n'
        f'<meta name="description" content="{esc(desc)}">')


def topnav(active, root=False):
    hrefs = {"overview": "index.html" if root else "../index.html",
             "log": "gallery/index.html" if root else "index.html",
             "paths": ("gallery/inference-paths.html" if root
                       else "inference-paths.html"),
             "lineage": ("gallery/research-lineage.html" if root
                         else "research-lineage.html"),
             "evolution": ("gallery/research-evolution.html" if root
                           else "research-evolution.html"),
             "flight": ("gallery/flight-path.html" if root
                        else "flight-path.html"),
             "notebook": ("gallery/lab-notebook.html" if root
                          else "lab-notebook.html")}
    links = []
    for key, label in NAV_PAGES:
        classes = [c for c in (("on" if key == active else ""),
                               ("cta" if key == "flight" else "")) if c]
        cls = f" class='{' '.join(classes)}'" if classes else ""
        links.append(f"<a href='{hrefs[key]}'{cls}>{label}</a>")
    return ("<nav class='topnav'><span class='brand'>Ballpark</span>"
            + "".join(links) + "</nav>")


def fmt_dur(s):
    if s is None:
        return "—"
    s = int(s)
    return f"{s // 60} m {s % 60:02d} s" if s >= 60 else f"{s} s"


def fmt_score(v):
    """The PRIMARY metric is the mission score, not a distance:
    (1 - usable_fix_rate) + false_fix_rate, minimized. 0 = every frame a
    usable fix; 1 = abstains everywhere; 2 = confidently wrong everywhere."""
    if v is None:
        return "—"
    if v >= 1e15:
        return "no prior best"
    if v >= FAIL:
        return "gated fail"
    return f"{v:.3f}"


def fmt_m(v):
    """Distances (diagnostics only — median/geomean/percentiles)."""
    if v is None:
        return "—"
    if v >= 1e15:
        return "no prior best"
    if v >= FAIL:
        return "gated fail"
    if v >= 1000:
        return f"{v/1000:,.2f} km"
    return f"{v:,.1f} m"


def chart_svg(exps):
    """Static SVG progress chart: worst-case error per experiment, log y,
    running-best step line, dashed target rule, Tufte marks. Marks carry
    data-id/data-tip for the JS chart<->table interlink."""
    W, H = 1000, 300
    ml, mr, mt, mb = 58, 16, 16, 34
    iw, ih = W - ml - mr, H - mt - mb
    if not exps:
        return ""
    finite = [e["primary_metric"] for e in exps
              if e["primary_metric"] and e["primary_metric"] < FAIL]
    ymax = min(max(finite + [1.0]) * 1.15, 2.05)
    ymin = 0.0

    def y(v):
        v = max(min(v, ymax), ymin)
        return mt + ih * (1 - (v - ymin) / max(ymax - ymin, 1e-9))

    n = len(exps)

    def x(i):
        return ml + iw * (0.5 if n == 1 else i / (n - 1)) * (0.96 if n > 1 else 1)

    parts = [f"<svg viewBox='0 0 {W} {H}' role='img' "
             f"aria-label='worst-case position error per experiment'>"]
    for v in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        if ymin <= v <= ymax:
            parts.append(f"<line class='tick-line' x1='{ml}' x2='{W-mr}' y1='{y(v):.1f}' y2='{y(v):.1f}'/>")
            parts.append(f"<text class='axis-lab' x='{ml-8}' y='{y(v)+3:.1f}' text-anchor='end'>{v:g}</text>")
    parts.append(f"<text class='axis-lab' x='{ml-8}' y='{mt-4}' text-anchor='end'>score</text>")
    parts.append(f"<line x1='{ml}' x2='{W-mr}' y1='{y(0.0):.1f}' y2='{y(0.0):.1f}' "
                 f"stroke='#8c2f1f' stroke-width='1.5' stroke-dasharray='6 5'/>")
    parts.append(f"<text class='axis-lab' x='{W-mr}' y='{y(0.0)+12:.1f}' "
                 f"text-anchor='end' fill='#8c2f1f'>0 — every frame a usable fix</text>")

    # Running-best step line, one segment per evaluation era. Within an era
    # the line only ever steps DOWN (the loop keeps only improvements). A
    # kept experiment that scores worse than the running best is, by loop
    # semantics, impossible — so it marks a deliberate eval-set change
    # (bootstrap data revisions): the line BREAKS there and restarts, and a
    # vertical rule labels the discontinuity.
    best = None
    segments, cur, resets = [], [], []
    for i, e in enumerate(exps):
        if e["kind"] == "holdout_check":
            continue
        v = e["primary_metric"]
        if not v or v >= FAIL or not e["kept"]:
            continue
        if best is None:
            best, cur = v, [(x(i), y(v))]
        elif v < best:
            cur.append((x(i), y(best)))
            best = v
            cur.append((x(i), y(v)))
        else:  # eval-set reset — new ruler, new segment
            segments.append(cur)
            resets.append(x(i))
            best, cur = v, [(x(i), y(v))]
    if cur:
        cur.append((x(n - 1), y(best)))
        segments.append(cur)
    # Reset labels: a paper-colored halo (paint-order stroke) keeps them
    # legible wherever the data line runs; stagger vertically only when two
    # rules sit close enough for their labels to collide.
    LABEL_W = 130
    prev_rx, prev_row = None, 0
    for rx in resets:
        row = (prev_row + 1) % 2 if (prev_rx is not None and rx - prev_rx < LABEL_W) else 0
        prev_rx, prev_row = rx, row
        parts.append(f"<line x1='{rx:.1f}' x2='{rx:.1f}' y1='{mt}' y2='{H-mb}' "
                     f"stroke='#9b998c' stroke-width='1' stroke-dasharray='3 4'/>")
        parts.append(f"<text class='axis-lab' x='{rx+5:.1f}' y='{mt+11+row*13}' "
                     f"stroke='#fffff8' stroke-width='4' "
                     f"style='paint-order:stroke'>eval set changed</text>")
    for seg in segments:
        if len(seg) > 1:
            d = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in seg)
            parts.append(f"<path d='{d}' fill='none' stroke='#111' stroke-width='2'/>")

    for i, e in enumerate(exps):
        v = e["primary_metric"]
        if v is None:
            continue
        kind = ("holdout check — logged, never drives keep/revert"
                if e["kind"] == "holdout_check"
                else "rejected — never trained" if is_rejected(e)
                else "kept" if e["kept"] else "discarded")
        if e.get("is_pivot"):
            kind += " · pivot-directed (patience spent)"
        tip = (f"<b>#{e['id']} {esc(e['title'])}</b>"
               f"<span class='t-note'>{fmt_m(v)} · {kind} · click to open</span>")
        tip_attr = esc(tip)
        common = f"class='pt' data-id='{e['id']}' data-tip=\"{tip_attr}\""
        if v >= FAIL:
            parts.append(f"<text {common} x='{x(i):.1f}' y='{mt+11}' text-anchor='middle' "
                         f"fill='#8c2f1f' font-weight='700' font-size='14'>×</text>")
        elif e["kind"] == "holdout_check":
            parts.append(f"<circle {common} cx='{x(i):.1f}' cy='{y(v):.1f}' r='4.5' "
                         f"fill='#fffff8' stroke='#8a6a1e' stroke-width='1.5'/>")
        elif e["kept"]:
            parts.append(f"<circle {common} cx='{x(i):.1f}' cy='{y(v):.1f}' r='4.5' fill='#111'/>")
        else:
            parts.append(f"<circle {common} cx='{x(i):.1f}' cy='{y(v):.1f}' r='4' fill='#b9b6a6'/>")
        if e.get("is_pivot"):
            py = H - 21
            parts.append(f"<path {common} d='M{x(i)-4.5:.1f},{py-7:.1f} "
                         f"L{x(i)+4.5:.1f},{py-7:.1f} L{x(i):.1f},{py:.1f} Z' fill='#8a6a1e'/>")
        parts.append(f"<text class='axis-lab' x='{x(i):.1f}' y='{H-12}' text-anchor='middle'>{e['id']}</text>")
    parts.append(f"<text class='axis-lab' x='{ml+iw/2:.0f}' y='{H-1}' text-anchor='middle'>experiment №</text>")
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# The full project history, across every evaluation era.
#
# experiments.sqlite holds only the CURRENT era, because the lineage was wiped
# each time the evaluation or the metric was corrected. Rendered alone it says
# the problem took five experiments, which is false: it took 83 across five
# eras and ten days of them were spent against a broken evaluation.
#
# lineage_history.sqlite (built by autoresearch.rescore_history) is the whole
# record placed on ONE ruler — every surviving model re-run through today's
# frozen scorer on today's eval set. See that module's docstring for what is
# and is not comparable.
# ---------------------------------------------------------------------------
HISTORY_DB = REPO_ROOT / "lineage_history.sqlite"

# One band per era. Low-chroma washes drawn from the site's own three hues, so
# the bands recede behind the data: grey for the eras spent lost, ochre as the
# evaluation is repaired, the accent for the era that worked. Identity is never
# colour-alone — every band is also labelled in place.
ERA_TINT = {
    "bootstrap": "rgba(107,106,96,.055)",
    "main":      "rgba(107,106,96,.105)",
    "slim_v1":   "rgba(138,106,30,.075)",
    "eval_v2":   "rgba(138,106,30,.135)",
    "mission":   "rgba(140,47,31,.085)",
}
ERA_INK = {
    "bootstrap": "#6b6a60", "main": "#6b6a60", "slim_v1": "#8a6a1e",
    "eval_v2": "#8a6a1e", "mission": "#8c2f1f",
}
# Short band captions. The full era description lives in the eras table and is
# surfaced on hover; these are what fits above a band.
ERA_SHORT = {
    "bootstrap": "bootstrap",
    "main": "4 areas · 6 lighting buckets · region holdout",
    "slim_v1": "berlin only",
    "eval_v2": "viewpoint holdout",
    "mission": "mission score",
}


def load_history():
    """Read the merged cross-era record. Returns (rows, eras), or ([], []) if
    the history DB has not been built yet — every caller degrades to the
    current era alone rather than failing."""
    if not HISTORY_DB.exists():
        return [], []
    import sqlite3
    conn = sqlite3.connect(HISTORY_DB)
    conn.row_factory = lambda cur, row: {d[0]: row[i]
                                         for i, d in enumerate(cur.description)}
    rows = conn.execute("SELECT * FROM history ORDER BY seq ASC").fetchall()
    eras = conn.execute("SELECT * FROM eras ORDER BY era_index ASC").fetchall()
    conn.close()
    return rows, eras


def history_chart_svg(rows, eras):
    """Every experiment the project has ever run, on one ruler, with a band
    per evaluation era.

    Two things are deliberately NOT the same encoding, because they are not
    the same claim:

      * the DOTS are today's mission score for that model — measured, on
        today's eval set, by today's frozen scorer;
      * black vs. grey is whether the loop KEPT it, decided at the time under
        whatever metric was then in force.

    So a black dot high on the chart is not a contradiction — it is the record
    of a metric that rewarded the wrong thing, which is the entire point of
    showing this. The running-best line is the honest one: the best mission
    score anyone had actually achieved as of that experiment, continuous
    across era boundaries because the ruler no longer changes at them."""
    if not rows:
        return ""
    W, H = 1000, 384
    ml, mr, mt, mb = 58, 16, 54, 46      # mt leaves room for the band captions
    iw, ih = W - ml - mr, H - mt - mb
    ymin, ymax = 0.0, 2.05
    n = len(rows)

    def y(v):
        return mt + ih * (1 - (max(min(v, ymax), ymin) - ymin) / (ymax - ymin))

    def x(i):
        return ml + iw * (0.5 if n == 1 else i / (n - 1))

    step = iw / max(n - 1, 1)
    p = [f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='mission score for "
         f"every experiment across all five evaluation eras'>"]

    # --- era bands, first so everything else sits on top of them ---
    for era in eras:
        lo = x(era["seq_start"] - 1) - step / 2
        hi = x(era["seq_end"] - 1) + step / 2
        lo, hi = max(lo, ml - 4), min(hi, W - mr + 4)
        if hi <= lo:
            continue
        key = era["key"]
        tip = esc(f"<b>{esc(era['label'])}</b><span class='t-note'>"
                  f"{esc(era['note'])}<br><br>Optimised then: "
                  f"{esc(era['ruler'])} · asked over {esc(era['eval_set'])}"
                  f"</span>")
        p.append(f"<rect class='era' data-tip=\"{tip}\" x='{lo:.1f}' y='{mt}' "
                 f"width='{hi-lo:.1f}' height='{ih}' "
                 f"fill='{ERA_TINT.get(key, 'rgba(107,106,96,.06)')}'/>")
        if era["era_index"]:
            p.append(f"<line x1='{lo:.1f}' x2='{lo:.1f}' y1='{mt}' y2='{mt+ih}' "
                     f"stroke='#d9d5c3' stroke-width='1'/>")
        # Caption above the band. Narrow bands get a tick and a label that
        # leans out of the band rather than an unreadable squeeze inside it.
        cx, wid = (lo + hi) / 2, hi - lo
        ink = ERA_INK.get(key, "#6b6a60")
        label = ERA_SHORT.get(key, era["label"])
        if wid >= len(label) * 5.4:
            p.append(f"<text class='era-lab' x='{cx:.1f}' y='{mt-10}' "
                     f"text-anchor='middle' fill='{ink}'>{esc(label)}</text>")
        else:
            row = mt - 10 - 13 * (era["era_index"] % 2)
            p.append(f"<line x1='{cx:.1f}' x2='{cx:.1f}' y1='{row+3}' y2='{mt-1}' "
                     f"stroke='{ink}' stroke-width='1'/>")
            p.append(f"<text class='era-lab' x='{cx:.1f}' y='{row}' "
                     f"text-anchor='middle' fill='{ink}'>{esc(label)}</text>")

    # --- gridlines and the goal rule ---
    for v in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        p.append(f"<line class='tick-line' x1='{ml}' x2='{W-mr}' "
                 f"y1='{y(v):.1f}' y2='{y(v):.1f}'/>")
        p.append(f"<text class='axis-lab' x='{ml-8}' y='{y(v)+3:.1f}' "
                 f"text-anchor='end'>{v:g}</text>")
    p.append(f"<text class='axis-lab' x='{ml-8}' y='{mt-4}' text-anchor='end'>score</text>")
    # 1.0 is not just another gridline: at or above it the model is worth less
    # than silence, which is the whole argument for this metric.
    p.append(f"<line x1='{ml}' x2='{W-mr}' y1='{y(1.0):.1f}' y2='{y(1.0):.1f}' "
             f"stroke='#9b998c' stroke-width='1' stroke-dasharray='2 4'/>")
    # Both rule captions hug the LEFT edge. Everything interesting on this
    # chart happens on the right — the descent through 1.0 and the landing on
    # 0 — so a right-anchored caption is exactly where the data will be.
    p.append(f"<text class='axis-lab' x='{ml+6}' y='{y(1.0)-5:.1f}' "
             f"text-anchor='start'>1 — no better than saying nothing</text>")
    p.append(f"<line x1='{ml}' x2='{W-mr}' y1='{y(0.0):.1f}' y2='{y(0.0):.1f}' "
             f"stroke='#8c2f1f' stroke-width='1.5' stroke-dasharray='6 5'/>")
    p.append(f"<text class='axis-lab' x='{ml+6}' y='{y(0.0)-6:.1f}' "
             f"text-anchor='start' fill='#8c2f1f'>0 — every frame a usable fix</text>")

    # --- best achieved so far, on today's ruler, unbroken across eras ---
    best, pts = None, []
    for i, r in enumerate(rows):
        v = r["mission_score"]
        if v is None or r["kind"] == "holdout_check":
            continue
        if best is None:
            best, pts = v, [(x(i), y(v))]
        elif v < best:
            pts += [(x(i), y(best)), (x(i), y(v))]
            best = v
    if pts:
        pts.append((x(n - 1), y(best)))
        p.append("<path d='M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts)
                 + "' fill='none' stroke='#111' stroke-width='2'/>")

    # --- one mark per experiment ---
    prov_note = {
        "rescored": "re-scored today from its exported model",
        "native": "measured natively — this era's own ruler",
        "derived": "rates recovered from the logged record (model files gone)",
        "gated": "failed a deployment gate or never trained",
        "incomparable": "ran on 10 m/px imagery — not answerable on today's eval set",
        "unrecoverable": "model and rates both lost",
        "holdout": "blind Hamburg holdout check — a different area, so no Berlin score exists",
    }
    for i, r in enumerate(rows):
        v, prov = r["mission_score"], r["provenance"]
        cx = x(i)
        verdict = ("holdout check" if r["kind"] == "holdout_check"
                   else "kept at the time" if r["kept"] else "discarded at the time")
        head = f"<b>{esc(r['era_label'])} #{r['src_id']} — {esc(r['title'] or '')}</b>"
        if v is not None:
            detail = (f"mission score {v:.3f} · usable "
                      f"{(r['usable_fix_rate'] or 0)*100:.1f}% · false fix "
                      f"{(r['false_fix_rate'] or 0)*100:.1f}%")
        else:
            detail = prov_note[prov]
        tip = esc(f"{head}<span class='t-note'>{esc(detail)} · {verdict}"
                  f"<br>{esc(prov_note[prov])}</span>")
        # Every era has a row in the table below, so every mark links to one.
        # The id must match that row's DOM id exactly: the current era keeps
        # bare numbers (older deep links point at those), earlier eras are
        # qualified because their numbering restarts.
        uid = str(r["src_id"]) if r["era"] == "mission" else f"{r['era']}-{r['src_id']}"
        common = f"class='pt' data-id='{uid}' data-tip=\"{tip}\""
        if prov == "holdout":
            p.append(f"<circle {common} cx='{cx:.1f}' cy='{mt+ih-6}' r='3.5' "
                     f"fill='#fffff8' stroke='#8a6a1e' stroke-width='1.5'/>")
        elif prov in ("incomparable", "unrecoverable"):
            # No score exists and none can be manufactured — say so with a
            # break in the record rather than a plotted zero.
            p.append(f"<line {common} x1='{cx:.1f}' x2='{cx:.1f}' "
                     f"y1='{mt+ih-7}' y2='{mt+ih}' stroke='#9b998c' "
                     f"stroke-width='1.5' stroke-dasharray='2 2'/>")
        elif prov == "gated" or v is None:
            p.append(f"<text {common} x='{cx:.1f}' y='{mt+11}' "
                     f"text-anchor='middle' fill='#8c2f1f' font-weight='700' "
                     f"font-size='13'>×</text>")
        elif r["kind"] == "holdout_check":
            p.append(f"<circle {common} cx='{cx:.1f}' cy='{y(v):.1f}' r='4' "
                     f"fill='#fffff8' stroke='#8a6a1e' stroke-width='1.5'/>")
        elif r["kept"]:
            p.append(f"<circle {common} cx='{cx:.1f}' cy='{y(v):.1f}' r='3.6' fill='#111'/>")
        else:
            p.append(f"<circle {common} cx='{cx:.1f}' cy='{y(v):.1f}' r='3' fill='#b9b6a6'/>")

    # --- x labels: per-era numbering, placed greedily with a hard minimum gap.
    # Numbering restarts inside every era, so two bands' labels can otherwise
    # collide across a boundary ("...64 65 1 2..." runs together). Each era
    # always gets its own #1 as an anchor; everything after it has to earn its
    # place by clearing MIN_GAP from whatever was drawn last, boundary or not.
    MIN_GAP = 24.0
    ticks = []          # [(x, label)] chosen, left to right
    for era in eras:
        lo_i, hi_i = era["seq_start"] - 1, era["seq_end"] - 1
        for i in range(lo_i, hi_i + 1):
            cx, lab = x(i), rows[i]["src_id"]
            clear = not ticks or cx - ticks[-1][0] >= MIN_GAP
            if i == lo_i:
                # An era's opening number anchors its band, so it always gets
                # drawn; the previous era's trailing number yields to it.
                while ticks and cx - ticks[-1][0] < MIN_GAP:
                    ticks.pop()
                ticks.append((cx, lab))
            elif clear:
                ticks.append((cx, lab))
    for cx, lab in ticks:
        p.append(f"<text class='axis-lab' x='{cx:.1f}' y='{mt+ih+15}' "
                 f"text-anchor='middle'>{lab}</text>")
    p.append(f"<text class='axis-lab' x='{ml+iw/2:.0f}' y='{H-8}' "
             f"text-anchor='middle'>experiment № — restarting at 1 with each "
             f"evaluation era</text>")
    p.append("</svg>")
    return "\n".join(p)


def history_rows_html(hist_rows, eras):
    """Table rows for the FOUR EARLIER eras.

    Deliberately not run through the current-era renderer. Those rows can
    show a heatmap, a worked example, a training log and an architecture
    figure because their artifacts sit in runs/ next to a metrics.json that
    the current scorer wrote. An old experiment's artifacts were written by a
    different scorer against a different eval set — rendering them in the same
    frame would present a picture of one measurement under the label of
    another. So these rows carry what genuinely survives: the design record
    the agent pre-registered, the verdict the loop reached at the time in its
    own units, and the re-measurement on today's ruler, each labelled as what
    it is."""
    if not hist_rows:
        return ""
    by_era = {e["era_index"]: e for e in eras}
    out = []
    for r in reversed(hist_rows):
        if r["era"] == "mission":       # the current era renders in full above
            continue
        era = by_era.get(r["era_index"], {})
        uid = f"{r['era']}-{r['src_id']}"
        num = f"{r['era_index'] + 1}.{r['src_id']}"
        v = r["mission_score"]
        prov = r["provenance"]
        # Two independent facts, and conflating them puts words in the record's
        # mouth: what the loop DECIDED at the time (kept / discarded / gated),
        # and whether a score exists on TODAY's ruler. A bootstrap experiment
        # can perfectly well have been kept and still be unscoreable now.
        if prov == "holdout":
            stat, cls = "HOLDOUT", "hold"
        elif prov == "gated":
            stat, cls = "GATED FAIL", "fail"
        elif prov in ("incomparable", "unrecoverable"):
            stat, cls = "NO SCORE", "kept" if r["kept"] else "disc"
        elif r["kept"]:
            stat, cls = "KEPT", "kept"
        else:
            stat, cls = "DISCARDED", "disc"
        if cls == "hold":
            why = ("Blind Hamburg holdout check. Logged for honesty, never "
                   "allowed to drive keep/revert.")
        elif cls == "fail":
            why = ("Failed a deployment gate: too large, too slow, or it "
                   "abstained on too many frames to be scoreable.")
        elif r["kept"]:
            why = ("The loop kept this — it beat the running best under the "
                   "metric then in force.")
        else:
            why = ("The loop discarded this — no improvement under the metric "
                   "then in force.")
        # Where the two rulers disagree, say so on the row rather than leaving
        # it to be inferred from the chart — that disagreement is the substance
        # of the whole exercise.
        if v is not None and v >= 1.0 and r["kept"] and cls not in ("hold", "fail"):
            why += f" On today's ruler it scores {v:.3f} — worse than silence."
        elif v is None and prov in ("incomparable", "unrecoverable"):
            why += (" It cannot be placed on today's ruler at all — see the row "
                    "for why.")
        era_metric_txt = "—"
        if r["era_metric"] is not None and r["era_metric"] < FAIL:
            em = r["era_metric"]
            era_metric_txt = (f"{em/1000:,.2f} km" if em >= 1000 else
                              f"{em:,.1f} m" if em >= 10 else f"{em:.3f}")
        tint = ERA_TINT.get(r["era"], "")
        out.append(f"""<tr class="row-main hist-row{' kept-row' if cls == 'kept' else ''}" id="r{uid}"
 style="--era-tint:{tint}" onclick="toggle('{uid}')">
<td><span class="caret">▸</span></td>
<td class="num" title="{esc(era.get('label', ''))} — experiment {r['src_id']} of that era">{num}</td>
<td class="title-cell"><b>{esc(r['title'] or '(untitled)')}</b>
  <span class="mono" style="color:var(--faint)"> {esc((r['git_commit'] or '')[:8])}</span>
  <div class="row-why why-{cls}">{esc(why)}</div></td>
<td><span class="cat">{esc(r['category'] or '—')}</span></td>
<td class="num">{fmt_score(v) if v is not None else '—'}</td>
<td class="num">{fmt_dur(r['duration_s'])}</td>
<td class="num">{cost_cell(r)}</td>
<td><span class="st st-{cls}">{stat}</span></td></tr>""")

        blocks = []
        if r["eli5"]:
            blocks.append(f"<div class='eb eb-eli'><div class='eb-h'>In plain "
                          f"words</div><p>{esc(r['eli5'])}</p></div>")
        for key, cls_, label in (("hypothesis", "eb-hyp", "Hypothesis"),
                                 ("method", "eb-met", "Method"),
                                 ("conclusion", "eb-con", "Conclusion at the time")):
            if r[key]:
                blocks.append(f"<div class='eb {cls_}'><div class='eb-h'>{label}</div>"
                              f"<p>{esc(r[key])}</p></div>")
        if v is not None:
            measured = (
                f"<div class='eb eb-res'><div class='eb-h'>Re-measured on "
                f"today's ruler</div><p>Mission score <b>{v:.3f}</b> &mdash; "
                f"<b>{(r['usable_fix_rate'] or 0)*100:.1f}%</b> of held-out "
                f"frames give a usable fix, <b>{(r['false_fix_rate'] or 0)*100:.1f}%</b> "
                f"are confident and wrong, {(r['abstain_rate'] or 0)*100:.1f}% "
                f"abstain. Median miss {fmt_m(r['median_error_m'])}. "
                f"{HIST_PROV[prov]}</p></div>")
        else:
            measured = (f"<div class='eb eb-res'><div class='eb-h'>Not on "
                        f"today's ruler</div><p>{HIST_PROV[prov]}</p></div>")
        blocks.append(measured)
        blocks.append(cost_block(r))
        blocks.append(
            f"<div class='eb eb-exp'><div class='eb-h'>What it was measured "
            f"against then</div><p>Optimised: {esc(era.get('ruler', '—'))}, "
            f"scoring <b>{era_metric_txt}</b>. Asked over: "
            f"{esc(era.get('eval_set', '—'))}. {esc(era.get('note', ''))}</p></div>")

        metrics = json.loads(r["rescored_json"] or "{}")
        cells = ""
        if metrics.get("areas"):
            cells = (f"<div class='score-head'>Scoreboard &mdash; re-scored "
                     f"today</div><div class='score-sub'>Produced by running "
                     f"this experiment's exported model through the current "
                     f"frozen scorer on the current eval set. It is not what "
                     f"this experiment was told at the time.</div>"
                     f"{cells_table(metrics)}")
        # 45 of the earlier eras' experiments were drawn by the figure agent
        # before figures were switched off for iteration speed. Those drawings
        # are the clearest thing in the whole record about what was actually
        # tried, so they lead the row exactly as the current era's do.
        fig = arch_block({"kind": r["kind"], "arch_svg": r["arch_svg"],
                          "id": uid, "title": r["title"] or ""})
        out.append(f"""<tr class="detail" id="d{uid}" style="display:none"><td colspan="8">
<div class="detail-inner">{fig}<div class="detail-grid">
<div class="explain">{''.join(blocks)}</div>
<div>{cells}
<div class="provenance">era {esc(era.get('label', ''))} &middot; ts {esc((r['ts'] or '')[:19])} &middot;
commit {esc((r['git_commit'] or '')[:12]) or '—'} &middot; artifacts {esc(r['artifacts_dir'] or '—')} &middot;
took {fmt_dur(r['duration_s'])}</div>
{source_block(r)}
</div></div></div></td></tr>""")
    return "".join(out)


HIST_PROV = {
    "rescored": "Measured, not converted: its exported model was re-run "
                "through the current scorer for this page.",
    "native": "Measured on this era's own ruler, which is the current one.",
    "derived": "Its run artifacts were deleted, so the usable- and false-fix "
               "rates were recovered arithmetically from the coverage and "
               "hit-rate this era's scorer logged at the time. The eval set "
               "and target were already identical to today's, so this is a "
               "derivation rather than an estimate.",
    "gated": "It failed a deployment gate, so there was no working model to "
             "score then and none to re-score now.",
    "incomparable": "It ran on 10 m/px imagery, before the switch to 1 m/px "
                    "orthophotos. Feeding it today's crops would not be asking "
                    "it the same question, so it is left unscored rather than "
                    "given a number that means nothing.",
    "unrecoverable": "Its run directory was deleted and its logged record does "
                     "not carry enough to reconstruct the rates. This one is "
                     "simply lost.",
    "holdout": "A blind Hamburg holdout check (§5). It measures a different "
               "area, so no Berlin mission score exists for it.",
}


def cells_table(metrics):
    areas = metrics.get("areas", [])
    if not areas:
        return "<p class='score-sub'>No per-area results recorded for this run.</p>"
    buckets = list(next((a["buckets"] for a in areas if a.get("buckets")), {}))
    if not buckets:
        return "<p class='score-sub'>No per-area results recorded for this run.</p>"
    worst = max((c["score"] for a in areas for c in a.get("buckets", {}).values()
                 if c["score"] is not None), default=None)
    rows = ["<table class='cells'><tr><th>area</th>"]
    rows += [f"<th>{b.replace('_', ' ')}</th>" for b in buckets]
    rows.append("</tr>")
    for a in areas:
        rows.append(f"<tr><td class='smcp'>{esc(a['area'])}</td>")
        for b in buckets:
            c = a.get("buckets", {}).get(b)
            if not c:
                rows.append("<td>—</td>")
                continue
            med = c.get("mission_score")
            classes = []
            if med is not None and c["score"] < 0.9:
                classes.append("cell-good")
            if c["score"] is not None and c["score"] >= FAIL:
                classes.append("cell-bad")
            if worst is not None and c["score"] == worst and worst < FAIL:
                classes.append("cell-worst")
            cls = f" class='{' '.join(classes)}'" if classes else ""
            val = "abstained" if med is None else f"{med:.3f}"
            uf = c.get("usable_fix_rate")
            uf_s = f" <span class='cov'>{100*uf:.0f}% usable</span>" if uf is not None else ""
            rows.append(f"<td{cls}><span class='num'>{val}</span>{uf_s} "
                        f"<span class='cov'>cov {c['coverage']:.2f}</span></td>")
        rows.append("</tr>")
    rows.append("</table>")
    return "".join(rows)


def gates_block(e, metrics):
    failed = [a["gates"].get("failed") for a in metrics.get("areas", [])
              if a.get("gates", {}).get("failed")]
    size = e["model_bytes_max"]
    lat = e["latency_ms_host_proxy"]
    size_s = f"<span class='ok num'>{size/1024:,.0f} KB</span>" if size else "—"
    lat_s = f"<span class='ok num'>{lat:.1f} ms</span>" if lat else "—"
    verdict = (f"<span class='bad'>gate violated — {esc('; '.join(failed))}</span>"
               if failed else "<span class='ok'>all deployment gates passed</span>")
    return (f"<div class='gates'><b>Fits the aircraft?</b> {verdict}<br>"
            f"largest per-area model {size_s} (limit 4,096 KB) · "
            f"single-frame inference {lat_s} on one CPU thread "
            f"(proxy limit 250 ms)</div>")


def heatmap_block(artifacts_dir, metrics):
    """Where the model was tested and how far off it was, per area — split
    out of figures() so it can sit in the scoreboard column, above the
    worked example, instead of trailing at the bottom of the row."""
    art = REPO_ROOT / (artifacts_dir or "")
    if not art.exists():
        return ""
    heat = sorted(art.glob("heatmaps/*.png"))
    if not heat:
        return ""
    med_range = {}
    for a in metrics.get("areas", []):
        meds = [c["median_error_m"] for c in a.get("buckets", {}).values()
                if c.get("median_error_m") is not None]
        if meds:
            med_range[a["area"]] = (min(meds), max(meds))
    figs = []
    for p in heat:
        rel = Path("..") / p.relative_to(REPO_ROOT)
        area = p.stem.replace("heatmap_", "")
        lo_hi = med_range.get(area)
        stat = (f"median miss {lo_hi[0]:,.0f}–{lo_hi[1]:,.0f} m across "
                f"lighting" if lo_hi else "")
        figs.append(f"<figure><a href='{rel}'><img src='{rel}' loading='lazy'></a>"
                    f"<figcaption><b>{esc(area)}</b>{stat}</figcaption></figure>")
    return (f"<div class='arch'>"
            "<div class='arch-h'>Where the model was tested — and how far off it was</div>"
            "<div class='figs-intro'>Each map is one full test area. Every dot is one "
            "held-out <i>viewpoint</i>: the ground under it was mapped during training, "
            "but this exact framing — 11–17 m off the nearest training vantage, at its "
            "own rotation — is one the model has never seen. It was "
            "shown a 128 m crop centered there and asked for its position. Dot color = the "
            f"distance between its answer and the truth — <b style='color:#3c9c3c'>green ≤ {TARGET_M:.0f} m "
            f"(at goal)</b>, <b style='color:#8a6a1e'>amber ≤ {TARGET_M*2.5:.0f} m</b>, "
            "<b style='color:#8c2f1f'>red beyond</b>. <i>A working model turns these maps "
            "green; spatial clusters of red reveal which parts of an area confuse it.</i></div>"
            f"<div class='thumbs maps'>{''.join(figs)}</div></div>")


def figures(artifacts_dir):
    art = REPO_ROOT / (artifacts_dir or "")
    if not art.exists():
        return ""
    samp = sorted(art.glob("samples/*.png"))
    out = []
    if samp:
        by_area = {}
        for p in samp:
            area = p.stem.split("_")[0]
            by_area.setdefault(area, []).append(p)
        inner = [
            "<div class='figs-intro'>In the current configuration, training and eval use "
            "the raw daytime reference imagery as fetched — no synthetic relighting, no "
            "low-light sensor simulation (that machinery still exists in the frozen pipeline "
            "and is used by the full spec's 6-lighting-condition setup, just switched off "
            "here). Below, one example 256 m patch per area, as-is. This illustrates the "
            "<i>dataset</i>, not this experiment's performance — the actual training set is "
            "thousands of distinct crops (see “training data” above), and this "
            "rendering only changes if the source imagery is re-fetched.</div>"]
        for area, ps in sorted(by_area.items()):
            figs = []
            for p in ps:
                rel = Path("..") / p.relative_to(REPO_ROOT)
                bucket = p.stem[len(area) + 1:].replace("_", " ")
                figs.append(f"<figure><a href='{rel}'><img src='{rel}' loading='lazy'></a>"
                            f"<figcaption>{esc(area)} · {esc(bucket)}</figcaption></figure>")
            inner.append(f"<div class='figs-h'>{esc(area)}</div>"
                         f"<div class='thumbs'>{''.join(figs)}</div>")
        out.append(f"<details class='trywrap'><summary>What the training imagery "
                   f"looks like (example patches — illustration, not the training set)"
                   f"</summary>{''.join(inner)}</details>")
    return f"<div class='figs'>{''.join(out)}</div>" if out else ""


def short_model(model_id):
    """'claude-sonnet-5' -> 'Sonnet', etc. — for compact inline labels.
    Falls back to the raw id (or '') rather than guessing at unknown ones."""
    if not model_id:
        return ""
    for key, label in (("opus", "Opus"), ("sonnet", "Sonnet"),
                        ("haiku", "Haiku"), ("fable", "Fable")):
        if key in model_id.lower():
            return label
    return model_id


def timings_block(e):
    """Where the experiment's wall time went — from the per-run timings.json
    the loop commits with every record (pod-era runs onward). Model names
    come from the row's own agent_model_design/_impl (whichever models
    actually ran that experiment — this varies over the project's life as
    the design/impl model assignment has changed), not a hardcoded guess."""
    p = REPO_ROOT / (e.get("artifacts_dir") or "") / "timings.json"
    if not p.exists():
        return ""
    try:
        t = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    design_m = short_model(e.get("agent_model_design"))
    impl_m = short_model(e.get("agent_model_impl"))
    try:
        n_areas = len(json.loads(e.get("metrics_json") or "null").get("areas") or [])
    except (json.JSONDecodeError, AttributeError):
        n_areas = 0
    train_label = f"train {n_areas} area{'s' if n_areas != 1 else ''}" if n_areas else "train"
    parts = [(f"design ({design_m})" if design_m else "design", t.get("agent_design_s")),
             (f"implement ({impl_m})" if impl_m else "implement", t.get("agent_impl_s")),
             (train_label, t.get("train_wall_s")),
             ("score", t.get("score_s")),
             ("samples", t.get("samples_s")),
             ("holdout", t.get("holdout_s")),
             ("publish", t.get("gallery_s"))]
    segs = [f"{esc(n)} <b class='num'>{fmt_dur(v)}</b>"
            for n, v in parts if v]
    if not segs:
        return ""
    total = t.get("total_s")
    tail = (f" · whole experiment <b class='num'>{fmt_dur(total)}</b>"
            if total else "")
    return ("<div class='score-sub' style='margin-top:10px'>"
            "<b>Where this experiment's time went:</b> "
            + " · ".join(segs) + tail + "</div>")


def train_block(artifacts_dir):
    """Summarize the actual training data used, from train_info.json."""
    p = REPO_ROOT / (artifacts_dir or "") / "train_info.json"
    if not p.exists():
        return ""
    try:
        infos = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    if not infos:
        return ""
    rows = ["<table class='cells' style='margin-top:12px'>"
            "<tr><th>training data</th><th>crops</th><th>epochs</th>"
            "<th>train time</th><th>device</th></tr>"]
    for i in infos:
        secs = i.get("train_seconds")
        rows.append(
            f"<tr><td class='smcp'>{esc(i['area'])}</td>"
            f"<td class='num'>{i['n_train_crops']:,}</td>"
            f"<td class='num'>{i['epochs']}</td>"
            f"<td class='num'>{f'{secs:,.0f} s' if secs is not None else '—'}</td>"
            f"<td class='mono'>{esc(i.get('device', '—'))}</td></tr>")
    rows.append("</table>")
    rows.append("<div class='score-sub' style='margin-top:6px'>crops are sampled "
                "fresh each run from the frozen train split (~45,000 distinct "
                "positions per area, times random rotation), never from eval "
                "blocks; how many to use is the experiment's own choice</div>")
    return "".join(rows)


def chain_of(e, exps):
    """Kept-ancestor chain (oldest -> self) as the overlay replay expects."""
    if e["kind"] == "holdout_check":
        return ""
    ids = [str(k["id"]) for k in exps
           if k["kept"] and k["kind"] != "holdout_check" and k["id"] < e["id"]]
    return ",".join(ids + [str(e["id"])])


def arch_block(e, chain_str=""):
    """The agent-drawn architecture diagram, when one exists.

    The loop runs with DRAW_FIGURES=0 on this branch (it cost ~15-20 min an
    iteration), so the current era's five figures were drawn as a backfill and
    45 more survive from the four-area era. Anything without one renders
    nothing rather than a placeholder. The text-only pipeline-stage fallback
    and the worked-example figure that used to live here have moved to
    worked_example_block() / the right column, since without a diagram the
    fallback sentence added nothing worth its own section."""
    if e["kind"] == "holdout_check":
        return ""
    svg = e.get("arch_svg") or ""
    svg = svg if svg.lstrip().startswith("<svg") else ""
    if not svg:
        return ""
    legend = (" — <span style='color:var(--faint)'>gray = frozen contract"
              "</span> · ink = the current design · "
              "<span class='chg'>red = this experiment's change</span>")
    attrs = (f" data-ovfig data-id='{e['id']}' data-chain='{chain_str}' "
             f"data-no='Fig. {e['id']}' data-title='{esc(e['title'])}'"
             if chain_str else "")
    return (f"<div class='arch'><div class='arch-h'>The design under test — "
            f"technical diagram{legend}</div>"
            f"<div class='arch-svg'{attrs}>{svg}</div></div>")


def worked_example_block(e):
    """One REAL worked example — the same held-out crop for every
    experiment, run through the run's actual exported ONNX model. Shown in
    the scoreboard column, above the scoreboard itself."""
    if e["kind"] == "holdout_check":
        return ""
    info = workedexample.ensure(e["artifacts_dir"])
    if not info:
        return ""
    rd = Path("..") / (e["artifacts_dir"] or "")
    fr, mp = rd / info["frame"], rd / info["map"]
    if info.get("has_field"):
        fstat = (f"red glow = its <b>actual internal probability field</b> "
                 f"recovered from the deployed model — the sharpest cell holds "
                 f"<span class='num'>{info['peak_pct']}%</span> of the probability mass "
                 f"(a uniform “no idea” field would be {info['uniform_pct']}%)")
    else:
        fstat = ("this design points at a coordinate directly — it has no "
                 "internal probability field to show")
    # "asis" is the internal bucket name for this branch's unmodified daytime
    # imagery — meaningless to a reader, where "at night" used to read fine.
    light = ("daytime reference imagery, unmodified" if info["bucket"] == "asis"
             else f"simulated {str(info['bucket']).replace('_', ' ')}")
    fig = f"""<div class='wex-row'>
<div class='wex-imgs'>
<figure class='wex-frame'><a href='{fr}'><img src='{fr}' loading='lazy'></a>
<figcaption><b>what the camera saw</b> — a real held-out 128 m crop of
{esc(info['area'])} ({light}); its true spot is the ○ on the map.
Every experiment is shown this same crop.</figcaption></figure>
<div class='wex-arr'>→</div>
<figure class='wex-map'><a href='{mp}'><img src='{mp}' loading='lazy'></a>
<figcaption><b>what this model actually computed</b> — {fstat}.
○ true location · × its answer.</figcaption></figure>
</div>
<div class='wex-stats'>
<div><div class='wex-num num'>{fmt_m(info['miss_m'])}</div>
<div class='wex-lab'>miss, this crop</div></div>
<div><div class='wex-num num'>{info['conf']:.2f}</div>
<div class='wex-lab'>self-reported confidence</div></div>
</div></div>"""
    return f"<div class='arch'><div class='arch-h'>One real test, end to end</div>{fig}</div>"



# --------------------------------------------------------------------------
# The agent-editable source, inline, per experiment.
#
# The whole search space is two files -- model/model.py and model/train.py --
# so an experiment IS a diff of those two. Showing the prose without the code
# describes the research; showing the code is the research.
#
# WHERE THE SOURCE COMES FROM, and why the distinction is load-bearing:
#
#   exact         runs/<id>/model_src/{model,train}.py -- the bytes that
#                 actually trained, snapshotted by loop.sh's snapshot_model_src()
#                 before any revert can destroy them. Only exists from
#                 2026-07-31 on, which is when that guard was added.
#   reconstructed read out of the experiment's own git commit tree. Sound ONLY
#                 for kept=1 rows: the loop commits model/ and THEN records
#                 HEAD, so a kept row's git_commit is its own commit.
#   unavailable   neither survives.
#
# `git_commit` is HEAD-at-logging-time, NOT "the commit holding this
# experiment's code". On a REVERTED row the loop ran `git checkout -- model/`
# and never committed, so HEAD is some unrelated commit -- experiment 2 of the
# mission era points at "overview: add 'The challenge, in one picture' hero",
# which touches no model file at all. Reading model/ out of that tree yields
# the CHAMPION's code wearing the reverted experiment's label: it renders
# perfectly and is a lie. Hence the kept=1 guard, and hence `unavailable` is a
# state we draw rather than a gap we paper over.
# --------------------------------------------------------------------------

SRC_FILES = ("model.py", "train.py")


def github_base():
    """Repo web URL for permalinks, from the origin remote. None if absent."""
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return None
    if not url:
        return None
    if url.startswith("git@"):                      # git@github.com:o/r.git
        url = "https://" + url[4:].replace(":", "/", 1)
    return url[:-4] if url.endswith(".git") else url


GITHUB = github_base()

_PY_KW = frozenset(keyword.kwlist) | frozenset(getattr(keyword, "softkwlist", []))
_PY_CONST = frozenset(("True", "False", "None", "self", "cls"))


def _span(cls, text):
    """Wrap text in a span, RE-OPENING it at every newline.

    A triple-quoted string or a long call spans lines; if its span crossed a
    line break the per-line gutter below would emit unbalanced HTML. Closing
    and reopening per line keeps every line independently well-formed.
    """
    parts = esc(text).split("\n")
    if not cls:
        return "\n".join(parts)
    return "\n".join(f"<span class='{cls}'>{p}</span>" if p else "" for p in parts)


def highlight_py(src):
    """Python -> highlighted HTML, using the stdlib tokenizer only.

    Deliberately no pygments: it happens to be importable in this venv but is
    not in requirements.txt, and a page generator that works only on the
    machine that grew a transitive dependency is not reproducible-from-nothing.
    On any tokenize error (truncated or non-UTF8 source) this falls back to
    escaped plain text rather than dropping the code.
    """
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return esc(src)
    # Absolute offsets, not row/col bookkeeping. NEWLINE and NL tokens carry
    # "\n" as their own text, so filling gaps line-by-line emits every line
    # break twice; slicing the original source between token spans cannot.
    starts, acc = [0], 0
    for ln in src.split("\n"):
        acc += len(ln) + 1
        starts.append(acc)
    at = lambda r, c: starts[r - 1] + c
    out, prev = [], 0
    for tt, tok, (srow, scol), (erow, ecol), _ in toks:
        s, e2 = at(srow, scol), at(erow, ecol)
        if s > prev:                                 # gap: indentation, blanks
            out.append(esc(src[prev:s]))
        if tt == tokenize.COMMENT:
            cls = "c"
        elif tt == tokenize.STRING:
            cls = "s"
        elif tt == tokenize.NUMBER:
            cls = "n"
        elif tt == tokenize.NAME:
            cls = "k" if tok in _PY_KW else "b" if tok in _PY_CONST else ""
        elif tt == tokenize.OP:
            cls = "o"
        else:
            cls = ""
        out.append(_span(cls, src[s:e2]))            # slice, not tok: identical
        prev = max(prev, e2)
    out.append(esc(src[prev:]))
    return "".join(out)


def numbered(html):
    """Line-number gutter. Safe because highlight_py never spans a newline."""
    return "".join(f"<span class='ln'></span>{ln}\n"
                   for ln in html.split("\n"))


def source_of(e):
    """(state, {filename: text}, permalink_or_None) for one experiment row.

    Accepts both the live `experiments` row and a `lineage_history` row --
    both carry artifacts_dir / git_commit / kept under those names.
    """
    commit = (e.get("git_commit") or "").strip()
    art = e.get("artifacts_dir")
    url = f"{GITHUB}/tree/{commit}/model" if (GITHUB and commit and e.get("kept")) else None

    if art:                                          # exact: the trained bytes
        d = REPO_ROOT / art / "model_src"
        got = {}
        for name in SRC_FILES:
            p = d / name
            if p.exists():
                try:
                    got[name] = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass
        if got:
            return "exact", got, url

    if commit and e.get("kept"):                     # reconstructed: kept only
        got = read_frozen_src(commit) or git_src(commit)
        if got:
            return "reconstructed", got, url

    return "unavailable", {}, None


FROZEN_SRC = REPO_ROOT / "site_assets" / "experiment_src"


def read_frozen_src(commit):
    """Recovered source, frozen into plain git by infra/freeze_experiment_src.py.

    CI must not need deep git history to render this page. `actions/checkout`
    defaults to fetch-depth 1, so `git show <old-commit>:model/model.py` fails
    there — and it fails SILENTLY, degrading all 16 reconstructed rows to
    "not retained" on the live site while they look right on the machine that
    rendered them. Cloning full history instead is not an option: .git is
    8.8 GB, in a job that has already run out of disk once.

    So the recovered text (147 KB total) is frozen to site_assets/ as ordinary
    git objects, exactly as the referenced images were after the LFS budget
    incident. Same lesson, same shape: publishing must not depend on fetching
    something large.
    """
    d = FROZEN_SRC / commit[:12]
    got = {}
    for name in SRC_FILES:
        p = d / name
        if p.exists():
            try:
                got[name] = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
    return got


def git_src(commit):
    """Read model/ out of a commit tree. Local dev and the freeze script only —
    returns {} on a shallow clone, which read_frozen_src() is there to cover."""
    got = {}
    for name in SRC_FILES:
        r = subprocess.run(["git", "show", f"{commit}:model/{name}"],
                           cwd=REPO_ROOT, capture_output=True, text=True)
        if r.returncode == 0 and r.stdout:
            got[name] = r.stdout
    return got


SRC_PROV = {
    "exact": "The exact bytes that trained this run, snapshotted before the "
             "keep/revert decision could touch them.",
    "reconstructed": "Read back out of this experiment&rsquo;s own commit. It was "
                     "kept, so the loop committed <code>model/</code> and then "
                     "recorded that commit &mdash; the tree is this "
                     "experiment&rsquo;s code.",
    "unavailable": "Source not retained. Until 2026-07-31 the loop discarded a "
                   "reverted experiment&rsquo;s code outright, and this row "
                   "predates that fix. Its design record, its verdict and its "
                   "exported model survive; the code does not. It is left blank "
                   "rather than filled with the parent&rsquo;s code, which is "
                   "what this row&rsquo;s recorded commit actually contains.",
}


def source_block(e):
    """The two agent-editable files, inline, collapsed, with provenance."""
    state, files, url = source_of(e)
    if state == "unavailable":
        return (f"<details class='trywrap srcwrap'><summary>The code for this "
                f"experiment &mdash; not retained</summary>"
                f"<div class='src-prov'>{SRC_PROV['unavailable']}</div></details>")
    link = (f" &middot; <a class='src-link' href='{esc(url)}' target='_blank' "
            f"rel='noopener'>on GitHub &#8599;</a>") if url else ""
    parts = [f"<div class='src-prov'>{SRC_PROV[state]}{link}</div>"]
    for name in SRC_FILES:
        if name not in files:
            continue
        body = files[name]
        parts.append(
            f"<div class='src-file'><div class='src-h'>model/{esc(name)}"
            f"<span class='src-lines'>{len(body.splitlines())} lines</span></div>"
            f"<pre class='src'><code>{numbered(highlight_py(body))}</code></pre></div>")
    label = "the exact code that trained it" if state == "exact" else "its code, from the commit"
    return (f"<details class='trywrap srcwrap'><summary>The code for this "
            f"experiment &mdash; {label}</summary>{''.join(parts)}</details>")


def prompt_block(e):
    """The exact prompt the headless agent was given for this experiment."""
    prompt = e.get("agent_prompt")
    if prompt:
        return (f"<details class='trywrap'><summary>The exact prompt given to the "
                f"headless agent</summary><pre class='prompt'>{esc(prompt)}</pre>"
                f"</details>")
    if e["kind"] == "holdout_check":
        return ""
    return ("<div class='provenance'>no headless prompt — designed interactively "
            "during the bootstrap session</div>")


def is_rejected(e):
    """A design the harness refused to even train — a demanded pivot that
    didn't hold up (see loop.sh's early- and post-implementation gates,
    2026-07-22) — distinct from a 'gated fail' that DID train but then hit
    a deployment constraint (ONNX size, latency). Both carry the same 1e9
    metric sentinel, so this is detected from the conclusion text loop.sh
    writes, not the metric."""
    return (e.get("conclusion") or "").startswith("REJECTED")


def status_of(e):
    if e["kind"] == "holdout_check":
        return "<span class='st-hold smcp'>holdout</span>"
    if is_rejected(e):
        return "<span class='st-rej smcp'>rejected</span>"
    if e["primary_metric"] is not None and e["primary_metric"] >= FAIL:
        return "<span class='st-fail smcp'>gated fail</span>"
    if e["kept"]:
        return "<span class='st-kept smcp'>kept</span>"
    return "<span class='st-disc smcp'>discarded</span>"


def _nums_from_conclusion(concl):
    """(metric, best_before) as strings, parsed from the harness-written
    conclusion, or (None, None). KEPT reads 'improved (BEST -> METRIC m)';
    REVERTED reads 'did not improve (METRIC m vs best BEST m)'."""
    concl = concl or ""
    m = re.search(r"improved \(([\d.eE+]+) -> ([\d.eE+]+)", concl)
    if m:
        return m.group(2), m.group(1)
    m = re.search(r"did not improve \(([\d.eE+]+) m vs best ([\d.eE+]+) m", concl)
    if m:
        return m.group(1), m.group(2)
    return None, None


def _m_str(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return "—"
    # These come from the harness-written conclusion, which quotes the PRIMARY
    # metric — the mission score, not a distance. (loop.sh's best_metric()
    # falls back to 1e18 before state/best.json exists; fmt_score maps that to
    # "no prior best" and FAIL=1e9 to "gated fail".)
    return fmt_score(v)


def _fail_detail(e):
    """Pin down WHY a 1e9 (worst-possible) experiment failed, from its
    metrics: a training/scoring crash (no per-area results at all) vs a
    real deployment-gate violation (trained, but over the ESP32-P4's size
    or latency budget — score.py's exact strings) vs the coverage floor
    (abstained on too many frames in some cell). Returns (short, long)."""
    try:
        metrics = json.loads(e.get("metrics_json") or "{}")
    except (ValueError, TypeError):
        metrics = {}
    areas = metrics.get("areas") or []
    if not areas:
        return ("training or scoring crashed — no result",
                "This design never produced a score at all: training or scoring failed outright "
                "(usually a bug in the proposed model or training code, or a model that couldn't be "
                "exported and scored). With no numbers to judge it on, it counts as a failure and "
                "was rolled back.")
    gate_fails = [a.get("gates", {}).get("failed") for a in areas
                  if a.get("gates", {}).get("failed")]
    if gate_fails:
        long_map = {"model too large": "the exported model is too big for the ESP32-P4's memory (>4 MiB)",
                    "missing model": "no usable model file was produced to load",
                    "latency over proxy budget": "a single inference is slower than the flight-time budget (>250 ms proxy)"}
        short_map = {"model too large": "too big for the ESP32-P4",
                     "missing model": "no model produced",
                     "latency over proxy budget": "too slow for the latency budget"}
        longs = "; ".join(dict.fromkeys(long_map.get(f, f) for f in gate_fails))
        shorts = "; ".join(dict.fromkeys(short_map.get(f, f) for f in gate_fails))
        return (f"over deployment budget: {shorts}",
                f"The model trained and scored, but it broke a hard aircraft limit: {longs}. A model "
                "that can't physically run on the UAV is failed regardless of how accurate it is, so "
                "it was rolled back.")
    return ("abstained on too many frames in a cell",
            "The model trained, but in at least one area × lighting combination it refused to answer on "
            "more than the allowed share of frames — and a model isn't allowed to pass by abstaining its "
            "way out of the hard cases. That cell is scored as a failure, which fails the whole "
            "experiment, so it was rolled back.")


def status_reason(e):
    """Plain-language ELI5 of WHY this experiment ended in its status —
    derived entirely from the harness-written conclusion / result / metrics
    (no LLM, no hand-authoring). Returns (kind, short, long): a status kind
    for colour, a terse phrase for the log row, and a paragraph for the
    detail view. The 'juicy' statuses (rejected / gated fail / discarded)
    get the fuller explanation; kept/holdout are brief by nature."""
    concl = e.get("conclusion") or ""
    result = e.get("result") or ""
    metric = e.get("primary_metric")

    if e["kind"] == "holdout_check":
        return ("hold", "blind Hamburg generalisation check",
                "Not a competing design — this is the periodic blind holdout. The current "
                "champion's training recipe is run on Hamburg, a city the loop never tunes "
                f"against, purely to check the method still generalises. It scored {fmt_score(metric)}; "
                "the number is logged for monitoring only and never affects which experiments "
                "are kept or reverted.")

    if is_rejected(e):
        if "REJECTED IN ERROR" in concl or "harness bug" in result.lower():
            return ("rej", "rejected by a harness bug — since fixed",
                    "Thrown out by a bug in the pivot-checker, not on its merits: it did propose "
                    "the required from-scratch rethink, but a gate misread the file header and "
                    "rejected it anyway. The bug was fixed afterwards. This run was never trained "
                    "or scored, so it counts as still-open rather than a real failure.")
        if "before implementation" in concl or "complete rethink" in concl:
            m = re.search(r"unchanged in the design:\s*(.+?)\s*$", result)
            tail = f" (untouched: {m.group(1).strip()})" if m else ""
            return ("rej", "pivot not honored — stages left unchanged",
                    "After a losing streak the loop demanded a *complete* architecture rethink — "
                    "every moving part of the design had to change. This proposal left parts of the "
                    f"champion untouched{tail}, so it was rejected before spending any GPU time. A "
                    "partial tweak doesn't count as a pivot.")
        m = re.search(r"champion's backbone \(([^)]+)\)", result) \
            or re.search(r"champion's backbone \(([^)]+)\)", concl)
        tail = f" ({m.group(1)})" if m else ""
        return ("rej", "pivot not honored — kept champion backbone",
                "The loop demanded a full pivot away from the champion's backbone, but the code that "
                f"got built still used it{tail}. The checker reads the actual source, not the agent's "
                "claim, so it was rejected before any training — re-tuning, truncating, or wrapping the "
                "same trunk doesn't count as a pivot.")

    if metric is not None and metric >= FAIL:
        short, long = _fail_detail(e)
        return ("fail", short, long)

    metric_s, best_s = _nums_from_conclusion(concl)
    if e["kept"]:
        got = _m_str(metric_s) if metric_s else fmt_score(metric)
        extra = ""
        if metric_s and best_s:
            try:
                d = float(best_s) - float(metric_s)
                if 0 < d < 1e14:
                    extra = f" — {d:.3f} better"
            except (ValueError, ZeroDivisionError):
                pass
        first = (best_s is None) or (_m_str(best_s) == "no prior best")
        if first:
            return ("kept", f"starting line: mission score {got}",
                    f"The first scoreable run of this lineage, so there was nothing to "
                    f"beat — it sets the mission score ({got}) that every later "
                    f"experiment must improve on. Its code is committed as the "
                    f"starting point later experiments branch from.")
        return ("kept", f"new best: {_m_str(best_s)} → {got}",
                f"The best design so far. It improved the mission score from "
                f"{_m_str(best_s)} to {got}{extra}, beating the previous champion — so "
                "its code was committed as the new best, and later experiments branch "
                "from here.")

    return ("disc", f"worse than champion: {fmt_score(metric)} vs {_m_str(best_s)}",
            f"The change trained and scored cleanly, but at {fmt_score(metric)} it didn't beat the champion's "
            f"{_m_str(best_s)}, so the loop discarded it and kept the previous best. A normal negative "
            "result — most experiments land here, and each still narrows down what doesn't work.")


HELP = f"""
<details class="help"><summary>What do the columns and marks mean?</summary>
<dl class="help-grid">
<dt>Mission score</dt><dd>The single number the loop optimizes, and
deliberately <b>the product requirement rather than a statistic about
errors</b>. The aircraft takes a vision fix every 5&ndash;10&nbsp;s and it is
its only drift correction, so per frame exactly three things can happen:
it is <b>confident and within {TARGET_M:.0f}&nbsp;m</b> (a <b>usable fix</b> &mdash; the
product), <b>not confident</b> (it abstains &mdash; safe, it waits), or
<b>confident and wrong</b> (a <b>false fix</b> &mdash; dangerous, it feeds a
wrong position into navigation). The score is
<b>(1 &minus; usable-fix rate) + false-fix rate</b>: <b>0</b> means every frame
is a usable fix, <b>1.0</b> means it abstains on everything, <b>2.0</b> means it
is confidently wrong on everything &mdash; strictly worse than silence, which is
the point. Error statistics (median, geometric mean, p10, p25) are all still
recorded, but none is optimized: each was tried as the target and each rewarded
something the aircraft does not want.</dd>
<dt>Region-holdout (diagnostic)</dt><dd>A small 1-in-32-block slice of Berlin
is kept genuinely out of training and scored separately. It is <b>logged and
never optimized</b> — it cannot move keep/revert. Its only job is to show
whether the model has real spatial structure or is just a lookup table over
memorized vantages, so expect it to read far worse than the headline number.
That gap is information, not failure.</dd>
<dt>gated fail (×)</dt><dd>The §6 score also enforces the aircraft's hard
limits: the exported model must fit the ESP32-P4 flight computer
(<b>≤ 4 MiB</b>) and answer within the latency budget (≤ 250 ms host proxy),
and it may not dodge hard cases by refusing to answer (a cell where it
abstains on &gt; 80% of frames counts as failed). Any violation scores the
whole experiment as failed regardless of accuracy.</dd>
<dt>rejected</dt><dd>Never trained at all. After a losing streak, the loop
demands a genuine pivot — every non-frozen part of the design must change,
not just the one piece that's gone stalest. If the proposed design doesn't
clear that bar (checked against the actual code diff, not just what the
agent claims it changed), it's rejected before spending any GPU time on
it.</dd>
<dt>cov (coverage)</dt><dd>Share of test frames the model was confident
enough to answer at all. Abstaining honestly on bad frames is allowed —
down to the 20% floor above.</dd>
<dt>Kept / Discarded</dt><dd>Karpathy-style loop discipline: branch from the
best code, run one focused experiment, <b>keep</b> the change (git commit)
only if the worst-case error improves — otherwise <b>revert</b> it. The
step line in the chart is the running best.</dd>
<dt>Eval-set reset (┆)</dt><dd>During the bootstrap phase the frozen
evaluation data itself was revised twice (10 m satellite → 1 m orthophotos;
then a split rebalance). Scores on different eval sets are measurements on
different rulers and must not be compared — the dashed vertical rule marks
the break, and the running-best line restarts there instead of pretending
continuity. From Phase 2 on, the eval set does not change.</dd>
<dt>Holdout (○)</dt><dd>Hamburg is the blind fifth area: structurally
different (port, river, spread-out), never seen by the loop, scored only as
a periodic read-only check. If its error diverges from the four development
areas, the pipeline has learned their quirks rather than a general method.
Its result never influences keep/revert.</dd>
<dt>Pivot-directed (▽)</dt><dd>After <b>4 consecutive</b> experiments in a
row fail to beat the running best, the harness injects a mandatory pivot
preamble into the next design prompt: do not refine the champion's current
mechanism again — propose from a design family absent from the recent
history. Marked ones ran under that directive; the streak resets every time
an experiment is kept.</dd>
<dt>Cost</dt><dd>Estimated cloud-GPU spend for that one experiment: its
measured wall time × a <b>$0.69/hr</b> rate, billed continuously by the clock
(not per phase), so it covers the whole experiment — design, implementation,
training, scoring, publishing — not GPU time alone. Shown only for the middle,
rented-GPU window; the bootstrap and later local eras run at ~$0 marginal
compute and show —.</dd>
<dt>Category</dt><dd>Which lever the experiment pulls: architecture, loss,
augmentation, relighting, training procedure, or quantization.</dd>
<dt>Where the deployment numbers went</dt><dd>Exported model size, weight
initialization and single-frame latency each used to have a column. All three
were dropped: nearly every row read the same, and the size and latency gates
are already folded into the mission score — a model that misses either scores
as a gated fail, so the column was restating the Status column. All three
remain in the expanded record, and init also on the
<a href="inference-paths.html">model designs</a> page.</dd>
<dt>Training set</dt><dd>Each area offers ~45,000 distinct training
positions (1 m/px, 128 m crops, times random rotation); how many crops an
experiment actually samples per lighting condition is its own choice and is
shown in the detail view. Training crops never overlap the eval blocks —
enforced by the frozen split, not by convention.</dd>
<dt>Agent prompt</dt><dd>Every loop experiment records the exact prompt its
headless agent received — expandable in the detail view, so any experiment
can be re-run or audited later.</dd>
<dt>In plain words</dt><dd>Each experiment's own jargon-free explanation of
what it tried, pre-registered alongside the technical design — read this
first if the title looks like alphabet soup.</dd>
<dt>One real test</dt><dd>The figure at the top of each detail view is not
an illustration: the same held-out Berlin viewpoint is fed to that
experiment's actual exported model, and the red glow on the map is the
probability field recovered from the deployed ONNX artifact itself —
with the true location (○), the model's answer (×), and the real miss
distance. Because every experiment sees the identical crop, any difference
between two figures is the mechanism change, not the example. The pipeline
sentence beneath names the stages; <b style="color:var(--accent)">red</b>
marks what that experiment changed (a “+” stage is training-only and never
flies).</dd>
</dl>
<div class="foot">Click any row — or any point in the chart — to see the
experiment's pre-registered hypothesis, method and expected outcome, the
measured result, the per-area × lighting scoreboard, and what the model
actually looked at.</div>
</details>"""


# License attribution for imagery-derived figures — required by the data
# sources' terms on any published page (dl-de/by-2-0, CC BY 4.0, Copernicus).
CREDITS_TMPL = """<footer style="margin:48px auto 24px;max-width:960px;color:#9b998c;
font-size:12.5px;line-height:1.5;border-top:1px solid #e6e4da;padding-top:10px">
{finished_note}Imagery-derived figures are based on open geodata:
© GeoBasis-DE/LGB (dl-de/by-2-0) · © Bayerische Vermessungsverwaltung (CC BY 4.0)
· © HVBG Hessen (dl-de/by-2-0) · © Freie und Hansestadt Hamburg, LGV (dl-de/by-2-0)
· Contains modified Copernicus Sentinel data.
Code: <a href="https://github.com/akaalias/ballpark/blob/main/LICENSE"
style="color:inherit">GPL-3.0-or-later</a> — free software; improved forks must
stay free —
<a href="https://github.com/akaalias/ballpark"
style="color:inherit">source repository</a>.</footer>
<a id="gh-ribbon" href="https://github.com/akaalias/ballpark"
target="_blank" rel="noopener">view on GitHub</a>"""


def credits_html():
    """Footer, with the concluded-research line spliced in when the research
    is finished. Kept as a template + function rather than a constant because
    the note depends on state/research_status, which changes without a code
    change."""
    return CREDITS_TMPL.replace("{finished_note}", finished_note())


PATHS_OUT = REPO_ROOT / "gallery" / "inference-paths.html"
OVERVIEW_OUT = REPO_ROOT / "index.html"
NOTEBOOK_OUT = REPO_ROOT / "gallery" / "lab-notebook.html"

# Hand-authored, not DB-derived: a condensed, dated narrative of what
# happened and why, compiled once from git history + Claude Code session
# transcripts (20-23 July 2026). Static content, not regenerated from
# experiments.sqlite — append future days by hand as the project continues.
NOTEBOOK_HTML = """
<section class="nb-day">
<h2>20 July 2026</h2>
<p class="daysub">Bootstrap &mdash; the pipeline, the harness, and the first real experiments</p>

<div class="nb-row"><span class="nb-time">19:05&ndash;19:23</span>
<p class="nb-text"><b>The repo is bootstrapped end-to-end in one
session.</b> Against the project spec, the frozen bbox-generic data
pipeline, a naive baseline model, SQLite lineage, an HTML gallery, and
<code>loop.sh</code> are scaffolded in one pass and proven with a real
train&rarr;score&rarr;log run, including confirming the Hamburg holdout
stays untouched.</p></div>

<div class="nb-row"><span class="nb-time">19:26&ndash;20:02</span>
<p class="nb-text"><b>Two open questions get resolved before more code is
written.</b> The deployment hardware: an ESP32-P4-Function-EV-Board paired
with the SmartSens SC2336 sensor turns out to be Espressif&rsquo;s own
documented reference pairing. The reference imagery: 10&nbsp;m/px
Sentinel-2 is swapped for 1&nbsp;m/px open state orthophotos, matching a
real ~100&nbsp;m-altitude camera footprint, and the relighting pipeline is
rebuilt for the larger rasters. A v2 baseline scores 2765.67&nbsp;m.</p></div>

<div class="nb-row"><span class="nb-time">20:06&ndash;21:26</span>
<p class="nb-text"><b>The dashboard is rebuilt in the project&rsquo;s house
style.</b> It&rsquo;s restyled to the same Tufte research-log language used
across earlier projects, then pushed through three more rounds of feedback
toward plain-language framing and precise miss-distance wording.
<code>loop.sh</code> is hardened for safe interruption &mdash; a clean stop
on Ctrl-C or a typed <code>exit</code> &mdash; and the train/eval split is
rebalanced to 360&nbsp;m blocks after the original 144&nbsp;m blocks were
starving some areas of eval crops.</p></div>

<div class="nb-pull"><span class="nb-time">21:51&ndash;22:41</span>
<div class="nb-quote"><p><b>A clean-slate reset breaks the project&rsquo;s
first real plateau.</b> The lineage resets to a clean baseline, and the
first post-reset experiment breaks the ~3.2&nbsp;km &ldquo;predicts the map
center&rdquo; floor that nine prior designs had all converged on &mdash; a
soft-argmax probability field, kept at 2489&nbsp;m. A concrete plateau rule
follows: after three consecutive reverts, a design must leave the refuted
mechanism family entirely.</p></div></div>

<div class="nb-row"><span class="nb-time">22:43&ndash;23:26</span>
<p class="nb-text"><b>Every experiment now gets its own hand-drawn
architecture diagram.</b> The design agent begins drawing one per
experiment, under a visual contract distinguishing frozen, current, and
just-changed stages. Experiment&nbsp;8 pivots to a residual encoder and is
kept at 2326.54&nbsp;m, closing out the day.</p></div>
</section>

<section class="nb-day">
<h2>21 July 2026</h2>
<p class="daysub">Infrastructure moves to a rented GPU; the public site takes shape</p>

<div class="nb-row"><span class="nb-time">22:15&ndash;07:56</span>
<p class="nb-text"><b>An overnight diagnosis finds a real blind spot in the
model&rsquo;s field head.</b> It has only ever seen a spatially-averaged
descriptor with no positional information, and the loop tests a
layout-aware fix; a pretrained MobileNetV3-Small backbone is explored in
parallel, resolving the from-scratch-vs-pretrained question directly.
Progress is slow &mdash; the loop repeatedly hits a connection error and
restarts the same iteration from scratch several times before anything
lands.</p></div>

<div class="nb-pull"><span class="nb-time">11:27&ndash;11:38</span>
<div class="nb-quote"><p><b>The pod&rsquo;s database becomes the one
production record.</b> Now that compute costs money on a rented RunPod
RTX&nbsp;4090, the policy is set explicitly: the laptop only ever pulls
from it. The rule is encoded in the sync scripts themselves, not left as
convention.</p></div></div>

<div class="nb-row"><span class="nb-time">08:32&ndash;09:02</span>
<p class="nb-text"><b>A new gallery page ships for every proposed
architecture.</b> &ldquo;Inference Paths&rdquo; is matched to the
typography of the researcher&rsquo;s existing personal site, along with a
frozen-contract figure showing the fixed camera-frame input and output
every design shares, its coordinates anchored so every architecture
diagram scroll-compares cleanly against it.</p></div>

<div class="nb-row"><span class="nb-time">11:32&ndash;14:50</span>
<p class="nb-text"><b>Four experiments land in a row, each a real
improvement.</b> 1856&nbsp;m &rarr; 1638&nbsp;m &rarr; 1369&nbsp;m &rarr;
1095&nbsp;m &rarr; 1055&nbsp;m.<svg class="nb-inline-spark" width="64" height="15" viewBox="0 0 64 15"><polyline points="2.0,13.0 17.5,10.2 33.0,7.3 48.5,4.5 62.0,2.0" fill="none" stroke="var(--ink)" stroke-width="1.1"/><circle cx="2.0" cy="13.0" r="1.6" fill="var(--ink)"/><circle cx="17.5" cy="10.2" r="1.6" fill="var(--ink)"/><circle cx="33.0" cy="7.3" r="1.6" fill="var(--ink)"/><circle cx="48.5" cy="4.5" r="1.6" fill="var(--ink)"/><circle cx="62.0" cy="2.0" r="1.6" fill="var(--ink)"/></svg> The loop is also restructured around two agent roles &mdash; one
designs an experiment, a second implements it &mdash; with per-phase timing
logged for each.</p></div>

<div class="nb-row"><span class="nb-time">11:51&ndash;12:15</span>
<p class="nb-text"><b>The project goes public.</b> The site and a matching
entry on the researcher&rsquo;s personal homepage ship together, with a
tagline chosen after two rounds of brainstorming, and an airloom-style
&ldquo;view on GitHub&rdquo; ribbon added to every page.</p></div>
</section>

<section class="nb-day">
<h2>22 July 2026</h2>
<p class="daysub">A plateau-enforcement mechanism that took two real fixes to actually hold</p>

<div class="nb-row"><span class="nb-time">00:51&ndash;01:14</span>
<p class="nb-text"><b>An overnight disk leak crashes the loop.</b> A
training run has been silently caching several gigabytes of rendered
images per experiment into a directory the loop was committing, filling
the pod&rsquo;s disk. Fixed by excluding scratch renders from persistence,
and the loop is relaunched.</p></div>

<div class="nb-row"><span class="nb-time">02:07&ndash;12:15</span>
<p class="nb-text"><b>The pivot check itself turns out to be broken.</b> The
loop runs through the night with no net progress, and by midday the pivot
check turns out to be testing the wrong signal entirely &mdash; a
declared-category heuristic rather than whether the frozen architecture
actually changed &mdash; so seven consecutive &ldquo;pivot&rdquo;
experiments have in fact left the same pretrained backbone untouched.</p></div>

<div class="nb-pull"><span class="nb-time">16:26&ndash;20:19</span>
<div class="nb-quote"><p><b>Two real bugs, not the prompt wording, were
blocking every pivot.</b> A first fix &mdash; a diagnostic naming which
stages hadn&rsquo;t changed &mdash; proves insufficient: its patience
threshold turns out to be hardcoded inside the checker script, ignoring the
loop&rsquo;s actual setting, so the intended stricter check silently
doesn&rsquo;t apply at the moment it&rsquo;s needed. A second, code-level
gate replaces it, checking the real backbone-construction diff before
training runs at all. Both root causes are found and fixed together; this
is the version that holds.</p></div></div>

<div class="nb-row"><span class="nb-time">14:59&ndash;15:54</span>
<p class="nb-text"><b>Monitoring and budget expectations get set
explicitly.</b> An hourly automated health check posts to a GitHub issue
only when something is actually broken. Budget context is set: roughly 100
experiments is the real ceiling, and the loop should not self-stop at the
target metric &mdash; only when budget runs out or the work is stopped by
hand.</p></div>

<div class="nb-row"><span class="nb-time">15:11&ndash;19:11</span>
<p class="nb-text"><b>Two infrastructure incidents get resolved in one
day.</b> A disk-exhaustion failure in the build pipeline, and a
git-history divergence between the pod and GitHub after both independently
applied the same fix. Policy tightens: every repository change now happens
on the pod exclusively; the laptop becomes read-only.</p></div>
</section>

<section class="nb-day">
<h2>23 July 2026</h2>
<p class="daysub">Nineteen reverted pivots, a forced redirect, and compute moves off the rented pod</p>

<div class="nb-row"><span class="nb-time">08:06&ndash;13:24</span>
<p class="nb-text"><b>A ten-hour GitHub outage gets root-caused and
fixed.</b> An oversized quarantine folder of rejected renders had been
accidentally committed; excluding the path structurally fixes it for good.
Loop vocabulary is also overhauled &mdash; &ldquo;iteration&rdquo; becomes
&ldquo;experiment&rdquo; throughout &mdash; and every experiment row gains
a deterministic, plain-language reason for its outcome.</p></div>

<div class="nb-row"><span class="nb-time">16:45&ndash;17:22</span>
<p class="nb-text"><b>The comparison baseline resets to a faster, fairer
standard.</b> The champion architecture, retrained on Berlin alone at a
fast budget, scores 1176.5&nbsp;m, retiring the older, heavier four-area
743&nbsp;m champion as historical so quick local experiments are judged
fairly. RunPod branding is dropped from the site as compute moves back to
local hardware.</p></div>

<div class="nb-pull"><span class="nb-time">18:21&ndash;19:04</span>
<div class="nb-quote"><p><b>Four forced pivots in a row all fail the same
way.</b> Each is required to avoid the champion&rsquo;s banned backbone
entirely &mdash; a residual VQ location codebook, a Gaussian-mixture
regression, a pretrained ShuffleNetV2 trunk, a memory-safe
learned-relighting redesign &mdash; all reverted. Across nineteen straight
reverted pivots, one pattern holds: decode-mechanism rewrites keep losing,
while the one round that changed the trunk came closest. The encoder, not
the decode head, is the real bottleneck.</p></div></div>

<div class="nb-row"><span class="nb-time">18:36&ndash;19:29</span>
<p class="nb-text"><b>A human-forced redirect finally lands a genuinely new
architecture.</b> A from-scratch dense scene-coordinate regression design
is forced directly into the next experiment, and a one-shot
forced-direction mechanism is added to the loop so a specific redirect can
be handed to a single run without further intervention. It scores
2360&nbsp;m &mdash; architecturally novel, still reverted.</p></div>

<div class="nb-row"><span class="nb-time">19:16&ndash;20:30</span>
<p class="nb-text"><b>Compute strategy pivots twice in one evening.</b> Off
the rented GPU pod back to local hardware, then toward a scale-to-zero
serverless GPU provider alongside a self-hosted always-on worker option
&mdash; removing the recurring overhead of a persistent rented pod.</p></div>
</section>

<section class="nb-day">
<h2>30 July 2026</h2>
<p class="daysub">A four-experiment losing streak, still short of the 743&nbsp;m champion</p>

<div class="nb-row"><span class="nb-time">19:26&ndash;19:41</span>
<p class="nb-text"><b>Figure-checking gets stricter.</b> Mechanical checks
for the architecture-diagram style contract land &mdash; palette, font,
stroke-width, gradients/icons/emoji, red-consistency &mdash; then the SVG
arcs get flattened so the line-crosses-label check can actually see curved
elements.</p></div>

<div class="nb-row"><span class="nb-time">17:15&ndash;20:42</span>
<p class="nb-text"><b>Experiment&nbsp;60</b> retries exp&nbsp;38&rsquo;s
illumination-invariant channel with a from-scratch trunk and a
cross-lighting consistency loss. 963.42&nbsp;m &mdash; closer than most
recent attempts, still reverted against the 743.07&nbsp;m champion.</p></div>

<div class="nb-pull"><span class="nb-time">18:42&ndash;21:53</span>
<div class="nb-quote"><p><b>The forced-pivot rule fires twice more, both
gated failures.</b> A quantization-freed depthwise-separable trunk with a
unified single-head field (experiment&nbsp;61) fails outright. Patience
spent again, the champion&rsquo;s <code>mobilenet_v3_small</code> trunk is
banned outright and a Fire-module SqueezeNet1.1 + FiLM-conditioned field
(experiment&nbsp;62) is forced under that ban &mdash; also a gated
failure.</p></div></div>

<div class="nb-row"><span class="nb-time">21:57</span>
<p class="nb-text"><b>A real training-speed bug gets fixed mid-streak.</b>
Bucket/realization images were being re-decoded from disk every epoch
&mdash; profiling shows this was ~78% of per-area wall-clock. Caching the
decode once per process is the whole fix.</p></div>

<div class="nb-row"><span class="nb-time">19:53&ndash;22:58</span>
<p class="nb-text"><b>Experiment&nbsp;63</b> &mdash; another forced pivot
under the mobilenet ban, this time a from-scratch trunk with
domain-contrastive pretraining, a fixed retinex channel, D4 symmetry, and
top-K-masked decode &mdash; scores 1074.76&nbsp;m. Four straight reverts;
the day closes with the same 743.07&nbsp;m champion still standing.</p></div>
</section>

<section class="nb-day">
<h2>31 July 2026</h2>
<p class="daysub">Three measurement bugs found in one day — the evaluation, the score, and what the harness kept — and everything measured before today invalidated</p>

<div class="nb-row"><span class="nb-time">morning</span>
<p class="nb-text"><b>The human proposes deliberately narrowing scope.</b>
After weeks of ~50&ndash;65-minute iterations optimizing a worst-case
across 4&nbsp;areas &times; 6&nbsp;lighting buckets, still far from the
20&nbsp;m target, the idea: one locale (Berlin), daytime imagery only,
treat &ldquo;overfitting to one place&rdquo; as a feature rather than a
failure mode, in exchange for much faster rounds &mdash; and re-derive a
fair accuracy milestone from what the actual image resolution can support,
rather than assuming the original number still applies.</p></div>

<div class="nb-row"><span class="nb-time">morning</span>
<p class="nb-text"><b>The plan gets grounded in real numbers before any
code changes.</b> The champion&rsquo;s own Berlin-only daytime score
(~650&ndash;720&nbsp;m) sets a realistic near-term milestone at
100&nbsp;m, not the original 20&nbsp;m (~35&times; off). Per-experiment
timing data shows the real speed levers aren&rsquo;t what they first
seem: the agent-drawn architecture figure (~15&ndash;20&nbsp;min, the
single largest non-training phase) and the six lighting buckets
concatenated into one training set are the actual costs &mdash; dropping
to one area is not, since areas already train in parallel.</p></div>

<div class="nb-pull"><span class="nb-time">morning</span>
<div class="nb-quote"><p><b>The berlin-slim branch is created and the whole
slice implemented in one pass.</b> Relighting collapses to a single as-is
daytime bucket (no synthetic ambient/gain/noise simulation), areas scope
to Berlin only, and the pivot-gate, holdout check, and figure-drawing step
are disabled as reversible one-line toggles, not deletions. The target
milestone moves to 100&nbsp;m everywhere it&rsquo;s referenced in the
gallery. Design and implementation are reassigned to Fable and Opus&nbsp;5
&mdash; a deliberate reversal of both models&rsquo; standing bans, flagged
before proceeding rather than silently applied.</p></div></div>

<div class="nb-row"><span class="nb-time">morning</span>
<p class="nb-text"><b>Two smoke tests catch real bugs before a live
run.</b> A hardcoded <code>midday.png</code> heatmap background (missed by
a substring grep) and a training-time realization generator that
would&rsquo;ve kept applying synthetic sensor noise despite relighting
being &ldquo;disabled&rdquo; both get fixed and reverified.</p></div>

<div class="nb-row"><span class="nb-time">morning</span>
<p class="nb-text"><b>A clean restart, at explicit request.</b> The
champion architecture &mdash; shaped by 63 rounds against the old
4-area/6-bucket problem, including machinery like a dark-lighting expert
head that no longer applies &mdash; is retired in favor of the original
from-scratch <code>TinyLocNet</code> baseline. The full experiment lineage
is wiped, not just the comparison metric, so berlin-slim&rsquo;s research
trail starts at Experiment&nbsp;1 with nothing to live up to; the old
62-row history is archived, not deleted.</p></div>

<div class="nb-row"><span class="nb-time">11:20</span>
<p class="nb-text"><b>A question about one roundabout unravels the whole
evaluation.</b> Looking at Berlin&rsquo;s error map, the human asks why the
Gro&szlig;er Stern &mdash; a seven-way star roundabout that exists exactly
once in the city, about as visually unmistakable as a landmark gets &mdash;
is surrounded entirely by red dots. If anything should be memorable, that
should be. Checking the frozen split directly: <b>zero</b> training crops
contain that roundabout, and <b>twelve</b> evaluation questions do. It sits
three pixels from the edge of its 360&nbsp;m block, and the block next door
is an eval block &mdash; so the anti-leakage buffer had deleted every single
training view of it.</p></div>

<div class="nb-pull"><span class="nb-time">11:20&ndash;11:40</span>
<div class="nb-quote"><p><b>The evaluation had been asking the model to
locate places it was never shown.</b> The one-in-five block holdout, plus a
92&nbsp;px no-leakage buffer around every eval block, means <b>28.2%</b> of
Berlin never appears inside any training crop &mdash; and <b>100%</b> of the
15,585 eval questions are centred on ground the model has never seen. Worse,
for <b>65%</b> of them the entire 128&nbsp;m frame contains not one familiar
pixel; the median eval frame is 0% familiar ground. Two independent
derivations agree: from the geometry alone, a train crop&rsquo;s content
stops 28&nbsp;px short of any eval block, so only eval centres within
36&nbsp;px of a block edge can see anything familiar at all &mdash;
predicting 64.0% blind, against 65.1% measured. This is the likely reason
roughly sixty experiments plateaued between 700 and 1000&nbsp;m: for most of
the test set the question is not hard, it is unanswerable.</p></div></div>

<div class="nb-row"><span class="nb-time">11:40</span>
<p class="nb-text"><b>The mismatch, stated plainly.</b> This branch was
founded on treating overfitting as a feature &mdash; build a visual memory
of one city. But the metric was built to defeat memorisation: withhold
regions, then grade the model on those regions. The architecture is a memory
system; the ruler was grading it as an extrapolator. It is also stricter
than the prior art the spec cites: DSAC and ACE evaluate on held-out camera
trajectories through a <i>fully mapped</i> scene, not on unmapped parts of
it. And in deployment the aircraft only ever flies over the box the model
was trained on, so the metric measured a capability the product never needs
while never measuring the one it does. The loop is stopped mid-experiment;
the split is being redesigned to hold out <i>viewpoints</i> rather than
<i>regions</i>.</p></div>

<div class="nb-row"><span class="nb-time">11:45&ndash;12:20</span>
<p class="nb-text"><b>The split is rebuilt around viewpoints.</b> Training
now covers every lattice position of the whole city &mdash; the model is
supposed to memorise its one box, so it is shown all of it. Evaluation moves
off-lattice: each test frame sits 11&ndash;17&nbsp;m from the nearest
training vantage (17&nbsp;m being the largest offset a 24&nbsp;m lattice
permits) and carries its own rotation. Mapped ground, unseen view &mdash;
which is what an aircraft over a mapped area actually faces. A small
one-in-thirty-two-block region stays genuinely untrained as a logged-only
diagnostic, never allowed to touch the metric, to catch a model that is a
lookup table with no spatial structure.</p></div>

<div class="nb-row"><span class="nb-time">12:20</span>
<p class="nb-text"><b>Measured again, on the same instruments.</b> Berlin
ever shown in training rises from 71.8% to <b>93.5%</b> (the remaining gap
is the deliberate diagnostic region and its buffer). Eval questions centred
on ground the model has seen: 0% &rarr; <b>100%</b>. Eval frames containing
no familiar pixel at all: 65.1% &rarr; <b>0%</b>. Distance from each test
frame to the nearest training vantage: 11.3&ndash;16.3&nbsp;m. The lineage
resets a second time in one day &mdash; scores from the old ruler cannot be
compared to the new one &mdash; and the five experiments run that morning
are archived rather than deleted.</p></div>

<div class="nb-pull"><span class="nb-time">13:10&ndash;13:40</span>
<div class="nb-quote"><p><b>A second measurement bug, found the same way as the
first &mdash; by looking at the map.</b> The human asks how an experiment whose
error map is dotted with green hits can be discarded, while one with a single
green dot in five hundred is kept. It is not the map that is wrong. A model
that guesses near the map centre produces a tight, mediocre, unimodal error
distribution &mdash; a decent <i>median</i> and no good tail. A model that
genuinely memorises places produces a <b>bimodal</b> one: some spots nailed,
the rest badly wrong when it misidentifies. The median rewards the guesser and
punishes the learner. Experiment&nbsp;2 located <b>8%</b> of Berlin to within
100&nbsp;m &mdash; the baseline managed 0.0%, experiment&nbsp;3 managed 0.2%
&mdash; and was reverted for a slightly worse median, while experiment&nbsp;3,
a marginally better centre-guesser, was kept. Tested against every candidate
statistic, the median was the <i>only</i> one that ranked experiment&nbsp;2
last; p25, p10, geometric mean and hit-rate all ranked it first.</p></div></div>

<div class="nb-row"><span class="nb-time">13:40</span>
<p class="nb-text"><b>The score becomes the geometric mean.</b> Chosen over a
percentile because it reads every frame rather than one cutoff, and because on
an error spanning two orders of magnitude the log scale is the natural one:
pulling a single spot from 2&nbsp;km to 50&nbsp;m &mdash; actually memorising
it &mdash; now moves the number meaningfully. Median, mean, p10, p25 and a
product-facing hit-rate are all still recorded, just no longer optimised. This
time no lineage reset was needed: the evaluation data never changed, only its
summary, so all three experiments were rescored from their saved weights.
Corrected, the history inverts &mdash; experiment&nbsp;2 becomes the champion
at 1323.6&nbsp;m, experiment&nbsp;3 the revert.</p></div>

<div class="nb-row"><span class="nb-time">13:40</span>
<p class="nb-text"><b>And the correction exposes a hole in the harness.</b>
Experiment&nbsp;2 was the best implementation the project had &mdash; and it no
longer existed. Reverting an experiment ran <code>git checkout -- model/</code>,
which deleted its source outright; only kept experiments ever entered the git
trail. That is safe only if the scoring metric is correct and permanent, an
assumption nobody had written down and which had just been proven false twice
in one day. Of the 60 archived experiments, the 13 kept ones survive in git;
the 47 reverted ones do not, though their design briefs and trained weights
remain, which is how experiment&nbsp;2 was rebuilt. Four separate code paths
could destroy an implementation; all four now snapshot it first.</p></div>

<div class="nb-pull"><span class="nb-time">14:30&ndash;15:10</span>
<div class="nb-quote"><p><b>&ldquo;There is ONLY the product requirement.&rdquo;</b>
Pressed on why the score was a statistic about errors at all, the answer does
not survive contact: the geometric mean had been chosen to make research
progress <i>visible</i>, which is the same species of mistake as the two bugs
already found today &mdash; optimising something adjacent to the goal instead
of the goal. So the metric becomes the goal. The aircraft takes a vision fix
every 5&ndash;10&nbsp;s and it is its only drift correction, so per frame
exactly three things can happen: <b>confident and within 100&nbsp;m</b> is a
<b>usable fix</b>; <b>not confident</b> is an abstention, which is safe because
it simply waits; <b>confident and wrong</b> is a <b>false fix</b>, which feeds
a wrong position into navigation and is worse than saying nothing.
<b>mission score = (1 &minus; usable-fix rate) + false-fix rate</b>: zero is
perfect, 1.0 is abstaining on everything, 2.0 is being confidently wrong on
everything. §6 had asserted since day one that honest abstention beats
confident guessing; no metric until now actually encoded it.</p></div></div>

<div class="nb-row"><span class="nb-time">15:10</span>
<p class="nb-text"><b>The baseline&rsquo;s new score is the argument for the
change.</b> The same naive model that the old rulers reported as
&ldquo;1993&nbsp;m median&rdquo; &mdash; a number that sounds like a
measurement &mdash; scores <b>2.001</b>: it is confident on every single frame
and wrong on every single one. Zero usable fixes, a 100% false-fix rate, the
worst value the scale can produce. Nothing about the model changed; only the
question being asked of it. Error statistics stay logged as diagnostics, and
the standing check is written into the scorer: <i>if the score improves while
the usable-fix rate does not, the metric is wrong again.</i> The lineage is
wiped a third time and the metric propagated through every prompt, document
and page.</p></div>

<div class="nb-pull"><span class="nb-time">the day in summary</span>
<div class="nb-quote"><p><b>What changed, and what it costs.</b> Three separate
things were found wrong today, and all three were found the same way &mdash; by
a human looking at a picture and saying that makes no sense, not by any check
the harness ran.</p>
<p><b>1. The evaluation asked an impossible question.</b> Holding out one map
block in five, plus a no-leakage buffer, left 28.2% of Berlin in no training
crop at all and put <b>100%</b> of test questions on ground the model had never
seen; 65% of test frames contained not one familiar pixel. For a system whose
entire job is to memorise one city, that is unanswerable by construction. Fixed
by holding out <i>viewpoints</i> instead of <i>regions</i>: train on all of
Berlin, test on frames 11&ndash;17&nbsp;m off the nearest training vantage.</p>
<p><b>2. The score rewarded the wrong behaviour &mdash; twice.</b> The median
favours a model that guesses the map centre (tight, mediocre, unimodal) over one
that genuinely memorises (bimodal: some places nailed, others badly missed). It
was caught reverting the only experiment that had begun to memorise anything.
Replacing it with a geometric mean fixed that symptom but was still a research
proxy chosen to make progress <i>visible</i> rather than to state what the
aircraft <i>needs</i> &mdash; the same mistake in nicer clothes. The score is now
the product requirement itself: usable fixes minus dangerous ones.</p>
<p><b>3. The harness was deleting its own work.</b> A reverted experiment had its
source discarded outright; only kept experiments entered the git trail. That is
safe only if the metric is correct and permanent, an assumption nobody had
written down and which had just been disproved twice in a day. Of 60 archived
experiments, the 13 kept ones survive; the 47 reverted ones do not.</p>
<p><b>What this means for everything before today: the record is void.</b> Every
experiment up to this point &mdash; roughly sixty-three of them, including the
743&nbsp;m champion the project chased for a week &mdash; was scored on an
evaluation that was largely unanswerable, using a statistic that preferred
guessing to learning. Their verdicts do not transfer. &ldquo;Reverted&rdquo;
mostly means &ldquo;did not help on an impossible test&rdquo;, and
&ldquo;kept&rdquo; may mean &ldquo;guessed the centre more tidily&rdquo;. The
lineage was therefore wiped rather than carried forward, and the old databases
archived; the design agent is deliberately <i>not</i> given the old conclusions,
because importing invalid refutations would steer it away from ideas that were
never actually refuted. The catalogue of approaches tried remains readable by a
human, but nothing about which of them <i>worked</i> survives.</p>
<p><b>And the plateau finally has an explanation.</b> Sixty-odd experiments sat
between 700&nbsp;m and 1&nbsp;km, and it was read as the problem being hard. On
the corrected evaluation the naive baseline scores <b>2.001</b> &mdash;
confidently wrong on every single frame, the worst value the scale allows &mdash;
while the one experiment the old metric threw away had already located 8% of
Berlin to within 100&nbsp;m. The work was not stalling because the task was
hard. It was stalling because the instruments were pointed at the wrong
thing.</p></div></div>

<div class="nb-row"><span class="nb-time">evening</span>
<p class="nb-text"><b>The champion is pointed at rural Prignitz &mdash; a
measurement, not an experiment.</b> Experiment&nbsp;5&rsquo;s code, unchanged,
trained and scored on the rural bounding box instead of Berlin: same
architecture, same eight epochs, same hyperparameters, only the box is
different. It needed no code change at all &mdash; the cell grid comes from the
raster (Berlin 55&times;54 = 2,970 cells, Prignitz 55&times;55 = 3,025), which is
the genericity requirement of &sect;0 being exercised rather than asserted.
Result: <b>0.113</b> against Berlin&rsquo;s 0.040.</p></div>

<div class="nb-row"><span class="nb-time">&nbsp;</span>
<p class="nb-text">The interesting part is <i>where</i> the 2.8&times; went.
The prediction beforehand was that fields would be indistinguishable and the
model would abstain more &mdash; the safe failure. It did not.
Abstention barely moved (3.0% &rarr; 3.8%) and usable fixes barely moved
(96.5% &rarr; 92.5%); <b>almost the entire loss is false fixes, 0.5% &rarr;
3.75%</b>, a 7.5&times; rise in exactly the outcome the mission score treats as
worse than silence. Median error is nearly unchanged at 29.5&nbsp;m, so when it
is right it is just as sharp: the gap is calibration, not precision. Belief
pooling protects against mass split across <i>adjacent</i> cells and against
mass scattered across <i>distant</i> ones, but not against mass landing
confidently on one wrong field that looks exactly like the right one &mdash; a
case Berlin never presents and Prignitz presents constantly. The unseen-region
diagnostic says the same thing from the other side: coverage <i>rises</i> to
64.2% (Berlin 38.2%) while staying 0% usable, i.e. on ground it has never seen
the rural model answers more readily and is wrong every time.</p></div>

<div class="nb-row"><span class="nb-time">&nbsp;</span>
<p class="nb-text">Nothing from this run was written to the experiment
database, <code>state/best.json</code> or the git trail. Scored against a
Berlin champion it would have been logged &ldquo;reverted&rdquo;, which would be
a meaningless verdict &mdash; the two areas are different questions, not
competing answers to one. It is recorded here and as a branch on the
<a href="research-evolution.html">evolution graph</a> instead. The research
question it names is sharper than Berlin&rsquo;s was: <i>cut false fixes
without spending usable ones.</i> Sharp enough that within the hour it had an
era of its own.</p></div>

<div class="nb-row"><span class="nb-time">22:00</span>
<p class="nb-text"><b>One model over both boxes: proposed, examined, dropped
&mdash; all before an experiment ran.</b> The idea was to stop letting each
area find its own setup and instead iterate toward a single model covering
Berlin <i>and</i> Prignitz, amending &sect;1&rsquo;s one-model-per-bounding-box
rule. It survived about an hour. The arithmetic was encouraging: the boxes sit
90&nbsp;km apart, so their union is 88&times;71&nbsp;km of which only
<b>1.6% is imaged</b> &mdash; a union grid would be 384,196 cells, nearly all
of it ground no camera has seen &mdash; while a grid over just the two imaged
boxes is 5,995 cells, barely twice Berlin&rsquo;s. It also needed no frozen
file changed: the scorer loads one ONNX per area, so a shared trunk exported
twice, each framed to its own raster, fits the existing interface. Dropped
anyway, on the grounds that unifying is a different question from the one
Prignitz actually poses, and running both at once would confound
them.</p></div>

<div class="nb-row"><span class="nb-time">&nbsp;</span>
<p class="nb-text"><b>That hour turned up a real measurement bug, and it is
still open.</b> Checking whether a shared model would fit the ESP32&#8209;P4
gave 4.88&nbsp;MB against a 4&nbsp;MB ceiling &mdash; an apparent blocker. It
wasn&rsquo;t. <code>MODEL_MAX_BYTES</code> was set in the <i>bootstrap
commit</i> and never revisited &mdash; <code>git log -S</code> returns exactly
one commit &mdash; with no derivation behind it. And it weighs the <b>fp32</b>
ONNX, while the deployment path this repo documents is esp&#8209;dl, which
ingests that ONNX for <b>int8</b> quantisation. Measured: the Berlin champion
is 2.94&nbsp;MB fp32 and <b>0.78&nbsp;MB int8</b>. The gate overstates every
model by about 3.8&times;, and the shared model that &ldquo;failed&rdquo; is
1.29&nbsp;MB quantised. Nothing in the pipeline quantises anything, so
&sect;3&rsquo;s &ldquo;quantization-aware training&rdquo; is a line in the spec
no experiment can currently act on. Deliberately <i>not</i> changed on the
spot: it is a frozen file, and the re-scoring that puts every past era on one
ruler runs the <i>current</i> scorer, so relaxing the gate could flip several
of the 17 archived gated-fails at once. That deserves its own step, with the
effect measured first.</p></div>

<div class="nb-row"><span class="nb-time">22:26</span>
<p class="nb-text"><b>A rural era opens &mdash; Prignitz alone.</b>
<code>AREAS=prignitz</code>, everything else about the narrow configuration
unchanged. The era&rsquo;s question is not Berlin&rsquo;s. Berlin had to prove
the approach worked at all; Prignitz already half works at 0.113, and the
entire gap is in one place, so the question is exact: <i>cut false fixes
without spending usable ones.</i></p></div>

<div class="nb-row"><span class="nb-time">22:58</span>
<p class="nb-text"><b>Experiment 1 &mdash; the seed.</b> The champion
architecture, no design agent, trained and scored on Prignitz natively so the
era starts from a number on its own board rather than a probe&rsquo;s.
<b>0.113</b>, reproducing the probe exactly: 92.5% usable, 3.75% false, 3.8%
abstaining, median 29.5&nbsp;m.</p></div>

<div class="nb-row"><span class="nb-time">23:30</span>
<p class="nb-text"><b>Experiment 2 &mdash; rotation-consensus decode. Refuted,
and informatively.</b> Training draws a fresh random heading per position per
epoch, so a genuinely memorised place ought to be heading-invariant while a
false match to a look-alike field should ride view-specific micro-texture that
does not survive being turned. The decode was changed to run the trunk over the
frame plus its 90/180/270&deg; rotations, average the four belief maps, and
decode the average &mdash; a wrong field would have to win from at least two
headings. Predicted false fixes 3.75% &rarr; &le;1.5%. Measured: <b>false fixes
went up</b>, 3.75% &rarr; 4.25%, usable 92.5% &rarr; 92.0%, mission 0.113
&rarr; 0.123. Reverted. The negative is worth more than the change would have
been: <b>the rural confusions are rotation-stable</b>. Two fields that fool the
model fool it from every heading, so this is not a viewing-angle artefact and
consensus at decode time cannot reach it. Whatever fixes it has to act during
training.</p></div>

<div class="nb-row"><span class="nb-time">23:30</span>
<p class="nb-text">Experiment 3 was designed and implemented on exactly that
reading &mdash; continuous-vantage training: jitter each training framing by up
to &plusmn;12&nbsp;px so vantages tile the ground instead of standing on the
24&nbsp;m lattice, since every eval view (and every real flight frame) stands
8&ndash;16&nbsp;px off-lattice, and a model that has only ever seen lattice
framings can tell fields apart by accidents of framing that vanish a few metres
away. It never trained. Recorded as the era&rsquo;s open move, not as a
result.</p></div>

</section>

<section class="nb-day">
<h2>1 August 2026</h2>
<p class="daysub">The rural era is rewound after four hours and three experiments; the trunk goes back to berlin-slim</p>

<div class="nb-row"><span class="nb-time">&nbsp;</span>
<p class="nb-text"><b>The era is closed and the repository rewound to the
commit before it opened.</b> <code>model/</code>, <code>state/best.json</code>
and the experiment database all return to the Berlin champion at <b>0.040</b>,
and the two Prignitz rows are removed from the lineage rather than archived as
an era &mdash; four hours and three experiments is a detour, not a chapter, and
carrying it as one would put a band on every chart in this site for work that
produced no kept result. Not a verdict on the rural problem: nothing about it
failed on its merits, and the 0.113 measurement and the rotation-stability
finding both stand. The reason is simpler &mdash; one question at a time, and
the question is still Berlin.</p></div>

<div class="nb-row"><span class="nb-time">&nbsp;</span>
<p class="nb-text">What survives the rewind, deliberately: this entry; the
detour drawn as a branch that leaves the berlin-slim line and stops, on the
<a href="research-evolution.html">evolution graph</a>; and the open finding
about the size gate above. The raw artifacts &mdash; run directories, exported
models, agent transcripts for all three experiments &mdash; are not deleted
either; they stay reachable at the git tag
<code>prignitz-detour-record</code>. The rule the project already had for
wiped eras applies here too: <i>re-measure or record, never quietly
erase</i>.</p></div>

</section>

<section class="nb-day">
<h2>28 August 2026</h2>
<p class="daysub">The flight test &mdash; a virtual UAV flies the champion from a to b</p>

<div class="nb-row"><span class="nb-time">morning</span>
<p class="nb-text"><b>The champion is put to the product&rsquo;s own
test.</b> A flight simulator (<code>sim/flightsim.py</code>, outside the
frozen pipeline, reading it untouched) flies a point-mass fixed-wing
through wind, gusts and gyro bias it cannot see, with the exported
champion&rsquo;s vision fixes as the only position correction &mdash;
camera frames are real orthophoto crops at the true pose, thresholds are
the frozen scorer&rsquo;s own, false fixes are accepted like any other,
and arrival is declared on the navigator&rsquo;s estimate but scored
against the truth. The commitment, fixed before the first run: whatever
the number is, it ships.</p></div>

<div class="nb-row"><span class="nb-time">&nbsp;</span>
<p class="nb-text"><b>The first Monte Carlo scores 53/100 &mdash; and the
bug is in the simulator, not the model.</b> Flights with perfect position
fixes still spiral off the map: the hand-built navigator had no heading
reference, so gyro bias integrated unbounded and guidance chased a
believed heading 100&deg;+ from the true one. Every real flight controller
carries a magnetometer; adding one (yaw aiding only &mdash; vision stays
the sole position source) takes the result to <b>91/100 flights arriving
within 100&nbsp;m</b>, median declared miss 45&nbsp;m. The control run
&mdash; same flights, vision disabled &mdash; arrives 0/100.</p></div>

<div class="nb-row"><span class="nb-time">&nbsp;</span>
<p class="nb-text"><b>The failures decompose cleanly.</b> 167 of 207
in-flight false fixes stand on the ~3% of Berlin the harness deliberately
never trained (the region-holdout diagnostic blocks); all three
catastrophic misses had their target on or beside that ground. Excluding
it, the in-flight false-fix rate is 0.77% &mdash; consistent with the
frozen eval&rsquo;s 0.5%. A deployed model trains on 100% of its box, so
the catastrophes are an artifact of flying over the research
configuration; they are reported as flown on the new
<a href="flight-path.html">flight path</a> page, which replays a
corner-to-corner 9&nbsp;km flight &mdash; one 6.5&nbsp;km false fix
survived, arrival 25.5&nbsp;m &mdash; over the Berlin orthophoto.</p></div>

</section>
"""


def render_notebook():
    """gallery/lab-notebook.html — a dated narrative of what happened and
    why, compiled once from git history + session transcripts. Content is
    static (NOTEBOOK_HTML above); the page shell reuses the same shared
    topnav/page_header/CREDITS as every other page."""
    html_page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{social_meta('gallery/lab-notebook.html', 'Lab notebook — Ballpark', 'The dated narrative of the research: what happened, what broke, and what was learned, day by day.')}
<title>Lab notebook &mdash; Ballpark</title>
<style>{CSS}</style></head><body>
{topnav('notebook')}
{compute_banner()}
<header class='page-head'>
<h1>Lab notebook</h1>
<p class='page-sub'>What we actually did, day by day &mdash; compiled from
the project&rsquo;s session history and commit log, not just the
experiment record. The <a href="index.html">research log</a> and <a
href="inference-paths.html">model designs</a> pages show what the loop
tried; this page is the surrounding story.</p></header>
<div class="nb-wrap">
{NOTEBOOK_HTML}
<p class="nb-source">Compiled from git history and Claude Code session
transcripts, 20 July &ndash; 1 August 2026 &mdash; condensed for readability; see
the <a href="index.html">research log</a> for the complete,
machine-generated experiment record.</p>
</div>
{credits_html()}</body></html>"""
    NOTEBOOK_OUT.parent.mkdir(exist_ok=True)
    NOTEBOOK_OUT.write_text(html_page)
    print(f"wrote {NOTEBOOK_OUT}")


def rsec(n, title, sub=""):
    """A numbered report section heading, in the register of the author's
    prior research reports: a light numeral, the title in serif at reading
    size, and a one-line italic summary so the section can be skimmed."""
    s = f"<p class='rsec-sub'>{sub}</p>" if sub else ""
    return (f"<div class='rsec'><span class='rsec-n'>{n}</span>"
            f"<h2 class='rsec-t'>{title}</h2></div>{s}")


def tile_grid():
    """Eight real 128 m Berlin tiles with their coordinates — the
    multiple-choice question made concrete. The crops and their lat/lon are
    baked as committed static files (assets/tiles/ — NOT gallery/, which is
    gitignored render output) by a one-off script run where data/ exists,
    because CI renders this page without the raster."""
    tj = REPO_ROOT / "assets" / "tiles" / "tiles.json"
    if not tj.exists():
        return ""
    d = json.loads(tj.read_text())
    cells = "".join(
        f"<figure class='tg-cell'>"
        f"<img src='assets/tiles/{t['file']}' loading='lazy' "
        f"alt='128 m aerial tile at {t['lat']}, {t['lon']}'>"
        f"<figcaption><span class='tg-idx'>tile {t['idx']:,} of "
        f"{d['n_cells']:,}</span>"
        f"{t['lat']:.5f}&nbsp;N &middot; {t['lon']:.5f}&nbsp;E"
        f"</figcaption></figure>"
        for t in d["tiles"])
    return (f"<div class='tile-grid'>{cells}</div>"
            "<p class='tg-cap'>Eight of the 2,970 answers. Every tile is "
            "128&nbsp;m of real Berlin and owns one coordinate, so "
            "&ldquo;which tile is this?&rdquo; is the same question as "
            "&ldquo;where am I?&rdquo; &mdash; a river bend is easy; the "
            "empty field at the end is why the model must also know when "
            "not to answer.</p>")


# The masonry packer for the model-designs mosaic — a real skyline layout,
# what Pinterest/Masonry.js actually do, replacing a first attempt with CSS
# grid row-spans + dense flow. The grid version left holes by construction:
# a 2-wide tile can only start where BOTH its columns are free, and the dead
# space that creates under uneven neighbours is unreachable to later tiles.
# Here each tile is measured at its real width and dropped at the lowest
# spot the skyline offers (leftmost on ties), so later small tiles fall into
# every valley. Cells go position:absolute inside the container ('packed');
# without JS the grid CSS remains as a gappy-but-sane fallback.
MASONRY_JS = """function dgpack(){
 var g=document.querySelector('.design-grid');if(!g)return;
 var gap=18,N=window.innerWidth<=640?2:(window.innerWidth<=1000?3:6),W=g.clientWidth,
     colW=(W-(N-1)*gap)/N;
 g.classList.add('packed');
 var colH=[],holes=[],i;for(i=0;i<N;i++)colH[i]=0;
 g.querySelectorAll('.dg-cell').forEach(function(c){
  var span=c.classList.contains('sz3')?3:(c.classList.contains('sz2')?2:1);
  if(span>N)span=N;
  c.style.width=(span*colW+(span-1)*gap)+'px';
  var h=c.offsetHeight,s,k,y;
  /* A pure skyline cannot reach the space under a wide tile that bridged
     columns of unequal height — measured on this very wall as seams of up
     to ~1,100 px. So those rectangles are remembered as holes, and later
     1-wide tiles drop into the topmost hole they fit before consulting
     the skyline at all. */
  if(span===1){
   var hi=-1;
   for(i=0;i<holes.length;i++)
    if(holes[i].y1-holes[i].y0>=h&&(hi<0||holes[i].y0<holes[hi].y0))hi=i;
   if(hi>=0){
    var ho=holes[hi];
    c.style.left=(ho.col*(colW+gap))+'px';c.style.top=ho.y0+'px';
    ho.y0+=h+gap;
    if(ho.y1-ho.y0<60)holes.splice(hi,1);
    return;
   }
  }
  var best=0,bestY=Infinity;
  for(s=0;s+span<=N;s++){
   for(y=0,k=s;k<s+span;k++)y=Math.max(y,colH[k]);
   if(y<bestY-0.5){bestY=y;best=s;}
  }
  for(k=best;k<best+span;k++){
   if(bestY-colH[k]>=60)holes.push({col:k,y0:colH[k],y1:bestY});
   colH[k]=bestY+h+gap;
  }
  c.style.left=(best*(colW+gap))+'px';
  c.style.top=bestY+'px';
 });
 g.style.height=(Math.max.apply(null,colH)-gap)+'px';
}
window.addEventListener('load',dgpack);
var dgrt;window.addEventListener('resize',function(){
 clearTimeout(dgrt);dgrt=setTimeout(dgpack,120)});"""


# "How to use it" — the minimum real inference client. Kept as a plain string
# (NOT inside render_overview's f-string body: it is full of braces) and kept
# honest against the actual contract: model/model.py exports input "frame"
# (1,3,128,128) float32/255 -> output "uvc" (u, v, conf), u/v normalised over
# the training raster, mapped to WGS84 via the raster's UTM georeference
# (data/berlin/meta.json), abstention below score.py's CONF_THRESHOLD = 0.3.
HOWTO_CODE = """# pip install onnxruntime opencv-python pyproj numpy
import cv2, numpy as np, onnxruntime as ort
from pyproj import Transformer

# the Berlin model's georeference (from the training raster's metadata)
ORIGIN = (385883.0, 5821316.0)          # UTM 33N of the map's top-left corner
SIZE   = (6939, 6828)                   # map size in px; 1 px = 1 m
to_wgs84 = Transformer.from_crs("EPSG:32633", "EPSG:4326")

sess = ort.InferenceSession("berlin.onnx")
cam  = cv2.VideoCapture(0)              # the downward-facing camera

ok, bgr = cam.read()                    # 1 - grab one frame
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, (128, 128))       # frame should cover ~128 m of ground
x = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0

u, v, conf = sess.run(None, {"frame": x})[0][0]   # 2 - ask the network
lat, lon = to_wgs84.transform(ORIGIN[0] + u * SIZE[0],
                              ORIGIN[1] - v * SIZE[1])

if conf &lt; 0.3:                          # 3 - trust it only when it is sure
    print("no fix - the model abstains")
else:
    print(f"{lat:.5f}, {lon:.5f}  (confidence {conf:.2f})")"""


def train_grid():
    """Fifty tiles of the actual training raster, full width — 'Our training
    data' at the end of the research section. Same static-asset scheme as
    tile_grid(): crops + coordinates baked into assets/tiles/train/ by a
    one-off local script, because CI has no raster."""
    tj = REPO_ROOT / "assets" / "tiles" / "train" / "train.json"
    if not tj.exists():
        return ""
    d = json.loads(tj.read_text())
    cells = "".join(
        f"<figure><img src='assets/tiles/train/{t['file']}' loading='lazy' "
        f"alt='128 m aerial training tile at {t['lat']}, {t['lon']}'>"
        f"<figcaption>{t['lat']:.5f} &middot; {t['lon']:.5f}</figcaption>"
        f"</figure>"
        for t in d["tiles"])
    return f"<div class='train-grid'>{cells}</div>"


def design_grid(hist):
    """Every agent-drawn model-design figure as a packed tapestry — a
    visual census of the whole search, feeding the inference-paths page
    for anyone who wants one at full size."""
    rows = [r for r in hist
            if (r.get("arch_svg") or "").lstrip().startswith("<svg")]
    if not rows:
        return ""
    # Newest first, so the wall opens on where the search ENDED — and the
    # champion (best re-scored mission score across every era) leads it,
    # visibly set apart from the also-rans it beat.
    rows = rows[::-1]
    scored = [r for r in rows if r.get("mission_score") is not None
              and r["mission_score"] < FAIL]
    champ_seq = min(scored, key=lambda r: r["mission_score"])["seq"] \
        if scored else None
    cells = []
    for r in rows:
        champ = r["seq"] == champ_seq
        tag = "kept" if r["kept"] else "reverted"
        no = f"{r['era_index'] + 1}.{r['src_id']}"
        title = esc((r["title"] or "").split(":")[0])
        label = "champion" if champ else tag
        # Tile size IS the verdict: reverted designs are one grid column,
        # kept designs two, the champion three — the mosaic's texture is
        # small failures packing around the ideas that survived.
        size = "sz3" if champ else ("sz2" if r["kept"] else "sz1")
        # data-ovfig: clicking opens the shared zoom/pan overlay in place —
        # the same viewer the champion figure uses — instead of navigating
        # away to the model-designs page.
        cells.append(
            f"<figure class='dg-cell {size} {tag}{' champ' if champ else ''}'>"
            f"<div class='dg-fig' data-ovfig data-no='{no}' "
            f"data-title='{esc(r['title'] or '')}'>{r['arch_svg']}</div>"
            f"<figcaption><b>{no}</b> "
            f"&middot; {label} &middot; {title}</figcaption></figure>")
    return f"<div class='design-grid'>{''.join(cells)}</div>"


def loop_cycle_svg():
    """The autoresearch loop as a four-step cycle — same figure vocabulary
    as the author's prior workshop site (approach.html), relabeled for this
    project: PROPOSE / TRAIN / SCORE / KEEP, a human supervisor steering
    into PROPOSE from outside, and the frozen ruler feeding SCORE."""
    return """<svg class="cycle" viewBox="0 0 640 420" role="img"
     aria-label="The loop: an agent proposes one change, the harness trains and scores it, and it is kept only if it beats the champion — while a human supervises from outside and a frozen ruler does the scoring">
  <defs>
    <marker id="cyc-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#8a857a"/>
    </marker>
    <marker id="cyc-arrow-h" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7.5" markerHeight="7.5" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#8c2f1f"/>
    </marker>
    <marker id="cyc-arrow-e" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7.5" markerHeight="7.5" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#8a6a1e"/>
    </marker>
  </defs>
  <path class="carrow" marker-end="url(#cyc-arrow)" d="M346.8,99.1 A125,125 0 0 1 415.9,168.2"/>
  <path class="carrow" marker-end="url(#cyc-arrow)" d="M415.9,261.8 A125,125 0 0 1 346.8,330.9"/>
  <path class="carrow" marker-end="url(#cyc-arrow)" d="M253.2,330.9 A125,125 0 0 1 184.1,261.8"/>
  <path class="carrow" marker-end="url(#cyc-arrow)" d="M184.1,168.2 A125,125 0 0 1 253.2,99.1"/>
  <text class="ccenter"     x="300" y="210">THE LOOP</text>
  <text class="ccenter-sub" x="300" y="231">experiment after experiment</text>
  <text class="clabel" x="300" y="50"  text-anchor="middle">PROPOSE</text>
  <text class="csub"   x="300" y="67"  text-anchor="middle">one focused change</text>
  <circle class="cnode" cx="300" cy="90" r="15"/>
  <text class="cnum" x="300" y="95" text-anchor="middle">1</text>
  <circle class="cnode" cx="425" cy="215" r="15"/>
  <text class="cnum" x="425" y="220" text-anchor="middle">2</text>
  <text class="clabel" x="448" y="211" text-anchor="start">TRAIN</text>
  <text class="csub"   x="448" y="228" text-anchor="start">on the whole city</text>
  <circle class="cnode" cx="300" cy="340" r="15"/>
  <text class="cnum" x="300" y="345" text-anchor="middle">3</text>
  <text class="clabel" x="300" y="374" text-anchor="middle">SCORE</text>
  <text class="csub"   x="300" y="391" text-anchor="middle">the mission penalty</text>
  <circle class="cnode" cx="175" cy="215" r="15"/>
  <text class="cnum" x="175" y="220" text-anchor="middle">4</text>
  <text class="clabel" x="152" y="211" text-anchor="end">KEEP</text>
  <text class="csub"   x="152" y="228" text-anchor="end">only if it beats the champion</text>
  <text class="chuman-lbl" x="632" y="58" text-anchor="end">HUMAN SUPERVISOR</text>
  <text class="chuman-sub" x="632" y="76" text-anchor="end">sets the levers,</text>
  <text class="chuman-sub" x="632" y="92" text-anchor="end">curates the inspiration</text>
  <circle class="chuman" cx="505" cy="75" r="5"/>
  <path class="carrow-h" marker-end="url(#cyc-arrow-h)" d="M505,75 C436,82 374,88 316,84"/>
  <text class="cexpert-lbl" x="8" y="350" text-anchor="start">FROZEN RULER</text>
  <text class="cexpert-sub" x="8" y="368" text-anchor="start">pipeline, eval set, scorer &mdash;</text>
  <text class="cexpert-sub" x="8" y="384" text-anchor="start">read-only to the agents</text>
  <circle class="cexpert" cx="172" cy="360" r="5"/>
  <path class="carrow-e" marker-end="url(#cyc-arrow-e)" d="M172,360 C212,360 252,352 288,346"/>
</svg>"""


def render_overview(exps):
    """index.html at the repo root — the project's front door: what this is,
    live status numbers from the lineage DB, and links into the gallery
    pages. Same airloom-style typography and shared topnav as the rest."""
    dev = [e for e in exps if e["kind"] != "holdout_check"]
    n_kept = sum(1 for e in dev if e["kept"])
    # The three tiles under the headline describe the SEARCH, not the model:
    # how many rulers it was measured against, how many attempts it took, and
    # what that came to. The model's own error stats are one line above and on
    # the research log; repeating them here spent the most valuable space on
    # the page restating the hero number.
    _hist, _eras = load_history()
    n_all_exp = sum(1 for r in _hist if r["kind"] != "holdout_check")
    n_eras = len(_eras)
    cost_all = sum((r["cost_agent"] or 0) + (r["cost_gpu"] or 0) for r in _hist)

    # Figures for "The answer" and "The verdict". Read from the champion's own
    # record and training info rather than typed in, so the prose cannot drift
    # from the model it describes the way the 0.183 numbers did.
    champ = next((e for e in reversed(dev)
                  if e["kept"] and e["primary_metric"] is not None
                  and e["primary_metric"] < FAIL), None)
    best_mb = (champ["model_bytes_max"] or 0) / 1e6 if champ else 0.0
    best_ms = champ["latency_ms_host_proxy"] if champ else 0.0
    cm = json.loads(champ["metrics_json"] or "{}") if champ else {}
    cell = next((c for a in cm.get("areas", [])
                 for c in a.get("buckets", {}).values()), {})
    hold = next((a.get("region_holdout") or {} for a in cm.get("areas", [])), {})
    usable_pct = f"{100*(cell.get('usable_fix_rate') or 0):.1f}%"
    false_pct = f"{100*(cell.get('false_fix_rate') or 0):.1f}%"
    p10_s = fmt_m(cell.get("p10_error_m"))
    holdout_med = fmt_m(hold.get("median_error_m"))
    # Lattice geometry. Read the vantage count out of the experiment that
    # introduced full-lattice coverage and named it in its own title — NOT
    # from train_info's n_train_crops, which is total VIEWS (581,696 = every
    # vantage seen several times over) and would overstate the lattice by 8x.
    n_cells, n_vantages = 2970, 72712
    for e in dev:
        m = re.search(r"all ([\d,]+) vantages", e["title"] or "")
        if m:
            n_vantages = int(m.group(1).replace(",", ""))
            break

    # The winning design's own figure — the one the agent drew before it was
    # allowed to train. Describing the architecture in prose and then not
    # showing the drawing that exists is a strange thing to do on a page whose
    # whole claim is "here is the technique that worked".
    # The contract row belongs under "The question"; the before/after maps are
    # evidence for "The answer", so they are placed separately.
    _chal = challenge_block(exps)
    chal_contract, _, chal_maps = _chal.partition("<!--SPLIT-->")
    tile_grid_html = tile_grid()
    train_grid_html = train_grid()
    design_grid_html = design_grid(_hist)

    champ_fig = ""
    _csvg = (champ.get("arch_svg") or "") if champ else ""
    # The figures live on a page where they are numbered, and cross-reference
    # each other ("byte-identical to Fig. 4"). Lifted onto the overview, which
    # has no numbered figures, that points at nothing — so name the relation
    # instead of the number.
    _csvg = re.sub(r"\bFig\.\s*\d+(\.\d+)?\b", "the previous experiment", _csvg)
    if _csvg.lstrip().startswith("<svg"):
        champ_fig = (
            f"<div class='contract-fig' data-ovfig "
            f"data-title='The design that works — experiment {champ['id']}'>"
            f"{_csvg}"
            f"<p class='contract-cap'>The champion, drawn by the design agent "
            f"<i>before</i> it was allowed to train. One camera crop enters at "
            f"the left; the trunk squeezes it to a short vector; the wide fan "
            f"is the choice over all {n_cells:,} map tiles; the small head "
            f"after it is the nudge inside the winning tile. Grey is the "
            f"harness's frozen contract, <span class='chg'>red is what this "
            f"experiment changed</span>, and the ochre lane at the bottom is "
            f"training-only scaffolding that never boards the aircraft. Click "
            f"to enlarge, or see every proposal on the "
            f"<a href='gallery/inference-paths.html'>model designs</a> "
            f"page.</p></div>")
    best = next((e["primary_metric"] for e in reversed(dev)
                 if e["kept"] and e["primary_metric"]
                 and e["primary_metric"] < FAIL), None)
    best_size = next((e["model_bytes_max"] for e in reversed(dev)
                      if e["kept"] and e["model_bytes_max"]), None)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    best_s = fmt_score(best) if best is not None else "—"
    size_s = f"{best_size/1024:,.0f} KB" if best_size else "—"
    # The mission score is in [0,2] and 0 is the goal, so "distance to goal"
    # is just the score itself, and progress is how far it has closed from the
    # first scoreable run toward 0.
    usable_best = med_best = false_best = abstain_best = None
    for e in reversed(dev):
        if e["kept"] and e.get("metrics_json"):
            try:
                cells = json.loads(e["metrics_json"])["areas"][0]["buckets"].values()
                worst = max(cells, key=lambda c: c.get("mission_score") or 0)
                usable_best = worst.get("usable_fix_rate")
                abstain_best = worst.get("abstain_rate")
                false_best = worst.get("false_fix_rate")
                med_best = worst.get("median_error_m")
            except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
                pass
            break
    usable_s2 = f"{100*usable_best:.0f}%" if usable_best is not None else "—"
    med_s = f"{med_best:,.0f} m" if med_best else "—"
    false_s = f"{100*false_best:.1f}%" if false_best is not None else "—"
    abst_s = f"{100*abstain_best:.0f}%" if abstain_best is not None else "—"
    # a bare 0.183 says nothing without its scale, so draw one: where this run
    # sits between 0 (every frame usable) and 2 (confidently wrong on all)
    pen_s = f"{best:.3f}" if best is not None and best < FAIL else "—"
    miss_s = f"{100*(1-usable_best):.0f}%" if usable_best is not None else "—"
    if usable_best is not None:
        scale_svg = (
            "<span class='sc-wrap'><span class='sc-bar'>"
            f"<span class='sc-dot' style='left:{100*usable_best:.1f}%'></span></span>"
            "<span class='sc-ends'><span>0% &mdash; never</span>"
            "<span>50%</span><span>100% &mdash; every frame</span></span></span>")
    else:
        scale_svg = ""
    baseline = next((e["primary_metric"] for e in dev
                     if e["primary_metric"] and e["primary_metric"] < FAIL),
                    None)
    if best is not None and baseline and baseline > 0 and best < baseline:
        progress_s = f"{100 * (baseline - best) / baseline:,.0f}%"
    elif best is not None and best <= 0.05:
        progress_s = "100%"
    else:
        progress_s = "0%"

    # The "One flight, replayed" section is owned by sim/render_flightpath.py
    # and embedded here verbatim (rel='' because this page lives at the repo
    # root). Imported lazily: that module imports this one for the page shell.
    from sim.render_flightpath import FP_CSS, flight_section
    flight_html = flight_section(
        "", "The full 100-flight record &mdash; the estimation-error chart, "
        "every arrival and every failure &mdash; is on the "
        "<a href='gallery/flight-path.html'>flight path</a> page.")

    html_page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{social_meta('', 'Ballpark — “Not all who wander are lost”: Can a UAV learn a city by heart — no GPS, no map on board, just a $4 flight computer?', 'Yes: a from-scratch neural memory that knows Berlin by heart. One downward photo in, (lat, lon, confidence) out — the map is the weights, one city in 3.1 MB. Found by an autonomous loop of coding agents in 81 experiments for $364.')}
<title>Ballpark &mdash; &ldquo;Not all who wander are lost&rdquo;: Can a UAV learn a city by heart &mdash; no GPS, no map on board, just a $4 flight computer?</title>
<style>{CSS}
{FP_CSS}</style><script>{PATHS_JS}</script></head><body>
{topnav('overview', root=True)}
{compute_banner()}
<div class="paths-wrap">
<h1 class="home-h1">&ldquo;Not all who wander are lost&rdquo;</h1>
<h2 class="home-q">Can a UAV learn a city by heart &mdash; no GPS, no map
on board, just a $4 flight computer?</h2>
<div class="bl-h">The bottom line</div>
<p class="bottom-line"><span class="hl">Yes! We found a special-purpose
neural memory architecture that &lsquo;remembers&rsquo; an exact
geo-location, given an in-flight daytime picture of Berlin.</span>
No GPS, no internet, no map on board, just a
{best_mb:.1f}&nbsp;MB file of weights. It works on {usable_pct} of camera
frames, and it took {n_all_exp} experiments and ${int(cost_all):,} to
find.</p>

<div class="stat-hero">
  <b>{usable_s2}</b><span>of camera frames give a usable position fix</span>
  {scale_svg}
  <p class="hero-sub">confident <i>and</i> within {TARGET_M:.0f}&nbsp;m. The
  loop minimises the <b>mission penalty</b>:
  <span class="pen">{pen_s}</span> = <span class="pen">{miss_s}</span> no fix
  + <span class="pen">{false_s}</span> wrong fix &mdash; a confident error
  counts double.</p>
</div>
{flight_html}

{rsec(2, "The question", "One frame, no map, no internet &mdash; and it has to know when it doesn&rsquo;t know.")}
<div class="pnote">
<p>A UAV flies over a city it has trained on. Its camera takes one picture
straight down. From that picture alone: where is it?</p>
<p>And &mdash; how much should the aircraft trust the answer? A wrong fix
flown with confidence is worse than no fix at all. The model has to know when
it doesn't know.</p>
</div>
{chal_contract}

{chal_maps}

{rsec(3, "The answer", "Stop asking for a coordinate. Ask which tile.")}
{champ_fig}
<div class="pnote">
<p>Ask a network for latitude and longitude and it fails badly: when it is
unsure it has no way to say so, and hedging toward the middle of the map is
what keeps its average error down. Our first attempt was confident on every
frame and wrong on every frame.</p>
<p><b>So we stopped asking for a coordinate and asked a multiple-choice
question instead.</b> Berlin is cut into {n_cells:,} tiles of 128&nbsp;m and
the network points at one. Now the answer carries its own certainty: sure,
and the belief piles onto a single tile; lost, and it smears across the city
&mdash; and the UAV says nothing instead. A second output nudges the fix to
a precise point inside the chosen tile.</p>
</div>

<div class="pnote">
<p>The result is one file of <b>{best_mb:.1f}&nbsp;MB</b> that answers in
<b>{best_ms:.1f}&nbsp;ms</b>. Nothing is looked up, nothing is matched
&mdash; the map <i>is</i> the weights. Each refinement on the way is its own
experiment in the <a href="gallery/index.html">research log</a>, with the
figures on the <a href="gallery/inference-paths.html">model designs</a>
page.</p>
</div>

<div class="stats">
  <div class="stat"><b>{n_eras}</b><span>research eras</span></div>
  <div class="stat"><b>{n_all_exp}</b><span>experiments</span></div>
  <div class="stat"><b>{fmt_usd(cost_all)}</b><span>total cost</span></div>
</div>

<div class="cta">
<a class="cta-link" href="assets/models/berlin.onnx" download>Download the
Berlin model &darr;</a>
<span class="cta-meta">berlin.onnx &middot; {best_mb:.1f}&nbsp;MB &middot;
one function: <i>estimate_position(frame) &rarr; (lat, lon,
confidence)</i></span>
</div>

<div class="pnote">
<p><b>One model per area, by design.</b> Asked about a block of Berlin held
out of training entirely, the same model returns <b>0% usable fixes</b>
&mdash; the map lives in the weights, so ground it was never shown is ground
it cannot place. The pipeline takes any bounding box; each trained model
knows only its own. Night flying is scoped out for now: everything here is
raw daytime imagery.</p>
</div>

{tile_grid_html}

{rsec(4, "How our research works", "The model was found, not designed &mdash; by a loop of coding agents.")}

<div class="contract-fig static">{contract_svg()}
<p class="contract-cap">The search space. The gray endpoints are frozen
&mdash; one camera crop in, one <i>(lat,&nbsp;lon,&nbsp;confidence)</i>
answer out &mdash; and the dashed box is everything an experiment may
change. Every design in the
<a href="gallery/inference-paths.html">model designs</a> gallery is one way
of filling it.</p>
</div>

<div class="approach">
<div class="approach-text">
<p>No human designed the champion. A
<a href="https://github.com/karpathy/autoresearch">Karpathy-style
autoresearch loop</a> did: each round, a coding agent reads the full
experiment history, proposes <b>one</b> focused change, and pre-registers
its hypothesis and expected outcome before any code is written. The harness
trains it, scores it, and keeps it only if it beats the champion.</p>
<p><b>Two roles, split clean.</b> The <i>agents</i> own the experiments. The
<i>frozen harness</i> owns the exam: the data pipeline, the held-out
viewpoints and the scorer are read-only, so the loop can rewrite the model
but never the ruler it is measured by. <i>We</i> supervise from outside
&mdash; set the levers, curate the inspiration, and decide when an era is
over.</p>
</div>
<div class="approach-cycle">
{loop_cycle_svg()}
<p class="cycle-cap">Each round keeps only what beats the best so far, and
the next experiment builds on it.</p>
</div>
</div>

<div class="strategies">
<div class="strategy"><h3 class="strategy-h">Pre-registered, one change at a time</h3>
<p>Hypothesis, method and expected outcome are written down <b>before</b>
implementation, and each experiment makes exactly one focused change &mdash;
so every verdict in the record says something.</p></div>
<div class="strategy"><h3 class="strategy-h">Pivoting on patience</h3>
<p>A run has a <b>patience</b>: spend it on reverted experiments and the
next design must abandon the current backbone for a genuinely different one
&mdash; enforced by reading the resulting source, not by asking nicely.</p></div>
<div class="strategy"><h3 class="strategy-h">A frozen ruler</h3>
<p>The score is the <b>product requirement</b> &mdash; usable fixes minus
confidently-wrong ones &mdash; and the agent cannot touch it. When a score
and the product disagree, the ruler gets fixed and the lineage is re-measured,
never quietly erased.</p></div>
<div class="strategy"><h3 class="strategy-h">Everything is recorded</h3>
<p>Every experiment is a git commit, a database row and a gallery entry
&mdash; kept and reverted alike. The <a href="gallery/research-lineage.html">
lineage</a>, <a href="gallery/research-evolution.html">evolution graph</a>
and <a href="gallery/lab-notebook.html">lab notebook</a> are rendered from
that record, not written after the fact.</p></div>
</div>

<div class="sec-h">Our model designs</div>
<div class="pnote">
<p>Every experiment's model design, drawn by the design agent itself
<i>before</i> it was allowed to train &mdash; camera frame entering on the
left, (lat, lon, confidence) leaving on the right, red marking what that
experiment changed. All fifty-nine that carry a figure, kept and reverted
alike; the <a href="gallery/inference-paths.html">model designs</a> page
has each at full size:</p>
</div>
{design_grid_html}

<div class="sec-h">Our training data</div>
<div class="pnote">
<p>Everything the model knows, it learned from one picture: an open-licensed
aerial orthophoto mosaic of Berlin from the Berlin/Brandenburg geoportal
(&copy;&nbsp;GeoBasis-DE/LGB, dl&#8209;de/by&#8209;2&#8209;0, modified),
fetched by the frozen pipeline and resampled to 1&nbsp;m per pixel &mdash;
6,939&nbsp;&times;&nbsp;6,828 pixels covering the whole bounding box. No
Google or Bing tiles anywhere: only open geodata, so the entire project can
be published. Training crops 128&nbsp;m frames from every position on that
mosaic; the held-out eval frames stand on the same ground, framed from
positions and headings training never used. A hundred of the {n_cells:,}
tiles:</p>
</div>
{train_grid_html}

{rsec(5, "How to use it", "One file, one function, three steps.")}
<div class="pnote">
<p><b>1. Download the model:</b> <a href="assets/models/berlin.onnx"
download>berlin.onnx</a> ({best_mb:.1f}&nbsp;MB). It runs anywhere ONNX
runs &mdash; a laptop, a Raspberry&nbsp;Pi, the ESP32&#8209;P4 target.</p>
<p><b>2. Feed it camera frames, 3. trust it only when it is sure</b> &mdash;
the whole client is this:</p>
</div>
<pre class="howto">{HOWTO_CODE}</pre>
<div class="pnote">
<p class="howto-note">The frame must look straight down over Berlin and
cover roughly 128&nbsp;m of ground (about 100&nbsp;m altitude with a
typical lens). Below confidence 0.3 the fix is discarded, exactly as the
research scores it.</p>
</div>

{rsec(6, "Explore the record", "Every experiment, every design, every dead end.")}
<div class="explore">
<a class="card" href="gallery/index.html"><b>The research log</b>
<span>Every experiment ever run, failures included: pre-registered
hypotheses, results, per-area × lighting scoreboards, the exact prompts
the agents received, and one real worked example per experiment — the same
held-out viewpoint through each model's actual deployed weights.</span></a>
<a class="card" href="gallery/research-lineage.html"><b>Experiment
lineage</b>
<span>The family tree of the search: every experiment as one node in
discovery order, arcs to the design it built on — hover any node to trace
its ancestry, kept trunk and dead branches alike.</span></a>
<a class="card" href="gallery/inference-paths.html"><b>Model
designs</b>
<span>The technical figures: each experiment's model design, drawn by the
agent itself before training, in one shared visual language — frozen
endpoints aligned so you can scroll and compare designs directly.</span></a>
</div>

{rsec(7, "Proven alternatives", "GPS-denied localisation is not unsolved &mdash; this just walks a different road.")}
<div class="pnote">
<p>GPS-denied visual localization is not an unsolved problem. The
established, field-tested family matches live camera frames against
<i>georeferenced reference imagery carried on the aircraft</i> — e.g.
<a href="https://github.com/TIERS/wildnav">WildNav</a>
(<a href="https://arxiv.org/abs/2210.09727">Vision-based GNSS-Free
Localization for UAVs in the Wild</a>), which matches UAV photographs
against satellite tiles with deep feature matching and demonstrated
GNSS-comparable accuracy in real flights. If you need working GPS-denied
navigation today, start there, not here.</p>
<p>This project deliberately walks a different road, for two reasons.
<b>Licensing:</b> the strongest matchers in that stack — Magic Leap's
<a href="https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/LICENSE">SuperGlue</a>
/ <a href="https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE">SuperPoint</a>
pretrained networks — are licensed for noncommercial research only, without
the right to sublicense, which is incompatible with a fully open-sourceable,
commercially usable system (permissive alternatives like
<a href="https://github.com/cvg/LightGlue">LightGlue</a> (Apache-2.0)
exist, but the whole approach still means shipping reference imagery on
the airframe — this project's hardest constraint rules that out).
<b>Curiosity:</b> the actual motivation is ground-level research — can an
autonomous loop of coding agents discover a genuinely different approach,
with no reference imagery on board and the map living entirely in the
network's weights, under a low-light premise? It may or may not end up
competitive with the field-tested systems above. Saying so out loud is
part of the experiment.</p>
<p><b>The vision, if it works:</b> a novel algorithm that lets anyone draw
a bounding box around a place they care about and generate their own tiny,
punchy, self-contained geo-boxed model — reliable, personalized, and free.
Open data in, open weights out; no reference imagery to license, no vendor
to call, no cloud to depend on. The map is yours, and it lives in a few
megabytes you own.</p>
</div>

<p class="psub num" style="margin-top:34px">updated {now} · experiments run on a single local
machine; the loop commits every result to git as it goes</p>
</div>
{OVERLAY_HTML}
<script>{MASONRY_JS}</script>
{credits_html()}</body></html>"""
    OVERVIEW_OUT.write_text(html_page)
    print(f"wrote {OVERVIEW_OUT}")


def contract_svg():
    """The meta-figure at the top of the inference-paths page: the frozen
    contract (camera frame in, (u, v, conf) out, both gray) with a dashed
    placeholder box for everything an experiment may redraw, and the ochre
    training-signals lane beneath it. Same glyph geometry and palette as the
    per-experiment figures (see archive/arch_svg_reference.py)."""
    INK, MUT, FAINT, ACC, OCH = ("#111111", "#6b6a60", "#9b998c",
                                 "#8c2f1f", "#8a6a1e")
    FONT = "Palatino,Georgia,serif"
    IC = 112  # inference lane center y

    def txt(x, y, s, size=10, color=MUT, w=400, anchor="middle", style=""):
        return (f"<text x='{x:.0f}' y='{y:.0f}' font-family='{FONT}' "
                f"font-size='{size}' fill='{color}' font-weight='{w}' "
                f"text-anchor='{anchor}' {style}>{s}</text>")

    def harrow(x1, x2, y, color=FAINT):
        return (f"<line x1='{x1}' y1='{y}' x2='{x2 - 5}' y2='{y}' "
                f"stroke='{color}' stroke-width='1'/>"
                f"<path d='M {x2},{y} l -6,-3 v 6 Z' fill='{color}'/>")

    b = []
    # lane labels, identical to the per-experiment figures
    b.append(txt(8, 26, "INFERENCE PATH — WHAT FLIES", 9, FAINT, 600, "start",
                 "letter-spacing='1.8'"))
    b.append(txt(8, 236, "TRAINING SIGNALS — NEVER FLY", 9, OCH, 600, "start",
                 "letter-spacing='1.8'"))
    # frozen input: the canonical terrain camera frame (glyph v2 — identical
    # geometry to the per-experiment figures and the prompt snippet)
    Y = IC - 38
    b.append(
        f"<g id='cam-terrain'>"
        f"<rect id='frozen-input' x='26' y='{Y}' width='76' height='76' "
        f"fill='#f6f4ea' stroke='{FAINT}' stroke-width='1.6'/>"
        f"<path d='M31 {Y+47} L97 {Y+27}' stroke='#e6e3d4' stroke-width='5' fill='none'/>"
        f"<path d='M59 {Y+6} L49 {Y+70}' stroke='#e6e3d4' stroke-width='3.5' fill='none'/>"
        f"<rect x='34' y='{Y+9}' width='12' height='8' fill='#d9d5c3' transform='rotate(-8 40 {Y+13})'/>"
        f"<rect x='78' y='{Y+8}' width='10' height='11' fill='#cfccbd'/>"
        f"<rect x='35' y='{Y+57}' width='13' height='8' fill='#d9d5c3'/>"
        f"<rect x='75' y='{Y+50}' width='10' height='9' fill='#cfccbd' transform='rotate(6 80 {Y+54})'/>"
        f"<rect x='54' y='{Y+31}' width='9' height='8' fill='#d9d5c3' opacity='.85'/>"
        f"<ellipse cx='86' cy='{Y+63}' rx='8' ry='6' fill='{OCH}' opacity='.12'/>"
        f"<circle cx='40' cy='{Y+28}' r='.7' fill='{MUT}' opacity='.5'/>"
        f"<circle cx='70' cy='{Y+14}' r='.7' fill='{MUT}' opacity='.5'/>"
        f"<circle cx='92' cy='{Y+36}' r='.7' fill='{MUT}' opacity='.5'/>"
        f"<circle cx='52' cy='{Y+52}' r='.7' fill='{MUT}' opacity='.5'/>"
        f"<circle cx='82' cy='{Y+70}' r='.7' fill='{MUT}' opacity='.5'/>"
        f"<circle cx='31' cy='{Y+42}' r='.7' fill='{MUT}' opacity='.5'/></g>")
    b.append(txt(64, IC - 44, "128²×3", 9, FAINT))
    b.append(txt(64, IC + 50, "camera frame", 10.5, MUT, 600))
    # What the camera frame actually is depends on the branch's lighting
    # buckets, so read them rather than hardcode: berlin-slim collapses them to
    # a single raw-daytime pass-through, and every figure on this page already
    # says "one daytime frame". Hardcoding "night" here contradicted all 59.
    b.append(txt(64, IC + 62, frame_caption(), 9.5, FAINT))
    b.append(txt(64, IC + 75, "frozen contract", 8.5, FAINT,
                 style="font-style='italic'"))
    b.append(harrow(108, 176, IC))
    # the placeholder: everything between the endpoints is the search space
    bx, by, bw, bh = 180, 40, 560, 144
    b.append(f"<rect x='{bx}' y='{by}' width='{bw}' height='{bh}' fill='none' "
             f"stroke='{ACC}' stroke-width='1.6' stroke-dasharray='9 7'/>")
    cx = bx + bw / 2
    b.append(txt(cx, IC - 14, "the experiment goes here", 14, INK, 600))
    b.append(txt(cx, IC + 6,
                 "architecture · feature extraction · decode · confidence — "
                 "the agent may redraw all of it", 10, MUT))
    b.append(txt(cx, IC + 22,
                 "each figure below is one proposal for the inside of this box",
                 9.5, FAINT, style="font-style='italic'"))
    # deployment gate note, hanging off the box like a margin annotation
    b.append(txt(972, by + bh + 16,
                 "whatever fills the box must export to one ONNX file "
                 "≤ 4 MiB and answer in ≤ 250 ms", 9, FAINT, anchor="end",
                 style="font-style='italic'"))
    # frozen output
    b.append(harrow(bx + bw + 6, 796, IC))
    ox = 812
    b.append(f"<line x1='{ox - 7}' y1='{IC}' x2='{ox + 7}' y2='{IC}' "
             f"stroke='{FAINT}' stroke-width='1.4'/>"
             f"<line x1='{ox}' y1='{IC - 7}' x2='{ox}' y2='{IC + 7}' "
             f"stroke='{FAINT}' stroke-width='1.4'/>"
             f"<circle cx='{ox}' cy='{IC}' r='3.9' fill='none' "
             f"stroke='{FAINT}' stroke-width='1.4'/>"
             f"<circle cx='{ox}' cy='{IC}' r='1.5' fill='{FAINT}'/>")
    b.append(txt(ox + 16, IC - 18, "frozen contract", 8.5, FAINT,
                 anchor="start", style="font-style='italic'"))
    b.append(txt(ox + 16, IC - 2, "(lat, lon, confidence)", 13, MUT, 600, "start"))
    b.append(txt(ox + 16, IC + 12, "position fix + confidence", 9, FAINT,
                 anchor="start"))
    # training-signals lane: scaffolding attached to the box, discarded later
    for lx in (cx - 90, cx + 90):
        b.append(f"<line x1='{lx}' y1='{by + bh}' x2='{lx}' y2='{252}' "
                 f"stroke='{OCH}' stroke-width='1' stroke-dasharray='2 4'/>")
    b.append(txt(cx, 250, "losses · supervision targets · samplers", 10, OCH, 600))
    b.append(txt(cx, 263,
                 "scaffolding that shapes the weights during training — torn "
                 "down before flight, never in the exported model", 9.5, OCH))
    return ("<svg viewBox='0 0 980 290' xmlns='http://www.w3.org/2000/svg' "
            "role='img'>" + "".join(b) + "</svg>")


def paths_status(e, is_current):
    """The one-line verdict under a figure's title.

    Historical rows carry `mission_score` (measured today) where current-era
    rows carry `primary_metric` (measured at the time). Both are the same
    number on the same ruler, but only one of them is what the loop was told,
    so the wording distinguishes them: an old design was kept on ITS metric
    and merely re-scores at this value now."""
    scored_now = "mission_score" in e
    v = e.get("mission_score") if scored_now else e.get("primary_metric")
    if is_rejected(e):
        return "<span class='fail'>rejected</span> — never trained"
    if e.get("provenance") == "gated" or (v is not None and v >= FAIL):
        return "<span class='fail'>gated fail</span> — reverted"
    if v is None:
        return "no score on today's ruler"
    if scored_now:
        verdict = ("<b>kept</b> at the time" if e["kept"]
                   else "discarded at the time")
        return (f"{verdict} — re-scored today: mission score "
                f"<span class='num'>{fmt_score(v)}</span>")
    if e["kept"]:
        cur = " <span class='chip-cur'>current design</span>" if is_current else ""
        return f"<b>kept</b> — new best, mission score <b class='num'>{fmt_score(v)}</b>{cur}"
    return f"discarded — mission score <span class='num'>{fmt_score(v)}</span>, reverted"


def render_paths(exps):
    """gallery/inference-paths.html — every experiment's pre-registered
    architecture figure (the `architecture_svg` each agent must draw before
    training), across every era, as one design record.

    Ordered NEWEST FIRST. Chronological order was right when the page covered
    one era of a dozen figures; across five eras and fifty it buries the
    design that actually works under everything that didn't."""
    hist, eras = load_history()
    era_by_key = {e["key"]: e for e in eras}
    if hist:
        figs = [dict(r, uid=f"{r['era']}-{r['src_id']}",
                     no=f"{r['era_index'] + 1}.{r['src_id']}",
                     anchor=(str(r["src_id"]) if r["era"] == "mission"
                             else f"{r['era']}-{r['src_id']}"))
                for r in hist
                if r["kind"] != "holdout_check"
                and (r["arch_svg"] or "").lstrip().startswith("<svg")]
    else:
        figs = [dict(e, uid=str(e["id"]), no=str(e["id"]), era="", era_index=0,
                     era_label="", anchor=str(e["id"]))
                for e in exps
                if e["kind"] != "holdout_check"
                and (e.get("arch_svg") or "").lstrip().startswith("<svg")]
    n_kept = sum(1 for e in figs if e["kept"])
    current_id = next((e["uid"] for e in reversed(figs) if e["kept"]), None)
    n_eras_with_figs = len({e["era"] for e in figs})
    figs = list(reversed(figs))          # newest first
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    # One dense subtitle, in the lineage page's register — the facts a reader
    # needs before the first figure, and nothing else. This page used to carry
    # three further paragraphs restating them at length.
    paths_sub = (
        f"{len(figs)} model designs, one figure per experiment &mdash; a "
        f"technical drawing of what one camera frame goes through, from pixels "
        f"to <i>(lat,&nbsp;lon,&nbsp;confidence)</i>. Every figure was drawn and "
        f"locked <b>before</b> that experiment trained, alongside a falsifiable "
        f"hypothesis, so what you compare is the design as <i>proposed</i>, not "
        f"redrawn to match a result already known. <b>Newest first</b>, grouped "
        f"by evaluation era: the lineage was wiped at each era boundary, so no "
        f"design descends from one in an earlier band &mdash; several late "
        f"figures are early ideas tried again once the measurement could "
        f"recognise them. Kept and reverted alike; click any figure to enlarge, "
        f"or open its full record in the <a href='index.html'>research log</a>.")

    body = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{social_meta('gallery/inference-paths.html', 'Model designs — Ballpark', 'All 59 agent-drawn model designs in one shared visual language, kept and reverted alike.')}
<title>Model Designs &mdash; Ballpark</title>
<style>{CSS}</style><script>{PATHS_JS}</script></head><body>
{topnav('paths')}
{compute_banner()}
{page_header("Everything we tried: Experiment model designs", paths_sub)}
<div class="paths-wrap">

<div class="pnote scope-note">
<p><b>Who designed these.</b> Not me. An <b>autonomous loop of coding agents</b>
designs, trains and scores one pre-registered experiment at a time, keeping
only what measurably helps; this site is its lab notebook. Every figure below
is an agent's own proposal, drawn before it was allowed to train.</p>
<p><b>The scope they were working in.</b> The project currently runs a
deliberately narrowed configuration: one locale (Berlin), raw daytime imagery
only, no synthetic low-light simulation — trading the full spec's four areas
and six lighting conditions for faster rounds while the loop searches for an
architecture worth generalizing back out. Designs from the earlier eras below
were built against that wider problem, which is part of why they look
heavier.</p>
</div>
<div class="contract-fig" data-ovfig data-title="The frozen contract — where the experiments happen">{contract_svg()}
<p class="contract-cap">The shape every figure on this page shares. The gray
endpoints are the harness's frozen contract — one camera crop in,
one <i>(lat,&nbsp;lon,&nbsp;confidence)</i> answer out — and the dashed box
is the entire search space: each experiment below is one way of filling it.
The ochre lane underneath holds the training signals: the losses, targets
and samplers that shape the weights during training and are torn down before
flight — they never board the aircraft.</p>
</div>
<div class="pkey">
  <span class="k"><span class="sw" style="border-top-color:var(--faint)"></span>frozen contract — harness endpoints</span>
  <span class="k"><span class="sw" style="border-top-color:var(--ink)"></span>the design under test</span>
  <span class="k"><span class="sw" style="border-top-color:var(--accent)"></span>what this experiment changed</span>
  <span class="k"><span class="sw" style="border-top-color:var(--ochre)"></span>training-only — never flies</span>
</div>
<p class="psub num">{len(figs)} proposals · {n_kept} kept ·
{n_eras_with_figs} of {len(eras) or 1} eras drawn · newest first ·
updated {now}</p>"""]

    prev_era = None
    for e in figs:
        # Newest first, so an era heading is emitted when the era CHANGES on
        # the way back through time. Without it, fifty figures from five
        # incompatible eras read as one continuous design lineage, which is
        # the misreading this whole page exists to avoid.
        if e["era"] != prev_era and era_by_key.get(e["era"]):
            era = era_by_key[e["era"]]
            prev_era = e["era"]
            body.append(
                f"<div class='era-head' style='--era-tint:"
                f"{ERA_TINT.get(e['era'], '')}'>"
                f"<span class='era-head-n' style='color:"
                f"{ERA_INK.get(e['era'], '#6b6a60')}'>"
                f"Era {e['era_index'] + 1}</span>"
                f"<span class='era-head-t'>{esc(era['label'])}</span>"
                f"<span class='era-head-s'>optimised {esc(era['ruler'])} · "
                f"asked over {esc(era['eval_set'])}</span></div>")
        try:
            arch = json.loads(e.get("arch_json") or "null")
        except (TypeError, json.JSONDecodeError):
            arch = None
        stages = arch.get("stages") if isinstance(arch, dict) else None
        changed = [s.get("name", "?") for s in (stages or []) if s.get("changed")]
        chg = (f"<span class='chg'>changed: {esc(', '.join(changed))}</span> · "
               if changed else "")
        kept_cls = " kept" if e["kept"] else ""
        eli5 = (f"<p><span class='fig-lead'>In plain words.</span> "
                f"{esc(e['eli5'])}</p>" if e.get("eli5") else "")
        # Ancestry runs within an era only — the lineage was wiped between
        # them, so a chain across a boundary would be a fiction.
        chain = [k["uid"] for k in figs
                 if k["kept"] and k["era"] == e["era"]
                 and k["no"] != e["no"]
                 and (k.get("seq") or 0) < (e.get("seq") or 0)] + [e["uid"]]
        body.append(f"""<section class="fig-entry{kept_cls}" id="e{e['uid']}" data-id="{e['uid']}" data-chain="{','.join(chain)}">
<div class="fig-head">
  <span class="fig-no num">Fig. {e['no']}</span>
  <span class="fig-title">{esc(e['title'])}</span>
  <span class="fig-status">{paths_status(e, e['uid'] == current_id)}</span>
</div>
<div class="fig-svg" data-ovfig data-id="{e['uid']}" data-chain="{','.join(chain)}" data-no="Fig. {e['no']}" data-title="{esc(e['title'])}">{e['arch_svg']}</div>
<div class="fig-cap">
{eli5}
<p class="fig-meta">{esc(e['category'] or '—')} · {esc(e['init_strategy'] or '—')}
· {chg}{esc((e['ts'] or '')[:10])} · commit
<span class="mono">{esc((e['git_commit'] or '')[:8])}</span> ·
<a href="index.html#r{e['anchor']}">full experiment record →</a></p>
</div>
</section>""")

    body.append("""</div>
""" + OVERLAY_HTML + credits_html() + "</body></html>")
    PATHS_OUT.parent.mkdir(exist_ok=True)
    PATHS_OUT.write_text("\n".join(body))
    print(f"wrote {PATHS_OUT} ({len(figs)} figures)")


ERAS = [{'d': '20 Jul', 't': '17:05', 'kind': 'start', 'title': 'Bootstrap — and two course corrections before a single experiment ran', 'trig': 'A spec, an empty repo, and a MacBook Air.', 'body': "The harness shipped on Sentinel-2 imagery at 10&nbsp;m/px. Within twenty minutes that was rejected and the frozen fetch stage rewritten for 1&nbsp;m/px open orthophotos — a 50&times; finer ground resolution, re-frozen before any research began. Hours later, looking at the six lighting renders, the human spotted that five of them were effectively identical: the simulated auto-exposure was re-brightening every bucket under its gain ceiling. The metric's &ldquo;six lighting conditions&rdquo; were very nearly a fiction. The loop was stopped and the relighting rebuilt.", 'q': 'Assume the UAV is at 100m altitude &mdash; find better satellite imagery that does NOT widen the sim-to-real gap', 'q2': 'honestly, all times other than night look the same to me', 'state': '7 bootstrap experiments, archived. Baseline 3,216.74&nbsp;m.'}, {'d': '20 Jul', 't': '20:35', 'kind': 'science', 'title': 'The human writes the anti-plateau rule himself', 'trig': 'The first loop runs flatten out.', 'body': 'Watching early experiments circle the same mechanism, he dictated the rule that would govern the loop for the next ten days — and which later hardened into code after five failed attempts to enforce it by prompt alone.', 'q': 'If three or more consecutive experiments were reverted, do not attempt another variation of the last refuted mechanism.', 'state': 'Loop running unattended overnight on a laptop.'}, {'d': '21 Jul', 't': '07:58', 'kind': 'infra', 'title': 'Rented GPU — decided in eleven minutes', 'trig': 'Experiments taking 20&ndash;50 minutes each on a MacBook Air.', 'body': "Not thermals, not a training failure — wall-clock per experiment. $200 went onto a RunPod account eleven minutes after the question was asked, and the <i>entire</i> loop moved to the pod, headless agents included. The pod's database was declared production, one-directional: local runs became testing only.", 'q': "I've added $200 to my runpod account.", 'state': 'Repo private on GitHub. Loop runs 24/7.'}, {'d': '21 Jul', 't': '12:11', 'kind': 'scope', 'title': 'The project goes public', 'trig': 'A decision to publish the research trail, not just the result.', 'body': "GitHub Pages, a homepage entry, and a title chosen from a shortlist. The same day he set the north star that still governs design decisions: that anyone should be able to draw a box and generate their own tiny, self-contained model, freely. He also stress-tested §6's premise — <i>how should it know about held-out spots inside Berlin?</i> — and let it stand. That question turned out to be the right one, ten days early.", 'q': 'I like &lsquo;Think globally, memorize locally&rsquo; a lot. And &lsquo;Not all who wander are lost&rsquo;.', 'state': 'Public site live: overview, research log, model designs, lineage.'}, {'d': '22 Jul', 't': '00:50', 'kind': 'infra', 'title': 'Infrastructure failures recorded as science', 'trig': 'A frozen site at one in the morning.', 'body': "Experiment 17's training code had been caching ~5.7&nbsp;GB of renders <i>per iteration</i> and the harness was committing them to Git LFS. The pod's 50&nbsp;GB volume filled at 23:34. Two experiments recorded as &ldquo;gated fail&rdquo; had not failed on their merits at all — they were disk crashes written into the research record as results. Later the same day GitHub Actions ran out of disk on 1.45&nbsp;GB of leftover debug PNGs.", 'q': 'Okay, the last 24h have been bumpy to say the least. Experiments break, commits can&rsquo;t get pushed to origin, now this.', 'state': '~40&nbsp;GB reclaimed. 25&nbsp;MB commit guard added. Hourly health check created.'}, {'d': '22 Jul', 't': '11:59', 'kind': 'science', 'title': 'Five attempts to make a pivot mean something', 'trig': 'The loop claiming to pivot while still building on MobileNetV3.', 'body': "A demanded pivot kept returning the champion's backbone with different machinery bolted around it — experiments 37, 38, 40. Four prompt-level fixes failed. What finally worked was enforcement in code: <code>backbonecheck.py</code>, which fingerprints backbone identity from the <i>post-implementation source</i> and rejects before training. The lesson generalised: prompt instructions are not guarantees.", 'q': 'That&rsquo;s not a PIVOT but a continuation.', 'q2': 'fuck this. fuck you. stop the current loop. &hellip; make the next experiment a fucking pivot', 'state': 'Champion 743&nbsp;m, unmoved. Trust in the harness at its lowest.'}, {'d': '23 Jul', 't': '13:00', 'kind': 'infra', 'title': 'Off RunPod — for process, not compute', 'trig': 'Three days of stuck commits, merge conflicts and platform friction.', 'body': "The decisive reframe came later that afternoon: the pain had never been RunPod, it was <b>two git writers on one branch</b> — the pod committing experiments while the laptop committed harness and docs. RunPod's last act was a stopped pod that could not resume because no GPU was free. Modal was chosen specifically because a <i>stateless</i> trainer that never touches git makes the two-writer problem impossible by construction.", 'q': 'It&rsquo;s been a fucking nightmare. I don&rsquo;t care about the compute.', 'state': 'Laptop = single writer. Modal A100 = stateless trainer, $30 credits.'}, {'d': '23 Jul', 't': '13:33', 'kind': 'scope', 'title': 'Scope narrowed to Berlin — and restored six hours later', 'trig': 'Local M1 training too slow for four areas.', 'body': 'Berlin-only, figures off, holdout skipped — a deliberate cut to buy iteration speed. But <code>state/best.json</code> did not follow the change, so for a while the loop was scoring four-area results against a Berlin-only yardstick. Once Modal made four areas cheap again, the cut was reversed and 743&nbsp;m restored as the number to beat.', 'q': 'I&rsquo;d rather see us move faster.', 'state': 'Back to the four-area §6 metric by 20:17.'}, {'d': '24&ndash;29 Jul', 't': '', 'kind': 'silence', 'title': 'Six days of silence — an abandonment, not a pause', 'trig': 'A training call that died at 00:25 and nobody noticed.', 'body': 'All four Modal A100 calls returned <i>cancelled by user or a failure</i> twenty-seven minutes in. The loop stopped. Git shows zero commits for six days; the only interaction in the entire window is a single <code>/compact</code> on the 29th. Read from the commit history alone this looks like a quiet week. It was a silent failure nobody was watching for.', 'q': 'I believe I cut that off because it took longer than expected.', 'state': 'Frozen: champion 743&nbsp;m, 54 experiments, 21 consecutive failures.'}, {'d': '30 Jul', 't': '18:26', 'kind': 'science', 'title': 'Attacking the premises instead of the hyperparameters', 'trig': 'A plateau six days old and a cold restart.', 'body': "Within three hours: a literature sweep asking whether the problem was already solved (no published system matches the constraint set — no on-board reference gallery, one compact model, km-scale); an offer to abandon the project's most-defended rule; and a 3D-shadow relighting idea, raised and parked. The constraint was then closed <i>empirically</i>: a retrieval-index probe ran three rounds and scored ~2,700&nbsp;m against the champion's 743&nbsp;m. Parked, not disproven — it never ran at real training scale.", 'q': 'Maybe the platonic idea of &lsquo;everything is in the weights&rsquo; is not worth keeping.', 'state': 'Four experiments, four reverts. Modal credits exhausted at 23:00.'}, {'d': '30 Jul', 't': '20:23', 'kind': 'infra', 'title': 'The real bottleneck was a PNG', 'trig': '&ldquo;Why does one experiment still take an hour?&rdquo;', 'body': 'Profiling found each ~120&nbsp;MB lighting render being decoded from disk every single epoch: <b>78% of training wall-clock</b>, against 14.5% for the actual forward and backward passes. Two prior theories — batch transfer and GPU compute — were wrong by an order of magnitude. Caching the decode cut training from ~40 to ~26 minutes.', 'q': 'yup, let&rsquo;s instrument to better understand what&rsquo;s taking so long.', 'state': "The evening's two biggest wins were both infrastructure. The metric did not move."}, {'d': '31 Jul', 't': '09:00', 'kind': 'scope', 'title': 'berlin-slim — trading scope for iteration speed', 'trig': 'Sixty-three experiments, still ~37&times; from target.', 'body': "A deliberate fork: one locale, daytime imagery only, no synthetic relighting, figures off, pivot gate off. The framing was explicit — <i>what if overfitting is a feature?</i> Nail one city first, worry later about whether the technique generalises. The milestone moved from 20&nbsp;m to 100&nbsp;m, derived from the champion's own Berlin-daytime numbers rather than picked.", 'q': 'I think we&rsquo;ve over-scoped and over-engineered the whole thing.', 'state': 'Fresh lineage. Baseline re-measured.'}, {'d': '31 Jul', 't': '11:20', 'kind': 'measure', 'title': 'The evaluation had been asking an impossible question', 'trig': 'One roundabout, surrounded by red dots.', 'body': 'Why was the Gro&szlig;er Stern — a seven-way star that exists once in Berlin — never located? Because <b>zero</b> training crops contained it and <b>twelve</b> eval questions did: it sits three pixels from an eval block, and the anti-leakage buffer had deleted every training view. Measured across the city: <b>28.2%</b> of Berlin appeared in no training crop, <b>100%</b> of eval questions stood on never-seen ground, and 65% of eval frames contained not one familiar pixel. For a system whose job is to memorise one city, that is unanswerable by construction. The split was rebuilt to hold out <i>viewpoints</i>, not <i>regions</i>.', 'q': 'a clear star-like circular roundabout &hellip; should score higher, no?', 'state': 'Training coverage 71.8% &rarr; 93.5%. Eval on seen ground 0% &rarr; 100%.'}, {'d': '31 Jul', 't': '13:10', 'kind': 'measure', 'title': 'The score rewarded guessing over learning — twice', 'trig': 'A map full of green dots, discarded; a map with one, kept.', 'body': 'A model that guesses the map centre has a tight, mediocre error distribution and a respectable <i>median</i>. A model that genuinely memorises is bimodal — some places nailed, others badly missed — and scores <i>worse</i> on a median. The metric had been reverting the only experiment that had begun to work. Replacing it with a geometric mean fixed the symptom, but that was still a proxy chosen to make progress visible rather than to state what the aircraft needs. The score is now the product requirement itself: usable fixes minus dangerous ones.', 'q': 'There is ONLY the product requirement.', 'state': 'Every <i>verdict</i> before today is void &mdash; 76 experiments judged on a broken ruler. The experiments themselves were later re-measured and put back on the record.'}, {'d': '31 Jul', 't': '13:40', 'kind': 'measure', 'title': 'And the harness had been deleting its own work', 'trig': 'The corrected metric promoting a previously-reverted experiment to champion.', 'body': 'Its implementation no longer existed. Reverting an experiment ran <code>git checkout -- model/</code>, discarding the source outright; only kept experiments ever entered the git trail. Safe only if the metric is correct and permanent — an assumption nobody had written down, and which had just been disproved twice in one day. Of sixty archived experiments, thirteen survive.', 'q': 'We can&rsquo;t LOSE our fucking experiment implementations EVER.', 'state': 'All four code-discarding paths now snapshot first.'}, {'d': '31 Jul', 't': '14:00', 'kind': 'result', 'title': 'With honest instruments, the approach works', 'trig': 'The first experiment measured on a correct evaluation and a correct metric.', 'body': 'Full-lattice coverage training — the same architecture the previous metric had thrown away as a failure — took the mission score from <b>2.001</b> (confident and wrong on every single frame, the worst value the scale allows) to <b>0.183</b>, and two further experiments took it to <b>0.040</b>. <b>96.5%</b> of camera frames now yield a usable fix and <b>0.5%</b> are confidently wrong; the typical miss when it answers is 27&nbsp;m, comfortably inside the 100&nbsp;m target. The region-holdout diagnostic reads 0% usable, confirming this is genuine memorisation rather than generalisation — exactly the design intent.', 'q': '', 'state': 'The plateau was never the task being hard. The instruments were pointed at the wrong thing. Re-measured on the corrected ones, the best of the previous 76 experiments scores 1.311 &mdash; worse than saying nothing at all.'}]

KIND = {'start': ('#4a473e', 'bootstrap'), 'infra': ('#8a6a1e', 'infrastructure'), 'science': ('#2f5d7c', 'research direction'), 'scope': ('#5b6e4a', 'scope'), 'silence': ('#9b998c', 'silence'), 'measure': ('#8c2f1f', 'measurement bug'), 'result': ('#3c9c3c', 'result')}

# The evolution figure's marks now live in autoresearch/evofig.py, which both
# holds the data AND lays it out. Three lists used to sit here looking like
# the source of truth while the hand-edited SVG literal below had moved past
# them -- SUPER held five entries while the figure drew six. Removed rather
# than fixed: two places to edit is how they drifted apart.

EVOLUTION_CSS = """
.evo{margin:0;padding:0 0 80px}
.evo-prose{max-width:980px;margin:0 auto;padding:0 16px}
/* unlimited horizontal space, scrolled — same idiom as the lineage diagram */
.evo-figure{overflow-x:auto;overflow-y:hidden;padding:8px 0 14px;
  scrollbar-width:thin;scrollbar-color:var(--rule) transparent}
.evo-figure::-webkit-scrollbar{height:8px}
.evo-figure::-webkit-scrollbar-thumb{background:var(--rule);border-radius:4px}
.evo-scrollhint{text-align:center;font:600 10px var(--serif);
  text-transform:uppercase;letter-spacing:.09em;color:var(--faint);margin:0 0 14px}
.evo-tick{stroke:var(--rule);stroke-width:1}
#evo-svg{display:block;height:auto}
.evo-branch{fill:none;stroke:var(--ink);stroke-width:3}
.evo-dormant{stroke:var(--faint);stroke-width:2;stroke-dasharray:5 6;opacity:.65}
.evo-dormantlab{fill:var(--faint);font:italic 10.5px var(--serif)}
.evo-era{pointer-events:none}
.evo-erabound{stroke:var(--rule);stroke-width:1}
.evo-eralab{font:600 11px var(--serif);font-feature-settings:"smcp" 1;
  letter-spacing:.06em;stroke:none;pointer-events:none}
.evo-grid{stroke:var(--rule-soft);stroke-width:1}
.evo-day{fill:var(--faint);font:600 9.5px var(--serif);letter-spacing:.08em;
  text-transform:uppercase;text-anchor:middle}
.evo-trunk{stroke:#111111;stroke-width:2.5;stroke-linecap:round}
.evo-node{fill:#111111;stroke:var(--paper);stroke-width:2;cursor:pointer}
.evo-tlab{fill:var(--ink);font:600 10.5px var(--serif);text-anchor:middle;cursor:pointer}
.evo-dead{fill:none;stroke:#9b998c;stroke-width:2;stroke-dasharray:5 4}
.evo-end line{stroke:#9b998c;stroke-width:2;stroke-linecap:round}
.evo-super{fill:none;stroke:#8c2f1f;stroke-width:2}
.evo-endr{fill:var(--paper);stroke:#8c2f1f;stroke-width:2}
/* An OPEN branch: ran, produced a real number, and is still live -- neither
   abandoned (grey ×) nor superseded (red ○). Ink, because it belongs to the
   living trunk; dashed and hollow-tipped, because it has not resolved. */
.evo-open{fill:none;stroke:#111111;stroke-width:2;stroke-dasharray:6 4;opacity:.85}
.evo-openend{fill:var(--paper);stroke:#111111;stroke-width:2}
.evo-rejoin{fill:none;stroke:#8c2f1f;stroke-width:1.2;stroke-dasharray:2 3;opacity:.6}
.evo-lab{font:600 10.5px var(--serif);cursor:pointer}
.evo-lab.d{fill:#6b6a60}
.evo-lab.s{fill:#8c2f1f}
.evo-lab.o{fill:var(--ink)}
.evo-endlab{font:600 9.5px var(--serif);letter-spacing:.02em;pointer-events:none}
.evo-endlab.s{fill:#8c2f1f}
.evo-endlab.d{fill:#6b6a60}
.evo-endlab.o{fill:var(--ink)}
.evo-pin{stroke:#8c2f1f;stroke-width:1;stroke-dasharray:2 2}
.evo-pindot{fill:#8c2f1f;cursor:pointer}
.evo-pinlab{fill:#8c2f1f;font:600 9px var(--serif);cursor:pointer;
  text-transform:uppercase;letter-spacing:.05em}
.evo-merge{fill:none;stroke:#111111;stroke-width:2}
.evo-arrow{fill:#111111}
.evo-mstart{fill:var(--paper);stroke:#111111;stroke-width:2}
.evo-mlab{fill:var(--ink);font:600 10.5px var(--serif);cursor:pointer;font-style:italic}
.evo-gap{fill:var(--paper);stroke:var(--rule);stroke-width:1;stroke-dasharray:3 3}
.evo-gaplab{fill:var(--faint);font:600 10px var(--serif);text-anchor:middle;
  letter-spacing:.07em;text-transform:uppercase}
.evo-svg-dim [data-k]:not(.on){opacity:.22}
.evo-key{display:flex;flex-wrap:wrap;gap:10px 26px;justify-content:center;
  margin:8px 0 22px;font-size:12px;color:var(--muted)}
.evo-key .k{display:inline-flex;align-items:center;gap:8px}
.evo-read{max-width:780px;margin:0 auto 26px;font-size:14px;line-height:1.7;
  color:#4a473e;text-align:center}
.evo-read b{color:var(--ink)}
#evo-tip{position:fixed;z-index:60;max-width:390px;background:var(--paper);
  border:1px solid var(--rule);box-shadow:0 6px 22px rgba(0,0,0,.13);
  padding:12px 15px;font-size:13px;line-height:1.6;color:#4a473e;
  opacity:0;pointer-events:none;transition:opacity .12s}
#evo-tip .tt{font:600 13.5px var(--serif);color:var(--ink);display:block;margin-bottom:5px}
#evo-tip .tk{font:600 9.5px var(--serif);text-transform:uppercase;
  letter-spacing:.09em;display:block;margin-bottom:7px}
#evo-tip .tk.kept{color:#111111}
#evo-tip .tk.abandoned{color:#6b6a60}
#evo-tip .tk.superseded{color:#8c2f1f}
#evo-tip .tk.open{color:#111111}
#evo-tip .tk.silence{color:var(--faint)}
"""

# Generated from data (autoresearch/evofig.py), not a pasted literal.
EVOLUTION_SVG = evofig.build_svg()

EVOLUTION_TIPS = r"""{"gap": {"t": "six days of silence", "d": "The loop had died at 00:25 on the 24th and nobody was watching. Git shows a quiet week; the transcripts show an abandonment. The only interaction in the whole window is a single /compact on the 29th.", "k": "silence"}, "d0": {"t": "Sentinel-2 10 m/px", "d": "The bootstrap's imagery source - too coarse for a 100 m-altitude camera footprint. Abandoned inside 20 minutes.", "k": "abandoned"}, "d1": {"t": "Prompt-only pivot rules", "d": "Four attempts to make 'pivot' mean something by instruction alone. Experiments 37, 38 and 40 all came back on the same backbone.", "k": "abandoned"}, "d2": {"t": "3D shadow relighting", "d": "Simulate real sun angles from 3D terrain instead of flat imagery. Raised and parked the same evening; still open.", "k": "abandoned"}, "d3": {"t": "Retrieval-index probe", "d": "Shipping a few MB of reference index - the project's most-defended constraint - was briefly on the table. Three rounds, ~2,700 m against the champion's 743 m. Parked, not disproven.", "k": "abandoned"}, "s0": {"t": "Region-holdout evaluation", "d": "In force for TEN of the project's twelve days. It left 28.2% of Berlin in no training crop and put 100% of test questions on ground the model had never seen - unanswerable for a memorisation system. Every result measured against it is void.", "k": "superseded"}, "s1": {"t": "Median error as the score", "d": "Rewards a model that guesses the map centre over one that genuinely memorises. It was caught reverting the only experiment that had begun to work.", "k": "superseded"}, "s2": {"t": "RunPod - rented 4090 pod", "d": "$200 committed eleven minutes after the question was asked; the entire loop moved onto the pod, headless agents included. Abandoned three days later - not for the GPU, for the process friction around it.", "k": "superseded"}, "s3": {"t": "Modal A100 - serverless", "d": "Chosen because a STATELESS trainer that never touches git makes the two-writer problem impossible by construction. $30 of credits, four areas fanned out in parallel, ~26 min/experiment after the decode fix. Ran until the credits ran out.", "k": "superseded"}, "s4": {"t": "Berlin-only scope cut", "d": "Narrowed for six hours to survive on an M1 - but state/best.json didn't follow, so the loop briefly scored four-area results against a Berlin-only yardstick.", "k": "superseded"}, "s5": {"t": "Geometric mean", "d": "Fixed the median's symptom but was still a research proxy, chosen to make progress visible rather than to state what the aircraft needs. Lasted two hours.", "k": "superseded"}, "i0": {"t": "pod disk full", "d": "A 5.7 GB-per-iteration render cache was being committed to Git LFS; the pod's 50 GB volume filled at 23:34. Two experiments recorded as 'gated fail' had not failed on their merits - they were disk crashes written into the research record as results.", "k": "incident"}, "i1": {"t": "Fable hits its cap", "d": "The design model hit 100% of its weekly quota. Three iterations burned 30-minute retry sleeps against a wall before anyone noticed; the design step was switched to Sonnet.", "k": "incident"}, "i2": {"t": "CI out of disk", "d": "GitHub Actions died on 1.45 GB of leftover debug PNGs - over half the repo's live LFS payload, a forgotten dump from early holdout work.", "k": "incident"}, "i3": {"t": "two writers on main", "d": "The pod and the laptop both committing to main all day: repeated rebase dances, and at least one merge that silently reverted the morning's pivot-enforcement work.", "k": "incident"}, "i4": {"t": "pod won't resume", "d": "RunPod's last act: a stopped pod could not restart because the host had no free GPU. Stopped pods don't reserve one. This is what turned a compromise into a full switch.", "k": "incident"}, "i5": {"t": "loop dies unnoticed", "d": "All four Modal calls returned 'cancelled by user or a failure' 27 minutes in, at 00:25. Nobody noticed for six days.", "k": "incident"}, "i6": {"t": "credits exhausted", "d": "$30 of Modal credits gone; the loop stopped two experiments short of its target and compute returned to the laptop.", "k": "incident"}, "m0": {"t": "enforce it in the source, not the prompt", "d": "Four rounds of instructing the loop to pivot had failed. The insight was that a prompt is not a guarantee: the check had to read the code that actually resulted. backbonecheck.py fingerprints the post-implementation source and rejects before training.", "k": "insight"}, "m1": {"t": "the problem is two writers, not the platform", "d": "Three days were spent blaming RunPod. The actual cause was the pod and the laptop both committing to main. Once that was named, the fix was a rule rather than a provider: one git writer, and a remote that only trains.", "k": "insight"}, "m2": {"t": "stop guessing, profile it", "d": "Two theories about the training bottleneck - batch transfer and GPU compute - were both wrong by an order of magnitude. Instrumenting it found a 120 MB PNG being re-decoded every epoch, 78% of wall-clock.", "k": "insight"}, "m3": {"t": "it is never shown the places it is tested on", "d": "One question about one roundabout: zero training crops contained the Grosser Stern, twelve eval questions did. Measured across the city, 100% of test questions stood on ground the model had never seen. The split was rebuilt around held-out viewpoints.", "k": "insight"}, "m4": {"t": "the score must BE the product requirement", "d": "A map full of green hits had been discarded while one with a single hit was kept. Median rewarded guessing the map centre; the geometric mean that replaced it was still a proxy. The score became what the aircraft actually needs: usable fixes minus dangerous ones.", "k": "insight"}, "m5": {"t": "it was starved, not refuted", "d": "The map-cell architecture had gate-failed at 84 seconds of training, seeing a fraction of the city. Given full-lattice coverage - the same architecture, nothing else changed - it took the mission score from 2.001 to 0.183.", "k": "insight"}, "t0": {"t": "Bootstrap", "d": "Spec, empty repo, MacBook Air. Frozen pipeline, naive baseline and the whole harness in one session.", "k": "kept"}, "t1": {"t": "1 m/px orthophotos", "d": "Sentinel-2 at 10 m/px rejected within 20 minutes; the fetch stage re-frozen 50x finer before any experiment ran.", "k": "kept"}, "t2": {"t": "Relighting rebuilt", "d": "Five of six lighting buckets were effectively identical - auto-exposure was re-brightening them all. Caught by eye, not by a test.", "k": "kept"}, "t3": {"t": "Goes public", "d": "GitHub Pages, the research trail published as it runs. North star set: anyone draws a box and gets their own model, freely.", "k": "kept"}, "t4": {"t": "Pivot enforced in code", "d": "Four prompt-level attempts failed to stop the loop rebuilding on the same backbone. backbonecheck.py fingerprints the post-implementation source instead.", "k": "kept"}, "t5": {"t": "One git writer", "d": "The rule that ended three days of chaos: the laptop commits, the remote only trains. Divergence becomes impossible by construction.", "k": "kept"}, "t6": {"t": "PNG decode fix", "d": "Each ~120 MB render was re-decoded every epoch: 78% of training wall-clock against 14.5% for the actual maths.", "k": "kept"}, "t7": {"t": "berlin-slim branch", "d": "Deliberate narrowing to buy iteration speed. What if overfitting is the feature?", "k": "kept"}, "t8": {"t": "0.040 - 96.5% usable", "d": "Four experiments on a corrected evaluation and a metric that states the product requirement. 96.5% of held-out frames give a usable fix, 0.5% are confidently wrong, median miss 27 m. The first of the four was the architecture the old metric had discarded as a failure.", "k": "kept"}, "t9": {"t": "back to berlin-slim", "d": "The rural detour rewound to the commit before it, and the trunk is the berlin-slim line again: model/, state/best.json and the experiment database all back to the 0.040 champion, with the Prignitz rows removed. Nothing about the detour failed on its merits - the decision was to keep pushing one question rather than run two.", "k": "kept"}, "d4": {"t": "Prignitz probe - 0.113", "d": "The Berlin champion, unchanged, trained and scored on the rural box - no code change needed at all, the cell grid comes from the raster. It scored 0.113 against Berlin's 0.040, and the loss was not where it was expected: usable fixes barely moved (92.5% vs 96.5%) and abstention barely moved (3.8% vs 3.0%), while FALSE fixes rose 7.5x, 0.5% to 3.75% - the one outcome the mission score treats as worse than silence. Fields produce confidently wrong answers where city blocks do not. That result opened a rural era at 22:26 the same evening, which ran three experiments in four hours: a native seed reproducing 0.113, a rotation-consensus decode that made false fixes WORSE (0.123, refuted - the rural confusions are rotation-stable), and a third designed but never trained. On 1 August the era was rewound to the commit before it and its rows removed from the database, to keep pushing Berlin rather than open a second front. The measurement stands; the era does not. Raw artifacts at git tag prignitz-detour-record.", "k": "abandoned"}}"""

EVOLUTION_JS = r"""
(function(){
  var tips={}; try{tips=JSON.parse(document.getElementById('evo-tips').textContent||'{}')}catch(e){}
  var svg=document.getElementById('evo-svg'), tip=document.getElementById('evo-tip');
  if(!svg||!tip) return;
  var KIND={kept:'on the trunk \u2014 this survived',abandoned:'tried and abandoned',
            superseded:'ran as the mainline, then replaced',silence:'nothing happened',
            open:'branched off \u2014 still open'};
  function show(k,ev){
    var d=tips[k]; if(!d) return;
    tip.innerHTML="<span class='tk "+d.k+"'>"+(KIND[d.k]||d.k)+"</span>"+
                  "<span class='tt'>"+d.t+"</span>"+d.d;
    tip.style.opacity=1;
    var x=Math.min(ev.clientX+16,window.innerWidth-tip.offsetWidth-10);
    var y=Math.min(ev.clientY+14,window.innerHeight-tip.offsetHeight-10);
    tip.style.left=x+'px'; tip.style.top=Math.max(8,y)+'px';
    svg.classList.add('evo-svg-dim');
    svg.querySelectorAll("[data-k='"+k+"']").forEach(function(n){n.classList.add('on')});
  }
  function hide(){
    tip.style.opacity=0; svg.classList.remove('evo-svg-dim');
    svg.querySelectorAll('.on').forEach(function(n){n.classList.remove('on')});
  }
  svg.querySelectorAll('[data-k]').forEach(function(n){
    n.addEventListener('mousemove',function(ev){show(n.getAttribute('data-k'),ev)});
    n.addEventListener('mouseleave',hide);
  });
  svg.addEventListener('mouseleave',hide);

  // Open by travelling to the newest work: start at the beginning, then
  // animate right so the arc of the project is visible on the way past.
  // (Setting the position instantly first made the smooth call a no-op —
  // there was nothing left to animate.)
  var fig=document.querySelector('.evo-figure');
  if(fig){
    var LATEST=4589.7;
    var target=function(){
      return Math.max(0,Math.min(LATEST-fig.clientWidth/2,
                                 fig.scrollWidth-fig.clientWidth));
    };
    fig.scrollLeft=0;
    var glide=function(){
      var t=target();
      if(t<=0) return;
      try{ fig.scrollTo({left:t,behavior:'smooth'}); }
      catch(e){ fig.scrollLeft=t; }
    };
    if('requestAnimationFrame' in window)
      requestAnimationFrame(function(){ setTimeout(glide,450); });
    else setTimeout(glide,450);
  }
})();
"""


EVOLUTION_OUT = REPO_ROOT / "gallery" / "research-evolution.html"


def evolution_era_bands(eras):
    """Era band rects for the hand-authored evolution SVG.

    That figure is a literal, generated once, with time on x — and the x scale
    is deliberately NON-linear: idle days are compressed so the busy ones get
    room. Rather than reproduce that scale (and drift from it the first time
    the figure is regenerated), read it back out of the figure itself: every
    day gridline in the SVG carries its own x and its own label, which is a
    complete description of the mapping. Each calendar day then gets its own
    linear scale between its neighbouring gridlines, and 20 July — which has
    no gridline because it starts off-canvas — is extrapolated from 21 July's.

    Returns an SVG fragment to splice in immediately after the opening <svg>
    tag, so the bands sit behind every mark in the figure."""
    grid = re.findall(r"<line class='evo-grid' x1='([\d.]+)'.*?"
                      r"<text class='evo-day' x='[\d.]+' y='\d+'>(\d+) Jul</text>",
                      EVOLUTION_SVG)
    if not grid or not eras:
        return ""
    day_x = {int(d): float(x) for x, d in grid}
    days = sorted(day_x)
    # viewBox width, so bands can be clamped to the canvas.
    m = re.search(r"viewBox='0 0 ([\d.]+) ([\d.]+)'", EVOLUTION_SVG)
    if not m:
        return ""
    VW, VH = float(m.group(1)), float(m.group(2))

    def x_of(day_float):
        """Day-of-July (fractional) -> x, piecewise-linear between gridlines."""
        lo = max([d for d in days if d <= day_float], default=days[0])
        hi = min([d for d in days if d > day_float], default=None)
        if hi is None:                      # past the last gridline
            lo2 = days[-2] if len(days) > 1 else days[-1] - 1
            per = (day_x[days[-1]] - day_x[lo2]) / max(days[-1] - lo2, 1)
            return day_x[days[-1]] + (day_float - days[-1]) * per
        if day_float < days[0]:             # 20 July, off the left edge
            per = day_x[days[1]] - day_x[days[0]] if len(days) > 1 else 300.0
            return day_x[days[0]] - (days[0] - day_float) * per
        span = hi - lo
        return day_x[lo] + (day_float - lo) * (day_x[hi] - day_x[lo]) / span

    def to_day(ts):
        # ISO timestamps, all inside July 2026 — day + fraction of day.
        try:
            t = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None
        return t.day + (t.hour * 3600 + t.minute * 60 + t.second) / 86400.0

    # The canvas is wider than the drawing (the figure leaves scroll room past
    # the last node). Ending the final band at the viewBox edge would paint a
    # long stretch of empty canvas as though work happened there, so it stops
    # where the drawing does.
    #
    # Read this from evofig, NOT by scraping the `evo-trunk` line: since the
    # fork became part of layout, that line stops at the SPLIT (4362.6) and
    # what continues to the right is the dormant-main line and the fork lane.
    # Scraping it silently cut the last era band 207 px short of the work it
    # was supposed to cover.
    content_end = min(VW, evofig.TRUNK_X1 + 60)

    parts, prev_row = [], 1
    for i, era in enumerate(eras):
        d0, d1 = to_day(era["ts_start"]), to_day(era["ts_end"])
        if d0 is None or d1 is None:
            continue
        # An era runs until the next one starts, so its band should too —
        # otherwise the gaps between eras read as unaccounted-for time.
        if i + 1 < len(eras):
            nxt = to_day(eras[i + 1]["ts_start"])
            if nxt is not None:
                d1 = nxt
        lo = max(0.0, x_of(d0) if i else 0.0)
        hi = min(content_end, x_of(d1) if i + 1 < len(eras) else content_end)
        if hi - lo < 1:
            continue
        key = era["key"]
        ink = ERA_INK.get(key, "#6b6a60")
        label = ERA_SHORT.get(key, era["label"])
        parts.append(f"<rect class='evo-era' x='{lo:.1f}' y='0' "
                     f"width='{hi-lo:.1f}' height='{VH:.0f}' "
                     f"fill='{ERA_TINT.get(key, 'rgba(107,106,96,.06)')}'/>")
        if i:
            parts.append(f"<line class='evo-erabound' x1='{lo:.1f}' y1='0' "
                         f"x2='{lo:.1f}' y2='{VH:.0f}'/>")
        # Captions sit low, under the trunk labels, where the figure is empty —
        # the top strip already carries the day gridline labels. The last three
        # eras are hours wide against the first two's days, so their bands are
        # far narrower than their names: those get staggered onto two rows with
        # a leader down to the band, rather than overprinting each other.
        cx, wid = (lo + hi) / 2, hi - lo
        if wid >= len(label) * 6.4:
            parts.append(f"<text class='evo-eralab' x='{cx:.1f}' y='{VH-10:.0f}' "
                         f"text-anchor='middle' fill='{ink}'>{esc(label)}</text>")
        else:
            prev_row = 0 if prev_row else 1
            ly = VH - 10 - 16 * prev_row
            parts.append(f"<line class='evo-erabound' x1='{cx:.1f}' y1='{ly+4:.0f}' "
                         f"x2='{cx:.1f}' y2='{VH-6:.0f}' stroke='{ink}'/>")
            parts.append(f"<text class='evo-eralab' x='{cx:.1f}' y='{ly:.0f}' "
                         f"text-anchor='middle' fill='{ink}'>{esc(label)}</text>")
    return "".join(parts)


# berlin-slim was a fork of the research framework itself, not another step
# along it: one locale, daytime only, relighting disabled, figures and pivot
# gate off. Everything that followed — the viewpoint evaluation, the mission
# score, the result — happened ON that fork; main has not moved since. Drawing
# it as trunk absorbed the fork into the mainline and quietly claimed the
# result for a branch that never produced it.
# The fork -- berlin-slim lifted onto its own lane -- used to be eight
# find-and-replace edits applied to the SVG literal after the fact. It is
# now part of layout in evofig.build_svg(), so a new mark can simply ask
# which line is live at its moment instead of inheriting the wrong one.

def _apply(svg: str, subs) -> str:
    for a, b in subs:
        if a not in svg:
            print(f"  evolution fork: pattern not found — {a[:70]}")
            continue
        svg = svg.replace(a, b, 1)
    return svg


def render_evolution(exps):
    """gallery/research-evolution.html — the research PROCESS as a graph, in
    the same idiom as the experiment lineage: a trunk of what survived, spurs
    ABOVE for directions tried and abandoned, spurs BELOW for approaches that
    ran AS the mainline before being replaced. x is real time.

    Deliberately not derived from experiments.sqlite — the turns that matter
    (why compute moved twice, why the evaluation was wrong for ten days) never
    appear as experiment rows. Reconstructed from the project's own
    conversation transcripts and git history.

    Colour is validated for CVD separation (#111111/#8c2f1f/#9b998c: worst
    all-pairs deltaE 20.6 protan, 27.1 normal), and every branch also carries a
    text label and a distinct end marker, so identity is never colour-alone.
    """
    key = (
      "<span class='k'><svg width='36' height='12'><line x1='0' y1='6' x2='36' y2='6' "
      "stroke='#111111' stroke-width='2.5'/><circle cx='18' cy='6' r='4.5' fill='#111111'/>"
      "</svg><b>the trunk</b> &mdash; what the project still does today</span>"
      "<span class='k'><svg width='36' height='12'><line x1='0' y1='6' x2='24' y2='6' "
      "stroke='#9b998c' stroke-width='2' stroke-dasharray='5 4'/>"
      "<line x1='27' y1='2' x2='34' y2='10' stroke='#9b998c' stroke-width='2'/>"
      "<line x1='27' y1='10' x2='34' y2='2' stroke='#9b998c' stroke-width='2'/>"
      "</svg><b>tried, then abandoned</b> &mdash; the &times; is where it stopped</span>"
      "<span class='k'><svg width='36' height='12'><line x1='0' y1='6' x2='27' y2='6' "
      "stroke='#8c2f1f' stroke-width='2'/><circle cx='30' cy='6' r='5' fill='#fffff8' "
      "stroke='#8c2f1f' stroke-width='2'/></svg><b>ran as the mainline, then replaced</b> "
      "&mdash; the &#9711; is the moment it was retired</span>"
      "<span class='k'><svg width='36' height='12'><line x1='0' y1='3' x2='24' y2='3' "
      "stroke='#111111' stroke-width='2'/><path d='M24,3 C30,3 32,7 32,10' stroke='#111111' "
      "stroke-width='2' fill='none'/><path d='M28.5,8 L32,12 L35.5,8 z' fill='#111111'/>"
      "</svg><b>an insight that merged in</b> &mdash; the arrow is where it changed the project</span>"
      "<span class='k'><svg width='22' height='12'><path d='M11,1 l5.5,5.5 l-5.5,5.5 "
      "l-5.5,-5.5 z' fill='#8c2f1f'/></svg><b>incident</b> &mdash; something broke</span>")
    # The open-branch swatch is drawn only when there IS an open branch. It
    # was left standing once after the last one closed, promising a marker
    # the figure no longer contains — a key that lies is worse than no key.
    if evofig.OPEN:
        key += (
          "<span class='k'><svg width='36' height='12'><line x1='0' y1='6' x2='27' y2='6' "
          "stroke='#111111' stroke-width='2' stroke-dasharray='6 4' opacity='.85'/>"
          "<circle cx='30' cy='6' r='5' fill='#fffff8' stroke='#111111' stroke-width='2'/>"
          "</svg><b>branched off, still open</b> &mdash; it ran and produced a real "
          "number, but nothing has been concluded yet</span>")
    _, eras = load_history()
    bands = evolution_era_bands(eras)
    # Splice the bands in right after the opening <svg …> so they sit behind
    # every mark, and extend the key with one swatch per era.
    forked = EVOLUTION_SVG
    evo_svg = (re.sub(r"(<svg\b[^>]*>)", lambda m: m.group(1) + bands,
                      forked, count=1) if bands else forked)
    if bands:
        key += "".join(
            f"<span class='k'><span class='era-sw' "
            f"style='background:{ERA_TINT.get(e['key'], '')}'></span>"
            f"<b>{esc(ERA_SHORT.get(e['key'], e['label']))}</b></span>"
            for e in eras)
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{social_meta('gallery/research-evolution.html', 'Research evolution — Ballpark', 'How the research itself branched: twelve days of eras, dead ends, incidents and insights, drawn as one graph.')}
<title>Research Evolution &mdash; Ballpark</title>
<style>{CSS}{EVOLUTION_CSS}</style></head><body>
{topnav('evolution')}
{compute_banner()}
{page_header("Research evolution: the shape of the search",
  "The <a href='research-lineage.html'>experiment lineage</a> shows what the loop "
  "tried. This shows what happened to the <i>research itself</i>. Time runs left to "
  "right. The dark line is the path that survived; above it are branches that left "
  "the trunk &mdash; grey and &times;-ended where they stopped" + (
    ", dark and hollow-ended if they are still open" if evofig.OPEN else "") +
  "; everything <b>below</b> ran as the mainline "
  "for a while before being replaced &mdash; including which machine the work ran "
  "on. <b>Hover anything</b> for what happened and why.")}
<div class="evo">
<div class="evo-prose"><div class="evo-key">{key}</div></div>
<p class="evo-scrollhint">scroll sideways &rarr; twelve days, 20 July &ndash; 1 August</p>\n<div class="evo-figure">{evo_svg}</div>
<div class="evo-prose"><p class="evo-read">The two long red lines are the story: for <b>ten of the
project&rsquo;s twelve days</b> the evaluation was asking a question the model could
not answer, and the score preferred guessing to learning. Every result measured
against them had to be thrown away.</p></div>
</div>
<div id="evo-tip"></div>
<script type="application/json" id="evo-tips">{EVOLUTION_TIPS}</script>
<script>{EVOLUTION_JS}</script>
{credits_html()}</body></html>"""
    EVOLUTION_OUT.parent.mkdir(exist_ok=True)
    EVOLUTION_OUT.write_text(body)
    print(f"wrote {EVOLUTION_OUT} (graph: {len(evofig.TRUNK_NODES)} trunk nodes, "
          f"{len(evofig.DEAD)} abandoned, {len(evofig.SUPER)} superseded, "
          f"{len(evofig.OPEN)} open)")


LINEAGE_OUT = REPO_ROOT / "gallery" / "research-lineage.html"

# Ported from the author's llm-heuristic-scientists-workshop lineage page
# (static/lineage.js + lineage.css), adapted to this repo's data model:
# nodes from experiments.sqlite, popover content inlined (no detail endpoint),
# click-through to the research log row.
LINEAGE_CSS = """
.lin-head{max-width:980px;margin:0 auto;padding:0 16px}
.lin-head .legend{margin:10px 0 4px;display:flex;flex-wrap:wrap;gap:10px 22px;
  font-size:13px;color:var(--muted)}
.lin-head .legend .k{display:inline-flex;align-items:center;gap:7px}
.lin-head .legend .ldot{width:9px;height:9px;border-radius:50%;display:inline-block}
.lin-head .legend .larc{width:18px;height:9px;border-top:1px solid var(--rule);
  display:inline-block;border-radius:9px 9px 0 0}
.lin-head .legend .lring{width:10px;height:10px;border-radius:50%;
  background:var(--paper);border:1.5px solid #8a6a1e;display:inline-block}
#diagram{overflow-x:auto;margin-top:14px;padding:24px 28px 16px;
  scrollbar-width:thin;scrollbar-color:var(--rule) transparent}
#diagram::-webkit-scrollbar{height:6px}
#diagram::-webkit-scrollbar-thumb{background:var(--rule);border-radius:3px}
#lin{display:block}
#lin text{font:12px/1 var(--serif);fill:var(--muted)}
#lin .edge{fill:none;stroke:#b2ac99;stroke-width:1}
#lin .nd{cursor:pointer}
#lin .nk{fill:var(--ink)}
#lin .ndd{fill:#9b998c}
#lin .nf{fill:var(--accent)}
#lin .nh{fill:var(--paper);stroke:#8a6a1e;stroke-width:1.6}
#lin .np{fill:none;stroke:#8a6a1e;stroke-width:1.5}
#lin .nu{stroke:#9b998c;stroke-width:1.5;stroke-dasharray:2 2;fill:none}
#lin .nlab{fill:var(--faint);font:10px/1 var(--serif);stroke:none}
#lin .hit{cursor:pointer}
/* Evaluation-era bands: washes behind the whole diagram, each captioned in
   place so the band is never identified by colour alone. */
#lin .eband{pointer-events:none}
#lin .ebound{stroke:var(--rule);stroke-width:1}
#lin .elab{font:600 10.5px var(--serif);font-feature-settings:"smcp" 1;
  letter-spacing:.05em;stroke:none}
#tip .pop-prov{color:var(--faint);font-style:normal}
#lin.dim .nd,#lin.dim .edge,#lin.dim .nlab{opacity:.07}
#lin.dim .nd.lit{opacity:.55}
#lin.dim .nd.lit-direct{opacity:1}
#lin.dim .nlab.lit{opacity:1;fill:var(--faint)}
#lin.dim .nlab.lit-direct{opacity:1;fill:var(--ink)}
#lin.dim .edge.lit{opacity:.4;stroke:var(--muted);stroke-width:1}
#lin.dim .edge.lit-direct{opacity:1;stroke:var(--ink);stroke-width:2}
#tip{position:fixed;z-index:30;pointer-events:none;opacity:0;transition:opacity .1s;
  background:var(--paper);color:var(--ink);border:1px solid var(--rule);
  box-shadow:0 6px 22px rgba(60,50,30,.16);border-radius:5px;
  font:13px/1.45 var(--serif);padding:12px 14px;max-width:430px}
#tip .pop-title{font-weight:700;font-size:14.5px;line-height:1.25;margin-bottom:2px}
#tip .pop-meta{color:var(--muted);font:11px/1.3 var(--serif);
  font-feature-settings:"smcp" 1;text-transform:uppercase;
  letter-spacing:.04em;margin-bottom:10px}
#tip .pop-h{font:800 10px/1 var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.05em;color:var(--faint);margin-bottom:4px}
#tip p{margin:0;font:italic 13px/1.5 var(--serif);color:var(--muted)}
#tip .pop-sec + .pop-sec{margin-top:11px}
#tip .pop-parent{font:600 11px/1.35 var(--serif);font-feature-settings:"smcp" 1;
  text-transform:uppercase;letter-spacing:.03em;color:var(--accent)}
"""

LINEAGE_JS = r"""
(function () {
  let data = {};
  try { data = JSON.parse(document.getElementById("lineage-data").textContent || "{}"); } catch (e) {}
  const nodes = data.nodes || [];
  const host = document.getElementById("diagram");
  const tip = document.getElementById("tip");
  if (!host) return;
  if (!nodes.length) { host.innerHTML = '<p class="empty">No experiments yet.</p>'; return; }
  const eras = data.eras || [];
  const esc = (s) => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  // The metric is a mission score, not a distance. This formatted it as
  // metres for as long as the metric WAS a distance, which rendered the
  // best result the project has ("0.040") as "0.0 m".
  const fmtScore = (v) => v == null ? null : v.toFixed(3);
  const provNote = {
    rescored: "re-scored today from its exported model",
    native: "measured on this era's own ruler",
    derived: "rates recovered from the logged record",
    gated: "failed a deployment gate",
    incomparable: "ran on 10 m/px imagery — not on this ruler",
    unrecoverable: "model and rates both lost",
    holdout: "blind Hamburg holdout — a different area",
  };

  const byKey = {}; nodes.forEach(n => byKey[n.key] = n);
  const parentsOf = {}; nodes.forEach(n => parentsOf[n.key] = (n.parents || []).filter(p => byKey[p]));

  const n = nodes.length;
  const padX = 28, topPad = 14, CAP = 260, minSpacing = 26;
  const labelOf = (nd) => "#" + nd.n + "  " + nd.title.slice(0, 30);
  const maxLabelLen = Math.max(1, ...nodes.map(nd => labelOf(nd).length));
  const labelH = Math.min(280, Math.round(maxLabelLen * 5.9) + 14);
  const avail = Math.max(320, (host.clientWidth || 1000) - 2 * padX - 8);
  const spacing = n > 1 ? Math.max(minSpacing, avail / (n - 1)) : 0;
  const Wpx = Math.round(2 * padX + (n - 1) * spacing);
  const xFor = (i) => padX + i * spacing;
  const xOf = {}; nodes.forEach((nd, i) => xOf[nd.key] = xFor(i));

  const edges = [];
  nodes.forEach(nd => parentsOf[nd.key].forEach(pk =>
    edges.push({ c: nd.key, p: pk, dx: Math.abs(xOf[nd.key] - xOf[pk]) })));
  const maxRy = edges.length ? Math.min(CAP, Math.max(...edges.map(e => e.dx / 2))) : 18;
  const ERA_LAB_H = eras.length ? 20 : 0;
  const baseY = topPad + ERA_LAB_H + maxRy;
  const H = Math.ceil(baseY + 12 + labelH);

  let g = "";
  // Era bands first, so every arc and node sits on top of its own band.
  eras.forEach((era, i) => {
    const lo = Math.max(0, xFor(era.start) - spacing / 2);
    const hi = Math.min(Wpx, xFor(era.end) + spacing / 2);
    if (hi <= lo) return;
    g += `<rect class="eband" data-era="${esc(era.key)}" x="${lo.toFixed(1)}" y="0" `
       + `width="${(hi - lo).toFixed(1)}" height="${H}" fill="${era.tint}"/>`;
    if (i) g += `<line class="ebound" x1="${lo.toFixed(1)}" y1="0" x2="${lo.toFixed(1)}" y2="${H}"/>`;
    g += `<text class="elab" x="${((lo + hi) / 2).toFixed(1)}" y="15" `
       + `text-anchor="middle" fill="${era.ink}">${esc(era.short)}</text>`;
  });
  edges.forEach(e => {
    const a = Math.min(xOf[e.p], xOf[e.c]), b = Math.max(xOf[e.p], xOf[e.c]);
    const rx = (b - a) / 2, ry = Math.min(CAP, rx);
    g += `<path class="edge" data-c="${esc(e.c)}" data-p="${esc(e.p)}" `
       + `d="M${a.toFixed(1)},${baseY} A${rx.toFixed(1)},${ry.toFixed(1)} 0 0 1 ${b.toFixed(1)},${baseY}"/>`;
  });
  nodes.forEach((nd, i) => {
    const x = xFor(i).toFixed(1);
    const cls = nd.kind === "failed" || nd.kind === "rejected" ? "nf" : nd.kind === "kept" ? "nk"
              : nd.kind === "holdout" ? "nh" : "ndd";
    const r = nd.kind === "kept" ? 4 : nd.kind === "holdout" ? 4 : 3;
    if (nd.kind === "unknown") {
      // No score exists and none can be reconstructed. Drawn as a gap in the
      // record rather than a dot, so it cannot be read as a result.
      g += `<line class="nd nu" data-key="${esc(nd.key)}" x1="${x}" y1="${baseY - 4}" `
         + `x2="${x}" y2="${baseY + 4}"/>`;
    } else {
      g += `<circle class="nd ${cls}" data-key="${esc(nd.key)}" cx="${x}" cy="${baseY}" r="${r}"/>`;
    }
    if (nd.pivot) { const R = 6.5, cx = parseFloat(x);
      g += `<path class="nd np" data-key="${esc(nd.key)}" `
         + `d="M${cx.toFixed(1)},${(baseY + R).toFixed(1)} `
         + `L${(cx + R * 0.866).toFixed(1)},${(baseY - R * 0.5).toFixed(1)} `
         + `L${(cx - R * 0.866).toFixed(1)},${(baseY - R * 0.5).toFixed(1)} Z"/>`; }
    g += `<circle class="hit" data-key="${esc(nd.key)}" cx="${x}" cy="${baseY}" r="9" fill="transparent"/>`;
    g += `<text class="nlab" data-key="${esc(nd.key)}" x="${x}" y="${baseY + 8}" text-anchor="start" `
       + `transform="rotate(90 ${x} ${baseY + 8})">${esc(labelOf(nd))}</text>`;
  });
  const viewW = host.clientWidth || window.innerWidth || 1000;
  const trail = Math.max(0, Math.round(viewW / 2) - padX);
  const Wsvg = Wpx + trail;
  host.innerHTML = `<svg id="lin" width="${Wsvg}" height="${H}" viewBox="0 0 ${Wsvg} ${H}">${g}</svg>`;

  const latestX = xFor(n - 1);
  const centerTarget = () =>
    Math.max(0, Math.min(latestX - host.clientWidth / 2, host.scrollWidth - host.clientWidth));
  host.scrollLeft = 0;
  let introRan = false;
  const intro = () => {
    if (introRan) return; introRan = true;
    const start = host.scrollLeft, dist = centerTarget() - start, dur = 1150;
    if (Math.abs(dist) < 1) return;
    const ease = (t) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    const t0 = performance.now();
    const step = (now) => {
      const p = Math.min(1, (now - t0) / dur);
      host.scrollLeft = start + dist * ease(p);
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };
  window.addEventListener("load", () => { host.scrollLeft = 0; setTimeout(intro, 350); });

  const svg = document.getElementById("lin");
  const q = (k) => CSS.escape(k);
  function lit(key){
    const nodeSet = new Set([key]); const edgeSet = new Set();
    const stack = [key];
    while (stack.length){
      const cur = stack.pop();
      (parentsOf[cur] || []).forEach(p => {
        edgeSet.add(cur + "|" + p);
        if (!nodeSet.has(p)){ nodeSet.add(p); stack.push(p); }
      });
    }
    const direct = new Set([key, ...(parentsOf[key] || [])]);
    svg.classList.add("dim");
    svg.querySelectorAll(".lit,.lit-direct").forEach(el => el.classList.remove("lit", "lit-direct"));
    nodeSet.forEach(k => svg.querySelectorAll(`[data-key="${q(k)}"]`).forEach(el =>
      el.classList.add(direct.has(k) ? "lit-direct" : "lit")));
    svg.querySelectorAll(".edge").forEach(e => {
      const c = e.getAttribute("data-c"), p = e.getAttribute("data-p");
      if (edgeSet.has(c + "|" + p)) e.classList.add(c === key ? "lit-direct" : "lit");
    });
  }
  function unlit(){ svg.classList.remove("dim"); svg.querySelectorAll(".lit,.lit-direct").forEach(el => el.classList.remove("lit", "lit-direct")); }

  function placePopover(nd){
    const svgRect = svg.getBoundingClientRect(), w = tip.offsetWidth;
    const sx = svgRect.width / Wsvg, sy = svgRect.height / H;
    const cx = svgRect.left + xOf[nd.key] * sx;
    const left = Math.max(8, Math.min(cx - w / 2, window.innerWidth - w - 12));
    const top = svgRect.top + baseY * sy + 10;
    tip.style.left = left + "px"; tip.style.top = top + "px";
  }
  function showPopover(nd){
    const met = nd.kind === "rejected" ? "rejected — never trained"
      : nd.kind === "failed" ? "gated fail"
      : nd.kind === "unknown" ? (provNote[nd.prov] || "no score")
      : (fmtScore(nd.metric) || "—") + (nd.kind === "holdout" ? " · blind holdout" : "");
    const rates = (nd.usable != null)
      ? `<div class="pop-sec"><div class="pop-h">On today's ruler</div>`
        + `<p>${(nd.usable * 100).toFixed(1)}% usable fixes · `
        + `${(nd["false"] * 100).toFixed(1)}% confidently wrong `
        + `<span class="pop-prov">(${esc(provNote[nd.prov] || "")})</span></p></div>`
      : "";
    const parents = (parentsOf[nd.key] || []).map(p =>
      `<span class="pop-parent">↳ #${esc((byKey[p] || {}).n || p)} `
      + `${esc((byKey[p] || {}).title || "").slice(0, 48)}</span>`).join("<br>");
    const era = nd.eraLabel
      ? `<div class="pop-sec"><div class="pop-h">Evaluation era</div><p>${esc(nd.eraLabel)}</p></div>` : "";
    tip.innerHTML =
      `<div class="pop-title">${esc(nd.title)}</div>`
      + `<div class="pop-meta">#${esc(nd.n)} · ${esc(nd.category || "")} · ${met}</div>`
      + (nd.summary ? `<div class="pop-sec"><div class="pop-h">In plain words</div><p>${esc(nd.summary)}</p></div>` : "")
      + rates + era
      + (parents ? `<div class="pop-sec"><div class="pop-h">Built on</div>${parents}</div>` : "");
    tip.style.opacity = "1";
    placePopover(nd);
  }

  svg.addEventListener("mouseover", e => {
    const m = e.target.closest("[data-key]"); if (!m) return;
    const key = m.getAttribute("data-key");
    lit(key);
    const nd = byKey[key]; if (nd) showPopover(nd);
  });
  svg.addEventListener("mouseout", e => {
    const m = e.target.closest("[data-key]"); if (!m) return;
    unlit(); tip.style.opacity = "0";
  });
  // Only the current era has rows in the research log; earlier eras' records
  // live in the archived DBs, so their nodes are hover-only rather than
  // linking to a row that isn't there.
  svg.addEventListener("click", e => {
    const m = e.target.closest("[data-key]"); if (!m) return;
    const nd = byKey[m.getAttribute("data-key")];
    if (nd && nd.link !== false) location.href = "index.html#r" + encodeURIComponent(nd.n);
  });
})();
"""


def render_lineage(exps):
    """gallery/research-lineage.html — the author's lineage-page idiom with
    this project's data: scrollable arc diagram, ancestry hover, eli5
    popovers, click-through to the log.

    Spans EVERY era, and the eras are deliberately not joined to each other.
    Each wipe of the lineage restarted the search from a fresh baseline, so
    an arc across a boundary would assert a descent that never happened. The
    diagram is five separate forests sharing one timeline — which is itself
    the finding: four of them died out."""
    hist, eras = load_history()
    if hist:
        nodes = []
        last_kept = None
        prev_era = None
        for r in hist:
            if r["era_index"] != prev_era:      # a wipe: nothing inherits across it
                last_kept, prev_era = None, r["era_index"]
            prov = r["provenance"]
            if prov == "holdout":
                kind = "holdout"
            elif prov == "gated":
                kind = "failed"
            elif prov in ("incomparable", "unrecoverable"):
                kind = "unknown"
            elif r["kept"]:
                kind = "kept"
            else:
                kind = "discarded"
            key = f"{r['era']}-{r['src_id']}"
            nodes.append({
                "key": key, "n": str(r["src_id"]),
                "title": r["title"] or "(untitled)",
                "summary": (r["eli5"] or r["hypothesis"] or "")[:260],
                "metric": r["mission_score"],
                "usable": r["usable_fix_rate"], "false": r["false_fix_rate"],
                "kind": kind, "pivot": False, "category": r["category"] or "",
                "era": r["era"], "eraLabel": r["era_label"], "prov": prov,
                # Only the current era has a row in the research log to open.
                "link": r["era"] == "mission",
                "parents": [last_kept] if last_kept is not None else [],
            })
            if kind == "kept":
                last_kept = key
        era_meta = [{
            "key": e["key"], "label": e["label"], "short": ERA_SHORT.get(e["key"], e["label"]),
            "tint": ERA_TINT.get(e["key"], "rgba(107,106,96,.06)"),
            "ink": ERA_INK.get(e["key"], "#6b6a60"),
            "start": e["seq_start"] - 1, "end": e["seq_end"] - 1,
            "ruler": e["ruler"], "evalSet": e["eval_set"], "note": e["note"],
        } for e in eras]
    else:
        nodes, era_meta = [], []
        last_kept = None
        for e in exps:
            gated = (e["primary_metric"] or 0) >= FAIL
            if e["kind"] == "holdout_check":
                kind = "holdout"
            elif is_rejected(e):
                kind = "rejected"
            elif gated:
                kind = "failed"
            elif e["kept"]:
                kind = "kept"
            else:
                kind = "discarded"
            nodes.append({
                "key": str(e["id"]), "n": str(e["id"]),
                "title": e["title"] or "(untitled)",
                "summary": (e["eli5"] or e["hypothesis"] or "")[:260],
                "metric": (None if gated or e["primary_metric"] is None
                           else e["primary_metric"]),
                "usable": None, "false": None, "link": True,
                "kind": kind, "pivot": bool(e.get("is_pivot")),
                "category": e["category"] or "", "era": "", "eraLabel": "",
                "prov": "native",
                "parents": [str(last_kept)] if last_kept is not None else [],
            })
            if kind == "kept":
                last_kept = e["id"]
    n_dev = sum(1 for nd in nodes if nd["kind"] != "holdout")
    data_json = json.dumps({"nodes": nodes, "eras": era_meta, "target": 0.0})
    if era_meta:
        lineage_sub = (
            f"All {n_dev} experiments, left → right in discovery order; each arc "
            f"links an experiment to the kept design it built on. The shaded "
            f"bands are the {len(era_meta)} <b>evaluation eras</b> &mdash; each "
            f"time the eval set or the metric was corrected the lineage was "
            f"wiped and the search restarted from a fresh baseline, so "
            f"<b>no arc crosses a band boundary</b>. Nothing in the last band "
            f"inherits from the four before it, which is exactly why the final "
            f"run had to rediscover from scratch what earlier eras had already "
            f"found. <b>Hover</b> to trace ancestry back to that era's root; "
            f"experiments in the current era <b>click through</b> to the "
            f"<a href='index.html'>research log</a>.")
        lineage_era_key = "".join(
            f"<span class='k'><span class='era-sw' style='background:{e['tint']}'>"
            f"</span>{esc(e['short'])}</span>" for e in era_meta)
    else:
        lineage_sub = (f"{n_dev} experiments, left → right in discovery order; "
                       f"each arc links an experiment to the kept design it built "
                       f"on. <b>Hover</b> to trace its ancestry back to the root; "
                       f"<b>click</b> to open its full record in the "
                       f"<a href='index.html'>research log</a>.")
        lineage_era_key = ""
    html_page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{social_meta('gallery/research-lineage.html', 'Experiment lineage — Ballpark', 'The family tree of the search: every experiment as a node, arcs to the designs it built on.')}
<title>Experiment Lineage &mdash; Ballpark</title>
<style>{CSS}{LINEAGE_CSS}</style></head><body>
{topnav('lineage')}
{compute_banner()}
{page_header("Experiment lineage: the family tree of the search", lineage_sub)}
<div class="lin-head">
<div class="legend">
  <span class="k"><span class="ldot" style="background:var(--ink)"></span>Kept (new best)</span>
  <span class="k"><span class="ldot" style="background:#9b998c"></span>Worse than best</span>
  <span class="k"><span class="ldot" style="background:var(--accent)"></span>Gated fail</span>
  <span class="k"><span class="lring"></span>Blind holdout check</span>
  <span class="k"><span class="larc"></span>Derived from its parent</span>
  {lineage_era_key}
</div>
</div>
<div id="diagram"></div>
<div id="tip"></div>
<script id="lineage-data" type="application/json">{data_json}</script>
<script>{LINEAGE_JS}</script>
{credits_html()}</body></html>"""
    LINEAGE_OUT.parent.mkdir(exist_ok=True)
    LINEAGE_OUT.write_text(html_page)
    print(f"wrote {LINEAGE_OUT} ({len(nodes)} nodes)")


def render():
    conn = connect()
    conn.row_factory = lambda cur, row: {d[0]: row[i] for i, d in enumerate(cur.description)}
    exps = conn.execute("SELECT * FROM experiments ORDER BY id ASC").fetchall()
    annotate_pivot(exps)
    hist_rows, hist_eras = load_history()
    n_dev = sum(1 for e in exps if e["kind"] != "holdout_check")
    n_kept = sum(1 for e in exps if e["kept"] and e["kind"] != "holdout_check")
    best = next((e["primary_metric"] for e in reversed(exps)
                 if e["kept"] and e["kind"] != "holdout_check"
                 and e["primary_metric"] and e["primary_metric"] < FAIL),
                None)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    era_band_key = "".join(
        f"<span class='era-sw' style='background:{ERA_TINT.get(e['key'], '')}'></span>"
        for e in hist_eras) or "<span class='era-sw'></span>"
    # "5 experiments, 4 kept" is true of the table below and false of the
    # project: the current era is the fifth, and the four experiments that
    # solved it stand on 78 that came before.
    if hist_rows:
        n_all = sum(1 for r in hist_rows if r["kind"] != "holdout_check")
        c_agent = sum(r["cost_agent"] or 0 for r in hist_rows)
        c_gpu = sum(r["cost_gpu"] or 0 for r in hist_rows)
        n_costed = sum(1 for r in hist_rows if r["cost_stages_n"])
        cost_line = ""
        if c_agent or c_gpu:
            cost_line = (
                f" Cost of the whole search: <b>{fmt_usd(c_agent + c_gpu)}</b> "
                f"&mdash; {fmt_usd(c_agent)} of agent tokens (equivalent API "
                f"cost; the work ran on a Max subscription) plus "
                f"{fmt_usd(c_gpu)} of rented GPU, recorded for {n_costed} of "
                f"{n_all} experiments.")
        status_meta = (f"{n_dev} experiment{'s' if n_dev != 1 else ''} in this "
                       f"era, {n_kept} kept &mdash; and {n_all - n_dev} before it, "
                       f"across {len(hist_eras) - 1} earlier evaluation "
                       f"{'eras' if len(hist_eras) > 2 else 'era'}. The chart "
                       f"below shows all {n_all}.{cost_line}")
    else:
        status_meta = (f"{n_dev} experiment{'s' if n_dev != 1 else ''}, "
                       f"{n_kept} kept.")

    # The chart shows the WHOLE project when the cross-era record has been
    # built, and falls back to the current era alone when it hasn't. Showing
    # only the current era is misleading on its own: it reads as though the
    # problem took five experiments, when five eras and 83 experiments went
    # into it and most of them were spent measuring the wrong thing.
    if hist_rows:
        n_hist = sum(1 for r in hist_rows if r["kind"] != "holdout_check")
        n_by = {}
        for r in hist_rows:
            n_by[r["provenance"]] = n_by.get(r["provenance"], 0) + 1
        n_era = len(hist_eras)
        chart_heading = (f"Mission score per experiment, all {n_hist} of them "
                         f"&mdash; (1 &minus; usable-fix rate) + false-fix rate, "
                         f"lower is better")
        chart_block = history_chart_svg(hist_rows, hist_eras)
        chart_caption = (
            f"<p class='chart-note'>Every experiment the project has run, across "
            f"{n_era} evaluation eras. The lineage was wiped at each era boundary, "
            f"so the numbering restarts inside every band &mdash; but the "
            f"<b>vertical scale is one ruler throughout</b>. "
            f"{n_by.get('rescored', 0)} of these were re&#8209;measured for this "
            f"chart by running their exported model through <i>today&rsquo;s</i> "
            f"frozen scorer on <i>today&rsquo;s</i> held&#8209;out viewpoints; "
            f"{n_by.get('native', 0)} are native to the current era; "
            f"{n_by.get('derived', 0)} had their run artifacts deleted and have "
            f"their usable- and false-fix rates recovered arithmetically from the "
            f"logged record; {n_by.get('gated', 0)} failed a deployment gate and "
            f"had no working model to score, then or now. Nothing was converted "
            f"by a fudge factor. "
            f"<b>Black means the loop kept it</b> &mdash; judged at the time, "
            f"under whatever metric was then in force, which is why kept "
            f"experiments sit high in the early bands: the old metric was "
            f"rewarding the wrong thing. The step line is the best mission score "
            f"anyone had actually achieved, and it is unbroken across the "
            f"boundaries because the ruler no longer changes at them.</p>")
    else:
        chart_heading = ("Mission score per experiment — (1 &minus; usable-fix "
                         "rate) + false-fix rate, lower is better")
        chart_block = chart_svg(exps)
        chart_caption = ""
    best_size = next((e["model_bytes_max"] for e in reversed(exps)
                      if e["kept"] and e["kind"] != "holdout_check"
                      and e["model_bytes_max"]), None)
    size_note = (f"currently <b class='num'>{best_size/1024:,.0f} KB</b>, hard limit "
                 f"<b class='num'>4 MiB</b>" if best_size else "hard limit <b>4 MiB</b>")
    if best is None:
        status_line = "No scoreable model yet."
    elif best <= 0.05:
        status_line = (f"<b>Goal reached:</b> mission score "
                       f"<b class='num'>{fmt_score(best)}</b> — very nearly every frame "
                       f"yields a usable fix.")
    else:
        status_line = (
            f"Status: best mission score is <b class='num'>{fmt_score(best)}</b> in its "
            f"hardest cell. That number is <b>(1 &minus; usable-fix rate) + false-fix "
            f"rate</b>: a frame counts as a <b>usable fix</b> only when the model is "
            f"confident <i>and</i> within {TARGET_M:.0f} m; abstaining is safe; being "
            f"confident and wrong is a <b>false fix</b>, which for a UAV is worse than "
            f"saying nothing at all. <b>1.0</b> means it abstains on everything, "
            f"<b>2.0</b> means it is confidently wrong on everything, <b>0</b> is the goal.")

    body = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
{social_meta('gallery/index.html', 'Research log — Ballpark', 'Every experiment the autonomous loop ran, failures included: pre-registered hypotheses, results, and the exact agent prompts.')}
<title>Research Log &mdash; Ballpark</title>
<style>{CSS}</style><script>{JS}</script></head><body>
{topnav('log')}
{compute_banner()}
{page_header("Where we are: The experiment record", f"Every experiment the autonomous loop has run, across all {len(hist_eras) or 1} evaluation eras — kept <i>and</i> discarded. Each row was pre-registered before training (hypothesis, method, expected outcome, architecture figure), then trained on a single GPU and scored. Every score on this page is measured by <b>one</b> frozen ruler: the <b>worst</b> mission score on held-out viewpoints — (1 &minus; usable-fix rate) + false-fix rate, the product requirement rather than an error statistic ({size_note}). The earlier eras were <i>judged</i> by different rulers at the time, and where those two disagree the row says so. One agent designs, one implements; failures stay on the record. New here? Start with the <a href='../index.html'>overview</a>.")}
<div class="status-callout">
  <div class="status-callout-h">Where we are right now</div>
  <p>{status_line}</p>
  <p class="status-meta">{status_meta}</p>
</div>
<header class="dash-head">
  <div class="dash-meta">
    <span class="k" title="This change improved the worst-case error and was committed."><span class="dot kept"></span>Kept improvement</span>
    <span class="k" title="No improvement — the code change was reverted; only the record remains."><span class="dot disc"></span>Discarded</span>
    <span class="k" title="Best mission score achieved so far."><span class="bar"></span>Running best</span>
    <span class="k" title="The goal: mission score 0 — every frame a usable fix."><span class="bar dash"></span>Goal (score 0)</span>
    <span class="k" title="Violated a deployment gate (model size, latency, or abstained too much) — scored as failure regardless of accuracy."><span class="x">×</span>Gated fail</span>
    <span class="k" title="Region-holdout check on genuinely untrained ground — logged for honesty, never used to decide keep/revert."><span class="ring"></span>Holdout check</span>
    <span class="k" title="This experiment ran after the design agent had gone 4+ consecutive tries without beating the running best — the harness injects a mandatory 'do not refine the champion again, pick an absent design family' directive into its prompt. Switched off in the current configuration — will not appear until reintroduced."><span class="tri"></span>Pivot-directed</span>
    <span class="k" title="No mission score exists for this experiment and none can be honestly reconstructed: either it ran on 10 m/px imagery before the switch to 1 m/px orthophotos (so it is not being asked the same question), or its model files and rate data are both gone."><span class="vrule"></span>Not on this ruler</span>
    <span class="k" title="Each shaded band is one evaluation era. Within a band the eval set and the optimized metric were held fixed; between bands one or both changed and the lineage was wiped. Hover a band for what it was measuring.">{era_band_key}Evaluation era</span>
    <span id="updated">updated {now}</span>
  </div>
</header>
<div class="dash-wrap">
<div class="chart-card">
<div class="chart-title">{chart_heading}</div>
{chart_block}</div>
<div class="tbl-card"><table class="main">
<thead><tr><th></th><th>#</th><th>Experiment</th>
<th title="Which lever the experiment pulls: architecture, loss, augmentation, relighting, training, quantization.">Category</th>
<th title="Mission score on held-out Berlin daytime viewpoints: (1 - usable-fix rate) + false-fix rate. A frame is a usable fix only if the model is confident AND within 100 m; abstaining is safe; confident-but-wrong is a false fix, worse than silence for a UAV. 0 = every frame usable, 1 = abstains everywhere, 2 = confidently wrong everywhere. The one number the loop optimizes -- deliberately the product requirement, not an error statistic.">Mission score</th>
<th title="Wall time of the whole experiment: agent design + training all areas + scoring.">Time</th>
<th title="Everything one experiment cost: the equivalent API cost of the tokens its design, implementation, figure and summary agents consumed - recorded by the harness at the time, not estimated from a price list - plus rented-GPU wall time during the pod window at $0.69/hr. The research ran on a claude.ai Max subscription, so the agent part is what it WOULD have cost billed per token, not money that left the account. An em dash means no accounting survives (the run directory was deleted), which is not the same as free.">Cost</th>
<th>Status</th></tr></thead><tbody>"""]
    body.append(live_row((max((e["id"] for e in exps), default=0) or 0) + 1))

    # Numbering restarted at 1 with each era, so a bare "#3" is ambiguous
    # across a table that now spans all of them. Rows carry <era>.<n>.
    cur_era = next((e for e in hist_eras if e["key"] == "mission"), None)
    cur_prefix = f"{cur_era['era_index'] + 1}." if cur_era else ""
    cur_era_label = esc(cur_era["label"]) if cur_era else ""
    cur_tint = ERA_TINT.get("mission", "") if cur_era else ""
    # The current era's rows come from experiments.sqlite, but their cost
    # accounting lives in the generated cross-era record — join on id so both
    # halves of the table cost the same way.
    hist_by_id = {r["src_id"]: r for r in hist_rows if r["era"] == "mission"}

    for e in reversed(exps):
        kept_cls = " kept-row" if (e["kept"] and e["kind"] != "holdout_check") else ""
        size = f"{e['model_bytes_max']/1024:,.0f} KB" if e["model_bytes_max"] else "—"
        lat = f"{e['latency_ms_host_proxy']:.1f} ms" if e["latency_ms_host_proxy"] else "—"
        pivot_tag = (" <span class='pivot-tag' title='Ran after 4+ consecutive "
                     "misses — the harness required a new design family this "
                     "round'>pivot</span>" if e.get("is_pivot") else "")
        sr_kind, sr_short, sr_long = status_reason(e)
        body.append(f"""<tr class="row-main{kept_cls}" id="r{e['id']}"
 style="--era-tint:{cur_tint}" onclick="toggle('{e['id']}')">
<td><span class="caret">▸</span></td>
<td class="num" title="{cur_era_label} — experiment {e['id']} of that era">{cur_prefix}{e['id']}</td>
<td class="title-cell"><b>{esc(e['title'])}</b>
  <span class="mono" style="color:var(--faint)"> {esc(e['git_commit'][:8])}</span>
  <div class="row-why why-{sr_kind}">{esc(sr_short)}</div></td>
<td><span class="cat">{esc(e['category'] or '—')}</span>{pivot_tag}</td>
<td class="num">{fmt_score(e['primary_metric'])}</td>
<td class="num">{fmt_dur(e.get('duration_s'))}</td>
<td class="num">{cost_cell(hist_by_id.get(e['id']))}</td>
<td>{status_of(e)}</td></tr>""")

        blocks = []
        if e.get("eli5"):
            blocks.append(f"<div class='eb eb-eli'><div class='eb-h'>In plain "
                          f"words</div><p>{esc(e['eli5'])}</p></div>")
        blocks.append(cost_block(hist_by_id.get(e["id"])))
        _why_head = {"kept": "Why it's the new best", "rej": "Why it was rejected",
                     "fail": "Why it failed the gate", "disc": "Why it was discarded",
                     "hold": "What this holdout check is"}.get(sr_kind, "Why this status")
        blocks.append(f"<div class='eb eb-why why-{sr_kind}'><div class='eb-h'>{_why_head}</div>"
                      f"<p>{esc(sr_long)}</p></div>")
        for key, cls, label in (("hypothesis", "eb-hyp", "Hypothesis"),
                                ("method", "eb-met", "Method"),
                                ("expected_outcome", "eb-exp", "Expected outcome"),
                                ("result", "eb-res", "Result"),
                                ("conclusion", "eb-con", "Conclusion")):
            if e.get(key):
                blocks.append(f"<div class='eb {cls}'><div class='eb-h'>{label}</div>"
                              f"<p>{esc(e[key])}</p></div>")
        metrics = json.loads(e["metrics_json"] or "{}")
        body.append(f"""<tr class="detail" id="d{e['id']}" style="display:none"><td colspan="8">
<div class="detail-inner">
{arch_block(e, chain_of(e, exps))}
<div class="detail-grid">
<div class="explain">{''.join(blocks)}</div>
<div>
{heatmap_block(e['artifacts_dir'], metrics)}
{worked_example_block(e)}
<div class="score-head">Scoreboard — mission score per area × lighting</div>
<div class="score-sub">mission score per cell — (1 &minus; usable-fix rate) + false-fix
rate; the worst cell (underlined) is the experiment's score. red = failed cell,
ink = better than abstaining on everything</div>
{cells_table(metrics)}
{gates_block(e, metrics)}
{train_block(e['artifacts_dir'])}
{timings_block(e)}
<div class="provenance">ts {esc(e['ts'][:19])} · commit {esc(e['git_commit'][:12])} ·
parent {esc((e['parent_commit'] or '')[:12]) or '—'} · artifacts {esc(e['artifacts_dir'] or '—')} ·
agent model {esc(e.get('agent_model') or '—')} · took {fmt_dur(e.get('duration_s'))}</div>
{source_block(e)}
{prompt_block(e)}
</div>
</div>
{figures(e['artifacts_dir'])}
</div></td></tr>""")

    body.append(history_rows_html(hist_rows, hist_eras))
    # The methodology note reads as a footnote to the whole page, so it sits
    # at the foot of it rather than between the chart and the table.
    body.append(f"</tbody></table>{HELP}"
                f"<div class='chart-foot'>{chart_caption}</div></div>")
    body.append("<div id='lightbox'><img alt=''><div class='lb-cap'></div>"
                "<div class='lb-hint'>click anywhere or press Esc to close</div></div>")
    body.append("<div id='tip'></div>" + OVERLAY_HTML
                + "<script>" + PATHS_JS + "</script>" + credits_html()
                + "</body></html>")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(body))
    print(f"wrote {OUT} ({len(exps)} experiments)")
    render_paths(exps)
    render_lineage(exps)
    render_evolution(exps)
    render_overview(exps)
    render_notebook()


if __name__ == "__main__":
    render()
