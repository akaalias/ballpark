"""The research-evolution figure, generated from data instead of hand-placed.

WHY THIS EXISTS
---------------
This figure used to be a 13.8 KB single-line SVG literal in gallery.py with
every coordinate baked in, plus an `evolution_fork_svg()` that did eight
find-and-replace edits on that string to lift the berlin-slim work onto its
own lane. Three lists (TRUNK, DEAD, SUPER) sat next to it looking like the
source of truth and were not: the literal had been hand-edited past them, so
SUPER held five entries while the figure drew six, and TRUNK held thirteen
while the figure drew nine.

The cost showed up the first time anything was added. Placing one small branch
took three attempts -- once outside the era band, once anchored to y=256, which
after the fork surgery is the DORMANT main line, so the new branch was drawn
descending from an abandoned trunk rather than from the champion that produced
it. Both were invisible in the markup and obvious in a screenshot. A figure
that cannot be reviewed by reading it is a figure that ships wrong.

So: the marks below are data, and this module lays them out. Adding a branch is
appending a tuple; moving it is changing its lane. The fork is part of layout
rather than surgery performed on the output.

THE TIME AXIS IS NOT LINEAR, AND THAT IS DELIBERATE
Each day gets width proportional to how much happened in it: the six silent
days of 24-29 July are 96 px each, while 22 and 31 July are ~1,000. The widths
were tuned by hand in the original figure and are preserved here verbatim
rather than re-derived, because they encode a judgement about what deserves
room -- not something to recompute. DAY_W is that judgement, written down.

Coordinates are recovered from the original figure to within 0.15 px, so this
regenerates it rather than redrawing it.
"""

# --- the time axis -------------------------------------------------------
# Left edge of each day, and how wide that day is drawn. Activity-weighted.
# Days are numbered from 20 July; 32 is 1 August, which is why DAY_LABEL
# exists rather than an f-string that assumes the month.
DAY_W = {20: 824.0, 21: 304.0, 22: 928.0, 23: 824.0, 24: 200.0, 25: 96.0,
         26: 96.0, 27: 96.0, 28: 96.0, 29: 96.0, 30: 928.0, 31: 1031.8,
         32: 346.7}
# 2026-09-28 the capture-holdout era opened (day 90), 2026-10-30 its loop
# started (day 122). The two months between hold one commit (the rename to
# Ballpark) and the month between held nothing: 6 px a day, so the quiet
# reads as quiet instead of stretching the figure by a screen per week.
for _d in range(33, 90):
    DAY_W[_d] = 6.0
DAY_W[90] = 928.0
for _d in range(91, 122):
    DAY_W[_d] = 6.0
DAY_W[122] = 520.0
LAST_DAY = 122
DAY_X = {21: 306.5}
for _d in range(22, LAST_DAY + 1):
    DAY_X[_d] = DAY_X[_d - 1] + DAY_W[_d - 1]
DAY_X[20] = DAY_X[21] - DAY_W[20]
GRID_MIN_W = 60.0         # a day narrower than this gets no gridline/label


def DAY_LABEL(d: int) -> str:
    """Day-of-project (20 = 20 July 2026, 32 = 1 August, 90 = 28 September,
    122 = 30 October) -> calendar label."""
    import datetime
    date = datetime.date(2026, 7, 20) + datetime.timedelta(days=d - 20)
    return f"{date.day} {date.strftime('%b')}"


TRUNK_Y = 256.0
BRANCH_Y = 216.0          # the berlin-slim fork's own lane
TRUNK_X0 = 52.0
TRUNK_X1 = DAY_X[LAST_DAY] + DAY_W[LAST_DAY] - 40.0   # where the line stops
VIEW_W, VIEW_H = int(TRUNK_X1 + 289), 608
SPLIT_D = 31.38           # berlin-slim: the trunk forks here
GAPS = [
    (24.0200, 30.7700, "six days of silence"),
    (32.35, 89.80, "two months of silence &mdash; one commit, the rename to Ballpark"),
    (90.95, 121.85, "a month on hold &mdash; the laptop was needed"),
]
GAP = GAPS[0]             # kept for anything that still reads the old name


def x(day: float) -> float:
    """Day (fractional) -> pixel. Piecewise-linear, one segment per day."""
    d = int(day)
    return DAY_X[d] + (day - d) * DAY_W[d]


# --- the marks -----------------------------------------------------------
# (day, label, 'above'|'below')  -- where the label sits relative to the line
TRUNK_NODES = [
    (20.73, "Bootstrap", "below"),
    (20.80, "1 m/px orthophotos", "above"),
    (20.86, "Relighting rebuilt", "below"),
    (21.5099, "Goes public", "above"),
    (22.85, "Pivot enforced in code", "below"),
    (23.60, "One git writer", "below"),
    (30.8699, "PNG decode fix", "below"),
    (31.38, "berlin-slim branch", "below"),
    (31.6001, "0.040 - 96.5% usable", "below"),
    (32.10, "back to berlin-slim", "below"),
    (90.75, "held-out photograph", "below"),
    (122.47, "loop resumes, locally", "above"),
]

# (from_day, to_day, lane_y, label, end_label) -- tried, then abandoned
DEAD = [
    (20.73, 20.80, 128, "Sentinel-2 10 m/px", "dropped in 20 min"),
    (22.50, 22.85, 128, "Prompt-only pivot rules", "never worked"),
    (30.88, 30.91, 64, "3D shadow relighting", "parked"),
    (30.90, 30.94, 128, "Retrieval-index probe", "parked, not disproven"),
    # The Prignitz probe and the era it opened are ONE line, because the
    # probe's 0.113 is the only reason the era existed. It ran three
    # experiments in four hours and was rewound to the commit before it on
    # 1 August; the trunk it left continues past its terminator, which is
    # the whole point of drawing it up here rather than as the live line.
    (31.6100, 31.9200, 128, "Prignitz probe · 0.113", "era opened, then closed"),
    # Google's tiles were the first "second photograph" tried, and the one that
    # can never be data: their terms forbid it. Kept as a hand-run smoke test.
    (90.55, 90.68, 64, "Google tiles as a second photograph", "licence: smoke test only"),
]

# (from_day, to_day, lane_y, label, end_label) -- ran as mainline, replaced
SUPER = [
    (20.75, 31.4701, 320, "Region-holdout evaluation", "→ held-out viewpoints"),
    (20.75, 31.5201, 384, "Median error as the score", "→ mission score"),
    (21.3299, 23.60, 448, "RunPod - rented 4090 pod", "→ Modal"),
    (23.60, 30.96, 448, "Modal A100 - serverless", "credits out → local M1"),
    (23.56, 23.85, 512, "Berlin-only scope cut", "→ 4 areas restored"),
    (31.5201, 31.5501, 512, "Geometric mean", "→ mission score"),
    # Lane 352 rather than 320: the region-holdout branch's end label sits at
    # y 324 right where this one would start, 82 px later on the same lane.
    (31.5501, 90.75, 352, "Same-photograph evaluation", "→ held-out photograph"),
]

# (day, label) -- something broke
INCIDENTS = [
    (22.03, "pod disk full"),
    (22.43, "Fable hits its cap"),
    (22.65, "CI out of disk"),
    (22.80, "two writers on main"),
    (23.58, "pod won&#39;t resume"),
    (24.02, "loop dies unnoticed"),
    (30.96, "credits exhausted"),
]

# (from_day, to_day, lane_y, label) -- an insight that changed the project
MERGES = [
    (22.5741, 22.85, 192, "enforce it in the source, not the prompt"),
    (23.2626, 23.60, 192, "the problem is two writers, not the platform"),
    (30.6830, 30.8699, 192, "stop guessing, profile it"),
    (31.2006, 31.4701, 192, "it is never shown the places it is tested on"),
    (31.2967, 31.5501, 128, "the score must BE the product requirement"),
    (31.4213, 31.6001, 64, "it was starved, not refuted"),
    (90.58, 90.75, 192, "it knew one photograph, not the city"),
]

# (from_day, to_day, lane_y, label, end_label) -- ran, still open.
# Hangs off whichever line is live at from_day, which after SPLIT_D is the
# fork lane -- the distinction that was got wrong by hand twice.
# Empty since 1 August: the Prignitz probe was the only entry and it is now
# a closed detour in DEAD. Kept as a list rather than deleted because the
# category is real and the next side-branch belongs here; render_evolution
# drops the legend swatch when it is empty.
OPEN = []


def _f(v):
    """Match the original figure's number formatting (one decimal, no -0.0)."""
    return f"{v + 0.0:.1f}"


def _branch(kind, key, x0, y0, x1, lane, lab, endlab, end_mark):
    """One branch: elbow off a horizontal line, run flat, terminate, label.

    The elbow's control offsets shrink for short branches so a 28 px stub does
    not overshoot its own endpoint -- the original figure's behaviour, kept.
    """
    s = min(1.0, (x1 - x0) / 45.0)
    c1, ex = x0 + 15.0 * s, x0 + 24.0 * s
    out = [f"<path class='evo-{kind}' data-k='{key}' d='M{_f(x0)},{y0:.0f} "
           f"C{_f(c1)},{y0:.0f} {_f(x0)},{_f(lane)} {_f(ex)},{_f(lane)} "
           f"L{_f(x1)},{_f(lane)}'/>"]
    if end_mark == "x":
        out.append(
            f"<g class='evo-end' data-k='{key}'>"
            f"<line x1='{_f(x1 - 4.5)}' y1='{_f(lane - 4.5)}' "
            f"x2='{_f(x1 + 4.5)}' y2='{_f(lane + 4.5)}'/>"
            f"<line x1='{_f(x1 - 4.5)}' y1='{_f(lane + 4.5)}' "
            f"x2='{_f(x1 + 4.5)}' y2='{_f(lane - 4.5)}'/></g>")
    else:
        out.append(f"<circle class='evo-{end_mark}' data-k='{key}' "
                   f"cx='{_f(x1)}' cy='{_f(lane)}' r='5'/>")
    cls = {"dead": "d", "super": "s", "open": "o"}[kind]
    out.append(f"<text class='evo-lab {cls}' data-k='{key}' x='{_f(ex + 8)}' "
               f"y='{_f(lane - 10)}' text-anchor='start'>{lab}</text>")
    out.append(f"<text class='evo-endlab {cls}' data-k='{key}' "
               f"x='{_f(x1 + 11)}' y='{_f(lane + 4)}' "
               f"text-anchor='start'>{endlab}</text>")
    return "".join(out)


def build_svg() -> str:
    """The whole figure, forked, in the original's element order (z-order)."""
    split_x = x(SPLIT_D)
    p = [f"<svg id='evo-svg' viewBox='0 0 {VIEW_W} {VIEW_H}' width='{VIEW_W}' "
         f"height='{VIEW_H}' role='img' "
         f"aria-label='research process as a branching graph'>"]

    for d in range(21, LAST_DAY + 1):             # day rules
        if DAY_W[d] < GRID_MIN_W:                 # compressed quiet days
            continue
        gx = x(d)
        p.append(f"<line class='evo-grid' x1='{_f(gx)}' y1='30' "
                 f"x2='{_f(gx)}' y2='574'/>")
        p.append(f"<text class='evo-day' x='{_f(gx + 6)}' y='22'>"
                 f"{DAY_LABEL(d)}</text>")

    for i, (ga, gb, glab) in enumerate(GAPS):     # the silences
        g0, g1 = x(ga), x(gb)
        gy = 242 if ga < SPLIT_D else 202         # on whichever line is live
        key = "gap" if i == 0 else f"gap{i}"
        p.append(f"<rect class='evo-gap' data-k='{key}' x='{_f(g0)}' y='{gy}' "
                 f"width='{_f(g1 - g0)}' height='28'/>")
        p.append(f"<text class='evo-gaplab' data-k='{key}' x='{_f((g0 + g1) / 2)}' "
                 f"y='{gy - 7}'>{glab}</text>")

    # The trunk stops at the fork; what continues is main, dormant since.
    p.append(f"<line class='evo-trunk' x1='{_f(TRUNK_X0)}' y1='{TRUNK_Y:.0f}' "
             f"x2='{_f(split_x)}' y2='{TRUNK_Y:.0f}'/>")
    p.append(f"<line class='evo-dormant' x1='{_f(split_x)}' y1='{TRUNK_Y:.0f}' "
             f"x2='{_f(TRUNK_X1)}' y2='{TRUNK_Y:.0f}'/>")
    p.append(f"<text class='evo-dormantlab' x='{_f((split_x + TRUNK_X1) / 2)}' "
             f"y='{TRUNK_Y + 15:.0f}' text-anchor='middle'>"
             f"main — unchanged since</text>")
    p.append(f"<path class='evo-branch' d='M{_f(split_x)},{TRUNK_Y:.0f} "
             f"C{_f(split_x + 15)},{TRUNK_Y:.0f} {_f(split_x + 22)},"
             f"{BRANCH_Y:.0f} {_f(split_x + 40)},{BRANCH_Y:.0f} "
             f"L{_f(TRUNK_X1)},{BRANCH_Y:.0f}'/>")

    def base_y(day):
        """Which line is live at this instant -- trunk, or the fork lane."""
        return BRANCH_Y if day > SPLIT_D else TRUNK_Y

    for i, (d0, d1, lane, lab, end) in enumerate(DEAD):
        p.append(_branch("dead", f"d{i}", x(d0), base_y(d0), x(d1), lane,
                         lab, end, "x"))
    for i, (d0, d1, lane, lab, end) in enumerate(SUPER):
        p.append(_branch("super", f"s{i}", x(d0), base_y(d0), x(d1), lane,
                         lab, end, "endr"))

    for i, (d, lab) in enumerate(INCIDENTS):      # pins below the red lanes
        px = x(d)
        p.append(f"<path class='evo-pindot' data-k='i{i}' d='M{_f(px)},442.5 "
                 f"l5.5,5.5 l-5.5,5.5 l-5.5,-5.5 z'/>")
        p.append(f"<line class='evo-pin' data-k='i{i}' x1='{_f(px)}' "
                 f"y1='455.0' x2='{_f(px)}' y2='468.0'/>")
        p.append(f"<text class='evo-pinlab' data-k='i{i}' x='{_f(px)}' "
                 f"y='479.0' text-anchor='middle'>{lab}</text>")

    for i, (d0, d1, lane, lab) in enumerate(MERGES):
        x0, x1 = x(d0), x(d1)
        ty = base_y(d1) - 9          # where the arrow's point lands
        # The elbow's second control point sits 16 px below the lane, but
        # must stay clear of the target: on a short drop (a merge landing
        # on the fork lane rather than the trunk) 16 px would sit PAST the
        # endpoint and bow the corner backwards.
        c2 = min(lane + 16, ty - 7)
        p.append(f"<path class='evo-merge' data-k='m{i}' d='M{_f(x0)},"
                 f"{_f(lane)} L{_f(x1 - 22)},{_f(lane)} C{_f(x1 - 7)},"
                 f"{_f(lane)} {_f(x1)},{_f(c2)} {_f(x1)},{_f(ty)}'/>")
        p.append(f"<path class='evo-arrow' data-k='m{i}' d='M{_f(x1 - 4.5)},"
                 f"{_f(ty - 2)} L{_f(x1)},{_f(ty + 7)} L{_f(x1 + 4.5)},"
                 f"{_f(ty - 2)} z'/>")
        p.append(f"<circle class='evo-mstart' data-k='m{i}' cx='{_f(x0)}' "
                 f"cy='{_f(lane)}' r='3.5'/>")
        p.append(f"<text class='evo-mlab' data-k='m{i}' x='{_f(x1 - 26)}' "
                 f"y='{_f(lane - 9)}' text-anchor='end'>{lab}</text>")

    for i, (d, lab, side) in enumerate(TRUNK_NODES):
        px, y0 = x(d), base_y(d)
        tick = y0 + 15 if side == "below" else y0 - 12
        p.append(f"<line class='evo-tick' x1='{_f(px)}' y1='{y0:.0f}' "
                 f"x2='{_f(px)}' y2='{_f(tick)}'/>")
        p.append(f"<circle class='evo-node' data-k='t{i}' cx='{_f(px)}' "
                 f"cy='{y0:.0f}' r='6'/>")
        ly = y0 + 24 if side == "below" else y0 - 24
        p.append(f"<text class='evo-tlab' data-k='t{i}' x='{_f(px)}' "
                 f"y='{ly:.0f}'>{lab}</text>")

    for i, (d0, d1, lane, lab, end) in enumerate(OPEN):
        p.append(_branch("open", f"o{i}", x(d0), base_y(d0), x(d1), lane,
                         lab, end, "openend"))

    return "".join(p) + "</svg>"


# Where the newest work sits, for the page's opening scroll.
LATEST_X = x(TRUNK_NODES[-1][0])
