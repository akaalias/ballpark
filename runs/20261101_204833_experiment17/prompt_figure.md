# Draw the architecture figure — one narrow job

A separate design agent already decided this iteration's experiment, and
it has already been implemented, trained, and scored — the record is
final. Your ONLY job is to draw its technical architecture diagram for
the gallery. You do not evaluate, second-guess, or change the design;
just illustrate what it already is.

Read the finished experiment record (path given below this prompt) for
`title`, `hypothesis`, `method`, and `architecture.stages` — that is the
diagram's content. Write exactly one field to the output path given below:
```json
{"architecture_svg": "<svg viewBox='0 0 980 300' …>…</svg>"}
```

`architecture_svg` is the **technical architecture diagram** of the model
that was tested — a proper ML-paper figure of the design, shown at the
top of this experiment's gallery entry. **Draw tensors and operations as
the things they are — NOT as labeled boxes:** the input image as a
pixel-textured square; conv feature maps as pseudo-3D slabs that shrink
spatially and deepen in channels; 1-D vectors as thin vertical tick-bars;
fully-connected layers as fans of thin crossing lines between bars;
spatial fields/grids as actually-drawn grids with shaded cells;
decodes/aggregations as thin lines converging from cells onto a point;
samplers/targets as small pictorial glyphs. Captions sit UNDER each
element (name ≤10.5px weight 600, sub-note ≤9.5px); tensor shapes
annotate the flow in small italics (128×128×3 → 8²×128 → 1024).
Losses/targets/samplers live in a lane below the inference path,
annotated with slanted dashed leader lines to small text — no boxes
around them. Draw the two FROZEN endpoints — the camera-frame input and
the (lat, lon, confidence) output — entirely in #9b998c with a small
italic "frozen contract" tag: they are fixed by the harness and outside
the design's search space. Everything between them is the design: draw
it in ink, with red only on what THIS experiment changed (per
`architecture.stages[].changed`). Style contract, so all experiments'
figures read as one paper: viewBox width 980 (height as tall as the
layout needs, 240–800; see the anchors below);
transparent background; ink #111111, captions #6b6a60,
annotations/arrows #9b998c; **#8c2f1f (red) reserved for exactly what
this experiment changed**; training-only elements in #8a6a1e (or red if
changed); font-family Palatino,Georgia,serif; stroke-width ≈1.2
elements, 1 arrows; fills only faint tints (opacity ≤ .12); no
gradients, no icons, no emoji. Inference flows left → right from camera
frame to (lat, lon, confidence).

**The camera frame is a fixed glyph — copy it verbatim.** It must look
like an actual nadir frame of terrain (streets, building footprints),
not abstract pixels, and be identical in every figure. Use exactly this
snippet, substituting Y for the frame's top edge (Y = H/2 − 38, so the
frame is centred on the canvas — see the anchors below; captions go under
it as usual — note
128²×3 is the model input crop, ~1 m/px, not the sensor's native
resolution):
```svg
<g id='cam-terrain'><rect id='frozen-input' x='26' y='Y' width='76'
   height='76' fill='#f6f4ea' stroke='#9b998c' stroke-width='1.6'/>
<path d='M31 Y+47 L97 Y+27' stroke='#e6e3d4' stroke-width='5' fill='none'/>
<path d='M59 Y+6 L49 Y+70' stroke='#e6e3d4' stroke-width='3.5' fill='none'/>
<rect x='34' y='Y+9' width='12' height='8' fill='#d9d5c3' transform='rotate(-8 40 Y+13)'/>
<rect x='78' y='Y+8' width='10' height='11' fill='#cfccbd'/>
<rect x='35' y='Y+57' width='13' height='8' fill='#d9d5c3'/>
<rect x='75' y='Y+50' width='10' height='9' fill='#cfccbd' transform='rotate(6 80 Y+54)'/>
<rect x='54' y='Y+31' width='9' height='8' fill='#d9d5c3' opacity='.85'/>
<ellipse cx='86' cy='Y+63' rx='8' ry='6' fill='#8a6a1e' opacity='.12'/>
<circle cx='40' cy='Y+28' r='.7' fill='#6b6a60' opacity='.5'/>
<circle cx='70' cy='Y+14' r='.7' fill='#6b6a60' opacity='.5'/>
<circle cx='92' cy='Y+36' r='.7' fill='#6b6a60' opacity='.5'/>
<circle cx='52' cy='Y+52' r='.7' fill='#6b6a60' opacity='.5'/>
<circle cx='82' cy='Y+70' r='.7' fill='#6b6a60' opacity='.5'/>
<circle cx='31' cy='Y+42' r='.7' fill='#6b6a60' opacity='.5'/></g>
```
(Y+n means the literal number Y plus n — compute the values. Every glyph
element stays strictly INSIDE the frame border — figcheck rejects a
figure without the `cam-terrain` group.) Draw the receptive-field square
and kernel-projection lines on top of it as usual.

**Anchor the frozen endpoints identically in every figure** so all
experiments' figures line up when compared down the gallery page:

- **Both endpoints sit on the canvas's horizontal centre line, y = H/2**,
  where H is your viewBox height. The camera-frame square is centred on it
  (top edge Y = H/2 − 38) and starts at x=26 (captions centred on x=53).
  The output is right-anchored on the same line: decode crosshair at
  (812, H/2), and the "(lat, lon, confidence)" text with
  text-anchor=start at x=828, baseline y = H/2 + 5 (13 px type). Its small
  "frozen contract" tag goes above it and "position fix + confidence"
  below it.
- Choose H first, then build around the centre line. A single inference
  path runs along it. Two parallel members (a committee) go one above and
  one below it, symmetrically, and rejoin on it before the output. Training
  signals go in their own band at the bottom, under a "TRAINING SIGNALS —
  NEVER FLY" heading with nothing else on that row.
- Mark the anchors machine-checkably: the camera-frame rect carries
  `id='frozen-input'`, the output text element carries
  `id='frozen-output'`. figcheck enforces both positions.

Lay the elements between them out on a running x-cursor with generous
spacing so nothing overlaps. Every label, line and shape stays inside the
canvas with a few px to spare — nothing may be cut off at any edge.

**Readability contract** (figcheck verifies this geometrically): no
text may overlap other text, and no line (arrows, converge fans,
leader lines) may pass through a label — a line may END at a label,
never cross one. Rules distilled from review rounds: dashed leaders
run as orthogonal L-routes through empty lanes, never diagonally
across components; converge/vote lines originate at their true
sources (the actual dots or window cells, not generic positions);
every element owns its caption — adjacent columns stagger their
caption rows instead of colliding; prefer moving a label into empty
horizontal space over stacking it against a neighbor. Width is fixed
at 980, but **height is yours**: grow the viewBox (240–800) whenever
more vertical room makes the layout cleaner rather than cramming.

**Save early, save often.** Write your first complete figure to the output
path within your first few turns, and overwrite it there after every
improvement. Your turn budget is capped: a run that hits the cap keeps
whatever is at the output path, and a figure that still has one or two
collisions is far better than none (experiment 10's agent refined a
draft in a scratch file for 46 turns and left nothing). Keep any helper
script or scratch file inside this run's directory (the folder of the
output path), never at the top of `runs/`.

**Look at it, then validate.** Geometry checks cannot see a chaotic
layout, so after every substantial revision render the SVG and look at the
picture: extract the SVG from the JSON into a file in this run's directory,
then `rsvg-convert -w 1600 -b white <that.svg> -o <that.png>` and Read the
PNG. Check, in the image itself: no text touching other text, a line or a
shape; no label cut off at an edge; dashed training leaders that run as
clean L-routes to the thing they annotate and stop there; no line that
goes nowhere; input and output level with each other on the centre line.
Then run `.venv/bin/python -m autoresearch.figcheck <your output path>` —
it checks the anchors, canvas bounds, centring, overlaps and well-formed
XML (escape `&` in text as `&amp;`). Revise until the picture looks clean
AND figcheck prints PASS. Consecutive conv layers must be tied by kernel-projection
lines (small kernel square on one face, faint lines converging to a cell
on the next) so the encoder reads as one computation, not boxes in a
row. `archive/arch_svg_reference.py` is the concrete reference
implementation of this entire visual language — read it and reuse its
helper geometry for any stage that already has one. **Read it; never run
it.** It is an archived script for a different era's experiments and, run
with its write flag, overwrites other experiments' figures in the database.
Your only output is the one file named below.

## Hard rules

- Edit ONLY the output path given below. You never edit `model/`,
  `experiment.json`, or any file listed in `/FROZEN`.
- Do not re-evaluate the experiment, change its hypothesis, or query the
  lineage DB for anything beyond what's needed to draw the figure
  consistently with prior stage names.

## Paths for this run (resolve the placeholders referenced above)
- Read the experiment record from: `runs/20261101_204833_experiment17/design_for_figure.json`
- Write ONLY this output file: `runs/20261101_204833_experiment17/figure.json` — a single JSON object of the
  form {"architecture_svg": "<svg …>"} and nothing else.
