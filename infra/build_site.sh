#!/usr/bin/env bash
# Assemble the public Pages site — used identically by CI and locally, so a
# locally-verified page IS the live page:
#
#   infra/build_site.sh            # renders + assembles into _site/
#   open _site/gallery/index.html  # exact preview of what goes live
#
# One renderer (autoresearch/gallery.py) produces the pages; this script only
# packages its untouched output next to the runs/ artifacts the pages link.
# The published copies of images are slimmed (resized/JPEG/deduped) — the
# full-res originals in runs/ remain the research record.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-_site}"
PY="${PY:-.venv/bin/python}"; [ -x "$PY" ] || PY=python3

$PY -m autoresearch.gallery
$PY -m sim.render_flightpath

rm -rf "$OUT"; mkdir -p "$OUT"
cp -R gallery "$OUT/gallery"
# Committed static page assets (e.g. the overview's geo-tile grid) — plain
# git, small, referenced as assets/... from the root index.html.
[ -d assets ] && cp -R assets "$OUT/assets"

# --- images -------------------------------------------------------------
# Preferred path: site_assets/, the ~20 slimmed images the pages actually
# reference, frozen into plain git by infra/freeze_site_assets.py. Using them
# means the build needs NO Git LFS at all — which is the point: on 2026-07-31
# the account's LFS budget ran out and every deploy failed at `git lfs pull`,
# even though the site's whole image payload is a few MB.
#
# Fallback (no manifest committed): hydrate from runs/ the old way. That still
# works locally, and keeps this script honest if the assets are ever deleted.
ASSETS_MANIFEST="site_assets/manifest.json"
USED_ASSETS=0
if [ -f "$ASSETS_MANIFEST" ]; then
  USED_ASSETS=1
  mkdir -p "$OUT/shared"
  cp site_assets/*.png site_assets/*.jpg "$OUT/shared/" 2>/dev/null || true
  # JSON/MD records stay: they are small, plain-git, and are the browsable
  # provenance the pages link to. Only the binaries came from LFS.
  rsync -a --prune-empty-dirs \
    --include '*/' --include '*.json' --include '*.md' \
    --exclude '*' runs/ "$OUT/runs/"
else
  echo "site_assets/manifest.json absent — hydrating images from runs/ (needs LFS)"
  rsync -a --prune-empty-dirs \
    --include '*/' --include '*.png' --include '*.json' --include '*.md' \
    --exclude '*' runs/ "$OUT/runs/"
fi
cp experiments.sqlite "$OUT/experiments.sqlite" 2>/dev/null || true
# The site's front door is the rendered project overview (gallery.py writes
# it to the repo root alongside the gallery pages).
cp index.html "$OUT/index.html"

# Slim the SITE COPY only: (1) heatmaps -> 1400px JPEG (photographic content,
# PNG was ~4.6 MB each); (2) all run images deduped by content hash into
# shared/ (per-run samples are near-identical copies of the same renders);
# HTML refs rewritten. Keeps the Pages artifact ~50 MB so uploads+deploys
# take about a minute, not a GPU-experiment's runtime.
$PY - "$OUT" "$USED_ASSETS" <<'PYSLIM'
import hashlib
import json
import sys
from pathlib import Path
from PIL import Image

out = Path(sys.argv[1])
used_assets = sys.argv[2] == "1"
renames = {}

if used_assets:
    # The frozen assets are already slimmed and deduped; all that is left is
    # to point the HTML at them. Rewriting from the manifest (rather than
    # re-deriving) means the published pixels are exactly what was verified
    # locally when the assets were frozen.
    manifest = json.loads(Path("site_assets/manifest.json").read_text())
    for src, name in manifest.items():
        renames[src] = f"shared/{name}"
    for html in out.glob("**/*.html"):
        t = html.read_text()
        for old, new in renames.items():
            # Pages sit one directory down (gallery/) or at the root, so both
            # spellings of the same reference have to resolve.
            depth_prefix = "../" if html.parent != out else ""
            t = t.replace(f"../{old}", f"../{new}").replace(
                f'"{old}', f'"{depth_prefix}{new}').replace(
                f"'{old}", f"'{depth_prefix}{new}")
        html.write_text(t)
    for d in sorted((p for p in out.glob("runs/**") if p.is_dir()), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass
    total = sum(f.stat().st_size for f in out.glob("**/*") if f.is_file())
    print(f"site artifact: {total/1e6:,.0f} MB "
          f"({len(set(renames.values()))} images from site_assets/, no LFS needed)")
    raise SystemExit(0)

for f in sorted(out.glob("runs/*/**/heatmap_*.png")):
    try:
        img = Image.open(f).convert("RGB")
    except Exception:
        continue  # unhydrated LFS pointer in a local preview; CI hydrates
    if img.width > 1400:
        img = img.resize((1400, round(img.height * 1400 / img.width)),
                         Image.LANCZOS)
    j = f.with_suffix(".jpg")
    img.save(j, quality=87, optimize=True)
    renames[f.relative_to(out).as_posix()] = j.relative_to(out).as_posix()
    f.unlink()

shared = out / "shared"
shared.mkdir(exist_ok=True)
seen = {}
for f in sorted(out.glob("runs/*/**/*")):
    if not f.is_file() or f.suffix.lower() not in (".png", ".jpg"):
        continue
    h = hashlib.sha256(f.read_bytes()).hexdigest()[:20]
    rel = f.relative_to(out).as_posix()
    if h in seen:
        renames[rel] = seen[h]
        f.unlink()
    else:
        tgt = shared / f"{h}{f.suffix}"
        tgt.write_bytes(f.read_bytes())
        f.unlink()
        seen[h] = tgt.relative_to(out).as_posix()
        renames[rel] = seen[h]

for html in out.glob("**/*.html"):
    t = html.read_text()
    for old, new in renames.items():
        t = t.replace(old, new)
    html.write_text(t)

for d in sorted((p for p in out.glob("runs/**") if p.is_dir()), reverse=True):
    try:
        d.rmdir()
    except OSError:
        pass
total = sum(f.stat().st_size for f in out.glob("**/*") if f.is_file())
print(f"site artifact: {total/1e6:,.0f} MB "
      f"({len(seen)} unique images from {len(renames)} refs)")
PYSLIM

cat > "$OUT/index_redirect_unused.html" <<'EOF'
EOF
rm -f "$OUT/index_redirect_unused.html"
touch "$OUT/.nojekyll"
# House wording (issue #24): "UAV", never "drone" — agent-written text in the
# record says "drone"; the site copy is rewritten, and the build stops if any
# is left.
$PY -m autoresearch.wording "$OUT"
echo "site assembled in $OUT/ — open $OUT/gallery/index.html to preview exactly what goes live"
