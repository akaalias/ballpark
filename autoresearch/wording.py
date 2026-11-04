"""House wording for everything the site publishes: "UAV", never "drone".

Alexis's rule (issue #24). Much of the site's text is agent output kept in the
research record (experiment.json summaries, design figures in
experiments.sqlite), and agents say "drone". The record stays as written; the
pages get the house word at the moment they are written:

- the renderers pass every page through `uav()` before writing it;
- infra/build_site.sh runs `python -m autoresearch.wording _site`, which
  rewrites the site copy (HTML, JSON, SVG, Markdown) and fails the build if
  "drone" is still anywhere in it.

    python -m autoresearch.wording DIR            # rewrite, then check
    python -m autoresearch.wording --check DIR    # check only
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_DRONE = re.compile(r"\b(drone)(s?)\b", re.IGNORECASE)
TEXT_SUFFIXES = {".html", ".json", ".svg", ".md", ".txt", ".xml"}


def uav(text: str) -> str:
    """drone -> UAV, drones -> UAVs, any case ("the drone's" -> "the UAV's")."""
    return _DRONE.sub(lambda m: "UAV" + ("s" if m.group(2) else ""), text)


def _files(root: Path):
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in TEXT_SUFFIXES:
            yield f


def rewrite(root: Path) -> int:
    changed = 0
    for f in _files(root):
        t = f.read_text(errors="surrogateescape")
        n = uav(t)
        if n != t:
            f.write_text(n, errors="surrogateescape")
            changed += 1
    return changed


def leftovers(root: Path) -> list[str]:
    hits = []
    for f in _files(root):
        for i, line in enumerate(f.read_text(errors="surrogateescape").splitlines(), 1):
            if _DRONE.search(line):
                hits.append(f"{f}:{i}")
    return hits


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    dirs = [Path(a) for a in argv if a != "--check"] or [Path("_site")]
    for d in dirs:
        if not check_only:
            print(f"wording: {rewrite(d)} files in {d} now say UAV")
        hits = leftovers(d)
        if hits:
            print(f'wording: "drone" is still in {len(hits)} places:', *hits[:20],
                  sep="\n  ", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
