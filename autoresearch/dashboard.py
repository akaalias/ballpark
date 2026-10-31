"""The loop's dashboard: one terminal screen that says what the autoresearch
loop is doing right now (issue #2). Read-only — it never writes to the
record; its only side effect is the graceful-stop flag (`s` toggles
state/stop, which loop.sh already honours).

    ./autoresearch/dashboard.sh            # live; q closes it, the loop keeps running
    .venv/bin/python -m autoresearch.dashboard --once [--width 160 --height 45]
                                            # one static render (for checking the layout)

Everything on screen comes from files the loop already writes:
  state/phase.json            phase, experiment number/target, start times, best,
                              the design once the design agent has written it
  runs/<ts>_experiment<N>/    the run in flight: train_berlin.log (one line per
                              epoch), agent_*.json (cost/turns when an agent
                              finishes), metrics.json (the score when it lands)
  runs/*/timings.json         how long each phase took in recent runs — the
                              basis of every "about … left"
  experiments.sqlite          recorded experiments (the recent list, the era's
                              baseline and best)
  state/loop.log              the loop's own output, when it is launched
                              detached (next step on issue #2)
Modelled on cryptos/tools/loop_tui.py so the two read as siblings.
"""
import collections
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from rich import box
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
PHASE = STATE / "phase.json"
STOP = STATE / "stop"
PIDF = STATE / "loop.pid"
LOOP_LOG = STATE / "loop.log"
DB = ROOT / "experiments.sqlite"
RUNS = ROOT / "runs"

ACCENT, MUTED, OK, BAD = "#c0614a", "grey58", "green3", "red3"
ACCENT_SOFT = "#e0a191"

# The loop's phases in the order loop.sh reports them (figure only when
# DRAW_FIGURES=1, which is the default again since 2026-10-31).
PHASES = ["design", "figure", "implement", "train", "score", "publish"]
# phase.json phase -> timings.json key(s) used for the estimate
TIMING_KEYS = {"design": ("agent_design_s",), "implement": ("agent_impl_s",),
               "train": ("train_wall_s",), "score": ("score_s",),
               "publish": ("samples_s", "gallery_s")}
DEFAULTS = {"design": 600, "figure": 480, "implement": 120, "train": 6000,
            "score": 60, "publish": 30}


# ---- small helpers ----------------------------------------------------------------------------------

def mins(sec):
    sec = int(max(0, sec))
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def day(t):
    d = datetime.fromtimestamp(t)
    return d.strftime("%H:%M") if d.date() == datetime.now().date() else d.strftime("%a %-d %b, %H:%M")


def bar(frac, width=30, colour=ACCENT):
    n = max(0, min(width, round(frac * width)))
    return Text("█" * n, style=colour) + Text("█" * (width - n), style="grey23")


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def loop_running():
    """True if a loop.sh of THIS repository is alive: the pid file (detached
    launch), else any process whose command names autoresearch/loop.sh."""
    if PIDF.exists() and alive(PIDF.read_text().strip()):
        return True
    try:
        out = subprocess.run(["pgrep", "-f", "autoresearch/loop.sh"], capture_output=True, text=True).stdout
        return any(p.strip() and int(p) != os.getpid() for p in out.split())
    except (OSError, ValueError):
        return False


def branch():
    try:
        return subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "?"


BRANCH = branch()


def phase_state():
    return read_json(PHASE) or {}


def run_dir(n):
    """The run directory of experiment n (newest if several)."""
    try:
        cands = sorted(RUNS.glob(f"*_experiment{n}"))
    except OSError:
        return None
    return cands[-1] if cands else None


def median_timings(k=5):
    files = sorted(RUNS.glob("*/timings.json"))[-k:]
    acc = {}
    for f in files:
        d = read_json(f) or {}
        for key, v in d.items():
            try:
                acc.setdefault(key, []).append(float(v))
            except (TypeError, ValueError):
                pass
    med = {key: sorted(v)[len(v) // 2] for key, v in acc.items()}
    est = {}
    for ph in PHASES:
        keys = TIMING_KEYS.get(ph)
        if keys and all(key in med for key in keys):
            est[ph] = sum(med[key] for key in keys)
        else:
            est[ph] = DEFAULTS[ph]
    # the figure agent is not in timings.json; take it from recent agent_figure.json
    figs = [read_json(f) for f in sorted(RUNS.glob("*/agent_figure.json"))[-k:]]
    ds = sorted(float(d["duration_ms"]) / 1000 for d in figs if d and d.get("duration_ms"))
    if ds:
        est["figure"] = ds[len(ds) // 2]
    est["total"] = med.get("total_s", sum(est[p] for p in PHASES))
    return est


EPOCH_RE = re.compile(r"epoch (\d+)/(\d+) loss=([\d.]+) top1=([\d.]+)")


def train_progress(rd):
    """(epochs done, epochs total, last line, per-capture top1) from the train log."""
    if not rd:
        return None
    log = rd / "train_berlin.log"
    try:
        lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
    except OSError:
        return None
    k, e, last_epoch = 0, None, None
    for ln in lines:
        m = EPOCH_RE.search(ln)
        if m:
            k, e, last_epoch = int(m.group(1)), int(m.group(2)), ln
    caps = {}
    if last_epoch:
        mc = re.search(r"top1_by_capture=(\{.*\})", last_epoch)
        if mc:
            try:
                caps = eval(mc.group(1), {"__builtins__": {}})  # a dict literal the trainer printed
            except Exception:  # noqa: BLE001
                caps = {}
    return {"done": k, "total": e, "last": lines[-1] if lines else "", "caps": caps,
            "mtime": log.stat().st_mtime if log.exists() else 0}


def training_processes():
    try:
        out = subprocess.run(["ps", "-eo", "pid,pcpu,command"], capture_output=True, text=True).stdout
    except OSError:
        return 0, 0.0
    n, cpu = 0, 0.0
    for ln in out.splitlines():
        if ("model.train" in ln or "model/train.py" in ln or "pipeline.score" in ln) and "grep" not in ln:
            n += 1
            try:
                cpu += float(ln.split()[1])
            except (IndexError, ValueError):
                pass
    return n, cpu


def recent(n=8):
    if not DB.exists():
        return [], {}
    try:
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        rows = c.execute("SELECT id, title, category, primary_metric, kept, duration_s, metrics_json, kind "
                         "FROM experiments ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        allrows = c.execute("SELECT id, primary_metric, kept, duration_s, metrics_json FROM experiments "
                            "WHERE kind != 'holdout_check' ORDER BY id ASC").fetchall()
        c.close()
    except sqlite3.Error:
        return [], {}
    era = {}
    if allrows:
        scored = [(r[1], r[0]) for r in allrows if r[1] is not None and r[1] < 1e17]
        era["n"] = len(allrows)
        era["kept"] = sum(1 for r in allrows if r[2])
        era["baseline"] = scored[0] if scored else None
        era["best"] = min(scored) if scored else None
        durs = sorted(r[3] for r in allrows if r[3])
        era["median_s"] = durs[len(durs) // 2] if durs else None
        era["last_metrics"] = read_json_s(allrows[-1][4])
        era["last_id"] = allrows[-1][0]
    return rows, era


def read_json_s(s):
    try:
        return json.loads(s or "{}")
    except ValueError:
        return {}


def rates(metrics):
    """(eval bucket name, score, usable, false) + train diagnostics from a metrics dict."""
    try:
        area = metrics["areas"][0]
        name, b = next(iter(area["buckets"].items()))
        diag = area.get("train_capture_diagnostics") or {}
        return name, b, diag
    except (KeyError, IndexError, TypeError, StopIteration, AttributeError):
        return None, None, {}


# ---- the activity feed ------------------------------------------------------------------------------

class Activity:
    """Events derived from the files the loop writes, read incrementally:
    phase changes, each training epoch as it lands, each agent's finish (with
    turns and cost), the score, the record. Plus state/loop.log lines when the
    loop runs detached."""

    def __init__(self):
        self.events = collections.deque(maxlen=60)
        self.seen = set()
        self.phase = None
        self.epochs = 0
        self.run = None
        self.log_pos = 0
        self.last_id = None
        self.seeded = False

    def add(self, t, text, key=None):
        if key is not None:
            if key in self.seen:
                return
            self.seen.add(key)
        self.events.append((t, text))

    def poll(self):
        pj = phase_state()
        n = pj.get("iter")
        rd = run_dir(n) if n else None
        if rd != self.run:
            self.run, self.epochs = rd, 0
        ph = pj.get("phase")
        if ph and ph != self.phase:
            self.phase = ph
            msg = {"idle": "loop finished its batch — idle", "synced": "record pushed",
                   "waiting": "paused: waiting for agent capacity"}.get(ph, f"phase {ph}")
            if ph == "design":
                msg = f"experiment {n}: {msg}"
            self.add(float(pj.get("phase_started") or time.time()), msg, key=("phase", n, ph, pj.get("phase_started")))
        if rd:
            for name, label in (("agent_design.json", "design agent finished"),
                                ("agent_figure.json", "figure drawn"),
                                ("agent_impl.json", "implementation agent finished")):
                f = rd / name
                if f.exists() and ("agent", str(rd), name) not in self.seen:
                    d = read_json(f)
                    if d is None or "duration_ms" not in d:
                        continue   # still being written
                    cost = d.get("total_cost_usd")
                    txt = f"{label}: {d.get('num_turns', '?')} turns, {mins(float(d['duration_ms']) / 1000)}"
                    if cost is not None:
                        txt += f", ${cost:.2f}"
                    if d.get("is_error") or d.get("subtype") not in (None, "success"):
                        txt += f" — {d.get('subtype')}"
                    self.add(f.stat().st_mtime, txt, key=("agent", str(rd), name))
            tp = train_progress(rd)
            if tp and tp["done"] > self.epochs:
                for ln in (rd / "train_berlin.log").read_text().splitlines():
                    m = EPOCH_RE.search(ln)
                    if m and int(m.group(1)) > self.epochs:
                        k, e = int(m.group(1)), int(m.group(2))
                        self.add(tp["mtime"] if k == tp["done"] else tp["mtime"] - (tp["done"] - k) * 60,
                                 f"epoch {k}/{e} · loss {m.group(3)} · top-1 {float(m.group(4)) * 100:.0f}%",
                                 key=("epoch", str(rd), k))
                self.epochs = tp["done"]
            mf = rd / "metrics.json"
            if mf.exists():
                name, b, _ = rates(read_json(mf) or {})
                if b:
                    self.add(mf.stat().st_mtime,
                             f"scored {b.get('mission_score', float('nan')):.3f} on {name} · usable "
                             f"{(b.get('usable_fix_rate') or 0) * 100:.1f}% · false fix "
                             f"{(b.get('false_fix_rate') or 0) * 100:.1f}%", key=("score", str(rd)))
        rows, era = recent(1)
        if rows:
            r = rows[0]
            if self.last_id is not None and r[0] != self.last_id:
                verdict = "kept — new best" if r[4] else "reverted"
                self.add(time.time(), f"experiment {r[0]} recorded: {verdict}", key=("record", r[0]))
            self.last_id = r[0]
        # the loop's own output, if it is being logged (detached launch)
        try:
            if LOOP_LOG.exists() and LOOP_LOG.stat().st_mtime >= float(pj.get("iter_started") or 0):
                with open(LOOP_LOG, "rb") as f:
                    if not self.seeded:
                        f.seek(0, 2)
                        self.log_pos = max(0, f.tell() - 4000)
                    f.seek(self.log_pos)
                    chunk = f.read()
                nl = chunk.rfind(b"\n")
                if nl >= 0:
                    for ln in chunk[: nl + 1].decode("utf-8", "replace").splitlines():
                        ln = ln.strip()
                        if ln and not ln.startswith("wrote /"):
                            self.add(time.time() if self.seeded else LOOP_LOG.stat().st_mtime, ln[:160])
                    self.log_pos += nl + 1
        except OSError:
            pass
        self.seeded = True


# ---- panels -----------------------------------------------------------------------------------------

def header():
    if STOP.exists():
        state = Text("stopping after this experiment", style=ACCENT)
    elif loop_running():
        state = Text("running", style=OK)
    else:
        state = Text("not running", style=BAD)
    t = Table.grid(expand=True)
    t.add_column(); t.add_column(justify="right")
    t.add_row(Text(f"BALLPARK · {BRANCH} · autoresearch loop", style="bold"),
              Text("compute: M1, local   ", style=MUTED) + Text("loop: ") + state
              + Text(f"   {time.strftime('%H:%M')}", style=MUTED))
    return Panel(t, box=box.HEAVY, style="bold")


def experiment_panel(tick, est):
    pj = phase_state()
    n, target = pj.get("iter"), pj.get("iterations")
    ph = pj.get("phase")
    now = time.time()
    rd = run_dir(n) if n else None
    design = pj.get("design") or {}
    if not design and rd:
        design = read_json(rd / "experiment.json") or read_json(RUNS / "pending_experiment.json") or {}
    lines = []
    if not n or ph in (None, "idle") or (ph == "synced" and not loop_running()):
        lines.append(Text("no experiment in flight — the loop is idle; the best result stands", style=MUTED))
        if pj.get("best") is not None:
            lines.append(Text(f"best so far {pj['best']:.3f}", style="default"))
        return Panel(Group(*lines), title="current experiment", title_align="left", border_style=ACCENT)

    lines.append(Text.assemble((f"experiment {n}", "bold"), (f" of {target}" if target else "", MUTED),
                               ("   " + (design.get("category") or "").upper(), ACCENT + " bold")))
    lines.append(Text(design.get("title") or "the design agent is writing the proposal…",
                      style="default" if design.get("title") else MUTED))
    # phase row
    steps = Text("phase    ", style=MUTED)
    cur = PHASES.index(ph) if ph in PHASES else -1
    ph_el = now - float(pj.get("phase_started") or now)
    for i, p in enumerate(PHASES):
        if i < cur:
            if p == "figure" and rd and not (rd / "agent_figure.json").exists():
                steps += Text(f"– {p}  ", style=MUTED)      # skipped (DRAW_FIGURES=0)
            else:
                steps += Text(f"✓ {p}  ", style=OK)
        elif i == cur:
            steps += Text(f"▶ {p}", style=ACCENT + " bold") + Text(f" {mins(ph_el)}  ", style=ACCENT)
        else:
            steps += Text(f"○ {p}  ", style=MUTED)
    if ph == "waiting":
        steps += Text("⏸ waiting for agent capacity", style=ACCENT + " bold")
    elif ph == "synced":
        steps += Text("✓ synced", style=OK)
    lines += [Text(""), steps]

    row = lambda name, *parts: Text(f"{name[:8]:<9}", style=MUTED) + Text.assemble(*parts)
    el = now - float(pj.get("iter_started") or now)
    # overall progress: phases done at their estimated weight + the current one
    weights = {p: est.get(p, DEFAULTS[p]) for p in PHASES}
    total = sum(weights.values())
    done_w = sum(weights[p] for p in PHASES[:max(cur, 0)])
    tp = train_progress(rd) if ph == "train" else None
    left = None
    if ph in PHASES:
        if ph == "train" and tp and tp["total"]:
            frac_ph = tp["done"] / tp["total"]
            per_epoch = ph_el / tp["done"] if tp["done"] else None
            left_ph = (tp["total"] - tp["done"]) * per_epoch if per_epoch else max(0.0, weights["train"] - ph_el)
        else:
            frac_ph = min(ph_el / weights[ph], 1.0) if weights[ph] else 0.0
            left_ph = max(0.0, weights[ph] - ph_el)
        frac = (done_w + frac_ph * weights[ph]) / total if total else 0.0
        left = left_ph + sum(weights[p] for p in PHASES[cur + 1:])
    else:
        frac = 1.0 if ph == "synced" else 0.0
    lines.append(Text(""))
    lines.append(row("done", bar(min(frac, 1.0)), (f"  {frac * 100:.0f}%  of this experiment (estimate)", "default")))
    if ph == "train":
        if tp and tp["total"]:
            lines.append(row("epochs", bar(tp["done"] / tp["total"], colour=ACCENT_SOFT),
                             (f"  {tp['done']}/{tp['total']}  {mins(ph_el)} so far", "default")))
        else:
            lines.append(row("epochs", bar(0.0, colour=ACCENT_SOFT), ("  loading the three photographs…", MUTED)))
    elif ph in PHASES:
        over = ph_el > weights[ph]
        lines.append(row(ph[:8], bar(min(ph_el / weights[ph], 1.0) if weights[ph] else 0, colour=BAD if over else ACCENT_SOFT),
                         (f"  {mins(ph_el)} of about {mins(weights[ph])} usual" + (" — over" if over else ""), BAD if over else "default")))
    lines.append(Text(""))
    lines.append(row("started", (day(float(pj.get("iter_started") or now)), "default")))
    if left is not None:
        lines.append(row("left", (f"about {mins(left)} (estimate)", "default")))
        lines.append(row("ends", (f"about {day(now + left)}", "default")))
    nproc, cpu = training_processes()
    try:
        load = os.getloadavg()[0]
    except OSError:
        load = 0.0
    if ph == "train":
        lines.append(row("cpu", (f"training on the M1 (mps), {cpu / 100:.1f} cores busy", OK) if nproc
                         else ("no training process found: check the run", BAD)))
    elif ph in ("design", "implement", "figure"):
        agent = {"design": "Fable 5.1", "implement": "Opus 5.5", "figure": "Sonnet"}[ph]
        lines.append(row("agent", (f"{agent} at work", OK)))
    lines.append(row("load", (f"{load:.1f} load average", "default")))
    if tp and tp["last"]:
        lines.append(row("log", (tp["last"][-120:], MUTED)))
    if tp and tp["caps"]:
        lines.append(row("top-1", ("  ".join(f"{k} {v * 100:.0f}%" for k, v in tp["caps"].items()), "default")))
    if design.get("eli5"):
        lines += [Text(""), Text("in plain words", style=MUTED), Text(design["eli5"], style="default")]
    elif design.get("hypothesis"):
        lines += [Text(""), Text("hypothesis", style=MUTED), Text(design["hypothesis"][:600], style="default")]
    return Panel(Group(*lines), title="current experiment", title_align="left", border_style=ACCENT)


def era_panel(est):
    pj = phase_state()
    rows, era = recent(1)
    target = pj.get("iterations")
    lines = []
    n = era.get("n", 0)
    if target:
        frac = min(n / target, 1.0) if target else 0
        lines.append(bar(frac, width=24) + Text(f"  {n}/{target} recorded in this batch", "default"))
    else:
        lines.append(Text(f"{n} experiments recorded", "default"))
    lines.append(Text(""))
    row = lambda name, *parts: Text(f"{name[:9]:<10}", style=MUTED) + Text.assemble(*parts)
    if era.get("baseline"):
        lines.append(row("baseline", (f"{era['baseline'][0]:.3f}", "default"), (f"  experiment {era['baseline'][1]}", MUTED)))
    if era.get("best"):
        lines.append(row("best", (f"{era['best'][0]:.3f}", "bold"), (f"  experiment {era['best'][1]} · {era.get('kept', 0)} kept of {n}", MUTED)))
    if era.get("median_s"):
        lines.append(row("per exp.", (mins(era["median_s"]), "default"), ("  median, start to record", MUTED)))
        remaining = max(0, (target or n) - n - 1)
        if target and n < target:
            # the one in flight ends at its own estimate; the rest at the median
            lines.append(row("batch", (f"{remaining} more after this one · about {mins(remaining * era['median_s'])} beyond it", "default")))
    name, b, diag = rates(era.get("last_metrics") or {})
    if b:
        lines.append(Text(""))
        lines.append(Text(f"last recorded (experiment {era.get('last_id')}) — score · usable · false fix", style=MUTED))
        t = Table.grid(padding=(0, 2))
        t.add_column(style=MUTED); t.add_column(justify="right"); t.add_column(justify="right"); t.add_column(justify="right")
        t.add_row(Text(f"{name}  (held out)", style=ACCENT + " bold"), Text(f"{b['mission_score']:.3f}", "bold"),
                  f"{(b.get('usable_fix_rate') or 0) * 100:.1f}%", f"{(b.get('false_fix_rate') or 0) * 100:.1f}%")
        for cap, d in diag.items():
            t.add_row(f"{cap}  (trained on)", f"{d.get('mission_score', 0):.3f}",
                      f"{(d.get('usable_fix_rate') or 0) * 100:.1f}%", f"{(d.get('false_fix_rate') or 0) * 100:.1f}%")
        lines.append(t)
    lines.append(Text(""))
    lines.append(Text(f"phase estimates from recent runs: design {mins(est['design'])} · figure {mins(est['figure'])} · "
                      f"implement {mins(est['implement'])} · train {mins(est['train'])} · score {mins(est['score'])}", style=MUTED))
    return Panel(Group(*lines), title="this era", title_align="left", border_style="grey50")


def recent_panel():
    rows, _ = recent(10)
    t = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1), header_style=MUTED)
    t.add_column("#", width=3); t.add_column("experiment", ratio=1, no_wrap=True, overflow="ellipsis")
    t.add_column("verdict", width=9); t.add_column("score", justify="right", width=6)
    t.add_column("usable", justify="right", width=6); t.add_column("time", justify="right", width=7)
    for eid, title, cat, score, kept, dur, mj, kind in rows:
        _, b, _ = rates(read_json_s(mj))
        usable = f"{(b.get('usable_fix_rate') or 0) * 100:.1f}%" if b else "–"
        verdict = "kept" if kept else "reverted"
        if score is not None and score >= 1e17:
            verdict, score_s = "gated", "×"
        else:
            score_s = f"{score:.3f}" if score is not None else "–"
        t.add_row(str(eid), title or "", Text(verdict, style=OK if kept else ("red3" if verdict == "gated" else "default")),
                  score_s, Text(usable, style=MUTED), mins(dur or 0))
    return Panel(t, title="recent experiments", title_align="left", border_style="grey50")


def activity_panel(act, rows):
    t = Table(box=None, expand=True, show_header=False, padding=(0, 1))
    t.add_column(width=5, style=MUTED); t.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    for ts, txt in sorted(act.events)[-max(3, rows):]:
        t.add_row(datetime.fromtimestamp(ts).strftime("%H:%M"), Text(txt, style="default"))
    return Panel(t, title="loop activity", title_align="left", border_style="grey50")


def footer():
    s = "take back the stop" if STOP.exists() else "stop gracefully (after this experiment)"
    return Text.assemble(("  s ", "bold reverse"), f" {s}     ", ("  q ", "bold reverse"),
                         " close the dashboard (the loop keeps running)")


def render(act, tick, width, height, est):
    import shutil
    from rich.console import Console
    lay = Layout()
    exp = experiment_panel(tick, est)
    w = width * 6 // 11
    need = len(Console(width=max(40, w)).render_lines(exp))
    top = max(height * 5 // 9, min(need, height - 3 - 1 - 8))
    lay.split_column(Layout(header(), size=3), Layout(name="top", size=top), Layout(name="bottom"), Layout(footer(), size=1))
    lay["top"].split_row(Layout(exp, ratio=6), Layout(era_panel(est), ratio=5))
    lay["bottom"].split_row(Layout(recent_panel(), ratio=6), Layout(activity_panel(act, max(3, height - top - 3 - 1 - 3)), ratio=5))
    return lay


def keys(state):
    try:
        import atexit
        import termios
        import tty
        fd = open("/dev/tty", "rb", buffering=0)
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        atexit.register(lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old))
    except (OSError, ImportError):
        return
    while True:
        ch = fd.read(1).lower()
        if ch == b"q":
            state["quit"] = True
            return
        if ch == b"s":
            if STOP.exists():
                STOP.unlink(missing_ok=True)
            else:
                STOP.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


def main(argv):
    once = "--once" in argv
    width = int(argv[argv.index("--width") + 1]) if "--width" in argv else None
    height = int(argv[argv.index("--height") + 1]) if "--height" in argv else None
    act = Activity()
    est = median_timings()
    if once:
        from rich.console import Console
        con = Console(width=width or 160, height=height or 45, force_terminal=True)
        act.poll()
        con.print(render(act, 0, con.width, con.height, est))
        return 0
    state = {"quit": False}
    threading.Thread(target=keys, args=(state,), daemon=True).start()
    tick = 0
    with Live(screen=True, auto_refresh=False) as live:
        while not state["quit"]:
            try:
                act.poll()
                if tick % 60 == 0:
                    est = median_timings()
                live.update(render(act, tick, live.console.size.width, live.console.size.height, est), refresh=True)
            except Exception as e:  # noqa: BLE001 — a half-written file must not take the dashboard down
                live.update(Panel(Text(f"reading the record: {e!r}"), title="dashboard"), refresh=True)
            tick += 1
            for _ in range(10):
                if state["quit"]:
                    break
                time.sleep(0.1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
