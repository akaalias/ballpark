#!/usr/bin/env bash
# FROZEN (see /FROZEN) — the autoresearch loop, CLAUDE.md §7. Phase 2 entrypoint.
#
# Repeatedly: ask a headless Claude Code instance to make ONE focused change to
# the agent-editable files (model/), then train + score on the development
# areas, log to SQLite, and keep (git commit) or revert by comparing the §6
# primary metric against the best so far. Every HOLDOUT_EVERY kept
# improvements, run the read-only Hamburg holdout check (§5) — logged, never
# fed back into keep/revert.
#
# Usage:   ./autoresearch/loop.sh [target-experiment-count]
#          The argument is a TARGET, not a count of runs: the loop keeps
#          designing/training/scoring experiments until the database holds
#          that many development experiments, then stops. So on a database
#          that already has 48 experiments, `loop.sh 100` runs 52 more;
#          `loop.sh 100` again later runs nothing. Default target: 100.
# Env:     AREAS          (default "berlin" on this branch — see CLAUDE.md
#                          "BRANCH OVERRIDE"; main-branch default is
#                          "berlin prignitz munich frankfurt")
#          PATIENCE       (default 4: consecutive non-kept experiments before
#                          the design prompt demands a pivot — pivot
#                          enforcement itself is force-disabled on this
#                          branch regardless of PATIENCE, see below)
#          EPOCHS         (default 8)
#          HOLDOUT_EVERY  (default 5)
#          HOLDOUT_ENABLED (default 0 on this branch; main-branch default 1 —
#                          §5 Hamburg holdout dropped for berlin-slim)
#          DRAW_FIGURES   (default 0 on this branch; main-branch default 1 —
#                          agent-drawn architecture SVG cut, ~15-20 min/iter)
#          CLAUDE_BIN     (default "claude")
#          SKIP_AGENT=1   (run train/score/log only — no code change; smoke test)
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-100}"
AREAS="${AREAS:-berlin}"
EPOCHS="${EPOCHS:-8}"
# Training budget knobs (env-overridable, champion code untouched). Crops per
# lighting bucket × bucket count = crops/epoch; total epochs = EPOCHS ×
# EPOCH_MULT (3, in train.py). berlin-slim branch runs 1 bucket (LIGHTING_
# BUCKETS collapsed, see CLAUDE.md "BRANCH OVERRIDE"), so defaults now give
# 6000 crops/epoch, not the main branch's 36k. Lower both for fast local
# exploration, e.g. TRAIN_CROPS=1500 EPOCHS=3.
TRAIN_CROPS="${TRAIN_CROPS:-6000}"
HOLDOUT_EVERY="${HOLDOUT_EVERY:-5}"
HOLDOUT_ENABLED="${HOLDOUT_ENABLED:-0}"  # berlin-slim branch: off by default (§5 dropped)
DRAW_FIGURES="${DRAW_FIGURES:-0}"  # berlin-slim branch: off by default (cuts ~15-20 min/iteration; gallery/heatmaps stay on)
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
PY=".venv/bin/python"
STATE="state"; mkdir -p "$STATE" runs

best_metric() { [ -f "$STATE/best.json" ] && $PY -c "import json;print(json.load(open('$STATE/best.json'))['primary'])" || echo 1e18; }

# How many development experiments the database already holds — the quantity
# the TARGET is measured against (holdout checks are diagnostics, not
# experiments, so they don't count). This is what makes `loop.sh N` mean
# "reach N experiments" rather than "run N more times".
dev_experiment_count() { sqlite3 experiments.sqlite "SELECT COUNT(*) FROM experiments WHERE kind='development';" 2>/dev/null || echo 0; }

# Sync health, updated after every push attempt (see step 6) — surfaced via
# phase.json below so an external health-check can tell the pod is stuck
# diverged from origin without SSHing in. PUSH_OK=0 / rising PUSH_AHEAD
# across successive checks means main.sync commits aren't landing (usually
# a real content conflict with a laptop-side commit — needs a human).
PUSH_OK=1
PUSH_AHEAD=0

# Real-time phase reporting for the public LIVE row: a one-file commit
# force-pushed to refs/heads/status (plumbing objects only — main's history
# stays untouched). The page fetches it from raw.githubusercontent.
report_phase() {
  # phase.json key names are kept ("iter"/"iterations"/"iter_started") for
  # backward compatibility with the live page and the external health-check
  # that already read them; the values now carry the experiment number, the
  # target, and the experiment's start time.
  # Design snippet: once the design agent has written a design (still at
  # runs/pending_experiment.json pre-implementation, moved to
  # $RUN_DIR/experiment.json once implementation starts), publish its
  # human-facing fields so the live row can show title/eli5/hypothesis
  # before training even starts, not just a phase name.
  DESIGN_JSON="null"
  for _df in "runs/pending_experiment.json" "${RUN_DIR:-}/experiment.json"; do
    if [ -n "$_df" ] && [ -f "$_df" ]; then
      DESIGN_JSON="$($PY -c "
import json,sys
try:
    d = json.load(open('$_df'))
except Exception:
    print('null'); sys.exit()
snip = {k: d.get(k) for k in ('title','category','init_strategy','eli5','hypothesis') if d.get(k)}
print(json.dumps(snip) if snip else 'null')
" 2>/dev/null || echo 'null')"
      break
    fi
  done
  printf '{"iter":%s,"iterations":%s,"iter_started":%s,"phase":"%s","phase_started":%s,"best":%s,"push_ok":%s,"push_ahead":%s,"design":%s}\n' \
    "${N:-0}" "$TARGET" "${EXPERIMENT_T0:-$(date +%s)}" "$1" "$(date +%s)" "$(best_metric)" \
    "$PUSH_OK" "$PUSH_AHEAD" "$DESIGN_JSON" \
    > "$STATE/phase.json" || true
  ( BLOB=$(git hash-object -w "$STATE/phase.json") &&
    TREE=$(printf '100644 blob %s\tphase.json\n' "$BLOB" | git mktree) &&
    COMMIT=$(git commit-tree "$TREE" -m "phase: $1") &&
    git push -qf origin "$COMMIT:refs/heads/status" ) 2>/dev/null || true
}

# Refuse to let any single oversized file into a commit. Bug history: one
# experiment once leaked 5.7 GB/experiment of training scratch (fixed
# separately by purging train_$area dirs below); a different run committed
# 1.4 GB of uncropped debug renders that sat unnoticed for a day and broke
# CI disk (2026-07-22). This catches anything of that shape before `git
# add` ever sees it, instead of relying on a human to notice later.
MAX_COMMIT_MB="${MAX_COMMIT_MB:-25}"
quarantine_oversized() {
  local dir="$1" big
  [ -d "$dir" ] || return 0
  big="$(find "$dir" -type f -size "+${MAX_COMMIT_MB}M" 2>/dev/null || true)"
  [ -z "$big" ] && return 0
  echo "WARNING: file(s) over ${MAX_COMMIT_MB}MB excluded from the commit (moved to state/quarantine/$RUN_ID, not pushed):"
  echo "$big"
  echo "$big" | while IFS= read -r f; do
    dest="state/quarantine/$RUN_ID/${f#"$dir"/}"
    mkdir -p "$(dirname "$dest")"
    mv "$f" "$dest"
  done
}

# NEVER lose an experiment's implementation. Every path that discards model/
# must call this first. Reverted code used to be thrown away outright, and on
# 2026-07-31 that cost the project its best-scoring implementation: the
# scoring metric was corrected, a previously-reverted experiment became the
# champion, and its source no longer existed. Cheap insurance; call it before
# every `git checkout -- model/`.
snapshot_model_src() {
  local dest="${1:-}"
  [ -n "$dest" ] || return 0
  mkdir -p "$dest/model_src" 2>/dev/null || return 0
  cp model/*.py "$dest/model_src/" 2>/dev/null || true
}

# Draw THIS experiment's architecture figure — the agent-drawn, figcheck-
# validated SVG in the same visual language as every prior figure (see
# autoresearch/prompt_figure.md and archive/arch_svg_reference.py). Runs for
# EVERY experiment regardless of the eventual keep/revert/reject outcome:
# the figure is record-keeping, not the metric, so it must not depend on
# whether the design survives the gates or improves the score. Reads the
# design from runs/pending_experiment.json and merges the resulting
# architecture_svg back into it, so the field rides along with the record
# through whichever exit path the experiment takes (the DB writer at each
# gate reads architecture_svg from the same file). Non-fatal by design: a
# failed or invalid draw just leaves that record without a figure rather
# than aborting the experiment.
FIGURE_MODEL="${FIGURE_MODEL:-${IMPL_MODEL:-claude-sonnet-5}}"
draw_figure() {
  local rundir="$1"
  local design="runs/pending_experiment.json"
  [ "${SKIP_AGENT:-0}" = "1" ] && return 0
  # Figures are token-expensive (the agent iterates against figcheck's
  # geometry — minutes and a real slice of subscription usage each). On by
  # default so every experiment is illustrated; set DRAW_FIGURES=0 at launch
  # to skip them entirely and conserve usage.
  [ "${DRAW_FIGURES:-1}" = "1" ] || return 0
  [ -f "$design" ] || return 0
  # Only draw when the design actually carries an architecture to illustrate.
  $PY -c "import json,sys; d=json.load(open('$design')); \
    sys.exit(0 if d.get('architecture',{}).get('stages') else 1)" 2>/dev/null || return 0
  report_phase figure
  local prompt="$rundir/prompt_figure.md" out="$rundir/figure.json"
  rm -f "$out"
  cat autoresearch/prompt_figure.md > "$prompt"
  { echo
    echo "## Paths for this run (resolve the placeholders referenced above)"
    echo "- Read the experiment record from: \`$design\`"
    echo "- Write ONLY this output file: \`$out\` — a single JSON object of the"
    echo "  form {\"architecture_svg\": \"<svg …>\"} and nothing else."
  } >> "$prompt"
  # --max-turns bounds cost: the agent stops the moment figcheck passes, so a
  # cap only bites when a layout won't converge — better a capped, slightly
  # imperfect figure than an open-ended grind (an uncapped run once churned
  # 8+ min on one figure). Merge whatever valid SVG it wrote even on a
  # non-zero exit, rather than discarding finished work.
  "$CLAUDE_BIN" -p "$(cat "$prompt")" \
    --model "$FIGURE_MODEL" \
    --permission-mode acceptEdits \
    --max-turns "${FIGURE_MAX_TURNS:-45}" \
    --allowedTools "Read,Write,Grep,Glob,Bash(.venv/bin/python:*)" \
    --output-format json </dev/null >"$rundir/agent_figure.json" \
    || echo "figure agent exited non-zero (turn limit?) — using whatever figure it wrote"
  # Merge the agent's architecture_svg into the design record (loop.sh does
  # the merge; the agent only ever writes its own output file, never the
  # record). Skips silently if no valid <svg> was produced.
  $PY - "$design" "$out" <<'PYFIG' || echo "figure merge skipped (no valid svg produced)"
import json, sys
design_path, fig_path = sys.argv[1], sys.argv[2]
fig = json.load(open(fig_path))
svg = (fig.get("architecture_svg") or "").strip()
if not svg.startswith("<svg"):
    raise SystemExit(1)
d = json.load(open(design_path))
d["architecture_svg"] = svg
json.dump(d, open(design_path, "w"), indent=2)
PYFIG
  $PY -m autoresearch.figcheck "$out" \
    || echo "WARNING: figure violates the anchor contract (cosmetic — kept anyway)"
}

# A dirty tree would blur lineage (a kept commit must contain exactly one
# experiment's change) — but the loop itself leaves state/ modified, so
# instead of aborting, checkpoint pending TRACKED changes into their own
# commit and start every experiment from a clean, attributable tree.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree dirty — committing pending changes as a pre-loop checkpoint."
  git add -u
  git commit -q -m "Pre-loop checkpoint: pending working-tree changes" || {
    echo "FATAL: checkpoint commit failed; commit or stash manually." >&2; exit 1; }
fi

# Graceful exit: Ctrl-C / SIGTERM discards only the in-flight experiment.
# Everything durable is already on disk — kept improvements are committed,
# finished experiments are in experiments.sqlite, best is in state/best.json —
# so a restart simply continues from the current best.
cleanup_interrupt() {
  echo ""
  echo "Interrupted — discarding the in-flight experiment (all completed"
  echo "experiments are already committed/logged). Safe to restart with:"
  echo "  ./autoresearch/loop.sh"
  snapshot_model_src "${RUN_DIR:-}"
  git checkout -- model/ 2>/dev/null || true
  rm -f runs/pending_experiment.json
  exit 130
}
trap cleanup_interrupt INT TERM

# Graceful stop request: type "exit" (or quit/stop) + Enter while the loop is
# running to finish the CURRENT experiment completely (train, score, log,
# keep/revert) and then stop. Also triggered by: touch state/stop
# (useful from another terminal or when running under nohup).
STOP_FLAG="$STATE/stop"
rm -f "$STOP_FLAG"
WATCHER_PID=""
if [ -t 0 ]; then
  ( while IFS= read -r line; do
      case "$line" in
        exit|quit|stop|q)
          touch "$STOP_FLAG"
          echo ">>> stop requested — will exit gracefully after the current experiment" ;;
      esac
    done ) &
  WATCHER_PID=$!
  echo "Type 'exit' + Enter to stop gracefully after the current experiment"
  echo "(Ctrl-C aborts the in-flight experiment immediately)."
fi
cleanup_watcher() { [ -n "$WATCHER_PID" ] && kill "$WATCHER_PID" 2>/dev/null || true; }
trap cleanup_watcher EXIT

# Run experiments until the database holds TARGET of them (see Usage). The
# ATTEMPTS cap is a safety net only: an experiment that fails before it logs a
# row (agent never produced a design, agent call errored) doesn't advance the
# count, so without a cap a persistent failure would spin forever.
ATTEMPTS=0
MAX_ATTEMPTS=$(( TARGET * 3 + 30 ))
while :; do
  if [ -f "$STOP_FLAG" ]; then
    rm -f "$STOP_FLAG"
    echo "Graceful stop requested — $(dev_experiment_count)/$TARGET experiments recorded so far."
    break
  fi
  DONE="$(dev_experiment_count)"
  if [ "$DONE" -ge "$TARGET" ]; then
    echo "Target reached: $DONE/$TARGET experiments recorded — nothing more to run."
    break
  fi
  ATTEMPTS=$(( ATTEMPTS + 1 ))
  if [ "$ATTEMPTS" -gt "$MAX_ATTEMPTS" ]; then
    echo "Safety stop: $ATTEMPTS attempts but only $DONE/$TARGET experiments recorded — experiments are failing to complete; investigate before continuing."
    break
  fi
  # This experiment's number IS the database id it will be assigned, so the
  # commit message, run directory, and gallery entry all name the same
  # "Experiment N".
  N="$(sqlite3 experiments.sqlite "SELECT COALESCE(MAX(id),0)+1 FROM experiments;" 2>/dev/null || echo 1)"
  # Drop any leftover design from the previous experiment before this one
  # reports a phase: report_phase publishes whatever design file it finds, so a
  # stale pending_experiment.json made the LIVE row show experiment N's
  # hypothesis and eli5 under experiment N+1's number until the new design
  # agent finished. Wrong attribution on a public page, not just a cosmetic lag.
  rm -f runs/pending_experiment.json
  RUN_ID="$(date -u +%Y%m%d_%H%M%S)_experiment$N"
  RUN_DIR="runs/$RUN_ID"
  mkdir -p "$RUN_DIR"
  PARENT_COMMIT="$(git rev-parse HEAD)"
  EXPERIMENT_T0="$(date +%s)"
  echo "=== Experiment $N ($DONE/$TARGET recorded)  run=$RUN_ID  parent=$PARENT_COMMIT ==="

  # 1. Two-stage agent (models chosen per task — design is the creative/
  #    strategic work, implementation is focused code editing):
  #      stage 1 DESIGN_MODEL (default Fable 5.1, claude-fable-5-1): pre-registers the experiment +
  #        an implementation_brief in runs/pending_experiment.json; no code edits.
  #      stage 2 IMPL_MODEL (default Opus 5.5, claude-opus-5-5): applies the brief to model/.
  # Snapshot the exact prompts handed to the agents — part of the experiment
  # record (§7 lineage) even when SKIP_AGENT skips the calls.
  cp autoresearch/prompt.md "$RUN_DIR/prompt.md"
  cp autoresearch/prompt_impl.md "$RUN_DIR/prompt_impl.md"
  # One-shot forced research direction: if state/directive.md exists, prepend it
  # to THIS experiment's design prompt (most prominent position), then consume it
  # so it steers exactly the next experiment and nothing after. Lets a human aim
  # the loop at a specific idea without editing the frozen prompt.
  if [ -f state/directive.md ]; then
    cat state/directive.md "$RUN_DIR/prompt.md" > "$RUN_DIR/.prompt_dir" \
      && mv "$RUN_DIR/.prompt_dir" "$RUN_DIR/prompt.md"
    mv state/directive.md "$RUN_DIR/directive_used.md"
    echo "FORCED DIRECTIVE injected into this experiment's design prompt (one-shot, consumed)."
  fi
  # Patience / auto-pivot (workshop pattern): after PATIENCE consecutive
  # non-kept experiments, the design prompt gets a mandatory-pivot preamble —
  # the guard against refining a dead line forever. The preamble is written
  # into the run's prompt snapshot so the record shows the prompt actually
  # used.
  LAST_KEPT_ID=$(sqlite3 experiments.sqlite "SELECT COALESCE(MAX(id),0) \
    FROM experiments WHERE kept=1 AND kind='development';" 2>/dev/null || echo 0)
  PLATEAU=$(sqlite3 experiments.sqlite "SELECT COUNT(*) FROM experiments \
    WHERE kind='development' AND id > $LAST_KEPT_ID;" 2>/dev/null || echo 0)
  PIVOT_DEMANDED=0
  FROZEN_STAGES=""  # only ever set inside the (disabled) pivot block below; must default to "" under set -u
  # berlin-slim branch override (see CLAUDE.md "BRANCH OVERRIDE"): pivot
  # enforcement forced off — reintroduce by deleting this line once a
  # plateau shows up worth forcing a rethink over.
  if false && [ "${PLATEAU:-0}" -ge "${PATIENCE:-4}" ]; then
    PIVOT_DEMANDED=1
    echo "PIVOT: $PLATEAU consecutive experiments without a new best — pivot directive injected"
    # Which stages have actually gone unquestioned across the streak (by
    # arch_json's own changed:true flags), not just which category labels
    # haven't been tried — see plateaucheck.py's docstring for why the
    # category-label version of this check let #30-34 all satisfy "pick a
    # design family from the list" while leaving the trunk/descriptor
    # untouched every single round.
    FROZEN_STAGES="$($PY -m autoresearch.plateaucheck "$LAST_KEPT_ID" "${PATIENCE:-4}" 2>/dev/null || true)"
    CHAMPION_BACKBONE="$($PY -m autoresearch.backbonecheck --fingerprint \
      model/model.py model/train.py 2>/dev/null || echo unknown)"
    cat - "$RUN_DIR/prompt.md" > "$RUN_DIR/prompt.md.tmp" <<PIVOTNOTE
## PATIENCE SPENT — THIS ITERATION MUST PIVOT

$PLATEAU consecutive experiments have failed to beat the current best.
A pivot means COMPLETELY RETHINKING THE ARCHITECTURE, not touching the one
stage named below as frozen while carrying the rest of the current design
over unchanged. Do not refine, extend, or swap one part of the champion's
mechanism and call it a pivot. Propose a genuinely different overall
design — the spec (§3) explicitly invites, among others: dispatcher +
lighting-condition-specialist models, learned relighting, a different
coordinate parameterization, quantization-aware capacity changes, or
training-data strategy overhauls — but whichever family you pick, it must
change how the WHOLE pipeline works end to end, not one isolated stage.

$FROZEN_STAGES

### The champion's backbone is OFF LIMITS this round

The current champion's backbone identity is: $CHAMPION_BACKBONE

Your design MUST NOT use it. Not re-tuned, not truncated at a different
layer, not wrapped in a dispatcher, not kept as one branch of an ensemble,
not reloaded from the same weights blob under a new class name. After
implementation the resulting source is scanned, and if any of those
identifiers still appear in model/ the experiment is REJECTED before
training. Rethinking the machinery bolted around an unexamined trunk is
exactly what the last five demanded pivots did; it is not a pivot.

A from-scratch trunk is an acceptable answer. So is a genuinely different
pretrained family. Carrying the same trunk across is not.

Every non-frozen stage in your architecture.stages list must be marked
"changed": true this round. If any stage is left exactly as before, this
experiment will be rejected before training even starts — this is checked
against your actual code diff, not just your own self-report.

PIVOTNOTE
    mv "$RUN_DIR/prompt.md.tmp" "$RUN_DIR/prompt.md"
  fi
  report_phase design
  AGENT_MODEL_DESIGN=""; AGENT_MODEL_IMPL=""
  T_DESIGN=0; T_IMPL=0
  if [ "${SKIP_AGENT:-0}" != "1" ]; then
    rm -f runs/pending_experiment.json
    T0=$(date +%s)
    "$CLAUDE_BIN" -p "$(cat "$RUN_DIR/prompt.md")" \
      --model "${DESIGN_MODEL:-claude-fable-5-1}" \
      --permission-mode acceptEdits \
      --allowedTools "Read,Write,Grep,Glob,Bash(.venv/bin/python:*),Bash(sqlite3:*)" \
      --output-format json </dev/null >"$RUN_DIR/agent_design.json" \
      || { echo "design agent failed (rate limit/usage cap?) — sleeping \
${AGENT_RETRY_SLEEP:-1800}s before next experiment"
           report_phase waiting
           sleep "${AGENT_RETRY_SLEEP:-1800}"; continue; }
    T_DESIGN=$(( $(date +%s) - T0 ))
    if [ ! -f runs/pending_experiment.json ]; then
      echo "design agent produced no runs/pending_experiment.json; skipping experiment"
      continue
    fi

    # Draw the architecture figure now, from the design alone — BEFORE any
    # gate below can reject the experiment — so every experiment ends up with
    # one regardless of outcome (kept, reverted, or rejected pre-training).
    # The SVG is merged into runs/pending_experiment.json and rides along to
    # whichever exit path this experiment takes.
    draw_figure "$RUN_DIR"

    # Early pivot-completeness check: if a pivot was demanded, verify the
    # DESIGN's own self-report already shows every non-frozen stage changed
    # BEFORE ever spending an implementation-agent call on it. Only the
    # backbone-diff verification (below, post-implementation) genuinely
    # needs a real code diff to exist — this half of the gate doesn't, so
    # checking it here saves ~2-3 min plus a whole extra Claude invocation
    # whenever the design alone already fails to be a complete rethink.
    AGENT_MODEL_DESIGN="$($PY -m autoresearch.agentmeta "$RUN_DIR/agent_design.json" 2>/dev/null || echo "${DESIGN_MODEL:-claude-fable-5-1}")"
    if [ -n "$FROZEN_STAGES" ]; then
      UNCHANGED_STAGES="$($PY - runs/pending_experiment.json <<'PYCHECK'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("(pending_experiment.json unreadable)")
    raise SystemExit
frozen_harness = {"Camera frame", "Output"}
stages = d.get("architecture", {}).get("stages", [])
unchanged = [s.get("name") for s in stages
             if s.get("name") not in frozen_harness and not s.get("changed")]
print(", ".join(unchanged))
PYCHECK
)"
      if [ -n "$UNCHANGED_STAGES" ]; then
        echo "REJECTED (before implementation): a complete architecture rethink was required, but the design's own self-report leaves these stages unchanged: $UNCHANGED_STAGES — skipping implementation and training."
        mv runs/pending_experiment.json "$RUN_DIR/experiment.json"
        echo '{"kind":"development","areas":[],"primary_mission_score":1e9,"target_m":100.0}' > "$RUN_DIR/metrics.json"
        $PY -m autoresearch.db --metrics "$RUN_DIR/metrics.json" \
          --experiment-file "$RUN_DIR/experiment.json" \
          --result "rejected before implementation: a complete architecture rethink was required, but these stages were already self-reported unchanged in the design: $UNCHANGED_STAGES" \
          --conclusion "REJECTED — pivot directive was not honored (partial change, not a complete rethink); experiment discarded before implementation" \
          --parent-commit "$PARENT_COMMIT" --artifacts-dir "$RUN_DIR" --kept 0 \
          --prompt-file "$RUN_DIR/prompt.md" \
          --agent-model-design "$AGENT_MODEL_DESIGN" --agent-model-impl "" \
          --duration-s "$(( $(date +%s) - EXPERIMENT_T0 ))" || true
        continue
      fi
    fi

    # Publish the design NOW, not at the end of the experiment. The gallery
    # was only rendered after scoring, so a design that landed at minute 5 of
    # a 20-minute experiment stayed invisible until the whole thing finished —
    # the LIVE row sat on "Design not finished yet" while the design existed.
    # (The row also tries to self-refresh from the status branch, but that
    # fetch hits raw.githubusercontent, which 403s for a PRIVATE repo, so the
    # render-time pre-fill is what actually makes it visible.) ~1-2s.
    $PY -m autoresearch.gallery >/dev/null 2>&1 || true
    report_phase implement
    T0=$(date +%s)
    "$CLAUDE_BIN" -p "$(cat "$RUN_DIR/prompt_impl.md")" \
      --model "${IMPL_MODEL:-claude-opus-5-5}" \
      --permission-mode acceptEdits \
      --allowedTools "Read,Edit,Write,Grep,Glob,Bash(.venv/bin/python:*),Bash(sqlite3:*)" \
      --output-format json </dev/null >"$RUN_DIR/agent_impl.json" \
      || { echo "impl agent failed; skipping experiment"
           snapshot_model_src "$RUN_DIR"
           git checkout -- model/ 2>/dev/null || true; continue; }
    T_IMPL=$(( $(date +%s) - T0 ))
    AGENT_MODEL_IMPL="$($PY -m autoresearch.agentmeta "$RUN_DIR/agent_impl.json" 2>/dev/null || echo "${IMPL_MODEL:-claude-opus-5-5}")"
    echo "agents finished (design: $AGENT_MODEL_DESIGN ${T_DESIGN}s, impl: $AGENT_MODEL_IMPL ${T_IMPL}s)"
  fi
  [ -f runs/pending_experiment.json ] && mv runs/pending_experiment.json "$RUN_DIR/experiment.json"
  # A SKIP_AGENT run is a real thing with a real purpose -- the baseline seed
  # that establishes the number every later experiment must beat -- so it gets
  # a proper self-describing record instead of "(no experiment design
  # provided)", which reads like a broken row in the gallery. This used to be
  # patched into the DB by hand after every lineage reset; now it is automatic.
  if [ ! -f "$RUN_DIR/experiment.json" ]; then
    if [ "${SKIP_AGENT:-0}" = "1" ]; then
      $PY - "$RUN_DIR/experiment.json" <<'PYSEED'
import json, sys
json.dump({
  "title": "Baseline seed: current model/ code, no design agent",
  "category": "other",
  "init_strategy": "from-scratch",
  "hypothesis": "Not a hypothesis -- a starting line. Run with SKIP_AGENT=1 so no "
                "design or implementation agent is involved: whatever is in model/ is "
                "trained and scored as-is to establish the number later experiments "
                "must beat, and to prove the harness works end to end.",
  "method": "Train the current model/ code unchanged and score it with the frozen "
            "scorer. No code change is proposed or applied.",
  "expected_outcome": "An honest starting mission score on the current metric, so "
                      "'kept' from here on means genuinely better.",
  "eli5": "This row is the starting line, not an idea being tested. The model that "
          "happens to be in the repo right now is trained and measured so there is a "
          "real number on the board; everything after this is an attempt to beat it.",
}, open(sys.argv[1], "w"))
PYSEED
    else
      echo '{"title":"(no experiment design provided)"}' > "$RUN_DIR/experiment.json"
    fi
  fi

  # 2. Enforce frozen files — hard-revert any agent edits to them.
  FROZEN_TOUCHED="$(git diff --name-only | grep -xF -f FROZEN || true)"
  if [ -n "$FROZEN_TOUCHED" ]; then
    echo "WARNING: agent touched frozen files, reverting:"; echo "$FROZEN_TOUCHED"
    echo "$FROZEN_TOUCHED" | xargs git checkout --
  fi
  # Enforce holdout blindness: agent must never read/copy holdout data or score it.
  # (score.py refuses hamburg without --holdout; loop only passes dev areas.)

  # 2b. Verify a demanded pivot actually REPLACED the champion's backbone,
  # checked against the post-implementation SOURCE — not the diff, and not
  # the agent's self-report.
  #
  # The previous version of this gate asked whether the diff contained a
  # removed line matching mobilenet_v3_small. It had two independent holes:
  # it only ran when plateaucheck happened to name the "Feature extractor"
  # stage (so a short streak, or a renamed stage, skipped it entirely), and
  # one removed line satisfied it (features[:9] -> features[:10], or a
  # reworded docstring, deletes a matching line and adds another just like
  # it). Demanded pivots shipped pretrained:mobilenet_v3_small five times
  # through #40 because nothing ever inspected the code that resulted.
  # backbonecheck.py compares champion identifiers against the candidate
  # source, so re-tuning, truncating, wrapping in a dispatcher, or keeping
  # the trunk as one ensemble branch all fail the same way.
  CARRIED_BACKBONE=""
  if [ "${PIVOT_DEMANDED:-0}" = "1" ]; then
    git show HEAD:model/model.py > "$RUN_DIR/.champ_model.py" 2>/dev/null || true
    git show HEAD:model/train.py > "$RUN_DIR/.champ_train.py" 2>/dev/null || true
    CARRIED_BACKBONE="$($PY -m autoresearch.backbonecheck \
      --champion "$RUN_DIR/.champ_model.py" "$RUN_DIR/.champ_train.py" \
      --candidate model/model.py model/train.py 2>/dev/null \
      | tr "\n" " " | sed "s/ *$//")"
    rm -f "$RUN_DIR/.champ_model.py" "$RUN_DIR/.champ_train.py"
  fi
  if [ -n "$CARRIED_BACKBONE" ]; then
    REJECT_REASON="a pivot was demanded but the implementation still carries the champion's backbone ($CARRIED_BACKBONE) — checked against the post-implementation source; a pivot must replace the trunk outright, not re-tune, truncate, wrap, or ensemble it"
    echo "REJECTED: $REJECT_REASON — skipping training."
    echo '{"kind":"development","areas":[],"primary_mission_score":1e9,"target_m":100.0}' > "$RUN_DIR/metrics.json"
    snapshot_model_src "$RUN_DIR"
    git checkout -- model/ 2>/dev/null || true
    $PY -m autoresearch.db --metrics "$RUN_DIR/metrics.json" \
      --experiment-file "$RUN_DIR/experiment.json" \
      --result "rejected before training: $REJECT_REASON" \
      --conclusion "REJECTED — pivot directive was not honored (self-report contradicted by the real diff); experiment discarded without training" \
      --parent-commit "$PARENT_COMMIT" --artifacts-dir "$RUN_DIR" --kept 0 \
      --prompt-file "$RUN_DIR/prompt.md" \
      --agent-model-design "$AGENT_MODEL_DESIGN" --agent-model-impl "$AGENT_MODEL_IMPL" \
      --duration-s "$(( $(date +%s) - EXPERIMENT_T0 ))" || true
    continue
  fi

  # 3. Train one model per development area — IN PARALLEL (areas are fully
  #    independent; per-area out-dirs avoid a write race on train_info.json,
  #    merged below) — then score (§6).
  report_phase train
  T0=$(date +%s); FAILED=0; PIDS=""
  for area in $AREAS; do
    # Cap math-library threads per process: torch defaults each process to a
    # thread pool sized to ALL host cores, so 4 concurrent trainings spawn
    # ~4x96 threads on a small cgroup quota and thrash (observed: GPU 10%,
    # 45+ min wall). TRAIN_THREADS overrides the default cap.
    if [ "${REMOTE_TRAIN:-0}" = "1" ]; then
      # Train on a remote GPU worker (stateless — no git on it, so the laptop
      # stays the sole writer). Same out-dir structure (models/*.onnx +
      # train_info.json) as local training, so the merge/score steps below are
      # unchanged. REMOTE_BACKEND: modal (serverless, default) | runpod (pod).
      if [ "${REMOTE_BACKEND:-modal}" = "runpod" ]; then
        infra/remote_train.sh "$area" "$RUN_DIR/train_$area" "$EPOCHS" "$TRAIN_CROPS" \
          >"$RUN_DIR/train_$area.log" 2>&1 &
      else
        $PY infra/modal_client.py "$area" "$RUN_DIR/train_$area" "$EPOCHS" "$TRAIN_CROPS" \
          >"$RUN_DIR/train_$area.log" 2>&1 &
      fi
    else
      OMP_NUM_THREADS="${TRAIN_THREADS:-8}" MKL_NUM_THREADS="${TRAIN_THREADS:-8}" \
      $PY -m model.train --area "$area" --out-dir "$RUN_DIR/train_$area" \
        --epochs "$EPOCHS" --max-crops-per-bucket "$TRAIN_CROPS" \
        >"$RUN_DIR/train_$area.log" 2>&1 &
    fi
    PIDS="$PIDS $!"
  done
  for pid in $PIDS; do wait "$pid" || FAILED=1; done
  for area in $AREAS; do echo "--- train $area:"; tail -2 "$RUN_DIR/train_$area.log"; done
  T_TRAIN=$(( $(date +%s) - T0 ))
  if [ "$FAILED" = "0" ]; then
    mkdir -p "$RUN_DIR/models"
    for area in $AREAS; do
      cp "$RUN_DIR/train_$area/models/"*.onnx "$RUN_DIR/models/" 2>/dev/null || FAILED=1
    done
    $PY - "$RUN_DIR" $AREAS <<'PYMERGE' || FAILED=1
import json, sys
from pathlib import Path
run = Path(sys.argv[1]); merged = []
for a in sys.argv[2:]:
    p = run / f"train_{a}" / "train_info.json"
    if p.exists():
        merged += json.loads(p.read_text())
(run / "train_info.json").write_text(json.dumps(merged, indent=2))
PYMERGE
    # Per-area training dirs are SCRATCH (agent code may cache gigabytes of
    # renders there — exp 17 wrote ~5.7 GB/experiment). Models and
    # train_info are merged out above; everything else must never reach
    # the record or LFS.
    for area in $AREAS; do rm -rf "$RUN_DIR/train_$area"; done
  fi
  report_phase score
  T0=$(date +%s)
  if [ "$FAILED" = "1" ]; then
    echo '{"kind":"development","areas":[],"primary_mission_score":1e9,"target_m":100.0}' > "$RUN_DIR/metrics.json"
  else
    $PY -m pipeline.score --areas "$(echo $AREAS | tr ' ' ',')" \
      --model-dir "$RUN_DIR/models" --out "$RUN_DIR/metrics.json" \
      --heatmap-dir "$RUN_DIR/heatmaps" || \
      echo '{"kind":"development","areas":[],"primary_mission_score":1e9,"target_m":100.0}' > "$RUN_DIR/metrics.json"
  fi
  T_SCORE=$(( $(date +%s) - T0 ))
  T0=$(date +%s)
  $PY -m autoresearch.samples --areas "$(echo $AREAS | tr ' ' ',')" --out "$RUN_DIR/samples" || true
  T_SAMPLES=$(( $(date +%s) - T0 ))

  METRIC="$($PY -c "import json;print(json.load(open('$RUN_DIR/metrics.json'))['primary_mission_score'])")"
  BEST="$(best_metric)"
  KEEP="$($PY -c "print(1 if $METRIC < $BEST else 0)")"

  # 4. Keep or revert (Karpathy-style), then log the completed experiment
  #    record (pre-registered design + measured result + conclusion).
  #
  # ALWAYS snapshot the source that produced this result, BEFORE the revert
  # below can destroy it. Until 2026-07-31 a reverted experiment's code was
  # simply thrown away (`git checkout -- model/`), leaving only prose and an
  # ONNX behind — so when the scoring metric was later corrected and a
  # previously-reverted experiment turned out to be the best result the
  # project had, its implementation no longer existed and had to be rebuilt
  # from its brief. A few KB of .py per experiment is nothing next to losing
  # the only copy of a champion.
  snapshot_model_src "$RUN_DIR"
  RESULT="mission score = $METRIC (previous best $BEST) — (1 - usable_fix_rate) + false_fix_rate, worst bucket; full breakdown in metrics.json"
  if [ "$KEEP" = "1" ]; then
    quarantine_oversized model
    git add -A model/ && git commit -q -m "Experiment $N: mission score $METRIC (previous best $BEST)

$(cat "$RUN_DIR/experiment.json")" || true
    echo "{\"primary\": $METRIC, \"run\": \"$RUN_ID\", \"commit\": \"$(git rev-parse HEAD)\"}" > "$STATE/best.json"
    KEPT_COUNT=$(( $(cat "$STATE/kept_count" 2>/dev/null || echo 0) + 1 )); echo "$KEPT_COUNT" > "$STATE/kept_count"
    CONCLUSION="KEPT — mission score improved ($BEST -> $METRIC); change committed as $(git rev-parse --short HEAD)"
    echo "KEPT: mission score $METRIC (improved from $BEST)"
  else
    git checkout -- model/ 2>/dev/null || true
    CONCLUSION="REVERTED — mission score did not improve ($METRIC vs best $BEST); change discarded"
    echo "REVERTED: mission score $METRIC (best remains $BEST)"
  fi
  DURATION_S=$(( $(date +%s) - EXPERIMENT_T0 ))
  $PY -m autoresearch.db --metrics "$RUN_DIR/metrics.json" \
    --experiment-file "$RUN_DIR/experiment.json" \
    --result "$RESULT" --conclusion "$CONCLUSION" \
    --parent-commit "$PARENT_COMMIT" --artifacts-dir "$RUN_DIR" --kept "$KEEP" \
    --prompt-file "$RUN_DIR/prompt.md" \
    --agent-model-design "$AGENT_MODEL_DESIGN" \
    --agent-model-impl "$AGENT_MODEL_IMPL" \
    --duration-s "$DURATION_S"

  # 5. Periodic read-only holdout check (§5) — logged, never drives keep/revert.
  T_HOLDOUT=0; T0=$(date +%s)
  KEPT_COUNT="$(cat "$STATE/kept_count" 2>/dev/null || echo 0)"
  if [ "${HOLDOUT_ENABLED:-0}" = "1" ] && [ "$KEEP" = "1" ] && [ $(( KEPT_COUNT % HOLDOUT_EVERY )) -eq 0 ]; then
    echo "--- holdout check (hamburg) ---"
    HODIR="$RUN_DIR/holdout"
    $PY -m model.train --area hamburg --out-dir "$HODIR" --epochs "$EPOCHS" --max-crops-per-bucket "$TRAIN_CROPS" && \
    $PY -m pipeline.score --areas hamburg --holdout \
      --model-dir "$HODIR/models" --out "$HODIR/holdout.json" \
      --heatmap-dir "$HODIR/heatmaps" && \
    echo '{"title":"Periodic blind holdout check (hamburg)","category":"other","hypothesis":"If the pipeline is genuinely generic, holdout error should track the development worst-case.","method":"Train current best model code on hamburg, score read-only (§5)."}' > "$HODIR/experiment.json" && \
    $PY -m autoresearch.db --metrics "$HODIR/holdout.json" \
      --experiment-file "$HODIR/experiment.json" \
      --result "informational only — never drives keep/revert" \
      --parent-commit "$PARENT_COMMIT" --artifacts-dir "$HODIR" \
      --kind holdout_check || echo "holdout check failed (non-fatal)"
  fi

  T_HOLDOUT=$(( $(date +%s) - T0 ))
  report_phase publish
  T0=$(date +%s)
  $PY -m autoresearch.gallery || true
  T_GALLERY=$(( $(date +%s) - T0 ))

  # Per-phase timing breakdown — committed with the record so the gallery's
  # experiment-detail view can render where the experiment's time went.
  $PY - <<PYTIMES || true
import json
json.dump({
    "agent_design_s": $T_DESIGN, "agent_impl_s": $T_IMPL,
    "train_wall_s": $T_TRAIN, "score_s": $T_SCORE,
    "samples_s": $T_SAMPLES, "holdout_s": $T_HOLDOUT,
    "gallery_s": $T_GALLERY,
    "total_s": $(( $(date +%s) - EXPERIMENT_T0 )),
}, open("$RUN_DIR/timings.json", "w"), indent=2)
PYTIMES

  # 6. Persist the complete experiment record — artifacts (incl. any holdout
  #    check), lineage DB, state — for kept AND reverted experiments, and
  #    push off-site. A reverted experiment's record lives only here and in
  #    SQLite, so this commit is what makes the full research trail survive
  #    the pod. Push failures are non-fatal: everything is committed locally
  #    and pushes retry implicitly next experiment — but see PUSH_OK/
  #    PUSH_AHEAD above: a real content conflict with a laptop commit won't
  #    resolve itself on retry, so it's surfaced via phase.json instead of
  #    just logged into a terminal nobody's watching.
  quarantine_oversized "$RUN_DIR"
  git add -A "$RUN_DIR" experiments.sqlite state/ 2>/dev/null || true
  git commit -q -m "Experiment $N record: $RUN_ID (mission score $METRIC, kept=$KEEP)" || true
  # Sync the branch we are ACTUALLY on, to its own remote branch.
  #
  # This used to hardcode `git pull --rebase origin main && git push origin
  # main`. On any branch other than main that is wrong in both halves: the
  # pull rebases the CURRENT branch onto origin/main (silently rewriting a
  # feature branch's history if it ever succeeds), and the push pushes the
  # local `main` ref, which the loop never touches — so the branch's own work
  # is never backed up. It only looked harmless because it failed every time
  # and logged a misleading "sync with origin/main failed".
  #
  # The laptop is the single writer here, so there is nothing to rebase onto:
  # just fast-forward-push this branch. A rejected push means someone else
  # moved the remote branch, which needs a human, not an automatic rebase.
  SYNC_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
  if git push -q origin "HEAD:refs/heads/$SYNC_BRANCH" 2>/dev/null; then
    PUSH_OK=1; PUSH_AHEAD=0
  else
    PUSH_OK=0
    PUSH_AHEAD="$(git rev-list --count "origin/$SYNC_BRANCH..HEAD" 2>/dev/null \
                  || git rev-list --count HEAD --not --remotes 2>/dev/null || echo -1)"
    echo "WARNING: push to origin/$SYNC_BRANCH failed (offline, no remote branch, or rejected) — $PUSH_AHEAD commit(s) ahead, record is committed locally only"
  fi
  report_phase "synced"
done
report_phase idle
echo "Loop finished. Best mission score: $(best_metric) — see gallery/index.html"
