#!/usr/bin/env bash
# Open the loop's dashboard (issue #2). q closes it; the loop keeps running.
cd "$(dirname "$0")/.." && exec .venv/bin/python -m autoresearch.dashboard "$@"
