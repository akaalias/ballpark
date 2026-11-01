#!/bin/bash
# Waits for the current loop round to end; relaunches with a bigger target only
# if it ended by reaching its target (not by a stop request or a crash).
cd /Users/alexisrondeau/Workshop/low-light-geolocalization-autoresearch || exit 1
while pgrep -f "bash .*autoresearch/loop.sh" >/dev/null; do sleep 60; done
if [ ! -e state/stop ] && tail -5 state/loop.log | grep -q "Target reached"; then
  echo "[relaunch_watch $(date '+%F %T')] round ended at target; relaunching with target ${1:-100}" >> state/loop.log
  REMOTE_TRAIN=1 nohup ./autoresearch/loop.sh "${1:-100}" >> state/loop.log 2>&1 < /dev/null &
else
  echo "[relaunch_watch $(date '+%F %T')] loop ended without reaching target (stop or error); not relaunching" >> state/loop.log
fi
