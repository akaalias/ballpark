#!/bin/bash
# Waits for the current loop to exit, then relaunches it with target $1.
# Relaunches after a normal finish, OR after a stop that was requested only
# to restart (state/restart_requested); a plain user stop is respected.
cd /Users/alexisrondeau/Workshop/low-light-geolocalization-autoresearch || exit 1
while pgrep -f "bash .*autoresearch/loop.sh" >/dev/null; do sleep 30; done
if [ -e state/restart_requested ]; then
  rm -f state/stop state/restart_requested
  echo "[relaunch_watch $(date '+%F %T')] restart requested; relaunching with target ${1:-100}" >> state/loop.log
  REMOTE_TRAIN=1 nohup ./autoresearch/loop.sh "${1:-100}" >> state/loop.log 2>&1 < /dev/null &
elif [ ! -e state/stop ] && tail -5 state/loop.log | grep -q "Target reached"; then
  echo "[relaunch_watch $(date '+%F %T')] round ended at target; relaunching with target ${1:-100}" >> state/loop.log
  REMOTE_TRAIN=1 nohup ./autoresearch/loop.sh "${1:-100}" >> state/loop.log 2>&1 < /dev/null &
else
  echo "[relaunch_watch $(date '+%F %T')] loop ended without target or restart request; not relaunching" >> state/loop.log
fi
