#!/bin/bash
# Watchdog for one trial: exits when the trial finishes OR stalls, so the orchestrator is woken up
# either way. Run it in the background right after launching the trial agent, one per trial.
# usage: watch_trial.sh TRIAL_DIR [STALL_MIN=25] [MAX_MIN=240]
#   exit 0  DONE     phases.log has a `done` line
#   exit 2  STALLED  no file under TRIAL_DIR changed for STALL_MIN minutes (an agent waiting on a
#                    background loop that will never fire looks exactly like this: a trial once sat
#                    idle for 16 hours because the orchestrator only waited for a completion message)
#   exit 3  TIMEOUT  MAX_MIN minutes since the watchdog started
# STALL_MIN must exceed the longest silent step: a full-context needle prefill writes nothing for
# minutes (a 261k prompt took ~11 min on a busy GPU), so keep it >= 20.
set -u
dir=$1; stall=${2:-25}; max=${3:-240}; t0=$(date +%s)
log=$dir/phases.log
while :; do
  if [ -f "$log" ] && grep -q " done$" "$log"; then echo "DONE $(date +%T) $dir"; exit 0; fi
  now=$(date +%s)
  last=$(find "$dir" -type f -mmin "-${stall}" -print -quit 2>/dev/null)
  if [ -z "$last" ] && [ $(( now - t0 )) -gt $(( stall * 60 )) ]; then
    echo "STALLED $(date +%T): nothing under $dir changed for ${stall} min. Last phase: $(grep -v '^#' "$log" 2>/dev/null | tail -1)"
    echo "GPU busy: $(cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | head -1)%"
    exit 2
  fi
  [ $(( now - t0 )) -gt $(( max * 60 )) ] && { echo "TIMEOUT after ${max} min: $(grep -v '^#' "$log" | tail -1)"; exit 3; }
  sleep 60
done
