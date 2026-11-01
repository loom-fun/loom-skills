#!/bin/bash
# Wait until LOGFILE contains a line matching REGEX (e.g. the final line of qualify.sh / finalize.sh), then
# print the last lines. Use this instead of `until ! pgrep -f SCRIPT` loops: pgrep -f matches the loop's
# own command line (three trials lost minutes to loops that never ended).
# usage: wait_log.sh LOGFILE REGEX [TIMEOUT_MIN=120]      e.g. wait_log.sh finalize.log '^FINALIZE '
set -u
log=$1; re=$2; t=${3:-120}; end=$(( $(date +%s) + t * 60 ))
until [ -f "$log" ] && grep -qE "$re" "$log"; do
  [ "$(date +%s)" -gt "$end" ] && { echo "TIMEOUT waiting for /$re/ in $log"; tail -5 "$log" 2>/dev/null; exit 1; }
  sleep 10
done
tail -5 "$log"
