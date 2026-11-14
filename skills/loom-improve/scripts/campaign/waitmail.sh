#!/bin/bash
# Usage: waitmail.sh NAME [CAMPAIGN_DIR]
# Blocks until CAMPAIGN_DIR/inbox/NAME.md grows, then prints the new lines and exits. CAMPAIGN_DIR defaults to $CAMPAIGN_DIR
# or the current directory. Run it as ONE background shell command (run_in_background) and read the output when it finishes;
# never poll in a foreground loop and never message a peer agent instead.
set -u
name=${1:?usage: waitmail.sh NAME [CAMPAIGN_DIR]}
dir=${2:-${CAMPAIGN_DIR:-.}}
f="$dir/inbox/$name.md"
mkdir -p "$dir/inbox"; touch "$f"
n=$(wc -l < "$f")
until [ "$(wc -l < "$f")" -gt "$n" ]; do sleep "${WAITMAIL_POLL_S:-30}"; done
tail -n +$((n + 1)) "$f"
