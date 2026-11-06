#!/bin/bash
# usage: check.sh [UPDATE=1]   - the card generator is deterministic and its output matches the golden files (CI)
# Builds info.json from fixture packages and evidence (make_info.py perf/gates/build, import_card.py), renders
# README.md (render_card.py) and diffs both against test/*/expected/. UPDATE=1 rewrites the expected files
# after an intended template change; review the diff before committing it.
set -eu
here=$(cd "$(dirname "$0")" && pwd); C=$here/..; tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
py() { python3 -I "$@"; }
F=$here/flint; D=$here/d1
# a current card: drafting, local perf + gates, a later DevLab result (an older failing agent run that the
# local pass supersedes, a full-context needle) and a Loom value from another run
py "$C/make_info.py" perf --loom-jsonl "$F/evidence/loom.jsonl" --llama-jsonl "$F/evidence/llama.jsonl" \
  --c4-loom "$F/evidence/c4_loom_runs.txt" --c4-llama "$F/evidence/c4_llama_runs.txt" --llamacpp-build b1355 \
  --llamacpp-drafting "draft-dflash, n-max 3" --date 2026-11-06 -o "$tmp/perf.json" 2>/dev/null
py "$C/make_info.py" gates "$F/evidence/gates.json" -o "$tmp/gates.json" 2>/dev/null
py "$C/make_info.py" build "$F/pkg" --repo loom-fun/ggml-org-Flint-loom --memory-bytes 67000000000 --output flint \
  --evidence "$tmp/perf.json" "$tmp/gates.json" "$F/evidence/devlab.json" -o "$tmp/flint_info.json" 2>/dev/null
# a pre-loom-card/1 decision-model card through the one-time importer
py "$C/import_card.py" "$D/repo" --run-date 2026-10-29 -o "$tmp/d1_ev.json" 2>/dev/null
py "$C/make_info.py" build "$D/repo" --repo loom-fun/LiquidAI-d1-2B-loom --evidence "$tmp/d1_ev.json" \
  -o "$tmp/d1_info.json" 2>/dev/null
bad=0
for n in flint d1; do
  py "$C/render_card.py" "$tmp/${n}_info.json" -o "$tmp/${n}_README.md" 2>/dev/null
  py "$C/render_card.py" "$tmp/${n}_info.json" -o "$tmp/${n}_README2.md" 2>/dev/null
  cmp -s "$tmp/${n}_README.md" "$tmp/${n}_README2.md" || { echo "FAIL $n: two renders differ"; bad=1; }
  if [ "${UPDATE:-0}" = 1 ]; then
    mkdir -p "$here/$n/expected"
    cp "$tmp/${n}_info.json" "$here/$n/expected/info.json"; cp "$tmp/${n}_README.md" "$here/$n/expected/README.md"
    echo "updated $n/expected"; continue
  fi
  for f in info.json README.md; do
    if diff -u "$here/$n/expected/$f" "$tmp/${n}_$f"; then echo "OK   $n/$f"; else echo "FAIL $n/$f differs from the golden file"; bad=1; fi
  done
  py "$C/render_card.py" --check "$here/$n/expected/info.json" "$here/$n/expected/README.md" >/dev/null \
    || { echo "FAIL $n: expected/README.md is not the render of expected/info.json"; bad=1; }
done
exit $bad
