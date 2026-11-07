#!/bin/bash
# usage: check.sh [UPDATE=1]   - the card generator is deterministic and its output matches the golden files (CI)
# Builds info.json from fixture packages and evidence (make_info.py perf/gates/build, import_card.py), renders
# README.md and REPORT.md (render_card.py) from fixture copies of every file a card repo holds (model.json,
# model.json.lock, llm_config.json, program/, info.json) and diffs them against test/*/expected/. UPDATE=1
# rewrites the expected files after an intended template change; review the diff before committing it.
set -eu
here=$(cd "$(dirname "$0")" && pwd); C=$here/..; tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
py() { python3 -I "$@"; }
F=$here/flint; D=$here/d1
GEN=0123456789abcdef0123456789abcdef01234567   # the card generator commit is an input, never read by the renderer
SRC=(--device "$F/evidence/device.json" --hrx-loom "$F/evidence/hrx_loom.json" --package "$F/pkg")
# a current card: a stamped package (llm_config.json "generator"), drafting, a local perf session, a second
# session that re-measured two rows a day later on another build of the programs (stale: the card marks those
# rows; its decode rows carry their drafted-token acceptance), local gates with the spec gate's acceptance, and
# DevLab gates (an older failing agent run that the local pass supersedes, a full-context needle); REPORT.md generated
py "$C/make_info.py" perf --loom-jsonl "$F/evidence/loom.jsonl" --llama-jsonl "$F/evidence/llama.jsonl" \
  --c4-loom "$F/evidence/c4_loom_runs.txt" --c4-llama "$F/evidence/c4_llama_runs.txt" --baseline-version b1355 \
  --baseline-args "-ngl 99 -cram 0 --spec-type draft-dflash --spec-draft-model ~/drafter.gguf --spec-draft-n-max 3 -ngld 99 -c 65536 -kvu -np 4" \
  --command "perf.py http://127.0.0.1:8431 loom --model flint --log loom.jsonl --no-campaign" \
  --date 2026-11-06 "${SRC[@]}" -o "$tmp/perf.json" 2>/dev/null
mkdir -p "$tmp/rebuilt"; cp -r "$F/pkg/." "$tmp/rebuilt/"; echo "// another build" >> "$tmp/rebuilt/program/llm.loom"
py "$C/make_info.py" perf --loom-jsonl "$F/evidence/verify_loom.jsonl" --llama-jsonl "$F/evidence/verify_llama.jsonl" \
  --metrics decode,prefill_1k "${SRC[@]}" --package "$tmp/rebuilt" -o "$tmp/perf2.json" 2>/dev/null
py "$C/make_info.py" gates "$F/evidence/gates.json" "${SRC[@]}" -o "$tmp/gates.json" 2>/dev/null
# the roofline (evidence/roofline.json: `make_info.py roofline --package pkg --acceptance 0.68` on the real GGUF; CI
# has neither the GGUF nor the gguf package) and two DevLab sweeps of Loom alone (the nightly's perf.jsonl and
# c4_steady.txt, then a later run of the same rows)
py "$C/make_info.py" devlab --loom-jsonl "$F/evidence/devlab_perf.jsonl" --c4-loom "$F/evidence/devlab_c4.txt" --run 1001 \
  --url https://github.com/loom-fun/loom-skills/actions/runs/1001/job/2002 --runner X2-LA03-S03-D07 -o "$tmp/devlab1.json" 2>/dev/null
py "$C/make_info.py" devlab --loom-jsonl "$F/evidence/devlab_perf.jsonl" --run 1002 --date 2026-11-07 \
  --url https://github.com/loom-fun/loom-skills/actions/runs/1002/job/2003 --runner X2-LA03-S04-D07 -o "$tmp/devlab2.json" 2>/dev/null
mkdir -p "$tmp/flint" "$tmp/d1"
cp -r "$F/pkg/." "$tmp/flint/"
py "$C/make_info.py" build "$tmp/flint" --repo loom-fun/ggml-org-Flint-loom --generator-commit $GEN --memory-bytes 67000000000 \
  --output flint --report generated --evidence "$tmp/perf.json" "$tmp/gates.json" "$tmp/perf2.json" "$F/evidence/devlab.json" \
  "$F/evidence/roofline.json" "$tmp/devlab1.json" "$tmp/devlab2.json" -o "$tmp/flint/info.json" 2>/dev/null
# --from-info: regenerating a published card with only the new evidence gives the same info.json
py "$C/make_info.py" build "$tmp/flint" --repo loom-fun/ggml-org-Flint-loom --generator-commit $GEN --memory-bytes 67000000000 \
  --output flint --report generated --evidence "$tmp/perf.json" "$tmp/gates.json" "$tmp/perf2.json" "$F/evidence/devlab.json" \
  "$tmp/devlab1.json" -o "$tmp/flint_old.json" 2>/dev/null
# (--repo comes from the old info.json; without --generator-commit it is this checkout's HEAD)
py "$C/make_info.py" build "$tmp/flint" --generator-commit $GEN --from-info "$tmp/flint_old.json" \
  --evidence "$F/evidence/roofline.json" "$tmp/devlab2.json" -o "$tmp/flint_regen.json" 2>/dev/null
py "$C/make_info.py" build "$tmp/flint" --from-info "$tmp/flint_old.json" -o "$tmp/flint_head.json" 2>/dev/null
head_sha=$(git -C "$C" rev-parse HEAD 2>/dev/null || echo unknown)
# the performance guardrails loom-org's org_status.py applies (card_lib.perf_flags, with the repo's programs: stale)
py -c "import sys; sys.path.insert(0, '$C'); import card_lib as c; print(c.dumps(c.perf_flags(c.load_json('$tmp/flint/info.json'), programs=c.programs_sha256('$F/pkg'))), end='')" \
  > "$tmp/flint/flags.json"
# the acceptance each drafted roofline row counts: its session's (perf.py's counters) where recorded, else the spec gate's
py "$here/roofline_rows.py" "$C" "$F/pkg" "$tmp/flint/info.json" > "$tmp/flint/roofline_rows.json"
# a pre-loom-card/1 decision-model card through the one-time importer: an unstamped package (versions
# recovered from the factory run and the report), its narrative report attached as REPORT.md
cp "$D/repo/model.json" "$D/repo/model.json.lock" "$tmp/d1/"
py "$C/import_card.py" "$D/repo" --lookups "$D/lookups.json" --report-out "$tmp/d1/REPORT.md" -o "$tmp/d1_ev.json" 2>/dev/null
py "$C/make_info.py" build "$tmp/d1" --repo loom-fun/LiquidAI-d1-2B-loom --generator-commit $GEN --report attached \
  --evidence "$tmp/d1_ev.json" -o "$tmp/d1/info.json" 2>/dev/null
bad=0
cmp -s "$tmp/flint/info.json" "$tmp/flint_regen.json" && echo "OK   flint: --from-info regenerates the same info.json" \
  || { echo "FAIL flint: --from-info + new evidence differs from building from all evidence"; diff "$tmp/flint/info.json" "$tmp/flint_regen.json"; bad=1; }
py -c "import json, sys; v = json.load(open(sys.argv[1]))['versions']['card_generator']; sys.exit(v != sys.argv[2])" \
  "$tmp/flint_head.json" "$head_sha" && echo "OK   flint: build without --generator-commit records the checkout's HEAD" \
  || { echo "FAIL flint: build without --generator-commit did not record $head_sha"; bad=1; }
for n in flint d1; do
  files="info.json README.md REPORT.md"; [ "$n" = flint ] && files="$files flags.json roofline_rows.json"
  py "$C/render_card.py" "$tmp/$n" 2>/dev/null
  py "$C/render_card.py" "$tmp/$n" -o "$tmp/${n}_README2.md" 2>/dev/null
  cmp -s "$tmp/$n/README.md" "$tmp/${n}_README2.md" || { echo "FAIL $n: two renders differ"; bad=1; }
  if [ "${UPDATE:-0}" = 1 ]; then
    mkdir -p "$here/$n/expected"
    for f in $files; do cp "$tmp/$n/$f" "$here/$n/expected/$f"; done
    echo "updated $n/expected"; continue
  fi
  for f in $files; do
    if diff -u "$here/$n/expected/$f" "$tmp/$n/$f"; then echo "OK   $n/$f"; else echo "FAIL $n/$f differs from the golden file"; bad=1; fi
  done
  # the golden README (and generated REPORT) must be exactly what the golden info.json renders to with the package files
  cp "$here/$n/expected/info.json" "$tmp/$n/info.json"; cp "$here/$n/expected/README.md" "$tmp/$n/README.md"
  [ "$n" = d1 ] && cp "$here/$n/expected/REPORT.md" "$tmp/$n/REPORT.md"
  py "$C/render_card.py" --check "$tmp/$n" "$tmp/$n/README.md" >/dev/null \
    || { echo "FAIL $n: expected/README.md is not the render of expected/info.json"; bad=1; }
done
exit $bad
