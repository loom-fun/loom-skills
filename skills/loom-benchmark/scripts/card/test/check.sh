#!/bin/bash
# usage: check.sh [UPDATE=1]   - the card generator is deterministic and its output matches the golden files (CI)
# Builds info.json from fixture packages and evidence (make_info.py perf/gates/build, import_card.py), renders
# README.md and REPORT.md (render_card.py) from fixture copies of every file a card repo holds (model.json,
# model.json.lock, llm_config.json, program/, info.json) and diffs them against test/*/expected/. UPDATE=1
# rewrites the expected files after an intended template change; review the diff before committing it.
set -eu
bad=0
here=$(cd "$(dirname "$0")" && pwd); C=$here/..; tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
py() { python3 -I "$@"; }
F=$here/flint; D=$here/d1
GEN=0123456789abcdef0123456789abcdef01234567   # the card generator commit is an input, never read by the renderer
SRC=(--device "$F/evidence/device.json" --hrx-loom "$F/evidence/hrx_loom.json" --package "$F/pkg")
# a current card: a stamped package (llm_config.json "generator"), drafting, a local perf session, a second
# session that re-measured two rows a day later on another build of the programs (stale: the card marks those
# rows; its decode rows carry their drafted-token acceptance), local gates with the spec gate's acceptance, and
# DevLab gates (an older failing agent run that the local pass supersedes, a full-context needle), a DevLab perf
# session of two rows on the same programs (the card's DevLab columns, next to the local values); REPORT.md generated.
py "$C/make_info.py" perf --loom-jsonl "$F/evidence/loom.jsonl" --llama-jsonl "$F/evidence/llama.jsonl" \
  --c4-loom "$F/evidence/c4_loom_runs.txt" --c4-llama "$F/evidence/c4_llama_runs.txt" --baseline-version b1355 \
  --baseline-args "-ngl 99 -cram 0 --spec-type draft-dflash --spec-draft-model ~/drafter.gguf --spec-draft-n-max 3 -ngld 99 -c 65536 -kvu -np 4" \
  --command "perf.py http://127.0.0.1:8431 loom --model flint --log loom.jsonl --no-campaign" \
  --date 2026-11-06 "${SRC[@]}" -o "$tmp/perf.json" 2>/dev/null
mkdir -p "$tmp/rebuilt"; cp -r "$F/pkg/." "$tmp/rebuilt/"; echo "// another build" >> "$tmp/rebuilt/program/llm.loom"
py "$C/make_info.py" perf --loom-jsonl "$F/evidence/verify_loom.jsonl" --llama-jsonl "$F/evidence/verify_llama.jsonl" \
  --metrics decode,prefill_1k "${SRC[@]}" --package "$tmp/rebuilt" -o "$tmp/perf2.json" 2>/dev/null
dlrows() {  # dlrows IN OUT FACTOR: the verify rows as a DevLab runner measured them (rates x FACTOR: a real gap)
  py -c "import json, sys; f = float(sys.argv[3]); rs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
for r in rs:
    r['decode_tps'] = round(r['decode_tps'] * f, 1); r['prefill'] = [dict(p, prefill_tps=round(p['prefill_tps'] * f, 1)) for p in r['prefill']]
open(sys.argv[2], 'w').write(''.join(json.dumps(r) + '\n' for r in rs))" "$@"
}
dlrows "$F/evidence/verify_loom.jsonl" "$tmp/dl_loom.jsonl" 0.6; dlrows "$F/evidence/verify_llama.jsonl" "$tmp/dl_llama.jsonl" 0.95
py "$C/make_info.py" perf --loom-jsonl "$tmp/dl_loom.jsonl" --llama-jsonl "$tmp/dl_llama.jsonl" \
  --metrics decode,prefill_1k "${SRC[@]}" --source-kind devlab --run 3003 -o "$tmp/perf_dl.json" 2>/dev/null
py "$C/make_info.py" gates "$F/evidence/gates.json" "${SRC[@]}" -o "$tmp/gates.json" 2>/dev/null
# the roofline (evidence/roofline.json: `make_info.py roofline --package pkg --acceptance 0.68` on the real GGUF; CI
# has neither the GGUF nor the gguf package)
mkdir -p "$tmp/flint" "$tmp/d1"
cp -r "$F/pkg/." "$tmp/flint/"
py "$C/make_info.py" build "$tmp/flint" --repo loom-fun/ggml-org-Flint-loom --generator-commit $GEN --memory-bytes 67000000000 \
  --output flint --report generated --evidence "$tmp/perf.json" "$tmp/perf_dl.json" "$tmp/gates.json" "$tmp/perf2.json" "$F/evidence/devlab.json" \
  "$F/evidence/roofline.json" -o "$tmp/flint/info.json" 2>/dev/null
# --from-info: regenerating a published card with only the new evidence gives the same info.json
py "$C/make_info.py" build "$tmp/flint" --repo loom-fun/ggml-org-Flint-loom --generator-commit $GEN --memory-bytes 67000000000 \
  --output flint --report generated --evidence "$tmp/perf.json" "$tmp/perf_dl.json" "$tmp/gates.json" "$tmp/perf2.json" "$F/evidence/devlab.json" \
  -o "$tmp/flint_old.json" 2>/dev/null
# (--repo comes from the old info.json; without --generator-commit it is this checkout's HEAD)
py "$C/make_info.py" build "$tmp/flint" --generator-commit $GEN --from-info "$tmp/flint_old.json" \
  --evidence "$F/evidence/roofline.json" -o "$tmp/flint_regen.json" 2>/dev/null
py "$C/make_info.py" build "$tmp/flint" --from-info "$tmp/flint_old.json" -o "$tmp/flint_head.json" 2>/dev/null
head_sha=$(git -C "$C" rev-parse HEAD 2>/dev/null || echo unknown)
# a card published before DevLab stopped measuring performance: its info.json carries Loom-only DevLab sweeps
# (evidence/devlab_sweeps.json, with their sources). It stays valid loom-card/2, --from-info carries the sweeps over
# unchanged, they are neither rendered nor judged (same README, REPORT and flags as without them), and a DevLab sweep
# in a new evidence file is ignored
mkdir -p "$tmp/swept"; cp -r "$F/pkg/." "$tmp/swept/"
py -c "import json, sys; sys.path.insert(0, sys.argv[1]); import card_lib as c
i, s = c.load_json(sys.argv[2]), c.load_json(sys.argv[3]); i['sources'].update(s['sources']); i['devlab'] = {'sweeps': s['sweeps']}
c.write_json(sys.argv[4], i)" "$C" "$tmp/flint/info.json" "$F/evidence/devlab_sweeps.json" "$tmp/swept_pub.json"
py -c "import json, sys; s = json.load(open(sys.argv[1]))['sweeps'][0]; s['source'] = {'kind': 'devlab', 'run': '1001', 'url': None}
json.dump({'evidence': 'loom-card-evidence/2', 'devlab': {'sweep': s}}, open(sys.argv[2], 'w'))" "$F/evidence/devlab_sweeps.json" "$tmp/sweep_ev.json"
py "$C/make_info.py" build "$tmp/swept" --generator-commit $GEN --from-info "$tmp/swept_pub.json" --evidence "$tmp/sweep_ev.json" \
  -o "$tmp/swept/info.json" 2>/dev/null
py "$C/render_card.py" "$tmp/swept" 2>/dev/null; py "$C/render_card.py" "$tmp/flint" -o "$tmp/flint_README0.md" 2>/dev/null
py -c "import sys; sys.path.insert(0, sys.argv[1]); import card_lib as c
p, f = c.load_json(sys.argv[2]), c.load_json(sys.argv[3])
assert c.validate(p) == [] and c.validate(f) == [], 'not valid'
assert f['devlab'] == p['devlab'] and len(f['devlab']['sweeps']) == 2, 'sweeps not carried over unchanged'
assert c.perf_flags(f) == c.perf_flags(c.load_json(sys.argv[4])), 'flags differ'" "$C" "$tmp/swept_pub.json" "$tmp/swept/info.json" \
  "$tmp/flint/info.json" && cmp -s "$tmp/swept/README.md" "$tmp/flint_README0.md" \
  && echo "OK   flint: a card with old DevLab sweeps stays valid, keeps them, renders and flags as without them" \
  || { echo "FAIL flint: old DevLab sweeps are not carried over harmlessly"; bad=1; }
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
cmp -s "$tmp/flint/info.json" "$tmp/flint_regen.json" && echo "OK   flint: --from-info regenerates the same info.json" \
  || { echo "FAIL flint: --from-info + new evidence differs from building from all evidence"; diff "$tmp/flint/info.json" "$tmp/flint_regen.json"; bad=1; }
py -c "import json, sys; v = json.load(open(sys.argv[1]))['versions']['card_generator']; sys.exit(v != sys.argv[2])" \
  "$tmp/flint_head.json" "$head_sha" && echo "OK   flint: build without --generator-commit records the checkout's HEAD" \
  || { echo "FAIL flint: build without --generator-commit did not record $head_sha"; bad=1; }
# gate merge rules (make_info.py gate_supersedes): on the same programs a later lower-level pass keeps the higher-level
# pass, a later failure always shows, and a pass on new programs replaces evidence on older ones; a published card's
# date-only gate vs a timestamped job of the same day behaves the same
gev() {  # gev OUT STATUS LEVEL TIME PROGRAMS_SHA256: one agent gate's evidence
  py -c "import json, sys; o, st, lv, t, p = sys.argv[1:]; json.dump({'evidence': 'loom-card-evidence/2', 'gates': [{'name': 'agent',
    'status': st, 'level': lv, 'time': t, 'source': {'kind': 'local'}, 'detail': None, 'programs_sha256': p}]}, open(o, 'w'))" "$@"
}
gate_case() {  # gate_case WHAT STATUS/LEVEL/SHA [--from-info F] --evidence EV...: the card's agent gate after the merge
  local what=$1 want=$2 got; shift 2
  py "$C/make_info.py" build "$tmp/flint" --repo loom-fun/ggml-org-Flint-loom --generator-commit $GEN --output flint "$@" \
    -o "$tmp/gc.json" 2>/dev/null
  got=$(py -c "import json, sys; g = json.load(open(sys.argv[1]))['gates']['agent']; print(g['status'], g['level'], g.get('programs_sha256'), sep='/')" "$tmp/gc.json")
  [ "$got" = "$want" ] && echo "OK   gates: $what" || { echo "FAIL gates: $what: $got, expected $want"; bad=1; }
}
gev "$tmp/g_full.json" pass full 2026-11-06T07:00:00 aaa
gev "$tmp/g_quick.json" pass quick 2026-11-07T07:00:00 aaa
gev "$tmp/g_qfail.json" fail quick 2026-11-07T07:00:00 aaa
gev "$tmp/g_new.json" pass full 2026-11-07T07:00:00 bbb
gate_case "quick pass after a full pass keeps the full pass" pass/full/aaa --evidence "$tmp/g_full.json" "$tmp/g_quick.json"
gate_case "quick pass after a full pass, evidence in the other order" pass/full/aaa --evidence "$tmp/g_quick.json" "$tmp/g_full.json"
gate_case "quick fail after a full pass shows the failure" fail/quick/aaa --evidence "$tmp/g_full.json" "$tmp/g_qfail.json"
gate_case "full pass on new programs replaces the old programs' full pass" pass/full/bbb --evidence "$tmp/g_full.json" "$tmp/g_new.json"
cp "$tmp/gc.json" "$tmp/g_pub.json"   # a published card (date-only gates): full pass on bbb on 2026-11-07
gev "$tmp/g_sameday.json" pass quick 2026-11-07T09:00:00 bbb
gate_case "published full pass, a same-day quick pass keeps it" pass/full/bbb --from-info "$tmp/g_pub.json" --evidence "$tmp/g_sameday.json"
gev "$tmp/g_sameday.json" fail quick 2026-11-07T09:00:00 bbb
gate_case "published full pass, a same-day quick fail shows" fail/quick/bbb --from-info "$tmp/g_pub.json" --evidence "$tmp/g_sameday.json"
# local and DevLab sessions both stay (card_lib.card_metrics: local headline, DevLab columns): a metric only DevLab
# measured asks for a local session (perf_flags "devlab"; org_status remeasure reason devlab-primary); a local value
# without a DevLab one on the same programs asks for a DevLab perf session ("devlab_perf"); the gap is never a flag
py "$C/make_info.py" build "$tmp/flint" --repo loom-fun/ggml-org-Flint-loom --generator-commit $GEN --output flint \
  --evidence "$tmp/gates.json" "$tmp/perf_dl.json" -o "$tmp/dl.json" 2>/dev/null
got=$(py -c "import sys; sys.path.insert(0, sys.argv[1]); import card_lib as c
p = c.programs_sha256(sys.argv[3])
for f in sys.argv[2], sys.argv[4]:
    fl = c.perf_flags(c.load_json(f), programs=p)
    print(','.join(sorted(fl['devlab'])), ','.join(c.remeasure_reasons(fl)), ','.join(sorted(fl['devlab_perf'])), sep='/', end=' ')" \
  "$C" "$tmp/dl.json" "$F/pkg" "$tmp/flint/info.json")
want="decode,prefill_1k/devlab-primary/ /stale/decode_2k,decode_32k,prefill_16k,prefill_2k,prefill_32k,prefill_8k,streams4 "
[ "$got" = "$want" ] && echo "OK   flags: local and DevLab sessions both count; each one missing is flagged, the gap never" \
  || { echo "FAIL flags: local/DevLab sessions: '$got', expected '$want'"; bad=1; }
# the spec gate's acceptance is a drafting statistic, not a timing: a DevLab nightly job's (pull_devlab.evidence, from
# its gates_spec.log, stamped with the job's programs) replaces a stale spec_acceptance and is not flagged devlab-primary;
# a job without a spec gate carries none
py "$C/make_info.py" gates "$F/evidence/gates.json" "${SRC[@]}" --package "$tmp/rebuilt" -o "$tmp/gates_old.json" 2>/dev/null
mkdir -p "$tmp/job/spec" "$tmp/job/nospec"
py -c "import json, sys; sys.path.insert(0, sys.argv[1]); import card_lib as c
p = c.programs_sha256(sys.argv[2]); d = sys.argv[3]
for sub, gates in (('spec', {'golden': {'pass': True}, 'spec': {'pass': True}}), ('nospec', {'golden': {'pass': True}})):
    json.dump({'repo': 'loom-fun/ggml-org-Flint-loom', 'slug': 'flint', 'level': 'quick', 'programs_sha256': p},
              open(f'{d}/{sub}/summary.json', 'w'))
    json.dump({'verdict': 'PASS', 'level': 'quick', 'ts': '2026-11-08T03:00:00', 'gates': gates}, open(f'{d}/{sub}/gates.json', 'w'))
open(f'{d}/spec/gates_spec.log', 'w').write('SPEC_CHECK PASS: 13/13 completions identical, 0 near-tie, acceptance 65% of 4278 proposals\n')
open(f'{d}/nospec/gates_spec.log', 'w').write('SPEC_CHECK PASS: 13/13 completions identical, 0 near-tie, acceptance 99% of 10 proposals\n')
from pull_devlab import evidence
for sub in ('spec', 'nospec'):
    s, g = c.load_json(f'{d}/{sub}/summary.json'), c.load_json(f'{d}/{sub}/gates.json')
    c.write_json(f'{d}/ev_{sub}.json', evidence([(s, g, f'{d}/{sub}', 'https://example.invalid/job/1')], 4004))" \
  "$C" "$F/pkg" "$tmp/job"
py "$C/make_info.py" build "$tmp/flint" --repo loom-fun/ggml-org-Flint-loom --generator-commit $GEN --output flint \
  --evidence "$tmp/perf.json" "$tmp/gates_old.json" -o "$tmp/acc_old.json" 2>/dev/null
py "$C/make_info.py" build "$tmp/flint" --generator-commit $GEN --from-info "$tmp/acc_old.json" \
  --evidence "$tmp/job/ev_spec.json" -o "$tmp/acc_new.json" 2>/dev/null
py "$C/make_info.py" build "$tmp/flint" --generator-commit $GEN --from-info "$tmp/acc_old.json" \
  --evidence "$tmp/job/ev_nospec.json" -o "$tmp/acc_nospec.json" 2>/dev/null
got=$(py -c "import sys; sys.path.insert(0, sys.argv[1]); import card_lib as c
p = c.programs_sha256(sys.argv[2])
def st(f):
    i = c.load_json(f); fl = c.perf_flags(i, programs=p)
    return f\"{c.card_metrics(i)['spec_acceptance'][0]['loom']}:{'stale' if 'spec_acceptance' in fl['stale'] else 'fresh'}:{'devlab' if 'spec_acceptance' in fl['devlab'] else 'local-ok'}\"
print(st(sys.argv[3]), st(sys.argv[4]), st(sys.argv[5]), sep='/')" "$C" "$F/pkg" "$tmp/acc_old.json" "$tmp/acc_new.json" "$tmp/acc_nospec.json")
want=$(py -c "import sys; sys.path.insert(0, sys.argv[1]); import card_lib as c
a = c.card_metrics(c.load_json(sys.argv[2]))['spec_acceptance'][0]['loom']; print(f'{a}:stale:local-ok/0.65:fresh:local-ok/{a}:stale:local-ok')" \
  "$C" "$tmp/acc_old.json")
[ "$got" = "$want" ] && echo "OK   flags: a DevLab spec gate's acceptance replaces a stale spec_acceptance (gates only: none)" \
  || { echo "FAIL flags: DevLab spec acceptance: $got, expected $want"; bad=1; }
# c4_steady.py runs that measured nothing (every stream failed: rate -0.0, a negative window) are not samples
printf '%s\n' '{"streams": 4, "window_s": 2.5, "agg_tps": 110.0}' '{"streams": 4, "window_s": -3.1, "agg_tps": -0.0}' 0 112.5 \
  > "$tmp/c4_bad.txt"
got=$(py -c "import sys; sys.path.insert(0, sys.argv[1]); from make_info import c4_samples; print(c4_samples(sys.argv[2]))" "$C" "$tmp/c4_bad.txt" 2>/dev/null)
[ "$got" = "[110.0, 112.5]" ] && echo "OK   c4: invalid c4_steady runs are skipped" || { echo "FAIL c4: $got, expected [110.0, 112.5]"; bad=1; }
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
