#!/bin/bash
# usage: check.sh [UPDATE=1]   - the card generator is deterministic and its output matches the golden files (CI)
# Builds info.json from fixture packages and evidence (make_info.py perf/gates/build, import_card.py), renders
# README.md and REPORT.md (render_card.py) from fixture copies of every file a card repo holds (model.json,
# model.json.lock, llm_config.json, program/, info.json) and diffs them against test/*/expected/. UPDATE=1
# rewrites the expected files after an intended template change; review the diff before committing it.
set -eu
bad=0
here=$(cd "$(dirname "$0")" && pwd); C=$here/..; tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
export LOOM_CARD_UPSTREAM=none   # offline: vision and audio the package lacks are n/a (no upstream probe)
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
# a card with a failing gate is published as is (loom-org step 2): it validates, renders the gate as a failure, and
# org_status counts it as failing (action fix)
mkdir -p "$tmp/failing"; cp -r "$F/pkg/." "$tmp/failing/"
py "$C/make_info.py" build "$tmp/failing" --repo loom-fun/ggml-org-Flint-loom --generator-commit $GEN --output flint --report generated \
  --evidence "$tmp/gates.json" "$tmp/g_qfail.json" -o "$tmp/failing/info.json" 2>/dev/null
py "$C/render_card.py" "$tmp/failing" 2>/dev/null
py -c "import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); import card_lib as c, render_card as r
from org_status import assess
assert c.validate(c.load_json(sys.argv[3] + '/info.json')) == [], 'not valid'
rows = [l for l in open(sys.argv[3] + '/README.md') if l.startswith('| \x60agent\x60 |')]
assert len(rows) == 1 and f'| {r.NO} fail' in rows[0], rows
assert assess('loom-fun/ggml-org-Flint-loom', sys.argv[3])['failing'] == ['agent']" "$C" "$here/../../../../loom-org/scripts" "$tmp/failing" \
  && echo "OK   failing: a card with a failing gate validates, renders it as fail and org_status counts it failing" \
  || { echo "FAIL failing: a card with a failing gate does not publish as failing"; bad=1; }
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
# a card measured on DevLab only (no local value): the DevLab values are the headline columns (llama.cpp, Loom, margin,
# roofline), the empty local and local ÷ DevLab columns are dropped, a footnote says where the numbers are from, and the
# session's device and SDK (pull_devlab.py --identity: runner.txt, install.log) and runs per value are on the card
py "$C/pull_devlab.py" --identity "$here/devlab_job" --device-out "$tmp/jd.json" --hrx-out "$tmp/jh.json" 2>/dev/null
py "$C/make_info.py" perf --loom-jsonl "$tmp/dl_loom.jsonl" --llama-jsonl "$tmp/dl_llama.jsonl" --metrics decode,prefill_1k \
  "${SRC[@]}" --source-kind devlab --run 3003 --device "$tmp/jd.json" --hrx-loom "$tmp/jh.json" -o "$tmp/perf_dl2.json" 2>/dev/null
mkdir -p "$tmp/dlonly"; cp -r "$F/pkg/." "$tmp/dlonly/"
py "$C/make_info.py" build "$tmp/dlonly" --repo loom-fun/ggml-org-Flint-loom --generator-commit $GEN --output flint \
  --evidence "$tmp/gates.json" "$tmp/perf_dl2.json" -o "$tmp/dlonly/info.json" 2>/dev/null
py "$C/render_card.py" "$tmp/dlonly" 2>/dev/null
py -c "import sys; r = open(sys.argv[1] + '/README.md').read()
assert '| metric | llama.cpp | Loom | margin | roofline | Loom % of roofline |\n' in r, 'header'
assert 'local ÷ DevLab' not in r and 'DevLab Loom' not in r, 'local columns'
row = [l for l in r.splitlines() if l.startswith('| decode tok/s')][0]
assert row.count('|') == 7 and '—' not in row.split('|')[2] and '—' not in row.split('|')[3] and '—' not in row.split('|')[4], row
assert 'measured on AMD DevLab Dispatch Strix Halo runners' in r and 'Runs per value: decode and prefill' in r
assert 'device not recorded' not in r and 'hrx-loom unknown' not in r, 'footnote'
assert 'X2-LA03-S03-D01' in r and '0.2.0.dev0' in r, 'device/sdk'" "$tmp/dlonly" \
  && echo "OK   devlab-only: a card with only DevLab values shows them as its headline columns" \
  || { echo "FAIL devlab-only: DevLab-only card does not render as expected"; bad=1; }
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
# a paused card (make_info.py build --from-info --next-steps --status paused): README.md is the same card with a final
# "Next steps" section after everything else, org_status reports paused; without the fields nothing changes
printf '%s\n' '- retry the agent gate on DevLab' '' 'measure `decode_32k` locally' > "$tmp/next.txt"
mkdir -p "$tmp/paused"; cp -r "$F/pkg/." "$tmp/paused/"
py "$C/make_info.py" build "$tmp/paused" --generator-commit $GEN --from-info "$tmp/failing/info.json" --next-steps "$tmp/next.txt" \
  --status paused -o "$tmp/paused/info.json" 2>/dev/null
py "$C/render_card.py" "$tmp/paused" 2>/dev/null
py "$C/make_info.py" build "$tmp/paused" --generator-commit $GEN --from-info "$tmp/paused/info.json" --next-steps /dev/null \
  --status active -o "$tmp/unpaused.json" 2>/dev/null
printf '\n## Next steps\n\nWork on this model is paused.\n\n- retry the agent gate on DevLab\n- measure `decode_32k` locally\n' \
  | cat "$tmp/failing/README.md" - | cmp -s - "$tmp/paused/README.md" && cmp -s "$tmp/failing/info.json" "$tmp/unpaused.json" \
  && py -c "import sys; sys.path.insert(0, sys.argv[1]); from org_status import assess
a, b = assess('loom-fun/ggml-org-Flint-loom', sys.argv[2]), assess('loom-fun/ggml-org-Flint-loom', sys.argv[3])
assert a.get('paused') is True and 'paused' not in b and a['failing'] == b['failing']" "$here/../../../../loom-org/scripts" \
  "$tmp/paused" "$tmp/failing" \
  && echo "OK   paused: next steps render last, org_status reports paused, and clearing them gives the card back" \
  || { echo "FAIL paused: a paused card's next steps do not render or clear as expected"; bad=1; }
# a document longer than the model's native context (perf.py: the Loom server's HTTP 400, logged in over_context with
# the prompt tokens make_info.py counts) is not applicable: org_status does not ask for it and the card shows n/a. It
# stays required when the refused prompt fits the native context (a package built smaller), when the native context
# is unknown, or once a session measured it.
echo '{"label": "loom", "kind": "loom", "time": "2026-11-07 10:00:00", "nonce": "1.0", "prefill_long": [{"prompt_tokens": 8650, "prefill_tps": 5000.0}, {"prompt_tokens": 17351, "prefill_tps": 4000.0}], "over_context": ["prefill_32000", "decode_tps_32k"], "decode_tps_32k": null}' \
  > "$tmp/oc_loom.jsonl"
py "$C/make_info.py" perf --loom-jsonl "$tmp/oc_loom.jsonl" "${SRC[@]}" -o "$tmp/perf_oc.json" 2>/dev/null
mkdir -p "$tmp/oc" "$tmp/oc_native"; cp -r "$F/pkg/." "$tmp/oc/"; cp -r "$F/pkg/." "$tmp/oc_native/"
py -c "import json, sys
c = json.load(open(sys.argv[1])); c['context_length'] = 32768; json.dump(c, open(sys.argv[1], 'w'))
for n, f in (34937, sys.argv[3]), (30000, sys.argv[4]):
    e = json.load(open(sys.argv[2])); e['performance']['session']['over_context'] = {'prefill_32k': n, 'decode_32k': n}
    json.dump(e, open(f, 'w'))" "$tmp/oc_native/llm_config.json" "$tmp/perf_oc.json" "$tmp/perf_oc_long.json" "$tmp/perf_oc_fits.json"
for e in long fits both; do
  ev="$tmp/perf_oc_long.json"; [ $e = fits ] && ev="$tmp/perf_oc_fits.json"; [ $e = both ] && ev="$ev $tmp/perf.json"
  py "$C/make_info.py" build "$tmp/oc_native" --repo loom-fun/ggml-org-Flint-loom --generator-commit $GEN --output flint \
    --evidence "$tmp/gates.json" $ev -o "$tmp/oc_$e.json" 2>/dev/null
done
py "$C/make_info.py" build "$tmp/oc" --repo loom-fun/ggml-org-Flint-loom --generator-commit $GEN --output flint \
  --evidence "$tmp/gates.json" "$tmp/perf_oc_long.json" -o "$tmp/oc/info.json" 2>/dev/null
cp "$tmp/oc_long.json" "$tmp/oc_native/info.json"; py "$C/render_card.py" "$tmp/oc_native" 2>/dev/null
py -c "import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); import card_lib as c
from org_status import assess
long = {'metric:prefill_32k', 'metric:decode_32k'}
miss = assess('loom-fun/ggml-org-Flint-loom', sys.argv[3])['missing']
assert not long & set(miss) and 'metric:prefill_16k' not in miss and 'metric:prefill_1k' in miss, miss
rows = [l for l in open(sys.argv[3] + '/README.md') if l.startswith('| prefill tok/s, 32k document |' + ' n/a |' * 8)]
assert rows and \"longer than the model's native 32768-token context\" in open(sys.argv[3] + '/README.md').read()
assert long <= set(assess('loom-fun/ggml-org-Flint-loom', sys.argv[4])['missing']), 'native context unknown'
facts = c.package_facts(c.load_package(sys.argv[3]))
for f in sys.argv[5:]:   # a refusal within the native context, and a measured value: still required
    assert c.over_context(c.load_json(f), c.native_context(facts['features'])) == set(), f" \
  "$C" "$here/../../../../loom-org/scripts" "$tmp/oc_native" "$tmp/oc" "$tmp/oc_fits.json" "$tmp/oc_both.json" \
  && echo "OK   over context: n/a only past the native context; required within it, without it, or once measured" \
  || { echo "FAIL over context: a refused long document is not judged against the native context"; bad=1; }
# c4_steady.py runs that measured nothing (every stream failed: rate -0.0, a negative window) are not samples
printf '%s\n' '{"streams": 4, "window_s": 2.5, "agg_tps": 110.0}' '{"streams": 4, "window_s": -3.1, "agg_tps": -0.0}' 0 112.5 \
  > "$tmp/c4_bad.txt"
got=$(py -c "import sys; sys.path.insert(0, sys.argv[1]); from make_info import c4_samples; print(c4_samples(sys.argv[2]))" "$C" "$tmp/c4_bad.txt" 2>/dev/null)
[ "$got" = "[110.0, 112.5]" ] && echo "OK   c4: invalid c4_steady runs are skipped" || { echo "FAIL c4: $got, expected [110.0, 112.5]"; bad=1; }
# a second llama.cpp backend (make_info.py perf --backend vulkan: a baseline-only DevLab session under
# performance.baselines): validates, renders as one more DevLab column and a footnote with its build, archive
# sha256 and the settings it could not run, changes no headline value or guardrail (card_lib.card_metrics/perf_flags),
# the latest Vulkan evidence replaces the earlier one, --from-info carries it over, a run that fetched other programs
# than the package is refused, and the card without it renders as before (the goldens)
dlrows "$F/evidence/verify_llama.jsonl" "$tmp/vk_llama.jsonl" 0.8
VK=(--backend vulkan --baseline-version b12211 --baseline-name llama.cpp --baseline-sha256 aabbccdd00112233445566778899aabbccddeeff00112233445566778899aabb
    --baseline-args "-cram 0 --spec-type draft-dflash -c 65536 -kvu -np 4" --differs "vision projector (--mmproj): not loaded, the workloads are text-only"
    --metrics decode,prefill_1k --source-kind devlab --run 5005 --device none --hrx-loom none --package "$F/pkg")
py "$C/make_info.py" perf --llama-jsonl "$tmp/vk_llama.jsonl" "${VK[@]}" -o "$tmp/vk1.json" 2>/dev/null
py "$C/make_info.py" perf --llama-jsonl "$tmp/dl_llama.jsonl" "${VK[@]}" -o "$tmp/vk2.json" 2>/dev/null
mkdir -p "$tmp/vk"; cp -r "$F/pkg/." "$tmp/vk/"
py "$C/make_info.py" build "$tmp/vk" --repo loom-fun/ggml-org-Flint-loom --generator-commit $GEN --memory-bytes 67000000000 --output flint \
  --report generated --evidence "$tmp/perf.json" "$tmp/perf_dl.json" "$tmp/gates.json" "$tmp/perf2.json" "$F/evidence/devlab.json" \
  "$F/evidence/roofline.json" "$tmp/vk1.json" -o "$tmp/vk/info.json" 2>/dev/null
py "$C/make_info.py" build "$tmp/vk" --generator-commit $GEN --from-info "$tmp/vk/info.json" --evidence "$tmp/vk2.json" -o "$tmp/vk_new.json" 2>/dev/null
py "$C/make_info.py" build "$tmp/vk" --generator-commit $GEN --from-info "$tmp/vk_new.json" -o "$tmp/vk_same.json" 2>/dev/null
py "$C/render_card.py" "$tmp/vk" 2>/dev/null
py -c "import sys; sys.path.insert(0, sys.argv[1]); import card_lib as c
base, vk, new = (c.load_json(f) for f in sys.argv[2:5]); p = c.programs_sha256(sys.argv[5])
assert c.validate(vk) == [] and c.validate(new) == [], 'not valid'
b = vk['performance']['baselines']['vulkan']; n = new['performance']['baselines']['vulkan']
assert b['baseline']['version'] == 'llamacpp-vulkan/b12211' and b['baseline']['sha256'].startswith('aabbcc') and len(b['baseline']['differs']) == 1
assert b['session']['metrics']['decode'] == {'baseline': 56.5, 'baseline_runs': [56.2, 56.8]} and 'loom' not in b['session']['metrics']['decode'], b['session']['metrics']['decode']
assert n['session']['metrics']['decode']['baseline'] != b['session']['metrics']['decode']['baseline'], 'the later Vulkan evidence did not replace the earlier'
assert new['performance']['baselines'].keys() == {'vulkan'} and 'baselines' not in base['performance']
assert vk['performance']['sessions'] == base['performance']['sessions'] and vk['sources'].keys() >= base['sources'].keys()
assert c.perf_flags(vk, programs=p) == c.perf_flags(base, programs=p) and \
    {k: v[0]['loom'] for k, v in c.card_metrics(vk).items()} == {k: v[0]['loom'] for k, v in c.card_metrics(base).items()}" \
  "$C" "$tmp/flint/info.json" "$tmp/vk/info.json" "$tmp/vk_new.json" "$F/pkg" \
  && cmp -s "$tmp/vk_new.json" "$tmp/vk_same.json" \
  && grep -q '| DevLab llama.cpp Vulkan |' "$tmp/vk/README.md" && grep -q 'Differs from llama.cpp: vision projector' "$tmp/vk/README.md" \
  && grep -q '^## llama.cpp Vulkan: DevLab Dispatch run 5005' "$tmp/vk/REPORT.md" \
  && echo "OK   vulkan: a baseline-only session renders as a DevLab column and changes no headline value or flag" \
  || { echo "FAIL vulkan: the second backend's session does not validate, render or leave the card's values alone"; bad=1; }
py "$C/make_info.py" perf --llama-jsonl "$tmp/vk_llama.jsonl" "${VK[@]}" --fetched-programs 0000 -o "$tmp/vk_bad.json" 2>/dev/null \
  && { echo "FAIL vulkan: a run that fetched other programs than the package was accepted"; bad=1; } \
  || echo "OK   vulkan: a run that fetched other programs than the package is refused"
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
# quant_of: any-case names (gguf-my-repo writes q4_k_m) without loosening the boundary guard
got=$(py -c "import sys; sys.path.insert(0, sys.argv[1]); import card_lib as c
print(*[c.quant_of(n) for n in ('hyperclovax-1.5b-q4_k_m.gguf', 'x-Q4_K_Mfoo', 'x-Q4_K_M_foo', 'model-q4-k-m', 'x-Q8_0-q4_0')])" "$C")
[ "$got" = "Q4_K_M Q4_K Q4_K_M None Q8_0" ] && echo "OK   quant_of" || { echo "FAIL quant_of: $got"; bad=1; }
# features: n/a when the upstream model lacks it, no (red) only when upstream has it and Loom does not, unknown (a neutral ?)
# when upstream cannot be read; an unreadable config.json never reads as absent, an org-less model_id falls back to the base_model tag
got=$(py -c "import sys; sys.path.insert(0, sys.argv[1]); import card_lib as c
H = 'https://huggingface.co/'
fake = {H + 'api/models/o/m-GGUF': {'siblings': [{'rfilename': 'mmproj-m.gguf'}], 'tags': ['base_model:o/m']},
        H + 'o/m/resolve/main/config.json': {'audio_config': {}},
        H + 'api/models/o/t-GGUF': {'siblings': [], 'tags': ['base_model:o/t']}, H + 'o/t/resolve/main/config.json': {'hidden': 1},
        H + 'api/models/o/g-GGUF': {'siblings': [], 'tags': ['base_model:o/g']},
        H + 'api/models/o/g': {'config': {'architectures': ['XForCausalLM']}},
        H + 'api/models/o/u-GGUF': {'siblings': [], 'tags': ['base_model:o/u']},
        H + 'api/models/o/u': {'config': {'architectures': ['XForConditionalGeneration']}}}
up = c.upstream_facts('o/m-GGUF', 'bare-name', fetch_json=fake.get)
txt = c.upstream_facts('o/t-GGUF', 'o/t', fetch_json=fake.get)
gated = c.upstream_facts('o/g-GGUF', 'o/g', fetch_json=fake.get)
und = c.upstream_facts('o/u-GGUF', 'o/u', 'http://x/y.gguf', fetch_json=fake.get, fetch_template=lambda u: 'x {{ tools }} enable_thinking')
cfg = {'chat': {'call_format': 'none', 'thinking': 'never'}, 'drafter': {'tuned': [{'kind': 'none'}, {'kind': 'dflash'}]}}
f = c.derive_features({}, cfg, 'chat', {}, {}, None, up)
g = c.derive_features({}, {'chat': {'call_format': 'hermes', 'thinking': 'never'}}, 'chat', {}, {}, None, txt)
old = c.derive_features({}, {'chat': {}}, 'chat', {}, {}, None, und)
old0 = c.derive_features({}, {'chat': {}}, 'chat', {}, {}, None, None)
st = lambda d, *ks: [d[k]['status'] for k in ks]
print(up['vision'], up['audio'], txt['vision'], txt['audio'], gated['audio'], und['audio'], und['template'],
      *st(f, 'image_input', 'audio', 'tool_calling', 'thinking', 'speculative_decoding'), c.drafting(f),
      *st(g, 'image_input', 'audio', 'tool_calling', 'speculative_decoding'), *st(old, 'tool_calling', 'thinking', 'reasoning_effort', 'audio'),
      *st(old0, 'tool_calling', 'audio'))" "$C")
[ "$got" = "True True False False False None {'tools': True, 'thinking': True, 'reasoning_effort': False} no no n/a n/a yes False n/a n/a yes n/a no no n/a unknown unknown unknown" ] && echo "OK   features: n/a vs no vs not determined" || { echo "FAIL features: $got"; bad=1; }
# an available but disabled drafter reads "available, disabled (METHOD)", the method from the data
got=$(py -c "import sys; sys.path.insert(0, sys.argv[1]); import render_card as r
print(r.feature_detail('speculative_decoding', {'status': 'yes', 'value': {'method': None, 'available': ['dflash']}}))" "$C")
[ "$got" = "available, disabled (dflash)" ] && echo "OK   unused drafter detail" || { echo "FAIL unused drafter detail: $got"; bad=1; }
# org_status consistency checks: a nightly entry that differs from the card (memory_bytes, llama.cpp build) or is missing,
# and a generator commit that is not on main: not an ancestor of main and in no merged PR (a throwaway git repo with one
# commit on main and one on a branch; the PR lookup is injected: merged, open, unknown)
git init -q -b main "$tmp/skills" && git -C "$tmp/skills" -c user.name=t -c user.email=t@t commit -q --allow-empty -m m
git -C "$tmp/skills" checkout -q -b side && git -C "$tmp/skills" -c user.name=t -c user.email=t@t commit -q --allow-empty -m s
py -c "import json, subprocess, sys; sys.path.insert(0, sys.argv[1]); from org_status import assess, consistency_checks
r = assess('loom-fun/ggml-org-Flint-loom', sys.argv[2])
g = lambda ref: subprocess.check_output(['git', '-C', sys.argv[3], 'rev-parse', ref], text=True).strip()
card = {'memory_bytes': r['memory_bytes'], 'llamacpp': r['llamacpp']}
assert card == {'memory_bytes': 67000000000, 'llamacpp': 'b1355'}, card
ok = {r['repo']: dict(repo=r['repo'], **card)}
r['generator'] = g('main')
assert consistency_checks(r, ok, sys.argv[3]) == []
assert consistency_checks(r, None, None) == []
assert len(consistency_checks(r, {}, None)) == 1
assert len(consistency_checks(r, {r['repo']: dict(ok[r['repo']], memory_bytes=69000000000)}, None)) == 1
assert len(consistency_checks(r, {r['repo']: dict(ok[r['repo']], llamacpp='b1368')}, None)) == 1
r['generator'] = g('side')
assert len(consistency_checks(r, ok, sys.argv[3], lambda c: [{'merged_at': None}])) == 1
assert len(consistency_checks(r, ok, sys.argv[3], lambda c: [])) == 1
assert consistency_checks(r, ok, sys.argv[3], lambda c: [{'merged_at': '2026-11-08T00:00:00Z'}]) == []   # squash-merged
assert consistency_checks(r, ok, sys.argv[3], lambda c: None) == []   # unknown never flags
from org_status import merged_prs
def fake(merged):   # GitHub: a refs/pull/N/merge commit is in no PR; its second parent (the PR head) is in the PR
    def get(path):
        if path == 'commits/M/pulls': return []
        if path == 'commits/M': return {'commit': {'message': 'Merge H into B'}, 'parents': [{'sha': 'B'}, {'sha': 'H'}]}
        if path == 'commits/H/pulls': return [{'merged_at': merged}]
        return None
    return get
assert merged_prs('M', fake('2026-11-08')) == [{'merged_at': '2026-11-08'}]   # merge-ref of a merged PR
assert merged_prs('M', fake(None)) == [{'merged_at': None}]   # of an open one: still flagged
assert merged_prs('X', lambda p: []) == []   # a plain orphan commit: no PR
def wave(files):   # an open PR (a campaign wave) that changes only the given files
    return lambda path: ([{'number': 7, 'merged_at': None}] if path.endswith('/pulls') else [{'filename': f} for f in files])
assert merged_prs('H', wave(['.github/devlab/port.json']))[0]['merged_at']   # config only: the base generated the programs
assert not merged_prs('H', wave(['.github/devlab/port.json', 'skills/x.py']))[0]['merged_at']
from org_status import changed_files   # a temporary DevLab commit on top of main: config only counts as main, skills/ stays flagged
cmp = lambda files: (lambda path: {'files': [{'filename': f} for f in files]})
assert changed_files('T', cmp(['.github/devlab/port.json'])) == ['.github/devlab/port.json']
r['generator'] = g('side')
none = lambda c: []
assert consistency_checks(r, ok, sys.argv[3], none, lambda c: ['.github/devlab/port.json', '.github/devlab/perf.json']) == []
assert len(consistency_checks(r, ok, sys.argv[3], none, lambda c: ['.github/devlab/port.json', 'skills/x.py'])) == 1
assert consistency_checks(r, ok, sys.argv[3], none, lambda c: None) != [] and consistency_checks(r, ok, sys.argv[3], lambda c: None, lambda c: None) == []" "$here/../../../../loom-org/scripts" "$tmp/flint" "$tmp/skills" \
  && echo "OK   org_status checks: nightly entry matches the card, generator commit is on main or in a merged PR" \
  || { echo "FAIL org_status checks: nightly/generator consistency flags are wrong"; bad=1; }
exit $bad
