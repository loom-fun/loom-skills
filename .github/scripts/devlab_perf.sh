#!/bin/bash
# One job of a DevLab performance experiment (.github/workflows/devlab-perf.yml): fetch a published loom-fun card,
# build package variants from this checkout (or another ref), and compare them with each other and with llama.cpp
# ON THIS RUNNER (runners differ from each other and from a dev machine: only same-job numbers are compared).
#
#   [WORK=~/devlab-perf] devlab_perf.sh EXPERIMENT.json JOB_NAME
#
# EXPERIMENT.json:
#   {"card": "loom-fun/<repo>", "memory_bytes": N, "kit": "gemma4" | null, "llamacpp": "b1368",
#    "llama_spec": "--spec-type draft-dflash --spec-draft-model {drafter} ...",   ({drafter}: the card's drafter GGUF)
#    "jobs": [{"name": "j", "variants": [V, ...], "steps": [S, ...]}]}   (a job may override card/kit/memory_bytes/llamacpp)
# Variant V: {"name": "card"}                                    the card's own programs, as published
#            {"name": "x", "build": {"ENV": "1"}, "build_args": [...], "ref": "origin/main",
#             "execution": {"draft_depth": 3}, "double": ["llm_verify:g_head_mv", "*:g_mv_q8_0"]}
#            build: build.py with these env vars (and extra flags); ref: build with that commit's starter;
#            execution: model.json overrides; double: ablation (each kernel.launch in PROGRAM matching REGEX twice).
# Job "thinking": off (default) | on: chat_template_kwargs.enable_thinking sent to BOTH engines in every step.
# Steps S: ab:R[:on|off]        every variant in ONE loom-serve (small context), ab_decode.py R interleaved rounds
#          perf:V:N[:long]      V served alone with its own settings: perf.py N times (+ --long)
#          c4:V                 c4_steady.py 4 streams on V
#          llama:N[:long|-][:M] llama.cpp (card build + llama_spec, n-max M; M=0: no drafting) perf.py N times, llama_decode.py tokens/step
#          llamac4[:M]          c4_steady.py on llama.cpp
#          textmatch:V          text_match.py: first 50 tokens of perf.py's prompts on V, then on llama.cpp
#          consistency:V:N      consistency.py N times (NOT DETERMINISTIC / concurrent mismatch counts)
#          determinism:V:C:R    determinism.py (issue #39's script, when present)
#          ktest:V              make_kernel_tests.py + run_kernel_tests.sh with V's kit
#          gates:V:LEVEL        gates.sh against llama.cpp
# Writes WORK/<job>/out (summary.json, *.jsonl, *.log). Every server it starts is stopped at exit.
set -uo pipefail
exp=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); job=${2:?usage: devlab_perf.sh EXPERIMENT.json JOB}
here=$(cd "$(dirname "$0")/../.." && pwd); SK=$here/skills
B=$SK/loom-benchmark/scripts; SV=$SK/loom-setup/scripts
WORK=${WORK:-$HOME/devlab-perf}; mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
D=$WORK/$job; out=$D/out; rm -rf "$D"; mkdir -p "$out"
export LOOM_SERVE_STATE=$D/state LLAMACPP_STATE=$D/llstate PYTHONDONTWRITEBYTECODE=1
PORT=${LOOM_PORT:-8161}; LLAMA_PORT=${LLAMA_PORT:-8131}; url=http://127.0.0.1:$PORT; lurl=http://127.0.0.1:$LLAMA_PORT
t0=$(date +%s); step() { echo "== $* ($(date +%T), +$(( $(date +%s) - t0 ))s)"; }
q() { python3 -c "import json,sys; e=json.load(open('$exp')); j=next(j for j in e['jobs'] if j['name']=='$job'); $1"; }
qv() { python3 -c "import json,sys; e=json.load(open('$exp')); j=next(j for j in e['jobs'] if j['name']=='$job'); v=next(v for v in j['variants'] if v['name']=='$1'); $2"; }
repo=$(q "print(j.get('card') or e['card'])"); mem=$(q "print(j.get('memory_bytes') or e['memory_bytes'])")
kit=$(q "print((j['kit'] if 'kit' in j else e.get('kit')) or '-')"); tag=$(q "print(j.get('llamacpp') or e.get('llamacpp') or '-')"); variants=$(q "print(' '.join(v['name'] for v in j['variants']))")
steps=$(q "print(' '.join(j['steps']))")
echo "job $job: variants $variants; steps $steps; runner $(hostname)"
hostname > "$out/runner.txt"

cleanup() {
  for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
  "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
  cp "$LOOM_SERVE_STATE"/*.log "$LLAMACPP_STATE"/*.log "$out/" 2>/dev/null
  echo "devlab_perf.sh: servers stopped"
}
trap cleanup EXIT

# 1. Loom SDK + tools venv
root=${LOOM_ROOT:-$WORK/loom}
if [ ! -x "$root/.venv/bin/loom-serve" ] || [ ! -x "$root/.tools/bin/python" ]; then
  step "install Loom into $root"; "$SV/install_loom.sh" "$root" ${LOOM_VERSION:+"$LOOM_VERSION"} > "$out/install.log" 2>&1 || { tail "$out/install.log"; exit 1; }
fi
export LOOM_ROOT=$root VENV=$root/.venv; T=$root/.tools/bin/python

# 2. The card (fetched once per runner)
card=$WORK/_card; base=https://huggingface.co/$repo/resolve/main
if [ "$(cat "$card/.fetched" 2>/dev/null)" != "$repo" ]; then   # a stale card from another job is refetched
  step "fetch $repo"; rm -rf "$card"
  "$root/.venv/bin/loom-fetch" "$base/model.json" --no-input --materialize=copy --cache "$WORK/fetch-cache" --output="$card" > "$out/fetch.log" 2>&1 || { tail "$out/fetch.log"; echo "FAIL: loom-fetch"; exit 1; }
  curl -fsSL --retry 3 "$base/llm_config.json" -o "$card/llm_config.json" || exit 1
  rm -rf "$WORK/fetch-cache"; echo "$repo" > "$card/.fetched"
fi
# build.py's config: the card's llm_config.json with its GGUFs pointed at the fetched files, plus the tensor table
"$T" - "$card" "$D/cfg.json" <<'EOF' || exit 1
import json, os, sys
import gguf
card, dst = sys.argv[1:]
c = json.load(open(os.path.join(card, "llm_config.json")))
par = lambda f: os.path.join(card, "parameter", os.path.basename(f))
c["gguf"] = par(c["gguf"])
if c.get("vision"):
    c["vision"]["mmproj"] = par(c["vision"]["mmproj"])
for k in ("drafter", "assistant"):
    if c.get(k) and c[k].get("gguf"):
        c[k]["gguf"] = par(c[k]["gguf"]) if os.path.exists(par(c[k]["gguf"])) else None
r = gguf.GGUFReader(c["gguf"])
c["tensors"] = {t.name: {"type": t.tensor_type.name.lower(), "shape": [int(x) for x in t.shape]} for t in r.tensors}
for k in ("model_id", "draft"):
    c.pop(k, None)
json.dump(c, open(dst, "w"), indent=1)
EOF
gguf=$("$T" -c "import json; print(json.load(open('$D/cfg.json'))['gguf'])")
drafter=$("$T" -c "import json; print((json.load(open('$D/cfg.json')).get('drafter') or {}).get('gguf') or '')")
cardid=$("$T" -c "import json; print(json.load(open('$card/model.json'))['model']['id'])")
# the card's own build settings (model id, depth, context, pool)
cardargs=$("$T" -c "
import json; m=json.load(open('$card/model.json')); c=json.load(open('$card/llm_config.json')); ex=m['presets']['chat']['execution']
a=['--context', str(ex['context_capacity']), '--pool', str(ex['pool_capacity']), '--prefill', str(ex['prefill_capacity'])]
d=c.get('draft') or {}
a += (['--draft', d['kind'], '--draft-depth', str(d['depth'])] if d.get('kind') else [])
print(' '.join(a))")

# 3. Packages: D/<variant>/pkg (+ D/<variant>/kit for kernel tests)
declare -A KIT
kit_for() {   # kit_for VARIANT REF -> builds D/<variant>/kit from REF's starter (or this checkout)
  local v=$1 ref=$2 PL=$SK/loom-port-llm
  if [ -n "$ref" ]; then
    git -C "$here" fetch -q origin "${ref#origin/}" 2>/dev/null
    rm -rf "$D/$v/src"; mkdir -p "$D/$v/src"
    git -C "$here" archive "$ref" skills/loom-port-llm | tar -x -C "$D/$v/src" || { echo "FAIL: no $ref"; return 1; }
    PL=$D/$v/src/skills/loom-port-llm; echo "starter from $ref ($(git -C "$here" log -1 --format=%h:%s "$ref" | cut -c1-80))"
  fi
  if [ "$kit" != - ]; then "$PL/reference/apply_example.sh" "$kit" "$D/$v/kit" | tail -1 >/dev/null || return 1
  else cp -r "$PL/starter" "$D/$v/kit"; fi
  KIT[$v]=$D/$v/kit
}
for v in $variants; do
  pkg=$D/$v/pkg; mkdir -p "$D/$v"
  if [ "$(qv "$v" "print('build' in v)")" = True ]; then
    kit_for "$v" "$(qv "$v" "print(v.get('ref') or '')")" || exit 1
    extra=$(qv "$v" "print(' '.join(v.get('build_args') or []))")
    envs=$(qv "$v" "print(' '.join(f'{k}={x}' for k, x in (v.get('build') or {}).items()))")
    step "build $v: env $envs; build.py $cardargs $extra"
    ( cd "${KIT[$v]}" && env $envs "$T" build.py "$D/cfg.json" "$pkg" --tokenizer "$card/tokenizer.json" --model-id "$cardid" $cardargs $extra ) > "$out/build_$v.log" 2>&1 \
      || { tail -30 "$out/build_$v.log"; echo "FAIL: build $v"; exit 1; }
    tail -3 "$out/build_$v.log"
  else
    kit_for "$v" "" || exit 1
    mkdir -p "$pkg"; cp -r "$card/program" "$card/model.json" "$card/tokenizer.json" "$card"/*.loom "$pkg/" 2>/dev/null
    ln -sfn "$card/parameter" "$pkg/parameter"
    "$T" - "$card" "$pkg" <<'EOF'
import json, os, sys
card, pkg = sys.argv[1:]
c = json.load(open(os.path.join(card, "llm_config.json")))
par = lambda f: os.path.join(card, "parameter", os.path.basename(f))
c["gguf"] = par(c["gguf"])
if c.get("vision"):
    c["vision"]["mmproj"] = par(c["vision"]["mmproj"])
for k in ("draft", "drafter", "assistant"):
    if isinstance(c.get(k), dict) and c[k].get("gguf"):
        c[k]["gguf"] = par(c[k]["gguf"])
json.dump(c, open(os.path.join(pkg, "llm_config.json"), "w"), indent=1)
EOF
  fi
  # ablation: every kernel.launch in PROGRAM matching REGEX ("PROGRAM:REGEX", PROGRAM * = any) runs twice
  "$T" - "$pkg/program/llm.loom" "$(qv "$v" "print(json.dumps(v.get('double') or []))")" <<'EOF'
import json, re, sys
p, rules = sys.argv[1], [r.split(":", 1) for r in json.loads(sys.argv[2])]
if rules:
    out, prog, n = [], None, 0
    for ln in open(p).read().split("\n"):
        m = re.match(r"command\.program\.def \w+ @(\w+)", ln)
        if m:
            prog = m.group(1)
        out.append(ln)
        if re.match(r"\s*kernel\.launch @", ln) and any(pr in ("*", prog) and re.search(rx, ln) for pr, rx in rules):
            out.append(ln); n += 1
    open(p, "w").write("\n".join(out)); print(f"ablation: doubled {n} launches")
EOF
  "$T" - "$pkg/model.json" "$(qv "$v" "print(json.dumps(v.get('execution') or {}))")" "$cardid-$v" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1])); ex = m["presets"]["chat"]["execution"]; ex.update(json.loads(sys.argv[2]))
json.dump(m, open(sys.argv[1], "w"), indent=1)
# a small-context copy with its own model id, for several packages in one loom-serve (ab steps)
m["model"]["id"] = sys.argv[3]; ex["context_capacity"] = 16384; ex["pool_capacity"] = 32768
json.dump(m, open(sys.argv[1].replace("model.json", "model_ab.json"), "w"), indent=1)
print("execution", {k: ex[k] for k in ex if k in ("draft_depth", "rows", "prefill_capacity")})
EOF
  cp "$pkg/model.json" "$out/model_$v.json"; sha256sum "$pkg"/program/*.loom > "$out/programs_$v.sha256"
done

# the benchmark scripts' chat hooks (bench_common.py: chat_probe) come from the job's kit
[ "$kit" != - ] && export LOOM_KIT=${KIT[${variants%% *}]}

serve_alone() { # serve_alone VARIANT
  for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
  "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
  step "serve $1"; "$SV/serve.sh" "perf-$1" "$PORT" amdgpu "$mem" "$D/$1/pkg/model.json" | tail -2
  curl -sf "$url/v1/models" >/dev/null
}
LSPEC=""
llama_up() {   # llama_up [N_MAX]: llama.cpp with the experiment's llama_spec (n-max overridden), restarted on change
  for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
  local spec; spec=$(q "print((e.get('llama_spec') or '').replace('{drafter}', '$drafter'))")
  [ -n "${1:-}" ] && spec=$(echo "$spec" | sed -E "s/--spec-draft-n-max [0-9]+/--spec-draft-n-max $1/")
  [ "${1:-}" = 0 ] && spec=""   # n-max 0: no drafting (reference)
  [ "$spec" = "$LSPEC" ] && curl -sf "$lurl/health" >/dev/null 2>&1 && return 0
  "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
  LLAMACPP_DIR=${LLAMACPP_DIR:-$("$B/fetch_llamacpp.sh" "$WORK/llamacpp" "$tag" | tail -1)}; export LLAMACPP_DIR LLAMA_CACHE_BUILD=$tag
  export LLAMA_SPEC=$spec; LSPEC=$spec
  step "llama.cpp $tag: $LLAMA_SPEC"
  local n; n=$(echo "$spec" | grep -oE "n-max [0-9]+" | grep -oE "[0-9]+")
  LLAMACPP_ARGS_FILE=$out/llamacpp_args_n${n:-0}.txt "$B/llamacpp_server.sh" "$gguf" "$LLAMA_PORT" 4 65536 unified | tail -1
}
TH=$(q "print(j.get('thinking') or 'off')")   # chat_template_kwargs.enable_thinking sent to both engines
echo "thinking: $TH (sent to both engines)"
declare -A res
for s in $steps; do
  IFS=: read -r kind p1 p2 p3 <<< "$s"
  [ "${p2:-}" = - ] && p2=""
  case $kind in
    ab)
      for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
      "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
      ms=(); ids=(); for v in $variants; do ms+=("$D/$v/pkg/model_ab.json"); ids+=("$cardid-$v"); done
      nv=${#ms[@]}; abmem=$(( 11000000000 * nv + 4000000000 ))
      step "ab: one loom-serve with $nv packages ($abmem bytes)"
      "$SV/serve.sh" abserve "$PORT" amdgpu "$abmem" "${ms[@]}" | tail -2
      "$T" "$here/.github/scripts/ab_decode.py" "$url" "${ids[@]}" --rounds "$p1" --think "${p2:-$TH}" --log "$out/ab.jsonl" 2>&1 | tee "$out/ab.log"
      res[$s]=$(grep -c SUMMARY "$out/ab.log")" variants measured"
      "$SV/stop.sh" abserve >/dev/null 2>&1;;
    perf)
      serve_alone "$p1" || { res[$s]=ERROR; continue; }
      step "perf $p1 x $p2 ${p3:-} (thinking $TH)"
      for i in $(seq 1 "$p2"); do
        (cd "$out" && "$T" "$B/perf.py" "$url" "$p1" --model "$cardid" --no-campaign --thinking "$TH" --log "perf_$p1.jsonl" ${p3:+--long} | tail -1)
      done
      "$T" "$here/.github/scripts/ab_decode.py" "$url" "$cardid" --rounds 2 --think "$TH" --log "$out/tps_$p1.jsonl" | grep SUMMARY
      res[$s]=$("$T" -c "
import json,statistics as st; r=[json.loads(l) for l in open('$out/perf_$p1.jsonl')]
k='decode_tps_32k' if '${p3:-}' else 'decode_tps'
print('$p1', k, [x.get(k) for x in r if x.get(k)], 'mean', round(st.mean(x[k] for x in r if x.get(k)), 2))");;
    c4)
      serve_alone "$p1" || { res[$s]=ERROR; continue; }
      step "c4 $p1"
      for i in 1 2 3; do (cd "$out" && "$T" "$B/c4_steady.py" "$url" 4 --model "$cardid" --thinking "$TH" | tail -1 | tee -a "c4_$p1.txt"); done
      res[$s]=$("$T" -c "import json,statistics as st; print('$p1 c4 median', st.median(json.loads(l)['agg_tps'] for l in open('$out/c4_$p1.txt')))");;
    llama)   # llama:N[:long|-][:N_MAX]
      llama_up "${p3:-}" || { res[$s]=ERROR; continue; }
      nm=$(echo "$LSPEC" | grep -oE "n-max [0-9]+" | grep -oE "[0-9]+"); nm=${nm:-0}; lf=perf_llamacpp_n$nm.jsonl
      step "llama.cpp n-max $nm perf x $p1 ${p2:-} (thinking $TH)"
      for i in $(seq 1 "$p1"); do
        (cd "$out" && "$T" "$B/perf.py" "$lurl" "llamacpp-n$nm" --kind llama --no-campaign --thinking "$TH" --log "$lf" ${p2:+--long} | tail -1)
      done
      "$T" "$here/.github/scripts/llama_decode.py" "$lurl" 2 --think "$TH" --log "$out/tps_llamacpp_n$nm.jsonl"
      res[$s]=$("$T" -c "
import json,statistics as st; r=[json.loads(l) for l in open('$out/$lf')]
k='decode_tps_32k' if '${p2:-}' else 'decode_tps'
print('llamacpp n-max $nm', k, [x.get(k) for x in r if x.get(k)], 'mean', round(st.mean(x[k] for x in r if x.get(k)), 2))");;
    llamac4)   # llamac4[:N_MAX]
      llama_up "${p1:-}" || { res[$s]=ERROR; continue; }
      nm=$(echo "$LSPEC" | grep -oE "n-max [0-9]+" | grep -oE "[0-9]+"); nm=${nm:-0}
      step "llama.cpp n-max $nm c4"
      for i in 1 2 3; do (cd "$out" && "$T" "$B/c4_steady.py" "$lurl" 4 --thinking "$TH" | tail -1 | tee -a "c4_llamacpp_n$nm.txt"); done
      res[$s]=$("$T" -c "import json,statistics as st; print('llamacpp n-max $nm c4 median', st.median(json.loads(l)['agg_tps'] for l in open('$out/c4_llamacpp_n$nm.txt')))");;
    textmatch)   # textmatch:V: first 50 generated tokens of perf.py's prompts, Loom V then llama.cpp (one at a time)
      serve_alone "$p1" || { res[$s]=ERROR; continue; }
      "$T" "$B/text_match.py" save "$url" "$out/text_$p1.json" --model "$cardid" --tokenizer "$card/tokenizer.json" --thinking "$TH" | tail -1
      llama_up || { res[$s]=ERROR; continue; }
      "$T" "$B/text_match.py" compare "$lurl" "$out/text_$p1.json" --tokenizer "$card/tokenizer.json" --thinking "$TH" | tee "$out/textmatch_$p1.log"
      res[$s]=$(grep TEXT_MATCH "$out/textmatch_$p1.log" | cut -c1-200);;
    consistency)
      serve_alone "$p1" || { res[$s]=ERROR; continue; }
      # as gates.sh runs it: --pkg and a llama.cpp reference (no drafting) to judge any divergence as a near-tie
      LLAMACPP_DIR=${LLAMACPP_DIR:-$("$B/fetch_llamacpp.sh" "$WORK/llamacpp" "$tag" | tail -1)}; export LLAMACPP_DIR
      ( unset LLAMA_SPEC; "$B/llamacpp_server.sh" "$gguf" "$LLAMA_PORT" 1 8192 | tail -1 )
      step "consistency $p1 x $p2"; nd=0; np=0; nc=0
      for i in $(seq 1 "$p2"); do
        "$T" "$B/consistency.py" "$url" --model "$cardid" --pkg "$D/$p1/pkg" --llama "$lurl" > "$out/consistency_${p1}_$i.log" 2>&1
        grep -q "NOT DETERMINISTIC" "$out/consistency_${p1}_$i.log" && nd=$((nd + 1))
        grep -q "^PASS$" "$out/consistency_${p1}_$i.log" && np=$((np + 1))
        grep -q "identical when concurrent: \[.*False" "$out/consistency_${p1}_$i.log" && nc=$((nc + 1))
        echo "  $i: $(grep -E "^(PASS|FAIL)$" "$out/consistency_${p1}_$i.log") $(head -1 "$out/consistency_${p1}_$i.log" | cut -c1-200)"
      done
      "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
      res[$s]="PASS $np/$p2, nondeterministic $nd/$p2, concurrent near-tie divergences $nc/$p2";;
    determinism)
      [ -f "$B/determinism.py" ] || { res[$s]="no determinism.py"; continue; }
      serve_alone "$p1" || { res[$s]=ERROR; continue; }
      step "determinism $p1 conc $p2 reps $p3"
      "$T" "$B/determinism.py" "$url" --model "$cardid" --conc "$p2" --reps "$p3" --log "$out/determinism_${p1}_c$p2.jsonl" \
        > "$out/determinism_${p1}_c$p2.log" 2>&1
      tail -3 "$out/determinism_${p1}_c$p2.log"; res[$s]=$(tail -1 "$out/determinism_${p1}_c$p2.log" | cut -c1-120);;
    ktest)
      serve_alone "$p1" >/dev/null 2>&1; for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
      step "kernel tests $p1"; kt=$D/$p1/ktest; rm -rf "$kt"
      ( cd "${KIT[$p1]}" && "$T" make_kernel_tests.py "$D/cfg.json" "$kt" ) > "$out/ktest_make_$p1.log" 2>&1 || { tail "$out/ktest_make_$p1.log"; res[$s]=ERROR; continue; }
      "${KIT[$p1]}/run_kernel_tests.sh" "$D/$p1/pkg" "$kt" > "$out/ktest_$p1.log" 2>&1; rc=$?
      grep -v "^PASS" "$out/ktest_$p1.log" | tail -15
      tot=$(grep -oE "^[0-9]+/[0-9]+ passed" "$out/ktest_$p1.log" | tail -1)
      res[$s]="$([ $rc = 0 ] && [ -n "$tot" ] && [ "${tot%%/*}" = "$(echo "${tot#*/}" | cut -d' ' -f1)" ] && echo PASS || echo FAIL) ($tot; $(grep -c '^SKIP' "$out/ktest_$p1.log") skipped)";;
    gates)
      serve_alone "$p1" || { res[$s]=ERROR; continue; }
      LLAMACPP_DIR=${LLAMACPP_DIR:-$("$B/fetch_llamacpp.sh" "$WORK/llamacpp" "$tag" | tail -1)}; export LLAMACPP_DIR LLAMA_CACHE_BUILD=$tag
      export LOOM_KIT=${KIT[$p1]}
      step "gates $p1 LEVEL=$p2"
      ( unset LLAMA_SPEC; FORCE=1 LEVEL=$p2 "$B/gates.sh" "$D/$p1/pkg" "$url" "$gguf" "$D/$p1/bench" "$LLAMA_PORT" ) > "$out/gates_$p1.out" 2>&1; rc=$?
      mkdir -p "$out/gates_$p1"; cp "$D/$p1"/bench/gates.json "$D/$p1"/bench/gates_*.log "$out/gates_$p1/" 2>/dev/null
      res[$s]=$(grep -h "^GATES" "$out/gates_$p1.out" | tail -1 | cut -c1-300); [ -n "${res[$s]}" ] || res[$s]="rc=$rc";;
    *) echo "unknown step $s"; res[$s]=ERROR;;
  esac
  echo "RESULT $job $s: ${res[$s]}"
done
for s in $steps; do echo "$s	${res[$s]:-}"; done > "$out/results.tsv"
"$T" - "$job" "$out" <<'EOF'
import json, os, sys
name, out = sys.argv[1:]
rows = [l.rstrip("\n").split("\t", 1) for l in open(os.path.join(out, "results.tsv"))]
json.dump({"job": name, "runner": open(os.path.join(out, "runner.txt")).read().strip(), "results": dict(rows)},
          open(os.path.join(out, "summary.json"), "w"), indent=1)
EOF
echo "SUMMARY $job"; cat "$out/results.tsv"
