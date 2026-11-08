#!/bin/bash
# One variant of a DevLab experiment (.github/workflows/devlab-experiment.yml): fetch a published loom-fun card,
# optionally rebuild its programs from THIS checkout's starter + reference example with build-time switches,
# optionally override model.json execution settings, serve it, and run the variant's steps.
#
#   [WORK=~/devlab-exp] devlab_experiment.sh EXPERIMENT.json VARIANT_NAME
#
# EXPERIMENT.json:
#   {"card": "loom-fun/<repo>", "memory_bytes": N, "kit": "gemma4" | null, "llamacpp": "b1368",   (a variant may override these)
#    "steps": ["consistency:20", "determinism:1:10", "determinism:4:5", "ktest"],     (default for every variant)
#    "variants": [{"name": "card"},                                   (the card's own programs, as published)
#                 {"name": "x", "build": {"ENV": "value", ...},       (rebuild: build.py with these env vars;
#                  "build_args": ["--draft-depth", "3"],                 build_args are extra build.py flags)
#                  "execution": {"draft_depth": 0},                    (model.json preset overrides)
#                  "config": {"rope_theta": 10000},                     (generator config overrides for a rebuild:
#                                                                         an injected bug the gates must catch)
#                  "steps": [...]}]}
# Steps: consistency:N   consistency.py N times (counts runs that are not deterministic / not consistent)
#        consistency:N:llama  the same with --pkg/--llama (the card's llama.cpp build): the gate's own verdict
#        pairs:R         determinism.py --pairs: each prompt twice in a row (prefix-cache miss, then hit), R reps
#        determinism:C:R determinism.py, R reps of each prompt with C requests in flight
#        admit:N         admission only: a needle prompt of N tokens on a fresh server, admitted or refused (503), cancelled
#        needle:N[:S]    one Loom-only needle probe of N tokens on the running server (read timeout S seconds): status, TTFT
#        status          the server's /v1/status (memory: weights, KV storage, workspace) after a short request, to status.json
#        ktest           the starter's kernel tests (make_kernel_tests.py + run_kernel_tests.sh) on the package
#        gates:LEVEL     gates.sh at LEVEL against the card's llama.cpp build ("llamacpp" in EXPERIMENT.json)
# Writes WORK/<variant>/out: summary.json, *.log. Every server it starts is stopped at exit.
set -uo pipefail
exp=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); name=${2:?usage: devlab_experiment.sh EXPERIMENT.json VARIANT}
here=$(cd "$(dirname "$0")/../.." && pwd); SK=$here/skills
B=$SK/loom-benchmark/scripts; SV=$SK/loom-setup/scripts; ST=$SK/loom-port-llm/starter
WORK=${WORK:-$HOME/devlab-exp}; mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
D=$WORK/$name; out=$D/out; rm -rf "$D/pkg" "$out"; mkdir -p "$out"
export LOOM_SERVE_STATE=$D/state LLAMACPP_STATE=$D/llstate PYTHONDONTWRITEBYTECODE=1
PORT=${LOOM_PORT:-8161}; LLAMA_PORT=${LLAMA_PORT:-8131}; url=http://127.0.0.1:$PORT
t0=$(date +%s); step() { echo "== $* ($(date +%T), +$(( $(date +%s) - t0 ))s)"; }
q() { python3 -c "import json,sys; e=json.load(open('$exp')); v=next(v for v in e['variants'] if v['name']=='$name'); $1"; }
# card, memory_bytes, kit and llamacpp: the variant's own, else the experiment's
f() { q "print(v.get('$1', e.get('$1')) or '-')"; }
repo=$(f card); mem=$(f memory_bytes); kit=$(f kit); tag=$(f llamacpp)
steps=$(q "print(' '.join(v.get('steps') or e.get('steps') or ['consistency:20']))")
echo "variant $name: $(q "print(json.dumps(v))")"

cleanup() {
  for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
  "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
  cp "$LOOM_SERVE_STATE"/*.log "$out/" 2>/dev/null
  echo "devlab_experiment.sh: servers stopped"
}
trap cleanup EXIT

# 1. Loom SDK + tools venv
root=${LOOM_ROOT:-$WORK/loom}
if [ ! -x "$root/.venv/bin/loom-serve" ] || [ ! -x "$root/.tools/bin/python" ]; then
  step "install Loom into $root"; "$SV/install_loom.sh" "$root" ${LOOM_VERSION:+"$LOOM_VERSION"} || exit 1
fi
export LOOM_ROOT=$root VENV=$root/.venv; T=$root/.tools/bin/python

# 2. The card (shared by the variants of one runner: fetched once into WORK/card)
card=$WORK/_card_$(echo "$repo" | tr "/" "_"); base=https://huggingface.co/$repo/resolve/main
if [ ! -f "$card/.fetched" ]; then
  step "fetch $repo"; rm -rf "$card"
  "$root/.venv/bin/loom-fetch" "$base/model.json" --no-input --materialize=copy --cache "$WORK/fetch-cache" --output="$card" || { echo "FAIL: loom-fetch"; exit 1; }
  curl -fsSL --retry 3 "$base/llm_config.json" -o "$card/llm_config.json" || exit 1
  rm -rf "$WORK/fetch-cache"; touch "$card/.fetched"
fi

# build.py's / make_kernel_tests.py's config: the card's llm_config.json with its GGUFs pointed at the fetched files, plus the tensor table
"$T" - "$card" "$D/cfg.json" "$(q "print(json.dumps(v.get('config') or {}))")" <<'EOF' || exit 1
import json, os, sys
import gguf
card, dst, over = sys.argv[1:]
c = json.load(open(os.path.join(card, "llm_config.json")))
par = lambda f: os.path.join(card, "parameter", os.path.basename(f))
c["gguf"] = par(c["gguf"])
for k in ("vision",):
    if c.get(k):
        c[k]["mmproj"] = par(c[k]["mmproj"])
for k in ("drafter", "assistant"):
    if c.get(k) and c[k].get("gguf"):
        c[k]["gguf"] = par(c[k]["gguf"]) if os.path.exists(par(c[k]["gguf"])) else None
r = gguf.GGUFReader(c["gguf"])
c["tensors"] = {t.name: {"type": t.tensor_type.name.lower(), "shape": [int(x) for x in t.shape]} for t in r.tensors}
for k in ("model_id", "draft"):
    c.pop(k, None)
c.update(json.loads(over))
json.dump(c, open(dst, "w"), indent=1)
EOF

# 3. The package: the card's own programs, or a rebuild from this checkout
pkg=$D/pkg
if [ "$(q "print('build' in v)")" = True ]; then
  # "ref": build with the starter/examples of another commit (bisecting a generator change)
  PL=$SK/loom-port-llm; ref=$(q "print(v.get('ref') or '')")
  if [ -n "$ref" ]; then
    git -C "$here" fetch -q origin "$ref" 2>/dev/null || git -C "$here" fetch -q --unshallow 2>/dev/null
    rm -rf "$D/src"; mkdir -p "$D/src"
    git -C "$here" archive "$ref" skills/loom-port-llm | tar -x -C "$D/src" || { echo "FAIL: no $ref"; exit 1; }
    PL=$D/src/skills/loom-port-llm; echo "starter from $ref ($(git -C "$here" log -1 --format=%h:%s "$ref" | cut -c1-80))"
  fi
  if [ "$kit" != - ]; then
    step "kit: starter + $kit example"; rm -rf "$D/kit"; "$PL/reference/apply_example.sh" "$kit" "$D/kit" | tail -1 || exit 1
    K=$D/kit
  else
    K=$PL/starter
  fi
  export LOOM_KIT=$K
  # the card's own build settings (model id, depth, context, pool) unless the variant overrides them
  args=$("$T" -c "
import json; m=json.load(open('$card/model.json')); c=json.load(open('$card/llm_config.json')); ex=m['presets']['chat']['execution']
a=['--model-id', m['model']['id'], '--context', str(ex['context_capacity']), '--pool', str(ex['pool_capacity']), '--prefill', str(ex['prefill_capacity'])]
d=c.get('draft') or {}
a += (['--draft', d['kind'], '--draft-depth', str(d['depth'])] if d.get('kind') else [])
print(' '.join(a))")
  extra=$(q "print(' '.join(v.get('build_args') or []))")
  envs=$(q "print(' '.join(f'{k}={x}' for k, x in (v.get('build') or {}).items()))")
  step "build: env $envs; build.py $args $extra"
  ( cd "$K" && env $envs "$T" build.py "$D/cfg.json" "$pkg" --tokenizer "$card/tokenizer.json" $args $extra ) > "$out/build.log" 2>&1 \
    || { tail -30 "$out/build.log"; echo "FAIL: build"; exit 1; }
  tail -5 "$out/build.log"
  for f in "$card"/program/*.loom; do cmp -s <(tail -n +2 "$f") <(tail -n +2 "$pkg/program/$(basename "$f")") && echo "same as the card: $(basename "$f")" || echo "differs from the card: $(basename "$f")"; done
else
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
  if [ "$kit" != - ]; then rm -rf "$D/kit"; "$SK/loom-port-llm/reference/apply_example.sh" "$kit" "$D/kit" | tail -1; export LOOM_KIT=$D/kit; fi
fi
"$T" - "$pkg/model.json" "$(q "print(json.dumps(v.get('execution') or {}))")" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1])); ex = m["presets"]["chat"]["execution"]; ex.update(json.loads(sys.argv[2]))
json.dump(m, open(sys.argv[1], "w"), indent=1); print("execution", ex)
EOF
cp "$pkg/model.json" "$out/model.json"; sha256sum "$pkg"/program/*.loom > "$out/programs.sha256"
model=$("$T" -c "import json; print(json.load(open('$pkg/model.json'))['model']['id'])")

llama_up() {   # the card's llama.cpp build on the card's GGUF (reference for near-tie diagnosis)
  [ "$tag" != - ] || { echo "EXPERIMENT.json has no llamacpp tag"; return 1; }
  curl -sf "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null && return 0
  LLAMACPP_DIR=${LLAMACPP_DIR:-$("$B/fetch_llamacpp.sh" "$WORK/llamacpp" "$tag" | tail -1)}; export LLAMACPP_DIR LLAMA_CACHE_BUILD=$tag
  "$B/llamacpp_server.sh" "$("$T" -c "import json; print(json.load(open('$pkg/llm_config.json'))['gguf'])")" "$LLAMA_PORT" 1 8192 | tail -1
}
serve() { "$SV/serve.sh" "exp-$name" "$PORT" amdgpu "$mem" "$pkg/model.json" | tail -2; curl -sf "$url/v1/models" >/dev/null; }
declare -A res
for s in $steps; do
  IFS=: read -r kind p1 p2 <<< "$s"
  case $kind in
    ktest)
      step "kernel tests"
      kt=$D/ktest; rm -rf "$kt"
      ( cd "${LOOM_KIT:-$ST}" && "$T" make_kernel_tests.py "$D/cfg.json" "$kt" ) > "$out/ktest_make.log" 2>&1 || { tail "$out/ktest_make.log"; res[$s]=ERROR; continue; }
      "${LOOM_KIT:-$ST}/run_kernel_tests.sh" "$pkg" "$kt" > "$out/ktest.log" 2>&1; rc=$?
      grep -v "^PASS" "$out/ktest.log" | tail -15; echo "PASS lines: $(grep -c '^PASS' "$out/ktest.log")"
      res[$s]=$([ $rc = 0 ] && echo PASS || echo FAIL);;
    consistency)
      curl -sf "$url/v1/models" >/dev/null || { step serve; serve || { res[$s]=ERROR; continue; }; }
      LLA=()
      if [ "$p2" = llama ]; then   # with --pkg/--llama: the gate's own verdict (near-tie rule for concurrent divergences)
        llama_up || { res[$s]=ERROR; continue; }; LLA=(--pkg "$pkg" --llama "http://127.0.0.1:$LLAMA_PORT")
      fi
      step "consistency x $p1 ${p2:-}"; nd=0; nc=0; np=0
      for i in $(seq 1 "$p1"); do
        "$T" "$B/consistency.py" "$url" --model "$model" "${LLA[@]}" > "$out/consistency_$i.log" 2>&1
        grep -q "^PASS$" "$out/consistency_$i.log" && np=$((np + 1))
        grep -q "NOT DETERMINISTIC" "$out/consistency_$i.log" && nd=$((nd + 1))
        grep -q "identical when concurrent: \[.*False" "$out/consistency_$i.log" && nc=$((nc + 1))
        echo "  $i: $(head -1 "$out/consistency_$i.log" | cut -c40-400)"
      done
      res[$s]="gate PASS $np/$p1, nondeterministic $nd/$p1, concurrent-mismatch $nc/$p1"
      grep -h "first divergence\|near-tie" "$out"/consistency_*.log | sort | uniq -c | sort -rn | head -8;;
    determinism)
      curl -sf "$url/v1/models" >/dev/null || { step serve; serve || { res[$s]=ERROR; continue; }; }
      step "determinism conc $p1 reps $p2"
      "$T" "$B/determinism.py" "$url" --model "$model" --conc "$p1" --reps "$p2" --log "$out/determinism_c$p1.jsonl" \
        > "$out/determinism_c$p1.log" 2>&1
      cut -c1-400 "$out/determinism_c$p1.log"; res[$s]=$(tail -1 "$out/determinism_c$p1.log" | cut -c1-120);;
    pairs)
      curl -sf "$url/v1/models" >/dev/null || { step serve; serve || { res[$s]=ERROR; continue; }; }
      step "prefix-cache pairs reps $p1"
      "$T" "$B/determinism.py" "$url" --model "$model" --pairs --reps "$p1" --log "$out/pairs.jsonl" > "$out/pairs.log" 2>&1
      cut -c1-400 "$out/pairs.log"; res[$s]=$(grep -c "a path gave two outputs" "$out/pairs.log")" prompts with a nondeterministic path, "$(grep -c "paths differ" "$out/pairs.log")" with hit != miss";;
    admit)
      # Admission only: a needle prompt of N tokens on a fresh server; admitted (it runs) or refused (503), then cancelled
      [ -f "$LOOM_SERVE_STATE/exp-$name.pid" ] && "$SV/stop.sh" "exp-$name" >/dev/null 2>&1
      step "serve (fresh)"; serve || { res[$s]=ERROR; continue; }
      step "admit $p1"
      res[$s]=$(cd "$B" && "$T" - "$pkg" "$url" "$model" "$p1" <<'EOF' 2> "$out/admit_$p1.err"
import sys, threading, time
import requests
from bench_common import Package
from needle import corpus, haystack
pkg, url, model, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
P = Package(pkg); ids = P.ids(corpus())
while len(ids) < n:
    ids = ids + ids
prompt, _ = haystack(P, ids, n, 1)
got = {}
def post():
    try:
        with requests.post(url + "/v1/chat/completions", stream=True, timeout=600, json={
                "model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1024, "stream": True}) as r:
            got["status"] = r.status_code; got["body"] = r.text[:200] if r.status_code != 200 else ""
    except Exception as e:
        got["error"] = str(e)[:200]
threading.Thread(target=post, daemon=True).start()
for _ in range(120):
    time.sleep(1)
    st = requests.get(url + "/v1/status").json()["data"][0]["loom"]["status"]
    if "status" in got or st["active_requests"]:
        break
print("admitted" if st["active_requests"] else f"refused {got}")
EOF
);;
    gates)
      curl -sf "$url/v1/models" >/dev/null || { step serve; serve || { res[$s]=ERROR; continue; }; }
      [ "$tag" != - ] || { echo "gates: EXPERIMENT.json has no llamacpp tag"; res[$s]=ERROR; continue; }
      "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
      LLAMACPP_DIR=${LLAMACPP_DIR:-$("$B/fetch_llamacpp.sh" "$WORK/llamacpp" "$tag" | tail -1)}; export LLAMACPP_DIR LLAMA_CACHE_BUILD=$tag
      step "gates LEVEL=$p1"
      gguf=$("$T" -c "import json; print(json.load(open('$pkg/llm_config.json'))['gguf'])")
      FORCE=1 LEVEL=$p1 "$B/gates.sh" "$pkg" "$url" "$gguf" "$D/bench" "$LLAMA_PORT" > "$out/gates.out" 2>&1; rc=$?
      cp "$D"/bench/gates.json "$D"/bench/gates_*.log "$out/" 2>/dev/null
      res[$s]=$(grep -h "^GATES" "$out/gates.out" | tail -1 | cut -c1-300); [ -n "${res[$s]}" ] || res[$s]="rc=$rc";;
    status)
      curl -sf "$url/v1/models" >/dev/null || { step serve; serve || { res[$s]=ERROR; continue; }; }
      curl -sf "$url/v1/chat/completions" -H "Content-Type: application/json" -d "{\"model\": \"$model\", \"messages\": [{\"role\": \"user\", \"content\": \"Say OK.\"}], \"max_tokens\": 8}" >/dev/null
      curl -sf "$url/v1/status" > "$out/status.json"; res[$s]=$(head -c 1500 "$out/status.json");;
    needle)
      curl -sf "$url/v1/models" >/dev/null || { step serve; serve || { res[$s]=ERROR; continue; }; }
      step "needle $p1"
      "$T" "$B/needle.py" "$pkg" "$url" --sizes "$p1" --probes 1 --no-over-capacity ${p2:+--timeout "$p2"} \
        --out "$out/needle_$p1.json" > "$out/needle_$p1.log" 2>&1
      res[$s]=$(grep -h "^OK\|^FAIL" "$out/needle_$p1.log" | tail -1 | cut -c1-240)
      [ -n "${res[$s]}" ] || res[$s]=$(tail -1 "$out/needle_$p1.log" | cut -c1-240);;
    *) echo "unknown step $s"; res[$s]=ERROR;;
  esac
  echo "RESULT $name $s: ${res[$s]}"
done
for s in $steps; do echo "$s	${res[$s]:-}"; done > "$out/results.tsv"
"$T" - "$name" "$out" <<'EOF'
import json, os, sys
name, out = sys.argv[1:]
rows = [l.rstrip("\n").split("\t", 1) for l in open(os.path.join(out, "results.tsv"))]
json.dump({"variant": name, "results": dict(rows)}, open(os.path.join(out, "summary.json"), "w"), indent=1)
EOF
echo "SUMMARY $name"; cat "$out/results.tsv"
