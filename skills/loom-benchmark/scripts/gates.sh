#!/bin/bash
# Correctness gates for a served package, one verdict file.
# usage: [LOOM_ROOT=~/loom] [LEVEL=fast|quick|full] LLAMACPP_DIR=DIR gates.sh PKG LOOM_URL GGUF BENCH_DIR [LLAMA_PORT=8130]
#   LEVEL=fast  (inner optimization loop, after each kept change, ~1-3 min): golden + compare
#               (20 short prompts + 1k-32k documents) and consistency (concurrent == sequential).
#   LEVEL=quick (after the port): fast + agent tool loop, sampling, needle at half the context
#               (at most 60k) compared with llama.cpp by the near-tie rule.
#   NO_LLAMA=1  no llama.cpp runs this model: reference-free checks only (consistency, agent rendering/
#               reuse/calls, sampling semantics, needle retrieval); report correctness as unverified.
#   LEVEL=full  (default; the final build, once): quick + one Loom-only probe at the full context
#               (proves the whole context is served). QUICK=1 is an alias for LEVEL=quick.
# llama.cpp references are cached in BENCH_DIR/llama_cache (they depend on the GGUF and the request,
# not on the Loom build), so only the first run pays for them. llama-server starts lazily, on the
# first cache miss: a later run that is fully cached never loads it. LLAMA_URL=http://... uses an
# already running llama-server (e.g. the shared one qualify.sh started) instead; it is left running.
# Writes BENCH_DIR/gates.json {"verdict", "level", "pkg_hash", "seconds", "gates": {name: {"pass", "log", "seconds"}}}
# and BENCH_DIR/gates_*.log. Exit 1 when any gate fails. Gates run sequentially (a shared GPU).
set -u
pkg=$(cd "$1" && pwd); url=$2; bench=$(mkdir -p "$4" && cd "$4" && pwd); lport=${5:-8130}
gguf=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")   # absolute: we cd into BENCH_DIR below
S=$(cd "$(dirname "$0")" && pwd); root=${LOOM_ROOT:-$HOME/loom}
bin=$root/.tools/bin; [ -d "$bin" ] || bin=$root/.tools/Scripts; T=$bin/python
level=${LEVEL:-full}; [ "${QUICK:-0}" = 1 ] && level=quick
case $level in fast|quick|full) ;; *) echo "LEVEL must be fast, quick or full"; exit 2;; esac
export LOOM_NO_AUDIT=1 PYTHONDONTWRITEBYTECODE=1 LLAMACPP_STATE=${LLAMACPP_STATE:-$bench/llamacpp-state} LLAMA_CACHE=$bench/llama_cache
ctx=$("$T" -c "import json;print(next(iter(json.load(open('$pkg/model.json'))['presets'].values()))['execution']['context_capacity'])")
model=$("$T" -c "import json;print(json.load(open('$pkg/llm_config.json'))['model_id'])")
# Long goldens that fit the context; the llama.cpp slot covers them (and needle sizes up to 64k).
long=$("$T" -c "print(','.join(str(n) for n in (1000, 2000, 16000, 32000) if n + 2000 <= $ctx))")
slot=$("$T" -c "print(min($ctx, 65536) + 1024)")
cd "$bench"
started_llama=0
if [ "${NO_LLAMA:-0}" = 1 ]; then
  L=""   # Loom-only: no golden/compare; agent, sampling and needle run their reference-free checks
elif [ -n "${LLAMA_URL:-}" ]; then
  L=$LLAMA_URL
else
  L=http://127.0.0.1:$lport
  : "${LLAMACPP_DIR:?set LLAMACPP_DIR to the llama.cpp ROCm build (or LLAMA_URL to a running llama-server)}"
  export LLAMA_START="'$S/llamacpp_server.sh' '$gguf' $lport 1 $slot"
  started_llama=1
fi
declare -A result secs
order=()
# A gate that already PASSED on this exact build (same model.json + program/*.loom) is not re-run:
# the inputs are identical (llama.cpp references are cached too). FORCE=1 re-runs everything.
pkg_hash=$(cat "$pkg/model.json" "$pkg"/program/*.loom 2>/dev/null | sha256sum | cut -c1-12)
passed_before() {  # passed_before NAME [NEEDLE_SIZE]
  [ "${FORCE:-0}" = 1 ] && return 1
  [ -f gates_runs.jsonl ] || return 1
  "$T" - "$pkg_hash" "$1" <<'PY'
import json, sys
h, name = sys.argv[1], sys.argv[2]
ok = any(r.get("pkg_hash") == h and r.get("gates", {}).get(name, {}).get("pass") for r in map(json.loads, open("gates_runs.jsonl")))
sys.exit(0 if ok else 1)
PY
}
run() {  # run NAME CMD...
  local name=$1; shift; local t0=$(date +%s)
  if passed_before "$name"; then
    echo "== $name: PASSED earlier on this exact build (pkg $pkg_hash), not re-run"
    result[$name]=0; secs[$name]=0; order+=("$name"); return
  fi
  echo "== $name"; "$@" > "gates_$name.log" 2>&1; local rc=$?
  tail -4 "gates_$name.log"; result[$name]=$rc; secs[$name]=$(( $(date +%s) - t0 )); order+=("$name")
}
t_start=$(date +%s)
LL=(); [ -n "$L" ] && LL=(--llama "$L")
if [ -n "$L" ]; then
run golden "$T" "$S/golden.py" "$pkg" golden.json --llama "$L" --long "$long"
# A prompt that would fail against the golden is re-recorded once by compare.py (llama.cpp's own
# run-to-run near-tie flips), instead of a full second recording.
run compare "$T" "$S/compare.py" "$pkg" golden.json --llama "$L" --url "$url"
grep -q '"verdict": "PASS"' gates_compare.log || result[compare]=1
fi
run consistency "$T" "$S/consistency.py" "$url" --model "$model" --pkg "$pkg" "${LL[@]}"
grep -q "^PASS$" gates_consistency.log || result[consistency]=1
if [ "$level" != fast ]; then
  run agent "$T" "$S/agent_golden.py" "$pkg" "$url" "${LL[@]}" --out agent_golden.json
  run sampling "$T" "$S/sampling_check.py" "$pkg" "$url" "${LL[@]}"
  # Needle: compared with llama.cpp (near-tie rule) up to the llama.cpp slot; the full-context
  # probe (LEVEL=full) proves the size is served.
  cmp=$(( ctx / 2 < 61440 ? ctx / 2 : 61440 ))
  needle_args=("${LL[@]}" --llama-max $(( slot - 1024 )) --out needle.json)
  if [ "$level" = quick ]; then
    needle_args+=(--sizes $cmp --probes 1)
    run needle "$T" "$S/needle.py" "$pkg" "$url" "${needle_args[@]}"
  else
    # The compared half-context probe passed on this build already (qualify): only the full-context probe is new.
    needle_args=("${LL[@]}" --llama-max $(( slot - 1024 )) --out needle_full.json)
    if passed_before needle; then needle_args+=(--sizes $(( ctx - 2048 )) --probes 1); else needle_args+=(--sizes $cmp $(( ctx - 2048 )) --probes 1 --top-probes 1); fi
    run needle_full "$T" "$S/needle.py" "$pkg" "$url" "${needle_args[@]}"
  fi
fi
if [ $started_llama = 1 ] && [ -f "$LLAMACPP_STATE/llamacpp-$lport.pid" ]; then
  kill "$(cat "$LLAMACPP_STATE/llamacpp-$lport.pid")" 2>/dev/null; rm -f "$LLAMACPP_STATE/llamacpp-$lport.pid"
fi
args=()
for n in "${order[@]}"; do args+=("$n" "${result[$n]}" "${secs[$n]}"); done
"$T" - "$level" "$pkg_hash" $(( $(date +%s) - t_start )) "${args[@]}" <<'EOF'
import json, sys, time
level, pkg_hash, total, rest = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
gates = {rest[i]: {"pass": rest[i + 1] == "0", "log": f"gates_{rest[i]}.log", "seconds": int(rest[i + 2])}
         for i in range(0, len(rest), 3)}
verdict = "PASS" if all(g["pass"] for g in gates.values()) else "FAIL"
out = {"verdict": verdict, "level": level, "pkg_hash": pkg_hash, "seconds": total,
       "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "gates": gates}
json.dump(out, open("gates.json", "w"), indent=1)
with open("gates_runs.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
print("GATES", verdict, f"level={level} {total}s", {n: ("PASS" if g["pass"] else "FAIL") for n, g in gates.items()})
sys.exit(0 if verdict == "PASS" else 1)
EOF
