#!/bin/bash
# Correctness gates for a served package, one verdict file.
# usage: [LOOM_ROOT=~/loom] [LEVEL=fast|quick|full] LLAMACPP_DIR=DIR gates.sh PKG LOOM_URL GGUF BENCH_DIR [LLAMA_PORT=8130]
#   LEVEL=fast  (inner optimization loop, after each kept change, ~1-3 min): golden + compare
#               (20 short prompts + 1k-32k documents) and consistency (concurrent == sequential).
#   LEVEL=quick (after the port): fast + speculative-decoding equivalence (spec_check.py, when the package
#               drafts), agent tool loop, sampling, needle at half the context
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
. "$(dirname "$0")/_lib.sh"
level=${LEVEL:-full}; [ "${QUICK:-0}" = 1 ] && level=quick
case $level in fast|quick|full) ;; *) echo "LEVEL must be fast, quick or full"; exit 2;; esac
export LOOM_NO_AUDIT=1 LLAMA_CACHE=$bench/llama_cache
# Long goldens that fit the context; the llama.cpp slot covers them (and needle sizes up to 64k).
long=$("$T" -c "print(','.join(str(n) for n in (1000, 2000, 16000, 32000) if n + 2000 <= $ctx))")
slot=$("$T" -c "print(min($ctx, 65536) + 1024)")
cd "$bench"
defer=0
# A package with image input: llama-server also loads its projector (MMPROJ), for the vision gate.
MMPROJ=${MMPROJ:-$("$T" -c "import json; print((json.load(open('$pkg/llm_config.json')).get('vision') or {}).get('mmproj', ''))" 2>/dev/null)}; export MMPROJ
if [ "${NO_LLAMA:-0}" = 1 ]; then
  L=""   # Loom-only: no golden/compare; agent, sampling and needle run their reference-free checks
elif [ -n "${LLAMA_URL:-}" ]; then
  L=$LLAMA_URL   # an already running llama-server (both models loaded: only if memory allows)
else
  L=http://127.0.0.1:$lport
  : "${LLAMACPP_DIR:?set LLAMACPP_DIR to the llama.cpp ROCm build (or LLAMA_URL to a running llama-server)}"
  export LLAMA_START="LLAMA_SPEC= '$S/llamacpp_server.sh' '$gguf' $lport 1 $slot"   # loads the projector too when MMPROJ is set
  defer=1   # one model at a time: references missing from the cache are filled with Loom stopped
fi
declare -A result secs done
order=()
# A gate that already PASSED on this exact build (same model.json + program/*.loom) is not re-run:
# the inputs are identical (llama.cpp references are cached too). FORCE=1 re-runs everything.
pkg_hash=$(cat "$pkg/model.json" "$pkg"/program/*.loom 2>/dev/null | sha256sum | cut -c1-12)   # includes the vision programs
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
pending=()
run() {  # run NAME CMD...: a gate whose llama.cpp references were all cached gets its verdict;
         # one with deferred references is pending and runs again after the fill
  local name=$1; shift; local t0=$(date +%s)
  [ -n "${done[$name]:-}" ] && return
  if passed_before "$name"; then
    echo "== $name: PASSED earlier on this exact build (pkg $pkg_hash), not re-run"
    result[$name]=0; secs[$name]=0; order+=("$name"); done[$name]=1; return
  fi
  echo "== $name"; rm -f "pending_$name.jsonl"
  if [ $defer = 1 ]; then LLAMA_DEFER=$bench/pending_$name.jsonl "$@" > "gates_$name.log" 2>&1; else "$@" > "gates_$name.log" 2>&1; fi
  local rc=$?
  secs[$name]=$(( ${secs[$name]:-0} + $(date +%s) - t0 ))
  if [ -s "pending_$name.jsonl" ]; then
    echo "   $(wc -l < "pending_$name.jsonl") llama.cpp references pending"; pending+=("$name"); return
  fi
  tail -4 "gates_$name.log"; result[$name]=$rc; order+=("$name"); done[$name]=1
}
# Speculative decoding must not change a token: record on the drafting server, then serve the same package with
# drafting off (model_target.json, draft_depth 0) on Loom's port, record again, restore, compare (spec_check.py).
spec_gate() {
  "$T" "$S/spec_check.py" record "$url" "$model" spec_draft.jsonl || return 1
  "$T" "$S/spec_check.py" card "$pkg" >/dev/null || return 1
  curl -s "$url/metrics" > spec_metrics.txt
  loom_down; local orig=("${loom_argv[@]}") rc
  loom_argv=("${orig[@]/model.json/model_target.json}"); loom_up || return 1
  "$T" "$S/spec_check.py" record "$url" "$model" spec_target.jsonl; rc=$?
  loom_down; loom_argv=("${orig[@]}"); loom_up || return 1
  [ $rc = 0 ] && "$T" "$S/spec_check.py" compare spec_target.jsonl spec_draft.jsonl --metrics-file spec_metrics.txt --pkg "$pkg" ${L:+--llama "$L"}
}
drafting=$("$T" -c "import json; print(1 if json.load(open('$pkg/llm_config.json')).get('draft') else '')" 2>/dev/null)
t_start=$(date +%s)
LL=(); [ -n "$L" ] && LL=(--llama "$L")
cmp=$(( ctx / 2 < 61440 ? ctx / 2 : 61440 ))
all_gates() {
  if [ -n "$L" ]; then
    run golden "$T" "$S/golden.py" "$pkg" golden.json --llama "$L" --long "$long"
    # A prompt that would fail against the golden is re-recorded once by compare.py (llama.cpp's own
    # run-to-run near-tie flips), instead of a full second recording. Needs the complete golden.
    if [ -n "${done[golden]:-}" ]; then
      run compare "$T" "$S/compare.py" "$pkg" golden.json --llama "$L" --url "$url"
      [ -n "${done[compare]:-}" ] && { grep -q '"verdict": "PASS"' gates_compare.log || result[compare]=1; }
    fi
  fi
  # Image input: prompt tokens + replies vs llama-server --mmproj (MMPROJ above) on vision_golden.py's images
  [ -n "$MMPROJ" ] && [ -n "$L" ] && run vision "$T" "$S/vision_golden.py" "$pkg" --url "$url" --llama "$L"
  run consistency "$T" "$S/consistency.py" "$url" --model "$model" --pkg "$pkg" "${LL[@]}"
  [ -n "${done[consistency]:-}" ] && { grep -q "^PASS$" gates_consistency.log || result[consistency]=1; }
  if [ "$level" != fast ]; then
    [ -n "$drafting" ] && run spec spec_gate
    run agent "$T" "$S/agent_golden.py" "$pkg" "$url" "${LL[@]}" --out agent_golden.json
    run sampling "$T" "$S/sampling_check.py" "$pkg" "$url" "${LL[@]}"
    # Needle: compared with llama.cpp (near-tie rule) up to the llama.cpp slot; the full-context
    # probe (LEVEL=full) proves the size is served.
    if [ "$level" = quick ]; then
      run needle "$T" "$S/needle.py" "$pkg" "$url" "${LL[@]}" --llama-max $(( slot - 1024 )) --out needle.json --sizes $cmp --probes 1
    else
      # The compared half-context probe passed on this build already (qualify): only the full-context probe is new.
      local sizes=(--sizes $cmp $(( ctx - 2048 )) --probes 1 --top-probes 1)
      passed_before needle && sizes=(--sizes $(( ctx - 2048 )) --probes 1)
      run needle_full "$T" "$S/needle.py" "$pkg" "$url" "${LL[@]}" --llama-max $(( slot - 1024 )) --out needle_full.json "${sizes[@]}"
    fi
  fi
}
for round in 1 2 3 4 5; do
  pending=()
  all_gates
  [ ${#pending[@]} = 0 ] && break
  # Re-recordings only become known once the first references exist, so this converges in a few rounds.
  echo "== round $round: filling llama.cpp references for ${pending[*]} with Loom stopped ($(date +%T))"
  loom_down
  files=(); for p in "${pending[@]}"; do files+=("pending_$p.jsonl"); done
  "$T" "$S/llama_fill.py" "$L" "${files[@]}" > llama_fill.log 2>&1; tail -1 llama_fill.log
  rm -f "${files[@]}"
  "$S/llamacpp_server.sh" --stop "$lport"
  loom_up || { echo "FAIL: could not restart Loom after the llama.cpp fill"; exit 1; }
done
for p in "${pending[@]}"; do result[$p]=1; order+=("$p"); echo "== $p: references still pending after 5 rounds (FAIL)"; done
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
