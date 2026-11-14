#!/bin/bash
# Correctness gates for a served package, one verdict file.
# usage: [LOOM_ROOT=~/loom] [LEVEL=dev|fast|quick|full] LLAMACPP_DIR=DIR gates.sh PKG LOOM_URL GGUF BENCH_DIR [LLAMA_PORT=8130]
#   LEVEL=dev   (the development loop: after each kept change in loom-optimize / loom-improve, a few minutes
#               once the llama.cpp references are cached): the gates that caught real bugs, at full
#               strength, plus a light touch of every other feature the package has:
#                 consistency (concurrent == alone, run-to-run determinism), agent tool loop (thinking on,
#                 and off when the template toggles it), golden + compare (12 short prompts + 1k/2k documents),
#                 vision (2 images, with a projector), spec (5-6 short requests drafted vs undrafted, when the
#                 package drafts), sampling (temperature 0 == greedy, seed reproduces; no 200-draw
#                 distribution).
#               Audit of the 26-29 sessions: consistency 3 real bugs, agent 3-4, compare 2 (first bring-up),
#               spec 0-1; vision and sampling 0. quick/full run nightly
#               on the published cards (.github/workflows/nightly-gates.yml).
#   LEVEL=fast  golden + compare (20 short prompts + 1k-32k documents) and consistency (concurrent == sequential).
#   LEVEL=quick (nightly CI; once before a new port's first card): fast + speculative-decoding equivalence (spec_check.py, when the package
#               drafts), agent tool loop (thinking on, and off when the template toggles it, as at dev), sampling, and the context admission
#               probe (below).
#   NO_LLAMA=1  no llama.cpp runs this model: reference-free checks only (consistency, agent rendering/
#               reuse/calls, sampling semantics); report correctness as unverified.
#   LEVEL=full  (default; the final build, once): quick. QUICK=1 is an alias for LEVEL=quick.
# Context admission (quick and full; a check, not a gate: no gate row, no card row): ctx_admit.py sends one request of
# ~the package's native context_capacity and cancels it once the server has accepted it (seconds, no long prefill). A
# refusal (4xx/503: a KV pool or memory budget below the context) fails the run: gates.json "context_admit"
# {"admitted", "tokens", "context", "detail"} (make_info.py ignores it), a "CONTEXT_ADMIT FAIL ..." line precedes the GATES line, which says FAIL (exit 1).
# llama.cpp references are cached in BENCH_DIR/llama_cache (they depend on the GGUF and the request,
# not on the Loom build), so only the first run pays for them. llama-server starts lazily, on the
# first cache miss: a later run that is fully cached never loads it. LLAMA_URL=http://... uses an
# already running llama-server (e.g. the shared one qualify.sh started) instead; it is left running.
# Writes BENCH_DIR/gates.json {"verdict", "level", "pkg_hash", "seconds", "gates": {name: {"pass", "log", "seconds"}}}
# and BENCH_DIR/gates_*.log. Exit 1 when any gate fails. Gates run sequentially (a shared GPU).
set -u
. "$(dirname "$0")/_lib.sh"
level=${LEVEL:-full}; [ "${QUICK:-0}" = 1 ] && level=quick
case $level in dev|fast|quick|full) ;; *) echo "LEVEL must be dev, fast, quick or full"; exit 2;; esac
export LOOM_NO_AUDIT=1 LLAMA_CACHE=$bench/llama_cache
# Long goldens that fit the context; the llama.cpp slot covers them.
long=$("$T" -c "print(','.join(str(n) for n in (1000, 2000, 16000, 32000) if n + 2000 <= $ctx))")
slot=$("$T" -c "print(min($ctx, 65536) + 1024)")
cd "$bench"
defer=0
# A package with image input: llama-server also loads its projector (MMPROJ), for the vision gate.
MMPROJ=${MMPROJ:-$("$T" -c "import json; print((json.load(open('$pkg/llm_config.json')).get('vision') or {}).get('mmproj', ''))" 2>/dev/null)}; export MMPROJ
if [ "${NO_LLAMA:-0}" = 1 ]; then
  L=""   # Loom-only: no golden/compare; agent and sampling run their reference-free checks
elif [ -n "${LLAMA_URL:-}" ]; then
  L=$LLAMA_URL   # an already running llama-server (both models loaded: only if memory allows)
else
  L=http://127.0.0.1:$lport
  : "${LLAMACPP_DIR:?set LLAMACPP_DIR to the llama.cpp ROCm build (or LLAMA_URL to a running llama-server)}"
  export LLAMA_START="LLAMA_SPEC=\"\${LLAMA_FILL_SPEC:-}\" '$S/llamacpp_server.sh' '$gguf' $lport 1 $slot"   # loads the projector too when MMPROJ is set
  export LLAMA_STOP="'$S/llamacpp_server.sh' --stop $lport"   # llama_fill.py restarts it for a request that needs other arguments
  defer=1   # one model at a time: references missing from the cache are filled with Loom stopped
fi
declare -A result secs done
order=()
# A gate that already PASSED on this exact build (same model.json + program/*.loom) is not re-run:
# the inputs are identical (llama.cpp references are cached too). FORCE=1 re-runs everything.
# A LEVEL=dev pass counts only for LEVEL=dev (its gates are lighter versions under the same names).
pkg_hash=$(cat "$pkg/model.json" "$pkg"/program/*.loom 2>/dev/null | sha256sum | cut -c1-12)   # includes the vision programs
passed_before() {  # passed_before NAME
  [ "${FORCE:-0}" = 1 ] && return 1
  [ -f gates_runs.jsonl ] || return 1
  "$T" - "$pkg_hash" "$1" "$level" <<'PY'
import json, sys
h, name, level = sys.argv[1], sys.argv[2], sys.argv[3]
ok = any(r.get("pkg_hash") == h and r.get("gates", {}).get(name, {}).get("pass") and (level == "dev" or r.get("level") != "dev")
         for r in map(json.loads, open("gates_runs.jsonl")))
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
  local lean=(); [ "$level" = dev ] && lean=(--lean)
  "$T" "$S/spec_check.py" record "$url" "$model" spec_draft.jsonl "${lean[@]}" || return 1
  "$T" "$S/spec_check.py" card "$pkg" >/dev/null || return 1
  curl -s "$url/metrics" > spec_metrics.txt
  loom_down; local orig=("${loom_argv[@]}") rc
  loom_argv=("${orig[@]/model.json/model_target.json}"); loom_up || return 1
  "$T" "$S/spec_check.py" record "$url" "$model" spec_target.jsonl "${lean[@]}"; rc=$?
  loom_down; loom_argv=("${orig[@]}"); loom_up || return 1
  [ $rc = 0 ] && "$T" "$S/spec_check.py" compare spec_target.jsonl spec_draft.jsonl --metrics-file spec_metrics.txt --pkg "$pkg" ${L:+--llama "$L"}
}
drafting=$("$T" -c "import json; print(1 if json.load(open('$pkg/llm_config.json')).get('draft') else '')") || exit 2
toggle=$("$T" -c "import json; print(1 if json.load(open('$pkg/llm_config.json')).get('chat', {}).get('thinking') == 'toggle' else '')") || exit 2
# A chat-only block (a template that ignores tools) has no tool loop to gate: the card's test (card_lib.tool_calling)
# decides, so the card shows the agent gates as not applicable rather than missing.
tools=$("$T" -c "import json, sys; sys.path.insert(0, '$S/card'); import card_lib; print(1 if card_lib.tool_calling(json.load(open('$pkg/llm_config.json')).get('chat') or {}) else '')") || exit 2
[ -n "$tools" ] || echo "== agent: skipped, tool calling unsupported by the chat template (not applicable)"
t_start=$(date +%s)
LL=(); [ -n "$L" ] && LL=(--llama "$L")
# Tool calling + history reuse (every level but fast): reasoning on, and off when the template toggles it (the card
# lists agent_think_off for such packages, card_lib.gate_applies). Nothing without tool calling (not applicable).
agent_gates() {
  [ -n "$tools" ] || return 0
  if [ -n "$toggle" ]; then
    run agent "$T" "$S/agent_golden.py" "$pkg" "$url" "${LL[@]}" --thinking on --out agent_golden.json
    run agent_think_off "$T" "$S/agent_golden.py" "$pkg" "$url" "${LL[@]}" --thinking off --out agent_think_off.json
  else
    run agent "$T" "$S/agent_golden.py" "$pkg" "$url" "${LL[@]}" --out agent_golden.json
  fi
}
# LEVEL=dev: the strongest gates in full, every other feature lightly; cheapest and strongest first.
dev_gates() {
  run consistency "$T" "$S/consistency.py" "$url" --model "$model" --pkg "$pkg" "${LL[@]}"
  [ -n "${done[consistency]:-}" ] && { grep -q "^PASS$" gates_consistency.log || result[consistency]=1; }
  agent_gates
  if [ -n "$L" ]; then
    run golden "$T" "$S/golden.py" "$pkg" golden_dev.json --llama "$L" --short 12 --long "$("$T" -c "print(','.join(str(n) for n in (1000, 2000) if n + 2000 <= $ctx))")"
    if [ -n "${done[golden]:-}" ]; then
      run compare "$T" "$S/compare.py" "$pkg" golden_dev.json --llama "$L" --url "$url" --out compare_dev.json
      [ -n "${done[compare]:-}" ] && { grep -q '"verdict": "PASS"' gates_compare.log || result[compare]=1; }
    fi
    [ -n "$MMPROJ" ] && run vision "$T" "$S/vision_golden.py" "$pkg" --url "$url" --llama "$L" --cases shapes,wide
  fi
  run sampling "$T" "$S/sampling_check.py" "$pkg" "$url" "${LL[@]}" --draws 0
  # Last: it restarts Loom twice (drafting off, then on again).
  [ -n "$drafting" ] && run spec spec_gate
}
all_gates() {
  # VISION_ONLY=1 [VISION_ARGS="--cases wide,wide_pre"]: just the vision gate (DevLab diagnostics for one failing case)
  if [ -n "${VISION_ONLY:-}" ]; then
    [ -n "$MMPROJ" ] && [ -n "$L" ] && run vision "$T" "$S/vision_golden.py" "$pkg" --url "$url" --llama "$L" ${VISION_ARGS:-}
    return
  fi
  [ "$level" = dev ] && { dev_gates; return; }
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
    agent_gates
    run sampling "$T" "$S/sampling_check.py" "$pkg" "$url" "${LL[@]}"
  fi
}
# Loom runs a gate kept across one invocation's deferred-reference rounds (consistency.py) never outlive it
rm -f "$bench"/pending_*.jsonl.loom.json
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
# Context admission probe (quick, full): the server is up here; a check, not a gate (see the header).
rm -f context_admit.json
if [ "$level" = quick ] || [ "$level" = full ]; then
  echo "== context admission"
  "$T" "$S/ctx_admit.py" "$pkg" "$url" --out context_admit.json > gates_context_admit.log 2>&1 || true
  [ -s context_admit.json ] || "$T" -c "import json; json.dump({'admitted': False, 'tokens': None, 'context': $ctx, 'detail': 'ctx_admit.py crashed (see gates_context_admit.log)'}, open('context_admit.json', 'w'))"
  tail -2 gates_context_admit.log
fi
args=()
for n in "${order[@]}"; do args+=("$n" "${result[$n]}" "${secs[$n]}"); done
"$T" - "$level" "$pkg_hash" $(( $(date +%s) - t_start )) "${args[@]}" <<'EOF'
import json, os, sys, time
level, pkg_hash, total, rest = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
gates = {rest[i]: {"pass": rest[i + 1] == "0", "log": f"gates_{rest[i]}.log", "seconds": int(rest[i + 2])}
         for i in range(0, len(rest), 3)}
admit = json.load(open("context_admit.json")) if os.path.exists("context_admit.json") else None
verdict = "PASS" if all(g["pass"] for g in gates.values()) and (admit is None or admit["admitted"]) else "FAIL"
out = {"verdict": verdict, "level": level, "pkg_hash": pkg_hash, "seconds": total,
       "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "gates": gates}
if admit is not None:   # a check, not a gate: not in "gates" (make_info.py and the card ignore it)
    out["context_admit"] = admit
json.dump(out, open("gates.json", "w"), indent=1)
with open("gates_runs.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
if admit is not None:
    print("CONTEXT_ADMIT", "PASS" if admit["admitted"] else "FAIL", admit["detail"])
print("GATES", verdict, f"level={level} {total}s", {n: ("PASS" if g["pass"] else "FAIL") for n, g in gates.items()})
sys.exit(0 if verdict == "PASS" else 1)
EOF
