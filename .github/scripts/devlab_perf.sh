#!/bin/bash
# One job of a DevLab performance experiment (.github/workflows/devlab-perf.yml): fetch a published loom-fun card,
# build package variants from this checkout (or another ref), and compare them with each other and with llama.cpp
# ON THIS RUNNER (runners differ from each other and from a dev machine: only same-job numbers are compared).
# This is the single source of card performance: the card variant (active wait) and llama.cpp on the same runner,
# ingested with make_info.py perf --source-kind devlab (loom-org SKILL.md step 3).
#
#   [WORK=~/devlab-perf] devlab_perf.sh EXPERIMENT.json JOB_NAME
#
# EXPERIMENT.json:
#   {"card": "loom-fun/<repo>", "memory_bytes": N, "kit": "gemma4" | null, "llamacpp": "b1368",
#    "llama_spec": "matched" (default),   the llama.cpp servers' drafting = the card's own (see below); or an explicit override
#                  "--spec-type draft-dflash --spec-draft-model {drafter} ...",   ({drafter}: the card's drafter GGUF;
#                                                                     {hf:OWNER/REPO/FILE}: that Hub file, e.g. a drafter llama.cpp loads and the card doesn't ship)
#    "jobs": [{"name": "j", "variants": [V, ...], "steps": [S, ...]}]}
#   BENCHMARK RULE (skills/loom-benchmark/reference/methodology.md): one drafting choice per model, identical on Loom and llama.cpp (ROCm, Vulkan), from
#   data/drafting_classification.json (MTP head in the GGUF > exact-model DFlash/assistant drafter > none; never ngram): "matched"
#   (the default) = that choice as llama.cpp arguments (card/matched_spec.py); Loom keeps its tuned depth, llama.cpp the drafter's
#   recommended n-max. The 1-stream rows load each engine once with it (llama@rocm.np1 / llama@vulkan.np1); streams4 loads both
#   without drafting (llamac4@rocm.nodraft / llamac4@vulkan.perslot.nodraft; Loom: variant {"name": "nodraft", "execution":
#   {"draft_depth": 0}}, step c4:nodraft). A server that cannot run the drafter fails its step = "not supported", never replaced.
#   n-max sweep (DIAGNOSTIC, not a card input: the card uses the documented n-max or the default): llama@rocm.np1:3:-:N /
#   llama@vulkan.np1:3:-:N (and the :long form), N in 1 2 3 4 d (d = llama.cpp default); card/pick_nmax.py tabulates it.
#   "llama_spec_vulkan": a Vulkan-only override, for a drafter the Vulkan build cannot run (record it with --differs).
#   A job may override card, kit, memory_bytes, llamacpp, llamacpp_vulkan, vulkan_kv and llama_spec: one experiment can
#   measure several cards.
#   "llamacpp_vulkan": "b12211": the official ggml-org Vulkan release (sha256-pinned in skills/loom-benchmark/data) that the
#   steps llama@vulkan, llamac4@vulkan and the other llama.cpp steps with @vulkan run, with the SAME llama_spec and workloads as
#   the ROCm "llamacpp" build (fetch_llamacpp.sh / llamacpp_server.sh LLAMACPP_BACKEND=vulkan). A server that cannot run a
#   setting fails its step (RESULT ... ERROR, llamacpp_start-vulkan.log): recorded as a difference, never replaced silently.
#   "vulkan_kv": "per-slot" (4 slots of 36864 tokens, no -kvu), "unified" (4 slots share one 65536-token pool, -kvu, as the ROCm
#   baseline) or "auto" (default: per-slot, except unified when the server drafts with DFlash). b12211's Vulkan backend has no
#   fused flash attention over K/V rows (the server logs "assigned to device CPU"), so with -kvu it attends the whole pool with a
#   dense mask: decode_32k and 4 streams lose 15-20 %; per-slot caches avoid it, but b12211 aborts a DFlash server with 4
#   concurrent streams on them (get_sub_batch: failed to add token). Every server argument is recorded: ingest the difference
#   with make_info.py perf --differs.
#   out/ also holds llamacpp_args-vulkan.txt (the exact server arguments), llamacpp_release-vulkan.sha256 and
#   programs_sha256.txt (the card's programs this job fetched: make_info.py perf --fetched-programs).
# Variant V: {"name": "card"}                                    the card's own programs, as published
#            {"name": "fun", "card": "loom-fun/<repo>"}              another card's published programs next to this job's card (a Strix
#                                                                    A/B of the staging package against the loom-fun one: ab / perf / textmatch)
#            {"name": "x", "build": {"ENV": "1"}, "build_args": [...], "ref": "origin/main",
#             "execution": {"draft_depth": 3}, "double": ["llm_verify:g_head_mv", "*:g_mv_q8_0"]}
#            "context": N: the served context of a reduced-context model on the R9700 (execution context_capacity / pool_capacity N;
#            a rebuild also gets --context N --pool N); rows beyond it are refused by the server (HTTP 400) and skipped by perf.py.
#            config: generator-config overrides for a rebuild (e.g. {"layers": 12}: timing only)
#            build: build.py with these env vars (and extra flags); ref: build with that commit's starter;
#            execution: model.json overrides; double: ablation (each kernel.launch in PROGRAM matching REGEX twice).
#            {"name": "parked", "active_wait": false}: the card's programs served with the SDK's default completion wait
#            (LOOM_ACTIVE_WAIT=0 in serve.sh) instead of the active wait every other variant gets.
# Job "thinking": off (default) | on: chat_template_kwargs.enable_thinking sent to BOTH engines in every step.
# Steps S: ab:R[:on|off]        every variant in ONE loom-serve (small context), ab_decode.py R interleaved rounds
#          perf:V:N[:long]      V served alone with its own settings: perf.py N times (+ --long)
#          c4:V[:N]             c4_steady.py N (default 4) streams on V
#          llama:N[:long|-][:M] llama.cpp (card build + llama_spec, n-max M) perf.py N times, llama_decode.py tokens/step
#          llamac4[:M]          c4_steady.py on llama.cpp
#          (llama[@vulkan|@rocm], llamac4[@vulkan|@rocm]: the same with the Vulkan build; files perf_llamacpp-vulkan_nN.jsonl. Modifiers
#          after @backend pick the server configuration, e.g. llama@vulkan.np1:2 : .np1 one slot (Vulkan 36864 tokens, ROCm 65536;
#          no 4-stream step), .perslot (Vulkan) 4 slots of 36864; files perf_llamacpp-vulkan.np1_nN.jsonl)
#          textmatch:V          text_match.py: first 50 tokens of perf.py's prompts on V, then on llama.cpp
#          consistency:V:N      consistency.py N times (NOT DETERMINISTIC / concurrent mismatch counts)
#          determinism:V:C:R    determinism.py (issue #39's script, when present)
#          func:V               V served alone: the functional checks the card claims (chat, tool calling, image input;
#                               r9700_func.py: the same checks as the Windows red-team harness), out/func_V.json
#          ktest:V              make_kernel_tests.py + run_kernel_tests.sh with V's kit
#          gates:V:LEVEL        gates.sh against llama.cpp
# Runner: "runner": "strix" (default) | "r9700" on the experiment or a job picks the DevLab runner (workflow runs-on). On the
#   R9700 (gfx1201, 32 GB VRAM, 31 GB host RAM, no HIP headers, no Vulkan loader) this script installs libvulkan1 +
#   mesa-vulkan-drivers (sudo -n apt-get), sets LOOM_MEMORY_CAP_GIB=34 (the budget is VRAM there, not host RAM: memcheck.sh
#   would otherwise refuse anything above 60 % of the host's RAM) and clamps memory_bytes to 34000000000, the R9700 pool the card's Quick start states (card_lib.GPUS; override with
#   "memory_bytes_r9700" on the job/experiment: the per-model budget from fit/verdict.json). llama.cpp ROCm = lemonade's
#   gfx120X build (fetch_llamacpp.sh finds it), Vulkan = b12211. A registration failure of a package (today: WMMA kernels
#   before the gfx12 twins land) ends that step at once as ERROR with the first diagnostics in the log.
#
# KERNEL TEST ON THE R9700 (copy into .github/devlab/perf.json on your branch, label your PR devlab-perf; the dual-target
# twins K_rdna4 run on gfx12 through run_kernel_tests.sh once the package carries them; "build": {} rebuilds the package from
# this branch with the card's own settings; a job that only tests kernels sets "kit" to the model's kit or null):
#   {"card": "loom-fun/unsloth-Qwen3-0.6B-loom", "memory_bytes": 34000000000, "kit": null, "llamacpp": "b1375",
#    "runner": "r9700",
#    "jobs": [{"name": "ktest-r9700", "variants": [{"name": "dual", "build": {}}], "steps": ["ktest:dual"]}]}
#   result: "PASS (N/N passed; M skipped)" in summary.json; per-case lines in ktest_dual.log (artifact devlab-perf-ktest-r9700).
#   Add "LOOM_RDNA4": "1" to the variant's build env when the branch's default is off; steps "perf:dual:1" / "c4:dual" time the
#   rebuilt package on the R9700 and "llama@vulkan.np1:1" / "llamac4" run the baselines on the same runner.
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
vtag=$(q "print(j.get('llamacpp_vulkan') or e.get('llamacpp_vulkan') or '-')")
vkv=$(q "print(j.get('vulkan_kv') or e.get('vulkan_kv') or 'auto')")
lkv=$(q "print(j.get('llama_kv_rocm') or '')")   # ROCm llama.cpp KV pool tokens when 65536 does not fit the GPU (r9700_jobs.py sets it)
RUNNER=$(q "print(j.get('runner') or e.get('runner') or 'strix')")
if [ "$RUNNER" = r9700 ]; then
  rm9=$(q "print(j.get('memory_bytes_r9700') or e.get('memory_bytes_r9700') or '')")
  if [ -n "$rm9" ]; then mem=$rm9; elif [ "$mem" -gt 34000000000 ]; then echo "r9700: memory_bytes $mem clamped to 34000000000 (the R9700 pool, card_lib.GPUS)"; mem=34000000000; fi
  export LOOM_MEMORY_CAP_GIB=${LOOM_MEMORY_CAP_GIB:-34}   # the budget lives in VRAM: memcheck.sh's 60 %-of-host-RAM rule does not apply
  if ! ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so\.1' || ! ls /usr/share/vulkan/icd.d/radeon_icd*.json >/dev/null 2>&1; then
    { sudo -n apt-get install -y libvulkan1 mesa-vulkan-drivers || { sudo -n apt-get update -qq && sudo -n apt-get install -y libvulkan1 mesa-vulkan-drivers; }; } >/dev/null 2>&1 \
      || echo "WARNING: could not install libvulkan1 mesa-vulkan-drivers (the @vulkan steps will fail)"
  fi
elif [ "$RUNNER" != strix ]; then echo "unknown runner $RUNNER (strix | r9700)"; exit 1; fi
BK=rocm; bkt=-   # the llama.cpp backend of the step running (set by the "@vulkan" suffix) and its file-name infix
echo "job $job: variants $variants; steps $steps; runner $RUNNER $(hostname); memory_bytes $mem"
hostname > "$out/runner.txt"

cleanup() {
  for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
  "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
  cp "$LOOM_SERVE_STATE"/*.log "$LLAMACPP_STATE"/*.log "$LLAMACPP_STATE"/*.mem "$out/" 2>/dev/null   # *.mem: MemAvailable/RssAnon of llama-server
  echo "devlab_perf.sh: servers stopped"
}
trap cleanup EXIT

# 1. Loom SDK + tools venv: the SDK the card pins in its llm_config.json (ensure_sdk.sh), else the default
step "Loom SDK"
curl -fsSL --retry 3 "https://huggingface.co/$repo/resolve/main/llm_config.json" -o "$out/llm_config.json" || { echo "FAIL: no llm_config.json in $repo"; exit 1; }
root=$("$SV/ensure_sdk.sh" "${LOOM_ROOT:-$WORK/loom}" "$out/llm_config.json" 2> "$out/install.log") || { tail "$out/install.log"; exit 1; }
export LOOM_ROOT=$root VENV=$root/.venv; T=$root/.tools/bin/python

# 2. The card (fetched once per runner and card)
fetch_card() {   # fetch_card REPO -> sets CARD_DIR: the card's files at one commit (a variant "card": ORG/REPO fetches another card)
  local r=$1 rv bs
  CARD_DIR=$WORK/_card/${r//\//_}; rv=$(curl -fsS --retry 3 "https://huggingface.co/api/models/$r/revision/main" | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])') || rv=main
  # the files are fetched at one commit (resolve/main can serve an older revision for a while after a publish), and a card
  # directory left by an earlier job on this runner is reused only when it holds that commit
  bs=https://huggingface.co/$r/resolve/$rv; CARD_BASE=$bs
  if [ "$rv" = main ] || [ "$(cat "$CARD_DIR/.fetched" 2>/dev/null)" != "$rv" ]; then
    step "fetch $r"; rm -rf "$CARD_DIR"
    HF_HOME="$WORK/hf-home" HF_HUB_CACHE="$WORK/hf-home/hub" "$root/.venv/bin/loom-fetch" "$bs/model.json" --no-input --materialize=copy --cache "$WORK/fetch-cache" --output="$CARD_DIR" > "$out/fetch.log" 2>&1 || { tail "$out/fetch.log"; echo "FAIL: loom-fetch $r"; exit 1; }
    curl -fsSL --retry 3 "$bs/llm_config.json" -o "$CARD_DIR/llm_config.json" || exit 1
    rm -rf "$WORK/fetch-cache" "$WORK/hf-home"; echo "$rv" > "$CARD_DIR/.fetched"
  fi
}
fetch_card "$repo"; card=$CARD_DIR; base=$CARD_BASE
python3 - "$card" <<'EOF' || { echo "FAIL: the fetched card's model.json and llm_config.json disagree (a stale fetch?)"; exit 1; }
import json, os, sys
m = json.load(open(os.path.join(sys.argv[1], "model.json"))); c = json.load(open(os.path.join(sys.argv[1], "llm_config.json")))
depth = next(iter(m["presets"].values()))["execution"].get("draft_depth", 0)
want = (c.get("draft") or {}).get("depth", 0)
sys.exit(0 if depth == want else f"model.json draft_depth {depth} != llm_config.json draft depth {want}")
EOF
"$T" -c "import sys; sys.path.insert(0, sys.argv[1]); from card_lib import programs_sha256; print(programs_sha256(sys.argv[2]))" \
  "$here/skills/loom-benchmark/scripts/card" "$card" > "$out/programs_sha256.txt"   # the published programs this job fetched
# build.py's config: the card's llm_config.json with its GGUFs pointed at the fetched files, plus the tensor table
"$T" - "$card" "$D/cfg.json" "$here/skills/loom-port-llm/starter" <<'EOF' || exit 1
import json, os, sys
import gguf
card, dst, starter = sys.argv[1:]
sys.path.insert(0, starter)
from formats import split_parts
c = json.load(open(os.path.join(card, "llm_config.json")))
par = lambda f: os.path.join(card, "parameter", os.path.basename(f))
c["gguf"] = par(c["gguf"])
if c.get("vision"):
    c["vision"]["mmproj"] = par(c["vision"]["mmproj"])
for k in ("drafter", "assistant"):
    if c.get(k) and c[k].get("gguf"):
        c[k]["gguf"] = par(c[k]["gguf"]) if os.path.exists(par(c[k]["gguf"])) else None
# every part of a split GGUF (the later layers' tensors are in the later parts)
c["tensors"] = {t.name: {"type": t.tensor_type.name.lower(), "shape": [int(x) for x in t.shape]} for p in split_parts(c["gguf"]) for t in gguf.GGUFReader(p).tensors}
for k in ("model_id", "draft"):
    c.pop(k, None)
json.dump(c, open(dst, "w"), indent=1)
EOF
gguf=$("$T" -c "import json; print(json.load(open('$D/cfg.json'))['gguf'])")
drafter=$("$T" -I -c "
import json, os, sys
sys.path.insert(0, sys.argv[1]); from matched_spec import drafter_file
f = drafter_file(json.load(open(sys.argv[2])))   # the drafter / assistant GGUF the package ships (fetched into the card's parameter dir)
p = os.path.join(sys.argv[3], 'parameter', os.path.basename(f)) if f else ''
print(p if p and os.path.exists(p) else '')" "$here/skills/loom-benchmark/scripts/card" "$card/llm_config.json" "$card")
cardid=$("$T" -c "import json; print(json.load(open('$card/model.json'))['model']['id'])")
# the card's own build settings (model id, depth, context, pool)
cardargs=$("$T" -c "
import json; m=json.load(open('$card/model.json')); c=json.load(open('$card/llm_config.json')); ex=next(iter(m['presets'].values()))['execution']
a=['--context', str(ex['context_capacity']), '--pool', str(ex.get('pool_capacity') or ex['context_capacity']), '--prefill', str(ex['prefill_capacity'])]
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
    extra=$(qv "$v" "print(' '.join(v.get('build_args') or []) + (f\" --context {v['context']} --pool {v['context']}\" if v.get('context') else ''))")   # "context": N = the served context of a reduced-context model (later flags win)
    envs=$(qv "$v" "print(' '.join(f'{k}={x}' for k, x in (v.get('build') or {}).items()))")
    step "build $v: env $envs; build.py $cardargs $extra"
    # "config": generator-config overrides for this variant's rebuild (e.g. {"layers": 12}: a truncated model; the step
    # time minus the layer-proportional part is the per-step fixed cost; outputs are garbage, only timing counts)
    vcfg=$(qv "$v" "print(json.dumps(v.get('config') or {}))")
    "$T" - "$D/cfg.json" "$D/$v/cfg.json" "$vcfg" <<'EOF2'
import json, sys
c = json.load(open(sys.argv[1])); c.update(json.loads(sys.argv[3])); json.dump(c, open(sys.argv[2], "w"), indent=1)
EOF2
    ( cd "${KIT[$v]}" && env $envs "$T" build.py "$D/$v/cfg.json" "$pkg" --tokenizer "$card/tokenizer.json" --model-id "$cardid" $cardargs $extra ) > "$out/build_$v.log" 2>&1 \
      || { tail -30 "$out/build_$v.log"; echo "FAIL: build $v"; exit 1; }
    tail -3 "$out/build_$v.log"
  else
    kit_for "$v" "" || exit 1
    vcard=$card; vrepo=$(qv "$v" "print(v.get('card') or '')")   # variant "card": ORG/REPO = that card's published programs (e.g. loom-fun's, next to the staging card's)
    if [ -n "$vrepo" ]; then fetch_card "$vrepo"; vcard=$CARD_DIR; fi
    mkdir -p "$pkg"; cp -r "$vcard/program" "$vcard/model.json" "$vcard/tokenizer.json" "$vcard"/*.loom "$pkg/" 2>/dev/null
    ln -sfn "$vcard/parameter" "$pkg/parameter"
    "$T" - "$vcard" "$pkg" <<'EOF'
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
        m = re.match(r"command\.program\.def\b.*\s@(\w+)\(", ln)
        if m:
            prog = m.group(1)
        out.append(ln)
        if re.match(r"\s*(kernel\.launch @|template\.apply<@)", ln) and any(pr in ("*", prog) and re.search(rx, ln) for pr, rx in rules):
            out.append(ln); n += 1
    open(p, "w").write("\n".join(out)); print(f"ablation: doubled {n} launches")
EOF
  "$T" - "$pkg/model.json" "$(qv "$v" "print(json.dumps(v.get('execution') or {}))")" "$cardid-$v" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1])); ex = next(iter(m["presets"].values()))["execution"]; ex.update(json.loads(sys.argv[2]))
json.dump(m, open(sys.argv[1], "w"), indent=1)
# a small-context copy with its own model id, for several packages in one loom-serve (ab steps)
m["model"]["id"] = sys.argv[3]; ex["context_capacity"] = 16384; ex["pool_capacity"] = 32768
json.dump(m, open(sys.argv[1].replace("model.json", "model_ab.json"), "w"), indent=1)
print("execution", {k: ex[k] for k in ex if k in ("draft_depth", "rows", "prefill_capacity")})
EOF
  cp "$pkg/model.json" "$out/model_$v.json"; sha256sum "$pkg"/program/*.loom > "$out/programs_$v.sha256"
done

serve_alone() { # serve_alone VARIANT
  for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
  "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
  local aw; aw=$(qv "$1" "print(0 if v.get('active_wait') is False else 1)")
  step "serve $1"; LOOM_ACTIVE_WAIT=$aw "$SV/serve.sh" "perf-$1" "$PORT" amdgpu "$mem" "$D/$1/pkg/model.json" > "$out/serve_$1.out" 2>&1; tail -2 "$out/serve_$1.out"
  grep -q '^FAILED' "$out/serve_$1.out" && { echo "REGISTRATION FAILED on $RUNNER for variant $1 (fast fail, see serve_$1.out):"; grep -m6 -E 'AMDGPU|error|ERROR' "$out/serve_$1.out" | cut -c1-300; return 1; }
  curl -sf "$url/v1/models" >/dev/null
}
LSPEC=""
llama_up() {   # llama_up [N_MAX]: llama.cpp with the experiment's llama_spec (n-max overridden), restarted on change
  for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
  # llama_spec absent or "matched": the package's own drafting (matched_spec.py: same method, same drafter GGUF, n-max = Loom's
  # depth; none when the package does not draft). Any other value is an explicit override (a drafter the card doesn't ship...).
  local spec; spec=$(q "print(j['llama_spec'] if 'llama_spec' in j else e.get('llama_spec') or 'matched')")
  [ "$BK" = vulkan ] && { local vs; vs=$(q "print(j.get('llama_spec_vulkan') or e.get('llama_spec_vulkan') or '')"); [ -n "$vs" ] && spec=$vs; }
  [ "$spec" = matched ] && { spec=$("$T" -I "$here/skills/loom-benchmark/scripts/card/matched_spec.py" "$repo") || { echo "FAIL: $repo is not in data/drafting_classification.json"; return 1; }; }
  spec=${spec//'{drafter}'/$drafter}
  [[ $MODS == *nodraft* ]] && spec=""   # .nodraft: the streams4 servers (benchmark rule: 4 streams run undrafted on every engine, Loom's occ_min gate too)
  # N_MAX (the step's last parameter): a number sets --spec-draft-n-max (adding it when the spec has none), "d" = llama.cpp's
  # default (the flag removed). Diagnostic n-max sweep only (llama@rocm.np1:3:-:N); card runs use no N.
  if [ -n "${1:-}" ] && [ -n "$spec" ]; then
    spec=$(echo "$spec" | sed -E "s/ ?--spec-draft-n-max [0-9]+//")
    [ "$1" = d ] || spec="$spec --spec-draft-n-max $1"
  fi
  local h; for h in $(echo "$spec" | grep -oE '\{hf:[^}]+\}' | sort -u); do   # {hf:OWNER/REPO/FILE}, fetched once per runner
    local f=${h#\{hf:}; f=${f%\}}; local dst=$WORK/hf/$f r=${f%/*}
    [ -f "$dst" ] || { mkdir -p "${dst%/*}" && curl -fsSL --retry 3 ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
      "https://huggingface.co/${r}/resolve/main/${f##*/}" -o "$dst.part" && mv "$dst.part" "$dst"; } || { echo "FAIL: fetch $f"; return 1; }
    spec=${spec//"$h"/$dst}
  done
  spec="$BK$MODS|$spec"   # the backend and its modifiers are part of what runs
  [ "$spec" = "$LSPEC" ] && curl -sf "$lurl/health" >/dev/null 2>&1 && return 0
  "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
  local dir
  if [ "$BK" = rocm ]; then
    LLAMACPP_DIR=${LLAMACPP_DIR:-$("$B/fetch_llamacpp.sh" "$WORK/llamacpp" "$tag" | tail -1)}; export LLAMACPP_DIR LLAMA_CACHE_BUILD=$tag; dir=$LLAMACPP_DIR
  else   # the official ggml-org Vulkan release: only this server uses it (LLAMACPP_DIR and the gates' ROCm build stay)
    dir=$(LLAMACPP_BACKEND=$BK "$B/fetch_llamacpp.sh" "$WORK/llamacpp-$BK" "$vtag" | tail -1) || return 1
    cp "$dir/release.sha256" "$out/llamacpp_release$bkt.sha256" 2>/dev/null
  fi
  export LLAMA_SPEC=${spec#*|}; LSPEC=$spec
  step "llama.cpp $BK ${dir##*/}: $LLAMA_SPEC"
  local np=4 kv=(65536 unified)   # one KV pool shared by 4 slots (-kvu)
  [[ $MODS == *np1* ]] && { np=1; kv=(65536); }   # .np1: one slot, the single-stream rows (the same slot policy as Loom's)
  [ "$BK" = rocm ] && [ -n "$lkv" ] && kv[0]=$lkv   # a model whose 65536-token KV pool does not fit beside its weights (32 GB R9700)
  if [ "$BK" = vulkan ]; then   # modifiers: .np1 (one slot of 36864 tokens), .perslot (4 slots of 36864); else vulkan_kv
    if [[ $MODS == *np1* ]]; then np=1; kv=(36864)
    elif [[ $MODS == *perslot* ]]; then kv=(36864)
    # auto: per-slot unless the server drafts with DFlash (b12211 aborts a DFlash server with 4 concurrent streams on per-slot caches)
    elif [ "$vkv" = per-slot ] || { [ "$vkv" = auto ] && [[ "$LLAMA_SPEC" != *draft-dflash* ]]; }; then kv=(36864); fi
  fi
  LLAMACPP_BACKEND=$BK LLAMACPP_DIR=$dir LLAMACPP_ARGS_FILE=$out/llamacpp_args$bkt.txt "$B/llamacpp_server.sh" "$gguf" "$LLAMA_PORT" $np "${kv[@]}" | tail -1
  sed -i -e "s#$WORK/hf/#hf:#g" -e "s#$card/#card:#g" "$out/llamacpp_args$bkt.txt" 2>/dev/null   # runner paths out of the recorded arguments
  [ "${PIPESTATUS[0]}" = 0 ] || { cp "$LLAMACPP_STATE/llamacpp-$LLAMA_PORT.log" "$out/llamacpp_start$bkt.log" 2>/dev/null; return 1; }
}
TH=$(q "print(j.get('thinking') or 'off')")   # chat_template_kwargs.enable_thinking sent to both engines
echo "thinking: $TH (sent to both engines)"
declare -A res
for s in $steps; do
  IFS=: read -r kind p1 p2 p3 <<< "$s"
  BK=rocm; bkt=; MODS=; case $kind in *@*) bkt=-${kind#*@}; BK=${kind#*@}; MODS=${BK#"${BK%%.*}"}; BK=${BK%%.*}; kind=${kind%@*};; esac   # llama@vulkan: the other backend's build
  [ "$BK" = rocm ] || [ "$vtag" != - ] || { echo "$s: set llamacpp_vulkan"; res[$s]=ERROR; continue; }
  [ "${p2:-}" = - ] && p2=""
  case $kind in
    ab)
      for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
      "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
      ms=(); ids=(); for v in $variants; do ms+=("$D/$v/pkg/model_ab.json"); ids+=("$cardid-$v"); done
      nv=${#ms[@]}; abmem=$(( 11000000000 * nv + 4000000000 )); [ "$RUNNER" = r9700 ] && [ "$abmem" -gt "$mem" ] && abmem=$mem
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
    probe)   # probe:V: stall_probe.py (when does a prefill stall? TTFT + GPU clocks per request)
      serve_alone "$p1" || { res[$s]=ERROR; continue; }
      step "stall probe $p1"
      "$T" "$here/.github/scripts/stall_probe.py" "$url" "$cardid" "$out/probe_$p1.jsonl" > "$out/probe_$p1.log" 2>&1; tail -3 "$out/probe_$p1.log" | cut -c1-200
      res[$s]="$(wc -l < "$out/probe_$p1.jsonl") requests";;
    c4)
      serve_alone "$p1" || { res[$s]=ERROR; continue; }
      step "c4 $p1"
      cn=${p2:-4}; cf=c4_$p1$([ "$cn" = 4 ] || echo "_n$cn").txt   # c4:V[:N] N concurrent streams (default 4)
      for i in 1 2 3; do (cd "$out" && "$T" "$B/c4_steady.py" "$url" "$cn" --model "$cardid" --thinking "$TH" | tail -1 | tee -a "$cf"); done
      res[$s]=$("$T" -c "import json,statistics as st; print('$p1 c4$([ "$cn" = 4 ] || echo " n$cn") median', st.median(json.loads(l)['agg_tps'] for l in open('$out/$cf')))");;
    llama)   # llama:N[:long|-][:N_MAX]
      llama_up "${p3:-}" || { res[$s]=ERROR; continue; }
      nm=$(echo "$LSPEC" | grep -oE "n-max [0-9]+" | grep -oE "[0-9]+"); nm=${nm:-0}; lf=perf_llamacpp${bkt}_n$nm.jsonl
      cp "$out/llamacpp_args$bkt.txt" "$out/llamacpp_args$bkt.n$nm.txt" 2>/dev/null   # the arguments of THIS n-max (the plain file is overwritten per server start)
      step "llama.cpp$bkt n-max $nm perf x $p1 ${p2:-} (thinking $TH)"
      for i in $(seq 1 "$p1"); do
        (cd "$out" && "$T" "$B/perf.py" "$lurl" "llamacpp${bkt}-n$nm" --kind llama --no-campaign --thinking "$TH" --log "$lf" ${p2:+--long} | tail -1)
      done
      [ -z "${p2:-}" ] && "$T" "$here/.github/scripts/llama_decode.py" "$lurl" 2 --think "$TH" --log "$out/tps_llamacpp${bkt}_n$nm.jsonl"
      res[$s]=$("$T" -c "
import json,statistics as st; r=[json.loads(l) for l in open('$out/$lf')]
k='decode_tps_32k' if '${p2:-}' else 'decode_tps'
print('llamacpp$bkt n-max $nm', k, [x.get(k) for x in r if x.get(k)], 'mean', round(st.mean(x[k] for x in r if x.get(k)), 2))");;
    llamac4)   # llamac4[:N_MAX]
      llama_up "${p1:-}" || { res[$s]=ERROR; continue; }
      nm=$(echo "$LSPEC" | grep -oE "n-max [0-9]+" | grep -oE "[0-9]+"); nm=${nm:-0}   # 0: no drafting
      step "llama.cpp$bkt n-max $nm c4"
      for i in 1 2 3; do (cd "$out" && "$T" "$B/c4_steady.py" "$lurl" 4 --thinking "$TH" | tail -1 | tee -a "c4_llamacpp${bkt}_n$nm.txt"); done
      res[$s]=$("$T" -c "import json,statistics as st; print('llamacpp$bkt n-max $nm c4 median', st.median(json.loads(l)['agg_tps'] for l in open('$out/c4_llamacpp${bkt}_n$nm.txt')))");;
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
      grep -q '"catalog"' "$card/llm_config.json" && { res[$s]="SKIP (catalog package: no generated kernel tests)"; continue; }
      serve_alone "$p1" >/dev/null 2>&1; for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
      step "kernel tests $p1"; kt=$D/$p1/ktest; rm -rf "$kt"
      ( cd "${KIT[$p1]}" && LLM_DRAFT=$("$T" -c "import json, sys; print(json.dumps(json.load(open(sys.argv[1])).get('draft')))" "$D/$p1/pkg/llm_config.json") \
          "$T" make_kernel_tests.py "$D/cfg.json" "$kt" ) > "$out/ktest_make_$p1.log" 2>&1 || { tail "$out/ktest_make_$p1.log"; res[$s]=ERROR; continue; }
      "${KIT[$p1]}/run_kernel_tests.sh" "$D/$p1/pkg" "$kt" > "$out/ktest_$p1.log" 2>&1; rc=$?
      grep -v "^PASS" "$out/ktest_$p1.log" | tail -15
      tot=$(grep -oE "^[0-9]+/[0-9]+ passed" "$out/ktest_$p1.log" | tail -1)
      res[$s]="$([ $rc = 0 ] && [ -n "$tot" ] && [ "${tot%%/*}" = "$(echo "${tot#*/}" | cut -d' ' -f1)" ] && echo PASS || echo FAIL) ($tot; $(grep -c '^SKIP' "$out/ktest_$p1.log") skipped)";;
    func)
      serve_alone "$p1" || { res[$s]=ERROR; continue; }
      step "functional checks $p1"
      curl -fsSL --retry 2 "$base/info.json" -o "$out/info_card.json" 2>/dev/null || rm -f "$out/info_card.json"
      fa=(); grep -q '"vision"' "$card/llm_config.json" && fa+=(--vision)
      "$T" "$here/.github/scripts/r9700_func.py" "$url" "$cardid" "$out/func_$p1.json" ${fa[@]+"${fa[@]}"} $([ -f "$out/info_card.json" ] && echo --info "$out/info_card.json") 2>&1 | tee "$out/func_$p1.log"
      curl -s -m 10 "$url/v1/status" > "$out/status_$p1.json" 2>/dev/null   # registration + memory plan as loom-serve reports it after the first requests
      res[$s]=$(tail -1 "$out/func_$p1.log");;
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
