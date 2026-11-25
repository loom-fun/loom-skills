#!/bin/bash
# One job of a DevLab performance experiment (.github/workflows/devlab-perf.yml): fetch a published loom-fun card,
# build package variants from this checkout (or another ref), and compare them with each other and with llama.cpp
# ON THIS RUNNER (runners differ from each other and from a dev machine: only same-job numbers are compared).
# This is the single source of card performance: the card variant (active wait) and llama.cpp on the same runner,
# ingested with make_info.py perf --source-kind devlab (loom-fleet reference/publish.md step 3).
#
#   [WORK=~/devlab-perf] devlab_perf.sh EXPERIMENT.json JOB_NAME
#
# EXPERIMENT.json:
#   {"card": "loom-fun/<repo>", "memory_bytes": N, "kit": "gemma4" | null, "llamacpp": "b1368",
#    "llama_spec": "matched" (default),   the llama.cpp servers' drafting = the card's own (see below); or an explicit override
#                  "--spec-type draft-dflash --spec-draft-model {drafter} ...",   ({drafter}: the card's drafter GGUF;
#                                                                     {hf:OWNER/REPO/FILE}: that Hub file, e.g. a drafter llama.cpp loads and the card doesn't ship; also in build_args)
#   Immutable external file: {hf:OWNER/REPO@40HEX/FILE#64HEXSHA256}; revision-specific cache, checksum checked on every use.
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
#   "llama_kv_vulkan": positive integer tokens per slot (or total tokens with unified KV); optional, keeps the
#   selected Vulkan slot policy and changes only its capacity when the default does not fit beside the weights.
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
#            build: build.py with these env vars (and extra flags); ref: build with that commit's starter AND its knowledge base (skills/loom-kb:
#            the choices the rebuild resolves; ab_jobs.py --base-ref/--head-ref --build writes fun/staging this way);
# SDK per variant (SDK A/B, fleet/regen.py plan --mode sdk-ab): {"sdk": "<version>" | "card"} on a variant = the hrx-loom the variant is
#            SERVED with. "card" (the default) = the pin of the package it serves (its llm_config.json "sdk", none = 0.1.0): ensure_sdk.sh gives
#            every distinct SDK its own root ($WORK/loom-sdk-<version>, installed once per runner) and serve.sh runs with that root's VENV, so an
#            old card on its old SDK and a package built from this checkout (or the same card) on the new SDK are measured in ONE job, ABBA (the steps list the order:
#            perf:fun:1 perf:staging:1 perf:staging:1 perf:fun:1 ...). out/sdk_<variant>.json = the root's loom-install.json, install_<variant>.log.
#            Job (or experiment) "sdk": the SDK of the TOOLS (build.py, perf.py, install.log = the identity ingest reads): a package built here
#            is built and stamped by it; without it the tools come from the job card's own pin. A card is always fetched with the SDK it pins.
#            The single-server `ab` step hosts all its variants in one loom-serve, so it refuses variants whose SDKs differ.
# Job "require_preset_r9700": true (r9700_jobs.py, a package with a fit-row r9700_execution): serving fails unless the card has its <first>-r9700 preset.
#            serve_args: extra loom-serve flags; execution: model.json overrides (every preset); double: ablation (each kernel.launch in PROGRAM matching REGEX twice).
#            {"name": "c3", "catalog": "github:ROCm/hrx-loom-library@v0.3.0", "model": "Qwen/Qwen3.8-27B", "preset": "chat", "sdk": "0.3.0"}: a catalog
#            CONTROL (an SDK regression's root cause: the library's own package at the tag matching that SDK, fetched and served with that SDK; "sdk" required),
#            served under the job card's model id; perf / profile / c4 steps only (no tokenizer, gates or kernel tests of its own)
#            {"name": "parked", "active_wait": false}: the card's programs served with the SDK's default completion wait
#            (LOOM_ACTIVE_WAIT=0 in serve.sh) instead of the active wait every other variant gets.
# Job/experiment "env": {"NAME": "value"}: environment for every step.
# Job "thinking": off (default) | on: chat_template_kwargs.enable_thinking sent to BOTH engines in every step.
# Steps S: ab:R[:on|off]        every variant in ONE loom-serve (small context), ab_decode.py R interleaved rounds
#          perf:V:N:ctx         perf.py --decode-ctx 16000,124000 (decode_tps_16k / decode_tps_128k rows)
#          perf:V:N[:long]      V served alone with its own settings: perf.py N times (+ --long); N runs = one session (its value: the median of
#                               its samples); one run per session = perf:V:1 repeated (a round each). Job "repeat_rows": R (fast models: 3) adds --repeat-rows R
#          profile:V[:LIST]     per-kernel device time of one prefill (pf2k, pf8k) and one drafted decode (dec; dec2k / dec32k: perf.py's decode_2k / decode_32k
#                               request without a nonce, identical text on every variant): V served per scenario with --print_device_statistics
#                               (the HAL's per-function dispatch statistics at shutdown), profile_requests.py + profile_stats.py: out/profile_V_<scenario>.{md,json,serve.log}
#          observe:V[:LIST]     diagnostic trace plus bounded Linux CPU/thread/wait/IO snapshots (observer; never a primary perf input)
#          trace:V[:LIST]       same diagnostic requests as profile without HAL timestamps (default: dec); observer-effect control
#          c4:V[:N]             c4_steady.py N (default 4) streams on V
#          llama:N[:long|-][:M] llama.cpp (card build + llama_spec, n-max M) perf.py N times, llama_decode.py tokens/step
#          llamac4[:M]          c4_steady.py on llama.cpp
#          (llama[@vulkan|@rocm], llamac4[@vulkan|@rocm]: the same with the Vulkan build; files perf_llamacpp-vulkan_nN.jsonl. Modifiers
#          after @backend pick the server configuration, e.g. llama@vulkan.np1:2 : .np1 one slot (Vulkan 36864 tokens, ROCm 65536;
#          no 4-stream step), .perslot (Vulkan) 4 slots of 36864; files perf_llamacpp-vulkan.np1_nN.jsonl)
#          textmatch:V          text_match.py: first 50 tokens of perf.py's prompts on V, then on llama.cpp
#          consistency:V:N      consistency.py N times (NOT DETERMINISTIC / concurrent mismatch counts)
#          determinism:V:C:R    determinism.py (issue #39's script, when present)
#          admit:V[:TOKENS]     ctx_admit.py on V served alone: a prompt of TOKENS (default: its context minus the answer) is admitted or refused (admit_V*.txt)
#          vllm:N[:long]        vLLM ROCm (vllm_server.sh --install) serving the card's checkpoint, the card's drafting (job "vllm_spec" = the --speculative-config JSON), perf.py --kind vllm N times
#                               (job "vllm_max_len": --max-model-len, default the served context); perf_vllm_nN.jsonl, vllm_args_on.txt, vllm_kv_on.txt, vllm_version.txt (a card with no llama.cpp baseline)
#          vllmc4[:N]           4 streams on vLLM without drafting, N (default 3) runs of c4_steady.py: c4_vllm_nN.txt
#          gufoperf:N[:long]    Gufo (job/experiment "gufo": {"image": DOCKER_IMAGE@sha256, "model": "OWNER/REPO@REV/FILE", "draft": "OWNER/REPO@REV/FILE",
#                               "speculative": "dflash2", "context": 65536}) serving ITS OWN model through its OpenAI server (gufo serve llm, the release
#                               container under docker), perf.py --kind gufo N times (client-timed like Loom): perf_gufo.jsonl, gufo_args_on.txt,
#                               gufo_version.txt, gufo_model.txt (file + sha256), gufo_metrics_on_*.txt (its /metrics after each session: drafted acceptance)
#          gufoc4[:N]           4 streams on Gufo without drafting (--speculative off --sessions 4), N (default 3) runs of c4_steady.py: c4_gufo.txt
# Local-machine evidence (a dev machine measuring a card at a non-default power mode, make_info.py perf --source-kind local):
#   POWER_LOG=1      out/power.tsv (epoch, package W, GPU MHz every 0.2 s from the amdgpu hwmon) and out/step_times.tsv (step, start, end);
#                    at the end power_steps.py writes out/power_steps.json (per step and per engine: median W and GHz of the busy samples).
#   MIN_MEMAVAIL_GB=N  refuse to start any server while MemAvailable is below N GB (a 27B server next to a leftover one OOMs a 128 GB machine).
#          fullctx:V[:TOKENS]   ctx_admit.py --complete: a request of ~the served context (or TOKENS) is prefilled to the end and answered (fullctx_V*.txt)
#          coldcapacity:V      opt-in experiment: cold sampled memory, exact full-context completion and structured overflow rejection
#          func:V               V served alone: the functional checks the card claims (chat, tool calling, image input;
#                               r9700_func.py: the same checks as the Windows red-team harness), out/func_V.json
#          logits:V[:N]         host logits readback (loom-skills#736): V's package through the Gufo bundle (loom-gufo-extension
#                               validation/devlab/build_bundle.sh: gufo + libloom_text_embed.so with their own loader; $GUFO_BUNDLE, or job/
#                               experiment "bundle": {"url": TARBALL, "sha256": H}; the devlab-perf workflow builds it in the run for a job with "bundle": {"build": {...}} and sets $GUFO_BUNDLE), logits_check.py: embed_probe (argmax of every returned row ==
#                               the device's token), gufo's host-sampled logprobs vs llama.cpp (same GGUF, undrafted), greedy text unchanged by
#                               the capability; logits_V.json (gate `logits`, make_info.py gates). N prompts (default 6). Without a bundle
#                               the step is "not supported" (never replaced by another path).
#          gufo:V               Gufo conformance on V's package served with --loom-logits=on (validation/devlab/run_conformance.sh: conformance.py
#                               --logits + gufo's logits_check.py); conformance_V.json (gate `gufo`), the raw files in out/gufo_V/
#          vllmdet:V            (safetensors card, or env REFERENCE_CHECKPOINT=ORG/REPO on a GGUF card) vLLM's run-to-run determinism: the dev golden set recorded twice
#          gates:V:LEVEL        with a safetensors card (llm_config source.kind) or REFERENCE_CHECKPOINT the reference is vLLM (scripts/vllm_reference.md), not llama.cpp
#          ktest:V              make_kernel_tests.py + run_kernel_tests.sh with V's kit
#          vtest:V[:HxW]        the image tower (a package with image input): starter/vision/tests (encoder kernels vs NumPy), then the whole encoder
#                               at HxW pixels (default 320x224) on the package's weights (a safetensors package: the checkpoint's own tensors) vs
#                               ref_vit.py, every output within 1.0 + 3 % of the reference; vtest_V.log, vtest_encoder_V.log
#          gates:V:LEVEL        gates.sh against llama.cpp
#            "elastic": true (variant): serve with --memory_bytes (the elastic state backing) on any GPU, instead of the device's card serve_args
#            (Strix Halo and the R9700: --pool_backing=fixed, no --memory_bytes; added to every variant that brings no serve_args with --pool_backing
#            and no memory_bytes of its own). The single-server `ab` step is always elastic (several models share one budget).
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
export LOOM_KB_ROOT=${LOOM_KB_ROOT:-$SK/loom-kb}   # kits copied out of the skills tree (ktest, --build) resolve choices from this checkout's knowledge base
B=$SK/loom-benchmark/scripts; SV=$SK/loom-setup/scripts
WORK=${WORK:-$HOME/devlab-perf}; mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
D=$WORK/$job; out=$D/out; rm -rf "$D"; mkdir -p "$out"
export LOOM_SERVE_STATE=$D/state LLAMACPP_STATE=$D/llstate PYTHONDONTWRITEBYTECODE=1
PORT=${LOOM_PORT:-8161}; LLAMA_PORT=${LLAMA_PORT:-8131}; url=http://127.0.0.1:$PORT; lurl=http://127.0.0.1:$LLAMA_PORT
t0=$(date +%s); step() { echo "== $* ($(date +%T), +$(( $(date +%s) - t0 ))s)"; }
q() { python3 -c "import json,sys; e=json.load(open('$exp')); j=next(j for j in e['jobs'] if j['name']=='$job'); $1"; }
qv() { python3 -c "import json,sys; e=json.load(open('$exp')); j=next(j for j in e['jobs'] if j['name']=='$job'); v=next(v for v in j['variants'] if v['name']=='$1'); $2"; }
# Job "kind": "image" (fleet/regen.py --programs production, an image package): no text steps. image_sdk_compare.py serves the pinned package
# ("card": ORG/REPO@REV) on the old and the new SDK (job "image": {"old_sdk", "new_sdk", "programs_sha256", "preset"}), ABBA sessions, the
# preset's deterministic request: identical pixels, programs hash checked before serving, timings per session; writes out/summary.json.
if [ "$(q "print(j.get('kind') or '')")" = image ]; then
  hostname > "$out/runner.txt"
  iold=$(q "print(j['image']['old_sdk'])"); inew=$(q "print(j['image']['new_sdk'])")
  for v in old new; do
    sv=$([ $v = old ] && echo "$iold" || echo "$inew")
    r=$(LOOM_VERSION=$sv "$SV/ensure_sdk.sh" "$WORK/loom-sdk-${sv//:/-}" "" 2> "$out/install_$v.log") || { tail -5 "$out/install_$v.log"; echo "FAIL: SDK $sv"; exit 1; }
    cp "$r/loom-install.json" "$out/sdk_$v.json"; eval "IROOT_$v=\$r"
  done
  "$IROOT_new/.tools/bin/python" "$here/.github/scripts/image_sdk_compare.py" "$exp" "$job" --old-root "$IROOT_old" --new-root "$IROOT_new" \
    --work "$D" --out "$out" --port "$PORT"
  exit $?
fi
# a card reference is ORG/REPO or ORG/REPO@REV (job "card" and variant "card"): REV pins the commit fetched instead of main
repo_ref=$(q "print(j.get('card') or e['card'])"); repo=${repo_ref%@*}; mem=$(q "print(j.get('memory_bytes') or e['memory_bytes'])")
kit=$(q "print((j['kit'] if 'kit' in j else e.get('kit')) or '-')"); tag=$(q "print(j.get('llamacpp') or e.get('llamacpp') or '-')"); variants=$(q "print(' '.join(v['name'] for v in j['variants']))")
steps=$(q "print(' '.join(j['steps']))")
vtag=$(q "print(j.get('llamacpp_vulkan') or e.get('llamacpp_vulkan') or '-')")
vkv=$(q "print(j.get('vulkan_kv') or e.get('vulkan_kv') or 'auto')")
lkv=$(q "print(j.get('llama_kv_rocm') or '')")   # ROCm llama.cpp KV pool tokens when 65536 does not fit the GPU (r9700_jobs.py sets it)
lvkv=$(q "v=j.get('llama_kv_vulkan'); assert v is None or (type(v) is int and v > 0), 'llama_kv_vulkan must be a positive integer'; print(v if v is not None else '')") || exit 1
# "repeat_rows": 3 on a job (a fast model, published decode > 100 tok/s: methodology rule 5): every perf.py run measures decode_2k,
# decode_32k and prefill_8k that many times (perf.py --repeat-rows); a session's value is the median of its samples
rrows=$(q "print(j.get('repeat_rows') or e.get('repeat_rows') or 1)")
RUNNER=$(q "print(j.get('runner') or e.get('runner') or 'strix')")
# "env": {"NAME": "value"} on the experiment or a job: exported for every step (gates.sh switches; the job's wins)
eval "$(q "import shlex; print('\n'.join(f'export {k}={shlex.quote(str(v))}' for k, v in {**(e.get('env') or {}), **(j.get('env') or {})}.items()))")"
# a runner other than strix is a device with its own memory pool (the registry's device.json: skills/loom-kb, runner_device.py): RDEV is its
# device key, RPOOL its memory_bytes, RSUF its preset suffix; the job's "memory_bytes_<runner>" / "require_preset_<runner>" are keyed by the runner
RDEV=""; RPOOL=0; RSUF=-
if [ "$RUNNER" != strix ]; then
  read -r RDEV RPOOL RSUF < <(python3 -I "$here/.github/scripts/runner_device.py" "$RUNNER") || { echo "unknown runner $RUNNER"; exit 1; }
fi
if [ -n "$RDEV" ]; then
  rm9=$(q "print(j.get('memory_bytes_$RUNNER') or e.get('memory_bytes_$RUNNER') or '')")
  if [ -n "$rm9" ]; then mem=$rm9; elif [ "$RPOOL" -gt 0 ] && [ "$mem" -gt "$RPOOL" ]; then echo "$RUNNER: memory_bytes $mem clamped to $RPOOL (the $RDEV pool, card_lib.GPUS)"; mem=$RPOOL; fi
  export LOOM_MEMORY_CAP_GIB=${LOOM_MEMORY_CAP_GIB:-$((RPOOL / 1000000000))}   # the budget lives in VRAM: memcheck.sh's 60 %-of-host-RAM rule does not apply
  if ! ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so\.1' || ! ls /usr/share/vulkan/icd.d/radeon_icd*.json >/dev/null 2>&1; then
    { sudo -n apt-get install -y libvulkan1 mesa-vulkan-drivers || { sudo -n apt-get update -qq && sudo -n apt-get install -y libvulkan1 mesa-vulkan-drivers; }; } >/dev/null 2>&1 \
      || echo "WARNING: could not install libvulkan1 mesa-vulkan-drivers (the @vulkan steps will fail)"
  fi
fi
BK=rocm; bkt=-   # the llama.cpp backend of the step running (set by the "@vulkan" suffix) and its file-name infix
echo "job $job: variants $variants; steps $steps; runner $RUNNER $(hostname); memory_bytes $mem"
hostname > "$out/runner.txt"

VPORT=${VLLM_PORT:-8132}; vurl=http://127.0.0.1:$VPORT; vpid=$D/vllm-$VPORT.pid
vllm_down() {   # stop the vLLM OpenAI server this job started (its process group: the API server and its EngineCore child)
  [ -f "$vpid" ] || return 0
  local p; p=$(cat "$vpid"); kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null
  for _ in $(seq 120); do kill -0 "$p" 2>/dev/null || break; sleep 0.5; done
  kill -9 -- -"$p" 2>/dev/null; rm -f "$vpid"
  # the recorded pid can be a launcher that already exited (the final MXFP4 card job: the spec-on server kept answering for 180 s after vllm_down, and the
  # no-draft c4 row failed): stop whatever serves this port by its command line, then the engine core it leaves behind
  pkill -TERM -f "vllm.entrypoints.openai.api_server.*--port $VPORT( |\$)" 2>/dev/null
  for _ in $(seq 60); do curl -sf "$vurl/v1/models" >/dev/null 2>&1 || break; sleep 1; done
  pkill -9 -f "vllm.entrypoints.openai.api_server.*--port $VPORT( |\$)" 2>/dev/null; pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  return 0
}
GPORT=${GUFO_PORT:-8133}; gurl=http://127.0.0.1:$GPORT; GCONT=gufo-$job-$$
gufo_down() {   # stop the Gufo container this job started (docker), and wait until its port is closed
  [ -n "${GUFO_MODE:-}" ] || return 0
  docker rm -f "$GCONT" >/dev/null 2>&1
  for _ in $(seq 60); do curl -sf "$gurl/v1/models" >/dev/null 2>&1 || break; sleep 1; done
  GUFO_MODE=; return 0
}
mem_guard() {   # mem_guard WHAT: MIN_MEMAVAIL_GB (default 0: off) GB of MemAvailable before a server starts
  local need=${MIN_MEMAVAIL_GB:-0} avail; [ "$need" -gt 0 ] || return 0
  for _ in $(seq 60); do
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$avail" -ge "$need" ] && { echo "MemAvailable ${avail} GB before $1" >> "$out/memavail.txt"; return 0; }
    sleep 2
  done
  echo "FAIL: MemAvailable ${avail} GB < $need GB before $1" | tee -a "$out/memavail.txt"; return 1
}
cleanup() {
  coldcapacity_stop
  vllm_down
  gufo_down
  [ -n "${PLOG_PID:-}" ] && kill "$PLOG_PID" 2>/dev/null
  for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
  "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1; VLLM_STATE=$D/vllm-state "$B/vllm_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
  cp "$D"/vllm-state/vllm-*.{log,mem} "$out/" 2>/dev/null
  cp "$LOOM_SERVE_STATE"/*.log "$LLAMACPP_STATE"/*.log "$LLAMACPP_STATE"/*.mem "$out/" 2>/dev/null   # *.mem: MemAvailable/RssAnon of llama-server
  echo "devlab_perf.sh: servers stopped"
}
trap cleanup EXIT

coldcapacity_stop() {
  if [ -n "${CAPACITY_MONITOR_PID:-}" ]; then
    touch "$CAPACITY_MONITOR_DIR/monitor.stop"
    wait "$CAPACITY_MONITOR_PID" || true
    CAPACITY_MONITOR_PID=""
  fi
}

# 1. Loom SDK + tools venv: the SDK the card pins in its llm_config.json (ensure_sdk.sh), else the default
step "Loom SDK"
curl -fsSL --retry 3 "https://huggingface.co/$repo/resolve/$([[ $repo_ref == *@* ]] && echo "${repo_ref#*@}" || echo main)/llm_config.json" -o "$out/llm_config.json" || { echo "FAIL: no llm_config.json in $repo"; exit 1; }
ROOT_BASE=${LOOM_ROOT:-$WORK/loom}
root=$("$SV/ensure_sdk.sh" "$ROOT_BASE" "$out/llm_config.json" 2> "$out/install.log") || { tail "$out/install.log"; exit 1; }   # the job card's own SDK (loom-fetch of the card)
# Explicit CI pins contain a colon, which uv rejects in a venv path (PATH separator).
# Keep the selector intact for installation and normalize only the directory name.
# job "sdk": the tools' SDK differs from the card's pin (an old card on its old SDK next to a package served here on the new one): install.log is that SDK's
jsdk=$(q "print(j.get('sdk') or e.get('sdk') or 'card')"); troot=$root
if [ "$jsdk" != card ]; then
  troot=$(LOOM_VERSION=$jsdk "$SV/ensure_sdk.sh" "$WORK/loom-sdk-${jsdk//:/-}" "" 2> "$out/install.log") || { tail "$out/install.log"; echo "FAIL: SDK $jsdk"; exit 1; }
fi
export LOOM_ROOT=$troot VENV=$troot/.venv; T=$troot/.tools/bin/python
echo "SDK roots: tools $troot (job sdk $jsdk); job card $root"
declare -A SDKROOT
sdk_root() {   # sdk_root VARIANT -> SDKR = the root of the SDK the variant is served with (variant "sdk"; "card" = the pin of its package, a built package = the tools' SDK)
  local v=$1 s r; SDKR=
  [ -n "${SDKROOT[$v]:-}" ] && { SDKR=${SDKROOT[$v]}; return 0; }
  s=$(qv "$v" "print(v.get('sdk') or 'card')")
  if [ "$s" != card ]; then r=$(LOOM_VERSION=$s "$SV/ensure_sdk.sh" "$WORK/loom-sdk-${s//:/-}" "" 2> "$out/install_$v.log") || { tail -5 "$out/install_$v.log" >&2; return 1; }
  elif [ "$(qv "$v" "print('build' in v)")" = True ]; then r=$troot
  else r=$("$SV/ensure_sdk.sh" "$ROOT_BASE" "$D/$v/pkg/llm_config.json" 2> "$out/install_$v.log") || { tail -5 "$out/install_$v.log" >&2; return 1; }; fi
  SDKROOT[$v]=$r; SDKR=$r; cp "$r/loom-install.json" "$out/sdk_$v.json" 2>/dev/null
  echo "variant $v: SDK $s -> root $r ($(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('hrx_loom'), d.get('source'), str(d.get('hrx_system_sha'))[:12])" "$r/loom-install.json" 2>/dev/null))"
}

# 2. The card (fetched once per runner and card)
fetch_card() {   # fetch_card REF -> sets CARD_DIR, CARD_BASE, CARD_REV: the card's files at one commit (REF = ORG/REPO[@REV]; a variant "card": REF fetches another card)
  local ref=$1 r=${1%@*} want=main rv bs
  [[ $ref == *@* ]] && want=${ref#*@}   # pinned: that revision instead of main (an old published revision, or a commit of the staging card)
  CARD_DIR=$WORK/_card/${r//\//_}; rv=$(curl -fsS --retry 3 "https://huggingface.co/api/models/$r/revision/$want" | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])') || rv=$want
  [ "$want" = main ] || CARD_DIR=${CARD_DIR}@${rv:0:12}   # a pinned card has its own directory next to the same card at main
  # the files are fetched at one commit (resolve/main can serve an older revision for a while after a publish), and a card
  # directory left by an earlier job on this runner is reused only when it holds that commit
  bs=https://huggingface.co/$r/resolve/$rv; CARD_BASE=$bs; CARD_REV=$rv
  if [ "$rv" = main ] || [ "$(cat "$CARD_DIR/.fetched" 2>/dev/null)" != "$rv" ]; then
    step "fetch $r"; rm -rf "$CARD_DIR"
    # a card with several presets (the R9700's `<first>-r9700`) needs one named: install the first (the default), then add the others' blocks
    # to the installed model.json (same variant and programs, so they only name other execution settings): both can then be served
    curl -fsSL --retry 3 "$bs/model.json" -o "$out/model_card.json" || { echo "FAIL: no model.json in $r"; exit 1; }
    pf=$(python3 -c "import json,sys; p=list(json.load(open(sys.argv[1]))['presets']); print('--preset=' + p[0] if len(p) > 1 else '')" "$out/model_card.json")
    # LOCAL_HF_CACHE=1 (a dev machine, not a runner): the weights come from (and stay shared with) the user's own HF cache,
    # never a second download or copy of a multi-GB GGUF
    if [ "${LOCAL_HF_CACHE:-0}" = 1 ]; then hfh=${HF_HOME:-$HOME/.cache/huggingface}; mat=share; else hfh=$WORK/hf-home; mat=copy; fi
    HF_HOME="$hfh" HF_HUB_CACHE="$hfh/hub" "$root/.venv/bin/loom-fetch" "$bs/model.json" $pf --no-input --materialize=$mat --cache "$WORK/fetch-cache" --output="$CARD_DIR" > "$out/fetch.log" 2>&1 || { tail "$out/fetch.log"; echo "FAIL: loom-fetch $r"; exit 1; }
    [ -z "$pf" ] || python3 - "$out/model_card.json" "$CARD_DIR/model.json" <<'EOF' || exit 1
import json, sys
src, dst = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
for n, p in src["presets"].items():
    if n not in dst["presets"]:
        assert p["variant"] in dst["variants"] and p["programs"] in dst["programs"], f"preset {n} needs another variant or program set"
        dst["presets"][n] = p
json.dump(dst, open(sys.argv[2], "w"), indent=1)
print("presets:", list(dst["presets"]))
EOF
    curl -fsSL --retry 3 "$bs/llm_config.json" -o "$CARD_DIR/llm_config.json" || exit 1
    rm -rf "$WORK/fetch-cache" "$WORK/hf-home"; echo "$rv" > "$CARD_DIR/.fetched"
  fi
}
fetch_card "$repo_ref"; card=$CARD_DIR; base=$CARD_BASE
echo "$repo@$CARD_REV" > "$out/card_revision.txt"   # the commit of the job's card this job fetched (a pinned REV resolved to the full sha)
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
if (c.get("source") or {}).get("kind") == "safetensors":   # an HF safetensors checkpoint: the tensor table from its header (HTTP range reads), the weights as fetched
    import inspect_safetensors
    src = inspect_safetensors.load_hf(c["source"]["repo"], c["source"]["revision"])
    bc = inspect_safetensors.build_config(src, drop_chat=True)
    c["tensors"] = bc["tensors"]
    for k in ("vision", "mrope"):   # the image tower is part of the config the branch's starter derives from the checkpoint
        if bc.get(k):
            c[k] = bc[k]
    c["weights"] = {"format": "safetensors", "files": [par(f) for f in c["source"]["files"]]}
else:
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
# the reference's model: the GGUF llama.cpp loads, or (a safetensors card) the checkpoint repo vLLM loads, pinned to the card's source revision
gguf=$("$T" -c "import json; c=json.load(open('$D/cfg.json')); s=c.get('source') or {}; print(s['repo'] if s.get('kind') == 'safetensors' else c.get('gguf') or '')")
safetensors=$("$T" -c "import json; print(1 if (json.load(open('$D/cfg.json')).get('source') or {}).get('kind') == 'safetensors' else '')")
[ -n "${REFERENCE_CHECKPOINT:-}" ] && { safetensors=1; gguf=$REFERENCE_CHECKPOINT; }   # vLLM forced on a GGUF package (adapter check)
[ -n "$safetensors" ] && [ -z "${REFERENCE_CHECKPOINT:-}" ] && export VLLM_REVISION=$("$T" -c "import json; print(json.load(open('$D/cfg.json'))['source'].get('revision') or '')")
[ -n "$safetensors" ] && export VLLM_STATE=$D/vllm-state
drafter=$("$T" -I -c "
import json, os, sys
sys.path.insert(0, sys.argv[1]); from matched_spec import drafter_file
f = drafter_file(json.load(open(sys.argv[2])))   # the drafter / assistant GGUF the package ships (fetched into the card's parameter dir)
p = os.path.join(sys.argv[3], 'parameter', os.path.basename(f)) if f else ''
print(p if p and os.path.exists(p) else '')" "$here/skills/loom-benchmark/scripts/card" "$card/llm_config.json" "$card")
cardid=$("$T" -c "import json; print(json.load(open('$card/model.json'))['model']['id'])")
# the card's own build settings (model id, depth, context, pool): v0.2 kits have no --pool (the server sizes the pool); a v0.1 card's pool_capacity is not passed to them
POOLFLAG=0; grep -q -- '"--pool"' "$SK/loom-port-llm/starter/build.py" && POOLFLAG=1
cardargs=$("$T" -c "
import json; m=json.load(open('$card/model.json')); c=json.load(open('$card/llm_config.json')); ex=next(iter(m['presets'].values()))['execution']
a=['--context', str(ex['context_capacity']), *(['--pool', str(ex.get('pool_capacity') or ex['context_capacity'])] if '$POOLFLAG' == '1' else []), '--prefill', str(ex['prefill_capacity'])]
d=c.get('draft') or {}
a += (['--draft', d['kind'], '--draft-depth', str(d['depth'])] if d.get('kind') else [])
print(' '.join(a))")

slug_arg() {   # slug_arg BUILD_PY CARD_REPO -> "--slug <repo name>" when that build.py takes one (a kit without it has no slug-matched choices), else nothing
  grep -q -- '"--slug"' "$1" && echo "--slug ${2##*/}"
  return 0
}
hf_sub() {   # {hf:OWNER/REPO/FILE}, or immutable {hf:OWNER/REPO@40HEX/FILE#64HEXSHA256}; pinned cache bytes are always verified
  local t=$1 h
  for h in $(echo "$t" | grep -oE '\{hf:[^}]+\}' | sort -u); do
    local f=${h#\{hf:}; f=${f%\}}; local dst=$WORK/hf/$f r=${f%/*}
    if [[ "$r" == *@* || "$f" == *#* ]]; then
      local rev name digest actual part
      if [[ ! "$f" =~ ^([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)@([0-9a-f]{40})/([A-Za-z0-9][A-Za-z0-9_.-]*)#([0-9a-f]{64})$ ]]; then
        echo "FAIL: invalid pinned HF marker (need OWNER/REPO@40HEX/FILE#64HEXSHA256): $f" >&2; return 1
      fi
      r=${BASH_REMATCH[1]}; rev=${BASH_REMATCH[2]}; name=${BASH_REMATCH[3]}; digest=${BASH_REMATCH[4]}
      dst=$WORK/hf-pinned/$r/$rev/$name
      if [ ! -f "$dst" ]; then
        mkdir -p "${dst%/*}" && part=$(mktemp "$dst.part.XXXXXX") || return 1
        curl -fsSL --retry 3 ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
          "https://huggingface.co/$r/resolve/$rev/$name" -o "$part" || { rm -f "$part"; echo "FAIL: fetch $f" >&2; return 1; }
        actual=$(sha256sum "$part") || { rm -f "$part"; return 1; }
        if [ "${actual%% *}" != "$digest" ]; then
          rm -f "$part"; echo "FAIL: checksum mismatch for pinned HF download $f" >&2; return 1
        fi
        mv "$part" "$dst" || return 1
      else
        actual=$(sha256sum "$dst") || return 1
        [ "${actual%% *}" = "$digest" ] || { echo "FAIL: checksum mismatch for pinned HF cache $f" >&2; return 1; }
      fi
      t=${t//"$h"/$dst}; continue
    fi
    [ -f "$dst" ] || { mkdir -p "${dst%/*}" && curl -fsSL --retry 3 ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
      "https://huggingface.co/$(echo "$f" | cut -d/ -f1-2)/resolve/main/$(echo "$f" | cut -d/ -f3-)" -o "$dst.part" && mv "$dst.part" "$dst"; } || { echo "FAIL: fetch $f" >&2; return 1; }
    t=${t//"$h"/$dst}
  done
  printf '%s\n' "$t"
}
# 3. Packages: D/<variant>/pkg (+ D/<variant>/kit for kernel tests)
declare -A KIT KBV
kit_for() {   # kit_for VARIANT REF -> builds D/<variant>/kit from REF's starter (or this checkout); KBV[v]: REF's knowledge base (the build resolves its choices from it)
  local v=$1 ref=$2 PL=$SK/loom-port-llm
  if [ -n "$ref" ]; then
    git -C "$here" fetch -q origin "${ref#origin/}" 2>/dev/null
    rm -rf "$D/$v/src"; mkdir -p "$D/$v/src"
    git -C "$here" archive "$ref" skills/loom-port-llm skills/loom-kb skills/loom-setup | tar -x -C "$D/$v/src" || { echo "FAIL: no $ref"; return 1; }
    PL=$D/$v/src/skills/loom-port-llm; KBV[$v]=$D/$v/src/skills/loom-kb; echo "starter from $ref ($(git -C "$here" log -1 --format=%h:%s "$ref" | cut -c1-80))"
  fi
  if [ "$kit" != - ]; then "$PL/reference/apply_example.sh" "$kit" "$D/$v/kit" | tail -1 >/dev/null || return 1
  else cp -r "$PL/starter" "$D/$v/kit" && "$T" "$D/$v/kit/kit_source.py" write "$D/$v/kit" --skills-dir "$PL" > /dev/null || return 1; fi   # the copy's SDK pin (KIT_SOURCE)
  KIT[$v]=$D/$v/kit
}
for v in $variants; do
  pkg=$D/$v/pkg; mkdir -p "$D/$v"
  catv=$(qv "$v" "print(v.get('catalog') or '')")
  if [ -n "$catv" ]; then   # a catalog control (release-checklist: SDK regressions): an hrx-loom-library package fetched with the variant's own SDK
    [ "$(qv "$v" "print(v.get('sdk') or 'card')")" != card ] || { echo "FAIL: catalog variant $v needs an explicit \"sdk\""; exit 1; }
    sdk_root "$v" || exit 1
    cm=$(qv "$v" "print(v.get('model') or '')"); cp_=$(qv "$v" "print(v.get('preset') or '')")
    step "catalog $v: $cm${cp_:+ preset $cp_} from $catv (SDK root $SDKR)"
    rm -rf "$pkg"
    HF_HOME="$WORK/hf-home-catalog" HF_HUB_CACHE="$WORK/hf-home-catalog/hub" "$SDKR/.venv/bin/loom-fetch" --catalog="$catv" --model="$cm" ${cp_:+--preset="$cp_"} \
      --no-input --materialize=share --cache "$WORK/catalog-cache" --output="$pkg" > "$out/fetch_$v.log" 2>&1 || { tail "$out/fetch_$v.log"; echo "FAIL: loom-fetch $cm ($catv)"; exit 1; }
    # served under the job's model id (every step addresses \$cardid); the chosen preset first (the one serve_alone picks)
    "$T" - "$pkg/model.json" "$cardid" "$cp_" <<'EOF' || exit 1
import json, sys
m = json.load(open(sys.argv[1])); m["model"]["id"] = sys.argv[2]   # (the card schema has no other model field: the catalog id is in catalog_V.txt)
if sys.argv[3]:
    m["presets"] = {sys.argv[3]: m["presets"][sys.argv[3]], **{k: x for k, x in m["presets"].items() if k != sys.argv[3]}}
json.dump(m, open(sys.argv[1], "w"), indent=1)
EOF
    echo "$catv $cm" > "$out/catalog_$v.txt"
  elif [ "$(qv "$v" "print('build' in v)")" = True ]; then
    kit_for "$v" "$(qv "$v" "print(v.get('ref') or '')")" || exit 1
    extra=$(qv "$v" "print(' '.join(v.get('build_args') or []) + (f\" --context {v['context']}\" + (f\" --pool {v['context']}\" if '$POOLFLAG' == '1' else '') if v.get('context') else ''))")   # "context": N = the served context of a reduced-context model (later flags win)
    extra=$(hf_sub "$extra") || exit 1   # a build_args {hf:OWNER/REPO/FILE} (a Hub drafter the card does not ship)
    envs=$(qv "$v" "print(' '.join(f'{k}={x}' for k, x in (v.get('build') or {}).items()))")
    # the card's slug (its repo name): the knowledge base's model node of the package (exec.*, the kernel choices of its format) applies to a rebuild only with it
    slugarg=$(slug_arg "${KIT[$v]}/build.py" "$repo")
    step "build $v: env $envs; build.py $slugarg $cardargs $extra"
    # "config": generator-config overrides for this variant's rebuild (e.g. {"layers": 12}: a truncated model; the step
    # time minus the layer-proportional part is the per-step fixed cost; outputs are garbage, only timing counts)
    vcfg=$(qv "$v" "print(json.dumps(v.get('config') or {}))")
    vcfg=$(hf_sub "$vcfg") || exit 1   # {hf:...} in a config value: e.g. {"assistant": {"gguf": "{hf:...}"}}, a Gemma-4 assistant the card does not ship
    "$T" - "$D/cfg.json" "$D/$v/cfg.json" "$vcfg" <<'EOF2'
import json, sys
c = json.load(open(sys.argv[1])); c.update(json.loads(sys.argv[3])); json.dump(c, open(sys.argv[2], "w"), indent=1)
EOF2
    # the package's slug (the card's repo name) selects the knowledge base's slug-matched model nodes, as fleet/build_fleet.py's canonical build does:
    # without it a model's program-level choices (e.g. Qwen3.6-35B-A3B decode.attn.kv_once.d256 on rdna4) are missing and the programs differ from the published ones
    ( cd "${KIT[$v]}" && env $envs ${KBV[$v]:+LOOM_KB_ROOT=${KBV[$v]}} "$T" build.py "$D/$v/cfg.json" "$pkg" $slugarg --tokenizer "$card/tokenizer.json" --model-id "$cardid" $cardargs $extra ) > "$out/build_$v.log" 2>&1 \
      || { tail -30 "$out/build_$v.log"; echo "FAIL: build $v"; exit 1; }
    tail -3 "$out/build_$v.log"
  else
    kit_for "$v" "" || exit 1
    vcard=$card; vrepo=$(qv "$v" "print(v.get('card') or '')")   # variant "card": ORG/REPO = that card's published programs (e.g. loom-fun's, next to the staging card's)
    if [ -n "$vrepo" ]; then fetch_card "$vrepo"; vcard=$CARD_DIR; echo "${vrepo%@*}@$CARD_REV" > "$out/card_revision_$v.txt"; fi
    mkdir -p "$pkg"; cp -r "$vcard/program" "$vcard/model.json" "$vcard/tokenizer.json" "$vcard"/*.loom "$pkg/" 2>/dev/null
    ln -sfn "$vcard/parameter" "$pkg/parameter"
    "$T" - "$vcard" "$pkg" <<'EOF'
import json, os, sys
card, pkg = sys.argv[1:]
c = json.load(open(os.path.join(card, "llm_config.json")))
par = lambda f: os.path.join(card, "parameter", os.path.basename(f))
if (c.get("source") or {}).get("kind") == "safetensors":
    c["weights"] = {"format": "safetensors", "files": [par(f) for f in c["source"]["files"]]}
else:
    c["gguf"] = par(c["gguf"])
if c.get("vision") and c["vision"].get("mmproj"):   # a safetensors tower has no projector file
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
        m = re.match(r"command\.(program\.)?def\b.*\s@(\w+)\(", ln)
        if m:
            prog = m.group(2)
        out.append(ln)
        if re.match(r"\s*(kernel\.launch @|template\.apply<@)", ln) and any(pr in ("*", prog) and re.search(rx, ln) for pr, rx in rules):
            out.append(ln); n += 1
    open(p, "w").write("\n".join(out)); print(f"ablation: doubled {n} launches")
EOF
  "$T" - "$pkg/model.json" "$(qv "$v" "print(json.dumps(v.get('execution') or {}))")" "$cardid-$v" "$(qv "$v" "print(json.dumps(v.get('cap_pool')))")" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1])); ex = next(iter(m["presets"].values()))["execution"]
cap = json.loads(sys.argv[4])   # "cap_pool": N (R9700, fixed backing commits the pool eagerly): a pool above N shrinks to N, the context with it
for pr in m["presets"].values():   # the overrides apply to every preset (build.py --prefill-r9700 adds a second one)
    # a v0.2 package has no pool_capacity (the server derives the pool): an override of it is dropped there (and the served context is context_capacity)
    pr["execution"].update({k: x for k, x in json.loads(sys.argv[2]).items() if k != "pool_capacity" or "pool_capacity" in pr["execution"]})
    if cap and (pr["execution"].get("pool_capacity") or 0) > cap:
        pr["execution"]["pool_capacity"] = cap
        pr["execution"]["context_capacity"] = min(pr["execution"]["context_capacity"], cap)
json.dump(m, open(sys.argv[1], "w"), indent=1)
# a small-context copy with its own model id, for several packages in one loom-serve (ab steps)
m["model"]["id"] = sys.argv[3]; ex["context_capacity"] = 16384
if "pool_capacity" in ex:   # a v0.1 card; a v0.2 card has none (the server derives the pool from --memory_bytes)
    ex["pool_capacity"] = 32768
json.dump(m, open(sys.argv[1].replace("model.json", "model_ab.json"), "w"), indent=1)
print("execution", {k: ex[k] for k in ex if k in ("draft_depth", "rows", "prefill_capacity")})
EOF
  cp "$pkg/model.json" "$out/model_$v.json"; sha256sum "$pkg"/program/*.loom > "$out/programs_$v.sha256"
  "$T" -I -c "import sys; sys.path.insert(0, sys.argv[1]); from card_lib import programs_sha256; print(programs_sha256(sys.argv[2]))" \
    "$here/skills/loom-benchmark/scripts/card" "$pkg" > "$out/programs_sha256_$v.txt" 2>/dev/null   # the aggregate hash of THIS variant's programs (a built variant's: make_info.py --fetched-programs)
  # variant "programs_sha256": the programs this variant must serve (fleet/regen.py --programs production: the frozen production programs); any other
  # hash ends the job before anything is measured
  vwant=$(qv "$v" "print(v.get('programs_sha256') or '')")
  if [ -n "$vwant" ] && [ "$vwant" != "$(cat "$out/programs_sha256_$v.txt")" ]; then
    echo "FAIL: variant $v serves programs $(cat "$out/programs_sha256_$v.txt"), not the expected $vwant: nothing measured"; exit 1; fi
done

pkg_preset() {   # pkg_preset MODEL_JSON -> the preset this runner serves ("" for a card with one preset): the R9700's <first><suffix> when it has one, else the first
  "$T" -c "import json,sys; p=list(json.load(open(sys.argv[1]))['presets']); r=p[0] + sys.argv[3]; print((r if sys.argv[2] and r in p else p[0]) if len(p) > 1 else '')" "$1" "$RDEV" "$RSUF"
}
serve_alone() { # serve_alone VARIANT
  vllm_down; gufo_down
  for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
  "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1; VLLM_STATE=$D/vllm-state "$B/vllm_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
  local aw sa rp vmem; aw=$(qv "$1" "print(0 if v.get('active_wait') is False else 1)")
  sa=$(qv "$1" "print(' '.join(v.get('serve_args') or []))")   # "serve_args": extra loom-serve flags
  # a package with a `<first preset>-r9700` preset (build.py --prefill-r9700) serves it on the R9700, as the card's R9700 Quick start does
  rp=""; [ -n "$RDEV" ] && rp=$("$T" -c "import json,sys; p=list(json.load(open(sys.argv[1]))['presets']); print(p[0] + sys.argv[2] if p[0] + sys.argv[2] in p else '')" "$D/$1/pkg/model.json" "$RSUF")
  # (the job's "require_preset_r9700" covers the job's own card; a variant of another card, e.g. the old package of an A/B, has no R9700 preset unless it sets the key itself)
  if [ -z "$rp" ] && [ -n "$RDEV" ] && [ "$(qv "$1" "print(v['require_preset_$RUNNER'] if 'require_preset_$RUNNER' in v else (j.get('require_preset_$RUNNER') or '') if 'card' not in v else '')")" = True ]; then
    echo "FAIL: the card has no <first>$RSUF preset (its fit row has ${RUNNER}_execution): publish the package with it first"; return 1; fi
  # With no device-specific alternate, select the first preset explicitly on every runner for a multi-preset card.
  [ -z "$rp" ] && rp=$(pkg_preset "$D/$1/pkg/model.json")
  if [ -n "$rp" ] && [[ "$sa" != *--preset=* ]]; then sa="--preset=$rp $sa"; fi
  SERVED_PRESET=$(printf '%s\n' $sa | sed -n 's/^--preset=//p' | head -1)   # the preset this server serves (admit: probes ITS context)
  vmem=$(qv "$1" "print(v.get('memory_bytes') or '$mem')")   # variant "memory_bytes": "none" with --pool_backing=fixed
  # every GPU is served as its card's Quick start serves it: the registry device's card.serve_args (card_lib.GPUS "serve_args": --pool_backing=fixed on the
  # R9700 and on Strix Halo, no --memory_bytes; the R9700's elastic trim + compaction stalls the long-prefill rows, hrx-loom 0.2.0's elastic backing
  # costs Strix prefill, notes u1_rootcause), unless the variant brings its own serve_args/memory_bytes or is "elastic": true (--memory_bytes on any GPU)
  # (+ --parameter_mode=preload where the package's weights fit the GPU's host copy limit: card_lib.package_serve_args, from the package's own model.json.lock sizes)
  sdk_root "$1" || { echo "FAIL: SDK of variant $1"; return 1; }; local vroot=$SDKR
  gsa=$("$T" -I -c "import os, sys; sys.path.insert(0, sys.argv[1]); import card_lib; pin = card_lib.load_json(sys.argv[4]) if os.path.isfile(sys.argv[4]) else None; print(' '.join(card_lib.package_serve_args(sys.argv[2], sys.argv[3], pin)))" "$here/skills/loom-benchmark/scripts/card" "${RDEV:-strix-halo}" "$D/$1/pkg" "$vroot/loom-install.json") \
    || { echo "FAIL: card_lib.package_serve_args(${RDEV:-strix-halo}) (the device's card serve flags)"; return 1; }
  local memneed=""
  if [ -n "$gsa" ] && [[ "$sa" != *--pool_backing=* ]] && [ "$(qv "$1" "print(bool(v.get('elastic') or v.get('memory_bytes')))")" = False ]; then
    [[ "$sa" == *--parameter_mode=* ]] && gsa=$(printf '%s\n' $gsa | grep -v -- '^--parameter_mode=' | tr '\n' ' ')   # the variant's own parameter mode wins
    sa="$gsa $sa"
    memneed=$vmem   # the elastic budget bounds what fixed backing commits: serve.sh's memcheck still guards the host (Strix memory is host RAM)
    vmem=none
  fi
  export LOOM_ROOT=$vroot VENV=$vroot/.venv   # the variant's SDK serves it (and runs gates / kernel tests on it)
  [ -n "${PROFILE_SA:-}" ] && sa="$sa $PROFILE_SA"   # the profile step: --print_device_statistics
  # VRAM still held by the process that was just stopped (a vLLM / llama.cpp server, the reference fill, the last loom-serve) makes the load fail
  # (RESOURCE_EXHAUSTED / "model is unavailable"): wait until it is released, and log what was held (vram_log.txt)
  "$B/wait_vram.sh" "serve $1" 1024 180 "$out/vram_log.txt"
  if [ -n "${COLD_CAPACITY_DIR:-}" ]; then
    CAPACITY_MONITOR_DIR=$COLD_CAPACITY_DIR; mkdir -p "$CAPACITY_MONITOR_DIR"
    rm -f "$CAPACITY_MONITOR_DIR/monitor.ready" "$CAPACITY_MONITOR_DIR/monitor.stop"
    printf 'cold' > "$CAPACITY_MONITOR_DIR/phase"
    "$T" "$here/.github/scripts/cold_capacity.py" monitor --pid-file "$LOOM_SERVE_STATE/perf-$1.pid" \
      --model "$D/$1/pkg/model.json" --out "$CAPACITY_MONITOR_DIR" > "$CAPACITY_MONITOR_DIR/monitor.log" 2>&1 &
    CAPACITY_MONITOR_PID=$!
    for _ in $(seq 50); do [ -f "$CAPACITY_MONITOR_DIR/monitor.ready" ] && break; sleep 0.1; done
    [ -f "$CAPACITY_MONITOR_DIR/monitor.ready" ] || { echo "FAIL: cold capacity monitor has no baseline"; coldcapacity_stop; return 1; }
    printf '%s\n' "${rp:-}" > "$CAPACITY_MONITOR_DIR/preset.txt"
  fi
  mem_guard "serve $1" || return 1
  step "serve $1 (SDK root $vroot)"; LOOM_MEMNEED=$memneed LOOM_ACTIVE_WAIT=$aw "$SV/serve.sh" "perf-$1" "$PORT" amdgpu "$vmem" "$D/$1/pkg/model.json" ${sa:+-- $sa} > "$out/serve_$1.out" 2>&1; tail -2 "$out/serve_$1.out"
  grep -q '^FAILED' "$out/serve_$1.out" && { echo "REGISTRATION FAILED on $RUNNER for variant $1 (fast fail, see serve_$1.out):"; grep -m6 -E 'AMDGPU|error|ERROR' "$out/serve_$1.out" | cut -c1-300; return 1; }
  curl -sf "$url/v1/models" >/dev/null || return 1
  # "ready" is the compiled registration; the weights load on the FIRST request, and until then every request is refused. Wait for that
  # first answer before any step sends its own (a cold runner lost the first server of a job to it: func, gates, consistency failed
  # for a reason that is not theirs). A model that does not load in 5 minutes FAILS the step (no restart that would make it pass): the
  # first-load failure is a user-facing bug; its evidence (the server's answer and log) is kept as cold_first_serve_V.txt.
  if ! "$T" "$B/wait_model.py" "$url" --wait 300 | tee "$out/cold_first_serve_$1.txt"; then
    echo "serve $1: the model did not load (cold_first_serve_$1.txt)"; cp "$LOOM_SERVE_STATE/perf-$1.log" "$out/cold_first_serve_$1.log" 2>/dev/null; return 1
  fi
  rm -f "$out/cold_first_serve_$1.txt"
  [ -z "${COLD_CAPACITY_DIR:-}" ] || printf 'loaded' > "$COLD_CAPACITY_DIR/phase"
  return 0
}
gufo_bundle() {   # gufo_bundle -> exports GUFO_BUNDLE (the experiment's/job's "bundle" tarball, fetched once per runner), or fails
  [ -n "${GUFO_BUNDLE:-}" ] && [ -x "$GUFO_BUNDLE/run.sh" ] && return 0
  local u h d; u=$(q "print((j.get('bundle') or e.get('bundle') or {}).get('url') or '')"); h=$(q "print((j.get('bundle') or e.get('bundle') or {}).get('sha256') or '')")
  [ -n "$u" ] && [ -n "$h" ] || return 1
  d=$WORK/_gufo-bundle/${h:0:12}
  if [ ! -x "$d/run.sh" ]; then
    step "fetch gufo bundle $u"; rm -rf "$d"; mkdir -p "$d"
    curl -fsSL --retry 5 ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} "$u" -o "$d.tgz" && [ "$(sha256sum "$d.tgz" | cut -d' ' -f1)" = "$h" ] \
      && tar -xzf "$d.tgz" -C "$d" --strip-components=1 || { echo "FAIL: gufo bundle $u (fetch, sha256 or untar)"; rm -rf "$d" "$d.tgz"; return 1; }
    rm -f "$d.tgz"
  fi
  export GUFO_BUNDLE=$d
}
free_gpu() { for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done; "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1; }
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
  spec=$(hf_sub "$spec") || return 1   # {hf:OWNER/REPO/FILE}, fetched once per runner
  spec="$BK$MODS|$spec"   # the backend and its modifiers are part of what runs
  [ "$spec" = "$LSPEC" ] && curl -sf "$lurl/health" >/dev/null 2>&1 && return 0
  "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1; gufo_down
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
    [ -z "$lvkv" ] || kv[0]=$lvkv
  fi
  mem_guard "llama.cpp $BK" || return 1
  LLAMACPP_BACKEND=$BK LLAMACPP_DIR=$dir LLAMACPP_ARGS_FILE=$out/llamacpp_args$bkt.txt "$B/llamacpp_server.sh" "$gguf" "$LLAMA_PORT" $np "${kv[@]}" | tail -1
  sed -i -e "s#$WORK/hf/#hf:#g" -e "s#$card/#card:#g" "$out/llamacpp_args$bkt.txt" 2>/dev/null   # runner paths out of the recorded arguments
  [ "${PIPESTATUS[0]}" = 0 ] || { cp "$LLAMACPP_STATE/llamacpp-$LLAMA_PORT.log" "$out/llamacpp_start$bkt.log" 2>/dev/null; return 1; }
}
vllm_up() {   # vllm_up on|off: vLLM ROCm (vllm_server.sh --install: the pinned wheel in a venv) serving the card's checkpoint as an OpenAI server, out of the box
  # (--max-model-len and, for "on", the job's "vllm_spec" --speculative-config are the only arguments); one engine at a time on the GPU.
  # A server already up in the same mode is reused (vllm:N and vllm:N:long are one load: 27B takes minutes to load); any other engine stops it.
  [ -f "$vpid" ] && [ "${VLLM_MODE:-}" = "$1" ] && kill -0 "$(cat "$vpid")" 2>/dev/null && curl -sf "$vurl/v1/models" >/dev/null 2>&1 && return 0
  vllm_down; gufo_down
  # the old server must be gone (its port closed) before the next one starts: a server that is still shutting down answered the new one's
  # readiness probe and the c4 rows ran on it (with drafting on)
  for _ in $(seq 180); do curl -sf "$vurl/v1/models" >/dev/null 2>&1 || break; sleep 1; done
  curl -sf "$vurl/v1/models" >/dev/null 2>&1 && { echo "the previous vLLM server still answers on $VPORT"; return 1; }
  for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
  "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
  [ -x "$B/vllm_server.sh" ] || { echo "no vllm_server.sh in this checkout (vLLM reference engine, PR #875)"; return 1; }
  export VLLM_STATE=$out
  local vpy ml spec ck rev args
  vpy=$("$B/vllm_server.sh" --install | tail -1) || return 1
  [ -x "$vpy" ] || { echo "vLLM install failed: $vpy"; return 1; }
  eval "$("$B/vllm_server.sh" --env)"   # the venv's own environment (MPI libs, the HIP dev packages Quark's MXFP4 emulation JIT-compiles with): one place, vllm_server.sh
  ml=$(q "print(j.get('vllm_max_len') or j.get('served_context') or 65536)")
  spec=$(q "print(j.get('vllm_spec') or '')")
  read -r ck rev < <("$T" -c "import json; s=json.load(open('$card/llm_config.json'))['source']; print(s['repo'], s.get('revision') or 'main')")
  args="--max-model-len $ml"; [ "$1" = on ] && [ -n "$spec" ] && args="$args --speculative-config '$spec'"
  echo "$args" > "$out/vllm_args_$1.txt"; "$vpy" -c "import vllm; print(vllm.__version__.split('+')[0])" > "$out/vllm_version.txt" 2>/dev/null
  "$B/wait_vram.sh" "vllm serve ($1)" 1024 180 "$out/vram_log.txt"
  step "vllm serve $ck@${rev:0:8}: $args"
  eval env VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-0} ${VLLM_EXTRA_ENV:-} setsid nohup "$vpy" -m vllm.entrypoints.openai.api_server --model "$ck" --revision "$rev" \
    --served-model-name vllm-card --port "$VPORT" $args > "$out/vllm_serve_$1.log" 2>&1 &
  echo $! > "$vpid"; VLLM_MODE=$1
  for _ in $(seq 1800); do
    curl -sf "$vurl/v1/models" >/dev/null 2>&1 && {
      echo "vLLM ready"
      # what the engine can hold at once (a request longer than the KV cache never finishes) and what it spent its memory on
      { grep -m1 -o "GPU KV cache size: [0-9,]* tokens" "$out/vllm_serve_$1.log"; grep -m1 -o "Model loading took [0-9.]* GiB" "$out/vllm_serve_$1.log"
        grep -m1 -o "Available KV cache memory: [0-9.]* GiB" "$out/vllm_serve_$1.log"; } | tee "$out/vllm_kv_$1.txt"
      return 0; }
    kill -0 "$(cat "$vpid")" 2>/dev/null || { echo "vLLM died:"; tail -15 "$out/vllm_serve_$1.log" | cut -c1-250; return 1; }
    sleep 1
  done
  echo "vLLM did not come up in 30 min"; return 1
}
gufo_sha() {   # gufo_sha FILE -> its content sha256: an HF cache snapshot entry links to blobs/<sha256> (one level: a deduplicating
  # store may link that blob on to a file named by another hash), else hash the file (a download into $WORK/hf)
  local b; b=$(basename "$(readlink "$1" 2>/dev/null)")
  if [[ $b =~ ^[0-9a-f]{64}$ ]]; then echo "$b"; else sha256sum "$1" | cut -d' ' -f1; fi
}
gufo_path() {   # gufo_path OWNER/REPO[@REV]/FILE -> a local file: the user's HF cache (LOCAL_HF_CACHE=1, never copied), else a download into $WORK/hf
  local f=$1 r rev=main name; r=${f%/*}; name=${f##*/}; [[ $r == *@* ]] && { rev=${r#*@}; r=${r%@*}; }
  if [ "${LOCAL_HF_CACHE:-0}" = 1 ]; then
    local hub=${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${r//\//--} c
    for c in "$hub"/snapshots/$([ "$rev" = main ] && echo '*' || echo "$rev")*/"$name"; do [ -f "$c" ] && { echo "$c"; return 0; }; done
  fi
  local dst=$WORK/hf/$r/$rev/$name; [ -f "$dst" ] || { mkdir -p "${dst%/*}" && curl -fsSL --retry 3 ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} "https://huggingface.co/$r/resolve/$rev/$name" -o "$dst.part" && mv "$dst.part" "$dst"; } || return 1
  echo "$dst"
}
gufo_up() {   # gufo_up on|off: Gufo's OpenAI server (gufo serve llm) in its release container, serving the job's "gufo" model; on = its drafting
  # (--speculative dflash2 with the draft, 1 session), off = no drafting, 4 sessions (the streams4 rule). Reused while up in the same mode.
  [ "${GUFO_MODE:-}" = "$1" ] && curl -sf "$gurl/v1/models" >/dev/null 2>&1 && return 0
  gufo_down; vllm_down
  for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
  "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
  command -v docker >/dev/null || { echo "no docker: the Gufo steps run its release container"; return 1; }
  local img model draft spec ctx mf df args vols
  img=$(q "print((j.get('gufo') or e.get('gufo') or {}).get('image') or '')"); model=$(q "print((j.get('gufo') or e.get('gufo') or {}).get('model') or '')")
  draft=$(q "print((j.get('gufo') or e.get('gufo') or {}).get('draft') or '')"); spec=$(q "print((j.get('gufo') or e.get('gufo') or {}).get('speculative') or 'off')")
  ctx=$(q "print((j.get('gufo') or e.get('gufo') or {}).get('context') or 65536)")
  [ -n "$img" ] && [ -n "$model" ] || { echo "job/experiment \"gufo\" needs image and model"; return 1; }
  mf=$(gufo_path "$model") || { echo "FAIL: gufo model $model"; return 1; }
  args="--model $mf --served-model-name gufo-card --context $ctx"
  if [ "$1" = on ] && [ "$spec" != off ]; then
    df=$(gufo_path "$draft") || { echo "FAIL: gufo draft $draft"; return 1; }
    args="$args --speculative $spec --dflash-model $df --sessions 1"
  else args="$args --speculative off --sessions 4"; fi
  # the container sees only the resolved files (an HF cache snapshot entry is a symlink into blobs/), read-only
  local rmf rdf; rmf=$(readlink -f "$mf"); vols="-v $rmf:$rmf:ro"; [ -n "${df:-}" ] && { rdf=$(readlink -f "$df"); vols="$vols -v $rdf:$rdf:ro"; }
  local rargs=${args//$mf/$rmf}; [ -n "${df:-}" ] && rargs=${rargs//$df/$rdf}
  echo "$args" | sed -e "s#${HF_HOME:-$HOME/.cache/huggingface}/hub/#hfcache:#g" -e "s#$WORK/hf/#hf:#g" > "$out/gufo_args_$1.txt"
  { echo "image $img"; docker image inspect --format '{{index .RepoDigests 0}}' "${img%@*}" 2>/dev/null; docker run --rm "$img" gufo --version 2>/dev/null | head -3; } > "$out/gufo_version.txt"
  { echo "model $model sha256 $(gufo_sha "$mf")"; [ -n "${df:-}" ] && echo "draft $draft sha256 $(gufo_sha "$df")"; } > "$out/gufo_model.txt"
  "$B/wait_vram.sh" "gufo serve ($1)" 1024 180 "$out/vram_log.txt"
  mem_guard "gufo $1" || return 1
  step "gufo serve ($1): $(cat "$out/gufo_args_$1.txt")"
  docker rm -f "$GCONT" >/dev/null 2>&1
  docker run -d --name "$GCONT" --network host --device /dev/kfd --device /dev/dri --ulimit memlock=-1 \
    --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" $vols \
    "$img" gufo serve --host 127.0.0.1 --port "$GPORT" llm $rargs > "$out/gufo_cid_$1.txt" 2>&1 || { cat "$out/gufo_cid_$1.txt"; return 1; }
  GUFO_MODE=$1
  for _ in $(seq 900); do
    curl -sf "$gurl/v1/models" >/dev/null 2>&1 && { echo "gufo ready"; return 0; }
    [ "$(docker inspect -f "{{.State.Running}}" "$GCONT" 2>/dev/null)" = true ] || { docker logs "$GCONT" > "$out/gufo_serve_$1.log" 2>&1; echo "gufo died:"; tail -15 "$out/gufo_serve_$1.log"; return 1; }
    sleep 1
  done
  echo "gufo did not come up in 15 min"; return 1
}
TH=$(q "print(j.get('thinking') or 'off')")   # chat_template_kwargs.enable_thinking sent to both engines
echo "thinking: $TH (sent to both engines)"
declare -A res
if [ "${POWER_LOG:-0}" = 1 ]; then   # local evidence of the power mode: package W and GPU clock through every step
  hw=$(ls -d /sys/class/drm/card*/device/hwmon/hwmon* 2>/dev/null | while read -r h; do [ "$(cat "$h/name" 2>/dev/null)" = amdgpu ] && [ -e "$h/freq1_input" ] && echo "$h"; done | head -1)
  if [ -n "$hw" ]; then
    echo "$hw" > "$out/power_source.txt"
    ( while :; do p=$(cat "$hw/power1_input" 2>/dev/null || cat "$hw/power1_average" 2>/dev/null); f=$(cat "$hw/freq1_input" 2>/dev/null)
        echo "$(date +%s.%N) ${p:-0} ${f:-0}"; sleep 0.2; done > "$out/power.tsv" ) & PLOG_PID=$!
  else echo "POWER_LOG=1: no amdgpu hwmon with freq1_input" | tee "$out/power_source.txt"; fi
fi
queue=($steps); qi=0   # a queue, not a fixed list: the step extend:TO:HAVE:ROT appends sessions (ab_extend.py)
while [ "$qi" -lt "${#queue[@]}" ]; do
  [ -n "${sprev:-}" ] && printf '%s\t%s\t%s\n' "$sprev" "$st0" "$(date +%s.%N)" >> "$out/step_times.tsv"   # the previous step's span (also one that ended on an error)
  s=${queue[$qi]}; qi=$((qi + 1)); st0=$(date +%s.%N); sprev=$s
  IFS=: read -r kind p1 p2 p3 <<< "$s"
  BK=rocm; bkt=; MODS=; case $kind in *@*) bkt=-${kind#*@}; BK=${kind#*@}; MODS=${BK#"${BK%%.*}"}; BK=${BK%%.*}; kind=${kind%@*};; esac   # llama@vulkan: the other backend's build
  [ "$BK" = rocm ] || [ "$vtag" != - ] || { echo "$s: set llamacpp_vulkan"; res[$s]=ERROR; continue; }
  [ "${p2:-}" = - ] && p2=""
  case $kind in
    ab)
      abr=(); for v in $variants; do sdk_root "$v" >/dev/null 2>&1; abr+=("${SDKR:-?}"); done
      abroots=$(printf '%s\n' "${abr[@]}" | sort -u | wc -l)
      [ "$abroots" -le 1 ] || { echo "ERROR: step $s hosts all variants in ONE loom-serve, but the variants use $abroots different SDKs (variant \"sdk\"): use perf:/c4: steps (one server per variant)"; res[$s]="ERROR: variants with different SDKs cannot share the ab server"; echo "RESULT $job $s: ${res[$s]}"; continue; }
      for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
      "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
      ms=(); ids=(); for v in $variants; do ms+=("$D/$v/pkg/model_ab.json"); ids+=("$cardid-$v"); done
      nv=${#ms[@]}; abmem=$(( 11000000000 * nv + 4000000000 )); [ -n "$RDEV" ] && [ "$abmem" -gt "$mem" ] && abmem=$mem
      "$B/wait_vram.sh" "ab serve" 1024 180 "$out/vram_log.txt"
      step "ab: one loom-serve with $nv packages ($abmem bytes)"
      "$SV/serve.sh" abserve "$PORT" amdgpu "$abmem" "${ms[@]}" | tail -2
      "$T" "$here/.github/scripts/ab_decode.py" "$url" "${ids[@]}" --rounds "$p1" --think "${p2:-$TH}" --log "$out/ab.jsonl" 2>&1 | tee "$out/ab.log"
      res[$s]=$(grep -c SUMMARY "$out/ab.log")" variants measured"
      "$SV/stop.sh" abserve >/dev/null 2>&1;;
    extend)   # extend:TO:HAVE:ROT: methodology "sequential" (2+2 sessions -> 6+6 where a ratio is within 2 points of its reject threshold)
      more=$("$T" "$here/.github/scripts/ab_extend.py" "$out" --to "$p1" --have "$p2" --rot "${p3:-0}" 2> "$out/extend.log"); cat "$out/extend.log"
      queue+=($more); res[$s]="$(echo $more | wc -w) more sessions"
      echo "$(echo "$more" | tr '\n' ' ')" > "$out/extended_steps.txt";;
    perf)
      serve_alone "$p1" || { res[$s]=ERROR; continue; }
      step "perf $p1 x $p2 ${p3:-} (thinking $TH)"
      # the server's raw /metrics before and after every perf.py run (metrics_V_NNN.txt: the cumulative speculative counters the drafted-token
      # acceptance is derived from; kept raw so a telemetry change of an SDK is visible in the artifact)
      mi=$(ls "$out"/metrics_"$p1"_*.txt 2>/dev/null | wc -l); curl -s "$url/metrics" > "$out/metrics_${p1}_$(printf %03d "$mi").txt" 2>/dev/null
      for i in $(seq 1 "$p2"); do
        (cd "$out" && "$T" "$B/perf.py" "$url" "$p1" --model "$cardid" --no-campaign --thinking "$TH" --repeat-rows "$rrows" --log "perf_$p1.jsonl" $([ "${p3:-}" = ctx ] && echo --decode-ctx 16000,124000 || echo ${p3:+--long}) | tail -1)
        mi=$((mi + 1)); curl -s "$url/metrics" > "$out/metrics_${p1}_$(printf %03d "$mi").txt" 2>/dev/null
      done
      echo "{\"session_end\": \"$p1\"}" >> "$out/perf_$p1.jsonl"   # one serve session ends (strix_ab_compare.py: per-session medians)
      "$T" "$here/.github/scripts/ab_decode.py" "$url" "$cardid" --rounds 2 --think "$TH" --log "$out/tps_$p1.jsonl" | grep SUMMARY
      res[$s]=$("$T" -c "
import json,statistics as st; r=[json.loads(l) for l in open('$out/perf_$p1.jsonl')]
k='decode_tps_124k' if '${p3:-}' == 'ctx' else 'decode_tps_32k' if '${p3:-}' else 'decode_tps'
print('$p1', k, [x.get(k) for x in r if x.get(k)], 'median', round(st.median(v), 2) if (v := [x[k] for x in r if x.get(k)]) else 'n/a (no row served: beyond the served context)')");;
    profile|trace|observe)   # profile:V[:SCENARIOS] (comma list of probe,pf2k,pf8k,dec; default all but probe): V served alone per scenario with --print_device_statistics (the HAL's
               # per-function dispatch statistics, printed at shutdown); one session per scenario, so each table holds only that scenario's requests
      capture_kind=$kind   # trace uses the same requests without device profiling (observer-effect control)
      capture_failed=0
      capture_scenarios=${p2:-pf2k,pf8k,dec}
      [ "$capture_kind" = profile ] || capture_scenarios=${p2:-dec}
      for sc in $(echo "$capture_scenarios" | tr ',' ' '); do
        profile_flags=""
        [ "$capture_kind" != profile ] || profile_flags="--device_profiling_mode=dispatch-events,executable-metadata --device_profiling_flush_interval_ms=5 --print_device_statistics"
        PROFILE_SA="$profile_flags" serve_alone "$p1" || { capture_failed=1; break; }
        step "$capture_kind $p1 $sc"
        capture_request_ok=1
        observer_pid=; observer_stop="$out/observe_${p1}_$sc.stop"
        if [ "$capture_kind" = observe ]; then
          rm -f "$observer_stop"
          "$T" "$B/thread_snapshot.py" "$(cat "$LOOM_SERVE_STATE/perf-$p1.pid")" --duration 120 --interval 0.2 --stacks \
            --stop-file "$observer_stop" --output "$out/observe_${p1}_$sc.threads.jsonl" > "$out/observe_${p1}_$sc.sampler.log" 2>&1 &
          observer_pid=$!
        fi
        ( cd "$B" && "$T" profile_requests.py "$url" "$cardid" "$sc" ) > "$out/${capture_kind}_${p1}_$sc.req.json" 2> "$out/${capture_kind}_${p1}_$sc.req.err" || capture_request_ok=0; tail -c 600 "$out/${capture_kind}_${p1}_$sc.req.json"; tail -3 "$out/${capture_kind}_${p1}_$sc.req.err"
        if [ -n "$observer_pid" ]; then
          touch "$observer_stop"
          wait "$observer_pid" || capture_request_ok=0
          "$T" -c 'import json,sys; rows=[json.loads(line) for line in open(sys.argv[1])]; assert any(row.get("event")=="sample" for row in rows), "no process/thread samples"' \
            "$out/observe_${p1}_$sc.threads.jsonl" || capture_request_ok=0
        fi
        "$SV/stop.sh" "perf-$p1" | tail -1; sleep 2   # a graceful stop: the statistics are printed at shutdown
        cp "$LOOM_SERVE_STATE/perf-$p1.log" "$out/${capture_kind}_${p1}_$sc.serve.log" 2>/dev/null
        [ "$capture_request_ok" = 1 ] || { capture_failed=1; break; }
        [ "$capture_kind" = profile ] || continue
        grep -c "dispatch " "$out/${capture_kind}_${p1}_$sc.serve.log" | sed "s/^/dispatch rows: /"; grep -A6 "IREE HAL device statistics" "$out/${capture_kind}_${p1}_$sc.serve.log" | head -8
        # verify passes of a decode scenario: decode_tokens - draft_accepted_inputs (both SDKs count them; 0.3.0 serving an ABI 1 program reports no proposals)
        [ "$sc" = probe ] || { steps_n=0; [[ $sc == dec* ]] && steps_n=$(python3 -c "
import json; r=json.loads(open('$out/${capture_kind}_${p1}_$sc.req.json').read().splitlines()[-1]); d=r.get('draft') or {}; w=r.get('warmup_draft') or {}
n=lambda k: d.get(k, 0) + w.get(k, 0)
print(round(n('decode_tokens') - n('draft_accepted_inputs')) if n('decode_tokens') else round(n('draft_proposed_tokens') / ${depth:-3}))" 2>/dev/null || echo 0)
          "$T" "$B/profile_stats.py" "$out/${capture_kind}_${p1}_$sc.serve.log" "$out/${capture_kind}_${p1}_$sc.req.json" --steps "$steps_n" --families "$B/profile_families.json" \
            --md "$out/${capture_kind}_${p1}_$sc.md" --json "$out/${capture_kind}_${p1}_$sc.json" > "$out/${capture_kind}_${p1}_$sc.table.txt" 2>&1; head -14 "$out/${capture_kind}_${p1}_$sc.table.txt" | cut -c1-200; }
      done
      if [ "$capture_failed" = 1 ]; then
        res[$s]="ERROR: diagnostic request failed"
      elif [ "$capture_kind" = profile ]; then
        res[$s]="$(ls "$out"/profile_${p1}_*.md 2>/dev/null | wc -l) tables"
      else
        res[$s]="$(ls "$out"/${capture_kind}_${p1}_*.req.json 2>/dev/null | wc -l) $capture_kind captures"
      fi;;
    probe)   # probe:V: stall_probe.py (when does a prefill stall? TTFT + GPU clocks per request)
      serve_alone "$p1" || { res[$s]=ERROR; continue; }
      step "stall probe $p1"
      # telemetry during the probe: /logz/stream (records, incl. the --heartbeat_ms snapshots) and /v1/observe; /metrics every 50 ms
      curl -sN -H "X-Loom-Model: $cardid" "$url/logz/stream" > "$out/logz_$p1.txt" 2>&1 & lp1=$!
      ( while true; do echo "T $(date +%s.%N)"; curl -s "$url/v1/sessions"; echo; curl -s "$url/v1/requests"; echo; sleep 0.1; done > "$out/sessions_$p1.txt" 2>&1 ) & lp2=$!
      lp3=$lp2
      "$T" "$here/.github/scripts/stall_probe.py" "$url" "$cardid" "$out/probe_$p1.jsonl" > "$out/probe_$p1.log" 2>&1; tail -3 "$out/probe_$p1.log" | cut -c1-200
      kill $lp1 $lp2 $lp3 2>/dev/null
      cp "$LOOM_SERVE_STATE/perf-$p1.log" "$out/servelog_$p1.log" 2>/dev/null   # heartbeat telemetry (--heartbeat_ms=N in serve_args)
      res[$s]="$(wc -l < "$out/probe_$p1.jsonl") requests";;
    c4)
      serve_alone "$p1" || { res[$s]=ERROR; continue; }
      step "c4 $p1"
      cn=${p2:-4}; cf=c4_$p1$([ "$cn" = 4 ] || echo "_n$cn").txt   # c4:V[:N] N concurrent streams (default 4)
      for i in 1 2 3; do (cd "$out" && "$T" "$B/c4_steady.py" "$url" "$cn" --model "$cardid" --thinking "$TH" | tail -1 | tee -a "$cf"); done
      echo "{\"session_end\": \"$p1\"}" >> "$out/$cf"   # one serve session ends
      res[$s]=$("$T" -c "import json,statistics as st; print('$p1 c4$([ "$cn" = 4 ] || echo " n$cn") median', st.median(r['agg_tps'] for l in open('$out/$cf') if 'agg_tps' in (r := json.loads(l))))");;
    llama)   # llama:N[:long|-][:N_MAX]
      llama_up "${p3:-}" || { res[$s]=ERROR; continue; }
      nm=$(echo "$LSPEC" | grep -oE "n-max [0-9]+" | grep -oE "[0-9]+"); nm=${nm:-0}; lf=perf_llamacpp${bkt}_n$nm.jsonl
      cp "$out/llamacpp_args$bkt.txt" "$out/llamacpp_args$bkt.n$nm.txt" 2>/dev/null   # the arguments of THIS n-max (the plain file is overwritten per server start)
      step "llama.cpp$bkt n-max $nm perf x $p1 ${p2:-} (thinking $TH)"
      for i in $(seq 1 "$p1"); do
        (cd "$out" && "$T" "$B/perf.py" "$lurl" "llamacpp${bkt}-n$nm" --kind llama --no-campaign --thinking "$TH" --repeat-rows "$rrows" --log "$lf" ${p2:+--long} | tail -1)
        echo "{\"session_end\": \"llamacpp$bkt-n$nm\"}" >> "$out/$lf"   # each llama.cpp run is a session (the server stays up): make_info.py perf takes the median of the runs
      done
      [ -z "${p2:-}" ] && "$T" "$here/.github/scripts/llama_decode.py" "$lurl" 2 --think "$TH" --log "$out/tps_llamacpp${bkt}_n$nm.jsonl"
      res[$s]=$("$T" -c "
import json,statistics as st; r=[json.loads(l) for l in open('$out/$lf')]
k='decode_tps_32k' if '${p2:-}' else 'decode_tps'
print('llamacpp$bkt n-max $nm', k, [x.get(k) for x in r if x.get(k)], 'median', round(st.median(v), 2) if (v := [x[k] for x in r if x.get(k)]) else 'n/a (no row served: beyond the served context)')");;
    vllm)   # vllm:N[:long]: the vLLM baseline of a safetensors card, N serve-session-less runs on one server (the card's drafting config), perf.py --kind vllm
      vllm_up on || { res[$s]=ERROR; echo "vllm: $(tail -2 "$out/vllm_serve_on.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)" > "$out/vllm_error.txt"; continue; }
      vkv=$(grep -o "cache size: [0-9][0-9,]* tokens" "$out/vllm_kv_on.txt" 2>/dev/null | grep -o "[0-9][0-9,]*" | tr -d ',')
      if [ -n "${p2:-}" ] && [ -n "$vkv" ] && [ "$vkv" -lt 40000 ]; then   # the long rows (8k/16k/32k documents, decode after 32k) can never finish: vLLM would wait for KV blocks forever
        echo "vLLM's KV cache holds $vkv tokens: the 8k/16k/32k rows are not measured" | tee "$out/vllm_note.txt"
        res[$s]="SKIP (vLLM KV cache $vkv tokens)"; continue
      fi
      step "vllm perf x $p1 ${p2:-} (thinking $TH)"
      for i in $(seq 1 "$p1"); do
        (cd "$out" && timeout 3600 "$T" "$B/perf.py" "$vurl" vllm --kind vllm --model vllm-card --no-campaign --thinking "$TH" --repeat-rows "$rrows" --log "perf_vllm_n$p1.jsonl" ${p2:+--long} | tail -1)
        echo "{\"session_end\": \"vllm\"}" >> "$out/perf_vllm_n$p1.jsonl"
      done
      res[$s]=$("$T" -c "
import json,statistics as st; r=[json.loads(l) for l in open('$out/perf_vllm_n$p1.jsonl') if l.startswith('{\"label')]
k='decode_tps_32k' if '${p2:-}' else 'decode_tps'
print('vllm', k, [x.get(k) for x in r if x.get(k)], 'median', round(st.median(v), 2) if (v := [x[k] for x in r if x.get(k)]) else 'n/a')");;
    vllmc4)   # vllmc4:N: 4 streams on vLLM without drafting (the same load-time exception as Loom's c4), N runs of c4_steady.py
      vllm_up off || { res[$s]=ERROR; echo "vllm: $(tail -2 "$out/vllm_serve_off.log" 2>/dev/null | tr '\n' ' ' | cut -c1-300)" > "$out/vllm_error.txt"; continue; }
      step "vllm c4"
      cn=${p1:-3}
      for i in $(seq 1 "$cn"); do (cd "$out" && timeout 1800 "$T" "$B/c4_steady.py" "$vurl" 4 --model vllm-card --thinking "$TH" 2> "c4_vllm_err.txt" | tail -1 | tee -a "c4_vllm_n$cn.txt"); done
      # 4 streams that never decode together (the engine serves fewer at a time: KV / state blocks): the engine's own message is the card's note
      grep -q agg_tps "$out/c4_vllm_n$cn.txt" 2>/dev/null || { echo "vLLM could not run 4 concurrent streams: $(tail -1 "$out/c4_vllm_err.txt" | cut -c1-200)" > "$out/vllm_note_c4.txt"; res[$s]="no 4-stream rate (vllm_note_c4.txt)"; vllm_down; continue; }
      echo "{\"session_end\": \"vllm\"}" >> "$out/c4_vllm_n$cn.txt"
      res[$s]=$("$T" -c "import json,statistics as st; print('vllm c4 median', st.median(r['agg_tps'] for l in open('$out/c4_vllm_n$cn.txt') if 'agg_tps' in (r := json.loads(l))))");;
    gufoperf)   # gufoperf:N[:long]: Gufo serving its own model with its drafting, perf.py --kind gufo N times on one server (each run a session, as llama.cpp's)
      gufo_up on || { res[$s]=ERROR; continue; }
      step "gufo perf x $p1 ${p2:-} (thinking $TH)"
      for i in $(seq 1 "$p1"); do
        (cd "$out" && timeout 3600 "$T" "$B/perf.py" "$gurl" gufo --kind gufo --model gufo-card --no-campaign --thinking "$TH" --repeat-rows "$rrows" --log "perf_gufo.jsonl" ${p2:+--long} | tail -1)
        echo "{\"session_end\": \"gufo\"}" >> "$out/perf_gufo.jsonl"
        curl -sf "$gurl/metrics" > "$out/gufo_metrics_on_$(date +%s).txt" 2>/dev/null
      done
      docker logs "$GCONT" > "$out/gufo_serve_on.log" 2>&1
      res[$s]=$("$T" -c "
import json,statistics as st; r=[json.loads(l) for l in open('$out/perf_gufo.jsonl') if l.startswith('{\"label')]
k='decode_tps_32k' if '${p2:-}' else 'decode_tps'
print('gufo', k, [x.get(k) for x in r if x.get(k)], 'median', round(st.median(v), 2) if (v := [x[k] for x in r if x.get(k)]) else 'n/a')");;
    gufoc4)   # gufoc4[:N]: 4 streams on Gufo without drafting (4 sessions), N (default 3) runs of c4_steady.py
      gufo_up off || { res[$s]=ERROR; continue; }
      step "gufo c4"
      for i in $(seq 1 "${p1:-3}"); do (cd "$out" && timeout 1800 "$T" "$B/c4_steady.py" "$gurl" 4 --model gufo-card --thinking "$TH" | tail -1 | tee -a "c4_gufo.txt"); done
      echo "{\"session_end\": \"gufo\"}" >> "$out/c4_gufo.txt"
      docker logs "$GCONT" > "$out/gufo_serve_off.log" 2>&1
      res[$s]=$("$T" -c "import json,statistics as st; print('gufo c4 median', st.median(r['agg_tps'] for l in open('$out/c4_gufo.txt') if 'agg_tps' in (r := json.loads(l))))");;
    llamac4)   # llamac4[:N_MAX]
      llama_up "${p1:-}" || { res[$s]=ERROR; continue; }
      nm=$(echo "$LSPEC" | grep -oE "n-max [0-9]+" | grep -oE "[0-9]+"); nm=${nm:-0}   # 0: no drafting
      step "llama.cpp$bkt n-max $nm c4"
      for i in 1 2 3; do (cd "$out" && "$T" "$B/c4_steady.py" "$lurl" 4 --thinking "$TH" | tail -1 | tee -a "c4_llamacpp${bkt}_n$nm.txt"); done
      echo "{\"session_end\": \"llamacpp$bkt-n$nm\"}" >> "$out/c4_llamacpp${bkt}_n$nm.txt"
      res[$s]=$("$T" -c "import json,statistics as st; print('llamacpp$bkt n-max $nm c4 median', st.median(r['agg_tps'] for l in open('$out/c4_llamacpp${bkt}_n$nm.txt') if 'agg_tps' in (r := json.loads(l))))");;
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
      # make_kernel_tests.py reads the package under test next to its config: a variant without its own cfg.json gets a copy, else the first
      # D/*/pkg found (another variant's, e.g. a kit build beside a served card) would shape this variant's cases
      [ -f "$D/$p1/cfg.json" ] || cp "$D/cfg.json" "$D/$p1/cfg.json"
      kenv=$(qv "$p1" "print(' '.join(f'{k}={x}' for k, x in (v.get('build') or {}).items()))")   # the variant's build env (e.g. GEMM_F16O) shapes the test cases too
      # the choices the knowledge base resolved for the rebuilt package (resolved.env of its llm_config.json: a model node's rules, not only the variant's build env) shape the cases too
      renv=$("$T" -c "import json, sys; print(' '.join(f'{k}={x}' for k, x in ((json.load(open(sys.argv[1])).get('resolved') or {}).get('env') or {}).items() if x is not None))" "$D/$p1/pkg/llm_config.json" 2>/dev/null)
      ( cd "${KIT[$p1]}" && env $renv $kenv LLM_DRAFT="$("$T" -c "import json, sys; print(json.dumps(json.load(open(sys.argv[1])).get('draft')))" "$D/$p1/pkg/llm_config.json")" \
          "$T" make_kernel_tests.py "$([ -f "$D/$p1/cfg.json" ] && echo "$D/$p1/cfg.json" || echo "$D/cfg.json")" "$kt" ) > "$out/ktest_make_$p1.log" 2>&1 || { tail "$out/ktest_make_$p1.log"; res[$s]=ERROR; continue; }
      "${KIT[$p1]}/run_kernel_tests.sh" "$D/$p1/pkg" "$kt" > "$out/ktest_$p1.log" 2>&1; rc=$?
      grep -v "^PASS" "$out/ktest_$p1.log" | tail -15
      tot=$(grep -oE "^[0-9]+/[0-9]+ passed" "$out/ktest_$p1.log" | tail -1)
      res[$s]="$([ $rc = 0 ] && [ -n "$tot" ] && [ "${tot%%/*}" = "$(echo "${tot#*/}" | cut -d' ' -f1)" ] && echo PASS || echo FAIL) ($tot; $(grep -c '^SKIP' "$out/ktest_$p1.log") skipped)";;
    vtest)   # vtest:V[:HxW]  the image tower on this GPU: the encoder kernel cases vs NumPy, then the whole encoder on the package's own weights vs ref_vit.py
      python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('vision') else 1)" "$D/$p1/pkg/llm_config.json" || { res[$s]="SKIP (no image input)"; continue; }
      for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
      step "vision tests $p1"; vt=$D/$p1/vtest; rm -rf "$vt"; mkdir -p "$vt"; vh=${p2:-320x224}
      export VISION_TEST_DIR=$vt VENV=${VENV:-$root/.venv} PY=$T; [ -n "$RDEV" ] && export VISION_TEST_RDNA4=1   # gfx12: the kernel cases launch the K_rdna4 twins
      ( cd "${KIT[$p1]}/vision/tests" && bash run_tests.sh ) > "$out/vtest_$p1.log" 2>&1; rc=$?
      # the whole encoder: a safetensors package reads the checkpoint's tensors (CFG.json CHECKPOINT), a GGUF package its mmproj
      eargs=$("$T" -c "import json,sys; c=json.load(open(sys.argv[1])); v=c['vision']; print(sys.argv[1] + ' ' + c['weights']['files'][0] if v.get('source') == 'safetensors' else v['mmproj'])" "$D/$p1/pkg/llm_config.json")
      ( cd "${KIT[$p1]}/vision" && "$T" test_encoder.py --nowrite $eargs "${vh%x*}" "${vh#*x}" ) > "$out/vtest_encoder_$p1.log" 2>&1
      tail -3 "$out/vtest_encoder_$p1.log"
      grep -q "^ENCODER PASS" "$out/vtest_encoder_$p1.log" || rc=1
      res[$s]="$([ $rc = 0 ] && echo PASS || echo FAIL) ($(grep -c '^PASS' "$out/vtest_$p1.log") kernel cases passed, $(grep -c '^FAIL\|^COMPILE' "$out/vtest_$p1.log") failed; encoder $vh: $(grep -o '^ENCODER [A-Z]*' "$out/vtest_encoder_$p1.log" | tail -1))";;
    logits)   # logits:V[:N_PROMPTS]  (loom-skills#736): V's package through the Gufo bundle with the logits capability on, checked on this GPU
      free_gpu   # the GPU is the bundle's (and the undrafted llama.cpp reference's) alone
      gufo_bundle || { echo '{"verdict": "NOT_SUPPORTED", "note": "no gufo bundle on this job (set bundle or GUFO_BUNDLE)", "gates": {}}' > "$out/logits_$p1.json"; res[$s]="not supported (no gufo bundle)"; continue; }
      LLAMACPP_DIR=${LLAMACPP_DIR:-$("$B/fetch_llamacpp.sh" "$WORK/llamacpp" "$tag" | tail -1)}; export LLAMACPP_DIR
      ( unset LLAMA_SPEC; "$B/llamacpp_server.sh" "$gguf" "$LLAMA_PORT" 1 8192 | tail -1 ) || { res[$s]=ERROR; continue; }
      step "logits $p1 (bundle $GUFO_BUNDLE)"
      "$T" "$here/.github/scripts/logits_check.py" "$D/$p1/pkg" "$out/logits_$p1.json" --bundle "$GUFO_BUNDLE" --llama "$lurl" --llama-stop "$B/llamacpp_server.sh --stop $LLAMA_PORT" \
        --llama-start "env -u LLAMA_SPEC '$B/llamacpp_server.sh' '$gguf' $LLAMA_PORT 1 8192" --prompts "${p2:-6}" --preset "$(pkg_preset "$D/$p1/pkg/model.json")" \
        > "$out/logits_$p1.log" 2>&1; rc=$?
      tail -3 "$out/logits_$p1.log" | cut -c1-300
      "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
      res[$s]=$(grep -h "^LOGITS" "$out/logits_$p1.log" | tail -1 | cut -c1-300); [ -n "${res[$s]}" ] || res[$s]="ERROR rc=$rc";;
    gufo)   # gufo:V  Gufo conformance on V's package (gufo run_conformance.sh: --loom-logits=on, conformance.py --logits, logits_check.py)
      free_gpu
      gufo_bundle || { echo '{"verdict": "NOT_SUPPORTED", "note": "no gufo bundle on this job (set bundle or GUFO_BUNDLE)", "gates": {}}' > "$out/conformance_$p1.json"; res[$s]="not supported (no gufo bundle)"; continue; }
      step "gufo conformance $p1 (bundle $GUFO_BUNDLE)"
      gp=$(pkg_preset "$D/$p1/pkg/model.json")
      EXTRA_GUFO_ARGS="${gp:+--loom-preset $gp}" BUNDLE=$GUFO_BUNDLE "$GUFO_BUNDLE/validation/devlab/run_conformance.sh" "$D/$p1/pkg" "$out/gufo_$p1" 8181 > "$out/gufo_$p1.log" 2>&1; rc=$?
      tail -4 "$out/gufo_$p1.log" | cut -c1-300
      res[$s]=$("$T" "$here/.github/scripts/gufo_gate.py" "$out/gufo_$p1" "$out/conformance_$p1.json" "$rc" 2>&1 | tail -1 | cut -c1-300);;
    smoke)   # smoke:V  one served chat request (the iterate preset's whole functional check: build, serve, answer; func:V is the full list)
      serve_alone "$p1" || { res[$s]=ERROR; continue; }
      step "smoke $p1"
      "$T" - "$url" "$cardid" > "$out/smoke_$p1.log" 2>&1 <<'PY'
import json, sys, urllib.request
url, model = sys.argv[1:3]
req = urllib.request.Request(url + "/v1/chat/completions", json.dumps({"model": model, "max_tokens": 16, "temperature": 0,
        "messages": [{"role": "user", "content": "Reply with one short sentence about the sea."}]}).encode(), {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=300))
m = r["choices"][0]["message"]
t = (m.get("content") or m.get("reasoning_content") or m.get("reasoning") or "").strip()
print("SMOKE", "PASS" if t else "FAIL", repr(t[:80]))
sys.exit(0 if t else 1)
PY
      res[$s]=$(tail -1 "$out/smoke_$p1.log" | cut -c1-200); [ -n "${res[$s]}" ] || res[$s]=ERROR;;
    func)
      serve_alone "$p1" || { res[$s]=ERROR; continue; }
      step "functional checks $p1"
      curl -fsSL --retry 2 "$base/info.json" -o "$out/info_card.json" 2>/dev/null || rm -f "$out/info_card.json"
      fa=(); { grep -q '"vision"' "$card/llm_config.json" || grep -q '"vision"' "$D/$p1/pkg/llm_config.json"; } 2>/dev/null && fa+=(--vision)   # the card's, or the rebuilt package's image input
      "$T" "$here/.github/scripts/r9700_func.py" "$url" "$cardid" "$out/func_$p1.json" ${fa[@]+"${fa[@]}"} $([ -f "$out/info_card.json" ] && echo --info "$out/info_card.json") 2>&1 | tee "$out/func_$p1.log"
      res[$s]=$(tail -1 "$out/func_$p1.log");;
    vllmdet)   # vllmdet:V : vLLM's run-to-run determinism: the dev golden prompt set recorded twice (salts a, b), token for token; the card's golden2 question
      [ -n "$safetensors" ] || { res[$s]="ERROR: not a safetensors package"; continue; }
      free_gpu   # nothing else on the GPU: only the reference runs
      export VLLM_VENV=$WORK/vllm-venv-${VLLM_VERSION:-0.18.0} LLAMA_CACHE=$D/$p1/bench/det_cache LLAMA_CACHE_BUILD=vllm-${VLLM_VERSION:-0.18.0}
      step "vLLM determinism $p1"; t1=$(date +%s)
      "$B/vllm_server.sh" "$gguf" "$LLAMA_PORT" 4096 > "$out/vllm_start.log" 2>&1; tail -2 "$out/vllm_start.log"
      ( cd "$D/$p1" && mkdir -p bench && cd bench && for salt in a b; do
          "$T" "$B/golden.py" "$D/$p1/pkg" "det_$salt.json" --llama "$lurl" --short 20 --long 1000,2000 --salt $salt || exit 1; done
        "$T" - <<'PY'
import json
a, b = (json.load(open(f"det_{s}.json")) for s in "ab")
same = sum(x["ids"] == y["ids"] for x, y in zip(a, b)); print(f"DETERMINISM {same}/{len(a)} prompts identical across two recordings")
PY
      ) > "$out/vllmdet_$p1.log" 2>&1; tail -1 "$out/vllmdet_$p1.log"
      cp "$D/$p1"/bench/det_*.json "$out/" 2>/dev/null; VLLM_STATE=$D/vllm-state "$B/vllm_server.sh" --stop "$LLAMA_PORT"
      res[$s]="$(grep -h DETERMINISM "$out/vllmdet_$p1.log" | tail -1) ($(( $(date +%s) - t1 )) s incl. install+load)";;
    admit)   # admit:V[:TOKENS]: ctx_admit.py on V served as configured (its execution context/pool): is a request of TOKENS (default ~ the served context) admitted? (the card's served context on the device)
      serve_alone "$p1" || { res[$s]=ERROR; continue; }
      step "context admission $p1 ${p2:-}"
      (cd "$out" && "$T" "$B/ctx_admit.py" "$D/$p1/pkg" "$url" ${p2:+--tokens "$p2"} ${SERVED_PRESET:+--preset "$SERVED_PRESET"} --out "admit_${p1}${p2:+_$p2}.json" 2>&1 | tee "admit_${p1}${p2:+_$p2}.txt" | tail -3)
      res[$s]=$(tail -1 "$out/admit_${p1}${p2:+_$p2}.txt" | cut -c1-200);;
    fullctx)   # fullctx:V[:TOKENS]: one request of ~the served context that must finish (ctx_admit.py --complete): the card's context is a measured one
      serve_alone "$p1" || { res[$s]=ERROR; continue; }
      step "full-context request $p1 ${p2:-}"
      (cd "$out" && "$T" "$B/ctx_admit.py" "$D/$p1/pkg" "$url" ${p2:+--tokens "$p2"} --complete 2>&1 | tee "fullctx_${p1}${p2:+_$p2}.txt" | tail -3)
      res[$s]=$(tail -1 "$out/fullctx_${p1}${p2:+_$p2}.txt" | cut -c1-200);;
    coldcapacity)   # explicitly selected experimental step; all ordinary serve/gate/perf paths are unchanged
      capdir="$out/coldcapacity_$p1"
      if COLD_CAPACITY_DIR="$capdir" serve_alone "$p1"; then
        cappreset=$(cat "$capdir/preset.txt")
        "$T" "$here/.github/scripts/cold_capacity.py" probe "$D/$p1/pkg" "$url" \
          ${cappreset:+--preset "$cappreset"} --out "$capdir" > "$capdir/probe.log" 2>&1
        caprc=$?
      else caprc=1; fi
      coldcapacity_stop
      if [ "$caprc" = 0 ] && "$T" -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(not(d["identity_ok"] and d["amd_cards_observed"] and d["driver_used_counters_observed"] and d["stopped_by_marker"]))' "$capdir/memory_summary.json"; then
        res[$s]=PASS
      else res[$s]=ERROR; fi
      tail -1 "$capdir/probe.log" 2>/dev/null;;
    gates)
      serve_alone "$p1" || { res[$s]=ERROR; continue; }
      if [ -n "$safetensors" ]; then   # the reference is vLLM (gates.sh picks it from source.kind); VLLM_VERSION / VLLM_ROCM_VARIANT / VLLM_EXTRA_ENV come from the job's "env"
        export VLLM_VENV=$WORK/vllm-venv-${VLLM_VERSION:-0.18.0}   # one install per runner and version, reused by later jobs
      else
        LLAMACPP_DIR=${LLAMACPP_DIR:-$("$B/fetch_llamacpp.sh" "$WORK/llamacpp" "$tag" | tail -1)}; export LLAMACPP_DIR LLAMA_CACHE_BUILD=$tag
      fi
      export LOOM_KIT=${KIT[$p1]}
      # The reference outputs are invariant inputs: they are restored before gates.sh runs, so a run whose requests are all cached never loads the
      # reference engine (vLLM was 72 % of a typical job). The committed store .github/devlab/reference_cache/<org>__<repo>__<ENGINE>.tgz
      # (skills/loom-benchmark/scripts/reference_cache.py), automatically: ENGINE = vllm-<VERSION> for a safetensors package (its checkpoint's repo),
      # llama-<BUILD> for a GGUF card (the card's repo). Keys: engine version + checkpoint revision / GGUF + request (prompt set, sampling): exact hits only.
      #  - "reference_cache": "PATH.tgz" (a job field, relative to the checkout) in addition: references an earlier job recorded (its gates_V/llama_cache.tgz);
      #    "reference_cache": "none" skips both (a deliberate cold recording).
      # A miss is recorded by this job (its llama_cache.tgz: `reference_cache.py merge` adds it to the store).
      rcache=$(q "print(j.get('reference_cache') or '')"); nref0=0; nref_hit=0
      mkdir -p "$D/$p1/bench"
      if [ "$rcache" != none ]; then
        if [ -n "$safetensors" ]; then
          rrepo=$("$T" -c "import json,sys; print((json.load(open(sys.argv[1])).get('source') or {}).get('repo', ''))" "$D/$p1/pkg/llm_config.json"); reng=vllm-${VLLM_VERSION:-0.18.0}
        else
          rrepo=$(q "print((j.get('card') or e.get('card') or '').split('@')[0])"); reng=llama-$tag
        fi
        [ -n "$rrepo" ] && nref_hit=$("$T" "$B/reference_cache.py" restore --repo "$rrepo" --engine "$reng" --into "$D/$p1/bench" 2>/dev/null || echo 0)
        echo "reference cache: $nref_hit entries restored from the committed store ($rrepo, $reng)"
      fi
      if [ -n "$rcache" ] && [ "$rcache" != none ]; then
        tar -xzf "$here/$rcache" -C "$D/$p1/bench" && echo "reference cache restored from $rcache"
      fi
      nref0=$("$T" "$B/reference_cache.py" count "$D/$p1/bench" 2>/dev/null || echo 0)
      step "gates $p1 LEVEL=$p2"
      ( unset LLAMA_SPEC; FORCE=1 LEVEL=$p2 "$B/gates.sh" "$D/$p1/pkg" "$url" "$gguf" "$D/$p1/bench" "$LLAMA_PORT" ) > "$out/gates_$p1.out" 2>&1; rc=$?
      mkdir -p "$out/gates_$p1"; cp "$D/$p1"/bench/gates.json "$D/$p1"/bench/gates_*.log "$out/gates_$p1/" 2>/dev/null
      if [ -n "$safetensors" ]; then   # the card's reference statement, install/load/memory logs of the vLLM reference
        cp "$D/$p1"/bench/reference.json "$D/vllm-state"/vllm-*.{log,mem,json} "$out/gates_$p1/" 2>/dev/null
      fi
      # the recorded references (reusable: keyed by engine version + checkpoint / GGUF + request): `reference_cache.py merge` adds them to the committed store
      tar -C "$D/$p1/bench" -czf "$out/gates_$p1/llama_cache.tgz" llama_cache $([ -f "$D/$p1/bench/reference.json" ] && echo reference.json) 2>/dev/null
      nref1=$("$T" "$B/reference_cache.py" count "$D/$p1/bench" 2>/dev/null || echo 0)
      vl=n/a; [ -n "$safetensors" ] && { vl=no; [ -f "$D/vllm-state/vllm-$LLAMA_PORT.log" ] && vl=yes; }   # vllm-PORT.log exists only when vllm_server.sh started the engine
      echo "references: $nref0 cached before gates, $(( nref1 - nref0 )) recorded in this job, vLLM loaded: $vl" | tee "$out/gates_$p1/reference_cache.txt"
      res[$s]=$(grep -h "^GATES" "$out/gates_$p1.out" | tail -1 | cut -c1-300); [ -n "${res[$s]}" ] || res[$s]="rc=$rc"
      [ -f "$out/gates_$p1/reference_cache.txt" ] && res[$s]="${res[$s]} [$(cat "$out/gates_$p1/reference_cache.txt")]";;
    *) echo "unknown step $s"; res[$s]=ERROR;;
  esac
  echo "RESULT $job $s: ${res[$s]}"
done
[ -n "${sprev:-}" ] && printf '%s\t%s\t%s\n' "$sprev" "$st0" "$(date +%s.%N)" >> "$out/step_times.tsv"
[ -n "${PLOG_PID:-}" ] && { kill "$PLOG_PID" 2>/dev/null; "$T" "$here/.github/scripts/power_steps.py" "$out" ${POWER_BAND:+--band $POWER_BAND} > "$out/power_steps.txt" 2>&1; cat "$out/power_steps.txt"; }
for s in "${queue[@]}"; do echo "$s	${res[$s]:-}"; done > "$out/results.tsv"
"$T" - "$job" "$out" <<'EOF'
import json, os, sys
name, out = sys.argv[1:]
rows = [l.rstrip("\n").split("\t", 1) for l in open(os.path.join(out, "results.tsv"))]
json.dump({"job": name, "runner": open(os.path.join(out, "runner.txt")).read().strip(), "results": dict(rows)},
          open(os.path.join(out, "summary.json"), "w"), indent=1)
EOF
echo "SUMMARY $job"; cat "$out/results.tsv"
