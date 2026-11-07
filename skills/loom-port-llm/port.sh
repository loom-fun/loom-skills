#!/bin/bash
# One command from a Hugging Face GGUF repo to a served, tested Loom package.
# usage: [LOOM_ROOT=~/loom] port.sh ORG/REPO[:FILE.gguf] WORKDIR [PORT] [MEMORY_GIB] [MODEL_ID]
#   positional: PORT (default 8160), MEMORY_GIB (default auto, see build.py --memory-gib), e.g. port.sh org/repo ~/work/m 8260 40
#   0. starts the end-to-end clock (WORKDIR/.e2e_start, unless it exists): qualify.sh sizes the
#      optimization budget from it
#   1. the GGUF (FILE, else pick_gguf.py: Q4_K_M by default, from the HF cache (or the repo that has it cached), formats the starter decodes
#      and that fits twice in GPU memory; header-only probe) and the base model's tokenizer.json
#   1b. find_reference.py: the best CATEGORY A loom-fun package (same architecture and family first; cards made by
#      the current skill only) -> WORKDIR/reference.json; its reference example kit is applied when it has one.
#      REFERENCE=off skips it, REFERENCE=loom-fun/REPO picks one
#   2. inspect_gguf.py: config + chat block derived from the template and verified (stops on any FAIL); then the
#      reference fills what inspect left open (cfg["reference"] records it; tune.py takes its drafting as a prior)
#   3. build.py at the native context, fitted to MEMORY_GIB
#   4. serve.sh (state in WORKDIR/state), kernel unit tests, smoke test (chat, thinking off, a tool call)
# Prints PORT_RESULT {...} with the package, model id, context and memory_bytes; the server keeps
# running (stop: LOOM_SERVE_STATE=WORKDIR/state stop.sh port). Exit codes: 2 download, 3 chat
# template needs a hand-written block, 4 build/serve, 5 kernel tests, 6 smoke test.
# Then run loom-benchmark/scripts/qualify.sh (or gates.sh). After hand-fixing cfg.json (exit 3), resume
# with FROM=build port.sh ORG/REPO WORKDIR [PORT] [MEMORY_GIB] [MODEL_ID] (stages 3-6; KIT=DIR uses your kit copy).
set -u
spec=$1; work=$(mkdir -p "$2" && cd "$2" && pwd); port=${3:-8160}; mem_gib=${4:-auto}; model_id=${5:-}
here=$(cd "$(dirname "$0")" && pwd); K=${KIT:-$here/starter}; S=$here/../loom-setup/scripts
root=${LOOM_ROOT:-$HOME/loom}
bin=$root/.tools/bin; [ -d "$bin" ] || bin=$root/.tools/Scripts
T=$bin/python; venv=$root/.venv; hf=$venv/bin/hf; [ -x "$hf" ] || hf=$venv/Scripts/hf
export LOOM_SERVE_STATE=$work/state PYTHONDONTWRITEBYTECODE=1
repo=${spec%%:*}; file=""; [ "$spec" != "$repo" ] && file=${spec#*:}
cd "$work"
[ -s .e2e_start ] || date +%s > .e2e_start
step() { echo "== $* ($(date +%T))"; }
ref=off; refpick=""   # 1b sets them (FROM=build keeps the reference cfg.json already records)

# FROM=build: resume after hand-fixing cfg.json (exit 3): reuse WORKDIR/gguf, tok and cfg.json, run 3-6.
if [ "${FROM:-}" = build ]; then
  gguf=$(ls "$work"/gguf/*.gguf | head -1); [ -f cfg.json ] && [ -f tok/tokenizer.json ] && [ -n "$gguf" ] || { echo "FROM=build needs WORKDIR/cfg.json, tok/tokenizer.json and gguf/"; exit 2; }
  [ -n "$model_id" ] || model_id=$("$T" -c "import json;print(json.load(open('cfg.json')).get('model_id') or '')")
  [ -n "$model_id" ] || { echo "FROM=build: pass MODEL_ID (5th argument)"; exit 2; }
else
step "1/6 download"
# A file already in the HF cache (any revision) is used as is; otherwise it is downloaded into the cache.
pick=$("$T" -I "$K/pick_gguf.py" "$repo" ${file:+--file "$file"}) || { echo "STOP: no usable GGUF in $repo (see above); pass ORG/REPO:FILE.gguf"; exit 2; }
file=$(printf '%s' "$pick" | cut -f1); cached=$(printf '%s' "$pick" | cut -f2)
if [ -n "$cached" ]; then
  echo "using the cached $file ($cached)"
else
  # Into the shared HF cache (reused by later runs and by llama.cpp tools), symlinked into WORKDIR/gguf.
  "$hf" download "$repo" "$file" > download.log 2>&1; echo "hf exit $?" >> download.log
  cached=$(grep -v '^hf exit' download.log | grep -o '/[^ ]*'"$(basename "$file")" | tail -1)
fi
[ -f "$cached" ] || { echo "download of $repo/$file failed (download.log):"; tail -5 download.log 2>/dev/null; exit 2; }
mkdir -p gguf && ln -sf "$cached" "gguf/$(basename "$file")"
gguf=$work/gguf/$(basename "$file")
# Tokenizer: the base model named in the GGUF, else the repo name without -GGUF.
tokrepos=$("$T" - "$gguf" "$repo" <<'EOF'
import re, sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1]); out = []
for k, f in r.fields.items():
    if re.fullmatch(r"general\.base_model\.\d+\.repo_url", k):
        out.append(f.contents().split("huggingface.co/")[-1].strip("/"))
repo = sys.argv[2]; base = re.sub(r"[-_.]gguf$", "", repo, flags=re.I)
out += [base, repo]
org, name = base.split("/", 1)
out += [f"Qwen/{name}", f"{org}/{name}-Instruct"]
# Not named in the GGUF and not at the guessed repos (flint: RecursiveSI/flint, not unsloth/flint):
# ask the Hub for repos of the same name, most downloaded first (non-GGUF repos carry tokenizer.json).
try:
    import json, urllib.request
    q = urllib.request.urlopen(f"https://huggingface.co/api/models?search={name}&sort=downloads&limit=10", timeout=20)
    out += [m["id"] for m in json.load(q) if "gguf" not in m["id"].lower()]
except Exception:
    pass
print(" ".join(dict.fromkeys(out)))
EOF
)
mkdir -p tok
for tr in $tokrepos; do
  "$hf" download "$tr" tokenizer.json --local-dir tok >/dev/null 2>&1 && { echo "tokenizer from $tr"; break; }
done
[ -f tok/tokenizer.json ] || { echo "no tokenizer.json found in: $tokrepos (download it from the base model into $work/tok)"; exit 2; }
[ -n "$model_id" ] || model_id=$(echo "$tokrepos" | cut -d' ' -f1)

# A published, gate-passing port of the same architecture (or one sharing building blocks) is the starting point:
# its kit (reference example), its reviewed config, its chat block and its measured drafting choice. Category A only.
ref=$(echo "${REFERENCE:-auto}" | tr 'A-Z' 'a-z'); refpick=""
case "$ref" in off|0|no|none) ref=off;; auto|on|1|yes) ;; *) refpick=${REFERENCE};; esac
if [ "$ref" != off ]; then
  step "1b reference (category A loom-fun packages)"
  "$T" -I "$here/find_reference.py" rank "$gguf" --repo "$repo" -o reference.json > reference.log 2>&1 || echo "reference lookup failed (reference.log); continuing from scratch"
  grep -v "^REFERENCE_RESULT" reference.log | head -8
  ex=$("$T" -c "
import json, sys; r = json.load(open('reference.json')); p = '$refpick'
x = next((x for x in r['ranked'] if (p and x['repo'] == p) or (not p and x['arch'] == r['target'].get('arch'))), None)
# No category A package of this architecture: an example kit of it still is a gate-passing port (examples.json)
ex = (x or {}).get('kit_example') or ('' if x or p else next((k for k, e in json.load(open('$here/reference/examples.json')).items()
                                                               if isinstance(e, dict) and e.get('arch') == r['target'].get('arch')), ''))
print(ex)" 2>/dev/null)
  if [ -z "${KIT:-}" ] && [ -n "$ex" ] && [ -d "$here/reference/$ex-example" ]; then
    rm -rf kit; "$here/reference/apply_example.sh" "$ex" "$work/kit" | tail -1 && K=$work/kit && export LOOM_KIT=$K
  fi
fi

step "2/6 inspect (chat block derived from the template and verified)"
"$T" -I "$K/inspect_gguf.py" "$gguf" cfg.json 2>&1 | tee inspect.log | grep -v "^chat OK"
irc=${PIPESTATUS[0]}
# The starter's coverage checks (formats, head_dim) don't apply to an example kit (gemma4: bf16, head_dim 512): its own
# build and kernel tests decide, as after apply_example.sh + FROM=build.
hd="\\|WARNING: no kernels\\|WARNING: head_dim"; [ "$K" = "$here/starter" ] || hd=""
# The template's chat block could not be derived: a same-architecture reference's block, verified against THIS
# model's template (chat_probe.py --verify), replaces the hand-written block when every probe matches.
if [ "$irc" = 0 ] && [ "$ref" != off ] && [ -s reference.json ] && [ -f cfg.json ] && grep -q "^chat FAIL\|not derivable\|WARNING: no chat block" inspect.log \
   && ! grep -q "Traceback$hd" inspect.log; then
  "$T" -I "$here/find_reference.py" apply cfg.json reference.json ${refpick:+--pick "$refpick"} --chat-failed | tee -a reference.log
  if "$T" -I "$K/chat_probe.py" "$gguf" --verify cfg.json > chat_verify.log 2>&1; then
    echo "chat block from the reference: verified against this template ($(grep -c '^OK' chat_verify.log) probes OK)"
    sed -i 's/^chat FAIL/chat (replaced by the reference) FAIL/; s/not derivable/not derivable (reference block verified)/; s/WARNING: no chat block/no chat block (reference block verified)/' inspect.log
  else
    echo "the reference's chat block does not match this template (chat_verify.log)"
  fi
fi
# ("not derivable automatically": the sed above marks a verified reference block between those words)
if [ "$irc" != 0 ] || grep -q "^chat FAIL\|not derivable automatically\|WARNING: no chat block$hd\|WARNING: unsupported layout\|Traceback" inspect.log; then
  arch=$("$T" -c "import json;print(json.load(open('cfg.json')).get('arch',''))" 2>/dev/null)
  for ex in $("$T" -c "import json; print(' '.join(k for k, v in json.load(open('$here/reference/examples.json')).items() if isinstance(v, dict) and v.get('arch') == '$arch'))" 2>/dev/null); do
    echo "HINT: reference/$ex-example is a working port of arch '$arch': $here/reference/apply_example.sh $ex kit cfg.json (merges the reviewed config too), then FROM=build KIT=kit $0 $spec $work $port $mem_gib $model_id"
  done
  echo "STOP: see inspect.log; fix cfg.json (chat block / formats) by hand as loom-port-llm/SKILL.md describes, then rerun build.py"; exit 3
fi

fi  # FROM=build

# Image input: when the repo ships a vision projector (mmproj GGUF) of a family starter/vision has an
# encoder plan for, cfg.json gets a "vision" block and the package an image component. VISION=0 skips it;
# MMPROJ=file picks a projector. An unsupported family prints the reason and the port stays text-only
# (reference/vision.md: adding a family).
if [ "${VISION:-1}" != 0 ] && ! "$T" -c "import json,sys; sys.exit(0 if json.load(open('cfg.json')).get('vision') else 1)"; then
  step "2b vision"
  HF=$hf "$T" "$K/vision/add_vision.py" cfg.json "$repo" ${MMPROJ:-} | tee vision.log | grep -v "^$"
fi

# Speculative decoding is on by default: an MTP/NextN block in the GGUF (inspect records cfg["mtp"]) or a DFlash
# drafter published for the base model (found and validated here) is used; draft/tune.py later keeps whichever
# decodes fastest, or none. SPEC=off disables it; DRAFTER_REPO=org/name[:file] picks a drafter.
spec_off=$(echo "${SPEC:-on}" | tr 'A-Z' 'a-z')
if [ "$spec_off" != off ] && [ "$spec_off" != 0 ] && ! "$T" -c "import json,sys; sys.exit(0 if json.load(open('cfg.json')).get('drafter') else 1)"; then
  step "2c drafter"
  "$T" "$K/draft/find_drafter.py" cfg.json "$repo" --download ${DRAFTER_REPO:+--repo "$DRAFTER_REPO"} > drafter.log 2>&1 || true
  grep "^DRAFTER_RESULT" drafter.log
fi

if [ "$ref" != off ] && [ -s reference.json ] && [ -f cfg.json ] && ! "$T" -c "import json,sys; sys.exit(0 if json.load(open('cfg.json')).get('reference') else 1)"; then
  "$T" -I "$here/find_reference.py" apply cfg.json reference.json ${refpick:+--pick "$refpick"} | tee -a reference.log
fi

step "3/6 build"
"$T" "$K/build.py" cfg.json pkg --tokenizer tok/tokenizer.json --model-id "$model_id" --memory-gib "$mem_gib" | tee build.log
[ "${PIPESTATUS[0]}" = 0 ] || { echo "STOP: build.py failed (build.log)"; exit 4; }
mem=$(grep -o -- "--memory_bytes=[0-9]*" build.log | cut -d= -f2)
ctx=$(grep -o "^context [0-9]*" build.log | cut -d' ' -f2)

step "4/6 serve"
"$S/stop.sh" port >/dev/null 2>&1
VENV=$venv "$S/serve.sh" port "$port" amdgpu "$mem" pkg/model.json || exit 4

step "5/6 kernel unit tests"
"$T" -I "$K/make_kernel_tests.py" cfg.json ktest >/dev/null && VENV=$venv "$K/run_kernel_tests.sh" pkg ktest | tee ktest.log | tail -3
grep -q "^FAIL\|Traceback" ktest.log && { echo "STOP: kernel tests failed (ktest.log); see the bring-up debugging order in SKILL.md"; exit 5; }
vmm=$("$T" -c "import json; print((json.load(open('cfg.json')).get('vision') or {}).get('mmproj', ''))")
if [ -n "$vmm" ]; then   # vision kernels vs NumPy, then the whole encoder vs ref_vit.py at a small size
  side=$("$T" -c "import sys; sys.path.insert(0, '$K/vision'); import encoder, preprocess as P; c = encoder.config('$vmm'); print(4 * c['patch'] * P.FAMILY[c['family']]['merge'])")
  VENV=$venv "$K/vision/tests/run_tests.sh" "$vmm" "$side" "$side" > vtest.log 2>&1; rc=$?; grep -c "^PASS" vtest.log | sed 's/^/vision kernel tests passed: /'; tail -1 vtest.log
  [ $rc = 0 ] || { echo "STOP: vision tests failed (vtest.log); see reference/vision.md"; exit 5; }
fi

step "6/6 smoke test"
if [ -n "$vmm" ]; then   # one image: the component compiles, rows reach the text model, the answer is sane
  "$T" - "$here/../loom-benchmark/scripts" "http://127.0.0.1:$port" <<'EOF' || exit 6
import sys; sys.path.insert(0, sys.argv[1])
import requests, vision_golden as V
im, _ = V.builtin_images()["shapes"]
model = requests.get(sys.argv[2] + "/v1/models", timeout=30).json()["data"][0]["id"]
r = V.request(sys.argv[2], model, V.data_url(im), "What shapes and colors are in this image?", 256, False)
said = (r.get("reasoning", "") + " " + r.get("content", "")).lower()
ok = "error" not in r and any(w in said for w in ("red", "blue", "green", "circle", "square", "triangle"))
print("image:", "OK" if ok else "FAIL", repr(r.get("content") or r.get("reasoning") or r.get("error"))[:150], "prompt tokens", r.get("prompt_tokens"))
sys.exit(0 if ok else 1)
EOF
fi
"$T" - "$work/pkg" "http://127.0.0.1:$port" "$here/../loom-benchmark/scripts/card" <<'EOF' || exit 6
import json, sys, requests
pkg, url = sys.argv[1], sys.argv[2]
sys.path.insert(0, sys.argv[3]); import card_lib   # tool_calling(): the test gates.sh and the card use
cfg = json.load(open(pkg + "/llm_config.json")); model = cfg["model_id"]; chat = cfg["chat"]
off = {"chat_template_kwargs": {"enable_thinking": False}} if chat.get("thinking") == "toggle" else {}
post = lambda body: requests.post(url + "/v1/chat/completions", json={"model": model, **body}, timeout=600).json()
ok = True
r = post({"messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}], "max_tokens": 600, "temperature": 0})
m = r["choices"][0]["message"]; good = "paris" in ((m.get("content") or "") + (m.get("reasoning_content") or "")).lower()
print("chat:", "OK" if good else "FAIL", repr((m.get("content") or "")[:80])); ok &= good
if card_lib.tool_calling(chat):
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    r = post({"messages": [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}], "tools": tools,
              "max_tokens": 600, "temperature": 0, **off})
    calls = r["choices"][0]["message"].get("tool_calls") or []
    good = bool(calls) and calls[0]["function"]["name"] == "get_weather" and "paris" in calls[0]["function"]["arguments"].lower()
    print("tool call:", "OK" if good else "FAIL", json.dumps(calls)[:200]); ok &= good
else:
    print("tool call: skipped (chat-only block: the template ignores tools)")
n = requests.post(url + "/v1/chat/completions/input_tokens", json={"model": model, "messages": [{"role": "user", "content": "hi"}]}).json()
print("input_tokens:", n)
sys.exit(0 if ok else 1)
EOF
# Speculative decoding must not change a single token: the same package with drafting off is the reference
# (loom-benchmark/scripts/spec_check.py). One model is loaded at a time.
draft=$("$T" -c "import json; print(json.load(open('pkg/llm_config.json')).get('draft') or {})" 2>/dev/null)
if [ "$draft" != "{}" ] && [ -n "$draft" ]; then
  SC=$here/../loom-benchmark/scripts/spec_check.py
  "$T" "$SC" record "http://127.0.0.1:$port" "$model_id" spec_draft.jsonl && "$T" "$SC" card pkg >/dev/null || exit 6
  curl -s "http://127.0.0.1:$port/metrics" > spec_metrics.txt
  "$S/stop.sh" port >/dev/null 2>&1
  VENV=$venv "$S/serve.sh" port "$port" amdgpu "$mem" pkg/model_target.json >/dev/null || exit 4
  "$T" "$SC" record "http://127.0.0.1:$port" "$model_id" spec_target.jsonl || exit 6
  "$S/stop.sh" port >/dev/null 2>&1
  VENV=$venv "$S/serve.sh" port "$port" amdgpu "$mem" pkg/model.json >/dev/null || exit 4
  "$T" "$SC" compare spec_target.jsonl spec_draft.jsonl | tail -3
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "STOP: drafting changed the output (spec_check); see starter/draft/README.md"; exit 6; }
fi
echo "PORT_RESULT $(printf '{"package": "%s", "model_id": "%s", "context": %s, "memory_bytes": %s, "port": %s, "gguf": "%s", "draft": %s}' "$work/pkg" "$model_id" "${ctx:-0}" "$mem" "$port" "$gguf" "$("$T" -c "import json; print(json.dumps(json.load(open('pkg/llm_config.json')).get('draft')))")")"
