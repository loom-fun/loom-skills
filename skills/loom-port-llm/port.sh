#!/bin/bash
# One command from a Hugging Face GGUF repo to a served, tested Loom package.
# usage: [LOOM_ROOT=~/loom] port.sh ORG/REPO[:FILE.gguf] WORKDIR [PORT] [MEMORY_GIB] [MODEL_ID]
#   positional: PORT (default 8160), MEMORY_GIB (default 56), e.g. port.sh org/repo ~/work/m 8260 40
#   0. starts the end-to-end clock (WORKDIR/.e2e_start, unless it exists): qualify.sh sizes the
#      optimization budget from it
#   1. download the GGUF (FILE, else pick_gguf.py: the best quant whose formats the starter supports
#      and that fits twice in GPU memory; header-only probe) and the base model's tokenizer.json
#   2. inspect_gguf.py: config + chat block derived from the template and verified (stops on any FAIL)
#   3. build.py at the native context, fitted to MEMORY_GIB
#   4. serve.sh (state in WORKDIR/state), kernel unit tests, smoke test (chat, thinking off, a tool call)
# Prints PORT_RESULT {...} with the package, model id, context and memory_bytes; the server keeps
# running (stop: LOOM_SERVE_STATE=WORKDIR/state stop.sh port). Exit codes: 2 download, 3 chat
# template needs a hand-written block, 4 build/serve, 5 kernel tests, 6 smoke test.
# Then run loom-benchmark/scripts/qualify.sh (or gates.sh). After hand-fixing cfg.json (exit 3), resume
# with FROM=build port.sh ORG/REPO WORKDIR [PORT] [MEMORY_GIB] [MODEL_ID] (stages 3-6; KIT=DIR uses your kit copy).
set -u
spec=$1; work=$(mkdir -p "$2" && cd "$2" && pwd); port=${3:-8160}; mem_gib=${4:-56}; model_id=${5:-}
here=$(cd "$(dirname "$0")" && pwd); K=${KIT:-$here/starter}; S=$here/../loom-setup/scripts
root=${LOOM_ROOT:-$HOME/loom}
bin=$root/.tools/bin; [ -d "$bin" ] || bin=$root/.tools/Scripts
T=$bin/python; venv=$root/.venv; hf=$venv/bin/hf; [ -x "$hf" ] || hf=$venv/Scripts/hf
export LOOM_SERVE_STATE=$work/state PYTHONDONTWRITEBYTECODE=1
repo=${spec%%:*}; file=""; [ "$spec" != "$repo" ] && file=${spec#*:}
cd "$work"
[ -s .e2e_start ] || date +%s > .e2e_start
step() { echo "== $* ($(date +%T))"; }

# FROM=build: resume after hand-fixing cfg.json (exit 3): reuse WORKDIR/gguf, tok and cfg.json, run 3-6.
if [ "${FROM:-}" = build ]; then
  gguf=$(ls "$work"/gguf/*.gguf | head -1); [ -f cfg.json ] && [ -f tok/tokenizer.json ] && [ -n "$gguf" ] || { echo "FROM=build needs WORKDIR/cfg.json, tok/tokenizer.json and gguf/"; exit 2; }
  [ -n "$model_id" ] || model_id=$("$T" -c "import json;print(json.load(open('cfg.json')).get('model_id') or '')")
  [ -n "$model_id" ] || { echo "FROM=build: pass MODEL_ID (5th argument)"; exit 2; }
else
step "1/6 download"
if [ -z "$file" ]; then
  file=$("$T" -I "$K/pick_gguf.py" "$repo") || { echo "STOP: no usable GGUF in $repo (see above); pass ORG/REPO:FILE.gguf"; exit 2; }
fi
# Into the shared HF cache (reused by later runs and by llama.cpp tools), symlinked into WORKDIR/gguf.
cached=$("$hf" download "$repo" "$file" 2>/dev/null | tail -1 | sed "s/^path=//")
[ -f "$cached" ] || { echo "download of $repo/$file failed"; exit 2; }
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

step "2/6 inspect (chat block derived from the template and verified)"
"$T" -I "$K/inspect_gguf.py" "$gguf" cfg.json 2>&1 | tee inspect.log | grep -v "^chat OK"
if [ "${PIPESTATUS[0]}" != 0 ] || grep -q "^chat FAIL\|not derivable\|WARNING: no chat block\|WARNING: no kernels\|WARNING: head_dim\|Traceback" inspect.log; then
  arch=$("$T" -c "import json;print(json.load(open('cfg.json')).get('arch',''))" 2>/dev/null)
  for ex in "$here"/reference/*-example; do
    [ -f "$ex/cfg.json" ] && [ "$("$T" -c "import json;print(json.load(open('$ex/cfg.json')).get('arch',''))" 2>/dev/null)" = "$arch" ] && \
      echo "HINT: reference/$(basename "$ex") is a working port of arch '$arch': $here/reference/apply_example.sh $(basename "$ex") kit; cp $ex/cfg.json's reviewed fields into cfg.json, then FROM=build KIT=kit $0 $spec $work $port $mem_gib MODEL_ID"
  done
  echo "STOP: see inspect.log; fix cfg.json (chat block / formats) by hand as loom-port-llm/SKILL.md describes, then rerun build.py"; exit 3
fi

fi  # FROM=build

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

step "6/6 smoke test"
"$T" - "$work/pkg" "http://127.0.0.1:$port" <<'EOF' || exit 6
import json, sys, requests
pkg, url = sys.argv[1], sys.argv[2]
cfg = json.load(open(pkg + "/llm_config.json")); model = cfg["model_id"]; chat = cfg["chat"]
off = {"chat_template_kwargs": {"enable_thinking": False}} if chat.get("thinking") == "toggle" else {}
post = lambda body: requests.post(url + "/v1/chat/completions", json={"model": model, **body}, timeout=600).json()
ok = True
r = post({"messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}], "max_tokens": 600, "temperature": 0})
m = r["choices"][0]["message"]; good = "paris" in ((m.get("content") or "") + (m.get("reasoning_content") or "")).lower()
print("chat:", "OK" if good else "FAIL", repr((m.get("content") or "")[:80])); ok &= good
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = post({"messages": [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}], "tools": tools,
          "max_tokens": 600, "temperature": 0, **off})
calls = r["choices"][0]["message"].get("tool_calls") or []
good = bool(calls) and calls[0]["function"]["name"] == "get_weather" and "paris" in calls[0]["function"]["arguments"].lower()
print("tool call:", "OK" if good else "FAIL", json.dumps(calls)[:200]); ok &= good
n = requests.post(url + "/v1/chat/completions/input_tokens", json={"model": model, "messages": [{"role": "user", "content": "hi"}]}).json()
print("input_tokens:", n)
sys.exit(0 if ok else 1)
EOF
echo "PORT_RESULT $(printf '{"package": "%s", "model_id": "%s", "context": %s, "memory_bytes": %s, "port": %s, "gguf": "%s"}' "$work/pkg" "$model_id" "${ctx:-0}" "$mem" "$port" "$gguf")"
