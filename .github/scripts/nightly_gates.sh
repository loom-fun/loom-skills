#!/bin/bash
# Nightly correctness gates on one published loom-fun card: deterministic, no LLM agent and no skill run.
# Fetches the latest card from Hugging Face, serves it, runs gates.sh (LEVEL=full by default) against the
# llama.cpp reference build the card was measured with. Correctness gates only: a card's DevLab perf session comes
# from a devlab-perf job (loom-org SKILL.md, step 3).
# Run by .github/workflows/nightly-gates.yml (one matrix job per card); runs by hand on any Strix Halo:
#
#   [LEVEL=full|quick|dev|fast] [WORK=~/loom-nightly] .github/scripts/nightly_gates.sh loom-fun/<card-repo>
#
# The card must be listed in .github/nightly-models.json (memory_bytes, reference example kit, llama.cpp build).
# Environment (all optional):
#   LEVEL         gates.sh level (default full)
#   WORK          work root (default ~/loom-nightly); this card uses WORK/<slug>/{pkg,bench,out,...}
#   LOOM_ROOT     Loom install with .venv and .tools (default WORK/loom; installed by install_loom.sh if missing).
#                 A card whose llm_config.json pins an SDK ("sdk": hrx-loom wheel) gets its own root next to it
#                 (loom-setup/scripts/ensure_sdk.sh)
#   LOOM_VERSION  hrx-loom version for a fresh install (default: the card's SDK pin, else install_loom.sh's validated
#                 version; set explicitly it overrides the pin)
#   LLAMACPP_DIR  a llama.cpp ROCm build to use instead of fetching the card's release
#   LLAMA_ROOT    where releases are unpacked (default WORK/llamacpp)
#   MATERIALIZE   loom-fetch --materialize (default copy; share avoids a second copy of the GGUFs)
#   FETCH_CACHE   loom-fetch --cache (default: loom-fetch's own)
#   LOOM_PORT / LLAMA_PORT   default 8160 / 8130
#   CLEAN=1       delete the fetched package (the GGUF copies) and FETCH_CACHE at exit
# Writes WORK/<slug>/out: gates.json, gates.out, gates_*.log, summary.json, serve.log.
# Every server this script or gates.sh starts is stopped at exit (state dirs under WORK/<slug>).
set -uo pipefail
repo=${1:?usage: nightly_gates.sh loom-fun/CARD_REPO}
here=$(cd "$(dirname "$0")/../.." && pwd); SK=$here/skills
B=$SK/loom-benchmark/scripts; SV=$SK/loom-setup/scripts
level=${LEVEL:-full}
WORK=${WORK:-$HOME/loom-nightly}; mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
read -r slug mem kit tag < <(python3 - "$here/.github/nightly-models.json" "$repo" <<'EOF'
import json, sys
m = next((m for m in json.load(open(sys.argv[1]))["models"] if m["repo"] == sys.argv[2]), None)
if m is None:
    sys.exit(f"{sys.argv[2]} is not in .github/nightly-models.json")
print(m["slug"], m["memory_bytes"], m.get("kit") or "-", m["llamacpp"])
EOF
) || exit 2
D=$WORK/$slug; pkg=$D/pkg; bench=$D/bench; out=$D/out
mkdir -p "$D" "$bench" "$out"
export LOOM_SERVE_STATE=$D/state LLAMACPP_STATE=$D/llstate PYTHONDONTWRITEBYTECODE=1
LOOM_PORT=${LOOM_PORT:-8160}; LLAMA_PORT=${LLAMA_PORT:-8130}; url=http://127.0.0.1:$LOOM_PORT
t0=$(date +%s)
step() { echo "== $* ($(date +%T), +$(( $(date +%s) - t0 ))s)"; }

cleanup() {
  for f in "$LOOM_SERVE_STATE"/*.pid; do [ -f "$f" ] && "$SV/stop.sh" "$(basename "$f" .pid)" >/dev/null 2>&1; done
  "$B/llamacpp_server.sh" --stop "$LLAMA_PORT" >/dev/null 2>&1
  cp "$LOOM_SERVE_STATE"/*.log "$LLAMACPP_STATE"/*.log "$out/" 2>/dev/null
  if [ "${CLEAN:-0}" = 1 ]; then
    rm -rf "$pkg"; [ -n "${FETCH_CACHE:-}" ] && rm -rf "$FETCH_CACHE"
    echo "removed the fetched package${FETCH_CACHE:+ and $FETCH_CACHE}"
  fi
  ss -ltn 2>/dev/null | grep -qE ":($LOOM_PORT|$LLAMA_PORT) " && echo "WARNING: port $LOOM_PORT or $LLAMA_PORT still listening"
  echo "nightly_gates.sh: servers stopped"
}
trap cleanup EXIT

# 1. Loom SDK (hrx-loom[hf]) + tools venv, as loom-setup installs them: the SDK the card pins (its llm_config.json),
# else the default
base=https://huggingface.co/$repo/resolve/main
curl -fsSL --retry 3 "$base/llm_config.json" -o "$out/llm_config.json" || { echo "FAIL: no llm_config.json in $repo"; exit 1; }
step "Loom SDK"
root=$("$SV/ensure_sdk.sh" "${LOOM_ROOT:-$WORK/loom}" "$out/llm_config.json") || exit 1
export LOOM_ROOT=$root VENV=$root/.venv
echo "hrx-loom $("$root/.venv/bin/python" -c 'import importlib.metadata as m; print(m.version("hrx-loom"))')"

# 2. The latest card. loom-fetch installs the files model.json lists; llm_config.json (the gates' view of the
# port: chat format, vision, drafter) and info.json (the published numbers) come from the repo next to it.
step "fetch $repo"
rm -rf "$pkg"
"$root/.venv/bin/loom-fetch" "$base/model.json" --no-input --materialize="${MATERIALIZE:-copy}" \
  ${FETCH_CACHE:+--cache "$FETCH_CACHE"} --output="$pkg" || { echo "FAIL: loom-fetch"; exit 1; }
for f in llm_config.json info.json; do curl -fsSL --retry 3 "$base/$f" -o "$pkg/$f" || [ $f = info.json ] || exit 1; done
curl -fsSL --retry 3 "$base/model.json" -o "$out/model.json"
# The published llm_config.json names its GGUFs by file name; the gates read them (GGUF metadata, the llama.cpp
# reference, the projector), so point them at this installation's parameter/ files.
python3 - "$pkg" <<'EOF'
import json, os, sys
p = sys.argv[1]; f = os.path.join(p, "llm_config.json"); c = json.load(open(f))
def fix(d, k):
    if isinstance(d, dict) and isinstance(d.get(k), str):
        q = os.path.join(p, "parameter", os.path.basename(d[k]))
        if os.path.exists(q):
            d[k] = q
fix(c, "gguf"); fix(c.get("vision"), "mmproj")
for k in ("draft", "drafter", "assistant"):
    fix(c.get(k), "gguf")
json.dump(c, open(f, "w"), indent=1)
print("model", c["model_id"], "gguf", c["gguf"], "vision", bool(c.get("vision")), "draft", bool(c.get("draft")),
      "thinking", c["chat"].get("thinking"))
EOF
gguf=$(python3 -c "import json; print(json.load(open('$pkg/llm_config.json'))['gguf'])")
# The card's README states the --memory_bytes it was validated with; a republished card may change it.
readme_mem=$(curl -fsSL --retry 3 "$base/README.md" 2>/dev/null | grep -o -- '--memory_bytes=[0-9]*' | head -1 | cut -d= -f2)
[ -n "$readme_mem" ] && [ "$readme_mem" != "$mem" ] && { echo "memory_bytes: $readme_mem from the card's README (nightly-models.json: $mem)"; mem=$readme_mem; }
[ -f "$gguf" ] || { echo "FAIL: $gguf missing after loom-fetch"; exit 1; }

# 3. The reference example's kit: its chat_probe renders and parses this card's chat format for the gates
if [ "$kit" != - ]; then
  step "kit: $kit example"
  rm -rf "$D/kit"; "$SK/loom-port-llm/reference/apply_example.sh" "$kit" "$D/kit" | tail -1 || exit 1
  export LOOM_KIT=$D/kit
fi

# 4. The llama.cpp reference build the card was measured with. The reference cache key names the build by its
# tag, not its path, so a cache restored on another runner still matches.
if [ -z "${LLAMACPP_DIR:-}" ]; then
  step "llama.cpp $tag"
  LLAMACPP_DIR=$("$B/fetch_llamacpp.sh" "${LLAMA_ROOT:-$WORK/llamacpp}" "$tag" | tail -1) || exit 1
fi
export LLAMACPP_DIR LLAMA_CACHE_BUILD=${LLAMA_CACHE_BUILD:-$tag}
echo "llama.cpp: $LLAMACPP_DIR (cache key build: $LLAMA_CACHE_BUILD)"

# 5. Serve
step "serve ($mem bytes)"
"$SV/serve.sh" "nightly-$slug" "$LOOM_PORT" amdgpu "$mem" "$pkg/model.json" | tail -3 || exit 1
curl -sf "$url/v1/models" >/dev/null || { echo "FAIL: loom-serve is not answering"; exit 1; }

# 6. Gates (FORCE=1: a nightly verdict never reuses an earlier pass)
step "gates LEVEL=$level"
FORCE=1 LEVEL=$level "$B/gates.sh" "$pkg" "$url" "$gguf" "$bench" "$LLAMA_PORT" > "$out/gates.out" 2>&1
rc=$?
grep -h "^GATES" "$out/gates.out" || tail -20 "$out/gates.out"
cp "$bench"/gates.json "$bench"/gates_*.log "$out/" 2>/dev/null

# 7. One summary for the workflow's table
# (runner, device, SDK and programs measured: the evidence pull_devlab.py records for the gates)
python3 - "$repo" "$slug" "$level" "$rc" $(( $(date +%s) - t0 )) "$out" "$B" "$pkg" <<'EOF'
import json, os, re, sys
repo, slug, level, rc, secs, out, B, pkg = sys.argv[1:]
g = json.load(open(os.path.join(out, "gates.json"))) if os.path.exists(os.path.join(out, "gates.json")) else {}
s = {"repo": repo, "slug": slug, "level": level, "verdict": g.get("verdict", "ERROR"), "seconds": int(secs),
     "gates_seconds": g.get("seconds"), "gates": {n: {"pass": x["pass"], "seconds": x["seconds"]} for n, x in g.get("gates", {}).items()}}
m = re.match(r"ephemeral-devlab-(.+?)-\d+$", os.environ.get("RUNNER_NAME", ""))
s["runner"] = m.group(1).upper() if m else os.environ.get("RUNNER_NAME") or os.uname().nodename
sys.path[:0] = [B, os.path.join(B, "card")]
try:
    from machine_info import detect_device, hrx_loom_identity
    s["device"], s["hrx_loom"] = detect_device(), hrx_loom_identity()
    from card_lib import programs_sha256
    s["programs_sha256"] = programs_sha256(pkg)
except Exception as e:  # noqa: BLE001  (evidence only: never fail the summary)
    print(f"summary: no device/SDK/programs evidence: {e}")
json.dump(s, open(os.path.join(out, "summary.json"), "w"), indent=1)
print("NIGHTLY", s["verdict"], slug, f"level={level} {secs}s", {n: "PASS" if x["pass"] else "FAIL" for n, x in s["gates"].items()})
EOF
exit $rc
