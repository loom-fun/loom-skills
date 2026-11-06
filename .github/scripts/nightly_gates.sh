#!/bin/bash
# Nightly correctness gates on one published loom-fun card: deterministic, no LLM agent and no skill run.
# Fetches the latest card from Hugging Face, serves it, runs gates.sh (LEVEL=full by default) against the
# llama.cpp reference build the card was measured with, and optionally a report-only perf run.
# Run by .github/workflows/nightly-gates.yml (one matrix job per card); runs by hand on any Strix Halo:
#
#   [LEVEL=full|quick|dev|fast] [WORK=~/loom-nightly] .github/scripts/nightly_gates.sh loom-fun/<card-repo>
#
# The card must be listed in .github/nightly-models.json (memory_bytes, reference example kit, llama.cpp build).
# Environment (all optional):
#   LEVEL         gates.sh level (default full)
#   WORK          work root (default ~/loom-nightly); this card uses WORK/<slug>/{pkg,bench,out,...}
#   LOOM_ROOT     Loom install with .venv and .tools (default WORK/loom; installed by install_loom.sh if missing)
#   LOOM_VERSION  hrx-loom version for a fresh install (default: install_loom.sh's validated version)
#   LLAMACPP_DIR  a llama.cpp ROCm build to use instead of fetching the card's release
#   LLAMA_ROOT    where releases are unpacked (default WORK/llamacpp)
#   MATERIALIZE   loom-fetch --materialize (default copy; share avoids a second copy of the GGUFs)
#   FETCH_CACHE   loom-fetch --cache (default: loom-fetch's own)
#   LOOM_PORT / LLAMA_PORT   default 8160 / 8130
#   PERF=1        after the gates: Loom perf.py + c4_steady.py, reported next to the card's published numbers
#   CLEAN=1       delete the fetched package (the GGUF copies) and FETCH_CACHE at exit
# Writes WORK/<slug>/out: gates.json, gates.out, gates_*.log, summary.json, serve.log, perf.jsonl, perf.md.
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

# 1. Loom SDK (hrx-loom[hf]) + tools venv, as loom-setup installs them
root=${LOOM_ROOT:-$WORK/loom}
if [ ! -x "$root/.venv/bin/loom-serve" ] || [ ! -x "$root/.tools/bin/python" ]; then
  step "install Loom into $root"
  "$SV/install_loom.sh" "$root" ${LOOM_VERSION:+"$LOOM_VERSION"} || exit 1
fi
export LOOM_ROOT=$root VENV=$root/.venv; T=$root/.tools/bin/python
echo "hrx-loom $("$root/.venv/bin/python" -c 'import importlib.metadata as m; print(m.version("hrx-loom"))')"

# 2. The latest card. loom-fetch installs the files model.json lists; llm_config.json (the gates' view of the
# port: chat format, vision, drafter) and info.json (the published numbers) come from the repo next to it.
step "fetch $repo"
base=https://huggingface.co/$repo/resolve/main
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

# 7. Report-only perf: Loom now vs the numbers the card publishes (info.json); never changes the verdict
if [ "${PERF:-0}" = 1 ]; then
  step "perf (report only)"
  curl -sf "$url/v1/models" >/dev/null || "$SV/serve.sh" "nightly-$slug" "$LOOM_PORT" amdgpu "$mem" "$pkg/model.json" | tail -1
  model=$("$T" -c "import json; print(json.load(open('$pkg/llm_config.json'))['model_id'])")
  (cd "$out" && rm -f perf.jsonl c4.jsonl
   "$T" "$B/perf.py" "$url" nightly --model "$model" --no-campaign --log perf.jsonl | tail -1
   "$T" "$B/c4_steady.py" "$url" 4 --model "$model" --log c4.jsonl | tail -1 > c4_steady.txt
   "$T" - "$B" "$pkg/info.json" <<'EOF' | tee perf.md
import json, os, sys
sys.path.insert(0, sys.argv[1])
from bench_common import perf_metrics
card = json.load(open(sys.argv[2])).get("metrics", {}) if os.path.exists(sys.argv[2]) else {}
now = {}
for line in open("perf.jsonl"):
    now.update(perf_metrics(json.loads(line)))
try:
    now["c4_steady_tps"] = json.loads(open("c4_steady.txt").read())["agg_tps"]
except Exception:  # noqa: BLE001
    pass
name = lambda m: m.replace("000", "k") if m.startswith("prefill_") else m
print("| metric | Loom tonight | Loom on the card | llama.cpp on the card |\n|---|---|---|---|")
for m, v in sorted(now.items()):
    c, l = card.get(name(m)), card.get(name(m) + "_llamacpp")
    c, l = (c[0] if isinstance(c, list) else c), (l[0] if isinstance(l, list) else l)
    print(f"| {name(m)} | {v:.1f} | {c if c is not None else '-'} | {l if l is not None else '-'} |")
EOF
  )
fi

# 8. One summary for the workflow's table
python3 - "$repo" "$slug" "$level" "$rc" $(( $(date +%s) - t0 )) "$out" <<'EOF'
import json, os, sys
repo, slug, level, rc, secs, out = sys.argv[1:]
g = json.load(open(os.path.join(out, "gates.json"))) if os.path.exists(os.path.join(out, "gates.json")) else {}
s = {"repo": repo, "slug": slug, "level": level, "verdict": g.get("verdict", "ERROR"), "seconds": int(secs),
     "gates_seconds": g.get("seconds"), "gates": {n: {"pass": x["pass"], "seconds": x["seconds"]} for n, x in g.get("gates", {}).items()}}
json.dump(s, open(os.path.join(out, "summary.json"), "w"), indent=1)
print("NIGHTLY", s["verdict"], slug, f"level={level} {secs}s", {n: "PASS" if x["pass"] else "FAIL" for n, x in s["gates"].items()})
EOF
exit $rc
