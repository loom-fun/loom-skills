#!/bin/bash
# A/B of Gufo revisions on ONE DevLab runner (.github/workflows/devlab-gufo-ab.yml): every variant is built here from source
# (Nix in Docker: nixos/nix with a persistent store volume, toolchain from gufo.cachix.org), then `gufo bench` runs in the
# experiment's order (ABBA...) and, with "exact": true, each variant serves greedy completions with top-5 logprobs
# (gufo_exact.py: short prompts and ~2k / ~8k-token documents) that must be byte-identical across variants.
#
#   WORK=DIR devlab_gufo_ab.sh EXPERIMENT.json
#
# EXPERIMENT.json: {"model": "OWNER/REPO@REV/FILE.gguf", "variants": [{"name", "repo": "OWNER/REPO", "rev": SHA}, ...],
#                   "order": [names...], "bench_args": "-p 2048,8192 -n 64 -r 2", "exact": true}
# Writes $WORK/out: build_<v>.log, bench_<i>_<v>.log, bench.tsv (variant, test, t/s), exact_<v>.json, summary.md.
set -euo pipefail
exp=$(realpath "$1"); here=$(cd "$(dirname "$0")" && pwd); repo_root=$(cd "$here/../.." && pwd)
W=${WORK:?WORK}; O=$W/out; mkdir -p "$O" "$W/src" "$W/hf"
VOL=${GUFO_AB_NIX_VOLUME:-gufo-ab-nix}
q() { python3 -c "import json,sys; e=json.load(open('$exp')); $1"; }
NIXCONF='experimental-features = nix-command flakes
sandbox = false
build-users-group =
accept-flake-config = true
extra-substituters = https://gufo.cachix.org
extra-trusted-public-keys = gufo.cachix.org-1:rfZe3n4lHINgyoACKzs5EthIZmBNvEyCCtufUpth0PE='
nixrun() {   # nixrun GPU(0|1) CMD: CMD in nixos/nix with the store volume and $W mounted at the same path
  local gpu=()
  [ "$1" = 1 ] && gpu=(--device /dev/kfd --device /dev/dri --ulimit memlock=-1 --security-opt seccomp=unconfined --ipc=host --network host)
  docker run --rm ${NAME:+--name "$NAME"} "${gpu[@]}" -v "$VOL":/nix -v "$W":"$W" -v "$repo_root":"$repo_root":ro -e HOME=/root \
    nixos/nix:latest sh -c "mkdir -p /root/.config/nix && printf '%s\n' '$NIXCONF' > /root/.config/nix/nix.conf && git config --global --add safe.directory '*' && $2; rc=\$?; chown -R $(id -u):$(id -g) $W/src $W/out 2>/dev/null; exit \$rc"
}
docker volume create "$VOL" >/dev/null
# trees an earlier run's build container left owned by root
docker run --rm -v "$W":"$W" nixos/nix:latest chown -R "$(id -u):$(id -g)" "$W/src" "$W/out" 2>/dev/null || true

# ---- model (Hugging Face, revision-pinned): resumable, HTTP/1.1 (a 17 GB HTTP/2 stream was cancelled mid-file), checked
m=$(q 'print(e["model"])'); r=${m%/*}; name=${m##*/}; rev=${r#*@}; r=${r%@*}
model=$W/hf/$r/$rev/$name
if [ ! -f "$model" ]; then
  mkdir -p "${model%/*}"
  for attempt in 1 2 3 4 5 6; do
    curl -fsSL --http1.1 -C - --retry 5 --retry-all-errors ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
      "https://huggingface.co/$r/resolve/$rev/$name" -o "$model.part" && { mv "$model.part" "$model"; break; }
    echo "download attempt $attempt failed; resuming"; sleep 10
  done
fi
[ -f "$model" ] || { echo "model download failed: $m"; exit 1; }
echo "model $m $(du -h "$model" | cut -f1)" | tee "$O/model.txt"

# ---- builds
mapfile -t vars < <(q 'print("\n".join(v["name"]+" "+v["repo"]+" "+v["rev"] for v in e["variants"]))')
for line in "${vars[@]}"; do
  read -r v vrepo vrev <<<"$line"; src=$W/src/$v
  [ -d "$src/.git" ] || git clone -q "https://github.com/$vrepo" "$src"
  git -C "$src" fetch -q origin "$vrev" 2>/dev/null || git -C "$src" fetch -q origin
  git -C "$src" checkout -q --force "$vrev"; git -C "$src" log --oneline -1 > "$O/rev_$v.txt"
  echo "== build $v ($vrepo@$vrev)"; start=$(date +%s)
  nixrun 0 "cd $src && nix develop path:$src --command bash -c 'eval \"\${preConfigure:-}\"; cmake --preset release -B build -DBUILD_TESTING=OFF >/dev/null && cmake --build build --target gufo --parallel \$(nproc)'" \
    > "$O/build_$v.log" 2>&1 || { tail -30 "$O/build_$v.log"; exit 1; }
  echo "built $v in $(( $(date +%s) - start )) s"; tail -1 "$O/build_$v.log"
done

# ---- runner state
{ date -Is; uname -r; grep -m1 "model name" /proc/cpuinfo; free -g | head -2; } > "$O/runner.txt"
H=$(ls -d /sys/class/drm/card*/device/hwmon/hwmon* 2>/dev/null | head -1 || true)
if [ -n "$H" ]; then ( while :; do echo "$(date +%s.%N) $(cat $H/power1_average 2>/dev/null || cat $H/power1_input 2>/dev/null) $(cat $H/freq1_input 2>/dev/null)"; sleep 0.2; done ) > "$O/power.tsv" & PL=$!; trap 'kill $PL 2>/dev/null' EXIT; fi

# ---- benchmarks, in the experiment's order
args=$(q 'print(e.get("bench_args","-p 2048,8192 -n 64 -r 2"))'); i=0
printf 'run\tvariant\ttest\ttps\n' > "$O/bench.tsv"
for v in $(q 'print(" ".join(e["order"]))'); do
  i=$((i+1)); src=$W/src/$v; echo "== bench $i $v"; echo "$i $v $(date +%s.%N) start" >> "$O/steps.tsv"
  NAME=gufo-ab-bench nixrun 1 "cd $src && nix develop path:$src --command ./build/gufo bench --model $model $args" > "$O/bench_${i}_$v.log" 2>&1 || { tail -20 "$O/bench_${i}_$v.log"; exit 1; }
  echo "$i $v $(date +%s.%N) end" >> "$O/steps.tsv"
  grep -E '^\|.*(pp|tg)[0-9]+' "$O/bench_${i}_$v.log" | awk -F'|' -v i=$i -v v=$v '{gsub(/ /,"",$6); split($7,a,"±"); gsub(/ /,"",a[1]); print i"\t"v"\t"$6"\t"a[1]}' | tee -a "$O/bench.tsv"
done

# ---- exactness: greedy text + top-5 logprobs byte-identical across variants
if [ "$(q 'print(int(bool(e.get("exact"))))')" = 1 ]; then
  text=$repo_root/skills/loom-benchmark/data/doc_32000.txt
  for line in "${vars[@]}"; do
    read -r v _ _ <<<"$line"; src=$W/src/$v
    NAME=gufo-ab-serve nixrun 1 "cd $src && nix develop path:$src --command ./build/gufo serve llm --model $model --served-model-name gufo-ab --context 65536 --speculative off --sessions 1 --host 127.0.0.1 --port 18731" > "$O/serve_$v.log" 2>&1 &
    for _ in $(seq 1 600); do curl -sf localhost:18731/v1/models >/dev/null && break; sleep 2; done
    python3 "$here/gufo_exact.py" http://127.0.0.1:18731 "$O/exact_$v.json" "$text" > "$O/exact_$v.log" 2>&1 || echo "exact $v failed: $(tail -2 "$O/exact_$v.log")"
    docker kill gufo-ab-serve >/dev/null 2>&1 || true; wait || true; sleep 5
  done
fi

# ---- summary
python3 - "$O" "$exp" <<'PY' | tee "$O/summary.md"
import json, sys, os, statistics, filecmp
o, exp = sys.argv[1], json.load(open(sys.argv[2]))
rows = [l.rstrip("\n").split("\t") for l in open(f"{o}/bench.tsv")][1:]
tests = sorted({r[2] for r in rows}, key=lambda t: (t[:2], int(t[2:])))
names = [v["name"] for v in exp["variants"]]
print("| test | " + " | ".join(f"{n} (runs)" for n in names) + f" | {names[-1]} vs {names[0]} |\n|---|" + "---|" * (len(names) + 1))
for t in tests:
    vals = {n: [float(r[3]) for r in rows if r[1] == n and r[2] == t] for n in names}
    med = {n: statistics.median(v) for n, v in vals.items() if v}
    cells = [f"{med[n]:.1f} ({', '.join(f'{x:.1f}' for x in vals[n])})" for n in names]
    print(f"| {t} | " + " | ".join(cells) + f" | {100 * (med[names[-1]] / med[names[0]] - 1):+.1f} % |")
ex = [f"{o}/exact_{n}.json" for n in names]
if exp.get("exact"):
    ok = all(os.path.exists(f) for f in ex) and all(filecmp.cmp(ex[0], f, shallow=False) for f in ex[1:])
    print(f"\nexactness (greedy text + top-5 logprobs, {len(json.load(open(ex[0]))) if os.path.exists(ex[0]) else 0} prompts): " + ("byte-identical" if ok else "DIFFERENT or missing"))
PY
