#!/bin/bash
# usage: probe.sh OUTDIR PKG MODEL_ID MEM PROFILE(families|-) "TOKENS:N TOKENS:N ..."
set -u
out=$1 pkg=$2 mid=$3 mem=$4 prof=$5 work=$6
SK=${SK:?set SK to the skills dir}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"; out=$(cd "$out" && pwd)
export LOOM_SERVE_STATE=$out/state VENV=${VENV:-$HOME/work/devlabgap/loom/.venv}
extra=(-- ${LOOM_EXTRA:-})
[ "$prof" != - ] && extra+=(--device_profiling_mode=$prof --device_profiling_output=$out/p.ireeprof --device_profiling_flush_interval_ms=50)
python3 "$here/sampler.py" "$out/samples.csv" 0.005 & sp=$!
"$SK/loom-setup/scripts/serve.sh" s 8461 amdgpu "$mem" "$pkg/model.json" "${extra[@]}" || { kill $sp; exit 1; }
python3 "$here/pp.py" http://127.0.0.1:8461 "$mid" 256 3 > /dev/null   # warm-up
spin=; [ -n "${SPIN:-}" ] && { taskset -c ${SPIN} sh -c "while :; do :; done" & spin=$!; }
[ -n "${DUTY:-}" ] && { taskset -c 3 python3 "$here/dutyspin.py" ${DUTY%/*} ${DUTY#*/} & spin=$!; }
for w in $work; do echo "## $w"; python3 "$here/pp.py" http://127.0.0.1:8461 "$mid" ${w%:*} ${w#*:} "$out/pp.jsonl"; done | tee "$out/pp.out"
[ -n "$spin" ] && kill $spin
curl -s http://127.0.0.1:8461/logz -H "X-Loom-Model: $mid" > "$out/logz.jsonl"
"$SK/loom-setup/scripts/stop.sh" s
kill $sp; wait $sp
