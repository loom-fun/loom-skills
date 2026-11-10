#!/bin/bash
# Refuse a load that could take the machine down (GPU memory is system RAM on Strix Halo).
# usage: memcheck.sh NEED_BYTES [WHAT]     exit 1 with the reason when the load is too big for the machine as it is now
#   The default plan stays within the memory cap (formats.memory_cap: 60 % of RAM, or LOOM_MEMORY_CAP_GIB); such a
#   load is not checked here. A load above the cap is an explicit per-model budget (a package whose resident
#   weights are bigger than the default allows: llm_config.json "memory", loom-port-llm/SKILL.md) and must leave
#   LOOM_MEMORY_HEADROOM_GIB (default 16) of MemAvailable on top: nothing else big may be resident, and
#   page cache counts as available. No-op without /proc/meminfo (Windows).
need=${1:?usage: memcheck.sh NEED_BYTES [WHAT]}; what=${2:-the load}
[ -r /proc/meminfo ] || exit 0
read -r total avail < <(awk '/^MemTotal/{t=$2} /^MemAvailable/{a=$2} END{print t*1024, a*1024}' /proc/meminfo)
cap=$(( total * 6 / 10 )); [ -n "${LOOM_MEMORY_CAP_GIB:-}" ] && cap=$(awk -v g="$LOOM_MEMORY_CAP_GIB" 'BEGIN{printf "%d", g*1073741824}')
head=$(awk -v g="${LOOM_MEMORY_HEADROOM_GIB:-16}" 'BEGIN{printf "%d", g*1073741824}')
[ "$need" -le "$cap" ] && exit 0
gib() { awk -v b="$1" 'BEGIN{printf "%.1f", b/1073741824}'; }
if [ "$avail" -lt $(( need + head )) ]; then
  echo "REFUSED: $what needs $(gib "$need") GiB, above the $(gib "$cap") GiB cap, and MemAvailable is $(gib "$avail") GiB (< $(gib $(( need + head ))) = need + $(gib "$head") GiB headroom)." >&2
  echo "  Stop other loom-serve/llama-server processes and drop big files from tmpfs (/tmp), or pick a smaller budget." >&2
  exit 1
fi
echo "memcheck: $what $(gib "$need") GiB above the $(gib "$cap") GiB cap; MemAvailable $(gib "$avail") GiB leaves $(gib $(( avail - need ))) GiB" >&2
