#!/bin/bash
# Wait until the GPU's memory is free before the next server starts, and log what was held.
#   usage: wait_vram.sh LABEL [MAX_MIB=1024] [WAIT_S=180] [LOGFILE]
# A loom-serve that starts while the previous process (a vLLM or llama.cpp server, the reference fill, the last loom-serve) still holds its VRAM
# fails to load (RESOURCE_EXHAUSTED / "model is unavailable"): a process group that has been killed frees its memory only when the driver has
# torn it down. Reads mem_info_vram_used of every DRM card (VRAM_FILES overrides the glob, for tests), prints
# "vram_used <MiB> MiB before LABEL (waited N s)", appends it to LOGFILE, and returns once the largest value is below MAX_MIB or has stopped
# falling for 20 s (a runner whose GPU holds that much by itself). Exit 0 always: the load that follows reports its own failure.
label=${1:?usage: wait_vram.sh LABEL [MAX_MIB] [WAIT_S] [LOGFILE]}; max=${2:-1024}; wait_s=${3:-180}; log=${4:-}
files=${VRAM_FILES:-/sys/class/drm/card*/device/mem_info_vram_used}
used() { cat $files 2>/dev/null | awk 'BEGIN{m=0} {if ($1>m) m=$1} END{printf "%d", m/1048576}'; }
t0=$(date +%s); prev=-1; still=0; cur=$(used)
if [ -z "$(ls $files 2>/dev/null)" ]; then echo "vram_used unknown before $label (no mem_info_vram_used)"; exit 0; fi
while [ "$cur" -ge "$max" ] && [ $(( $(date +%s) - t0 )) -lt "$wait_s" ]; do
  sleep 2; cur=$(used)
  if [ "$cur" -ge "$prev" ] && [ "$prev" -ge 0 ]; then still=$((still + 2)); else still=0; fi
  prev=$cur
  [ "$still" -ge 20 ] && break
done
line="vram_used $cur MiB before $label (waited $(( $(date +%s) - t0 )) s)"
echo "$line"; [ -n "$log" ] && echo "$(date +%T) $line" >> "$log"
exit 0
