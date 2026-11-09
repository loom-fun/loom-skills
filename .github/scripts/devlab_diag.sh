#!/bin/bash
# The "diag" step of devlab_experiment.sh: why a Loom kernel runs slower on this runner than on a local Strix Halo.
#   devlab_diag.sh PKG OUT MEM ROUNDS
# 1. The machine: kernel, boot flags, amdgpu module parameters, KFD topology of the GPU node, firmware, clocks,
#    power profile and caps, CPU governor (OUT/sysfacts.txt).
# 2. gemma-4-E2B prefill A/B (perf.py --long), ROUNDS interleaved rounds of four packages: the old flash2_d512
#    (28.9 KB LDS, 256 lanes), the same kernel with its K tile padded to 36 KB and to 57.6 KB of LDS (no other
#    change), and the published one (57.6 KB, 512 lanes). GPU clock/power/busy sampled every 0.5 s (OUT/sample_*.log).
set -uo pipefail
pkg=$1; out=$2; mem=$3; rounds=${4:-2}
here=$(cd "$(dirname "$0")/../.." && pwd); SV=$here/skills/loom-setup/scripts; B=$here/skills/loom-benchmark/scripts
T=${T:?}; D=$(dirname "$pkg")
f=$out/sysfacts.txt
{
  echo "## uname"; uname -a; echo "## cmdline"; cat /proc/cmdline; echo "## os"; grep PRETTY /etc/os-release
  echo "## amdgpu module"; modinfo -F version amdgpu 2>/dev/null; cat /sys/module/amdgpu/version 2>/dev/null
  for p in /sys/module/amdgpu/parameters/*; do echo "$(basename "$p")=$(cat "$p" 2>/dev/null)"; done
  echo "## kfd topology"
  for n in /sys/class/kfd/kfd/topology/nodes/*; do
    grep -q "simd_count [1-9]" "$n/properties" 2>/dev/null || continue
    echo "# $n"; cat "$n/properties"; echo "# mem_banks"; cat "$n"/mem_banks/*/properties 2>/dev/null
    echo "# caches"; for c in "$n"/caches/*; do tr '\n' ' ' < "$c/properties"; echo; done | sort | uniq -c
  done
  for c in /sys/class/drm/card*/device; do
    [ -f "$c/pp_dpm_sclk" ] || continue; echo "## $c"
    for x in power_dpm_force_performance_level pp_power_profile_mode pp_dpm_sclk pp_dpm_mclk pp_dpm_fclk pp_dpm_socclk \
             mem_info_vram_total mem_info_gtt_total current_link_speed; do echo "# $x"; cat "$c/$x" 2>/dev/null; done
    echo "# fw_version"; for x in "$c"/fw_version/*; do echo "$(basename "$x")=$(cat "$x" 2>/dev/null)"; done
    for h in "$c"/hwmon/hwmon*; do echo "# hwmon $h"; for x in "$h"/*_input "$h"/*_cap* "$h"/*_average "$h"/*_label; do
      [ -f "$x" ] && echo "$(basename "$x")=$(cat "$x" 2>/dev/null)"; done; done
  done
  echo "## cpu"; lscpu | grep -E "Model name|^CPU\(s\)|MHz"; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor \
    /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference /sys/firmware/acpi/platform_profile 2>/dev/null
  echo "## thp"; cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag
  echo "## ttm"; cat /sys/module/ttm/parameters/* 2>/dev/null; echo "## mem"; free -g
  echo "## iommu"; ls /sys/class/iommu 2>/dev/null; dmesg 2>/dev/null | grep -iE "iommu|amdgpu.*(fw|firmware|lds|cu_mode)" | head -20
  echo "## firmware files"; ls -la /lib/firmware/amdgpu 2>/dev/null | grep -iE "gc_11_5_1|psp_14_0_1|smu_14_0_1|dcn_3_5_1|vcn_4_0_6" 
  echo "## amd-smi"; command -v amd-smi >/dev/null && { amd-smi static 2>&1 | head -80; amd-smi firmware 2>&1 | head -60; }
  echo "## rocminfo"; command -v rocminfo >/dev/null && rocminfo 2>&1 | grep -iE "gfx|Compute Unit|SIMDs|LDS|Workgroup|Wavefront|Max Waves|segment" | head -40
} > "$f" 2>&1
cat "$f"

sampler() {   # GPU sclk, power, busy, temp every 0.5 s
  local c; c=$(ls -d /sys/class/drm/card*/device 2>/dev/null | while read -r c; do [ -f "$c/pp_dpm_sclk" ] && echo "$c" && break; done)
  local h; h=$(ls -d "$c"/hwmon/hwmon* 2>/dev/null | head -1)
  while :; do echo "$(date +%s.%N | cut -c1-14) sclk=$(grep '\*' "$c/pp_dpm_sclk" 2>/dev/null | awk '{print $2}') busy=$(cat "$c/gpu_busy_percent" 2>/dev/null) pw=$(cat "$h/power1_average" "$h/power1_input" 2>/dev/null | head -1) t=$(cat "$h/temp1_input" 2>/dev/null)"; sleep 0.5; done
}
old=$D/e2b_old
[ -f "$old/.fetched" ] || { "$LOOM_ROOT/.venv/bin/loom-fetch" "https://huggingface.co/loom-fun/unsloth-gemma-4-E2B-it-loom/resolve/233724ca3f7e/model.json" \
  --no-input --materialize=copy --cache "$D/fetch-cache" --output="$old" > "$out/fetch_old.log" 2>&1 && touch "$old/.fetched"; } || { echo "FAIL: fetch old"; exit 1; }
"$T" "$here/.github/scripts/lds_variant.py" "$old" "$D/e2b_oldpad36" g_attention_flash2_d512 k_bytes 23808 || exit 1
"$T" "$here/.github/scripts/lds_variant.py" "$old" "$D/e2b_oldpad57" g_attention_flash2_d512 k_bytes 45312 || exit 1
seq_pk=(); for r in $(seq 1 "$rounds"); do
  if [ $((r % 2)) = 1 ]; then seq_pk+=("$old" "$D/e2b_oldpad36" "$D/e2b_oldpad57" "$pkg"); else seq_pk+=("$pkg" "$D/e2b_oldpad57" "$D/e2b_oldpad36" "$old"); fi
done
for p in "${seq_pk[@]}"; do
  n=$(basename "$p"); [ "$p" = "$pkg" ] && n=e2b_new
  "$SV/stop.sh" diag >/dev/null 2>&1; "$SV/serve.sh" diag 8471 amdgpu "$mem" "$p/model.json" | tail -1
  sampler > "$out/sample_${n}_$(date +%s).log" & sp=$!
  "$T" "$B/perf.py" http://127.0.0.1:8471 "$n" --model google/gemma-4-E2B-it --long --log "$out/diag_perf.jsonl" --no-campaign \
    | grep prefill_tps | tr '\n' ' '; echo " <- $n"
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
done
"$SV/stop.sh" diag >/dev/null 2>&1
"$T" - "$out/diag_perf.jsonl" <<'PY'
import json, sys, statistics as st, collections
r = collections.defaultdict(list)
for l in open(sys.argv[1]):
    x = json.loads(l); r[x["label"]].append([p["prefill_tps"] for p in x["prefill_long"]])
for k, v in r.items(): print("DIAG", k, "8k/16k/32k median", [round(st.median(c)) for c in zip(*v)], "runs", v)
PY
