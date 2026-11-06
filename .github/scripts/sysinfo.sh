#!/bin/bash
# The machine facts that change GPU speed on Strix Halo, one "key: value" per line (reads only; no GPU work).
# Compare a DevLab runner with the dev machine: bash .github/scripts/sysinfo.sh > sysinfo.txt
r() { printf '%s: %s\n' "$1" "$(eval "$2" 2>/dev/null | tr '\n' ' ' | sed 's/ *$//')"; }
r host "hostname"
r product "cat /sys/class/dmi/id/sys_vendor /sys/class/dmi/id/product_name"
r bios "cat /sys/class/dmi/id/bios_version /sys/class/dmi/id/bios_date"
r kernel "uname -r"
r cmdline_gpu "tr ' ' '\n' < /proc/cmdline | grep -iE 'amdgpu|ttm|iommu|gtt|amd_pstate|mitigations'"
r mem_total_kb "awk '/MemTotal/{print \$2}' /proc/meminfo"
r ttm_pages_limit "cat /sys/module/ttm/parameters/pages_limit"
r ttm_page_pool_size "cat /sys/module/ttm/parameters/page_pool_size"
r amdgpu_gttsize "cat /sys/module/amdgpu/parameters/gttsize"
for c in /sys/class/drm/card*/device; do
  [ -f "$c/mem_info_vram_total" ] || continue
  r vram_total_mib "echo \$((\$(cat $c/mem_info_vram_total)/1048576))"
  r gtt_total_mib "echo \$((\$(cat $c/mem_info_gtt_total)/1048576))"
  r vram_used_mib "echo \$((\$(cat $c/mem_info_vram_used)/1048576))"
  r gtt_used_mib "echo \$((\$(cat $c/mem_info_gtt_used)/1048576))"
  r dpm_level "cat $c/power_dpm_force_performance_level"
  r power_profile "grep '\*' $c/pp_power_profile_mode"
  r sclk_levels "cat $c/pp_dpm_sclk"
  r mclk_levels "cat $c/pp_dpm_mclk"
  r fclk_levels "cat $c/pp_dpm_fclk"
  r power_cap_uw "cat $c/hwmon/*/power1_cap"
  r power_cap_max_uw "cat $c/hwmon/*/power1_cap_max"
  r gpu_temp_mc "cat $c/hwmon/*/temp1_input"
done
r platform_profile "cat /sys/firmware/acpi/platform_profile"
r platform_profile_choices "cat /sys/firmware/acpi/platform_profile_choices"
r cpu_governor "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
r cpu_epp "cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"
r thp "cat /sys/kernel/mm/transparent_hugepage/enabled"
r numa_balancing "cat /proc/sys/kernel/numa_balancing"
r rocm "cat /opt/rocm/.info/version"
r amdgpu_fw "cat /sys/kernel/debug/dri/*/amdgpu_firmware_info | head -3"
r loadavg "cat /proc/loadavg"
r gpu_processes "ps -eo pid,comm,rss --sort=-rss | grep -iE 'loom|llama|python|ollama|vllm' | head -5"
