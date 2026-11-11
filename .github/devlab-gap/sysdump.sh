#!/bin/bash
# Dump GPU/driver/firmware/memory configuration relevant to GPU perf (no root needed).
d=$(ls -d /sys/class/drm/card*/device | grep -v -- - | head -1)
echo "## host"; hostname; uname -a; cat /proc/cmdline
echo "## bios"; cat /sys/class/dmi/id/{bios_version,bios_date,board_name,product_name,sys_vendor} 2>/dev/null
echo "## cpu policy"; cat /sys/devices/system/cpu/cpu0/cpufreq/{scaling_driver,scaling_governor,energy_performance_preference} 2>/dev/null; cat /sys/devices/system/cpu/amd_pstate/status 2>/dev/null
echo "## mem"; grep -E "MemTotal|MemAvailable|HugePages_Total|AnonHugePages" /proc/meminfo; cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag
echo "## ttm"; cat /sys/module/ttm/parameters/* 2>/dev/null | paste -sd' '; for f in /sys/module/ttm/parameters/*; do echo "$(basename $f)=$(cat $f)"; done
echo "## amdgpu vram/gtt"; for f in mem_info_vram_total mem_info_vis_vram_total mem_info_gtt_total mem_info_vram_used mem_info_gtt_used; do echo "$f=$(cat $d/$f 2>/dev/null)"; done
echo "## dpm"; for f in power_dpm_force_performance_level power_dpm_state pp_dpm_sclk pp_dpm_mclk pp_dpm_fclk pp_dpm_socclk pp_cur_state pp_power_profile_mode pp_features; do echo "-- $f"; cat $d/$f 2>/dev/null; done
echo "## hwmon"; for f in $d/hwmon/hwmon*/*; do [ -f $f ] && echo "$(basename $f)=$(cat $f 2>/dev/null)"; done
echo "## fw"; for f in $d/fw_version/*; do echo "$(basename $f)=$(cat $f 2>/dev/null)"; done
echo "## vbios"; cat $d/vbios_version 2>/dev/null
echo "## amdgpu params"; for f in /sys/module/amdgpu/parameters/*; do echo "$(basename $f)=$(cat $f 2>/dev/null)"; done
echo "## kfd topology"; for n in /sys/class/kfd/kfd/topology/nodes/*; do echo "-- $n"; cat $n/properties 2>/dev/null | grep -E "simd_count|max_engine_clk|gfx_target|cu_per|local_mem|fw_version|capability|debug_prop|num_cp_queues|sdma_fw|max_waves"; for m in $n/mem_banks/*; do cat $m/properties | paste -sd' '; done; done
echo "## kfd sched"; cat /sys/module/amdgpu/parameters/sched_policy /sys/module/amdgpu/parameters/mes 2>/dev/null
echo "## numa/iommu"; ls /sys/class/iommu 2>/dev/null; dmesg 2>/dev/null | grep -i -E "iommu|amdgpu.*(vram|gtt|mtype|mes)" | head -20
echo "## modinfo"; modinfo amdgpu 2>/dev/null | grep -E "^(version|vermagic|filename)"
