"""amdgpu gpu_metrics v3.0 (Strix Halo / SMU 14) decoder, ctypes mirror of kgd_pp_interface.h struct gpu_metrics_v3_0."""
import ctypes as C
class GM30(C.Structure):
    _fields_ = [("structure_size", C.c_uint16), ("format_revision", C.c_uint8), ("content_revision", C.c_uint8),
        ("temperature_gfx", C.c_uint16), ("temperature_soc", C.c_uint16), ("temperature_core", C.c_uint16 * 16), ("temperature_skin", C.c_uint16),
        ("average_gfx_activity", C.c_uint16), ("average_vcn_activity", C.c_uint16), ("average_ipu_activity", C.c_uint16 * 8),
        ("average_core_c0_activity", C.c_uint16 * 16), ("average_dram_reads", C.c_uint16), ("average_dram_writes", C.c_uint16),
        ("average_ipu_reads", C.c_uint16), ("average_ipu_writes", C.c_uint16),
        ("system_clock_counter", C.c_uint64),
        ("average_socket_power", C.c_uint32), ("average_ipu_power", C.c_uint16), ("average_apu_power", C.c_uint32),
        ("average_gfx_power", C.c_uint32), ("average_dgpu_power", C.c_uint32), ("average_all_core_power", C.c_uint32),
        ("average_core_power", C.c_uint16 * 16), ("average_sys_power", C.c_uint16), ("stapm_power_limit", C.c_uint16),
        ("current_stapm_power_limit", C.c_uint16),
        ("average_gfxclk_frequency", C.c_uint16), ("average_socclk_frequency", C.c_uint16), ("average_vpeclk_frequency", C.c_uint16),
        ("average_ipuclk_frequency", C.c_uint16), ("average_fclk_frequency", C.c_uint16), ("average_vclk_frequency", C.c_uint16),
        ("average_uclk_frequency", C.c_uint16), ("average_mpipu_frequency", C.c_uint16),
        ("current_coreclk", C.c_uint16 * 16), ("current_core_maxfreq", C.c_uint16), ("current_gfx_maxfreq", C.c_uint16),
        ("throttle_residency_prochot", C.c_uint32), ("throttle_residency_spl", C.c_uint32), ("throttle_residency_fppt", C.c_uint32),
        ("throttle_residency_sppt", C.c_uint32), ("throttle_residency_thm_core", C.c_uint32), ("throttle_residency_thm_gfx", C.c_uint32),
        ("throttle_residency_thm_soc", C.c_uint32), ("time_filter_alphavalue", C.c_uint32)]
FIELDS = ["average_gfxclk_frequency", "average_fclk_frequency", "average_uclk_frequency", "average_socclk_frequency",
          "average_gfx_activity", "average_dram_reads", "average_dram_writes", "average_socket_power", "average_gfx_power",
          "average_apu_power", "stapm_power_limit", "current_stapm_power_limit", "current_gfx_maxfreq", "temperature_gfx",
          "throttle_residency_spl", "throttle_residency_fppt", "throttle_residency_sppt", "throttle_residency_thm_gfx",
          "throttle_residency_prochot", "time_filter_alphavalue"]
def read(path):
    b = open(path, 'rb').read()
    if len(b) < C.sizeof(GM30): b = b + bytes(C.sizeof(GM30) - len(b))
    g = GM30.from_buffer_copy(b[:C.sizeof(GM30)])
    return {f: getattr(g, f) for f in FIELDS}
if __name__ == "__main__":
    import glob, sys
    p = glob.glob('/sys/class/drm/card*/device/gpu_metrics')[0]
    print(C.sizeof(GM30), len(open(p, 'rb').read())); print(read(p))
