"""Rows per workgroup (R) of the decode matrix-vector kernels, shared by gen_llm.py (which emits the
variants) and make_kernel_tests.py (which must test the variant the model launches: a disagreement
silently SKIPs the case in run_kernel_tests.sh).

Catalog lever: halve R while the grid is under-filled. Env: MV_ADAPT=0 (always R=16), MV_WG1 / MV_WG_LONG
(1-token workgroup targets; K >= 8192 uses MV_WG_LONG), MV_WG4 (a separate 4-token target).
The 1- and 4-token variants of a projection use the same R unless MV_WG4 is set: R changes the summation
order, so different R makes a request's output depend on whether it was decoded alone or batched
(consistency.py fails). MV_WG4 opts into that anyway (more c4 throughput: record it in the report).
"""
import os

ADAPT = os.environ.get("MV_ADAPT", "1") not in ("", "0")
WG1 = int(os.environ.get("MV_WG1", "4096")); WG4 = int(os.environ.get("MV_WG4", "128"))
WG_LONG = int(os.environ.get("MV_WG_LONG", "640"))
SAME_R = "MV_WG4" not in os.environ


def pick_r(n, K, TG):
    if not ADAPT or n is None:
        return 16
    if SAME_R:
        TG = 1
    target = (WG_LONG if K >= 8192 else WG1) if TG == 1 else WG4
    R = 16
    while R > 1 and n // R < target and n % (R // 2) == 0:
        R //= 2
    return R
