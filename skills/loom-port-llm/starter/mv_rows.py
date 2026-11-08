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


def iq_r(n, K, TG, sw=False):
    """Rows per workgroup of the int8-activation (ACT_Q8) matvecs. A row's arithmetic does not depend on R (explicit
    ops, explicit butterfly), so R is free per variant without changing any result:
    - 8+ tokens: 16 rows (fewer activation reloads; Qwen3.8-27B 17408x5120 t16: 395 -> 368 us).
    - 16 tokens, gate/up (swiglu) or K >= 8192: 32 rows. Each workgroup loads its 16 tokens' int8 activations for the
      whole K once; 32 rows halve that per weight row. Muse-Glimmer t16 (interleaved): gate/up 6656x19968 -5.5 %
      (with BFLY_PERM), down 19968x6656 -3.7 %; K = 4096 / 6656 single-weight projections measured flat or +4 %.
      MV_Q8_R32=0 turns it off.
    - Pair kernels (sw == 2: attention q + gate, k + v in one launch): 16 rows, halved (down to 4) while fewer than
      32 row groups would remain. Muse-Glimmer t16 (interleaved, vs two single launches): q + gate 6656x4096
      R16 -9 % (R32 -1 %), k + v 6656x256 R8 -36 % (R16 -16 %, R32 +52 %)."""
    R = pick_r(n, K, TG)
    if TG >= 8 and n and n % 16 == 0 and os.environ.get("MV_Q8_R16", "1") == "1":
        R = 16
    if sw == 2 and n:
        R = 16
        while R > 4 and n // R < 32 and n % (R // 2) == 0:
            R //= 2
        return R
    waves = (K // 32 + 31) // 32
    lds = waves * 32 * TG * (2 if sw else 1) * 4   # the workgroup's F32 partials (waves x R x TG per weight)
    if (TG >= 16 and n and n % 32 == 0 and (sw or K >= 8192) and K // 32 <= 1024 and lds <= 48 * 1024
            and os.environ.get("MV_Q8_R32", "1") == "1"):
        R = 32
    return R


def flash2_deep_from(cfg):
    """First key position at which gen_flash2.py's prefill attention runs two 16-token tiles per workgroup (pairs of
    uniform tiles of one row) for G = 3 or 4 query heads per KV head at head_dim <= 128 (one tile by default there);
    0 = never. Shared with make_kernel_tests.py (which tests both kernels the program launches). Each staged K/V tile
    then serves twice the queries: deep prefill is bandwidth-bound on the KV cache (Qwen3-4B, G = 4, a 131k prompt:
    2152 s with one tile, 530 s with two), while shallower tiles keep one (two lost 3-16 % at 1k-32k, G = 3 / 4).
    FLASH2_TS_DEPTH overrides (0: off); FLASH2_TS forces one kernel."""
    v = os.environ.get("FLASH2_TS_DEPTH")
    if v:
        return int(v)
    G = cfg["heads"] // cfg["kv_heads"]
    deep = cfg["head_dim"] <= 128 and G in (3, 4) and not cfg.get("attn_sinks") and not os.environ.get("FLASH2_TS")
    return 32768 if deep else 0
