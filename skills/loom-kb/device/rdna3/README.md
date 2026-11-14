# RDNA3 family (gfx11-generic, `@rdna3`)

Wave 32, WMMA 16x16 fragments (`vector<16xf16>`). The base target of every program file: a kernel `K` carries `target(@rdna3)`
and nothing else of its text changes, so a build for this family only is byte-identical to a build without the dual-target pass.
Devices: `strix-halo` (gfx1151), `radeon-rx7900xtx` (gfx1100, spec values). A discrete RDNA3 card shares gfx11-generic: a device row (card entry, roofline preset,
runner, fit table), no new kernels.

Data: `family.json` (read by `registry.families()`, `dual_target.py`).

## WMMA runs on the VALU (why m256 helps)

On RDNA3 WMMA runs on the SIMD's VALU, so the weight dequant serializes with it: the m128 GEMM spends ~190 VALU instructions per 32
WMMAs; m256 halves that per MMA (~115; see `topic/prefill.md`, "WMMA GEMM"). The gfx12 twins have the same property (below in `device/rdna4`).
