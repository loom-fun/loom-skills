# RDNA3 family (gfx11-generic, `@rdna3`)

Wave 32, WMMA 16x16 fragments (`vector<16xf16>`). The base target of every program file: a kernel `K` carries `target(@rdna3)`
and nothing else of its text changes, so a build for this family only is byte-identical to a build without the dual-target pass.
Devices: `strix-halo` (gfx1151). A discrete RDNA3 card (gfx1100) shares gfx11-generic: a device row (card entry, roofline preset,
runner, fit table), no new kernels.

Data: `family.json` (read by `registry.families()`, `dual_target.py`).
