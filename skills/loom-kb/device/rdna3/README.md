# RDNA3 family (gfx11-generic, `@rdna3`)

Wave 32, WMMA 16x16 fragments (`vector<16xf16>`). The base target of every program file: a kernel `K` carries `target(@rdna3)`
and nothing else of its text changes, so a build for this family only is byte-identical to a build without the dual-target pass.
Devices: `strix-halo` (gfx1151), `radeon-rx7900xtx` (gfx1100, spec values), `gfx1150` (Radeon 890M, results from a named machine, `fleet/devices/gfx1150.json`), and `gfx1152` (Radeon 860M, results from leased XSJ hosts). A discrete RDNA3 card shares gfx11-generic: a device row (card entry, roofline preset,
runner, fit table), no new kernels.

Data: `family.json` (read by `registry.families()`, `dual_target.py`).

## Getting started on desktop Radeon

On Linux with ROCm tools installed, run `rocminfo` and match the GPU agent's `Name` and `Marketing Name` to your card. The Radeon RX 7800 XT (gfx1101, 16 GB) is an RDNA3 device. [Using rocminfo](https://rocm.docs.amd.com/projects/rocminfo/en/latest/how-to/use-rocminfo.html) · [AMD GPU specifications](https://rocm.docs.amd.com/en/docs-7.2.3/reference/gpu-arch-specs.html) · [AMD product specifications](https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7800-xt.html)

If your GPU has no device entry in this knowledge base, use `rdna3` to find the family guidance for a local Loom package. From the loom-skills checkout, replace the example path with that package's `llm_config.json`.

```sh
python3 skills/loom-kb/kb.py route --config /path/to/llm_config.json --device rdna3 --topic sdk
```

For installation and fetching, follow [Loom setup](../../../loom-setup/SKILL.md) and the package's Quick start.
