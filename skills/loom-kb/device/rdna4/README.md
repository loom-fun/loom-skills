# RDNA4 family (gfx12-generic, `@rdna4`)

Wave 32, WMMA 16x16 fragments with 8-wide operand vectors (`vector<8xf16>`: RDNA4 halves the fragment registers and doubles the dense
f16 rate per CU). A kernel with a gfx12 version `K` -> `K_rdna4` selects it per device (`dual_target.py`). The twins are text
rewrites of the RDNA3 kernels (`gen_gemm_rdna4.twin()`, `flash2_rdna4.to_rdna4()`, MoE inline in `gen_moe.py`); tuning that only
exists here: `flash2_b2`, `trim`, `vpack`, `prefetch`, `f16o_fuse`. Devices: `radeon-r9700` (gfx1201, 32 GB class), `gfx1201-16gb` (gfx1201, 16 GB class: RX 9070 XT, RX 9070; results from a named machine, `fleet/devices/gfx1201-16gb.json`).

**gfx1201 is selected by VRAM class**: arch gfx1201 and VRAM >= 30 GiB -> `radeon-r9700` (32 GB class); 14 to 30 GiB -> `gfx1201-16gb` (16 GB class); below -> not supported yet (a 12 GB class can be added later from the `build.py` fit and one measurement).

Data: `family.json`.

Measurements on the 16 GB class (`gfx1201-16gb`, a Radeon RX 9070 XT, hrx-loom 0.2.0). The R9700's packages run unchanged (same
chip, same programs); per model only the context differs: `fleet/devices/gfx1201-16gb.json` (32 of the 50 R9700 cards fit), by `loom-benchmark`
`test_9070xt.py` (hrx-loom 0.2.0, 2026-11-17, a desktop machine). Serving (the registry's card serve_args), elastic backing:
- `--memory_bytes=15000000000`: 17.1 GB board, ~1.1-1.5 GB held by a desktop session.
- `--state_slab_bytes=4mib` and `--weight_slab_bytes=64mib`: elastic state is committed per KV plane in 64 MiB slabs by default (82 planes =
  5.5 GB before the first request) and weights/workspace in 1 GiB slabs; the small slabs give +20-38 % context on 0.2.0 with perf rows within
  +-3 % (5 models, ABBA).
- `--parameter_mode=preload`: on a stock Ubuntu host (8 MiB RLIMIT_MEMLOCK, Linux 7.0) the default file mode and direct mode fail at the first
  request (io_uring "buffer pin preparation failed (12)": the amd file-staging path pins 8 x 2 MiB lanes). Alternative: `systemd-run --user
  -p SystemCallFilter=~io_uring_setup -p SystemCallErrorNumber=ENOSYS loom-serve ...`.
- hrx-loom 0.2 registers a model only when the budget holds one full `context_capacity` ("shared pool byte limit cannot address one model
  context"): a model whose native context does not fit needs the 9070 XT preset (`context_capacity` = its max_context). Registration is a loose
  bound: the max context is the largest at which a full-length request completes and the next answer is correct.
- Per model: gpt-oss-20b registers only with a smaller prefill chunk (`prefill_capacity` 1024: 38912 tokens, prefill -8 to -12 % vs the
  R9700; 256: 59392 tokens but prefill -33 %); LFM2.5-24B only with 256 (3k context).
- Performance (5 models, hrx-loom 0.2.0, ±3 % rule): decode and 4 streams match the R9700 card, prefill does not (-3 to -17 %): the 16 GB
  class has its own card section (`share_section_with` null). llama.cpp ROCm b1375 matches across the two boards (36 of 39 rows within
  ±3 %): the gap is elastic backing (the same R9700 served with these settings loses as much prefill), not the hardware.
- qwen35 hybrids (Qwen3.5, Ornith): 1-2 % above the max context a request completes and the next answer is garbled, just above that the server
  crashes, instead of the clean 503 every other architecture gives Repro and evidence: `gfx1201-16gb/long-context.md`.
