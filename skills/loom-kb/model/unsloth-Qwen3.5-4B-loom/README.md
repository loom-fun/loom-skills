# Model: unsloth-Qwen3.5-4B-loom

Qwen3.5-4B Q4_K_M (qwen35 hybrid, DFlash drafter, vision). Device settings beyond the family's:

- **gfx1150** (Strix Point) (gfx1150, 32 GB laptop, GTT 15 GiB): served context 63488 (`exec.context_capacity`, `exec.context_cap`; the native 262144
  does not register under the GTT budget on hrx-loom 0.2.0). Draft depth 3 and prefill chunk 1024 stay: they measured best there.
  Measurements and provenance: `fleet/devices/gfx1150.json`; device facts: `device/rdna3/gfx1150`.
