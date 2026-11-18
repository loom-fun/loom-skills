# Model: unsloth-Qwen3.5-9B-loom

Qwen3.5-9B Q4_K_M (qwen35 hybrid, DFlash drafter, vision). Device settings beyond the family's:

- **gfx1150** (Strix Point) (gfx1150, 32 GB laptop, GTT 15 GiB): served context 40960 (`exec.context_capacity`, `exec.context_cap`; the native 262144
  does not register under the GTT budget on hrx-loom 0.2.0, and above ~41k a full-length request crashes the server or garbles the next answer
  instead of a clean 503). Draft depth 3 and prefill chunk 1024 stay the card's.
  Measurements and provenance: `fleet/devices/gfx1150.json`; device facts: `device/rdna3/gfx1150`.
