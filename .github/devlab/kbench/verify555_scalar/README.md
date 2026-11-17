# Shipped scalar2head ×2token attention experiment

Opt-in custom suite only: `.github/devlab/verify555_scalar.json`, suite `verify555_scalar`. This branch is independent of the pinned WMMA experiment. No serving generator imports this code.

The fixed reference is the current shipped Qwen3-4B d128/GQA4 attention generator from this branch's pinned source. Its long-context body uses8-key online-softmax blocks and keeps all4heads of one token in a wave. The candidate retains4queries per wave but tiles2heads ×2consecutive tokens, reusing each K/V block across both axes. Even waves own one head pair; odd waves exit for paired spans. Incomplete, dead or mismatched groups use the original GQA path.

The span path starts at position4095, matching the shipped GQA handover. This matters: the existing short-context per-head body uses16-key blocks and has a different partial reduction tree. Each candidate query keeps its own split range,8-key alignment, masks, maximum, sum, FMA order and partial buffer indices.

Offline SDK0.2/gfx1151 compilation: baseline216VGPR, candidate149VGPR, both0LDS and no scratch operations in ISA. This is not a speed or correctness claim. DevLab compares bit-exact partial buffers for spans1/3/4/5/8crossing32k plus a4token span straddling4095. Candidate buffers start42, reference0; every tested shape has all64splits active, so skipped writes fail. The kernel benchmark interleaves5rounds of the4token32k shape.

The suite intentionally reuses deterministic input/case generation from the sibling experimental harness; its reference kernel is regenerated from shipped scalar source, never the archived WMMA generator. Any production enablement still requires kernel exactness, per-row end-to-end perf, spec and consistency gates. No fleet scope is enabled by this experiment.
