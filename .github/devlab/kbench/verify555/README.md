# Issue 555 verify-span SDK experiment

Explicit opt-in only: `.github/devlab/verify555.json`; no serving generator imports these files.

`source_mma.py` is the parked `perfgap-verify32k-mma` generator, retained here as an experimental control. `gen.py` supplies the current page-address helper and migrates the obsolete vector constant spelling to SDK0.2. The candidate stages the16query columns once into4KiBLDS, then loads their fragments near each QK MMA. This preserves the MMA and softmax operation ordering. It is restricted to T4/G4/d128 (Qwen3-4B geometry).

Offline gfx1151 compilation: baseline218VGPR/18944LDSbytes; candidate168VGPR/23040LDSbytes. Both still emit488scratch instructions. This is a codegen observation, not a speed claim. Historical full waitcnt drains do not reproduce unchanged with SDK0.2: QK loads have partial waits. Pure independent WMMA controls omit the pressure from real attention.

The DevLab suite compares every partial accumulator and max/sum bit for spans1,3,4,5,8 crossing32k boundaries, then interleaves baseline/candidate kernel timing at T4/32k. Passing this experiment alone does not establish equivalence with the shipped scalar path, spec/consistency gates, or end-to-end improvement. Those are required before any production enablement.
