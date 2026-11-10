# GPU agent validation plan

Objective: validate real GPU porting with Claude Code, Codex and Pi, using a small
model with a published cousin, and **no needle-in-a-haystack tests**.

- Target: `Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF`, revision
  `ebb2015119c907b064c512bf053e945850b5875f`, file
  `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`.
- Cousin: `loom-fun/bartowski-Qwen2.5-0.5B-Instruct-loom`, revision
  `dad263bdf7c45f2e41c05b02df6cb7fae99f20c3`. Its generated `loom-card/2` card
  and qwen2 architecture were verified before selecting the target. This is a
  different fine-tuning of the same small architecture, not a new kernel family.
- Hardware: local AMD Radeon 8060S / gfx1151. Trials run serially with separate
  work directories and ports, leaving unrelated resident servers alone.
- SDK: the existing PyPI 0.1.0 Windows installation at `~/loom010`, source
  revision `d01d1c18b154189fd073777329052d23f6071f14`, wheel SHA256
  `d368cf8be58f00f3e9c765df4031964d2fc07fb086a9b5241830f05e95725f83`.
  Reusing a fixed SDK isolates porting behavior from installation differences.
- Trials use frozen skills and independently build/serve their own packages.
  HF download caches and the fixed SDK/reference binaries may be shared; prior
  agents' packages, fixes and transcripts are excluded from each new trial.
- `SKIP_NEEDLE=1` excludes every needle gate in dev/quick/full and prevents direct
  `needle.py` execution. Evidence records the exclusion; it is never a pass.

Acceptance for **each** agent: actual GPU compilation and serving; kernel tests;
verified chat template; golden comparison against GPU llama.cpp; consistency,
tool/history and sampling gates where applicable; measured decode/prefill and a
report stating needle coverage was excluded. Check inherited reference settings
and retain only a justified measured improvement; exhaustive tuning is outside
this validation request. No requirement to invent a speedup when already tuned.

Compare the same checkpoint and reference build across agents. Run final gates
on the final package, inspect authoritative logs and artifacts rather than only
the agents' summaries, and stop trial-owned servers between agents. If a common
script defect is found, fold back the general fix, freeze a new revision, and
revalidate affected earlier packages. Do not publish target models to HF.

Results: in progress. Raw transcripts and packages are under `.agent-smoke/gpu/`
(gitignored); verified summaries will be recorded here and in the trial ledger.
