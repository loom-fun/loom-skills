# Campaign lessons (loom-org)

Detail behind the rules in SKILL.md. Rules first, reasons after.

## Publishing: verify the live card
- `make_info.py build` used to write `./info.json` (the current directory) when `-o` was omitted, while
  `render_card.py` and `publish_package.py upload` read `PKG/info.json`. Wrapper scripts that omitted `-o` published
  the OLD info.json next to regenerated programs (Nov 12: 12 cards carried gates and sessions of other programs, and
  card-only perf republishes never landed). The default is now `PKG/info.json`.
- After every upload, fetch `https://huggingface.co/loom-fun/REPO/resolve/main/info.json` and confirm the new session
  id (`performance.sessions[-1].source`) and the gates' `programs_sha256` equal the published package's; an upload
  that "succeeded" is not a published card until the live file says so.

## Memory budget
- `make_info.py build --from-info` copies the old `card` block; only `--memory-bytes` changes it. A regenerated card
  once kept its old budget while build.py printed a larger one. The build now refuses a carried value when the published gates' programs
  are not the package's; card-only republishes of the same programs are unaffected.
- loom-serve charges a near-full request about the whole KV pool plus 0.5-1.7 GB beyond build.py's estimate. Uncapped
  packages absorb that in pool headroom; a capped one (pool == context) got HTTP 503 "insufficient completion capacity"
  on a full-context request at the printed budget, on every runner, and was admitted 0.5 GB higher. build.py now
  reserves 2 GiB more workspace when capped.
- Audit `memory_bytes` across cards, nightly entries and the sessions' recorded budgets after any regeneration.

## Stale programs on DevLab
Runners can fetch the previous Hub revision for a few minutes after an upload (the run's summary shows an old
`programs_sha256` while `model.json` is unchanged). `pull_devlab.py --package` skips such jobs; re-dispatch. A
nightly or perf run that reports old programs after more than the wait is a runner program-cache bug to investigate.

## Regeneration A/B
Same runner, interleaved rounds, mean of several: published programs vs regenerated, on decode, decode_32k, prefill
1k to 32k, streams4. Runners differ between rows; only an on/off pair from one runner is comparable. Past cases:
- Cards generated from an unmerged branch had 2x the 32k decode of the same model rebuilt from main.
- A multi-token verify attention won +21 to +107 % at 8 KV heads and lost 7 to 30 % at 2-4 KV heads, hence a
  minimum-KV-heads gate.
- Fixing a family-wide prefill weakness (flash subtiles by query-heads-per-KV-head, MoE and gated-delta-net prefill
  kernels) cleared whole groups of `far` flags at once; per-model tuning did not.

## Gate rules
Principled noise rules measure llama.cpp's own variation at the divergence (re-scoring under its other numerics,
capped), accept only more than before, and ship tests with constructed cases and mutants that must still fail. A
recording from llama.cpp with its image encoder on the CPU flipped its own top-1 token on one image; Loom matching
that recording is evidence, a larger constant is not. See `loom-benchmark` for the rules themselves.

## Correctness bugs the sweep surfaced
- Tokenizer: loom-serve's tokenizer differs from HF's when a later pre-tokenizer stage splits pieces further (an
  individual-digits `Digits` stage); build.py rewrites it (`loom-port-llm`). Compare token counts on long text.
- Batch-dependent prefill attention numerics made one coder model fail consistency (the same prompt alone vs in a
  batch): prefill attention tiles must not depend on what else is in the batch.
- Kernel paths can depend on KV heads (above). None of these show in golden compare; the full set found them.

## Orchestration
- State: a handoff file read first after any restart (workers, PRs, queue, rules, next actions), plus a status page
  built from `org_status` output with a log and an ETA. A page update is counts, log, ETA and scope changes together.
- Worker brief: the rules (GPU, DevLab, quant, publishing), PR acceptance criteria, and a final-report format. Fresh
  workers are cheap; resumed ones keep context. Workers' reports are summaries; verify with `org_status`.
- Reviews: independent reviewer per PR, against the failure it fixes and the criteria. Gate-rule PRs get a strict one.
- tmpfs: a scratchpad in `/tmp` was RAM; GGUF copies and kernel-test tables there crashed a campaign. Use disk.
- Upload refusals by a permission classifier: the worker reports the exact command; the orchestrator checks the card
  and runs it.
- Local GPU crash: concurrent servers exhausted memory and restarted the container, stopping every worker. Hence the
  single local-queue owner.
