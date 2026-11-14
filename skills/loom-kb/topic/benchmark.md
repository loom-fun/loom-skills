# Topic: benchmark (gate internals, golden interpretation, the Hugging Face card)

## Contents

- How golden.py and compare.py work, and how to read a divergence
- Hugging Face model card (loom-fun repos)

## How golden.py and compare.py work, and how to read a divergence

- **How golden.py works.** It renders each prompt with the package's chat block
  and sends the *token ids* to llama.cpp `/completion` (greedy, `n_probs: 5`). Both
  engines therefore see identical prompts: 20 short prompts × 64 tokens, plus 1
  prompt per long document × 64 tokens (the two question kinds alternate across sizes).
- **How compare.py works.** It asks Loom for the same completion through chat,
  rebuilds the raw text from the reasoning/answer channels, re-tokenizes it, and
  finds the first divergence. It reports the **golden log-prob gap**
  logp(golden) − logp(loom token) at that point.
- **Interpretation.** Greedy decoding through different summation orders
  *always* diverges eventually at near-ties.
  - **PASS**: every divergence is a near-tie (`bench_common.near_tie`): gap ≤ 1.0 with the token in the
    golden top-5, or Loom picked llama.cpp's second choice where llama.cpp's first has p < 0.5
    (llama.cpp itself is unsure there: on flint its own re-recording moved by gap 1.05).
    Typical healthy ports have max gap 0.1–0.6, and full matches on a quarter or
    more of the prompts.
  - **The bar includes llama.cpp's own rounding noise at that position.** A divergence the rule above
    rejects is judged again: llama.cpp re-scores the same position (prompt + its own tokens before it)
    under two numerics it uses itself, the context in one batch (MMQ matmuls) and in micro-batches of 8
    (MMVQ: another q8_1 dot-product formula and summation order, and the decode path a reused history
    took; `bench_common.llama_views`). The noise is how far the gap moves across the golden and the
    views that list both tokens, capped at 0.5. The divergence passes when the smallest of those gaps is
    ≤ 1.0 + the noise, or when a view ranks Loom's token and not the golden's (llama.cpp itself prefers
    it: gemma-4-26B, `_` at p 1.0 token by token is unlisted batched, where `/` is top-1). A view without
    Loom's token never widens the bar (`test_near_tie.py`).
    Why: both engines round activations to 8 bits at every matmul, so a single rounding-level change
    (f16 vs f32 KV, MMVQ vs MMQ, ties-to-even vs away) can move a gap by about 1 nat where the model is
    truly tied (gemma-3-1b, first token: an f64 forward puts the two candidates 0.004 apart, the golden
    gap is 1.36, the views 1.28 and 1.06). A fixed larger constant would let real bugs through everywhere.
    Limit: a mild long-context RoPE error (base 5e5 vs 1e6) is below this bar's resolution, since
    llama.cpp's own views disagree that much there (16k: 1.24 golden, 0.98 and 1.44 views).
    Only the first divergence of a prompt (turn, probe) is judged, as before. compare.py, agent_golden.py,
    spec_check.py uses it (`near_tie_at`); vision_golden.py (image prompts) does not. No
    micro-batch view where llama.cpp cannot extend its cache in small steps (recurrent state).
  - **FAIL**: a large gap, or a token outside every view's top list. That is a real bug, even
    if the text reads fine.
  - Two llama.cpp builds (Vulkan vs ROCm) disagree with each other in the same
    near-tie way, which is useful context for readers.
- llama.cpp ROCm itself changes greedy output between identical runs at near-ties
  (seen at 1k and 16k on Qwen3-4B). That is why there are two recordings: `--alt`
  scores each prompt against whichever recording Loom follows further. Before
  declaring FAIL on a single prompt, re-record it; a gap that disappears was noise.
  - A plain re-run of a *text* prompt at `-np 1` is often bit-identical, so it proves little.
    The informative check is llama.cpp's sensitivity to its own summation order. The gates measure
    it at the divergence already (one batch vs micro-batches of 8, above). For a FAIL that remains,
    re-record the one prompt with the server started under other legitimate numerics: `LLAMA_SPEC="-ub 128"`,
    `"-ub 2048 -b 2048"` and `"-fa off"` (llamacpp_server.sh appends LLAMA_SPEC). Then compare
    the top-5 at the divergence.
  - If llama.cpp's own top-1 changes across these settings, or Loom's token falls within the
    1.0 gap under one of them, the FAIL is a near-tie. Document it with the per-setting top-5;
    don't change the kernels to chase it. Example, Flint 15.5k summary at token 15: llama.cpp
    picked ` across` (default), ` combine` (`-ub 128`, `-ub 2048`) and ` summarize` (`-fa off`).
    Loom's ` final` had gap 1.39 (default), 1.37 (`-ub 128`), 1.06 (`-ub 2048`) and 0.72 (`-fa off`).
  - If llama.cpp stays on one token under every setting and Loom's gap stays above 1.0, it is
    *not* proven noise. Before changing kernels, check that both sides compute the same model:
    - Near-exact reference: dequantize the GGUF to F32 (`llama-quantize --allow-requantize IN OUT F32`)
      and run it on the CPU (`LLAMA_SPEC="-ngl 0 --no-op-offload -nkvo --no-mmproj-offload -fa off
      -ctk f32 -ctv f32"`). If it agrees with llama.cpp's GPU run, Loom is the outlier.
    - Vision: dump llama.cpp's image embeddings (`MTMD_DEBUG_EMBEDDINGS=out.bin llama-mtmd-cli ...
      --image X --jinja`: int32 rows, int32 width, then f32 data) and compare them with Loom's
      encoder. Row cosines far below 0.999 are a preprocessing or encoder difference, not numerics.
      Feeding llama.cpp the image already resized to the processed size isolates the resize.
    - Example, Bolt Q4_K_M vision `photo` (1024x768) at token 7: 17 recordings with llama.cpp b1355
      all chose ` grid` (gap 1.02–1.54 to Loom's `,`), and the F32 CPU run agreed (gap 1.40). Loom's
      kernels were not the cause: f32 activations, matvec-only prefill and attention variants all
      gave the same reply. b1355 letterboxed Gemma 4 images (row cosine to Loom's embeddings 0.32 at
      worst); upstream llama.cpp 7bd18083 (lemonade b1368) resizes without padding, as Loom does.
      Against b1368 all six images, `photo` included, match token by token (57 tokens).
- Golden ids come from llama.cpp's `tokens`. `completion_probabilities` omits
  byte-fallback tokens (emoji, partial UTF-8), so top-5 lists are kept only while
  they line up.
- Document sizes are approximate. With a 262k-vocab tokenizer (Gemma),
  `doc_32000` is about 35k tokens, which is over a 32768 context. Either build the
  package (and llama-server) with a 40960 context, or use `--long 1000,2000,16000`
  and say so.
- Long goldens (16k/32k) are what catch KV paging, prefill chunking and attention
  masking bugs.
- `consistency.py` catches mixed-sequence epoch bugs (padding, tile uniformity, fallback lists) and
  races (each prompt also runs alone twice and must be identical on the same path). A concurrent-vs-alone
  difference passes only when every prompt is deterministic alone and every divergence is a near-tie by the
  compare rule (gap ≤ 1 nat, measured with `--pkg/--llama`): decode tokens that share an epoch with a
  prefill go through the f16 GEMM path instead of the f32 matvec path. Anything else is a bug.
  "Same path" means equal `cached_tokens`: a repeat that hits the prefix cache (its history retained from
  the concurrent phase) recomputes only the uncached tail with the decode-shaped kernels, so it can flip a
  near-tie against the first, uncached run without any race (issue #39); such a prompt runs alone again
  until two runs share a path, and the hit-vs-miss difference takes the near-tie rule. To hunt a suspected
  race, `determinism.py URL --model ID --conc 1 --reps 20` (alone, never a cache hit: any difference is a
  race), `--pairs` (cache miss vs hit) and `--conc 4` (mixed epochs) separate the three causes.
- 503 errors on long prompts mean `--memory_bytes` is too small for the KV pool plus the prefill scratch
  (loom-kb `porting` topic, bring-up debugging order step 5; even on a fresh server, whatever the message says about retained state),
  not a model bug.

## Hugging Face model card (loom-fun repos)

A published card's `README.md`, `info.json` and `REPORT.md` are **generated, never hand-written**: the skill
supplies data, `scripts/card/` turns it into the card. Every loom-fun card has the same sections, rows and
footnotes (lead-in, quick start, performance vs llama.cpp, features, correctness gates, files); only the values
differ. No prose of your own on a card: no "what's new", no explanations, no caveats. A value without evidence
renders as not measured / no evidence / unknown; never fill one in by hand. A card with failing gates is published
as it is (`loom-org` step 2): each renders as fail, and only new evidence changes it.

`info.json` (schema `card/info_schema.json`, `"schema": "loom-card/3"`; a `loom-card/2` file reads as its one GPU, Strix
Halo) holds only what no package file records. Per GPU (`devices.strix-halo`, `devices.radeon-r9700`: card_lib.GPUS): measurement sessions (per-run values, commands, the baseline's version and exact server arguments),
the spec gate's acceptance, gate evidence, the device and Loom SDK of each source, dates and versions. The card
reads everything else from the repo's own files: sources, revisions, sha256 and sizes from `model.json.lock`;
context, rows, presets and drafting depth from `model.json`; features, the drafter and the generating
loom-skills commit and SDK from `llm_config.json` (`"generator"`, stamped by `build.py`).

1. Evidence files (`loom-card-evidence/2`) from the run's own outputs, made **on the machine that measured**
   (each records `scripts/machine_info.py`'s detected device and the SDK under `$LOOM_ROOT`):
   - `card/make_info.py perf --perf-log bench/perf_log.jsonl --baseline-version llamacpp-rocm/bNNNN
     --baseline-args "$(cat bench/llamacpp_args.txt)" --package PKG -o ev_perf.json` (finalize.sh's `final` /
     `llama-final` rows and the llama-server arguments it recorded; or `--loom-jsonl`/`--llama-jsonl` +
     `--c4-loom`/`--c4-llama` for separate same-session logs, `--command` for each measuring command, and
     `--metrics` for a later session that re-measured some rows);
   - `card/make_info.py gates bench/gates.json [ktest.out --time ...] --package PKG -o ev_gates.json` (also the
     spec gate's `acceptance N% of M proposals` from `gates_spec.log`);
   - `card/pull_devlab.py loom-fun/REPO --package DIR -o ev_devlab.json`: the latest nightly-gates run on AMD
     DevLab Dispatch: gates of every finished level (job URL as evidence), and the spec gate's acceptance from the
     job's `gates_spec.log` as the card's `spec_acceptance`, stamped with the job's programs (a drafting statistic of
     model, drafter and text, not a timing: it counts from any source). A perf session recorded with
     `--source-kind devlab` (a scripted DevLab port, a `devlab-perf` job: `loom-org` step 3) is the card's performance
     (`card_lib.card_metrics`: per metric the last session); `perf` refuses `--source-kind local`, so finalize's
     `perf --perf-log` record is a development/report record, not card evidence;
   - `card/make_info.py roofline --package DIR -o ev_roofline.json` (tools venv; CPU only, reads the GGUF from the
     Hugging Face cache): the hardware ceiling of every card metric (`loom-roofline` `roofline.py`) with the
     device constants and the roofline.py sha256 and commit. Drafting packages: decode and 4-stream ceilings are
     verify passes per second x (1 + acceptance x depth) tokens per pass (the drafter's cost not counted: an upper
     bound). Per row the acceptance is the one measured in the same session as the card's value (perf.py records
     each decode request's, c4_steady.py its streams', from loom-serve's `/metrics` counters); a row whose session
     recorded none falls back to the spec gate's, and the card says so (the gate's text can accept far more or
     less than perf.py's essay: 41 % vs ~17 % on a small drafted model, a 1.8x too-high ceiling). Without either
     they have none. A first card has no `DIR/info.json` yet: pass the run's evidence (`--evidence ev_perf.json
     ev_gates.json`), or build once, run roofline, and build again.
   Cards for two GPUs (methodology rule 12): pass `--gpu strix-halo|radeon-r9700` to `perf`, `gates`, `roofline` and
   `pull_devlab.py` (the R9700's roofline uses `roofline.py --device radeon-r9700s`, set by `--gpu`); a GPU without evidence is
   `--gpus strix-halo,radeon-r9700` on `build` (`not_measured`), `build --fit VERDICT.json` records the R9700's memory
   budget and served context or marks a package that does not fit it, `--org loom-fun-staging` writes the card's repo under
   another org.
2. `card/make_info.py build DIR --repo loom-fun/REPO --memory-bytes N --output OUT --report generated
   --evidence ev_*.json -o DIR/info.json` (`--generator-commit` defaults to this checkout's HEAD; to regenerate a
   published card with only new evidence: `--from-info DIR/info.json`, re-read from the Hub right before, which
   also supplies `--repo`). DIR holds the repo's `model.json`,
   `model.json.lock`, `llm_config.json` (and `program/`). Per gate the latest evidence wins, so a later failing
   run shows as a failure, except that on the same programs a later pass at a lower level (dev < fast < quick <
   full) keeps the earlier higher-level pass (a published date-only gate and a same-day timestamped job: the new
   evidence is the later); per metric the last session that measured it (`build --from-info` drops performance sessions not from DevLab/factory
   and old `devlab` sweeps; `--reset-performance` drops the old sessions to re-measure from scratch). `--report attached` instead keeps a
   REPORT.md written by the run (the factory's) or an older card's narrative, as is.
   A new or replaced package: `card/publish_package.py prepare PKG DIR --source ORG/GGUF-REPO` first, on the machine
   that holds the GGUFs (copies the package without weights, pins each file's origin at the revision it was validated
   on, writes `model.json.lock` with `loom-fetch --update-lock`), then `make_info.py build DIR ...` on that DIR.
3. `card/render_card.py DIR` writes `DIR/README.md` (and `DIR/REPORT.md` for `--report generated`: setup,
   commands, every run's value, the roofline, gate evidence). The performance table shows per metric llama.cpp,
   Loom, the margin, the roofline and Loom as % of it, then the Vulkan baseline as one more column; footnotes say
   what the roofline is. Loom-only DevLab
   sweeps that older cards recorded in `info.json` (`devlab`) stay valid and are carried over by `--from-info`, but
   are not rendered. When republishing a card, upload only those three files: `publish_package.py upload DIR --repo
   REPO --card-only --parent-commit SHA` (SHA: the repo commit whose info.json the card was built from; the upload
   fails instead of overwriting a card another worker published since). Never re-upload a repo's programs, `model.json`, lock or weights for a card change.
   A whole package: `publish_package.py upload DIR --repo REPO` (removes files an in-place update leaves stale).

**Performance guardrails** (`card_lib.perf_flags`, applied by `loom-org`'s `org_status.py`): a metric whose Loom
value is below half the efficiency Loom ports have demonstrated (far from roofline) sends the model back to a worker
with `loom-optimize`, and to a deep debugging session if it stays far after that (`loom-org` step 5). A
value measured on other programs (stale) is re-measured with a fresh `devlab-perf` job first. Flags are only `far`
and `stale`.

The card's Versions line shows the schema, the loom-skills commit and hrx-loom build that generated the
programs, and the card generator commit. `loom-setup/scripts/install_loom.sh` writes `$LOOM_ROOT/loom-install.json`
(version, hrx-system commit, CI artifact or PyPI release, wheel sha256), which `build.py` stamps into the package
and the evidence records for every measurement. A schema change that only adds keys, metrics or gates keeps
`loom-card/3` (readers ignore unknown names); renaming, removing or retyping a key bumps it.
To change what every card says, change `render_card.py` (and `card/test/check.sh`'s golden files with
`UPDATE=1`, CI `check-cards`), then re-render all cards. `card/import_card.py` maps cards published before
loom-card/1 to evidence (one time; their versions are recovered from the factory run and the report they
name, and their narrative report becomes REPORT.md verbatim). The loom-factory workflow's publish step runs
steps 1-3 itself and uploads the run's REPORT.md.
