# Benchmark methodology: the one normative reference

Every card number, report and "Loom vs llama.cpp" claim follows these rules. The numbers are data in `methodology.json` next to this
file (a test checks that the numbers stated here match it). Other skills link here and summarize at most
two or three lines; if they disagree with this file, this file wins and the other text is a bug. Code: `scripts/card/matched_spec.py`
(`TIE_BAND`, `outcome`, `tally`, `rule_violations`) and `data/drafting_classification.json`; tests: `scripts/card/test/check.sh`.

## Contents

- Drafting (one choice per model)
- Which configuration each row runs
- Reading the result
- Card text and publishing (incl. rule 15: reuse stored measurements)

## Drafting (one choice per model)

1. **One drafting choice per model, used by both Loom and llama.cpp (ROCm and Vulkan).** In order: an MTP head in the card's own
   GGUF (`nextn_predict_layers`), else a DFlash drafter/assistant published for this exact model by its maker or Z-Lab (or a
   faithful GGUF conversion of that release), else no drafting. Never n-gram (opt-in experiments only, never on a card). Never a
   drafter trained for another model. Provenance you cannot verify means no drafting. Decided by availability, never by which
   engine is faster. Source: `data/drafting_classification.json` (+ `.md` with the provenance links). *Why: a speed-picked
   configuration is best-of-configurations in disguise.*
2. **Draft depth is per engine and need not match.** Loom: its tuned depth. llama.cpp: the drafter's documented
   `--spec-draft-n-max`, else llama.cpp's default. Both are recorded on the card. *Why: depth is engine-specific; the Nov 13 sweep
   showed tuning llama.cpp's n-max moves its geomean 0-6 % against the default, mostly noise.* (`pick_nmax.py` and the `:N` n-max
   steps stay as a diagnostic, not a card input.)

## Which configuration each row runs

3. **Single-stream rows (decode, decode_2k, decode_32k, prefill_1k..32k): each engine is loaded once with that choice for all rows.**
   Runtime-adaptive behaviour a real user also gets is allowed (Loom's `ctx_max` / `occ_min` gates). No per-metric load-time setting
   changes, no best-of-configurations. *Why: a user loads a server once.*
4. **streams4 is the one load-time exception: both engines are loaded with drafting off and serve 4 parallel requests** (Loom: the
   same package with a `draft_depth` 0 override, variant `nodraft`; llama.cpp: a 4-slot server without `--spec-*`).
   *Why: at 4 concurrent streams drafting usually costs more than it saves (measured on most cards); this is the one load-time exception the methodology allows, applied to both engines.*
5. **Fair measurement.** Same GGUF, same DevLab runner and job, same prompts (nonce'd; no prompt-cache reuse), Loom with the active
   wait, timing source disclosed (Loom client-side, llama.cpp server `timings`), one pinned llama.cpp build per backend. Compare
   engines only within one runner/job (ratios, never across runners). **Sampling, identical for Loom and llama.cpp (ROCm and
   Vulkan): every metric is the MEDIAN over sessions (rounds); a session's value is the median of its samples.** A session is one
   serve session of the engine (a llama.cpp round: one `perf.py` run). A fast model (published or expected decode above ~100
   tok/s) gets 6 sessions, and `perf.py --repeat-rows 3` measures decode_2k, decode_32k and prefill_8k three times per session
   (median of 3 samples); every other model, and every other row, 3 sessions of one sample. streams4 is the median of its
   `c4_steady.py` runs (a session's runs, then the median over sessions). The Strix A/B uses the same code
   (`make_info.loom_sessions`, `card_lib.sampling_plan`). The card says which method each value was measured with
   (`aggregate: "median"` on the session; a session without it was measured earlier as the mean of N rounds and is never
   relabelled). *Why: a fast model's rate is bimodal per serve session and a mean follows the outlier.*
6. **A backend that cannot run the chosen drafter: that cell is "not supported" with the reason.** Never substitute another
   configuration. *Why: a substitute is a different comparison.*

## Reading the result

7. **Outcome per row against the better llama.cpp backend**: win > +3 %, tie within +-3 % (`TIE_BAND`, the measured same-runner
   noise), loss < -3 %. Not-supported and unmeasured cells are neutral. Cards and reports give three counts: wins every row,
   wins or ties every row, loses at least one row. Report the honest count; integrity beats win count.
8. **Correct means every required gate passed on the live programs (`programs_sha256`).** A gate stamped with another build renders
   "not re-run on this package", never as a pass. A card with a failing gate is published as such; gates are never loosened.
9. **Roofline.** Acceptance counts only from the displayed session on the current programs; missing or 0 acceptance gives the
   single-pass ceiling for that row; streams4 is always single-pass; MoE streams4 uses the shared-experts bound; no ceilings for rows
   beyond the native context; a value above its ceiling gets a footnote, never a tuned number (`loom-roofline`).

## Card text and publishing

10. **Card text.** A public card holds only what `render_card.py` generates, plus short caveats. Allowed: the title and first
    paragraph (with "Learn more on the Loom ecosystem website"), Quick start (pinned install, `loom-fetch`, `loom-serve`), the
    performance table with its generated Methodology bullets (5-8, same template on every card: what is compared, the drafting
    choice and each engine's depth, 4 streams undrafted, what the rows mean with the tie band, the server arguments, every "not
    supported" cell with its reason, the measurement sessions) and footnotes, Features, Correctness (a gate without evidence says
    "no evidence"; one measured on other programs says "not re-run on this package"), Files, versions and sources, and, only for
    a paused card, Next steps. `REPORT.md` is generated from `info.json` (never attached or hand-written). Never hand-write a
    paragraph, and never put history in text: no "I/we", "tried", "previously", "ruled out", ideas, run/PR/agent chatter, local
    paths, or what a card does not do beyond the standard. A caveat is one short factual line (`differs`, at most 240 characters,
    one sentence): `not supported: <reason>`, `MTP head present but not yet supported by Loom (<reason>)`, `llama.cpp loads <repo>
    (a conversion of the same <drafter> drafter)`, a gate not run / not re-run. `card_lib.free_text_errors` (run by `validate`,
    `render_card.py`, `publish_package.py upload`) and `card/test/check.sh` enforce this. The hrx-loom install is pinned to the
    version the card recorded.
11. **Publishing.** `make_info.py build PKG -o PKG/info.json` (always `-o`), `publish_package.py upload ... --parent-commit SHA`,
    then fetch the live `info.json` and confirm the new session id (`performance.sessions[-1].source`) and the gates'
    `programs_sha256` equal the package's before saying "published". Publish partial progress; never fake a cell.
12. **Cards for two GPUs** (`loom-card/3`: Strix Halo `strix-halo`, gfx1151; AMD Radeon AI PRO R9700 `radeon-r9700`, gfx1201, 32 GB).
    Every rule above holds per GPU: each GPU has its own table, measured in one job on one runner of that GPU (never compare or
    mix values across GPUs), its own llama.cpp ROCm and Vulkan baselines on that runner, its own roofline (`roofline.py --device`:
    `strix-halo`, `radeon-r9700s` for the R9700 DevLab runner) and its own gate evidence. Evidence names its GPU (`make_info.py
    perf|gates|roofline --gpu`; else the GPU its source device names, else Strix Halo). The package is the same on both: one
    `model.json`, the context is native on a GPU with room for its KV cache; on a GPU whose memory holds less (`--fit`: weights + KV
    + 2 GB must fit the GPU's memory), `--memory_bytes` is that GPU's memory, the card states the context served there (`served_context`)
    and a row or gate that needs more context than that (`decode_32k`, `prefill_32k`, `golden`'s 32k documents) reads `doesn't fit` in every column of that GPU (Loom, llama.cpp, roofline): an accepted outcome, not a failure and not "no evidence". `--memory_bytes` for the R9700 is 34.0 GB (the board reports 31.86 GiB). A package that does not fit a GPU's memory is that GPU's `unsupported` state (one factual
    line). A GPU without results for this package is `not_measured` (one factual line, no numbers copied from another package or
    program). Publishing goes to one org (`publish_package.py upload --org`, default `loom-fun`); the staging flow uses `--org
    loom-fun-staging` with the same repo names, and a card whose `info.json` repo is another org is refused.
13. **A/B verdicts use a band per row** (`methodology.json` `rows`; same runner, ABBA, medians of sessions; for unchanged inputs only to confirm a row that moved, rule 15). A row PASSes when its new/old
    ratio is within the band; there is no geomean: a gain on one row never pays for a loss on another. Not the card's tie band (rule 7),
    which stays +-3 % on every row. Bands (+-, and the ratio below which a 2+2-session canary rejects):
    <!-- bands -->
    - prefill_8k, prefill_16k, prefill_32k: +-2 %, reject below 0.95
    - prefill_1k, prefill_2k: +-3 %, reject below 0.94
    - decode: +-3.5 %, reject below 0.92
    - decode_32k: +-3 %, reject below 0.92
    - decode_2k, streams4: +-4 %, reject below 0.92
    <!-- /bands -->
    The bands were checked on true A/A pairs (same programs on both sides, `band_calibration_aa` in `methodology.json`): a row whose A/A
    spread exceeded its band in more than 5 % of pairs was widened (decode: 3 -> 3.5 %). A ratio within 2 points of its reject threshold extends 2+2 sessions to 6+6. `strix_ab_compare.py --band X` forces one flat band.
    Acceptance rows are informational: reported, never a rerun or out_of_margin verdict alone. The ratchet's two runs may be concurrent replicas of one submission (`ab_jobs.py --replicas 2`) on different runners (`promote_check.py` records the runner names; one runner twice = two session sets, said so); the rule is unchanged. `--sequential` extends 2+2 to 6+6 inside the job. **Runner health:** an OLD side >10 % off the published card (Strix: loom-fun; R9700: staging) on any decode/prefill row (`runner_guard`) makes the run `degraded_runner` (compare verdict `rerun`): never a promotion or a regression; rerun on another runner.
14. **An unchanged default path keeps its evidence.** When `fleet/stage_equiv.py` proves a package's default-path stages identical to the measured one's, `make_info.py equivalent` carries that GPU's performance rows (the card names the proof); any other change re-measures.
    Host logits are the gates `logits` and `gufo` (`devlab_perf.sh` steps `logits:V`, `gufo:V`), never a performance row. The `logits` gate's llama.cpp comparison follows the golden (`logits_check.py`): top-1 equal per step (a divergence is judged by `near_tie` and ends that prompt); the other top-5 tokens both engines list agree within a tolerance by probability (larger p >= 0.5: 1.0 nat, >= 0.01: 1.5, >= 1e-4: 3.0, below: not compared), tiers set from the measured differences of one GGUF through the two engines' quantized matmuls (small dense <= 0.7 nat; mid-size/MoE 1.2 at p~0.1, 2.7 at p~1e-3); token spellings are normalized (special tokens, a sentencepiece piece's leading space) and tokens that print alike count by their best entry; llama.cpp's references are recorded and its server stopped before Gufo loads.
15. **Unchanged inputs keep their measurements (owner rule).** A stored row stays valid, and is reused, while every input it measured is unchanged, on the same GPU type (`strix-halo`, `radeon-r9700`, ...; any runner of that type): the model files (the GGUF, its drafter or MTP head, and for Loom the package programs), the engine binary (the llama.cpp build tag per backend, ROCm and Vulkan; the hrx-loom version), and the run configuration (llama.cpp server flags and drafting classification, Loom preset and wait mode). Never re-run a row only to have it "in the same job". An SDK or package change re-measures Loom only, never llama.cpp; a card reuses its stored llama.cpp rows; an old-vs-new A/B compares against the old package's stored rows and runs a same-runner ABBA job (rule 13) only to confirm a row that moved beyond its band (`methodology.json` `rows`). The card names each side's run and date, so a cross-job comparison is stated. Re-measure only when an input changed, the stored row came from a `degraded_runner`, or the row was never measured.

Local qualification (`qualify.sh`, `llama_spec_tune.py`, `tune.py`) is a development loop: its tables guide the optimizer, are
not card numbers, and may try configurations the card rule excludes. Audit a card with `matched_spec.rule_violations`.
