# fleet/: the 51 published models as build inputs

Dev infrastructure (not installed with the skills). A change to the kit is judged by what it does to the programs of every
published model, without a GPU and without any weights.

| file | what |
|---|---|
| `manifest.json` | per model: staging repo@revision, model id, kit example, build args (context, pool, prefill, rows, max_tokens, draft_depth), every GGUF it reads (repo@revision, size, dump) |
| `<slug>/llm_config.json` | the published config (minus `tensors`, `generator`, `draft`: outputs of a build) overlaid on `inspect_gguf.py` of the GGUF |
| `<slug>/tokenizer_min.json` | the tokenizer's added tokens (content, id): all the programs read of the tokenizer (`gen_llm.py` stop-token restore) |
| `<slug>/gguf/<file>.json.gz` + `blobs/` | GGUF header dumps (metadata, tensor infos, size, rope_freqs data); vocab arrays and chat templates in the shared content-addressed `blobs/` |
| `gguf_stub.py` | `dump` a GGUF (file or URL, HTTP range reads) / `stub` a sparse header-only file `inspect_gguf.py` and `build.py` accept |
| `st_stub.py` | `dump` an HF safetensors checkpoint (directory or repo, HTTP range reads of the headers) into `<slug>/safetensors/<file>.json.gz`: tensor table, config.json, generation_config, tokenizer_config, chat template, added tokens; `build_fleet.py` builds such a model with `inspect_safetensors.py` from the dump (manifest `files.<name>.kind: safetensors`) |
| `add_model.py` | add or refresh a model from its package directory (reads only small files) |
| `build_fleet.py` | `--ref REF` / `--tree PATH` `--out DIR`: programs-only build of the fleet, CPU only, both targets; `--own-inputs` (with `--ref`, the impact base side): a model whose inputs changed since REF is built from REF's own inputs |
| `impact_plan.py` | CI input-based selection, independent base/head CPU shards, and fail-closed aggregation; scope lint always runs |
| `diff_fleet.py` | `BASE_DIR HEAD_DIR --json --md`: per model changed files, blocks, GPU target, card rows, strata, model.json |
| `scope_lint.py` | `--base REF --pr-body FILE [--impact IMPACT.json]`: does a PR stay inside its `loom-scope` block (files per kind, scope leak, protected files, registry legacy values, kb lint); `--local` builds the impact itself |
| `ownership.json` | protected core files, files each PR kind may change, kernel family -> files and rows_map rule ids (data; protected) |
| `route_load.py` | instruction load: lines a port / optimize run reads per fleet model (routed files + workflow and kb SKILL.md) vs the pre-refactor set at `--base`; `--md`, `--json` |
| `replays/` | measure 2: past PRs replayed through the guards (`replay.py NAME`, `mkpatches.py`, `summarize.py`); evidence per PR and `REPLAYS.md` |
| `sdk_delta_report.py` | `EVIDENCE.json [-o REPORT.md] [--band 3] [--strict]`: the per-model per-GPU SDK delta report (version entry section 5b) generated from `regen.py evidence`: verdict per pair (two thirds of the rows >= -band %), delta per row, every regression over the band with its explanation (`--strict` exits 1 on an unexplained one) |
| `sdk_inventory.py` | `snapshot` the production org's public package repos (org/repo -> head) / `drift` between two inventories: the accepted inventory of an SDK migration |
| `regen.py` | regenerate many cards, one command per step (`plan submit poll collect rerun ingest publish status accept adopt`), state in one JSON file; `--mode card` for a new device: no A/B |
| `devices/` | per-device results of GPUs with no DevLab runner (gfx1201 16 GB class, gfx1150): per model fits / max context / serve flags / preset overrides / evidence with provenance; schema `devices/device_results.schema.json`, `devices/README.md`; `make_info.py build` turns an entry into the card's GPU block, preset and section |
| `rows_map.json` | kernel family -> card rows / gates (data; first matching rule wins) |

    python fleet/build_fleet.py --ref origin/r9700 --out $WORK/base --jobs 12      # ~2 min on 12 cores
    python fleet/build_fleet.py --tree .           --out $WORK/head --jobs 12
    python fleet/diff_fleet.py $WORK/base $WORK/head --json impact.json --md impact.md

Needs `python` with `gguf` (numpy) installed (inspect_gguf.py). The manifest's inputs are the CURRENT staging packages; a ref
is built with its own kit but these inputs. Proof that stubs lose nothing: `build_fleet.py --real DIR` builds from the HF
cache GGUFs and full tokenizers instead; the programs and model.json must be byte-identical to the stub build.
Adding a model: `python fleet/add_model.py SLUG PACKAGE_DIR --staging-revision REV` (PACKAGE_DIR = snapshot of its staging repo).

## Regenerating cards: `regen.py`

    regen.py plan --base BRANCH --device radeon-r9700 --mode card --all --state S.json   # BRANCH: the integration branch; a new device: one card measurement per model, no A/B
    regen.py submit S.json; (loop, >= 120 s:) regen.py poll S.json; regen.py collect S.json; regen.py ingest S.json
    regen.py plan --base BRANCH --device radeon-r9700 --replicas 2 --sequential --no-baselines --priority SLUG --state T.json   # a narrow change: the pair in ONE submission
    regen.py submit S.json --max-inflight auto                               # a sweep in waves (the device's runner count); a --priority plan waits one wave at most
    regen.py rerun S.json && regen.py submit S.json                          # ONE confirming -r2 job where the ratchet asks for it
    regen.py publish S.json --dry-run                                        # card-only upload to loom-fun-staging (no flag: uploads)
    regen.py package S.json --dry-run                                        # sdk-ab: the v0.2 package to staging before ingest (SDK A/B, below)
    regen.py status S.json --status-json status.json                         # models.<slug>.<gpu> stage / verdict / run
    regen.py plan --base BRANCH --device strix-halo --device radeon-r9700 --mode sdk-ab --all --state S.json   # SDK migration A/B (below)

**SDK A/B (`--mode sdk-ab`)**: per model and device ONE job, ABBA on one runner: `fun` = the published staging card (the staging head at plan time; the manifest revision can lag it) served on
its own pin (hrx-loom 0.1.0), `staging` = the package BUILT in the job from the job's branch (`--base`) and served on hrx-loom 0.2.0 (variants name
their `"sdk"`; `devlab_perf.sh` gives each SDK its own root and runs `serve.sh` with that root's VENV; the one-server `ab` step refuses mixed SDKs),
plus the card job's func / ktest / gates and llama.cpp baselines. `collect` writes `perf` = % delta v0.2 vs v0.1 for decode, prefill (8k), longctx
(decode_32k) and 4stream (positive = v0.2 faster) per model and device; `status --status-json` writes it as `models.<slug>.perf.<gpu>`. A metric more
than 3 % slower queues ONE confirming `-r2` job (`rerun`); slower than 3 % in both runs = `needs_decision` with both numbers. `ingest` builds the v0.2
card on the staging head and puts the delta line into the publish message. Order: `plan -> submit -> poll -> collect -> package -> ingest -> publish`.

**Loom-only sweep (`--mode sdk-loom`)**: when the old card's llama.cpp rows and Loom rows are already measured and neither changed, re-measuring them is waste. Per model
and device ONE job serves the package built in the job on the new SDK alone (variants `staging` + `stagingnd`, `ab_jobs.py --sdk-ab --loom-only`): Loom perf short and
long at the card job's session count, c4, func, ktest, gates; no llama.cpp steps, no old-card variant. `collect` takes decode, prefill (8k), longctx (decode_32k) and
4stream as the % delta of the job's Loom rows against the Loom rows of the OLD card's `info.json` for that device (the card at the old revision; another job on the
same runner type). A gate failure holds. A metric more than 3 % off either way (or one the old card has and the job did not measure) makes the job's decision
`confirm`; `rerun` then queues ONE targeted ABBA job for that model and device: the `sdk-ab` job without baselines (old card on its own pin against the new package, same
job), and its verdict decides: slower than 3 % there = `needs_decision` with the numbers, else the first job promotes. State per job: `perf` (the cross-job deltas, with the ABBA
ones on the confirmed rows), `perf_cross`, `confirmed` = {row: ABBA %}; `status --status-json` writes `models.<slug>.perf.<gpu>` and `models.<slug>.perf_confirmed.<gpu>`
(the confirmed rows). `package` is unchanged; `ingest` builds the device's table from the first job's Loom rows (`make_info.py perf --baseline-from-info`) and the previous
card's llama.cpp rows (ROCm: copied per row with its original run and date as `baseline_from`; Vulkan: the published session stays), never re-measured, and the card's
methodology says "llama.cpp measured on D (run), Loom on D2 (run): same runner type, different jobs" instead of "same job" (`render_card.py`; a same-job card keeps its wording).
Order: `plan -> submit -> poll -> collect -> (rerun -> submit -> poll -> collect) -> package -> ingest -> publish`.

**Frozen SDK migration (`--programs production`)**: the release ships the production programs unchanged on the new SDK, so the sweep measures exactly those.

    python fleet/sdk_inventory.py snapshot -o /tmp/inv.json && python fleet/sdk_inventory.py drift skills/loom-migrate/versions/vOLD-to-vNEW-production.json /tmp/inv.json
    regen.py plan --base BRANCH --device strix-halo --device radeon-r9700 --mode sdk-loom --programs production --all --state S.json
    regen.py submit S.json --max-inflight auto; (loop >= 120 s) regen.py poll S.json; regen.py collect S.json; regen.py rerun S.json && regen.py submit S.json
    regen.py evidence S.json -o evidence.json; regen.py package S.json; regen.py ingest S.json; regen.py publish S.json; regen.py eligible S.json --out-dir release/

`--new-sdk` defaults to the newest verified release (`sdk_pin.VERIFIED_RELEASES`); its inventory `skills/loom-migrate/versions/vOLD-to-vNEW-production.json`
pins every production package (`_drift` records what moved since it was first accepted), the sibling `-held.json` lists packages out of the release. Each
job serves the production package at its pinned revision on the new SDK (nothing built; the programs hash is checked before anything is measured), the
devices come from the card itself (runner-less ones are recorded as carried), image packages get an image-service job (identical pixels on both SDKs),
and every perf session leaves the server's raw `/metrics` (`metrics_V_NNN.txt`: the speculative counters). `evidence` writes the per model / device /
metric record (`sdk-evidence/1`: old card row and session, new row, run, job, conclusion, programs hash, wheel, gates, admission probe, acceptance);
`eligible` derives eligible.json / held.json / pending.json from it, fail closed. Phase 2 (the kit's programs against the frozen ones, both on the new
SDK, ABBA in one job): `regen.py plan --mode sdk-ab --old-sdk target --no-baselines ...`.

`regen.py package S.json [--dry-run] [--slugs S...] [--ref REF]` is the step between `collect` and `ingest` (`make_info.py` refuses programs other than the
ones the job built, so the v0.2 package must be on staging before the card is made). It builds the base once, CPU only (`build_fleet.py --ref origin/<base>
--with-gen`), snapshots each staging repo at its head (small files, never weights), and replaces `program/**`, the package-root `*.loom`, `gen/**`,
`llm_config.json` and `model.json` with the canonical build's; `tokenizer.json`, `model.json.lock` and the card files stay (the old card until `ingest`).
`model.json` is the canonical one with the published `origins` and parameter-file `origin`s carried over; it is checked against the published one first
(`pool_capacity`, origins and the per-GPU presets aside): a field the canonical build adds is listed, a removed or changed one stops that model. The
upload is `publish_package.py upload` (not `--card-only`) to loom-fun-staging with `--parent-commit` = the snapshot head, so a concurrent publish fails it.
State per job: `package = {revision, programs_sha256, base}`; a rerun skips a published model, and a head that already holds the canonical programs is
recorded without an upload. A job whose `programs_sha256_staging.txt` differs from the canonical build's stops that model (`needs_decision`); `ingest`
stops a model without a package, with a package whose hash is not the job's, or whose staging head no longer holds the package's programs. Default
selection: collected, promoted jobs; `--slugs` takes named models in any stage. `--dry-run` runs everything but the upload (publish_package.py's own dry run)
and changes no state.

A validated optimization may intentionally change a preset's prefill chunk. Name that one transition explicitly:
`regen.py package S.json --dry-run --slugs SLUG --ref MERGED_REF --prefill-transition SLUG:chat:1024:2048`.
Every selected job must have a matching program digest and `model_staging.json` whose complete preset execution equals
the canonical build. Assembly checks the exact published/canonical old/new values; context, rows, drafting, parameter
origins and all other model-field guards remain unchanged. Omit the flag for ordinary SDK migrations. Review the
complete assembled package and its card before uploading; do not substitute a programs-only directory for fit sizing.

Time model (measured on this campaign's 86 DevLab jobs; new GPU, 35 fitting models, 3 runners, `--mode card`): about 12.8 runner-h, 6.3 h wall
(3.8 h of it measuring), 1 engineer with about 2 h of active work. `--mode card` costs about 0.75x of an A/B job (0.6x for large models); a
hand-run A/B regeneration of the same fleet needs about 15 engineer-h. The tests (tests/test_regen.py) run on a downloaded job of this campaign.

## Impact CI cost and coverage

The `impact` required check always runs scope lint, the independent serving-setting check,
KB lint, and a sampled DevLab plan. It does not build programs for card rendering,
documentation, test, or other explicitly identified non-build surfaces. Selection comes
from the changed paths, never from the author's declared scope: a false scope must still
fail lint. An edit confined to existing fleet configs/tokenizers/header dumps builds all
owners of those inputs. Shared kit/KB changes (including reference examples), manifest
or build-tool changes, and unknown paths conservatively build every model.

`impact_plan.py` records selected models, unchanged inputs, and its reasons in the PR
report. For build work it distributes models across up to four shards, with base and
head revisions on independent runners (up to eight CPU jobs). Each shard still uses the
builder's process pool, capped at two processes per runner; GPU/device subjobs are unnecessary because this check
only generates both targets' programs on CPU. Exact caches include the immutable revision, target kit/KB
trees, HEAD's fleet input/tool tree, selected model list, and installed dependency versions.
Both revisions intentionally use HEAD's manifest/config/header inputs, matching
`build_fleet.py --ref` semantics. The planner also validates each revision's own
`fleet/manifest.json`: selected additions are accounted as absent from BASE and built
at HEAD; removals are reported from the catalog comparison.

The final `impact` job fails if planning or any build fails, any expected shard or model
is absent, its revision/model set is wrong, or any build reports a skip. Only a selected
model proven absent from the historical BASE manifest receives a truthful catalog-absence
marker; missing/corrupt catalogs and failures for cataloged BASE models remain fatal.
Non-input PRs still
fail for invalid/missing scope blocks and protected-file violations. No GPU correctness
or performance gate is removed by this CPU optimization. When introducing a new build
dependency outside the kit/KB/fleet trees, update the planner's explicit non-input contract
and cache inputs, with a test, rather than extending the skip list implicitly.
