# Publish, sweep, regenerate

How a package and its generated card reach the Hub, how the card's evidence is filled on DevLab, and when a regenerated
package may replace a published one. The sampled promotion check for targeted work (`fleet/promote_check.py`, the baseline
store) is [targeted-fix.md](targeted-fix.md) section 5; the ratchet's design is in [architecture.md](architecture.md);
`loom-benchmark/scripts/card/promote_staging.py` moves approved staging packages to their final repos on the owner's go-ahead; for an SDK
migration, first tag the production repos with the outgoing SDK version (`loom-migrate` SKILL.md step 6, `scripts/tag_org.py`).

**Release tags** (owner rule, 2026-11-16). loom-skills: every promotion to `main` (with its loom-fun cards) tags the merge commit
`sdkX.Y.Z-uN`: X.Y.Z is the hrx-loom SDK version the release targets; N counts the promotions on that SDK, starting at 0 and resetting
to 0 on a new SDK (`sdk0.1.0-u0`, `sdk0.1.0-u1`, `sdk0.2.0-u0`, ...). No `v` prefix and no dot before N: the tag is deliberately not semver
(a semver tool would read `-N` as a pre-release) and stays unambiguous if the SDK ever uses four version parts.

<a id="card-tags"></a>**Card tags** (loom-fun): a card tag `vX.Y.Z` means exactly one thing: the last commit of that card repo that worked
with SDK X.Y.Z, before the card moved to a newer SDK, i.e. the repo's head at the time of the move. It is created only at that move,
never on a new card. `scripts/tag_org.py` tags every `*-loom` repo whose head still pins the outgoing SDK, including cards the campaign did
not migrate (their head stays the last commit on that SDK). Which loom-skills commit built a card is in its `info.json`
(`versions.card_generator`), so cards carry no release tag.

Which models need this work is [populate-org.md](populate-org.md).

## 2. Publish the initial result

The worker publishes the package and its generated card (`loom-benchmark` "Hugging Face model card"; never
hand-written): `card/publish_package.py prepare PKG DIR --source REPO` (the GGUF repo), then `make_info.py build`
and `render_card.py DIR`, then `publish_package.py upload DIR --repo loom-fun/NAME` (`--card-only` for a card
update: README.md, info.json and REPORT.md only; `--parent-commit SHA` with the repo commit whose info.json the card
was built from, so a concurrent publish fails instead of being overwritten). Gates not yet run render as "no evidence"; that's expected and honest. The repo is now category A
and `org_status` shows it as `sweep` until step 3 fills it.

**Publish every port, done or not**, as soon as it has a package and gate evidence: failing gates, a failing
audit or perf flags don't hold it back. The card is the record of progress: a failing gate renders as fail, never
dropped or loosened, and `org_status` marks the repo `fix` (populate-org.md step 4). Add the card to
`.github/nightly-models.json` in the same PR, and republish as gates go green. Only `done` (below) needs every gate
passing.

**Card text rule: every fetch line names its preset** (owner rule, 2026-11-17). The renderer always prints
`loom-fetch huggingface:REPO:PRESET ...` (legacy pins: `--preset=PRESET`), single-preset cards included (`...:chat`), on every GPU block.
A preset added to a published card later must never break a published command: hrx-loom 0.1.0 and 0.2.0 both refuse a multi-preset
card fetched without a preset ("select explicitly"). A card or doc that prints a fetch line without the preset is a bug
(`tests/test_fetch_preset.py` fails on it).

**The nightly entry matches the card exactly**: `memory_bytes`, the llama.cpp build and `features` (why: the nightly
gates serve with the entry's values, so a mismatch gates a different configuration than the card claims).
`org_status` prints a `CHECK:` for a missing or differing entry. A republish that changes any of them updates the
entry in a PR. A catalog-package card (`kit: "qwen4exp"`, loom-kb topic `catalog-ports`) has a `memory_bytes` above the
60 % default and an SDK pin in its `llm_config.json`: the nightly installs that SDK (`ensure_sdk.sh`) and needs a runner
with the headroom (`memcheck.sh`); refresh the pin before its `expires` (loom-setup "SDK pins").

**Memory budget** (one number for the whole card; details in the loom-kb `org` topic (`kb.py route --topic org`)):
- On every regeneration pass `make_info.py build --memory-bytes N` explicitly, N being build.py's printed value for
  the regenerated programs. It is not picked up from build.py, and a carried-over old value mislabels the card (#151
  now refuses it when the programs changed).
- The gates, the devlab-perf job and the nightly entry all use that same N.
- A memory-capped package (budget below context plus pool headroom) keeps a measured 2 GiB margin (build.py, #130):
  with pool equal to the context and no margin, a full-context request is refused with a 503.

**Pausing a model:** list what would be done next, one step per line in a file, and republish the card only (DIR
as in step 3.2): `make_info.py build DIR --from-info DIR/info.json --status paused --next-steps FILE`,
`render_card.py DIR`, `publish_package.py upload DIR --repo loom-fun/REPO --card-only --parent-commit $sha`. The
README then ends with a "Next steps" section. `org_status` keeps the repo's action and reports `paused: true`. To
resume, rebuild with `--status active --next-steps /dev/null`. A paused card keeps its evidence; use it also for a
model dropped from the target list (reason first in the steps).

## 3. Sweep on DevLab Dispatch: missing gates, the roofline and the performance job

DevLab is an elastic pool of Strix Halo runners; sweep many cards' gates and performance at once there, never on
the local GPU. Hand the card over and move the local GPU on to the next item.

1. The card is in `.github/nightly-models.json` (card, memory, kit, llama.cpp build; step 2). Run `nightly-gates.yml`
   (it runs on schedule, or `gh workflow run nightly-gates.yml -f models=loom-fun/REPO,... -f level=full`, or label
   a PR `nightly-gates`).
   **Wait at least 2 minutes after an upload before dispatching**: runners can still fetch the previous revision of the
   programs. If a run measured old programs, re-dispatch it.
2. When it finishes, per card, in a directory holding the repo's current `model.json`, `model.json.lock`,
   `llm_config.json` and `info.json` (re-read `info.json` from the Hub right before, at a known commit:
   `sha=$(curl -s https://huggingface.co/api/models/loom-fun/REPO/revision/main | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])')`,
   then `curl -L -o DIR/info.json https://huggingface.co/loom-fun/REPO/resolve/$sha/info.json`):
   - `card/pull_devlab.py loom-fun/REPO --package DIR -o ev_devlab.json`: the gates, with the run, job URL and
     runner (skips jobs that measured an older revision of the package: it ingests only jobs whose `programs_sha256` equals the package's), and the spec gate's acceptance stamped with
     the job's programs: acceptance depends on the model, drafter and text, not the machine, so a sweep on the
     published programs replaces a stale `spec_acceptance` (speed stays local-only);
   - `card/make_info.py roofline --package DIR -o ev_roofline.json` (tools venv, CPU only: reads the GGUF), once
     per card or after a session changed its acceptance (a first card, with no info.json yet: add
     `--evidence ev_perf.json ev_gates.json`). A drafted row's ceiling counts the acceptance measured in the row's own
     session (perf.py, c4_steady.py), else the spec gate's, and the card says which;
   - `card/make_info.py build DIR --from-info DIR/info.json --evidence ev_devlab.json ev_roofline.json ...` (the
     repo comes from info.json, the generator commit from the checkout), `render_card.py DIR`, and
     `publish_package.py upload DIR --repo loom-fun/REPO --card-only --parent-commit $sha` (fails if another worker
     published since: re-read and rebuild).
3. A card with metrics missing or measured on other programs (`sweep`, `remeasure`): one `devlab-perf` job per card
   (required: it is the only source of card performance; llama.cpp steps are omitted when the card's stored llama.cpp rows are reusable, methodology rule 17) (`.github/devlab/perf.json`: card, memory_bytes, kit and llamacpp from `nightly-models.json`,
   llama_spec omitted = `matched`: llama.cpp gets the card's classified drafting choice (same method and drafter, its documented n-max;
   none if the classification says none; `../../loom-benchmark/reference/methodology.md`), an explicit `llama_spec` only overrides it (`{drafter}` = the
   card's drafter, `{hf:OWNER/REPO/FILE}` a Hub file, e.g. an MTP head the card doesn't ship); variant `{"name": "card"}`, steps `perf:card:1` x3 (6 for a fast model, decode > ~100 tok/s: also job field `"repeat_rows": 3`, perf.py `--repeat-rows`), the same count of `perf:card:1:long`, `c4:card`, `llama@rocm.np1:3 llama@rocm.np1:3:long` (6 for a fast model)
   `llamac4@rocm.nodraft` (every metric = the median over sessions of each session's median sample, methodology rule 5; `r9700_jobs.py` and `strix_ab_jobs.py` write exactly this) (files `perf_llamacpp-rocm.np1_nN.jsonl`, `c4_llamacpp-rocm_nN.txt`); several cards go in one experiment, one job each with its own card fields), then
   `make_info.py perf --loom-jsonl perf_card.jsonl --llama-jsonl perf_llamacpp_nN.jsonl --c4-loom c4_card.txt
   --c4-llama c4_llamacpp_nN.txt --package DIR --source-kind devlab --run ID --url JOB_URL --device JOB_DEVICE.json
   --hrx-loom JOB_HRX.json --baseline-version llamacpp-rocm/BUILD --baseline-args "$(cat llamacpp_args.txt)"
   --fetched-programs "$(cat programs_sha256.txt)"` on the job's artifact (write JOB_DEVICE.json and JOB_HRX.json with
   `card/pull_devlab.py --identity JOBDIR --device-out JOB_DEVICE.json --hrx-out JOB_HRX.json`: the runner and the SDK
   the job recorded; `none` if absent; `--source-kind devlab` is perf's default, `--source-kind local` is refused except for a power-mode card, below), and
   rebuild the card as above.
   **Vulkan baseline** (every card, one job each; the same workloads and `llama_spec` as the ROCm baseline): job fields
   `llamacpp_vulkan: "b12211"`, variant `{"name": "card"}`, steps `llama@vulkan.np1:2 llama@vulkan.np1:2:long llamac4@vulkan.perslot.nodraft` (the same drafting choice as ROCm on 1-stream rows)
   (stored ROCm and Loom rows are reused while their inputs are unchanged, rule 17); at the card's own `memory_bytes`. Ingest:
   `make_info.py perf --backend vulkan --llama-jsonl perf_llamacpp-vulkan_nN.jsonl --c4-llama c4_llamacpp-vulkan_nN.txt
   --baseline-version b12211 --baseline-args "$(cat llamacpp_args-vulkan.txt)" --baseline-sha256 "$(cat
   llamacpp_release-vulkan.sha256)" --fetched-programs "$(cat programs_sha256.txt)" --differs "..." --package DIR
   --source-kind devlab --run ID --url JOB_URL --device none --hrx-loom none` (refused if the job fetched other
   programs than DIR's), then `make_info.py build --from-info` as above. Its numbers are another job's: compare
   engines by ratios, not across runners. KV layout (`vulkan_kv`, default auto): per-slot caches, except unified (as ROCm) for DFlash servers, where b12211 aborts on per-slot caches; the unified pool makes b12211 Vulkan attend with a dense mask (no GPU fused flash attention over K/V rows), so record that as a `--differs` line too. Every server argument that differs from the ROCm baseline, or that the Vulkan
   server could not run, goes in as `--differs "setting: reason"` (shown on the card), never changed silently. Nothing in `org_status` reads it.
4. Missing metrics come from a devlab-perf job (item 3), never from a local session.

## Regeneration safety

- Before publishing, A/B it against the card's own published programs on the same DevLab runner: decode, decode_32k,
  prefill 1k to 32k, streams4. Bisect main when a main build loses.
- Check that the card's generator commit is on main or in a merged PR (squash merges leave PR commits off main's
  history; `org_status` flags the rest). A card built from an unmerged branch silently loses that branch's features
  when regenerated from main (decode_32k halved this way); merge the feature first or keep the card.
- A kernel path that wins at some KV-head counts can lose at others: gate paths by shape and A/B across the
  org's shapes, not one card.

## Performance: DevLab is the source

Every performance number on a card comes from a `devlab-perf` job (Loom with the card's programs and llama.cpp on the
same runner). Loom used to measure well below a local session on DevLab (prefill -25 to -52 %): the DevLab runners'
SMU firmware (BIOS 1.12, SMU 0x0a640600) lowers the memory-fabric clocks (FCLK/UCLK) while the CPU is idle, and Loom
parked its CPU while the GPU worked. Loom is now served with an active wait (`serve.sh` turns `--device=amdgpu` into
`amdgpu://?completion_spin_ns=4294967295`; `LOOM_ACTIVE_WAIT=0` opts out, for A/B only), and DevLab matches or beats
the old local numbers. Local runs are for development only (the optimize/port loop, kernel A/B, quick checks), and
a Loom number measured without the active wait is not comparable with a card's.

**Exception: a card measured at a non-default power mode** (methodology rule 18; owner rule 2026-11-20). The DevLab Strix Halo
runners run at 85 W; a card that states another power mode (the local GMKtec EVO-X2 in BIOS performance mode, ~123 W) comes from
ONE local job that measures every engine of that GPU's table at that power, interleaved:
`WORK=DIR LOCAL_HF_CACHE=1 POWER_LOG=1 POWER_BAND="120 125" MIN_MEMAVAIL_GB=40 flock GPU.lock .github/scripts/devlab_perf.sh EXP.json JOB`
with the card's normal steps (`perf:card:1` x3 short and long, `c4:nodraft`, `llama@rocm.np1:1` / `llama@vulkan.np1:1` rounds,
`llamac4@rocm.nodraft`, `llamac4@vulkan.perslot.nodraft`) in ABBA rounds, plus `gufoperf:1[:long]` / `gufoc4` when the card shows
Gufo (experiment field `gufo`: its pinned container image, ITS OWN model file and drafter; devlab_perf.sh header). Ingest each engine
with `make_info.py perf --source-kind local --power-mode 123w --power-evidence OUT/power_steps.json --run JOB --device DEV.json`
(DEV.json names the machine; Vulkan and Gufo with `--backend vulkan|gufo`, Gufo also `--baseline-name Gufo --baseline-model
OWNER/REPO/FILE --baseline-model-sha256 SHA --baseline-label "Gufo (UD-Q4_K_XL)"` and `--differs` lines for its drafting), the
roofline with `make_info.py roofline --power 123w`. Gates still come from DevLab (final preset), and the other GPUs' rows keep their
DevLab sources (labelled so on the card).
