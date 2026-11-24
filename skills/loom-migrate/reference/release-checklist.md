# End-to-end SDK release checklist (every hrx-loom release)

The whole procedure from "a new SDK is out" to "everything is promoted", including the repos outside loom-skills. Copy it into the
release's tracking issue (one issue, maintained until promotion) and check items off. First written in campaign sdk020 (SDK 0.1.0 -> 0.2.0,
tracking issue loom-fun/loom-skills#1172). SKILL.md has the per-step detail for the loom-skills side; release-specific facts
(flags, issue numbers) belong in the version entry, not here.

## Contents
- 1. Organize
- 2. Migrate and measure (loom-skills, staging)
- 3. Per-GPU serve lines and stock hosts
- 4. Dependent repos and pages
- 5. Promotion, in order
- 6. Rules learned the hard way

## 1. Organize
- [ ] One integration branch (`sdk<NNN>`) is the single promotion branch; parallel projects (new GPUs, features) PR into it, not `main`.
- [ ] Nothing to `main` or the production HF org until the owner's single go (section 5); staging org only.
- [ ] Tracking issue with this checklist; one PM talks to the owner; projects report to the PM.
- [ ] Measure only on the stable release. A release candidate is pilot data: when the stable lands, retarget pins and re-measure.

## 2. Migrate and measure (loom-skills, staging)
- [ ] Research, version entry, mechanical steps, fleet proof (SKILL.md steps 1-4); diff RC -> stable for breaking changes too.
- [ ] SDK pin everywhere: `skills/loom-setup/scripts/install_loom.sh`, loom-setup docs, card Quick start pin, CI install steps and
      `.github/nightly-models.json`, starter kit pins (the version entry lists the files).
- [ ] Sweep = new package on the new SDK only (methodology rule 17): Loom rows + gates. Reuse the stored llama.cpp rows and the old card's
      Loom rows; a same-runner ABBA job only confirms a row outside its band.
- [ ] In-job builds equal the published package (same build flags incl. `--slug`, programs hash compared before measuring).
- [ ] Every slowdown root-caused before it is accepted: kernel GPU time vs wall time per dispatch, a catalog-model control, per-GPU split.
      Decide "SDK regression vs our kernels vs a changed default" with evidence. A changed default is a card setting, not an
      upstream bug.
- [ ] Gate failures triaged; a failure that already existed on the old SDK is not a regression and is out of scope.
- [ ] Final render of every card on staging (design, Quick start, device sections); live verification (programs hash, pin, sessions).
- [ ] Smoke the staging cards right before the go: SKILL.md step 5 smoke (serve, one chat, one tool call) on a sample per GPU.

## 3. Per-GPU serve lines and stock hosts
- [ ] Each GPU's serve flags and presets are data (`fleet/devices/*.json`, card presets), rendered as one self-contained Quick start
      block per GPU. Re-check every GPU's default memory mode on the new SDK (elastic vs fixed: speed, eager allocations such as
      `cache_target_bytes` under fixed backing).
- [ ] Every rendered fetch line names its preset (`loom-fetch huggingface:REPO:PRESET ...`), single-preset cards included: adding a preset
      later must never break a published command (0.1.0 and 0.2.0 refuse a multi-preset card fetched without one).
      `tests/test_fetch_preset.py` enforces it on all goldens.
- [ ] Run every GPU's Quick start on a STOCK Ubuntu host (default `ulimit -l`, no root). DevLab runners have raised memlock limits,
      so DevLab alone never proves this; any workaround flag goes on the serve line only where it fits (version entry has the details).
- [ ] Windows: a stratified sample of cards (`fleet/sample_plan.py` strata) on DevLab Windows runners via each card's own Quick
      start (owner decision in sdk020: the full set only if the sample fails).
- [ ] New GPUs: device results file in `fleet/devices/` validated with `skills/loom-benchmark/scripts/card/make_info.py devices`,
      KB device node, skill pointer, card section.

## 4. Dependent repos and pages (one PR each, unmerged until promotion)
- [ ] loom-dev website (loom-fun/loom-dev): install/fetch/serve text, new GPUs browsable.
- [ ] HF org pages: staging org page (`spaces/<staging-org>/README`) updated now; the production page prepared for promotion.
- [ ] Gufo upstream PR (gufo-org/gufo, the Loom extension support PR): rebased, works with the new SDK; stays draft.
- [ ] loom-fun/loom-gufo-extension: SDK pin (Nix), API, docs, conformance against the staging cards.
- [ ] loom-fun/weaver: pins, runtime, doctor, templates, packaging, release notes.
- [ ] Lemonade (lemonade-sdk/lemonade): a draft PR into the Loom backend PR's branch updating it for the new SDK.
- [ ] Upstream SDK bugs: write them up with source refs; file only with the owner's OK.

## 5. Promotion, in order (owner's go)
Before approval, freeze the eligible staging commits, production parents, outgoing tags and held repos. Prepare and verify the exact production package bytes on DevLab using [the reviewed bundle procedure](prepared-promotion.md).

1. [ ] Verify or create the outgoing SDK tags at the exact production parents in the reviewed release tag plan. The `tag_org.py --only REPO...` dry run can check explicitly named text repos, including suffixed names. Use the release's exact tag plan for image packages, whose SDK evidence has no `llm_config.json`. The promoted cards get tags when they in turn move to a newer SDK.
2. [ ] Merge `sdk<NNN>` -> `main`; tag the merge commit `sdk<new>-uN` (publish.md "Release tags").
3. [ ] Replay the reviewed bundle against the pinned production parents. Archive the returned commit IDs and exact readback receipts.
4. [ ] Publish the production HF org page.
5. [ ] Merge the website PR; check the device pages.
6. [ ] Merge the extension and weaver PRs; take the Lemonade PR out of draft; mark the Gufo PR ready if its maintainers agree.
7. [ ] Clean up: close experiment PRs, delete merged branches, archive campaign dirs, final report.

## 6. Rules learned the hard way
- Re-measure only what changed (rule 17); every DevLab job is a cost.
- Cancel unstarted jobs when the target changes; a plain cancel may not take, force-cancel and confirm the queue is empty.
