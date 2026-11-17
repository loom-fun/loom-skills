# End-to-end SDK release checklist (every hrx-loom release)

The whole procedure from "a new SDK is out" to "everything is promoted", including the repos outside loom-skills. Copy it into the
release's tracking issue (one issue, maintained until promotion) and check items off. Learned in campaign sdk020 (SDK 0.1.0 -> 0.2.0,
2026-11-16/17, tracking issue loom-fun/loom-skills#1172). SKILL.md has the per-step detail for the loom-skills side.

## Contents
- 1. Organize
- 2. Migrate and measure (loom-skills, staging)
- 3. Per-GPU serve lines and stock hosts
- 4. Dependent repos and pages
- 5. Promotion, in order
- 6. Rules learned the hard way

## 1. Organize
- [ ] One integration branch (`sdk<NNN>`) is the single promotion branch; parallel projects (new GPUs, features) PR into it, not `main`.
- [ ] Nothing to `main` or the production HF org until the owner's single go; staging org only.
- [ ] Tracking issue with this checklist; one PM talks to the owner; projects report to the PM.
- [ ] Measure only on the stable release. A release candidate is pilot data: when the stable lands, retarget pins and re-measure.

## 2. Migrate and measure (loom-skills, staging)
- [ ] Research, version entry, mechanical steps, fleet proof (SKILL.md steps 1-4); diff RC -> stable for breaking changes too.
- [ ] Sweep = new package on the new SDK only (methodology rule 15): Loom rows + gates. Reuse the stored llama.cpp rows and the old card's
      Loom rows; a same-runner ABBA job only confirms a row outside its band.
- [ ] In-job builds equal the published package (same build flags incl. `--slug`, programs hash compared before measuring).
- [ ] Every slowdown root-caused before it is accepted: kernel GPU time vs wall time per dispatch, a catalog-model control, per-GPU split.
      Decide "SDK regression vs our kernels vs a changed default" with evidence. A changed default (e.g. a new pool backing mode) is
      a card setting, not an upstream bug.
- [ ] Gate failures triaged; a failure that already existed on the old SDK is not a regression and is out of scope.
- [ ] Final render of every card on staging (design, Quick start, device sections); live verification (programs hash, pin, sessions).

## 3. Per-GPU serve lines and stock hosts
- [ ] Each GPU's serve flags and presets are data (`fleet/devices/*.json`, card presets), rendered as one self-contained Quick start
      block per GPU. Re-check every GPU's default memory mode on the new SDK (elastic vs fixed: speed, eager allocations such as
      `cache_target_bytes` under fixed backing).
- [ ] Run every GPU's Quick start on a STOCK Ubuntu host (default 8 MiB `ulimit -l`, no root). DevLab runners have raised limits
      (Strix 15.6 GiB, R9700 3.9 GiB), so DevLab alone never proves this. 0.2.0 needs `--parameter_mode=preload` (ROCm/hrx-system#3332)
      where the host copy of the weights fits.
- [ ] Windows sample on DevLab Windows runners (~6 stratified cards via the card's own Quick start); all cards only if the sample fails.
- [ ] New GPUs: device results file validated (`make_info.py devices`), KB device node, skill pointer, card section.

## 4. Dependent repos and pages (one PR each, unmerged until promotion)
- [ ] loom-dev website (loom-fun/loom-dev): install/fetch/serve text, new GPUs browsable.
- [ ] HF org pages: staging org page (`spaces/<staging-org>/README`) updated now; the production page prepared for promotion.
- [ ] Gufo upstream PR (gufo-org/gufo, the Loom extension support PR): rebased, works with the new SDK; stays draft.
- [ ] loom-fun/loom-gufo-extension: SDK pin (Nix), API, docs, conformance against the staging cards.
- [ ] loom-fun/weaver: pins, runtime, doctor, templates, packaging, release notes.
- [ ] Lemonade (lemonade-sdk/lemonade): a draft PR into the Loom backend PR's branch updating it for the new SDK.
- [ ] Upstream SDK bugs: write them up with source refs; file only with the owner's OK.

## 5. Promotion, in order (owner's go)
1. [ ] Tag each production card repo's head `v<old SDK>` (`scripts/tag_org.py`; card tag rule: `loom-fleet` reference/publish.md "Card tags").
2. [ ] Merge `sdk<NNN>` -> `main`; tag the merge commit `sdk<new>-uN` (publish.md "Release tags").
3. [ ] Promote staging cards to the production org (`promote_staging.py`); verify every live card.
4. [ ] Publish the production HF org page.
5. [ ] Merge the website PR; check the device pages.
6. [ ] Merge the extension and weaver PRs; take the Lemonade PR out of draft; mark the Gufo PR ready if its maintainers agree.
7. [ ] Clean up: close experiment PRs, delete merged branches, archive campaign dirs, final report.

## 6. Rules learned the hard way
- Re-measure only what changed (rule 15); every DevLab job is a cost.
- Cancel unstarted jobs when the target changes; a plain cancel may not take, force-cancel and confirm the queue is empty.
- Report owner-facing status as open items only, times in US Eastern.
