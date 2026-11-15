# Measure 2: replaying past PRs through the new architecture

Question: do past PRs come out narrowly scoped, or do they touch every model, and do the lint rules catch the ones that should have been narrow?
Method: each PR's intent is replayed twice on origin/refactor, CPU only (no GPU, no DevLab).

- **A, as originally done**: the way the PR was merged: a changed default (the kit's default and the registry's `legacy` value, which must agree) or
  the PR's own file edits, under the `loom-scope` block its author would plausibly have declared (targeted at the models they measured).
- **B, as the architecture wants it**: the same intent as an opt-in `scope.json` rule at the measured scope (or a device rule), same measurements.
- Per variant: `build_fleet.py` (both targets, 51 models) of base and head, `diff_fleet.py`, `scope_lint.py` with the declared block and that
  impact, `sample_plan.py` (default strata and `--granularity shape`). A **base** is the pre-PR state: the shipped feature turned off by its
  own kit default and registry legacy value (`base.patch`); it is the same tree for A and B. Evidence per replay: `fleet/replays/<name>/`
  (`meta.json`, `base.patch`, `A.patch`, `B.patch`, `*.scope`, `*.impact.md`, `*.lint.txt`, `*.plan.txt`, `*.plan.shape.txt`, `*.summary.json`).
  Tools: `replay.py NAME` (runs one replay), `mkpatches.py` (regenerates the patches), `summarize.py` (the tables below).
  Merged card / tooling PRs (#692, #694, #686) are replayed from the commit: base = origin/refactor with the PR's files as they were before the
  commit, head = after it.

## Summary

| PR | original: touched / leak / lint | scoped: touched / lint | sampler representatives | narrow? |
|---|---|---|---|---|
| #357 | 17 / 11 / FAIL (protected, registry, leak); A2: FAIL (protected, registry) | 6 / pass | A 15 models (28 strata), 10 by shape; B 5 models (8 strata), 4 by shape | A: no, B: yes |
| #451 | 15 / 0 / FAIL (protected, registry); A2: FAIL (protected, registry, leak) | 8 / pass | A 10 models (20 strata), 9 by shape; B 5 models (10 strata), 5 by shape | A: no, B: yes |
| D7 | 10 / 5 / FAIL (protected, registry, leak) | 5 / pass | A 6 models (12 strata), 6 by shape; B 3 models (6 strata), 3 by shape | A: no, B: yes |
| #573 / #574 | 50 / 50 / FAIL (protected, registry) | 50 / FAIL (leak) | A 46 models (88 strata), 40 by shape; B 46 models (88 strata), 40 by shape | A: no, B: no |
| #656 / #665 | 1 / 0 / FAIL (files) | 1 / pass | A 1 models (2 strata), 1 by shape; B 1 models (2 strata), 1 by shape | A: no, B: yes |
| #480 | 50 / 40 / FAIL (files, registry, leak); A2: FAIL (registry) | 7 / pass | A 41 models (76 strata), 33 by shape; B 5 models (10 strata), 5 by shape | A: no, B: yes |
| #692 | 0 / 0 / pass | 0 / 1 PRs: tooling pass | none (no program touched) | yes: nothing to sample |
| #694 | 0 / 0 / FAIL (files, protected) | 0 / 4 PRs: architecture pass, card pass, docs pass, tooling pass | none (no program touched) | yes: nothing to sample |
| #686 | 0 / 0 / FAIL (files, protected) | 0 / 5 PRs: architecture pass, card pass, docs pass, tooling pass, unclassified FAIL | none (no program touched) | yes: nothing to sample |

| PR | files edited, original (protected) | files edited, scoped (protected) |
|---|---|---|
| #357 | 2 (1) | 2 (0) |
| #451 | 2 (1) | 5 (0) |
| D7 | 2 (1) | 1 (0) |
| #573 / #574 | 7 (3) | 1 (0) |
| #656 / #665 | 1 (0) | 1 (0) |
| #480 | 3 (0) | 3 (0) |
| #692 | 3 (0) | 3 in 1 PRs by kind (0 in the architecture PR) |
| #694 | 28 (2) | 28 in 4 PRs by kind (2 in the architecture PR) |
| #686 | 55 (3) | 55 in 5 PRs by kind (3 in the architecture PR) |


Reading the table. touched = models whose programs or model.json differ between base and head (of 51). leak = touched models the declared scope
does not match. lint = `scope_lint.py` on the declared block (rules that failed). A2 = the same default flip declared `horizontal`. sampler =
models the sampler would run on DevLab (both GPUs together, strata in parentheses; "by shape" = `--granularity shape`). The files table counts
the files each variant edits (in brackets: protected core files: `fleet/ownership.json`); the code behind a switch is the same for A and B and is
not counted.

## Per PR

**#357 K/V-once decode attention (non-drafted head_dim 128).** A (default on): 17 models change, 11 outside the Qwen3 / Qwen3-VL cards the PR
measured (Qwen2.5 x3 and Qwen2.5-VL, LFM3 x2, Llama-family x3, Mellum, Qwen3-30B-A3B). Lint fails: `attn_rules.py` is protected, the `legacy` value of
`decode.attn.kv_once` changed, 11 leaks; declared `horizontal` it still fails (protected, registry): a default flip can only be an
`architecture` PR. B (rule `decode.attn.kv_once` on `arch/llm/dense/qwen3` and `qwen3vl`, `not_facets drafted`, `head_dim 128`): 6 models (the
Qwen3 0.6B / 1.7B / 4B / 14B and Qwen3-VL 4B / 8B cards), lint passes, no protected file. The siblings the PR only "expects to benefit" (kv4
Qwen2.5, Llama, ...) come back as one rule each after their own A/B. Sampler: B runs 5 models (Qwen3-VL-8B, Qwen3-1.7B + 0.6B, Qwen3-4B,
Qwen3-14B) on each GPU; A 15 of its 17.

**#451 K/V-once in drafted packages (head_dim 128, and 256 for qwen35 hybrids).** A (`ATTN_GQA_DRAFT`, `ATTN_GQA_D256` on): 15 models, lint fails
(protected, registry). The sampler's strata for A include drafted qwen35moe head_dim 256 / kv <= 2 (representatives Ornith-1.5-35B-A3B on both GPUs,
Ornith-1.5-122B on Strix, Sage-A1.5 (MTP) on both): the D7 regression would have been sampled at PR time, which is the answer to "would it pick a
drafted qwen35moe hd256/kv2 model": yes, 3 representatives in 3 strata. B (rules on the families the PR measured: qwen2vl, llama, flint, qwen3 and
hybrid/qwen35 with `+drafted`; d256 only on `qwen35`): 8 models, none of them qwen35moe, so the D7 regression cannot occur and needs no later rule.
Lint passes. The PR's Ornith-1.5-9B "noise" row is in B because it is a qwen35 model; the rule's evidence line names the measured cards.

**D7 (drafted qwen35moe decode_32k).** Base = refactor without the rule (the state of #451 A). A (`ATTN_GQA_D256=0` as the default): 10 models
change, 5 outside qwen35moe+drafted: it removes #451's gain from the dense qwen35 hybrids (Qwen3.5-4B / 9B, Ornith-1.5-9B, Qwen3.6-27B,
Qwen3.8-27B) as well; lint fails (protected, registry, 5 leaks). B (the merged rule on `arch/llm/moe/qwen35moe` + drafted): 5 models, lint passes,
1 file. Sampler B: 3 models; A: 6.

**#573 / #574 RDNA4 GEMM and flash twins.** One replay: the kit cannot switch the two apart (`LOOM_RDNA4`). Base built with the twins off
(`--rdna4 0`, gfx11 programs only), head with them on. All 50 models that have a WMMA kernel change; gfx12 gets the twins (`K_rdna4`) and
**the gfx11 kernel bodies are byte-identical**: the gfx11 kernel header only gains `target(@rdna3)`, plus template and `device` declarations. A
edits 7 files (the default at six kit sites and the registry legacy; 3 protected) and fails lint (protected, registry). B is one rule of the
`device/rdna4` node and **also fails** lint (50 "changes gfx11 programs" leaks): finding 1 below. The normalized one-family view
(`replay_impact.py`, twins and tags normalized away): gfx12 50 models, gfx11 1 model (E2B: its dual launch group changes the command programs;
no gfx11 kernel). The sampler runs 46 models per GPU for A and B alike: a new GPU target is fleet-wide by nature, the narrow part is the
GPU, not the model set.

**#656 / #665 E2B R9700 prefill stack, opt-in.** The PR was already narrow: `rdna4_f16_ffn` in the model's llm_config. A: 1 model, but the
flag lives in the package config (`fleet/<slug>/llm_config.json` here), which no targeted PR may edit (lint: files); in the repo the flag is
produced by a port run, so the PR looked narrow only because its author hand-picked one model. B (rule on `arch/llm/dense/gemma4`:
`heads 8, kv_heads 1, devices rdna4` sets `prefill.gemm.f16o_swiglu`, from which the trims and the d512 flash derive): 1 model, lint passes,
and the E4B / 12B / 26B / 31B siblings are excluded by the match, not by luck.

**#480 flash2 masked path only in the last key tile.** A (`FLASH2_PEEL` on): 50 models change (every model with a flash2 prefill kernel),
40 outside the six blocked cards' families; lint fails (files: the two flash generators, registry, leaks); declared horizontal it still fails
(registry). B (rules on `gemma4` with `sliding_window <= 512`, `spark2_5`, `lfm3`): 7 models (E2B, E4B, Bolt-E4B, Spark-X2.5 / X2.6,
LFM3-1.7B, LFM3-Omni-2B), lint passes. Sampler: 41 of 50 for A (33 by shape), 5 for B.

**#692 / #694 / #686 card and methodology PRs.** Program impact empty for all three (0 models, no GPU). #692 (three A/B scripts under
`.github/scripts/`) passes as `kind: tooling`. #694 (28 files) and #686 (55) fail as one `tooling` PR (files outside the kind, protected
`make_info.py`, `info_schema.json`, `.github/workflows/nightly-gates.yml`). Split by what each file is, every part passes under its own kind:
#694 = architecture (2: card schema + producer) + card + docs + tooling; #686 = architecture (3) + card (24) + docs (10) + tooling (16);
2 files of #686 have no kind (finding 2).

## Findings for the guards

1. **A device row that adds a twin target is read as a gfx11 change.** `diff_fleet.py` classes every changed kernel block by the `target`
   tag; adding the dual-target machinery rewrites each gfx11 kernel header (`target(@rdna3)`) and adds templates, so `scope_lint.py` fails
   `device/rdna4` PRs with "changes gfx11 programs" (50 of 50, #573 B) although no gfx11 kernel body changed. Recommendation: classify blocks on the
   text with the target tag and the template/decl blocks normalized (what `replay_impact.py` does). Until then the first dual-target PR of a family
   needs `leak:` under horizontal / architecture; later device rows without twins are unaffected.
2. **`ownership.json` has no home for two tooling paths**: `.github/devlab/**` (experiment configs) and `skills/loom-roofline/*.py`
   (#686 touched both). Add them to the `tooling` allow list (architecture PR).
3. **Defaults live twice.** Every switch has its default in the kit (`attn_rules.py`, `gen_flash2*.py`, ...) and its `legacy` in
   `choices.py`; the replay edits 2 sites per flag (#357, #451, D7, #480: 2 to 3) and 7 for `LOOM_RDNA4`. The scoped variants never edit a default.
   Reading the kit defaults from the registry would make a default flip one edit; not done here (not in scope).
4. **Sampling cannot shrink a fleet-wide change.** With the default strata a default flip samples nearly every touched model (#480: 41 of 50,
   #357: 15 of 17; by shape 33 and 10). Scoping is what makes the sample small: the same intent as B samples 5 models. The sampler works
   because strata are fine; the saving comes from the scope, not from the stratification.
5. **Scoped variants are narrow by construction and the lint agrees**: of the 6 program replays B touched exactly the declared nodes' models
   and passed lint in 5 (the 6th is finding 1); A leaked outside its author's measured scope in 4 (#357 11, #480 40, D7 5, #573 50) and was
   lint-clean in none (a default flip needs `architecture`: protected `attn_rules.py` / `dual_target.py`, a changed registry `legacy`, or files
   outside the kind).

## How the replays were approximated

- The code behind each switch (gqa body, drafted paths, twin generators, peel) is already in the kit at HEAD and is the same for A and B; the
  base turns the feature off by its default only. Only the policy (default vs scope rule) is compared, not the original diffs of the generators.
- #451 / D7 are chained on #357's A head (K/V-once on globally); `d256` is part of #451 as merged (the rule D7 later restricted).
- #357's B covers only the families the PR names (Qwen3, Qwen3-VL); #480's B the six blocked cards' families.
- #573 and #574 are one replay; the build flag `--rdna4` forces `LOOM_RDNA4`, so B's rule does not change the build, the programs are those of
  `--rdna4 1`; `A.patch` / `B.patch` are the policy expressions the lint reads.
- #480: only `FLASH2_PEEL` is switchable; its dead-block skip, edge handling and lane-0 entry stay in the kit for both variants.
- #350 (GDN recurrence) is not replayed: the decode / prefill split interlocks with the MTP single-form rule; #480 stands for the family-specific case.
- Card PRs: the PR's own file states on top of origin/refactor; the kb structure lint is skipped there (old SKILL.md / methodology.md copies do
  not exist in the new layout). The impact there is built as for the others (0 models).
- Sampler output is the plan only (which models); verdicts need DevLab and belong to measure 5.
