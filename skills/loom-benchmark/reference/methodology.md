# Benchmark methodology: the one normative reference

Every card number, report and "Loom vs llama.cpp" claim follows these rules. Other skills link here and summarize at most
two or three lines; if they disagree with this file, this file wins and the other text is a bug. Code: `scripts/card/matched_spec.py`
(`TIE_BAND`, `outcome`, `tally`, `rule_violations`) and `data/drafting_classification.json`; tests: `scripts/card/test/check.sh`.

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
   wait, 3 rounds averaged (streams4: median of 3), timing source disclosed (Loom client-side, llama.cpp server `timings`), one
   pinned llama.cpp build per backend. Compare engines only within one runner/job (ratios, never across runners).
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

10. **Card text.** `render_card.py` generates a short Methodology section (5-8 bullets, same template on every card) from the card's
    own data: what is compared, the drafting choice and each engine's depth, 4 streams undrafted, what the rows mean (with the tie
    band), the server arguments, every "not supported" cell with its reason. Never hand-write it; no paragraph about configurations
    that "differ". The hrx-loom install is pinned to the version the card recorded.
11. **Publishing.** `make_info.py build PKG -o PKG/info.json` (always `-o`), `publish_package.py upload ... --parent-commit SHA`,
    then fetch the live `info.json` and confirm the new session id (`performance.sessions[-1].source`) and the gates'
    `programs_sha256` equal the package's before saying "published". Publish partial progress; never fake a cell.

Local qualification (`qualify.sh`, `llama_spec_tune.py`, `tune.py`) is a development loop: its tables guide the optimizer, are
not card numbers, and may try configurations the card rule excludes. Audit a card with `matched_spec.rule_violations`.
