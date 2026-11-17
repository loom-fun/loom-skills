---
name: loom-migrate
description: Migrates the loom-skills kit, fleet of model cards and skills to a new hrx-loom SDK release - research the release, apply the version entry's mechanical steps (IR renames, program ABI, card schema, install/pin), prove the fleet parses and links under the new SDK, A/B the old cards against the new packages on DevLab, publish to staging, and record simplifications and optimization opportunities. Holds one entry per adjacent SDK pair (v0.1.0 to v0.2.0 first) and sweep and rewrite scripts. Use for "new Loom SDK", "migrate to hrx-loom X", "SDK upgrade", "port the fleet to a new SDK", "my package fails to parse or load on the new SDK", or "what changed between hrx-loom versions".
---

# Migrate to a new hrx-loom SDK

**Bundled paths: first shell action.** Before running commands, set `LOOM_SKILL_DIR` in the shell to the
absolute directory containing **this** `SKILL.md`, using its path from the skill listing or read tool.
In Claude Code, run `export LOOM_SKILL_DIR="${CLAUDE_SKILL_DIR}"` (the host substitutes the path).
In Codex, Pi or another host, export `LOOM_SKILL_DIR` to the loaded file's directory. Use a
shell-compatible absolute path (Git Bash on Windows). Rebind before using another skill's commands
and in every new shell; it is not the working directory or SDK root. Never run an unresolved placeholder.
Run bundled commands by quoted absolute path; do not run `scripts/...` from the working directory.
Install the complete Loom suite: sibling paths (`loom-fleet`, `loom-setup`, `loom-kb`) are required.

An SDK release changes what the kit emits (source IR, program ABI), what cards say (model.json schema, SDK pin, Quick start) and how things
install and run. The work is the same for every adjacent pair; only the list of changes differs. This skill holds the procedure (below)
and one **version entry** per pair in `versions/`. A user who owns a hand-written or catalog package uses the entry and the rewrite
script on their own tree; a campaign uses the whole procedure.

## Which job?

| Request | Do |
|---|---|
| Migrate one package or tree I own to a newer SDK | Read `versions/<from>-to-<to>.md`, section 3a: install the new wheel, `rewrite_<from>_<to>.py`, the manual items, `sweep.py`, serve and chat. (A package the repo's kit generated: regenerate it with the kit instead) |
| Migrate the fleet of cards | The whole procedure; run it as a campaign (`loom-fleet` `reference/campaign.md`) |
| The next release is out and there is no entry for the pair | Procedure step 1 (research), write the entry from `versions/TEMPLATE.md`, then continue |
| A package fails with `unknown op`, `unknown model field`, a publish/prepare signature error on the new SDK | Find the symptom in the entry's "Gotchas" and "Breaking changes" |

Entries: [versions/v0.1.0-to-v0.2.0.md](versions/v0.1.0-to-v0.2.0.md) (hrx-loom 0.1.0 to 0.2.0). Template for the next:
[versions/TEMPLATE.md](versions/TEMPLATE.md).

## The procedure (any adjacent pair)

Order matters: each step's proof is the next step's precondition. Big outputs go on disk (campaign dir `eng/`), never `/tmp`.

1. **Research first, before any migration code.** Write `notes/sdk<NNN>_research.md` in the campaign dir.
   - Release page and release notes (the migration table is the contract: "Previous / New / Migration" per surface), the upstream
     source diff between the tags (hrx-system, clone it), the library diff (hrx-loom-library: the catalog programs are the reference ABI),
     and the **wheel itself** (install it into a scratch venv; verify its sha256 against the release's `SHA256SUMS`).
   - **Empirical sweep first**: build the fleet's programs with the OLD kit (programs-only: `fleet/build_fleet.py`), then run
     `scripts/sweep.py --sdk NEWVENV` on it. The histogram of first-error classes is the real breaking-change list for source. It
     replaces guessing from the notes; the notes explain it and name what the sweep cannot see (load-time ABI, schema, behaviour).
   - Diff `--help` of every tool the repo calls (grep `skills/ .github/ fleet/ scripts/ tests/` for `loom-*`), old vs new.
   - Classify every change (below), map it to the files in loom-skills it touches (file:line), and mark each HARD (parse or load fails),
     SILENT (behaviour or numbers change), INSTALL, or N/A.
2. **Split by surface and write the version entry** (`versions/<from>-to-<to>.md`) while the engineers work: breaking changes with exact
   before and after, the files, the step (and script), the check, and the gotchas as they are found. It is the deliverable that outlives
   the campaign; update it from every engineer's gotcha lines before it is final.
3. **Mechanical steps.** Apply the entry's steps in order. Prefer a script over hand edits and keep it idempotent, so it also works on
   a user's tree and so a second run proves nothing was missed. Generators (the kit) are fixed at the emitter, not by rewriting output:
   run the rewrite script on a *copy* of the old fleet as the reference result, then make the emitters produce the same text.
   - Fleet tools that compare programs across the boundary (diff, equivalence, link) must accept both spellings until the old side is gone.
   - The kit emits only the new IR after migration (no dual-SDK emission): the old side of every A/B is an already published card.
4. **Fleet proof (CPU only).**
   - `fleet/build_fleet.py --tree . --out DIR` (programs only), then `scripts/sweep.py --sdk NEWVENV DIR --json OUT`: 0 parse failures, and
     every package links (a link failure is a load failure on a GPU). The old SDK on the old fleet is the control (same script, the old venv).
   - `fleet/diff_fleet.py OLDDIR NEWDIR`: unlike a normal change the program diff is NOT empty. It must be *explained*: summarize it
     by class and count (renames, ABI lines, pool/pin lines) and show that after mapping the renames back the remainder is only the
     deliberate ABI/schema lines. Anything unexplained is a bug in the migration.
   - Unit tests and `fleet/scope_lint.py --local` as for any PR (`loom-fleet` `reference/targeted-fix.md`).
5. **GPU proof on DevLab** (never the local GPU; `loom-fleet` `reference/campaign.md` "DevLab"). Smoke first: serve, one chat, one tool
   call, per representative (a dense small model, a hybrid/recurrent one, a vision one, and one other GPU family), with tight timeouts.
   Then the SDK A/B: `fleet/regen.py plan --mode sdk-ab`: per model one job in which the *published old card on the old SDK* and the
   *new package on the new SDK* run ABBA on the same runner, metrics decode / prefill / long-context / 4-stream, same gates. The
   ratchet: a regression over 3 % on a metric triggers one confirming rerun, then a decision with the numbers. Poll >= 120 s with one loop.
   When the old card and llama.cpp are already measured and unchanged, `--mode sdk-loom` is the cheaper sweep: Loom only on the new SDK, deltas
   against the old card's Loom rows, ONE targeted ABBA job only for a row outside +-3 % (`fleet/README.md` "SDK A/B"); the card then says
   Loom and llama.cpp come from different jobs on the same runner type.
6. **Publish to the staging org** (`loom-fleet` `reference/publish.md`): each card shows its own new-SDK measurement and the delta line
   against the old card; publish partial progress; verify the live card (session and programs hash). Never production.
   **Before promoting the staging cards to the production org** (owner's go-ahead), tag every production repo's current head with the
   OUTGOING SDK version (`scripts/tag_org.py --org loom-fun --tag v<old>`), so the old card stays addressable at
   `.../resolve/v<old>/model.json`; after the promotion tag the new state with the new version.
7. **Report the perf delta** per model and device: equal or better on a supermajority of metrics, and every regression over 3 % explained
   (an upstream issue, or here). A result without its explanation is not reported as "fine".
8. **Simplification ledger** (`notes/simplifications.md`): what the new SDK lets us delete (a flag, a computation, a workaround). Land what
   is inside the campaign scope; list the rest with the reason. Measure a simplification that depends on behaviour (for example
   dropping an explicit tuning flag) with an A/B before claiming it.
9. **Opportunity issues, not work** (`notes/opportunities.md`): when the release opens an optimization (a fixed overhead, a new
   primitive), open an issue in the skills repo with the evidence and the expected gain and do not pursue it in this campaign. File
   it only if a measurement or the source shows the gain; "might help" is not an issue.
10. **Close the loop**: fold every gotcha into the version entry, run the fresh-agent dry run (a new agent migrates one card using only
    this skill and the repo), and fix whatever it had to guess.

## Breaking-change classes (how to triage a release)

| Class | Symptom on the new SDK | How to find it | Typical fix |
|---|---|---|---|
| Source IR | `loom-opt`/`loom-link`: `error [PARSE/...] unknown op`, then undefined SSA values | `sweep.py` parse sweep (cheap, complete for syntax) | op rename/split: regex rewrite by operand type or attribute |
| Program ABI | server fails to load: export signature mismatch, missing export role, record validation | release notes "source ABI" rows, the fixture programs in the SDK tree and library at the new tag, `sweep.py` link (partly) | edit the templates/emitters; the fixture is the reference; a GPU smoke is the proof |
| Card schema | `unknown model field 'X'` at serve; readers raise `KeyError` | diff of the card parser/validator between tags; every card key the repo writes or reads (grep) | drop or rename the key in writers and tolerate absence in readers |
| Install and pin | wheel not where it was (PyPI vs release asset), catalog tag, pin schema | the release page, `pip download`, `loom.json` of the catalog | installer source, pin stamping, card Quick start, CI install steps |
| Device runtime | numbers move, kernel A/B not comparable, a device selector disappears | notes "runtime" rows, the loom-test/benchmark driver, completion/wait policy | re-baseline in the same job; keep explicit tuning flags until an A/B shows they are redundant |
| Request semantics | gates that send a field get a different answer (400 became ignored; a null means default) | the server's request decoder diff | adjust gates and integrations; note what silently changed |

A change can sit in two classes (a pool-size change is card schema *and* program bounds). Record both.

## Scripts (run them, do not read them)

| Script | Does |
|---|---|
| `"${LOOM_SKILL_DIR}/scripts/sweep.py" --sdk VENV FLEET_DIR [--json OUT] [--no-link]` | version-agnostic: `loom-opt` parse of every program file and `loom-link` link of every variant/service of every package (a fleet dir or one package), the first-error-class histogram, JSON out. Run it with the old venv as control and the new venv as the test |
| `"${LOOM_SKILL_DIR}/scripts/rewrite_v010_v020.py" [--check] [--json OUT] [--no-widen-pool] [--py] PATH...` | the v0.1.0 to v0.2.0 source migration on a file, package or tree: `vector.constant` to `splati`/`splatf` by element type, `command.program.*` to `command.*`, the `publish_epoch` control-state argument, the weights-policy import key range, `pool_capacity` out of cards, the KV pool bounds, the one-token conv guard, the vision finite-batch ABI; idempotent; prints `manual: FILE:LINE: ... section N` for what needs a hand edit |
| `"${LOOM_SKILL_DIR}/scripts/tag_org.py" --org ORG --tag vX.Y.Z [--message M] [--only REPO...] [--force] [--dry-run]` | tags the head of every `*-loom` repo of the org with the OUTGOING SDK version; skips a repo that has the tag; refuses a repo whose head is not on that SDK (its `llm_config.json` pin) unless `--force`; exit 1 if any repo failed, was refused, or has the tag on another commit; used before promotion |

A new pair gets its own `rewrite_<from>_<to>.py` beside these and a row here; keep scripts standard-library only.

## Rules that hold in every migration

- Pin the exact SDK version everywhere (cards, CI, installers); a stable release comes from PyPI (record the wheel sha256); a pre-release comes from the GitHub release asset, checked against the release's `SHA256SUMS`, because it is not on PyPI. When the stable release lands after a candidate, diff `candidate..stable` and re-run the offline proofs (sweep, link check) before re-pinning.
- Baselines for the perf delta are the *published old card on the old SDK*, same job, same runner, ABBA; never numbers from another day.
- Do not change a model's programs beyond the migration. A speedup found along the way is an opportunity issue.
- Old published cards keep working on the old SDK: a package with no pin means the old version, explicitly.
- Upstream bugs go to the upstream notes and the owner; do not file them in the SDK's repos yourself.
- Every gotcha is one line, `symptom -> cause -> fix`, in the version entry.
