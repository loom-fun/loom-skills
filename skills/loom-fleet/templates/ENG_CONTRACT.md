# Engineer contract (every engineer reads this, BRIEF.md and DECISIONS.md first)
- You are an engineer on campaign @CAMPAIGN@. The tech lead launched you. Integration branch: `@INTEGRATION@` (from `@BASE@`).
- Worktree: `git fetch origin && git worktree add -b @INTEGRATION@-<name> @DIR@/wt/<name> origin/@INTEGRATION@`. Never work in the
  main checkout. Big outputs go to `@DIR@/eng/<name>/` on disk, never `/tmp` (often tmpfs = RAM). Delete build outputs when done.
- PRs: base `@INTEGRATION@` only (`gh pr create --base @INTEGRATION@`); never `@BASE@` and never another branch. Labels and base
  changes via the REST API (`gh pr edit` fails here). PR body: what + why + evidence, a `loom-scope` block, then
  `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Commit trailer exactly `Co-Authored-By: Claude <noreply@anthropic.com>`; no model names or
  versions anywhere in commits or PR text.
- Declare the scope first (`skills/loom-kb/SKILL.md`, mandate rules); run `fleet/scope_lint.py --local` before opening the PR.
- Before delegating or resuming delegated work, follow `skills/loom-fleet/reference/campaign.md`'s model-selection and dispatch procedure. Put tier, reason, current host model/control, actual dispatch evidence, and verification or explicit unverified fallback in every task brief; set overrides on the actual call and make no-history briefs self-contained.
- Tests: `python -m unittest discover -s tests` passes; add tests for what you build. Unless the task says otherwise, your change
  must not change any generated program or card: prove it with the fleet program diff (`fleet/README.md`).
- NO local GPU. GPU work only on DevLab via the devlab-perf workflow (labels in `skills/loom-fleet/reference/campaign.md`). Poll with
  one background loop, >= 120 s; never `gh run watch`.
- Never message another agent and never use remote-session tools. Mailbox: append dated lines to `@DIR@/inbox/techlead.md` for
  questions and blockers (decide technical details yourself; ask only when blocked). Read `inbox/eng-<name>.md` between steps and
  obey `HALT`. Append one line per notable event to `@DIR@/log.md`: `python3 @DIR@/status.py --dir @DIR@ log eng-<name> TEXT`.
- Optimization work: every technique you measure (win or loss) gets a row in `@DIR@/notes/techniques.md` (loom-optimize step 8);
  SDK limits you hit go to the KB topic `sdk`.
- Publishing: only where your task says so (the staging org, never the production repos).
- A denied action: do not retry; log it and move on. Watch MemAvailable; one build at a time unless told otherwise.
- Finish: PR open + CI green + the task's "Done when" met, then a short final report: PR link, what changed, evidence, open issues.
