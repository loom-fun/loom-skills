# Cross-agent acceptance

One source suite lives in `skills/`. The installer adapts discovery locations only:
Claude Code uses `.claude/skills`, Codex and current Pi use `.agents/skills`.
Claude's plugin remains another distribution of that same source.

Each skill binds `LOOM_SKILL_DIR` from its own loaded file path. Claude Code's
`${CLAUDE_SKILL_DIR}` substitution supplies the exact path in the skill's opening
instructions. Other agents use the skill path provided by their loader. There is
no requirement that either variable already exist in the process environment.
The agent sets the shell variable before executing the examples and repeats that
binding in new shells or when switching skills. All skill-root command paths are quoted.

Native references:
- [Claude skills and substitutions](https://code.claude.com/docs/en/skills)
- [Codex skills and discovery](https://learn.chatgpt.com/docs/build-skills)
- [Pi skills and invocation](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/skills.md)

## CPU checks

Install test dependencies with `python3 -m pip install -r tests/requirements.txt`,
then run `python3 -m unittest discover -s tests -v`. CI runs this on Linux and Windows.
The checks exercise the actual complete suite in temporary user roots containing
spaces, verify bundled and sibling resources, and execute a bundled command from
an unrelated working directory. They also cover updates, removal, collisions,
local modifications, symlinks where available, and failed-replacement recovery.
These are installation checks, not runs of the agent models.

## Fresh-session smoke test

For each agent, make a separate scratch project whose path contains spaces.
Install the complete suite with `python3 scripts/install_skills.py --dest PATH`,
where PATH is `PROJECT/.claude/skills` for Claude or `PROJECT/.agents/skills` for
Codex and Pi. Launch a fresh session in PROJECT, not the skills checkout. Also
test Claude separately with `claude --plugin-dir /absolute/path/to/loom-skills`.
Use the same task, changing only the invocation prefix:

| Agent/distribution | Prefix |
|---|---|
| Claude copy | `/loom-improve` |
| Claude plugin | `/loom:loom-improve` |
| Codex | `$loom-improve` |
| Pi | `/skill:loom-improve` |

> Perform a CPU-only portability smoke test. Do not modify the skills, install the
> SDK, download models or start GPU servers. Follow the loaded skill's bundled-path
> instructions to run its scripts/kpi.py --help from a working directory outside
> the installed skill tree. Then locate the sibling loom-port-llm skill and run
> its starter/kit_source.py --help after rebinding the skill directory. Use the
> system Python; these two commands use only the standard library. Report the
> loaded SKILL.md paths, exact commands, exit codes and whether both worked.
> If subagents are unavailable, perform the task serially.

Pass requires actual command output, both exit codes zero, no unresolved path
tokens, and no missing sibling resources. Record agent version, model, source
commit, OS, installation method and transcript. A denied tool or failed login is
an environment block, not a pass. Test a direct sub-skill invocation: do not load
the umbrella first to supply missing instructions.

## End-to-end Loom trial

On supported AMD hardware, run one frozen revision through Claude, Codex and Pi,
using the same model, GGUF, SDK, reference build and resource budget. Fill
`skills/loom-improve/trial_prompt.md` separately for each with `{AGENT}`, isolated
work directories and ports. Run one trial at a time to avoid GPU contention.
Include a Pi/serial run without subagents. Preserve all gates and report phases,
friction, wall clock, final full gates/audit and quiet-GPU performance.

Record outcomes in `skills/loom-improve/reference/results.md`, identifying the
agent and model. Do not label support validated until the smoke test and complete
trial pass for that agent. Historical Claude trial results are not evidence for
Codex or Pi.

## Verification recorded November 9, 2026

Initial smoke runs used Windows, Python 3.15 and Git Bash with the portability
working tree based on `f18ddc48a5a1d350184c28234c484c524083d0e2`. The subsequent
frozen GPU-trial revision `451d63e` passes 27 CPU regression tests. All nine
skill metadata validations passed. Claude plugin validation passed with the
existing missing-version warning. Linux and Windows installation CI passed;
Linux is not claimed as locally tested.

| Agent | Distribution / model | Observed result |
|---|---|---|
| Claude Code 2.1.302 | Plugin and project copy; Sonnet 5.5 | Native invocation and both help commands passed |
| Codex CLI 0.167.0-alpha.3.2 | Project copy; CLI default with user config disabled | Both help commands passed, using absolute paths containing spaces |
| Pi 0.73.1 | Project copy; existing Lemonade `user.Qwen3.5-35B-A3B-ThinkingCoder` | Both help commands passed after recoverable tool mistakes |

Codex's first nested read-only run was blocked before shell startup; the user
approved the workspace-write / automatic-review retry, which passed. Its JSONL
output did not expose a model ID, so none is inferred here.

Pi's first trial used relative paths and changed into skill directories. The
instructions were tightened to require absolute commands, and its second trial
bound the correct absolute directories. That trial recovered from applying
`bash -n` to a Python file and a Windows file-tool path typo before completing
both commands. The file-tool/Git-Bash path distinction is now explicit. These
are successful smoke tasks with recorded friction, not evidence of flawless
agent behavior or full model-porting quality. The subsequent real GPU trials
are recorded in [GPU agent validation](gpu-agent-validation.md), including their
actual tool/history coverage and trial-isolation limitations.

Local transcripts are kept under `.agent-smoke/` (gitignored); no credentials or
raw session logs are committed.
