# Checking agent compatibility

Run the CPU regression suite (Python 3.12+, no extra dependencies):

```bash
python3 -m unittest discover -s tests -v
```

CI runs it on Linux and Windows. It covers installation, user-file protection,
paths with spaces, Windows subprocesses, gate exclusions and streaming timing.

For a manual smoke test, run this in a scratch project with spaces in its path,
choosing the agent under test:

```bash
npx skills add loom-fun/loom-skills --skill '*'
```

Start a fresh agent session and invoke `loom-improve` directly:
`/loom-improve` in Claude, `$loom-improve` in Codex, `/skill:loom-improve` in Pi.
For Claude's plugin, use `/loom:loom-improve`.

Ask the agent to run `loom-improve/scripts/kpi.py --help`, then the sibling
`loom-port-llm/starter/kit_source.py --help`, using absolute paths from outside
both skill directories. Both commands must exit zero. This checks skill loading
and resource paths; it does not test GPU porting.

For GPU validation, use `skills/loom-improve/trial_prompt.md` in a separate work
directory per agent. Record outcomes in
[the trial ledger](../skills/loom-improve/reference/results.md).

Verified on Windows (2026-11-10) with `skills` 1.7.6: Claude Code 2.1.302,
Codex 0.167.0-alpha.3.2 and Pi 0.73.1 each loaded the npx-installed suite and
passed both commands. All nine skills and their bundled files were present.
