# Trial prompt (fill the {PLACEHOLDERS}; send verbatim to a fresh agent session)

You are a fresh {AGENT} agent testing a set of skills exactly as a user would experience them.
Follow the skills literally; do not use knowledge from outside them unless they fail you (and if so,
note it).

THE USER'S REQUEST (treat this as the prompt you received):
"get {HF_REPO} working on loom-serve, optimize it, and report the performance relative to llamacpp rocm"

SKILLS: they are in {SNAPSHOT}/skills/. Start by reading {SNAPSHOT}/skills/loom/SKILL.md and follow
it. Set LOOM_SKILL_DIR to the directory containing the skill being read
(e.g. {SNAPSHOT}/skills/loom-port-llm), rebinding when switching skills. Do NOT use installed skills; use
these files. Do not read anything else under {HOME} (in particular not the skills repo checkout, other
trial directories, or earlier work dirs: earlier experiments live there, including earlier ports of
this exact model, and using them would invalidate this test) other than your own dirs, the HF cache
and the llama.cpp builds. The skills' own reference examples are fair game. Give any subagent you
spawn the same restriction.

ENVIRONMENT CONSTRAINTS:
- Install root: LOOM_ROOT={TRIAL}/root . Create it fresh with the loom-setup install script. Do not use ~/loom.
- Work dir: {TRIAL}/work . Put LOOM_SERVE_STATE and LLAMACPP_STATE inside it. Write packages, test
  dumps and any file over ~100 MB there, not to /tmp or your scratchpad (tmpfs: it uses RAM the GPU
  shares). Delete kernel-test `.npy` dumps once their tests pass.
- Ports: Loom {LOOM_PORT} (and the next 9 for variants), llama.cpp {LLAMA_PORT} (and the next 9).
  Check they are free. Other trials may be running on other ports: leave their processes alone.
- llama.cpp: the skills download the latest lemonade-sdk/llamacpp-rocm release themselves (pre-approved);
  installed builds under {HOME}/.cache/lemonade are approved too. No other binaries.
- Hugging Face downloads are fine. Respect the skills' GPU memory hard rules.
- Do not ask the user anything; run the whole job end to end in this one turn.
- Instead of publishing an Artifact, leave the final report in the work dir and copy it to
  {TRIAL}/report.md (and report.html).
- When you finish, leave the final optimized loom-serve running on port {LOOM_PORT}, stop every
  llama-server you started, and do not leave background wait loops running.

INSTRUMENTATION (required, for profiling the skill): append a line `$(date +%s) <phase-name>` to
{TRIAL}/phases.log at the START of every phase and once at the very end (`done`). Phases: install,
port, qualify, optimize, finalize, report, done (add sub-phases such as port_kernels, port_chat,
port_debug for manual work). Also note in that file (as `# ...` comment lines) every time you hit a
problem, a skill instruction was wrong/missing, or you had to improvise, with an estimate of the
minutes it cost.

FINAL ANSWER: reply with (1) the phase timestamps/durations, (2) final decode TG and prefill PP numbers
for Loom and llama.cpp, (3) gate verdicts, (4) a candid list of where time went and which skill steps
felt slow, redundant, or confusing, (5) the final package path, model id, memory_bytes and port.
