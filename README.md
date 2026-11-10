# Loom agent skills

Shared skills for Claude Code, Codex and Pi that let a fresh session use
[AMD Loom](https://github.com/ROCm/hrx-system) (the `hrx-loom` compiler, runtime and
`loom-serve` model server) from a cold start:

- install the SDK;
- fetch and serve catalog models (LLM, ASR, TTS, image, embeddings);
- run a browser playground;
- **port an arbitrary GGUF LLM to Loom**;
- optimize it without breaking correctness;
- benchmark it against llama.cpp ROCm.

The skills were developed and verified on an AMD Ryzen AI Max+ 395 "Strix Halo"
(gfx1151 GPU + XDNA NPU, 128 GB).

## Install

Install the complete suite with [the skills CLI](https://github.com/vercel-labs/skills) (requires Node.js):

```bash
npx skills add loom-fun/loom-skills --skill '*'
```

Choose Claude Code, Codex or Pi when prompted. This installs into the current
project; add `-g` for all projects. Install all nine skills together: they share
sibling scripts. Start a fresh agent session afterward.

| Agent | Project skill directory | Invocation |
|---|---|---|
| Claude Code | `.claude/skills` | `/loom` |
| Codex | `.agents/skills` | `$loom` |
| Pi | `.agents/skills` | `/skill:loom` |

Update with `npx skills update`; remove with `npx skills remove`.

**Without Node.js:** use the Python 3.12+ fallback (standard library only):

```bash
git clone https://github.com/loom-fun/loom-skills
cd loom-skills
python3 scripts/install_skills.py
# Windows PowerShell: python scripts/install_skills.py
```

The fallback installs for all three agents in their user skill directories.
Use `--agent codex` to target one agent, `--dest PATH` for a custom directory,
and `--uninstall` to remove unchanged copies. It preserves unmanaged and locally
modified skills.

**Claude Code plugin.** The existing plugin and automatic skill-directory substitution remain supported:

```text
/plugin marketplace add loom-fun/loom-skills
/plugin install loom@loom
```

Use either the plugin or a skills install in Claude to avoid duplicates.
Update the plugin with `claude plugin marketplace update loom && claude plugin update loom@loom`.
Private repositories require read access and working Git credentials for either installation method.

## Use

### Recommended: port, optimize and report in one prompt

```
get <hf-org>/<model> working on loom-serve, optimize it, and report the performance relative to llamacpp rocm
```

The agent runs the whole pipeline without stopping, within the model's roofline time budget (about
25 min for a 0.8B to 70 min for a 30B dense model in recent trials), as four scripted steps
with the agent's judgment only where it is needed:
1. `loom-port-llm/port.sh`: pick the GGUF (header-only probe: supported formats, fits in GPU
   memory), port, build at the native context, serve, kernel tests, smoke test;
2. `loom-benchmark/scripts/qualify.sh`: pick the fastest llama.cpp ROCm build, run the
   correctness gates against it, measure both baselines, run the **roofline analysis**
   (per-metric targets) and size the optimization budget;
3. optimize under a rate-of-approach controller, with fast gates after each kept change;
4. `loom-benchmark/scripts/finalize.sh`: the full gates (including a full-context probe)
   on the final build, final perf, the audit, and a generated report.

It stops to ask only for a gated-model license. Loom (PyPI) and the latest lemonade-sdk/llamacpp-rocm
release (the reference) are downloaded without asking.

**What a port includes.** The model's native context length (40,960 for Qwen3-0.6B,
262,144 for Qwen3.5-0.8B), tool calling in the model's own format, reasoning with an
on/off switch, history reuse between agent turns, and per-request sampling
(`temperature`, `top_p`, `top_k`, `seed`). The chat glue is derived from the model's own
Jinja chat template and verified against it. Dense transformers and Qwen3.5-style
gated delta-net hybrids are covered by the starter kit. The llama.cpp reference
outputs are cached, so re-running the gates on an optimized build only costs the Loom side.

**Recent fresh-agent trials (Claude Code, hrx-loom 0.1.0, Strix Halo, November 2026)**, one prompt each, full
gates and audit PASS, against the fastest llama.cpp ROCm build on the same GGUF: Qwen3.5-0.8B in 24 min
(decode +27 %, prefill 1k/2k +22/+28 %), Bolt-E4B (Gemma 4) in 52 min (prefill 1k/2k +44/+58 %),
flint (40B MoE) in 58 min (decode +25 %), Muse-Glimmer-30B in 71 min. Every trial and its lessons:
`skills/loom-improve/reference/results.md`.

**Effort control.** With no target and no time given, the **whole job** gets the model's
roofline budget (`loom-roofline/e2e_budget.py`): the optimization phase gets what is left after
the port and the qualification, minus a reserve for the final gates and the report. Add a time or a target
to the prompt to change it, for example "…optimize it for 3 hours…" or "…until decode is
50 % faster than llama.cpp…". The roofline step checks a stated target against the
hardware ceiling and says so up front if it is out of reach.

### Other examples

Ask in plain language, for example:
- "install Loom and serve moonshine";
- "port Qwen3-4B-Instruct-2507 to loom-serve and validate it against llama.cpp";
- "benchmark my Loom port";
- "optimize this model" (roofline → targets → about 30 min of optimization → report);
- "what's the roofline for this model?" or "is +50 % decode achievable?".

The skill descriptions help the agent pick the right one. You can also invoke a skill by
name:

| Skill | What it does |
|---|---|
| `loom` | Start here: overview, hard safety rules, which skill to use |
| `loom-setup` | Install the SDK (`hrx-loom` from PyPI) and the matching model library, `loom-fetch`, start/stop/inspect `loom-serve` (GPU and NPU) |
| `loom-demo-ui` | Web playground: chat, transcribe, speak, image, embed. It finds models from the servers' `/v1/status` |
| `loom-port-llm` | Config-driven starter kit that turns a GGUF LLM (dense, or a Qwen3.5-style gated delta-net hybrid) into a Loom package at its native context length, with tool calling, reasoning and history reuse derived from the model's own chat template: inspect → generate kernels → build → kernel unit tests → golden, agent and full-context gates |
| `loom-roofline` | Roofline for decode/prefill/c4 from the GGUF, calibrated against achieved Loom efficiency; picks per-metric targets vs llama.cpp and a time budget; a rate-of-approach controller decides continue / pivot / stop |
| `loom-optimize` | The measure / ablate / change / verify loop, a kernel A/B helper, and a catalog of optimizations that worked (and that didn't) |
| `loom-benchmark` | Token-by-token golden comparison against llama.cpp ROCm, a multi-turn tool-calling gate, a full-context needle gate, a batching-consistency check, throughput workloads, reporting rules |
| `loom-improve` | Improve the skills themselves: the KPIs (wall clock, agentic correctness vs llama.cpp, speed), fresh-agent trials on a reference model suite, a stall watchdog, a KPI regression check and a fixed quiet-GPU evaluator. Prompt: "your goal is to improve the loom skill at X" |
| `loom-org` | Populate an HF org (loom-fun) with the top models that fit Strix Halo: rank and pick models, classify what the org has (category A current, B older skill), then loop: port and optimize from the best category A reference, publish, sweep expensive gates on DevLab, fix bugs in the skill and regenerate, until every model is correct and fully assessed |

A starter-covered model goes from prompt to report within its roofline budget (port ≈1 min,
qualify ≈10 min, then optimization and finalize). An architecture the starter doesn't
cover adds the hand port; `loom-port-llm` has a playbook for it (llama.cpp's own compute
graph and intermediate tensors via `llama_trace.py`, numpy node-by-node checks), and the
optimization budget shrinks to fit.

## Runtime requirements

- Linux or Windows 11 (Git Bash), and an AMD GPU (ROCm-class; developed on gfx1151).
  Windows was verified end to end on Strix Halo with Qwen3-0.6B; see the `loom` skill's
  Windows section. The NPU path needs an
  XDNA device.
- Python ≥ 3.12 with `venv`, and `git`. The SDK comes from PyPI
  ([`hrx-loom`](https://pypi.org/project/hrx-loom/), Linux glibc ≥ 2.39 or Windows 11).
  `gh` is only needed for development wheels from CI (`install_loom.sh ROOT ci-latest`).
- For the golden reference: a llama.cpp ROCm build from
  [lemonade-sdk/llamacpp-rocm](https://github.com/lemonade-sdk/llamacpp-rocm). The skills
  download its latest release by themselves (pre-approved); they ask before running a binary
  from any other source.

## Safety notes baked into the skills

- **Memory limit.** Never serve a model whose device-resident weights plus KV exceed the GPU's
  memory budget: overcommitting unified memory hard-crashes the machine.
- **`--memory_bytes`.** Always pass it to `loom-serve`.
- **Downloaded binaries.** Ask before running one, except Loom from PyPI and lemonade-sdk/llamacpp-rocm releases (pre-approved).
- **Golden references.** They are external (llama.cpp), never a model written by
  hand.

## Repository layout

```
.claude-plugin/{plugin.json,marketplace.json}   # plugin + single-plugin marketplace
skills/<skill>/SKILL.md                          # nine shared skills, with scripts and references
scripts/install_skills.py                        # native discovery adapters and managed copy install
install.sh                                       # Bash wrapper for the Python installer
LICENSE                                          # MIT
THIRD_PARTY_NOTICES.md
```

Each skill resolves bundled paths from its own `SKILL.md` location, then binds
`LOOM_SKILL_DIR` for shell commands. Claude Code supplies `${CLAUDE_SKILL_DIR}` automatically;
Codex and Pi supply the loaded file's path. This works independently of the shell working directory.
Subagents are optional; serial execution preserves the same correctness gates.

## Contributing

```bash
python3 -m unittest discover -s tests -v
```

CI runs these regressions on Linux and Windows. See [compatibility checks](docs/agent-compatibility.md)
for fresh-agent smoke tests and GPU trials.

## License

MIT; see [LICENSE](LICENSE). Bundled third-party material keeps its own license; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
