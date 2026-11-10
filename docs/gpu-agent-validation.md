# GPU agent validation

Objective: validate real GPU porting with Claude Code, Codex and Pi, using a small
model with a published cousin, and **no needle-in-a-haystack tests**.

- Windows pilot: `Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF`, revision
  `ebb2015119c907b064c512bf053e945850b5875f`, file
  `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf`.
- Pilot cousin: `loom-fun/bartowski-Qwen2.5-0.5B-Instruct-loom`, revision
  `dad263bdf7c45f2e41c05b02df6cb7fae99f20c3`. Its generated `loom-card/2` card
  and qwen2 architecture were verified before selecting the target. This is a
  different fine-tuning of the same small architecture, not a new kernel family.
- Final common target: `unsloth/Qwen3-0.6B-GGUF`, revision
  `50968a4468ef4233ed78cd7c3de230dd1d61a56b`, file `Qwen3-0.6B-Q4_K_M.gguf`.
  Force cousin `loom-fun/ggml-org-Qwen3-1.7B-loom`, revision
  `caee8b360bf82572dd561da478f1eb4aed847f2d` (verified generated `loom-card/2`,
  qwen3, reasoning toggle and Hermes tools). Do not reuse the catalog's exact
  0.6B port: independently build each trial from the GGUF and the 1.7B cousin.
  The coder pilot passed golden/tool-prompt parity but emitted no actual calls
  in either engine. Qwen3 is selected for stronger tool/reasoning coverage,
  rather than counting those vacuous tool-call checks as broad validation.
- Hardware: local AMD Radeon 8060S / gfx1151. Trials run serially with separate
  work directories and ports, leaving unrelated resident servers alone.
- SDK: the existing PyPI 0.1.0 Windows installation at `~/loom010`, source
  revision `d01d1c18b154189fd073777329052d23f6071f14`, wheel SHA256
  `d368cf8be58f00f3e9c765df4031964d2fc07fb086a9b5241830f05e95725f83`.
  Reusing a fixed SDK isolates porting behavior from installation differences.
- Trials use frozen skills and independently build/serve their own packages.
  HF download caches and the fixed SDK/reference binaries may be shared; prior
  agents' packages, fixes and transcripts are excluded from each new trial.
- `SKIP_NEEDLE=1` excludes every needle gate in dev/quick/full and prevents direct
  `needle.py` execution. Evidence records the exclusion; it is never a pass.

Acceptance for **each** agent: actual GPU compilation and serving; kernel tests;
verified chat template; golden comparison against GPU llama.cpp; consistency,
tool/history and sampling gates where applicable; measured decode/prefill and a
report stating needle coverage was excluded. Check inherited reference settings
and retain only a justified measured improvement; exhaustive tuning is outside
this validation request. No requirement to invent a speedup when already tuned.

Compare the same checkpoint and reference build across agents. Run final gates
on the final package, inspect authoritative logs and artifacts rather than only
the agents' summaries, and stop trial-owned servers between agents. If a common
script defect is found, fold back the general fix, freeze a new revision, and
revalidate affected earlier packages. Do not publish target models to HF.

## Results (November 9, 2026)

All three agents independently built and served the final Qwen3 target on the
AMD GPU using frozen skill revision `451d63e309fe9cf022d35136e75dd6a85b035568`.
The coordinator checked the artifacts and live deployments, rather than relying
on the agents' completion messages.

| Agent | Agent model | Kernel tests | Final scoped gates | Actual tool calls, thinking on / off | CLI wall time | Package hash |
|---|---|---|---|---|---|---|
| Claude Code 2.1.302 | Sonnet 5.5 | 57/57 | 7/7 PASS | 4 / 4 | 18.8 min | `ef40a339b617` |
| Codex CLI 0.167.0-alpha.3.2 | CLI default; model ID not exposed | 57/57 | 7/7 PASS | 4 / 4 | 17.5 min | `248f6f6bd57f` |
| Pi 0.73.1 | Lemonade `user.Qwen3.5-35B-A3B-ThinkingCoder` | 57/57 | 7/7 PASS | 4 / 4 | 22.7 min | `642bafdcc886` |

The seven gates are golden generation, comparison, consistency, speculative
decoding equivalence, agent/tool history with thinking on, the same with thinking
off, and sampling. Final full-level runs reused successful checks for the exact
package hash; the underlying executed logs were inspected. Needle and
needle_full are explicitly **skipped**, not counted among those seven passes.
All three audits have zero failing checks, with warnings retained.

Each checkpoint path was verified to refer to the exact pinned HF snapshot file
(396,705,472 bytes). Each package records the same clean skills revision, SDK
source revision and wheel hash, and the selected 1.7B cousin revision. Native
40,960-token context is preserved. Final live processes were checked for the
fixed SDK executable and `--device=amdgpu`. Independent HTTP probes obtained a
structured `get_weather` call for Paris, supplied a tool response, and received
a final answer using the supplied 18°C/sunny fixture. Gate logs additionally
verify rendering, history reuse, and both reasoning modes. This validates the
port and protocol behavior, not general model answer quality.

All three retained measured ngram depth 3. Pi also attempted a prefill setting
through an environment variable that the builder does not consume; its candidate
had the same executable package hash and still used prefill capacity 1024. No
gain is attributed to that attempt.

### Reference measurements

Claude and Codex used installed ROCm llama.cpp `c08d28d` with `ngram-simple`.
Pi's own discovery selected fetched build `b1372`. A separate coordinator
supplement against the common `c08d28d` reference passed all seven gates from a
fresh reference cache, final audit, and another independent live tool-response
probe. Pi's original results remain intact. The common executable SHA256 is
`84992a9de949743eaeaa3700c4f5371c85e5a5fc5be180dbfdee5cc1c822b851`.

Observed final tok/s, **Loom / common reference**; Pi uses the coordinator
supplement. These are separate sessions, not a controlled agent performance
ranking. The report uses the upper median when two short rounds are available.

| Agent | Decode | Decode 2k | Prefill 1k | Prefill 2k | Four-stream throughput |
|---|---|---|---|---|---|
| Claude | 250.36 / 212.71 | 268.11 / 100.75 | 12720.1 / 3586.0 | 12639.6 / 2633.6 | 507.9 / 289.1 |
| Codex | 259.50 / 206.15 | 204.86 / 99.18 | 10662.8 / 3404.8 | 10515.3 / 2345.9 | 492.7 / 273.9 |
| Pi, common-reference supplement | 272.73 / 160.00 | 216.27 / 88.90 | 10719.9 / 3291.3 | 10401.3 / 2380.1 | 518.1 / 242.1 |

Measurements also include long decode and prefill. Shared-machine variability and baseline-efficiency
warnings prevent treating these sessions as stable idle-GPU speedup claims or
as a ranking of the agents. CLI times include different cache and supervision
costs and should not be used as an agent speed comparison.

### Friction and limits

- The Windows pilot exposed checkpoint/path, Bash subprocess, process restart,
  capability detection and EOS timing defects; general fixes were folded into
  the frozen revision before the final cohort. The initial coder pilot is not
  counted as one of the final three runs because neither engine produced actual
  tool calls.
- Claude encountered a coordinator SDK-environment mistake during a manual
  restart. The final deployment was corrected and independently verified; the
  runner now exports both `LOOM_ROOT` and `VENV` for later trials.
- Public reference discovery scans exact-target metadata as well as cousin
  metadata, and `show reference.json` displays its top-ranked result rather than
  the forced cousin. No exact-target kernels or weights were reused; selected
  reference configs report `taken: []`, since inspection derived every key.
  These are not blinded or pristine cold-cache trials. Codex explicitly marked
  its overall trial status failed for that isolation condition while separately
  marking GPU validation passed. That worker result is preserved; the
  coordinator's GPU acceptance is based on the executed evidence above.
- Pi recovered from malformed checkpoint arguments and Windows process-control
  mistakes. The coordinator removed its unused, verified candidate server before
  completing the validation. Passing the workflow does not imply flawless agent
  operation.
- GitHub's latest-release API returned 403 during some discovery attempts;
  installed builds remained available. The fixed-reference supplement removes
  baseline differences from the controlled comparison.
- SDK installation was reused rather than tested. This is one small supported
  architecture on one Windows/AMD host, not evidence for every model or OS.
- No needle retrieval or full-context capacity probe ran. Ordinary long-document
  golden and performance requests are distinct from those excluded tests.

Raw transcripts, packages, reports and coordinator audits are under
`.agent-smoke/gpu/` (gitignored). [Compact audited evidence](validation/agent-gpu-qwen3-2026-11-09.json)
records the revisions, package hashes, comparisons and measurements. No
credentials, weights or raw session transcripts are committed. All trial-owned
servers were stopped after verification; the unrelated resident agent server
was left running.
