# Family gemma4: porting

## Porting: features the starter lacks (Gemma-3n / Gemma-4)

- **Gemma-3n / Gemma-4 family (arch `gemma3n`, `gemma4`).** `apply_example.sh gemma4`: a complete,
  gate-passing port whose README lists every feature the starter lacks (PLE, KV sharing, mixed
  head_dim 256/512, KV heads per layer kind and V = K on global layers without `attn_v` (12B/26B/31B), MoE
  layers beside the dense FFN (26B-A4B), proportional partial RoPE, V norm without weights, GEGLU, softcap, bf16); each
  one silently breaks output if missed. Read llama.cpp's `src/models/gemma*.cpp` for a new variant.
  Sliding layers with a small window (<= ~1k keys) don't need flash prefill: split-K is fine there
  (a trial spent 15 min on it for no gain).

## Gemma-4 assistant drafter (`gen_assist.py`)

llama.cpp src/models/gemma4-assistant.cpp, mapped onto gen_draft.py's MTP machinery (per-row carry of the target's
final normalized hidden, `g_mtp_prepare` / `g_mtp_gather` / `g_mtp_advance`). Each round: target embedding *
sqrt(H) concatenated with the hidden -> `nextn.pre_projection` -> 4 layers (query-only attention, GELU FFN, post
norms, `layer_output_scale`) -> the assistant's tied head (argmax) and `nextn.post_projection` (the next round's
hidden). The attention reuses the trunk's own decode kernels for the two target layers (parsed from the program text,
`target_attention`): the assistant's QA heads are placed in the target's QH-head layout so head a reads KV head
a // (QA/KVH), and `g_as_visible` clamps every round's queries to the anchor's context (positions < P). The propose
stage therefore binds the target's origins table and arena (build_draft.py); warm only saves the carry.
Nemotron-3.5-Bolt-E4B with the E4B assistant: acceptance equals llama.cpp's (depth 1: 75 % vs 77 %, depth 3: 57 % vs
57 %), spec gate 16/16 exact including images, 46.9 -> 72.7 tok/s at depth 3.

## Lessons from the E4B port (from the gemma4 kit)

Lessons the agent paid for:
- `llama_trace.py ... --grep '-0$|-23$|-24$'` (a sliding, a global and a KV-shared layer) gave the whole
  spec in one run; the kernels were right the first time.
- Bugs in a hand-written `output_next` show up only on real outputs: an empty span must not change
  state (see the loom-kb `porting` topic), and the reasoning close is `<channel|>` with no newline.
- Sliding layers with a window of <= ~1k keys don't need flash prefill (split-K is fine; a trial wasted
  15 min on it). head_dim > 256 needs split-K attention generalized to D/32 values per lane and flash2's
  D/16 fragment loop.
- Known gap: with thinking ON, history after a tool result is not reused (the template re-renders the
  reasoning with a `\n` before `<channel|>` that the model does not generate). loom-serve resumes a retained
  row only when the next prompt extends its history exactly (hrx-system `input.c`: the transcript must be a
  prefix; an idle row cannot be cut back to the common prefix, and the prefix cache only captures
  intermediate frontiers of long prefills), so neither the chat program nor storage can reuse the common
  prefix: the turn is prefilled again (~200 tokens per tool step in the agent gate). The agent gate requires
  reuse only where the runtime can give it (`agent_golden.py`: history the model verifiably generated and
  the template rewrote is not required to be reused).
  Likewise for templates whose thinking-off generation prompt adds an empty thought channel (Gemma 4
  12B/26B): the template does not render it in history, so the next turn re-prefills from there.

## Gemma4 QAT target packages

The exact Unsloth31B and26B-A4B QAT releases use Q4_0/F32 target tensors even though
their upstream filenames end in `UD-Q4_K_XL`. The dense31B path uses existingQ4_0
matrix kernels; the26B-A4B path additionally needs Q4_0 expert decode and grouped
prefill. Expert staging preserves the signed half-scale and centered nibble codes.

Scope is text/chat/tools with thinking toggle; vision still needs QAT-specific
integration. Existing non-QAT packages and their evidence are separate.

Drafting: the exact-QAT assistant (Google's `gemma-4-<M>-it-qat-q4_0-unquantized-assistant`,
Unsloth's GGUF `mtp-gemma-4-<M>-it.gguf` at the root of the target repo, = `MTP/...-Q4_0.gguf`)
runs on the `assist` path (`draft/gen_assist.py`). Unlike E2B/E4B's it lists
`head_count_kv` per layer (31B [16,16,16,4], 26B-A4B [8,8,8,2]: sliding layers read target
layer L-2 with `kv_heads`, the full layer reads L-1 with `kv_heads_global`), so the head
grouping is per layer kind; its weights are Q4_0; `find_drafter.py` finds it in the target's
own repo first. It is the classification's choice (rule 1: exact-model drafter). The Anbeeld
DFlash drafters are trained for the non-QAT targets: measured for comparison only, never a
card choice.

Initial fleet sizing is a CPU-generated input, not measured device fit or native
context support. Device-specific settings and four model/device qualification records
must come from the exact QAT weights before final publication. Cold host/device peaks,
completed full-limit requests and structured over-limit rejection are required; keep
sampled memory counters separate and distinguish served context from native context.
