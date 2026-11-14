# Facet: hybrid-gdn: porting (recurrent state, other recurrent layers, cfg check)

## Recurrent state (hybrid models)

`storage.loom`'s geometry record declares recurrent **slots** in the arena: words
4–7 are origin, stride, length (bytes per slot) and capacity. The runtime assigns
each resident row a reader slot and a writer slot and records them, with the fork
position, in the row's origins entry (`encode_state`): words 0 (reader byte offset),
1 (writer byte offset) and 3 (fork position). A span that starts at the fork
position continues the reader's state (e.g. a reused prefix or a retained agent
turn); any later span continues the row's own writer; a span at position 0 starts
from zeros. Every span publishes its final state to the writer. Prefix reuse for
hybrids happens only at retained frontiers (the end of a previous turn), which is
exactly what agent loops need. `gen_gdn.py` implements this protocol.

## Other recurrent layers (Mamba, RWKV, other delta-net variants)

- **Other recurrent layers (Mamba, RWKV, other delta-net variants).** Follow
  `gen_gdn.py`: per-span kernels read the reader slot at the fork position (else the
  row's writer slot), start from zeros at position 0, and publish the final state to
  the writer slot; `build.py` sizes the slot (`gdn_geometry()["slot_bytes"]`). Test the
  chain like `make_kernel_tests.py`'s `gdn_chain` (a span that continues a stored state).

## Porting check for hybrids

- Hybrids (`recurrent_layers` present): check `gdn` against llama.cpp's
  `src/models/qwen35.cpp` (k/v heads, head dims, conv kernel), that `kv_layers`
  counts only the attention layers, `rope_dims` (Qwen3.5: 64 of 256), `attn_gate:
  "fused_q"`, and `ffn_norm_tensor: "post_attention_norm"`.

## Folded into the starter from trials

**Folded into the starter from trials** (already on; don't redo): the gated delta-net conv and
history-commit kernels spread channels over `GDN_CB` workgroups per token when fewer than 64
tokens are in flight (Qwen3.5-0.8B decode +18 %, identical output; `GDN_CB=1` restores the old grid).
