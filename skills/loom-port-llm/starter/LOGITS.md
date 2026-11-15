# Opt-in logits readback (loom-skills#736)

Generated packages sample on the device. This feature lets a host that links `libloom_text_embed` (the Gufo extension) read the
full-vocabulary logits row of chosen spans, to sample on the host (grammars, logprobs, min_p, penalties). Every package carries it;
it is off at load and off per span, and the default stage compiles to the same code with it off (tests/test_logits_fold.py).

## Switches (both opt-in)
1. **Load-time capability.** Card config key `llm.logits` = `true` (bool, default false). The embed layer adds it to the card's
   `config` JSON before loading (`variants.<v>.services.chat.config`, next to `llm.card`); to try it from a file, set
   `"llm.logits": true` there in a copy of `model.json`. It reaches `storage.loom` (the `outputs` allocation grows) and every
   `@llm_advance` stage (the export launch is compiled in). Without it the allocation and the stage are the default ones.
2. **Per-span flag.** Bit 30 of the sampling record's `top_k` (control.loom `request_options`): send
   `"top_k": 1073741824 + k` (k <= 64 is the device top-k; the value is ignored when the host samples). Packages built before this
   change read it as k = 64: harmless. A span without the bit exports nothing. Only spans that select a token (the last token of a
   prompt, every decode step) have a row.

## Where the row is
After an ordinary epoch (undrafted: `loom_serve_text_model_epoch`, no output limits), the `outputs` allocation (state allocation 5)
holds, besides the span tokens in its first bytes:

| field | value |
|---|---|
| region start | byte 32768 (after the 8192 span tokens of 4 B) |
| row of span i | `32768 + i * row_bytes`, i = the span's index in the epoch plan |
| row_bytes | `ceil(vocab / 128) * 512` (f32, little endian) |
| valid values | the first `vocab` floats: the logits as the sampler sees them (a logit softcap is applied, argmax unaffected) |
| padding | the rest of the row is `-3e38` |

The embed layer downloads it with the runtime's transfer API (`loom_serve_text_state_allocation(state, 5)` +
`loom_serve_execution_transfer`, the pattern of `loom_text_embed.c` checkpoint images). At T=0 `argmax(row)` is the device-selected
token (ties: lowest index). The device token is still returned through the ordinary feedback; the host feeds its own sampled token
back through the forced-token path.

Allocation size with the capability: `32768 + span_capacity * row_bytes` (slot 5 only; e.g. 8 spans x 151,936 vocab = +4.9 MB).
Drafted packages: advance (undrafted) epochs export; verify, warm and propose stages are untouched.

## Exercise it without Gufo
- Kernel: `make_kernel_tests.py` emits `@logits_export` (bit-exact copy, an unflagged and an unselected span export nothing);
  `run_kernel_tests.sh` runs it with the other kernel tests on a GPU.
- Program: `fleet/build_fleet.py` builds every package's programs offline; `loom-opt llm.loom --pass=canonicalize --pass=cse --pass=dce
  --pass=symbol-dce --config=llm.context=... --config=llm.logits=true|false` shows the stage with and without the export launch.
- Package: set `"llm.logits": true` in the card config, serve with the embed layer, send a request whose `top_k` carries bit 30, and
  download allocation 5 after the epoch (layout above).
- Opt out of the code entirely for a model: registry choice `feature.logits_export` = false (env `LLM_LOGITS=0`).
