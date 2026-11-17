# qwen35 hybrids over the elastic memory budget: garbled output, then a server crash (hrx-loom 0.2.0)

Notes only (not filed upstream). Seen on the gfx1201 16 GB class (Radeon RX 9070 XT, Linux 7.0, hrx-loom 0.2.0 from PyPI), 2026-11-17,
while measuring `fleet/devices/gfx1201-16gb.json`. Nothing in it is specific to that GPU: it is a property of elastic backing with a
`--memory_bytes` budget, so it can affect any GPU served that way.

## What happens

A text request whose context is a little larger than the budget can hold should be refused at admission with HTTP 503 ("retained state
leaves insufficient completion capacity"), and is for every other architecture in the fleet. For the `qwen35` architecture (Qwen3.5,
Ornith: gated delta-net recurrent layers plus attention) there are three bands instead:

| request vs the largest context that works | result |
|---|---|
| far above | clean 503 at admission (correct) |
| a few % above | admitted, then the server process dies mid-request ("Remote end closed connection without response") |
| 1-2 % above the working maximum | admitted and COMPLETED, but the next request's answer is garbled multi-script text |

The bands move with the budget (14e9 instead of 15e9: every threshold drops), so this is the admission accounting against Loom's own
budget, not physical VRAM: likely the recurrent state (or another per-request allocation of this architecture) is not counted when the
request is admitted.

## Repro

Package: the card's package built on 0.2.0 (`build.py --memory-gib 4096`), its default preset, `context_capacity` overridden to the request
size through a deployment document. Server (fresh per probe):

    loom-serve --model=deployment.json '--device=amdgpu://?completion_spin_ns=4294967295' \
      --memory_bytes=15000000000 --state_slab_bytes=4mib --weight_slab_bytes=64mib --parameter_mode=preload

    deployment.json: {"name": MODEL_ID, "model": {"path": "PKG/model.json"},
                      "definition": {"kind": "text", "context_capacity": N}}

Then one chat request of N - 128 prompt tokens (real text, `max_tokens` 1, thinking off), then "What is the capital of France? Answer in one
word." (`max_tokens` 1024, temperature 0). `skills/loom-benchmark/scripts/test_9070xt.py PKG --loom-root ROOT --weight-slab-bytes 64mib`
runs exactly this as a bisection and records each probe (`confirm`) and the garbled answers (`wrong_answers`).

## Evidence (context_capacity N per probe, fresh server each)

| model | budget | admitted, correct after | admitted, garbled after | crash | clean 503 |
|---|---|---|---|---|---|
| Qwen3.5-4B | 15e9 | 151552 | 152576 | 154624 | 162816 |
| Qwen3.5-4B | 14e9 | 134144 | 135168 | 136192 | 151552 |
| Qwen3.5-9B | 15e9 | 87040 | 88064 | 89088 | 97280 |
| Ornith-1.5-9B | 15e9 | 93184 | 94208 | 95232, 97280 | 130048 |
| Ornith-1.0-9B | 15e9 | 224256 | (none seen) | 225280 | 229376 |

Garbled answers (the content field, first characters): Qwen3.5-4B `"!覆[,]euxigiauroง itzer餐限显arenteket boutallon影虾…"`,
Qwen3.5-9B `"!нду独有禾 Fakiseauáros ANEL Pics不成功jadienheim…"`, Ornith-1.5-9B `"!отре籁جري筏uzziAINER…"`. A fresh server answers
"Paris" at every context up to the working maximum. Every other architecture of the 32 models that fit 16 GB (Qwen2.5, Qwen3, Qwen3-VL,
gemma-3/4, LFM, Llama, Nemotron, gpt-oss, Mellum, HyperCLOVAX, Spark) gave only clean 503s above its maximum.

## Consequences

- The 16 GB class's `max_context` for these models is the largest context with a correct answer after a full-length request, and the preset
  sets `context_capacity` to it, so a single request cannot cross the line. Several concurrent long requests could (the budget is shared by
  the rows): a risk wherever qwen35 models are served elastic with a budget (the Strix Halo cards too).
- A fix belongs in the SDK's admission (count the recurrent state against the budget) or in the qwen35 programs' declared storage sizes
  (`storage.loom`), not in the device data.
