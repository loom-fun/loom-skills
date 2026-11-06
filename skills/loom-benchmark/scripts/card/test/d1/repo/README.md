---
base_model: LiquidAI/d1-2B-GGUF
license: other
tags:
- loom
- loom-serve
- gguf
- strix-halo
---

# d1-2B-GGUF for AMD Loom

A [Loom](https://github.com/ROCm/hrx-system) package (`.loom` programs and a `model.json` card) for the
unmodified GGUF checkpoint [LiquidAI/d1-2B-GGUF](https://huggingface.co/LiquidAI/d1-2B-GGUF) (`Q4_K_M`), generated on an AMD Ryzen AI Max+ 395
"Strix Halo" by the [loom-skills](https://github.com/loom-fun/loom-skills)
factory workflow.

The weights are not included. `loom-fetch` (from the Loom SDK, installed with its `hf` extra by the first command) installs the programs
from this repository and the GGUF from [LiquidAI/d1-2B-GGUF](https://huggingface.co/LiquidAI/d1-2B-GGUF), checked against `model.json.lock`:

```bash
pip install "hrx-loom[hf]"
loom-fetch https://huggingface.co/loom-fun/LiquidAI-d1-2B-loom/resolve/main/model.json --materialize=copy --output=d1-2B-GGUF
loom-serve --model=d1-2B-GGUF/model.json --device=amdgpu
```

This is a decision model, not a chat model: it answers `choice`, `score` and `noul` questions about a
text or JSON state in one forward pass, without generating text. The server listens on `127.0.0.1:8080` and
answers them at `/v1/systemone`:

```bash
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "LiquidAI/d1-2B-GGUF",
  "state": "I was charged twice this month, please refund one of them.",
  "questions": {
    "refund": {
      "type": "noul",
      "instructions": "Is the customer asking for a refund?"
    },
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Charges, refunds, invoices",
        "technical": "App or site faults"
      }
    }
  }
}'
```

Choice and score answers come back with their full probability maps, and noul answers with the probability of
`true`.

[LiquidAI/d1-2B-GGUF](https://huggingface.co/LiquidAI/d1-2B-GGUF) is tagged `image-text-to-text`, but this Loom package takes text input only.
