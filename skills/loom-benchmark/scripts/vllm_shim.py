#!/usr/bin/env python3
"""llama-server look-alike on top of vLLM: the golden reference engine for safetensors packages.

usage: vllm_shim.py CHECKPOINT [--port 8130] [--max-model-len N] [--revision REV] [--mock]
       (stop: SIGTERM; vllm_server.sh starts and stops it)

The gates talk to a reference server through llama-server's /completion (bench_common.llama_completion), so one small
shim covers every gate with no change to them. It serves the subset the gates use and nothing else:
  GET  /health            {"status": "ok"} once the engine is loaded
  GET  /props             {"engine", "version", "checkpoint", "revision", "dtype", "quantization"}: what the card states
  POST /completion        prompt = list of token ids, or {"prompt_string", "multimodal_data": [base64 images]}
                          (vision_golden.py; "<__media__>" marks the image, replaced by IMAGE_PLACEHOLDER);
                          n_predict, n_probs, temperature 0 (greedy: anything else is refused), return_tokens.
                          Answer: {"content", "tokens", "completion_probabilities": [{"id","token","logprob",
                          "top_logprobs": [{"id","token","logprob"}]}], "tokens_evaluated", "timings": {"prompt_n"}}
  A `_micro_batch` request (llama.cpp's decode-kernel view through a slot cache) has no vLLM equivalent: answered with
  an empty view, which bench_common.llama_views leaves out.
Greedy decoding with the checkpoint's own quantization config (quark: vLLM emulates the W4A4 activation quantization
where the GPU has no native MXFP4); bf16, one request at a time (the gates are sequential), prefix caching off so every
request is evaluated in full (the llama.cpp `cache_prompt: false` equivalent).
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MARKER = "<__media__>"
# Qwen-VL family: what vLLM's processor expands into the image tokens. A package of another family overrides it.
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def completion_response(token_ids, logprobs, text, n_probs, prompt_n, token_text=None):
    """Map vLLM's output to llama-server's /completion JSON.
    token_ids: generated ids; logprobs: per position {id: (logprob, decoded_token)} (vLLM: the top-k plus the sampled
    token); the chosen token is the one that was sampled, and each position lists its n_probs best, best first."""
    cp = []
    for tid, lp in zip(token_ids, logprobs or []):
        top = sorted(lp.items(), key=lambda kv: -kv[1][0])[:n_probs]
        chosen = lp.get(tid)
        cp.append({"id": tid, "token": chosen[1] if chosen else "", "logprob": chosen[0] if chosen else None,
                   "top_logprobs": [{"id": i, "token": t, "logprob": p} for i, (p, t) in top]})
    return {"content": text, "tokens": list(token_ids), "completion_probabilities": cp,
            "tokens_evaluated": prompt_n, "timings": {"prompt_n": prompt_n, "predicted_n": len(token_ids)},
            "stop": True}


def parse_request(body, placeholder=IMAGE_PLACEHOLDER):
    """-> (engine prompt dict, images as bytes, n_predict, n_probs). Raises ValueError on what the shim cannot answer."""
    if body.get("temperature", 0) not in (0, 0.0, None):
        raise ValueError("references are greedy: temperature must be 0")
    n_predict, n_probs = int(body.get("n_predict", 64)), int(body.get("n_probs", 0))
    prompt = body.get("prompt")
    images = []
    if isinstance(prompt, list):
        eng = {"prompt_token_ids": [int(t) for t in prompt]}
    elif isinstance(prompt, dict):
        text = prompt["prompt_string"]
        images = [base64.b64decode(m) for m in prompt.get("multimodal_data") or []]
        if text.count(MARKER) != len(images):
            raise ValueError(f"{text.count(MARKER)} media markers for {len(images)} images")
        eng = {"prompt": text.replace(MARKER, placeholder)}
    elif isinstance(prompt, str):
        eng = {"prompt": prompt}
    else:
        raise ValueError("prompt must be token ids, a string or {prompt_string, multimodal_data}")
    return eng, images, n_predict, n_probs


def _derived_max_len(checkpoint, revision=None):
    try:
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(checkpoint, revision=revision)
        c = getattr(c, "text_config", None) or c
        return int(getattr(c, "max_position_embeddings", 0) or 0) or None
    except Exception:
        return None


class VllmEngine:
    """vLLM offline engine (imported lazily: the unit tests never load it)."""

    def __init__(self, checkpoint, max_model_len, revision=None, dtype="bfloat16", seed=0, extra=None, gpu_util=0.85, images=0, max_batched=None):
        import vllm
        self.version = vllm.__version__
        derived = _derived_max_len(checkpoint, revision)
        if derived and max_model_len > derived:   # vLLM refuses a longer window than the checkpoint declares: longer goldens are skipped by golden.py
            print(f"max_model_len {max_model_len} -> {derived} (the checkpoint's max_position_embeddings)", flush=True)
            max_model_len = derived
        kw = dict(model=checkpoint, dtype=dtype, seed=seed, max_model_len=max_model_len, enable_prefix_caching=False, gpu_memory_utilization=gpu_util,
                  limit_mm_per_prompt={"image": images, "video": 0}, **(extra or {}))
        if max_batched:   # smaller encoder-cache profile: gfx1151's math SDPA fallback allocates tokens^2 for the image profile (256 GiB at 16384)
            kw["max_num_batched_tokens"] = max_batched
        if revision:
            kw["revision"] = revision
        self.llm = vllm.LLM(**kw)
        self.checkpoint, self.revision, self.dtype = checkpoint, revision, dtype
        qc = getattr(self.llm.llm_engine.model_config, "quantization", None)
        self.quantization = qc

    def generate(self, eng_prompt, images, n_predict, n_probs):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0, max_tokens=n_predict, logprobs=max(n_probs, 1), seed=0)
        if images:
            from PIL import Image
            eng_prompt = dict(eng_prompt, multi_modal_data={"image": [Image.open(io.BytesIO(b)).convert("RGB") for b in images]})
        out = self.llm.generate([eng_prompt], sp, use_tqdm=False)[0]
        o = out.outputs[0]
        lps = [{i: (l.logprob, l.decoded_token or "") for i, l in pos.items()} for pos in (o.logprobs or [])]
        return list(o.token_ids), lps, o.text, len(out.prompt_token_ids)

    def props(self):
        return {"engine": "vllm", "version": self.version, "checkpoint": self.checkpoint, "revision": self.revision,
                "dtype": self.dtype, "quantization": self.quantization or "none (bf16 weights)"}


class MockEngine:
    """Deterministic stand-in (--mock, tests): continues a prompt with ids (last + 1 + i) % 1000."""
    version, checkpoint, revision, dtype, quantization = "mock", "mock", None, "bfloat16", None

    def generate(self, eng_prompt, images, n_predict, n_probs):
        ids = eng_prompt.get("prompt_token_ids") or [ord(c) for c in eng_prompt["prompt"]]
        gen = [(ids[-1] + 1 + i) % 1000 for i in range(n_predict)]
        lps = [{t: (-0.1, f"t{t}"), (t + 1) % 1000: (-2.5, f"t{(t + 1) % 1000}")} for t in gen]
        return gen, lps, "".join(f"t{t}" for t in gen), len(ids) + len(images)

    def props(self):
        return {"engine": "mock", "version": "mock"}


def make_server(engine, port, placeholder=IMAGE_PLACEHOLDER, host="127.0.0.1"):
    lock = threading.Lock()   # one request at a time: the gates are sequential and the engine is not re-entrant

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/health"):
                self._send(200, {"status": "ok"})
            elif self.path.startswith("/props"):
                self._send(200, engine.props())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/completion"):
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if body.get("_micro_batch"):
                    return self._send(200, {"completion_probabilities": [], "note": "no micro-batch view on vLLM"})
                eng, images, n_predict, n_probs = parse_request(body, placeholder)
                with lock:
                    ids, lps, text, prompt_n = engine.generate(eng, images, n_predict, n_probs)
                self._send(200, completion_response(ids, lps, text, n_probs, prompt_n))
            except ValueError as e:
                self._send(400, {"error": str(e)})
            except Exception as e:   # an engine failure must not look like an answer
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):
            pass

    return ThreadingHTTPServer((host, port), H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--port", type=int, default=8130)
    ap.add_argument("--max-model-len", type=int, default=36864)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--image-placeholder", default=IMAGE_PLACEHOLDER)
    ap.add_argument("--gpu-util", type=float, default=float(os.environ.get("VLLM_GPU_UTIL", 0.85)))
    ap.add_argument("--images", type=int, default=int(os.environ.get("VLLM_IMAGES", 0)),
                    help="images per prompt (0: text only; vLLM then skips the vision encoder's memory profile, which tries a 256 GiB attention matrix on a VL checkpoint)")
    ap.add_argument("--max-batched-tokens", type=int, default=int(os.environ.get("VLLM_MAX_BATCHED_TOKENS", 0)) or None)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    engine = MockEngine() if a.mock else VllmEngine(a.checkpoint, a.max_model_len, a.revision, gpu_util=a.gpu_util, images=a.images, max_batched=a.max_batched_tokens)
    print(f"READY vllm_shim {engine.props()}", flush=True)
    make_server(engine, a.port, a.image_placeholder).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
