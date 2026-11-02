"""Shared helpers for the Loom benchmark scripts.

A *package* is a directory with model.json + tokenizer.json + llm_config.json
(written by loom-port-llm/starter/build.py). llm_config.json's "chat" block is
the single source of truth for prompt rendering, so llama.cpp (golden, fed raw
token ids) and Loom (fed chat messages) see byte-identical prompts.
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
# chat_probe.simulate() is the Python model of the starter's control.loom rendering. A port that
# extends the kit (a copied starter with its own chat_probe.py) sets LOOM_KIT=that dir.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "loom-port-llm", "starter"))
if os.environ.get("LOOM_KIT"):
    sys.path.insert(0, os.path.abspath(os.environ["LOOM_KIT"]))
import chat_probe  # noqa: E402

PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain what a hash map is in two sentences.",
    "What is 17 * 23?",
    "List three prime numbers greater than 50.",
    "Translate 'good morning' into Spanish and German.",
    "Write a haiku about compilers.",
    "What does the Unix command `grep -r` do?",
    "Why is the sky blue? Keep it short.",
    "Give me a SQL query that counts rows in a table named users.",
    "What is the difference between a process and a thread?",
    "Name the planets of the solar system in order.",
    "Fix the bug: def add(a, b): return a - b",
    "What is the time complexity of binary search?",
    "Summarize the plot of Romeo and Juliet in one sentence.",
    "Write a bash one-liner to count lines in all .py files.",
    "What year did the first moon landing happen?",
    "Convert 100 degrees Fahrenheit to Celsius.",
    "What is a mixture-of-experts language model?",
    "Write a regular expression that matches an email address.",
]


DOC_SIZES = (1000, 2000, 8000, 16000, 32000)


def perf_metrics(row):
    """{metric: tok/s} of one perf_log.jsonl row, in roofline.json metric names: decode_tps[_1k/_2k/_32k],
    c4_steady, and prefill_<document size> (each prefill sample bucketed by its prompt tokens)."""
    out = {k: row[k] for k in ("decode_tps", "decode_tps_1k", "decode_tps_2k", "decode_tps_32k", "c4_steady")
           if row.get(k) is not None}
    for p in row.get("prefill", []) + row.get("prefill_long", []):
        out[f"prefill_{min(DOC_SIZES, key=lambda x: abs(x - p['prompt_tokens']))}"] = p["prefill_tps"]
    return out


# The smaller documents are byte-exact prefixes of the two stored ones (prompts and reference-cache keys unchanged).
DOC_PREFIX = {1000: (2000, 4688), 8000: (32000, 36510), 16000: (32000, 73839)}


def doc(n):
    """Real-text document of ~n tokens (Loom docs markdown): 1000 2000 8000 16000 32000."""
    base, size = DOC_PREFIX.get(n, (n, None))
    data = open(os.path.join(DATA, f"doc_{base}.txt"), "rb").read()
    return (data if size is None else data[:size]).decode("utf-8")


class Package:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.path = path
        self.cfg = json.load(open(os.path.join(path, "llm_config.json")))
        self.chat = self.cfg["chat"]
        self.model_id = self.cfg["model_id"]
        self.tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))

    def render(self, user):
        """Exactly what control.loom renders for one user message + generation prompt."""
        return self.render_messages([{"role": "user", "content": user}])

    def render_messages(self, messages, tools=None, enable_thinking=None):
        """control.loom's rendering of a conversation (chat_begin + messages + chat_end)."""
        return chat_probe.simulate(self.chat, messages, tools, gen=True, enable_thinking=enable_thinking)

    def ids(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def raw_from_chat(self, message, finished, thinking=None, prompt=None):
        """Rebuild the model's raw output text from Loom's parsed channels so it can be
        re-tokenized and compared token-by-token with the golden ids. `thinking` is the
        request's thinking selection (default: the template default). A kit (LOOM_KIT) whose
        chat_probe.py defines raw_from_chat(chat, message, finished, thinking, prompt) overrides this;
        `prompt` is the rendered prompt when known (a format whose prompt can already open the
        reasoning channel, e.g. Gemma 4 after tool results, needs it)."""
        c = self.chat
        if hasattr(chat_probe, "raw_from_chat"):
            try:
                return chat_probe.raw_from_chat(c, message, finished, thinking, prompt)
            except TypeError:   # a kit written before the prompt argument existed
                return chat_probe.raw_from_chat(c, message, finished, thinking)
        on = c.get("thinking_default", c.get("starts_in_reasoning")) if thinking is None else thinking
        reasoning = message.get("reasoning_content")
        content = message.get("content") or ""
        calls = chat_probe.render_calls(c, message.get("tool_calls") or [])   # wraps pythonic call lists
        answer = content + (c["content_call_separator"] if content and calls else "") + calls
        if not (c.get("starts_in_reasoning") and on):
            return answer
        end = c.get("think_end", "</think>")
        after = c["think_close"].split(end, 1)[1] if end in c["think_close"] else ""
        text = reasoning or ""
        if answer or finished:
            text += end + after + answer
        return text


def _gguf_fingerprint(path):
    """Identity of a GGUF without hashing gigabytes: name, size and the first 4 MiB."""
    if not path or not os.path.exists(path):
        return str(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(4 << 20))
    return f"{os.path.basename(os.path.realpath(path))}:{os.path.getsize(path)}:{h.hexdigest()[:16]}"


_FP = {}


def llama_completion(url, body, pkg=None, salt="", timeout=7200):
    """llama.cpp /completion with a persistent on-disk cache.

    The llama.cpp reference only depends on the GGUF, the llama.cpp build and the request,
    never on the Loom build, so every gate run after the first reuses it: re-running the gates
    on an optimized build costs Loom time only. Cache: $LLAMA_CACHE or ./llama_cache (bench/).
    `salt` separates deliberate re-recordings (golden2: llama.cpp is not run-to-run deterministic).
    On a miss with $LLAMA_DEFER set (gates.sh: one model at a time), the request is appended to that
    file and a placeholder without tokens is returned; llama_fill.py answers the pending requests with
    only llama.cpp loaded, and the gate re-runs from the cache. Otherwise, when nothing answers at
    `url` and $LLAMA_START is set, that command is run once to start llama-server.
    """
    gguf = pkg.cfg.get("gguf") if pkg is not None else os.environ.get("LLAMA_GGUF")
    if gguf not in _FP:
        _FP[gguf] = _gguf_fingerprint(gguf)
    key = hashlib.sha256(json.dumps({"gguf": _FP[gguf], "build": os.environ.get("LLAMACPP_DIR", ""),
                                     "body": body, "salt": salt}, sort_keys=True).encode()).hexdigest()
    d = os.environ.get("LLAMA_CACHE", "llama_cache")
    path = os.path.join(d, key[:2], key + ".json")
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except ValueError:
            pass
    defer = os.environ.get("LLAMA_DEFER")
    if defer:
        with open(defer, "a") as f:
            f.write(json.dumps({"key": key, "gguf": gguf, "body": body, "salt": salt}) + "\n")
        return {"_suspect": "deferred: answered by llama_fill.py, then the gate re-runs", "_pending": True,
                "content": "", "completion_probabilities": []}
    ensure_llama(url)
    for attempt in range(2):
        r = requests.post(url + "/completion", json=body, timeout=timeout).json()
        # llama.cpp must have evaluated the whole prompt: fewer tokens means it answered from some
        # cached state (seen: an old build answering the previous prompt). Retry once, never cache it.
        n_eval = (r.get("timings") or {}).get("prompt_n")
        ok = n_eval is None or not isinstance(body.get("prompt"), list) or n_eval == len(body["prompt"])
        if ok:
            break
        print(f"[llama] WARNING: evaluated {n_eval} of {len(body['prompt'])} prompt tokens; retrying", flush=True)
    if not ok:
        r["_suspect"] = f"llama.cpp evaluated {n_eval} of {len(body['prompt'])} prompt tokens"
        return r
    if "completion_probabilities" in r or "tokens" in r:  # don't cache errors
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".{os.getpid()}"
        json.dump(r, open(tmp, "w"))
        os.replace(tmp, path)
    return r


# Every llama.cpp reference is greedy with the samplers forced off: some GGUFs set penalty_repeat
# (LFM2.5: 1.05), which llama.cpp applies even at temperature 0, while Loom decodes greedily.
GREEDY = {"temperature": 0, "cache_prompt": False, "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}


def llama_greedy(url, ids, n_predict, pkg, salt="", n_probs=5):
    """Greedy llama.cpp continuation of prompt ids -> (response, generated ids, aligned top lists).
    completion_probabilities omits byte-fallback tokens (emoji, partial UTF-8), so the ids come from
    `tokens` and the top-n lists are trusted only while they line up with them."""
    r = llama_completion(url, dict(GREEDY, prompt=ids, n_predict=n_predict, n_probs=n_probs, return_tokens=True), pkg, salt)
    cp = r.get("completion_probabilities", [])
    gen = r.get("tokens") or [x["id"] for x in cp]
    k = next((i for i, x in enumerate(cp) if i >= len(gen) or x["id"] != gen[i]), len(cp))
    return r, gen, cp[:k]


def near_tie(top, gold_id, loom_id, max_gap=1.0):
    """Is Loom's token at a divergence a near-tie of llama.cpp's greedy token? top: {id: logprob} from
    llama.cpp at that position. Yes when the log-prob gap is <= max_gap, or when llama.cpp itself is
    unsure there: Loom picked its second choice and its first has probability < 0.5 (flint: p 0.38 vs
    0.13, gap 1.03, while llama.cpp's own re-recording of the same prompt sat at gap 1.05). A token
    llama.cpp is confident against, or ranks lower, is a bug."""
    if gold_id not in top or loom_id not in top:
        return False
    if top[gold_id] - top[loom_id] <= max_gap:
        return True
    ranked = sorted(top, key=top.get, reverse=True)
    return ranked[:2] == [gold_id, loom_id] and math.exp(top[gold_id]) < 0.5


def top_logprobs(tops, k):
    """{id: logprob} at generated position k, or {} past the aligned lists (a byte-split token)."""
    return {t["id"]: t["logprob"] for t in tops[k]["top_logprobs"]} if k < len(tops) else {}


def ensure_llama(url):
    try:
        if requests.get(url + "/health", timeout=3).ok:
            return
    except requests.RequestException:
        pass
    cmd = os.environ.get("LLAMA_START")
    if not cmd:
        raise SystemExit(f"llama.cpp is not answering at {url} (and LLAMA_START is not set)")
    print(f"[llama] cache miss: starting llama-server ({cmd})", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode:
        raise SystemExit(f"could not start llama-server: {cmd}")


def stream(url, model, messages, max_tokens, extra=None, timeout=None, on_token=None):
    """Streams one chat completion (greedy unless `extra` says otherwise). Returns client-side TTFT,
    decode tok/s, usage, the text (reasoning + content), the finish reason and the HTTP status;
    on_token(t) is called with time.perf_counter() at every streamed token."""
    t0 = time.perf_counter()
    first = finish = None
    n = pt = 0
    text = ""
    body = {"messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": 0,
            "stream_options": {"include_usage": True}}
    if model:
        body["model"] = model
    body.update(extra or {})
    with requests.post(url + "/v1/chat/completions", json=body, stream=True, timeout=timeout,
                       headers={"Content-Type": "application/json"}) as r:
        if r.status_code != 200:
            return dict(status=r.status_code, error=r.text[:300], ttft=None, total=time.perf_counter() - t0,
                        tokens=0, prompt_tokens=0, decode_tps=None, text="", finish=None)
        for line in r.iter_lines():
            if not line.startswith(b"data: {"):
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
                pt = d["usage"]["prompt_tokens"]
            for ch in d.get("choices", []):
                dl = ch.get("delta", {})
                piece = (dl.get("reasoning_content") or "") + (dl.get("content") or "")
                if piece:
                    now = time.perf_counter()
                    first = now - t0 if first is None else first
                    if on_token:
                        on_token(now)
                    text += piece
                finish = ch.get("finish_reason") or finish
    total = time.perf_counter() - t0
    return dict(status=200, ttft=first, total=total, tokens=n, prompt_tokens=pt, text=text, finish=finish,
                decode_tps=(n - 1) / (total - first) if n > 1 and first else None)
