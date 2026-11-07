"""Shared helpers for the Loom benchmark scripts.

A *package* is a directory with model.json + tokenizer.json + llm_config.json
(written by loom-port-llm/starter/build.py). llm_config.json's "chat" block is
the single source of truth for prompt rendering, so llama.cpp (golden, fed raw
token ids) and Loom (fed chat messages) see byte-identical prompts.
"""
import glob
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
DRAFT_COUNTERS = ("draft_proposed_tokens", "draft_accepted_inputs")


def draft_counters(text):
    """loom-serve's cumulative speculative decoding counters {draft_proposed_tokens, draft_accepted_inputs} from its
    /metrics text ({} when it has none: no drafting, or llama-server)."""
    vals = {}
    for line in (text or "").splitlines():
        for k in DRAFT_COUNTERS:
            if f'field="{k}"' in line:
                vals[k] = float(line.split()[-1])
    return vals


def server_draft_counters(url):
    """draft_counters() of a running server's /metrics, or {} when it serves none."""
    try:
        r = requests.get(url + "/metrics", timeout=30)
        return draft_counters(r.text) if r.status_code == 200 else {}
    except requests.RequestException:
        return {}


def acceptance_between(before, after):
    """Accepted / proposed drafted tokens between two draft_counters() snapshots, or None (nothing proposed)."""
    prop = after.get("draft_proposed_tokens", 0) - before.get("draft_proposed_tokens", 0)
    acc = after.get("draft_accepted_inputs", 0) - before.get("draft_accepted_inputs", 0)
    return round(acc / prop, 3) if prop > 0 else None


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


def no_dummy_prefix(tok):
    """The tokenizer without a SentencePiece-style dummy prefix (Metaspace, prepend_scheme "first": a "▁" before
    the first word of every encoded string). The gates encode a reply on its own; with the prefix its first word
    becomes "▁The" where the model, continuing its prompt, generated "The" (a token-0 mismatch on every reply that
    opens with a word). A rendered prompt opens with a special token, where "first" adds nothing, so prompts encode
    as the server encodes them. Other tokenizers are returned unchanged."""
    from tokenizers import Tokenizer
    spec = json.loads(tok.to_str())
    pres = [spec.get("pre_tokenizer") or {}]
    pres += pres[0].get("pretokenizers", []) if pres[0].get("type") == "Sequence" else []
    changed = False
    for p in pres:
        if p.get("type") == "Metaspace" and p.get("prepend_scheme") == "first":
            p["prepend_scheme"], changed = "never", True
    return Tokenizer.from_str(json.dumps(spec)) if changed else tok


class Package:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.path = path
        self.cfg = json.load(open(os.path.join(path, "llm_config.json")))
        self.chat = self.cfg["chat"]
        self.model_id = self.cfg["model_id"]
        self.tok = no_dummy_prefix(Tokenizer.from_file(os.path.join(path, "tokenizer.json")))

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
    The key names the llama.cpp build by $LLAMACPP_DIR; $LLAMA_CACHE_BUILD (e.g. "b1355") replaces that path
    where it differs between machines (nightly CI restores the cache on whichever runner it lands on).
    On a miss with $LLAMA_DEFER set (gates.sh: one model at a time), the request is appended to that
    file and a placeholder without tokens is returned; llama_fill.py answers the pending requests with
    only llama.cpp loaded, and the gate re-runs from the cache. Otherwise, when nothing answers at
    `url` and $LLAMA_START is set, that command is run once to start llama-server.
    """
    gguf = pkg.cfg.get("gguf") if pkg is not None else os.environ.get("LLAMA_GGUF")
    if gguf not in _FP:
        _FP[gguf] = _gguf_fingerprint(gguf)
    key = hashlib.sha256(json.dumps({"gguf": _FP[gguf], "build": os.environ.get("LLAMA_CACHE_BUILD") or os.environ.get("LLAMACPP_DIR", ""),
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
    mb = body.get("_micro_batch")   # llama_views: the prompt in micro-batches of at most mb tokens
    start = body.get("_score_from")   # llama_scores: top lists at every position from there
    for attempt in range(2):
        if mb:
            r, ok = _micro_batched(url, {k: v for k, v in body.items() if k != "_micro_batch"}, mb, timeout), True
            break
        if start is not None:
            r, ok = _scored(url, {k: v for k, v in body.items() if k != "_score_from"}, start, timeout), True
            break
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


def _micro_batched(url, req, mb, timeout):
    """Answer /completion `req` with its prompt evaluated in micro-batches of at most mb tokens, each request
    extending the slot's cache (the first starts afresh). A server that rolls back to a context checkpoint
    before extending (SWA layers) re-evaluates a fixed few tokens more per request, so the step shrinks by
    that and the prompt starts over; when no step of 2 or more keeps every batch within mb (recurrent state: the
    whole prefix each time; at most len/2 requests), the answer is an empty view."""
    p, step = req["prompt"], mb
    while step >= 2:
        m = 0
        while m < len(p):
            m2 = min(m + step if m else mb, len(p))
            last = {} if m2 == len(p) else {"n_probs": 0}
            r = requests.post(url + "/completion", json=dict(req, prompt=p[:m2], cache_prompt=m > 0, **last), timeout=timeout).json()
            e = (r.get("timings") or {}).get("prompt_n", 0)
            if e > mb:
                break
            m = m2
        else:
            return r
        step -= e - mb
    return {"completion_probabilities": [], "note": f"no micro-batch view: llama.cpp re-evaluates more than {mb} tokens per request"}


def _scored(url, req, start, timeout):
    """llama.cpp's top lists for the token after each prefix req["prompt"][:L], L = start .. len - 1: the first
    request evaluates prompt[:start] afresh, each next one extends the slot's cache by one token (decode)."""
    p, out = req["prompt"], []
    for L in range(start, len(p)):
        r = requests.post(url + "/completion", json=dict(req, prompt=p[:L], cache_prompt=L > start), timeout=timeout).json()
        if not r.get("completion_probabilities"):
            return {"completion_probabilities": [], "note": f"no top list at position {L}: {str(r)[:200]}"}
        out.append(r["completion_probabilities"][0])
    return {"completion_probabilities": out}


# Every llama.cpp reference is greedy with the samplers forced off: some GGUFs set penalty_repeat
# (LFM2.5: 1.05), which llama.cpp applies even at temperature 0, while Loom decodes greedily.
GREEDY = {"temperature": 0, "cache_prompt": False, "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}


def llama_greedy(url, ids, n_predict, pkg, salt="", n_probs=5, **extra):
    """Greedy llama.cpp continuation of prompt ids -> (response, generated ids, aligned top lists).
    completion_probabilities omits byte-fallback tokens (emoji, partial UTF-8), so the ids come from
    `tokens` and the top-n lists are trusted only while they line up with them. `extra`: more request fields."""
    r = llama_completion(url, dict(GREEDY, prompt=ids, n_predict=n_predict, n_probs=n_probs, return_tokens=True, **extra), pkg, salt)
    cp = r.get("completion_probabilities", [])
    gen = r.get("tokens") or [x["id"] for x in cp]
    k = next((i for i, x in enumerate(cp) if i >= len(gen) or x["id"] != gen[i]), len(cp))
    return r, gen, cp[:k]


def llama_views(url, ids, pkg):
    """llama.cpp's own top-10 log-probs for the token after `ids` under two numerics it uses itself: `ids` in one
    batch (its prefill kernels: MMQ matmuls) and in micro-batches of 8 through the slot's cache (its decode
    kernels: MMVQ, another q8_1 dot-product formula and summation order). A view llama.cpp could not compute
    (pending under LLAMA_DEFER, or no cache reuse: recurrent state) is left out."""
    views = []
    for extra in ({}, {"_micro_batch": 8, "id_slot": 0}):
        _, _, cp = llama_greedy(url, list(ids), 1, pkg, n_probs=10, **extra)
        if cp:
            views.append(top_logprobs(cp, 0))
    return views


def llama_scores(url, ids, cont, pkg):
    """llama.cpp's top-10 at each token of `cont` (Loom's output) after the prompt `ids`, teacher-forced on
    Loom's own tokens -> [{id: logprob}] per position, or [] while pending (LLAMA_DEFER) or unavailable."""
    if not cont:
        return []
    r = llama_completion(url, dict(GREEDY, prompt=list(ids) + list(cont), n_predict=1, n_probs=10, id_slot=0,
                                   _score_from=len(ids)), pkg)
    cp = r.get("completion_probabilities") or []
    return [{t["id"]: t["logprob"] for t in x["top_logprobs"]} for x in cp] if len(cp) == len(cont) else []


def trajectory_gaps(tops, cont):
    """Per token of Loom's output: llama.cpp's top-1 log-prob minus that of Loom's token (a lower bound,
    from the last listed log-prob, where llama.cpp does not list it). ~0 where Loom picks llama.cpp's choice."""
    return [max(t.values()) - t.get(c, min(t.values())) for t, c in zip(tops, cont)]


NOISE_CAP = 0.5   # nats: the most llama.cpp's own spread may widen the bar


def exact_gaps(tops, gold_id, loom_id):
    """logp(gold) - logp(loom) in each of llama.cpp's views of one position that lists both tokens."""
    return [v[gold_id] - v[loom_id] for v in tops if gold_id in v and loom_id in v]


def gap_noise(tops, gold_id, loom_id):
    """llama.cpp's own noise at one position: how far the exact gap moves across its views, capped at NOISE_CAP."""
    exact = exact_gaps(tops, gold_id, loom_id)
    return min(max(exact) - min(exact), NOISE_CAP) if exact else 0.0


def near_tie(top, gold_id, loom_id, max_gap=1.0, views=()):
    """Is Loom's token at a divergence a near-tie of llama.cpp's greedy token? top: {id: logprob} from
    llama.cpp at that position. Yes when the log-prob gap is <= max_gap, or when llama.cpp itself is unsure
    there: Loom picked its second choice and its first has probability < 0.5 (flint: p 0.38 vs 0.13, gap
    1.03, while llama.cpp's own re-recording of the same prompt sat at gap 1.05). A token llama.cpp is
    confident against, or ranks lower, is a bug.
    With `views` (llama_views: llama.cpp's own other numerics there): yes when the smallest gap among top and
    the views is <= max_gap plus their spread (max - min), capped at NOISE_CAP. Only views listing both
    tokens count: one without Loom's token never widens the bar. One without gold that lists Loom's token is
    llama.cpp itself ranking Loom's token higher, which passes (gemma-4-26B: `_` at p 1.0 token by token,
    unlisted batched, where `/` is top-1)."""
    if gold_id not in top:
        return False
    tops = (top, *views)
    exact = exact_gaps(tops, gold_id, loom_id)
    if any(loom_id in v and gold_id not in v for v in tops) or exact and min(exact) <= max_gap + gap_noise(tops, gold_id, loom_id):
        return True
    ranked = sorted(top, key=top.get, reverse=True)
    return ranked[:2] == [gold_id, loom_id] and math.exp(top[gold_id]) < 0.5


def near_tie_at(url, ids, top, gold_id, loom_id, pkg, max_gap=1.0):
    """near_tie for the token after `ids` (the prompt + llama.cpp's own tokens before the divergence); when the
    plain rule rejects it and a llama.cpp URL is given, judged again with llama.cpp's own views there (llama_views)."""
    return near_tie(top, gold_id, loom_id, max_gap) or bool(url) and near_tie(top, gold_id, loom_id, max_gap, llama_views(url, ids, pkg))


def text_divergence(tok, loom_ids, gids):
    """First divergence of Loom's output ids from llama.cpp's greedy ids `gids`, judged on text: (k, loom_id,
    loom_text) with k the first golden position whose token leaves Loom's text, loom_id Loom's token there and
    loom_text Loom's text from there; None when both texts agree as far as both go. Loom's ids are its parsed
    reply re-tokenized, and the model may pick a split the tokenizer would not ('.\\n' '\\n' '\\n' where the
    tokenizer merges '.\\n\\n\\n'): the same text split differently is no divergence. From the first differing
    id the rest is compared as text; a golden token that ends inside a UTF-8 character is judged with the
    tokens that complete it. When the texts part inside Loom's first differing token (Loom ' Monday', llama.cpp
    ' Mon' ','), the divergence is that token's position. Without a split difference this is the first differing id."""
    k = next((i for i, (x, y) in enumerate(zip(loom_ids, gids)) if x != y), None)
    if k is None:
        return None
    text = tok.decode(loom_ids[k:], skip_special_tokens=False)
    first = len(tok.decode(loom_ids[k:k + 1], skip_special_tokens=False))
    at, done = k, 0
    for j in range(k, len(gids)):
        s = tok.decode(gids[k:j + 1], skip_special_tokens=False)
        if s.endswith("�"):
            continue
        if text.startswith(s):
            at, done = j + 1, len(s)
        elif s.startswith(text):
            return None
        elif done < first:   # the texts part inside Loom's token (' Monday' vs ' Mon' ','): Loom chose it at k
            return k, loom_ids[k], text
        else:
            loom_id = (tok.encode(text[done:], add_special_tokens=False).ids or [None])[0]
            return at, loom_id, text[done:]
    return None


def loom_runs_key(pkg, model, prompts):
    """Identity of a set of Loom runs a gate keeps across gates.sh's deferred-reference rounds (consistency.py): the
    model id, the prompts, the package as gates.sh's pkg_hash sees it (model.json, llm_config.json, program/*.loom)
    and the installed Loom SDK (machine_info.hrx_loom_identity: version and build)."""
    from machine_info import hrx_loom_identity
    files = [os.path.join(pkg, "model.json"), os.path.join(pkg, "llm_config.json")] + sorted(glob.glob(os.path.join(pkg, "program", "*.loom")))
    h = hashlib.sha256(json.dumps([model, prompts, hrx_loom_identity()], sort_keys=True).encode())
    for f in files:
        h.update(f.encode())
        if os.path.isfile(f):
            with open(f, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def kept_runs(state, key):
    """The runs kept in STATE (a JSON file) when they were made under KEY (loom_runs_key), else None."""
    try:
        kept = json.load(open(state)) if state and os.path.exists(state) else {}
    except ValueError:
        return None
    return kept if kept.get("key") == key else None


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


def thinking_extra(mode):
    """Request fields that select thinking identically on Loom and llama.cpp: on / off send
    chat_template_kwargs.enable_thinking; template sends nothing (each engine's own default, and they differ:
    llama-server turns thinking on, a Loom package follows the chat template, off for e.g. Nemotron-3.5-Bolt)."""
    return {} if mode == "template" else {"chat_template_kwargs": {"enable_thinking": mode == "on"}}


def stream(url, model, messages, max_tokens, extra=None, timeout=None, on_token=None):
    """Streams one chat completion (greedy unless `extra` says otherwise). Returns client-side TTFT,
    decode tok/s, usage, the text (reasoning + content), the finish reason and the HTTP status;
    tokens: usage.completion_tokens, or the streamed token chunks when the server sends no usage block;
    on_token(t) is called with time.perf_counter() at every streamed token."""
    t0 = time.perf_counter()
    first = finish = None
    n = pt = chunks = 0
    usage = False
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
                usage = True
                n = d["usage"]["completion_tokens"]
                pt = d["usage"]["prompt_tokens"]
            for ch in d.get("choices", []):
                dl = ch.get("delta", {})
                piece = (dl.get("reasoning_content") or "") + (dl.get("content") or "")
                if piece:
                    chunks += 1
                    now = time.perf_counter()
                    first = now - t0 if first is None else first
                    if on_token:
                        on_token(now)
                    text += piece
                finish = ch.get("finish_reason") or finish
    total = time.perf_counter() - t0
    n = n if usage else chunks
    return dict(status=200, ttft=first, total=total, tokens=n, prompt_tokens=pt, text=text, finish=finish,
                decode_tps=(n - 1) / (total - first) if n > 1 and first else None)
