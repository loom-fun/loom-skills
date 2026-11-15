#!/usr/bin/env python3
"""Host logits readback check of one package on one GPU (devlab_perf.sh step `logits:V`; loom-skills#736).

The package is opened through libloom_text_embed (the embed layer Gufo links; loom-serve never sees the capability) with
`llm.logits` on, and driven one row at a time:

  (a) argmax: with the span flagged for logits at T=0, every returned row's argmax is the token the device selected, for
      every step of every prompt (the row's padded columns hold -3e38 and are never the argmax);
  (b) llama.cpp: the row's log-softmax top-5 vs llama.cpp's own top-5 log-probs for the same greedy continuation (same
      GGUF, same runner, same prompt token ids), within LOGPROB_TOL nats on the tokens both list; a greedy divergence is
      judged by bench_common.near_tie (the golden gate's rule) and ends that prompt's comparison;
  (c) default spans unchanged: the greedy continuation of a handle with the capability OFF, of the capability-ON handle with
      the span not flagged, and with the span flagged are identical token for token.

Writes logits_V.json, a gates.json that make_info.py gates ingests (gate `logits`, detail "N/N steps ...").

usage: logits_check.py PACKAGE OUT.json --lib libloom_text_embed.so [--llama http://127.0.0.1:8130] [--steps 32]
                       [--prompts 6] [--preset P] [--memory-bytes N] [--rows 4]
The embed library comes from --lib or $LOOM_EMBED_LIB. Without one the check is "not supported" (exit 2, a JSON saying so),
never replaced by another path. The comparison logic (check_*) takes plain lists and is unit tested without a GPU
(tests/test_logits_check.py); EmbedLib is the only part that touches the library.
"""
import argparse
import ctypes
import datetime
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "skills", "loom-benchmark", "scripts"))

PAD = -1e30            # padded columns of a row are -3e38 (g_head_mv): anything below this is padding
LOGPROB_TOL = 0.30     # nats: Loom's and llama.cpp's log-prob of a token both list (quantized matmul numerics differ)
TOPK = 5
MIN_COMPARED = 8       # greedy steps compared against llama.cpp on at least one prompt before the gate can pass


# ---------------------------------------------------------------- comparisons (pure)

def argmax(row, vocab):
    """Index of the largest of the first `vocab` columns (ties: the lowest index, as the device's select does)."""
    best, bi = -math.inf, -1
    for i in range(min(vocab, len(row))):
        if row[i] > best:
            best, bi = row[i], i
    return bi


def log_softmax_top(row, vocab, k=TOPK):
    """[(id, logprob)] of the k largest real columns, logprobs over the first `vocab` columns."""
    r = row[:vocab]
    m = max(r)
    lse = m + math.log(sum(math.exp(x - m) for x in r))
    top = sorted(range(len(r)), key=lambda i: (-r[i], i))[:k]
    return [(i, r[i] - lse) for i in top]


def check_argmax(row, vocab, token):
    """(ok, why) for one step: the argmax of the returned row is the device-selected token, padding is masked."""
    if len(row) < vocab:
        return False, f"row has {len(row)} columns, vocabulary is {vocab}"
    if any(math.isnan(x) for x in row[:vocab]):
        return False, "NaN in the row"
    a = argmax(row, vocab)
    if a != token:
        return False, f"argmax {a} != device token {token}"
    if any(x > PAD for x in row[vocab:]):
        return False, "a padded column is not masked"
    return True, ""


def check_vs_llama(rows, vocab, llama_ids, llama_top, near_tie, tol=LOGPROB_TOL):
    """Compare Loom's rows (one per greedy step, from the same prompt) with llama.cpp's greedy continuation.
    llama_ids: its greedy token ids; llama_top: per step {id: logprob} (top-n). near_tie(top, gold, loom) -> bool.
    Returns (compared_steps, problems): stops at the first greedy divergence (judged as a near-tie, or a problem)."""
    compared, problems = 0, []
    for k, (row, gid, ltop) in enumerate(zip(rows, llama_ids, llama_top)):
        top = log_softmax_top(row, vocab, TOPK)
        lid = top[0][0]
        if lid != gid:
            if not near_tie(ltop, gid, lid):
                problems.append(f"step {k}: Loom top-1 {lid} vs llama.cpp {gid}, not a near-tie")
            break
        for i, lp in top:
            if i in ltop and abs(lp - ltop[i]) > tol:
                problems.append(f"step {k}: token {i} logprob {lp:.3f} vs llama.cpp {ltop[i]:.3f}")
        compared += 1
    return compared, problems


def check_default_unchanged(off, on_unflagged, on_flagged):
    """The greedy continuation is identical with the capability off, on (span unflagged) and on (span flagged)."""
    if off != on_unflagged:
        return False, f"capability on changes the unflagged output at token {first_diff(off, on_unflagged)}"
    if off != on_flagged:
        return False, f"a flagged span changes the output at token {first_diff(off, on_flagged)}"
    return True, ""


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def verdict(checks, compared_max, llama_used):
    """The gate: pass only when every check passed, and (with a llama.cpp reference) enough steps were compared."""
    bad = [c for c in checks if not c["ok"]]
    if llama_used and compared_max < MIN_COMPARED and not bad:
        bad = [{"name": "llama", "why": f"only {compared_max} steps compared (need {MIN_COMPARED})"}]
    return not bad, bad


# ---------------------------------------------------------------- the embed library (ctypes)

class EmbedLib:
    """libloom_text_embed.so at ABI 2 (loom-gufo-extension loom-embed): the calls this check needs.
    ABI 2 (open_ex, logits_info, row_logits); a span is flagged by the request key "logits": true (eng/logitsdesign/DESIGN.md section 5)."""

    class Options(ctypes.Structure):   # loom_text_embed_options_t (ABI 1)
        _fields_ = [("model_path", ctypes.c_char_p), ("preset", ctypes.c_char_p), ("device_uri", ctypes.c_char_p),
                    ("memory_bytes", ctypes.c_uint64), ("rows", ctypes.c_uint32), ("checkpoints", ctypes.c_uint32)]

    class OptionsEx(ctypes.Structure):   # loom_text_embed_options_ex_t (ABI 2)
        pass

    class LogitsInfo(ctypes.Structure):   # loom_text_embed_logits_info_t
        _fields_ = [("supported", ctypes.c_uint32), ("enabled", ctypes.c_uint32), ("width", ctypes.c_uint32), ("max_rows", ctypes.c_uint32)]

    def __init__(self, path):
        self.L = L = ctypes.CDLL(path)
        c, vp = ctypes, ctypes.c_void_p
        self.OptionsEx._fields_ = [("struct_size", c.c_uint32), ("base", self.Options), ("logits", c.c_uint32)]
        L.loom_text_embed_open_ex.argtypes = [c.POINTER(self.OptionsEx), c.POINTER(vp), c.c_char_p, c.c_size_t]
        L.loom_text_embed_logits_info.argtypes = [vp, c.POINTER(self.LogitsInfo)]
        L.loom_text_embed_error.restype = c.c_char_p
        L.loom_text_embed_error.argtypes = [vp]
        L.loom_text_embed_close.argtypes = [vp]
        L.loom_text_embed_vocabulary_size.argtypes = [vp]
        L.loom_text_embed_vocabulary_size.restype = c.c_int32
        L.loom_text_embed_row_begin.argtypes = [vp, c.c_uint32, vp, c.c_uint64, c.c_int]
        L.loom_text_embed_row_prefill.argtypes = [vp, c.c_uint32, c.POINTER(c.c_int32), c.c_size_t]
        L.loom_text_embed_row_decode.argtypes = [vp, c.c_uint32]
        L.loom_text_embed_row_token.argtypes = [vp, c.c_uint32]
        L.loom_text_embed_row_token.restype = c.c_int32
        L.loom_text_embed_row_is_eos.argtypes = [vp, c.c_uint32]
        L.loom_text_embed_row_end.argtypes = [vp, c.c_uint32]
        L.loom_text_embed_chat_prepare.argtypes = [vp, c.c_char_p, c.c_size_t, c.POINTER(c.c_int32), c.c_size_t,
                                                   c.POINTER(c.c_size_t), c.POINTER(vp)]
        L.loom_text_embed_request_free.argtypes = [vp]
        L.loom_text_embed_row_is_logit.argtypes = [vp, c.c_uint32]
        L.loom_text_embed_row_logits.argtypes = [vp, c.c_uint32, c.POINTER(c.c_float), c.c_size_t, c.POINTER(c.c_size_t)]
        self.h = None

    def open(self, package, logits, preset=None, rows=4, memory_bytes=0):
        c = ctypes
        base = self.Options(os.path.join(package, "model.json").encode(), preset.encode() if preset else None, None, memory_bytes, rows, 0)
        o = self.OptionsEx(c.sizeof(self.OptionsEx), base, 1 if logits else 0)
        err, h = c.create_string_buffer(512), c.c_void_p()
        if self.L.loom_text_embed_open_ex(c.byref(o), c.byref(h), err, len(err)):
            raise RuntimeError(f"embed open (logits={int(logits)}): {err.value.decode(errors='replace')}")
        self.h = h
        self.vocab = self.L.loom_text_embed_vocabulary_size(h)
        info = self.LogitsInfo()
        self.L.loom_text_embed_logits_info(h, c.byref(info))
        self.width, self.supported, self.enabled = info.width, bool(info.supported), bool(info.enabled)
        if logits and not (self.supported and self.enabled):
            raise RuntimeError(f"embed open: logits capability not available (supported={self.supported} enabled={self.enabled})")
        return self

    def close(self):
        if self.h:
            self.L.loom_text_embed_close(self.h)
            self.h = None

    def _ok(self, rc, what):
        if rc:
            raise RuntimeError(f"{what}: {(self.L.loom_text_embed_error(self.h) or b'').decode(errors='replace')}")

    def begin(self, row, user_text, max_tokens, flag):
        """Chat-prepare a greedy request, reserve the row, flag the span: -> the prompt's token ids."""
        c = ctypes
        body = json.dumps({"messages": [{"role": "user", "content": user_text}], "temperature": 0, "max_tokens": max_tokens,
                           **({"logits": True} if flag else {})}).encode()   # the request key "logits": true flags the span
        n, req = c.c_size_t(), c.c_void_p()
        self._ok(self.L.loom_text_embed_chat_prepare(self.h, body, len(body), None, 0, c.byref(n), c.byref(req)), "chat_prepare")
        self.L.loom_text_embed_request_free(req)   # the counting call's request
        ids = (c.c_int32 * n.value)()
        req = c.c_void_p()
        self._ok(self.L.loom_text_embed_chat_prepare(self.h, body, len(body), ids, n.value, c.byref(n), c.byref(req)), "chat_prepare")
        try:
            rc = self.L.loom_text_embed_row_begin(self.h, row, req, n.value + max_tokens + 1, 1)
            if rc == 1:
                raise RuntimeError("row_begin: no capacity")
            self._ok(rc, "row_begin")
        finally:
            self.L.loom_text_embed_request_free(req)
        if bool(self.L.loom_text_embed_row_is_logit(self.h, row)) != bool(flag):
            raise RuntimeError(f"row_is_logit disagrees with the request key (flag={flag})")
        return list(ids), ids, n.value

    def prefill(self, row, ids, n):
        self._ok(self.L.loom_text_embed_row_prefill(self.h, row, ids, n), "row_prefill")

    def token(self, row):
        return self.L.loom_text_embed_row_token(self.h, row)

    def eos(self, row):
        return bool(self.L.loom_text_embed_row_is_eos(self.h, row))

    def decode(self, row):
        self._ok(self.L.loom_text_embed_row_decode(self.h, row), "row_decode")

    def end(self, row):
        self.L.loom_text_embed_row_end(self.h, row)

    def row_logits(self, row):
        """The full logits row of the row's pending selection: `width` floats (ids the head does not have are <= -1e38)."""
        c = ctypes
        buf, n = (c.c_float * self.width)(), c.c_size_t()
        self._ok(self.L.loom_text_embed_row_logits(self.h, row, buf, self.width, c.byref(n)), "row_logits")
        return list(buf)[:n.value]


def generate(emb, row, user_text, steps, flag, rows_out=None):
    """Greedy continuation of one prompt on `row`: -> (prompt ids, generated ids). With rows_out a list, the logits row of
    every selection is appended (flag on)."""
    ids, buf, n = emb.begin(row, user_text, steps, flag)
    try:
        emb.prefill(row, buf, n)
        gen = []
        while len(gen) < steps:
            t = emb.token(row)
            if rows_out is not None:
                rows_out.append(emb.row_logits(row))
            gen.append(t)
            if emb.eos(row) or len(gen) == steps:
                break
            emb.decode(row)
        return ids, gen
    finally:
        emb.end(row)


# ---------------------------------------------------------------- the run

def run(a):
    lib = a.lib or os.environ.get("LOOM_EMBED_LIB")
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    out = {"verdict": "FAIL", "level": "dev", "ts": now, "gates": {}, "checks": []}
    if not lib or not os.path.isfile(lib):
        out["verdict"] = "NOT_SUPPORTED"
        out["note"] = f"no embed library ({lib or 'LOOM_EMBED_LIB unset'}): host logits cannot be checked on this runner"
        json.dump(out, open(a.out, "w"), indent=1)
        print("LOGITS NOT_SUPPORTED", out["note"])
        return 2
    from bench_common import PROMPTS, Package, llama_greedy, near_tie
    pkg = Package(a.package)
    emb = EmbedLib(lib)
    prompts = PROMPTS[:a.prompts]
    checks, compared_max, stats = [], 0, {"steps": 0, "prompts": 0}
    # (c) reference continuations with the capability off
    emb.open(a.package, logits=False, preset=a.preset, rows=a.rows, memory_bytes=a.memory_bytes)
    off = {}
    for p in prompts:
        off[p] = generate(emb, 0, p, a.steps, flag=False)
    emb.close()
    emb.open(a.package, logits=True, preset=a.preset, rows=a.rows, memory_bytes=a.memory_bytes)
    vocab = emb.vocab
    try:
        for p in prompts:
            ids0, g_off = off[p]
            ids1, g_un = generate(emb, 0, p, a.steps, flag=False)
            rows = []
            ids2, g_fl = generate(emb, 0, p, a.steps, flag=True, rows_out=rows)
            ok, why = check_default_unchanged(g_off, g_un, g_fl)
            checks.append({"name": "default_unchanged", "prompt": p[:40], "ok": ok, "why": why})
            checks.append({"name": "prompt_ids", "prompt": p[:40], "ok": ids0 == ids2 == pkg.ids(pkg.render(p)),
                           "why": "the embed layer's prompt tokens differ from the package's chat program"})
            bad = [(k, check_argmax(r, vocab, t)[1]) for k, (r, t) in enumerate(zip(rows, g_fl)) if not check_argmax(r, vocab, t)[0]]
            checks.append({"name": "argmax", "prompt": p[:40], "ok": not bad and len(rows) == len(g_fl),
                           "why": f"step {bad[0][0]}: {bad[0][1]}" if bad else ("rows missing" if len(rows) != len(g_fl) else "")})
            stats["steps"] += len(rows)
            stats["prompts"] += 1
            if a.llama:
                r, lids, cp = llama_greedy(a.llama, ids2, a.steps, pkg)
                ltop = [{t["id"]: t["logprob"] for t in x["top_logprobs"]} for x in cp]
                n, problems = check_vs_llama(rows, vocab, lids[:len(ltop)], ltop, near_tie)
                compared_max = max(compared_max, n)
                checks.append({"name": "llama", "prompt": p[:40], "ok": not problems, "compared": n,
                               "why": "; ".join(problems[:3])})
    finally:
        emb.close()
    ok, bad = verdict(checks, compared_max, bool(a.llama))
    n_ok = sum(1 for c in checks if c["ok"])
    detail = (f"{stats['steps']} steps on {stats['prompts']} prompts, argmax = device token"
              + (f", top-{TOPK} within {LOGPROB_TOL} of llama.cpp over {compared_max}+ steps" if a.llama else "")
              + ", greedy text unchanged")
    out.update({"verdict": "PASS" if ok else "FAIL", "checks": checks, "vocab": vocab, "stats": stats,
                "gates": {"logits": {"pass": ok, "detail": detail if ok else "; ".join(f"{b.get('name')}: {b.get('why')}" for b in bad)[:200]}}})
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"LOGITS {out['verdict']} {n_ok}/{len(checks)} checks; {detail}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("package")
    ap.add_argument("out")
    ap.add_argument("--lib")
    ap.add_argument("--llama", help="llama.cpp server URL (same GGUF, same runner): enables check (b)")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--preset")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--memory-bytes", type=int, default=0)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
