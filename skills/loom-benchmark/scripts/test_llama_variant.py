"""Unit test of the llama-server variants (bench_common.SERVER_VARIANTS, llama_fill.py): python test_llama_variant.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as bc

os.environ["LLAMA_CACHE"] = tempfile.mkdtemp(); os.environ["LLAMA_GGUF"] = os.devnull
bc.ensure_llama = lambda url: None
posted = []
class Resp:
    def json(self): return {"completion_probabilities": [{"id": 1}], "tokens": [1]}
bc.requests.post = lambda url, json=None, timeout=None: posted.append(json) or Resp()
body = {"prompt": "p", "_server": "cpu-encoder"}
ok = True
def check(name, cond):
    global ok
    ok &= bool(cond); print(("ok  " if cond else "FAIL"), name)

r = bc.llama_completion("u", body)   # a running server cannot be switched: left unanswered, never cached, nothing sent
check("a variant request on a running server is pending, not sent", r.get("_pending") and not posted)
os.environ["LLAMA_FILL_VARIANT"] = "cpu-encoder"   # llama_fill.py started the server as that variant
r = bc.llama_completion("u", body)
check("the variant server answers; `_server` is not sent to it", r.get("tokens") == [1] and posted == [{"prompt": "p"}])
del os.environ["LLAMA_FILL_VARIANT"]
check("the answer is cached under the variant's key", bc.llama_completion("u", body).get("tokens") == [1])
posted.clear(); r = bc.llama_completion("u", {"prompt": "p"})
check("the default request is a different key", r.get("tokens") == [1] and len(posted) == 1)

import vision_golden as vg
class Pkg:
    chat = {}; cfg = {"gguf": ""}
    render_messages = staticmethod(lambda m, enable_thinking: m[0]["content"])
seen = []
bc.llama_completion = lambda url, body, pkg, salt: seen.append(body) or {}
vg.reference(Pkg, "u", "data:image/png;base64,AA", "q", 8, server="cpu-encoder"); vg.reference(Pkg, "u", "data:image/png;base64,AA", "q", 8)
check("vision_golden.reference: `_server` only when asked", seen[0].get("_server") == "cpu-encoder" and "_server" not in seen[1])

# the vision gate's prefill views (vision_golden.prefill_view / judge): Thu/Thursday as on the Ornith-1.5-122B chart
class Tok:
    names = {1: "Thu", 2: "Thursday", 3: "x"}
    decode = lambda self, ids, skip_special_tokens=True: "".join(self.names[i] for i in ids)
class VPkg(Pkg):
    tok = Tok(); raw_from_chat = staticmethod(lambda m, f, t: "Thursday"); ids = lambda self, t: [2]
    def __init__(self): pass
p = VPkg()
def view(top, evaluated):
    return lambda url, body, pkg, salt="": seen.append(body) or {"completion_probabilities": [{"top_logprobs": [{"id": i, "logprob": v} for i, v in top.items()]}],
                                                              "tokens_evaluated": evaluated}
bc.llama_completion = view({1: -0.65, 2: -0.74}, 103)
v = vg.prefill_view(p, "u", "data:image/png;base64,AA", "q", [3, 3, 3], 100)
check("prefill_view: the prefix text and the image go in one prompt; its top-10 comes back", v == {1: -0.65, 2: -0.74} and seen[-1]["prompt"]["prompt_string"].endswith("xxx") and seen[-1]["n_predict"] == 1)
check("prefill_view: tokens evaluated other than prompt + prefix (a re-tokenized prefix): no view", vg.prefill_view(p, "u", "data:image/png;base64,AA", "q", [3, 3], 100) is None)
bc.llama_completion = lambda *a, **k: {"_pending": True, "completion_probabilities": []}
check("prefill_view: pending: no view", vg.prefill_view(p, "u", "data:image/png;base64,AA", "q", [3], 100) is None)
ref = {"tokens_evaluated": 5, "tokens": [3, 1], "completion_probabilities": [{"id": 3, "top_logprobs": [{"id": 3, "logprob": 0.0}]},
       {"id": 1, "top_logprobs": [{"id": 1, "logprob": -0.06}, {"id": 2, "logprob": -2.8}]}]}
got = {"message": {}, "finish": "stop", "prompt_tokens": 5}
import bench_common
bench_common.text_divergence = lambda tok, ids, gold: (1, 2, "Thursday")
check("judge: a 2.8 nat gap in the decode view fails without views", vg.judge(p, ref, got)[0] == "FAIL")
check("judge: ... and passes when llama.cpp's prefill view has them 0.1 apart", vg.judge(p, ref, got, lambda pre: [{1: -0.65, 2: -0.74}])[0] == "PASS")
check("judge: ... but not when its prefill views agree with the decode view (a bug is wide in every view)", vg.judge(p, ref, got, lambda pre: [{1: -0.1, 2: -3.0}])[0] == "FAIL")
check("judge: the views are asked for only after a plain FAIL", vg.judge(p, dict(ref, completion_probabilities=ref["completion_probabilities"][:1] + [{"id": 1, "top_logprobs": [{"id": 1, "logprob": -0.3}, {"id": 2, "logprob": -0.5}]}]), got, lambda pre: 1 / 0)[0] == "PASS")
sys.exit(0 if ok else 1)
