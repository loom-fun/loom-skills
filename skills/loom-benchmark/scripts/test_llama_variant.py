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
sys.exit(0 if ok else 1)
