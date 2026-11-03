"""Compare ref_vit.py's intermediate tensors with a llama-mtmd-debug trace (sums of named tensors).
usage: trace_compare.py MMPROJ LOG rainbow|gray SIZE name...       e.g. after_posemb layer_out-0 layer_out-11 projected
LOG: llama-mtmd-debug -m MODEL.gguf --mmproj MMPROJ -p encode -n SIZE --image rainbow > LOG 2>&1
ref_vit records a tensor with tr(name, x); add tr() calls to a new encoder for the names its llama.cpp graph uses (cb(...))."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_vit as R

W, KV = R.load(sys.argv[1])
fam = KV.get("clip.vision.projector_type") or KV.get("clip.projector_type")
out = R.ENCODERS[fam](W, KV, R.synth(sys.argv[3], int(sys.argv[4])))
R.TRACE["projected"] = out
sums = {}
for blk in open(sys.argv[2], errors="replace").read().split("common_debug_cb_eval:")[1:]:
    name = blk.split(" = ")[0].strip(); m = re.search(r"sum = (\S+)", blk)
    if m and name not in sums:
        sums[name] = float(m.group(1))
for n in sys.argv[5:]:
    r = R.TRACE.get(n); l = sums.get(n)
    rel = (float(r.sum()) - l) / abs(l) if r is not None and l else float("nan")
    print(f"{n:24s} llama.cpp {l}  ref {None if r is None else round(float(r.sum()), 4)}  rel {rel:+.4f}")
