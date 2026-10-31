"""llama.cpp's own intermediate tensors for a prompt, per graph node: the reference for a new architecture.

usage: llama_trace.py LLAMACPP_DIR MODEL.gguf "prompt text" [--out trace.json] [--grep REGEX]
  -> trace.json {node: {"op", "inputs", "shape", "sum", "rows": [[first 3 ... last 3] per token]}}
     and a one-line-per-node summary on stdout (filter with --grep, e.g. '-0$|-1$' for layers 0-1).
  The summary is also the architecture's compute graph (op + input names + shapes, in execution
  order): `--grep '-0$'` shows one layer's dataflow, which is the spec a new port must implement.

Runs llama-eval-callback (ships with the lemonade llama.cpp builds; copied to a temp dir if it
lacks the exec bit) with the GPU build, greedy, 1 token. Node names follow llama.cpp's graph
builder (src/models/<arch>.cpp): attn_norm-N, Qcur-N, Kcur-N, Vcur-N, kqv_out-N, ffn_inp-N,
ffn_out-N, l_out-N, result_norm, result_output, plus arch-specific ones (per_layer_*, ffn_moe_*).

Use it when a hand-written port is coherent-but-wrong or garbage and the kernel tests pass: write
(or reuse) a numpy forward of the first layers on the same token ids, and compare node by node
(sum and edge values): the first node that differs names the misunderstood op (rope pairing, a
missing norm, a scale, a gate). It replaces guessing from text output. The prompt is tokenized by
llama.cpp (with its BOS rule): take the ids from the trace's 'inp_embd' shape and the
'tokens' line, or pass plain words so both tokenizers agree.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ap = argparse.ArgumentParser()
ap.add_argument("llamacpp_dir")
ap.add_argument("gguf")
ap.add_argument("prompt")
ap.add_argument("--out", default="trace.json")
ap.add_argument("--grep", default=None)
a = ap.parse_args()
d = os.path.abspath(a.llamacpp_dir)
exe = os.path.join(d, "llama-eval-callback")
if not os.path.exists(exe):
    sys.exit(f"{exe} not found (any lemonade llama.cpp build has it)")
tmp = None
if not os.access(exe, os.X_OK):
    tmp = tempfile.mkdtemp()
    shutil.copy(exe, tmp)
    exe = os.path.join(tmp, "llama-eval-callback")
    os.chmod(exe, 0o755)
env = dict(os.environ, LD_LIBRARY_PATH=d + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
r = subprocess.run([exe, "-m", os.path.abspath(a.gguf), "-p", a.prompt, "-n", "1", "-ngl", "99", "--temp", "0"],
                   cwd=d, env=env, capture_output=True, text=True, errors="replace")
text = r.stdout + r.stderr
if tmp:
    shutil.rmtree(tmp, ignore_errors=True)
head = re.compile(r"common_debug_cb_eval:\s+(\S+) = \((\w+)\)\s+(\w+)\((.*)\) = \{([\d, ]+)\}")
nodes, cur = {}, None
for line in text.splitlines():
    m = head.search(line)
    if m:
        name, dtype, op, inputs, shape = m.groups()
        cur = {"op": op, "dtype": dtype, "inputs": re.findall(r"([\w.\-]+)\{", inputs),
               "shape": [int(x) for x in shape.split(",")], "rows": []}
        nodes[name] = cur
        continue
    if cur is None:
        continue
    if line.strip().startswith("sum ="):
        cur["sum"] = float(line.split("=")[1])
        cur = None
    elif line.strip().startswith("[") and re.search(r"[-\d]", line):
        vals = re.findall(r"-?\d+\.\d+(?:e[-+]?\d+)?|-?inf|nan", line)
        if vals:
            cur["rows"].append([float(v) for v in vals])
if not nodes:
    sys.exit("no tensors parsed; llama-eval-callback output:\n" + text[-2000:])
json.dump(nodes, open(a.out, "w"))
pat = re.compile(a.grep) if a.grep else None
for n, v in nodes.items():
    if pat is None or pat.search(n):
        print(f"{n:24} {v['op']:10} {'(' + ', '.join(v['inputs']) + ')':52} {str(v['shape']):20} "
              f"sum={v.get('sum', float('nan')):.6g} row0={v['rows'][0] if v['rows'] else []}")
print(f"{len(nodes)} nodes -> {a.out}", file=sys.stderr)
