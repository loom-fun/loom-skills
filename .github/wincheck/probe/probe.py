"""Serve one package dir, record registration facts, greedy outputs and (optionally) perf. usage: probe.py PKG_DIR OUT.json [--perf]"""
import json, os, subprocess, sys, time, urllib.request
pkg, outp = sys.argv[1], sys.argv[2]; perf = "--perf" in sys.argv
URL = "http://127.0.0.1:8080"
def http(p, body=None, t=600):
    r = urllib.request.Request(URL + p, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=t).read())
t0 = time.time()
log = open(outp + ".serve.log", "w")
srv = subprocess.Popen(["loom-serve", f"--model={pkg}/model.json", "--device=amdgpu://?completion_spin_ns=4294967295", "--memory_bytes=17000000000"], stdout=log, stderr=subprocess.STDOUT)
R = {"pkg": pkg}
st = None
while time.time() - t0 < 600:
    try:
        st = http("/v1/status", t=10)["data"][0]["loom"]
        if st.get("registration") in ("ready", "failed", "prepared") or http("/v1/models", t=10)["data"]:
            break
    except Exception:
        pass
    time.sleep(1)
R["ready_s"] = round(time.time() - t0, 1)
try:
    st = http("/v1/status", t=10)["data"][0]["loom"]
except Exception as e:
    st = {"error": str(e)}
comp = st.get("compilation") or {}
R["registration"] = st.get("registration"); R["error"] = (st.get("error") or {}).get("message")
R["preparation_s"] = round((st.get("preparation_duration_ns") or 0) / 1e9, 2)
R["compilation"] = {k: comp.get(k) for k in ("kernels", "reused_kernels", "entries", "errors", "stage_duration_sum_ns")}
R["diagnostics"] = sorted({d.get("message", "")[:300] for d in comp.get("diagnostics") or []})[:10]
models = []
try: models = http("/v1/models", t=10)["data"]
except Exception: pass
if models:
    mid = models[0]["id"]
    doc = open(os.path.join(os.path.dirname(__file__), "..", "..", "..", "skills", "loom-benchmark", "data", "doc_32000.txt")).read()[:7000]
    prompts = ["What is the capital of France? Answer in one word.", "Write a haiku about memory bandwidth.",
               "Summarize this document in three sentences:\n\n" + doc]
    R["outputs"] = []
    for p in prompts:
        r = http("/v1/chat/completions", {"model": mid, "messages": [{"role": "user", "content": p}], "max_tokens": 96, "temperature": 0,
                                          "chat_template_kwargs": {"enable_thinking": False}})
        R["outputs"].append({"text": r["choices"][0]["message"].get("content"), "usage": r.get("usage")})
    if perf:
        sd = os.path.join(os.path.dirname(__file__), "..", "..", "..", "skills", "loom-benchmark", "scripts")
        p = subprocess.run([sys.executable, os.path.join(sd, "perf.py"), URL, "probe", "--model", mid, "--no-campaign", "--repeats", "5", "--log", outp + ".perf.jsonl"],
                           cwd=sd, capture_output=True, text=True, timeout=3600)
        js = [l for l in p.stdout.splitlines() if l.startswith('{"')]
        if js:
            o = json.loads(js[-1]); R["perf"] = {"decode": o.get("decode_tps"), "decode_2k": o.get("decode_tps_2k"), "prefill": o.get("prefill")}
        else:
            R["perf_error"] = p.stdout[-1500:] + p.stderr[-1500:]
srv.terminate(); srv.wait(30)
json.dump(R, open(outp, "w"), indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "outputs"})); [print(repr((o["text"] or "")[:160])) for o in R.get("outputs", [])]
