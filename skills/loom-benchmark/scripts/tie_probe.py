"""Prepare the paired divergence input from the two native text_match.py captures."""
import json
from pathlib import Path
import sys
from bench_common import Package, doc, llama_greedy, thinking_extra

mode, folder, pkgdir, *args = sys.argv[1:]
out = Path(folder)
pkg = Package(pkgdir)
if mode == "prepare":
    from ratchet_prompts import PROMPTS as prompts
    captures = [json.loads((out/f"text_{v}.json").read_text()) for v in ("fun","staging")]
    assert captures[0]["thinking"] == captures[1]["thinking"]
    cases=[]
    for i,prompt in enumerate(prompts):
        texts=[c["texts"][i] for c in captures]
        if texts[0] == texts[1]:
            continue
        mode_th = captures[0]["thinking"]
        messages = [{"role":"user","content":prompt}]
        rendered=pkg.render_messages(messages, enable_thinking={"off":False,"on":True,"template":None}[mode_th])
        ids = [pkg.ids(pkg.raw_from_chat({"content":t}, False, mode_th=="on", rendered)) for t in texts]
        pos=next(j for j,(x,y) in enumerate(zip(*ids)) if x!=y)
        candidates=[t[pos] for t in ids]
        body={"model":pkg.model_id,"messages":messages,"temperature":0,"max_tokens":58,"logits":True,**thinking_extra(mode_th)}
        (out/f"tie_body_{i}.json").write_text(json.dumps(body))
        (out/f"tie_prefix_{i}.txt").write_text(" ".join(map(str,ids[0][:pos])))
        cases.append(dict(prompt=i,position=pos,candidates=candidates,pieces=[pkg.tok.decode([x],skip_special_tokens=False) for x in candidates],common=ids[0][:pos]))
    (out/"tie_cases.json").write_text(json.dumps(cases))
elif mode == "reference":
    for c in json.loads((out/"tie_cases.json").read_text()):
        i=c["prompt"]
        a,b=[json.loads((out/f"tie_{v}_{i}.json").read_text()) for v in ("fun","staging")]
        assert a["prompt_ids"]==b["prompt_ids"]
        _,ids,cp=llama_greedy(args[0],a["prompt_ids"]+c["common"],1,pkg,n_probs=10)
        assert len(ids)==len(cp)==1
        (out/f"tie_reference_{i}.json").write_text(json.dumps({"choice":ids[0],"top":cp[0]["top_logprobs"]}))
