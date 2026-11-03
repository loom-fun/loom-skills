"""Adds image input to a port: picks the repo's vision projector (mmproj GGUF), downloads it to the HF cache,
and writes cfg["vision"] for build.py. Exit 0 with VISION_RESULT {...}; exit 3 when the repo has a projector
whose family has no encoder plan yet (the text port is unaffected: follow reference/vision.md to add it).

usage: add_vision.py CFG.json ORG/REPO [MMPROJ_FILE]      (a local path for MMPROJ_FILE skips the download)
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Prompt markers llama.cpp's mtmd puts around a family's image embeddings (tools/mtmd/mtmd.cpp img_beg/img_end).
MARKERS = {
    "gemma4v": ("<|image>", "<image|>"),
    "qwen3vl_merger": ("<|vision_start|>", "<|vision_end|>"),
    "muse-glimmer": ("<|image_start|>", "<|image_end|>"),
}
# Kits whose chat program renders whole conversations: image requests use the generic per-message renderer;
# the generation prompt and the output parser's first state come from (generic state -> kit state map) or
# from the kit's own chat_end ("format").
END_STATES = {"gemma4": [3, 1], "atem": "format"}
# Families whose text model uses multi-axis RoPE for image tokens (llama.cpp MTMD_POS_TYPE_MROPE).
MROPE_FAMILIES = ("qwen3vl_merger",)
PREFERENCE = ("F16", "BF16", "F32", "Q8_0")


def pick(files):
    mm = [f for f in files if re.search(r"mmproj", f, re.I) and f.endswith(".gguf")]
    for p in PREFERENCE:      # whole token: "F16" must not match "BF16"
        for f in mm:
            if re.search(rf"(^|[-_.]){p}([-_.]|$)", os.path.basename(f), re.I):
                return f
    return mm[0] if mm else None


def main():
    cfg_path, repo = sys.argv[1], sys.argv[2]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    given = sys.argv[3] if len(sys.argv) > 3 else None
    if given and os.path.exists(given):
        path = given
    else:
        import urllib.request
        files = [x["rfilename"] for x in json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=30)).get("siblings", [])]
        name = given or pick(files)
        if not name:
            print("VISION_RESULT " + json.dumps({"vision": False, "reason": "no mmproj in the repo"}))
            return 0
        hf = os.environ.get("HF", "hf")   # the SDK venv's hf CLI (port.sh passes it)
        out = subprocess.run([hf, "download", repo, name], capture_output=True, text=True)
        path = (out.stdout.strip().splitlines() or [""])[-1].replace("path=", "")
        if out.returncode or not os.path.exists(path):
            sys.exit(f"STOP: downloading {repo}/{name} failed: {out.stderr[-500:]}")
    import encoder
    try:
        vc = encoder.config(path)
    except SystemExit as e:   # an unsupported family: the encoder plan stops with the reason
        print(e); print("VISION_RESULT " + json.dumps({"vision": False, "mmproj": path, "reason": "unsupported projector family"}))
        return 3
    fam = vc["family"]
    beg, end = MARKERS[fam]
    block = {"mmproj": path, "family": fam, "img_begin": beg, "img_end": end}
    if fam in MROPE_FAMILIES:   # the text model rotates image tokens by (t, h, w): rope.dimension_sections
        import gguf
        r = gguf.GGUFReader(cfg["gguf"]); arch = cfg.get("arch", "")
        f = r.fields.get(f"{arch}.rope.dimension_sections")
        if f is None:
            sys.exit(f"STOP: {fam} images need M-RoPE but {arch}.rope.dimension_sections is missing from the GGUF")
        sections = [int(f.parts[i][0]) for i in f.data]
        cfg["mrope"] = {"sections": sections, "interleaved": not arch.startswith("qwen2")}
        block["mrope"] = True
    fmt = (cfg.get("chat") or {}).get("call_format")
    if fmt in END_STATES:
        block["end_states"] = END_STATES[fmt]
    cfg["vision"] = block
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("VISION_RESULT " + json.dumps({"vision": True, "family": fam, "mmproj": path, "encoder": f"{vc['layers']}x{vc['E']}", "out": vc["out"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
