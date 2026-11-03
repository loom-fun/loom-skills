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
PREFERENCE = ("F16", "BF16", "F32", "Q8_0")


def pick(files):
    mm = [f for f in files if re.search(r"mmproj", f, re.I) and f.endswith(".gguf")]
    for p in PREFERENCE:
        for f in mm:
            if p.lower() in f.lower():
                return f
    return mm[0] if mm else None


def main():
    cfg_path, repo = sys.argv[1], sys.argv[2]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    given = sys.argv[3] if len(sys.argv) > 3 else None
    if given and os.path.exists(given):
        path = given
    else:
        from huggingface_hub import HfApi, hf_hub_download
        name = given or pick([s.rfilename for s in HfApi().model_info(repo).siblings])
        if not name:
            print("VISION_RESULT " + json.dumps({"vision": False, "reason": "no mmproj in the repo"}))
            return 0
        path = hf_hub_download(repo, name)
    import encoder
    try:
        vc = encoder.config(path)
    except SystemExit as e:   # an unsupported family: the encoder plan stops with the reason
        print(e); print("VISION_RESULT " + json.dumps({"vision": False, "mmproj": path, "reason": "unsupported projector family"}))
        return 3
    fam = vc["family"]
    if fam == "qwen3vl_merger" and not cfg.get("rope_sections"):
        print("NOTE: Qwen-VL images need M-RoPE positions in the text model; cfg has no rope_sections")
    beg, end = MARKERS[fam]
    block = {"mmproj": path, "family": fam, "img_begin": beg, "img_end": end}
    fmt = (cfg.get("chat") or {}).get("call_format")
    if fmt in END_STATES:
        block["end_states"] = END_STATES[fmt]
    cfg["vision"] = block
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("VISION_RESULT " + json.dumps({"vision": True, "family": fam, "mmproj": path, "encoder": f"{vc['layers']}x{vc['E']}", "out": vc["out"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
