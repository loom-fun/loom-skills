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
    "qwen2.5vl_merger": ("<|vision_start|>", "<|vision_end|>"),
    "muse-glimmer": ("<|image_start|>", "<|image_end|>"),
}
# Kits whose chat program renders whole conversations: image requests use the generic per-message renderer;
# the generation prompt and the output parser's first state come from (generic state -> kit state map) or
# from the kit's own chat_end ("format").
END_STATES = {"gemma4": [3, 1], "atem": "format"}
# Families whose text model uses multi-axis RoPE for image tokens (llama.cpp MTMD_POS_TYPE_MROPE).
MROPE_FAMILIES = ("qwen3vl_merger", "qwen2.5vl_merger")
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
    # Gemma 4 other than E2B/E4B decodes image tokens non-causally on its sliding layers (llama.cpp
    # mtmd_decode_use_non_causal: a gemma4v projector whose width is not 1536 or 2560, i.e. the 12B/26B/31B text
    # models). The attention kernels are causal only, so those ports stay text-only until that mask exists.
    if cfg.get("arch") == "gemma4" and cfg.get("hidden") not in (1536, 2560):
        print("VISION_RESULT " + json.dumps({"vision": False, "reason": "gemma4 images need bidirectional attention "
                                             "over image tokens on sliding layers (not implemented): text-only port"}))
        return 0
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
        # The path is the last line ("path=..." or bare, by hf version); hints may follow on either stream.
        found = [w.replace("path=", "") for w in (out.stdout + "\n" + out.stderr).split()
                 if w.replace("path=", "").endswith(os.path.basename(name))]
        path = next((p for p in reversed(found) if os.path.exists(p)), "")
        if out.returncode or not path:
            sys.exit(f"STOP: downloading {repo}/{name} failed (exit {out.returncode}): "
                     f"stdout {out.stdout[-300:]!r} stderr {out.stderr[-300:]!r}")
    import encoder
    try:
        vc = encoder.config(path)
    except SystemExit as e:   # an unsupported family: the encoder plan stops with the reason
        print(e); print("VISION_RESULT " + json.dumps({"vision": False, "mmproj": path, "reason": "unsupported projector family"}))
        return 3
    fam = vc["family"]
    beg, end = MARKERS[fam]
    block = {"mmproj": path, "family": fam, "img_begin": beg, "img_end": end}
    if fam in MROPE_FAMILIES or vc["deepstack"]:   # text-side facts from the model's GGUF
        import gguf
        r = gguf.GGUFReader(cfg["gguf"]); arch = cfg.get("arch", "")
    if fam in MROPE_FAMILIES:   # the text model rotates image tokens by (t, h, w): rope.dimension_sections
        f = r.fields.get(f"{arch}.rope.dimension_sections")
        if f is None:
            sys.exit(f"STOP: {fam} images need M-RoPE but {arch}.rope.dimension_sections is missing from the GGUF")
        sections = [int(f.parts[i][0]) for i in f.data]
        cfg["mrope"] = {"sections": sections, "interleaved": not arch.startswith("qwen2")}
        block["mrope"] = True
    if vc["deepstack"]:   # DeepStack: the text model adds feature slice k+1 after layer k (llama.cpp qwen3vl.cpp)
        f = r.fields.get(f"{arch}.n_deepstack_layers")
        n = int(f.parts[f.data[0]][0]) if f is not None else 0
        if n != len(vc["deepstack"]) or vc["out"] != cfg["hidden"]:
            sys.exit(f"STOP: the projector has {len(vc['deepstack'])} deepstack layers of width {vc['out']}, "
                     f"the text model {arch}.n_deepstack_layers = {n} and hidden {cfg['hidden']}")
        block["deepstack"] = n
    fmt = (cfg.get("chat") or {}).get("call_format")
    if fmt in END_STATES:
        block["end_states"] = END_STATES[fmt]
    cfg["vision"] = block
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("VISION_RESULT " + json.dumps({"vision": True, "family": fam, "mmproj": path, "encoder": f"{vc['layers']}x{vc['E']}", "out": vc["out"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
