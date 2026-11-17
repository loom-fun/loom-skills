"""Image tower of a safetensors checkpoint: the cfg["vision"] block (inspect_safetensors.py writes it into llm_config.json).

The encoder plan (encoder.py) is written against the tensor names of llama.cpp's mmproj converter. A checkpoint's tower is the
same model under HF names (model.visual.*), so the per-family map (hf_names/<family>.json "vision") gives each mmproj name its HF
key and, for the small f32 tensors (biases, norms, positions), the load-time preparation (bf16 -> f32 copy: the weight policy
`vision_weights.loom`); matrices are read as stored (bf16). Standard library only (no gguf, no numpy): the checkpoint's header and
config.json are the whole input.

cfg["vision"] = {"source": "safetensors", "family", "img_begin", "img_end", "mrope": True,
                 "config": {the encoder config of encoder.config(): patch, E, NH, D, ffn, layers, eps, out, mean, std, merge, pos_n,
                            act, deepstack, fmt, patch_layout: "hf3d", out_row},
                 "tensors": {mmproj name: {"type": "BF16"|"F32", "shape": [GGUF order], "bytes": n, "hf": key, "prep": "copy"?}}}
"""
import math

GGML = {"BF16": "BF16", "F16": "F16", "F32": "F32"}


def _dig(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


def block(config, header, vmap, preprocessor=None):
    """config: the checkpoint's config.json; header: {hf key: {dtype, shape, bytes}}; vmap: hf_names/<family>.json["vision"];
    preprocessor: preprocessor_config.json (optional). Returns cfg["vision"], or exits with the reason."""
    vc = config.get(vmap["config"]) or {}
    pre = vmap["prefix"]
    k = {n: vc.get(v) for n, v in vmap["config_keys"].items()}
    for n in ("patch", "E", "NH", "ffn", "layers", "out", "merge", "pos_n", "temporal"):
        if not isinstance(k[n], int):
            raise SystemExit(f"vision: config.json {vmap['config']}.{vmap['config_keys'][n]} is missing")
    act = vmap["acts"].get(k["act"])
    if act is None:
        raise SystemExit(f"vision: activation {k['act']!r} has no encoder kernel (hf_names vision.acts)")
    if k["temporal"] != 2:
        raise SystemExit(f"vision: temporal_patch_size {k['temporal']}: the patch embedding kernel sums two taps (a still image repeats its frame)")
    side = math.isqrt(k["pos_n"])
    if side * side != k["pos_n"]:
        raise SystemExit(f"vision: {k['pos_n']} position embeddings are not a square grid")
    E, P = k["E"], k["patch"]
    deep = list(k["deepstack"] or [])
    pp = preprocessor or {}
    mean = pp.get("image_mean") or vmap["preprocess_defaults"]["image_mean"]
    std = pp.get("image_std") or vmap["preprocess_defaults"]["image_std"]
    if pp.get("patch_size", P) != P or pp.get("merge_size", k["merge"]) != k["merge"] or pp.get("temporal_patch_size", 2) != 2:
        raise SystemExit("vision: preprocessor_config.json disagrees with config.json vision_config (patch / merge / temporal size)")

    tens = {}

    def add(internal, hf, prep=None):
        t = header.get(pre + hf)
        if t is None:
            raise SystemExit(f"vision: {internal}: HF tensor {pre + hf} is not in the checkpoint")
        if t["dtype"] not in ("BF16", "F16", "F32"):
            raise SystemExit(f"vision: {pre + hf} is {t['dtype']}: the encoder reads bf16 / f16 / f32 tensors")
        shape = list(reversed(t["shape"]))
        if prep:
            if t["dtype"] != "BF16":
                raise SystemExit(f"vision: {pre + hf}: prep {prep} expects bf16, found {t['dtype']}")
            n = math.prod(t["shape"])
            tens[internal] = {"type": "F32", "shape": shape, "bytes": 4 * n, "hf": pre + hf, "prep": prep}
        else:
            tens[internal] = {"type": GGML[t["dtype"]], "shape": shape, "bytes": t["bytes"], "hf": pre + hf}

    pw = header.get(pre + vmap["patch_embed"]["weight"])
    if pw is None or pw["shape"] != [E, 3, k["temporal"], P, P]:
        raise SystemExit(f"vision: patch embedding {pre + vmap['patch_embed']['weight']} {pw and pw['shape']} != [{E}, 3, {k['temporal']}, {P}, {P}]")
    add("v.patch_embd.weight", vmap["patch_embed"]["weight"])
    tens["v.patch_embd.weight"]["shape"] = [P, P, 3, E]     # the converter's per-tap shape; the file keeps both taps (layout hf3d)
    if pre + vmap["patch_embed"]["bias"] in header:
        add("v.patch_embd.bias", vmap["patch_embed"]["bias"], "copy")
    for n, s in vmap["global"].items():
        add(n, s["hf"], s.get("prep"))
    for i in range(k["layers"]):
        for n, s in vmap["block"]["tensors"].items():
            add(f"v.blk.{i}.{n}", vmap["block"]["prefix"].format(i=i) + s["hf"], s.get("prep"))
    for j, i in enumerate(deep):
        for n, s in vmap["deepstack"]["tensors"].items():
            add(f"v.deepstack.{i}.{n}", vmap["deepstack"]["prefix"].format(k=j) + s["hf"], s.get("prep"))
    pos = tens["v.position_embd.weight"]["shape"]
    if pos[1] != k["pos_n"] or pos[0] != E:
        raise SystemExit(f"vision: position table {pos} != [{E}, {k['pos_n']}]")
    out = tens["mm.2.weight"]["shape"][1]
    if out != k["out"]:
        raise SystemExit(f"vision: projector output {out} != out_hidden_size {k['out']}")
    conf = {"patch": P, "E": E, "NH": k["NH"], "D": E // k["NH"], "ffn": k["ffn"], "layers": k["layers"], "eps": vmap["eps"],
            "out": out, "mean": mean, "std": std, "merge": k["merge"], "pos_n": side, "act": act, "deepstack": deep,
            "out_row": out * (1 + len(deep)), "fmt": {"BF16": "bf16", "F16": "f16", "F32": "f32"}[tens["v.blk.0.ffn_up.weight"]["type"]],
            "patch_layout": "hf3d"}
    return {"source": "safetensors", "family": vmap["family"], "img_begin": vmap["img_begin"], "img_end": vmap["img_end"],
            "mrope": True, "config": conf, "tensors": tens}


def prepared(vision):
    """{hf key: elements} of the tensors the weight policy converts at load (bf16 -> f32)."""
    return {t["hf"]: t["bytes"] // 4 for t in vision["tensors"].values() if t.get("prep")}
