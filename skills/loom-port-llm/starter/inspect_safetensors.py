"""Reads a Hugging Face safetensors checkpoint and writes the LLM_CONFIG json that gen_llm.py, build.py and the benchmark scripts
consume: the same shape inspect_gguf.py writes (internal GGUF-style tensor names, formats, shapes; the "chat" block from the
chat template), plus the HF key of every tensor and a "source" block. Standard library only for the reading; the chat block
needs chat_probe.py (jinja2).

    python inspect_safetensors.py SOURCE OUT.json [--revision REV] [--checkpoint PATH]

SOURCE: a local directory (config.json, tokenizer files, *.safetensors), an HF repo id (HTTP range reads of the headers; small
files are downloaded; no weights), or a header dump written by `dump` (fleet/st_stub.py uses it). --checkpoint: the local
safetensors file(s) the built package will link (cfg["weights"]); without it the HF cache is searched.

Per family the HF -> internal name map is data (hf_names/<family>.json: "a new family is configuration"); the producer's weight
convention (hf_names/quark.json: U8 nibbles + U8 E8M0 `weight_scale` -> format mxfp4s, excluded layers stay bf16) is separate.
"""
import argparse
import base64
import gzip
import json
import math
import os
import struct
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = os.path.join(HERE, "hf_names")
HF = "https://huggingface.co"


# ---------------------------------------------------------------------------------------------------------------- sources
class Source:
    """What the config needs of a checkpoint: header (name -> {dtype, shape}), file sizes, config.json, generation_config.json,
    tokenizer_config.json, the chat template text, the tokenizer's added tokens, and where it came from."""

    def __init__(self, header, config, generation_config, tokenizer_config, chat_template, added_tokens, origin, files, preprocessor=None):
        self.header, self.config, self.generation_config = header, config, generation_config or {}
        self.preprocessor = preprocessor or {}   # preprocessor_config.json (image mean / std, patch and merge size), when the repo has one
        self.tokenizer_config, self.chat_template, self.added_tokens = tokenizer_config or {}, chat_template or "", added_tokens
        self.origin, self.files = origin, files   # files: {name: size}

    def dump(self):
        return {"format": "safetensors-header-dump", "header": self.header, "files": self.files, "config": self.config,
                **({"preprocessor_config": self.preprocessor} if self.preprocessor else {}),
                "generation_config": self.generation_config, "tokenizer_config": self.tokenizer_config,
                "chat_template": self.chat_template, "added_tokens": self.added_tokens, "origin": self.origin}

    @classmethod
    def from_dump(cls, d):
        return cls(d["header"], d["config"], d["generation_config"], d["tokenizer_config"], d["chat_template"], d["added_tokens"],
                   d["origin"], d["files"], d.get("preprocessor_config"))


def parse_header(read):
    """read(offset, length) -> bytes; returns ({name: {dtype, shape}}, file size is the caller's)."""
    n = struct.unpack("<Q", read(0, 8))[0]
    raw = json.loads(read(8, n))
    raw.pop("__metadata__", None)
    return {k: {"dtype": v["dtype"], "shape": v["shape"], "bytes": v["data_offsets"][1] - v["data_offsets"][0]} for k, v in raw.items()}


def _added(tok):
    return [{"content": t["content"], "id": t["id"], "special": bool(t.get("special"))} for t in tok.get("added_tokens", [])]


def _http(url, rng=None):
    req = urllib.request.Request(url, headers={"User-Agent": "loom-skills-inspect"})
    if os.environ.get("HF_TOKEN"):
        req.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[0] + rng[1] - 1}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read(), r.headers


def load_local(path):
    files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    header, sizes = {}, {}
    for f in files:
        fp = os.path.join(path, f)
        with open(fp, "rb") as fh:
            header.update(parse_header(lambda o, n: (fh.seek(o), fh.read(n))[1]))
        sizes[f] = os.path.getsize(fp)
    rd = lambda f: json.load(open(os.path.join(path, f))) if os.path.exists(os.path.join(path, f)) else None
    ct = os.path.join(path, "chat_template.jinja")
    tok = rd("tokenizer.json") or {}
    return Source(header, rd("config.json"), rd("generation_config.json"), rd("tokenizer_config.json"),
                  open(ct).read() if os.path.exists(ct) else None, _added(tok), {"kind": "safetensors", "path": os.path.abspath(path)}, sizes,
                  rd("preprocessor_config.json"))


def load_hf(repo, revision="main"):
    info = json.loads(_http(f"{HF}/api/models/{repo}/revision/{revision}")[0])
    sha, names = info["sha"], [s["rfilename"] for s in info["siblings"]]
    get = lambda f: _http(f"{HF}/{repo}/resolve/{sha}/{f}")[0]
    getj = lambda f: json.loads(get(f)) if f in names else None
    header, sizes = {}, {}
    for f in sorted(n for n in names if n.endswith(".safetensors")):
        url = f"{HF}/{repo}/resolve/{sha}/{f}"
        read = lambda o, n, url=url: _http(url, (o, n))[0]
        header.update(parse_header(read))
        sizes[f] = int(_http(url, (0, 1))[1]["Content-Range"].rpartition("/")[2])
    tok = getj("tokenizer.json") or {}
    return Source(header, getj("config.json"), getj("generation_config.json"), getj("tokenizer_config.json"),
                  get("chat_template.jinja").decode() if "chat_template.jinja" in names else None, _added(tok),
                  {"kind": "safetensors", "repo": repo, "revision": sha}, sizes, getj("preprocessor_config.json"))


def load_source(spec, revision="main"):
    if os.path.isdir(spec):
        return load_local(spec)
    if os.path.isfile(spec):
        op = gzip.open if spec.endswith(".gz") else open
        return Source.from_dump(json.load(op(spec, "rt")))
    return load_hf(spec, revision)


# ---------------------------------------------------------------------------------------------------------------- the maps
def family_for(config):
    archs = set(config.get("architectures") or [])
    for f in sorted(os.listdir(NAMES)):
        if f.endswith(".json") and f != "quark.json":
            m = json.load(open(os.path.join(NAMES, f)))
            if archs & set(m.get("architectures", [])):
                return m
    sys.exit(f"no name map in hf_names/ for architectures {sorted(archs)}: add hf_names/<family>.json (see hf_names/qwen35.json)")


def producer_for(config):
    qc = config.get("quantization_config") or {}
    if not qc:
        return None
    for f in sorted(os.listdir(NAMES)):
        if f.endswith(".json") and f != "qwen35.json":
            p = json.load(open(os.path.join(NAMES, f)))
            if p.get("producer") and p.get("quant_method") == qc.get("quant_method"):
                return p
    sys.exit(f"quantization_config.quant_method {qc.get('quant_method')!r}: no producer convention in hf_names/")


def dig(d, path, default=None):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def tensor_entries(src, fam, prod, layer_types, n_mtp):
    """internal name -> config tensor entry for the family's map, resolved against the header (the producer's convention
    decides each weight's format)."""
    H, out, used = src.header, {}, set()
    excluded = set((src.config.get("quantization_config") or {}).get(prod["exclude_key"] if prod else "exclude", []))

    def entry(internal, spec, hf):
        t = H.get(hf)
        if t is None:
            sys.exit(f"{internal}: HF tensor {hf} is not in the checkpoint")
        used.add(hf)
        shape = list(reversed(t["shape"]))        # GGUF order: [K (fastest), rows]
        if spec.get("prep"):
            if t["dtype"] != "BF16":
                sys.exit(f"{hf}: prep {spec['prep']} expects bf16, found {t['dtype']}")
            shp = [t["shape"][-1], t["shape"][0]] if spec.get("shape") else list(t["shape"])   # conv1d [C, 1, K] -> internal [K, C]
            e = {"type": "f32", "shape": shp, "hf": hf, "prep": {"src": "bf16", "op": spec["prep"]}}
            out[internal] = e
            return
        stem = hf[:-len(".weight")] if hf.endswith(".weight") else None
        sc = f"{stem}.weight{prod['scale_suffix']}" if prod and stem else None
        if sc in H:
            if t["dtype"] != prod["weight_dtype"] or H[sc]["dtype"] != prod["scale_dtype"]:
                sys.exit(f"{hf}: expected {prod['weight_dtype']} nibbles + {prod['scale_dtype']} scales, found {t['dtype']} / {H[sc]['dtype']}")
            rows, half = t["shape"]
            k = half * 2
            gs = prod["weight"]["group_size"]
            if H[sc]["shape"] != [rows, k // gs]:
                sys.exit(f"{hf}: scale shape {H[sc]['shape']} != [{rows}, {k // gs}]")
            if stem in excluded:
                sys.exit(f"{hf}: quantized in the file but listed in the producer's exclude list")
            used.add(sc)
            out[internal] = {"type": prod["format"], "shape": [k, rows], "hf": hf, "hf_scale": sc}
            return
        if t["dtype"] == "BF16":
            out[internal] = {"type": "bf16", "shape": shape, "hf": hf}
        elif t["dtype"] in ("F32",):
            out[internal] = {"type": "f32", "shape": shape, "hf": hf}
        elif t["dtype"] == "F16":
            out[internal] = {"type": "f16", "shape": shape, "hf": hf}
        else:
            sys.exit(f"{hf}: dtype {t['dtype']} without a scale tensor: no format for it (add a producer convention)")

    for n, spec in fam["global"].items():
        if n == "output.weight" and spec["hf"] not in H:
            continue                                # tied embeddings
        entry(n, spec, spec["hf"])
    lp = fam["layer"]["prefix"]
    for i, kind in enumerate(layer_types):
        for part in ("common", kind):
            for n, spec in fam["layer"][part].items():
                entry(f"blk.{i}.{n}", spec, lp.format(i=i) + spec["hf"])
    layers = len(layer_types)
    assert n_mtp <= 1, "one MTP block: the map's global MTP tensors (fc, pre_fc norms, head norm) are per model"
    for j in range(n_mtp):
        bi = layers + j
        bp = fam["mtp"]["block"].format(j=j)
        for n, spec in fam["layer"]["common"].items():
            entry(f"blk.{bi}.{n}", spec, bp + spec["hf"])
        for n, spec in fam["layer"]["full_attention"].items():
            entry(f"blk.{bi}.{n}", spec, bp + spec["hf"])
        for n, spec in fam["mtp"]["global"].items():
            entry(f"blk.{bi}.{n}", spec, spec["hf"])
    return out, used


# ---------------------------------------------------------------------------------------------------------------- the config
def rope_freq_factors(rp, rope_dims):
    """llama3 rope scaling -> one factor per rotary pair (llama.cpp's converter writes these as rope_freqs.weight; a pair's frequency is
    divided by its factor): 1 for high frequencies, `factor` for low ones, smoothly interpolated between."""
    if not rp or rp.get("rope_type") != "llama3":
        return None
    import math
    th, f = float(rp["rope_theta"]), float(rp["factor"])
    lo, hi, old = float(rp["low_freq_factor"]), float(rp["high_freq_factor"]), float(rp["original_max_position_embeddings"])
    out = []
    for j in range(rope_dims // 2):
        wavelen = 2 * math.pi * th ** (2 * j / rope_dims)
        if wavelen < old / hi:
            out.append(1.0)
        elif wavelen > old / lo:
            out.append(f)
        else:
            sm = (old / wavelen - lo) / (hi - lo)
            out.append(1.0 / ((1 - sm) / f + sm))
    return out


def build_config(src, checkpoint=None, drop_chat=False, drop_vision=False):
    cfg_hf = src.config
    fam, prod = family_for(cfg_hf), producer_for(cfg_hf)
    tc = cfg_hf.get(fam["text_config"]) if fam.get("text_config") else cfg_hf
    c = {k: dig(tc, v) for k, v in fam["config"].items()}
    layers = c["layers"]
    if fam.get("layer_types"):   # hybrid families name each layer's kind; a plain transformer has none
        layer_types = [("linear_attention" if t == fam["recurrent_type"] else "full_attention") for t in tc[fam["layer_types"]]]
    else:
        layer_types = ["full_attention"] * layers
    assert len(layer_types) == layers, (len(layer_types), layers)
    n_mtp = int(dig(tc, fam["mtp_layers"], 0) or 0) if fam.get("mtp") and any(k.startswith("mtp.") for k in src.header) else 0
    tensors, used = tensor_entries(src, fam, prod, layer_types, n_mtp)
    name = (src.origin.get("repo") or os.path.basename(src.origin.get("path", "model"))).rpartition("/")[2]
    tokens = {t["content"]: t["id"] for t in src.added_tokens}
    eos = tokens.get(src.tokenizer_config.get("eos_token"))
    if isinstance(src.tokenizer_config.get("eos_token"), dict):
        eos = tokens.get(src.tokenizer_config["eos_token"].get("content"))
    gen_eos = src.generation_config.get("eos_token_id", [])
    gen_eos = [gen_eos] if isinstance(gen_eos, int) else list(gen_eos)
    eos = eos if eos is not None else (gen_eos[0] if gen_eos else tc.get("eos_token_id"))
    im_end, endoftext = tokens.get("<|im_end|>"), tokens.get("<|endoftext|>")
    rd = fam["rope_dims"]
    vocab = tensors["token_embd.weight"]["shape"][1]
    cfg = {
        "name": name, "arch": fam["arch"], "gguf": None,
        "hidden": c["hidden"], "layers": layers, "heads": c["heads"], "kv_heads": c["kv_heads"], "head_dim": c["head_dim"],
        "ffn": c["ffn"], "vocab": vocab, "context_length": c["context_length"], "rms_eps": c["rms_eps"],
        "rope_theta": float(c["rope_theta"]), "rope_type": fam["fixed"]["rope_type"], "qk_norm": fam["fixed"]["qk_norm"],
        "attn_gate": fam["fixed"]["attn_gate"], "post_norms": fam["fixed"]["post_norms"], "embed_norm": fam["fixed"]["embed_norm"],
        "embed_scale": fam["fixed"]["embed_scale"], "sliding_window": fam["fixed"]["sliding_window"],
        "sliding_layers": fam["fixed"]["sliding_layers"], "nope_layers": fam["fixed"]["nope_layers"],
        "tied_embeddings": "output.weight" not in tensors, "eos_id": eos,
        "extra_eos_ids": sorted({i for i in (im_end, endoftext, *gen_eos) if i is not None and i != eos}),
        "bos_id": tokens.get(src.tokenizer_config.get("bos_token")) if src.tokenizer_config.get("add_bos_token") else None,
        "chat": None,
        "rope_dims": int(dig(tc, rd["head_dim"]) * (dig(tc, rd["factor"]) if rd.get("factor") else 1.0)),
    }
    if fam.get("gdn"):
        cfg["recurrent_layers"] = [i for i, k in enumerate(layer_types) if k == "linear_attention"]
        g = {k: dig(tc, v) for k, v in fam["gdn"].items() if k != "v_order"}
        g["inner"] = g["v_heads"] * g["head_v"]
        g["v_order"] = fam["gdn"]["v_order"]
        cfg["gdn"] = g
        cfg["kv_layers"] = layers - len(cfg["recurrent_layers"])
    ff = rope_freq_factors(dig(tc, "rope_parameters") or dig(tc, "rope_scaling"), cfg["rope_dims"])
    if ff:
        cfg["rope_freq_factors"] = ff
    cfg["ffn_norm_tensor"] = fam["fixed"]["ffn_norm_tensor"]
    if n_mtp:
        cfg["mtp"] = {"layers": [layers + j for j in range(n_mtp)]}
    cfg["attn_sinks"] = fam["fixed"]["attn_sinks"]
    # mrope (image positions) belongs to the image component (cfg["vision"] below); without it text positions are sequential on all
    # three axes, which plain neox RoPE over the rotary dims computes.
    cfg["tensors"] = tensors
    cfg["formats"] = sorted({t["type"] for t in tensors.values()})   # the weight formats in the file (the knowledge base routes quant nodes on them)
    if prod:   # the producer's quantization_config as the checkpoint states it (loom-kb facet/quark routes on this key)
        cfg[prod["producer"]] = cfg_hf["quantization_config"]
    files = sorted(src.files)
    cfg["source"] = {"kind": "safetensors", **{k: v for k, v in src.origin.items() if k in ("repo", "revision")}, "files": files,
                     "family": fam["family"], "producer": prod["producer"] if prod else None, "bytes": sum(src.files.values())}
    if checkpoint:
        cfg["weights"] = {"format": "safetensors", "files": [os.path.abspath(p) for p in checkpoint]}
    unmapped = sorted(k for k in src.header if k not in used and not k.startswith("model.visual.") and not k.startswith("mtp."))
    if unmapped:
        print("NOTE: checkpoint tensors the name map does not use:", unmapped[:6], f"({len(unmapped)} total)")
    vis = sorted(k for k in src.header if k.startswith(fam.get("vision", {}).get("prefix", "model.visual.")))
    if vis and fam.get("vision") and not drop_vision and cfg_hf.get(fam["vision"]["config"]):
        sys.path.insert(0, os.path.join(HERE, "vision"))
        import st_source
        cfg["vision"] = st_source.block(cfg_hf, src.header, fam["vision"], src.preprocessor)
        mr = {k: dig(tc, v) for k, v in fam["mrope"].items() if k != "pad_to"}   # image positions: (t, h, w) rotary sections
        cfg["mrope"] = {"sections": list(mr["sections"]) + [0] * (fam["mrope"].get("pad_to", 4) - len(mr["sections"])), "interleaved": bool(mr["interleaved"])}
        vprep = sum(1 for t in cfg["vision"]["tensors"].values() if t.get("prep"))
        print(f"vision: {fam['vision']['family']} tower {cfg['vision']['config']['layers']} x {cfg['vision']['config']['E']}, {len(cfg['vision']['tensors'])} tensors "
              f"({vprep} prepared f32 copies, matrices as stored), M-RoPE sections {cfg['mrope']['sections']}")
    elif vis:
        print(f"NOTE: {len(vis)} vision tensors (model.visual.*): no image component (the family's map has no vision block, config.json has no vision_config, or drop_vision)")
    if src.chat_template and not drop_chat:
        import chat_probe
        bos = src.tokenizer_config.get("bos_token") or ""
        bos = bos.get("content", "") if isinstance(bos, dict) else bos
        chat, notes = chat_probe.derive(src.chat_template, bos)
        for n in notes:
            print("chat note:", n)
        for nm, ok, detail in chat_probe.verify(src.chat_template, chat, bos):
            print("chat", "OK  " if ok else "SKIP" if ok is None else "FAIL", nm, "-", detail)
        cfg["chat"] = chat
        if cfg["bos_id"] is None and chat.get("bos") and chat["bos"] == bos:   # HF tokenizers often add bos in tokenizer.json, not add_bos_token
            cfg["bos_id"] = tokens.get(bos)
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source")
    ap.add_argument("out", nargs="?", help="the llm_config.json to write (or -o)")
    ap.add_argument("-o", "--output", help="the llm_config.json to write")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--checkpoint", action="append", default=[], help="local safetensors file of the weights (repeat per shard)")
    a = ap.parse_args()
    src = load_source(a.source, a.revision)
    cfg = build_config(src, a.checkpoint)
    fmts = sorted({t["type"] for t in cfg["tensors"].values()})
    print(f"{cfg['name']}: arch={cfg['arch']} L={cfg['layers']} H={cfg['hidden']} heads={cfg['heads']}/{cfg['kv_heads']} d={cfg['head_dim']} "
          f"ffn={cfg['ffn']} vocab={cfg['vocab']} formats={fmts} source={cfg['source']['producer']}")
    out = a.output or a.out
    if not out:
        sys.exit("usage: inspect_safetensors.py SOURCE OUT.json (or -o OUT.json)")
    json.dump(cfg, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
