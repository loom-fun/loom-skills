"""The image tower's tensors from a safetensors checkpoint, in the form ref_vit.py takes (numpy; development and tests only).

load(paths, vision) -> (W, KV): W[mmproj name] = float32 array shaped like the GGUF loader's (the HF shape: GGUF dims reversed);
the conv3d patch embedding is split into the converter's two temporal taps (v.patch_embd.weight / .1); the f32 tensors are the
prepared copies the weight policy writes (bf16 -> f32); matrices are the bf16 values as stored (widened). KV: the clip.* metadata
ref_vit reads, from cfg["vision"]["config"]. `vision` is cfg["vision"] (inspect_safetensors.py)."""
import json, struct
import numpy as np


def _index(paths):
    t = {}
    for p in paths:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            t[k] = (p, 8 + n + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0], v["dtype"], v["shape"])
    return t


def read(index, key):
    p, off, n, dt, shape = index[key]
    raw = np.fromfile(p, np.uint8, n, offset=off)
    if dt == "BF16":
        a = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif dt == "F16":
        a = raw.view(np.float16).astype(np.float32)
    else:
        a = raw.view(np.float32)
    return a.reshape(shape)


def load(paths, vision):
    ix = _index(paths); c = vision["config"]; W = {}
    for name, t in vision["tensors"].items():
        a = read(ix, t["hf"])
        if name == "v.patch_embd.weight":   # [E, 3, 2, P, P]: tap 0 and tap 1
            W[name], W[name + ".1"] = a[:, :, 0].copy(), a[:, :, 1].copy()
        else:
            W[name] = a
    KV = {"clip.vision.patch_size": c["patch"], "clip.vision.embedding_length": c["E"], "clip.vision.attention.head_count": c["NH"],
          "clip.vision.attention.layer_norm_epsilon": c["eps"], "clip.vision.image_mean": c["mean"], "clip.vision.image_std": c["std"],
          "clip.vision.spatial_merge_size": c["merge"], "clip.use_silu": c["act"] == "silu"}
    return W, KV


def raw_weights(paths, vision):
    """{mmproj name: int8 array}: the bytes the encoder program's `command.parameter` views see: the file's bytes, or for a prepared
    tensor (policy vision_weights.loom) the converted f32 values."""
    ix = _index(paths); out = {}
    for name, t in vision["tensors"].items():
        p, off, n, dt, shape = ix[t["hf"]]
        if t.get("prep"):
            out[name] = np.ascontiguousarray(read(ix, t["hf"]).astype(np.float32)).reshape(-1).view(np.int8)
        else:
            out[name] = np.fromfile(p, np.uint8, n, offset=off).view(np.int8)
    return out
