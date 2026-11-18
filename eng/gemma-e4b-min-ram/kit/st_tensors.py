"""Tensors of a safetensors checkpoint as the kernel tests and the numerics tests see them (numpy only, memory-mapped).

The config's tensor table (inspect_safetensors.py) names each internal tensor, its format and its HF key(s); this reads the bytes
the generated programs read (the file's own bytes: nibble plane, scale plane; bf16 matrices) and the values the programs'
kernels compute from them (the numpy references): MXFP4 planes dequantized per the Quark / vLLM convention, bf16 widened, the
prepared f32 tensors (bf16 norms +1, A_log -> -exp, copies) as the weight policy produces them.

The MXFP4 reference below is an independent reimplementation of vLLM's mxfp4_w4a8_gemv_rdna3 decode (csrc/rocm/
mxfp4_w4a8_gemv_rdna3.cu: kE2M1x2 table, low nibble = even k, scale = 2^(e - 127)); tests/test_safetensors_source.py checks the
starter's reader against it on real rows.
"""
import json
import struct

import numpy as np  # noqa: E402  (tests that import this module skip without numpy)

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float64)
DTYPES = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U8": np.uint8, "I8": np.int8}


def e2m1_values():
    """The 16 codes: bit 3 is the sign."""
    return np.concatenate([E2M1, -E2M1])


def dequant_mxfp4(nibbles, scales):
    """nibbles u8 [rows, K/2], scales u8 [rows, K/32] -> float32 [rows, K]: element 2j = low nibble of byte j, 2j+1 = high nibble;
    value = E2M1(code) * 2^(e - 127) per 32 elements."""
    rows, half = nibbles.shape
    lut = e2m1_values()
    lo, hi = lut[nibbles & 15], lut[nibbles >> 4]
    v = np.stack([lo, hi], axis=-1).reshape(rows, half * 2)
    s = np.exp2(scales.astype(np.float64) - 127.0)
    return (v.reshape(rows, -1, 32) * s[:, :, None]).reshape(rows, half * 2).astype(np.float32)


def bf16_to_f32(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


def prepared(values_f32, op):
    """The weight policy's elementwise ops (draft/prep_weights.py ELEMENT_OPS) on f32 values."""
    return {"copy": lambda v: v, "add1": lambda v: v + np.float32(1.0), "negexp": lambda v: -np.exp(v.astype(np.float64)).astype(np.float32)}[op](values_f32)


class Checkpoint:
    """The tensors of one or more safetensors files by HF key (memory-mapped, nothing is read until used)."""

    def __init__(self, paths):
        self.t = {}
        for p in paths:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                head = json.loads(f.read(n))
            head.pop("__metadata__", None)
            for k, v in head.items():
                a, b = v["data_offsets"]
                self.t[k] = (p, 8 + n + a, b - a, v["dtype"], v["shape"])

    def raw(self, key):
        p, off, n, _, _ = self.t[key]
        return np.memmap(p, dtype=np.uint8, mode="r", offset=off, shape=(n,))

    def array(self, key):
        p, off, n, dt, shape = self.t[key]
        return np.memmap(p, dtype=DTYPES[dt], mode="r", offset=off, shape=tuple(shape))


class Tensor:
    """The attributes make_kernel_tests.py reads of a GGUF tensor: name, shape (GGUF order [K, rows]), tensor_type.name, data (the
    file bytes), plus scale (the second plane of a split format) and values (float32 [rows, K] or the 1-D prepared values)."""

    class Type:
        def __init__(self, name):
            self.name = name

        def __eq__(self, o):
            return getattr(o, "name", o) == self.name

        def __hash__(self):
            return hash(self.name)

    def __init__(self, name, entry, ck):
        self.name, self.entry = name, entry
        self.shape = tuple(entry["shape"])
        self.tensor_type = Tensor.Type({"mxfp4s": "MXFP4S", "bf16": "BF16", "f32": "F32", "f16": "F16"}[entry["type"]])
        self.ck = ck
        self.scale = ck.raw(entry["hf_scale"]) if entry.get("hf_scale") else None
        self._data = None

    @property
    def data(self):
        # a prepared f32 tensor is read through its policy: the data is the prepared f32 values (what the program's view sees)
        if self.entry.get("prep"):
            return self.values()
        return self.ck.raw(self.entry["hf"])

    def values(self):
        e, ck = self.entry, self.ck
        if e["type"] == "mxfp4s":
            rows = ck.t[e["hf"]][4][0]
            return dequant_mxfp4(np.asarray(ck.raw(e["hf"])).reshape(rows, -1), np.asarray(ck.raw(e["hf_scale"])).reshape(rows, -1))
        a = ck.array(e["hf"])
        v = bf16_to_f32(np.asarray(a)) if ck.t[e["hf"]][3] == "BF16" else np.asarray(a, np.float32)
        if e.get("prep"):
            return prepared(v, e["prep"]["op"]).reshape(-1)
        return v


def load(cfg):
    """{internal name: Tensor} of the config's checkpoint (cfg["weights"]["files"])."""
    ck = Checkpoint(cfg["weights"]["files"])
    return {n: Tensor(n, e, ck) for n, e in cfg["tensors"].items()}
