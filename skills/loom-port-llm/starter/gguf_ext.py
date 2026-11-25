"""GGML tensor types the gguf Python package (pinned in the tools venv) does not know yet, registered at import time so
GGUFReader can read a file that holds them and gguf.quants.dequantize gives the exact reference values (kernel tests).

- Q2_0 (42, upstream ggml): 64 values per block, f16 d + 16 bytes of 2-bit codes; value = (q - 1) d.
- PQ2_0 (142, PrismML fork): the same codes at 128 values per block (34 bytes).
- PTQ1_0 (143, PrismML fork, Ternary Bonsai): 128 trits per 28-byte block: qs[24] (5 trits per byte, two stages of 16 and 8
  bytes), qh[2] (4 trits per byte), f16 d. Trit n of byte q is ((q * 3^n mod 256) * 3) >> 8; value (t - 1) d. Value order inside
  the block: v = m + 16 n (bytes 0..15), 80 + m + 8 n (bytes 16..23), 120 + h + 2 n (qh). Port of ggml-quants.c
  dequantize_row_ptq1_0 (PrismML-Eng/llama.cpp).

Import this module before opening such a GGUF (inspect_gguf.py, make_kernel_tests.py, roofline.py, chat_probe.py do)."""
import numpy as np
import gguf
from gguf import constants as _c, quants as _q

EXTRA = {"Q2_0": (42, 64, 18), "PQ2_0": (142, 128, 34), "PTQ1_0": (143, 128, 28)}   # name: (type id, block values, block bytes)


def _member(name, value):
    T = _c.GGMLQuantizationType
    if value in T._value2member_map_:
        return T._value2member_map_[value]
    m = int.__new__(T, value)
    m._name_, m._value_ = name, value
    T._value2member_map_[value] = m
    T._member_map_[name] = m
    T._member_names_.append(name)
    return m


def _codes2(blocks, qk):
    n = blocks.shape[0]
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
    qs = blocks[:, 2:2 + qk // 4].reshape(n, qk // 4, 1) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4)
    q = (qs & 3).reshape(n, qk).astype(np.int8) - np.int8(1)
    return d * q.astype(np.float32)


def ptq1_0_trits(blocks):
    """(n, 128) int8 trits in {-1, 0, 1} of n PTQ1_0 blocks (uint8 rows of 28 bytes), in value order."""
    n = blocks.shape[0]
    p5 = np.array([1, 3, 9, 27, 81], dtype=np.uint8).reshape(1, 5, 1)
    s0 = (blocks[:, 0:16].reshape(n, 1, 16) * p5).reshape(n, 80)
    s1 = (blocks[:, 16:24].reshape(n, 1, 8) * p5).reshape(n, 40)
    sh = (blocks[:, 24:26].reshape(n, 1, 2) * p5[:, :4]).reshape(n, 8)
    q = np.concatenate([s0, s1, sh], axis=1)
    return ((q.astype(np.uint16) * 3) >> 8).astype(np.int8) - np.int8(1)


def ptq1_0_scale(blocks):
    return blocks[:, 26:28].copy().view(np.float16).astype(np.float32)


def _register():
    mem = {k: _member(k, v[0]) for k, v in EXTRA.items()}
    for k, (tid, qk, nb) in EXTRA.items():
        _c.GGML_QUANT_SIZES.setdefault(mem[k], (qk, nb))
    if mem["PTQ1_0"] in _q._type_traits:
        return

    def deq(fn):
        return classmethod(lambda cls, blocks: fn(blocks))
    for k, fn in (("Q2_0", lambda b: _codes2(b, 64)), ("PQ2_0", lambda b: _codes2(b, 128)),
                  ("PTQ1_0", lambda b: ptq1_0_scale(b) * ptq1_0_trits(b).astype(np.float32))):
        if mem[k] in _q._type_traits:
            continue
        # a subclass of the package's base class registers itself (__init_subclass__) for gguf.quants.dequantize
        type(k, (_q.__dict__["__Quant"],), {"dequantize_blocks": deq(fn),
                                             "quantize_blocks": classmethod(lambda cls, b: NotImplemented)}, qtype=mem[k])


_register()
PTQ1_0 = _c.GGMLQuantizationType._value2member_map_[143]


def q2_0_to_ptq1_0(blocks):
    """Repack Q2_0 ternary rows (uint8, n x 36: two 64-value blocks per 128 values) to PTQ1_0 blocks (n x 28): the load-time
    weight policy's numpy oracle (quantize_row_ptq1_0_ref's packing; d = the scale of a half with a nonzero trit)."""
    n = blocks.shape[0]
    b = blocks.reshape(n, 2, 18)
    qs = b[:, :, 2:].reshape(n, 2, 16, 1) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 1, 4)
    t = (qs & 3).reshape(n, 128).astype(np.uint16)            # trit + 1 in 0..2
    assert t.max() <= 2, "a Q2_0 code 3 (+2) is not ternary"
    d0, d1 = b[:, 0, :2].copy(), b[:, 1, :2].copy()
    nz0 = (t[:, :64] != 1).any(1)
    d = np.where(nz0[:, None], d0, d1)
    out = np.zeros((n, 28), np.uint8)
    def pack(vals, ntr, extra):
        q = np.zeros(vals.shape[:-1], np.uint16)
        for k in range(ntr):
            q = q * 3 + vals[..., k]
        q = q * (3 ** extra)
        return ((q * 256 + 242) // 243).astype(np.uint8)
    out[:, 0:16] = pack(np.stack([t[:, m + 16 * np.arange(5)] for m in range(16)], 1), 5, 0)
    out[:, 16:24] = pack(np.stack([t[:, 80 + m + 8 * np.arange(5)] for m in range(8)], 1), 5, 0)
    out[:, 24:26] = pack(np.stack([t[:, 120 + h + 2 * np.arange(4)] for h in range(2)], 1), 4, 1)
    out[:, 26:28] = d
    return out
