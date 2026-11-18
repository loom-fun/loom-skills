"""Kernel-test cases of the speculative-decoding bookkeeping kernels (draft/gen_draft.py, gen_llm.py g_token_info) against
numpy simulations of their contracts: g_token_info, g_spec_rows, g_spec_accept, g_spec_rmsnorm, and the MTP drafter's
g_mtp_prepare / g_mtp_gather / g_mtp_advance / g_mtp_pair / g_mtp_save / g_mtp_copy / g_mtp_concat.

Metadata (the epoch packet): 4 i32 header words {count, eos, mixed, 0}, then 9-word span records {length, position, -,
row, cursor (first packed input), -, flags (1 selected, 2 proposes, 4 speculative, credit << 8), -, -}; with M-RoPE a table
of `count` i32 offsets follows the first `count` records. Token ids are packed with stride 8 when header word 2 (mixed) is
set, else 1. Every output buffer starts filled with a sentinel, so a word a kernel must not write is checked too.

cases(lib, save, rng, ks) -> [check.case text]: only for the kernels the library LIB defines."""
import re

import numpy as np

SENT = -7          # i32 sentinel of untouched output words
KS = ""            # make_kernel_tests.py's file suffix (KTEST_HEADS_SFX child runs): save() appends it to every name
FSENT = -3.25      # f32 sentinel


def _def(lib, name):
    m = re.search(r'(?ms)^kernel\.def[^\n]*export\("' + name + r'"\).*?^\}\n', lib)
    return m.group(0) if m else None


def _width(lib, name):
    """The hidden width a row kernel was generated for: its 32-value group count (%groups) times 32, or None."""
    m = re.search(r"^\s*%groups = index\.constant (\d+) : index$", _def(lib, name) or "", re.M)
    return int(m.group(1)) * 32 if m else None


def _eps(lib, name):
    m = re.search(r"^\s*%eps = scalar\.constant ([0-9.e+-]+) : f32$", _def(lib, name) or "", re.M)
    return float(m.group(1)) if m else None


def _ty(a):
    return "tensor<" + "x".join(map(str, a.shape)) + "x" + {"float32": "f32", "int32": "i32", "int8": "i8", "float16": "f16"}[str(a.dtype)] + ">"


class _Case:
    """Builds one check.case: inputs read from .npy files, outputs filled with a sentinel, exact or close expectations."""

    def __init__(self, name, save):
        self.name, self.save, self.lines, self.checks = name, save, [], []

    def lit(self, var, v):
        self.lines.append(f"  %{var} = check.literal value({v}) : index")
        return f"%{var}", "index"

    def inp(self, var, a):
        self.save(f"{self.name}_{var}", a)
        self.lines.append(f'  %{var} = check.file.read.npy path("{self.name}_{var}{self._sfx()}.npy") : {_ty(a)}')
        return f"%{var}", _ty(a)

    def out(self, var, expect, init=None, close=None):
        """An output buffer: initial contents INIT (default: the sentinel), checked against EXPECT."""
        init = np.full(expect.shape, SENT if expect.dtype == np.int32 else FSENT, expect.dtype) if init is None else init
        v, t = self.inp(var, init)
        self.save(f"{self.name}_{var}_e", expect)
        self.lines.append(f'  %{var}_e = check.file.read.npy path("{self.name}_{var}_e{self._sfx()}.npy") : {t}')
        if close is None:
            self.checks.append(f"  check.expect.equal actual(%{var}) expected(%{var}_e) : {t}")
        else:
            self.checks.append(f"  check.expect.close actual(%{var}) expected(%{var}_e) atol({close}) rtol({close}) nan(different) : {t}")
        return v, t

    def _sfx(self):
        return KS

    def launch(self, kern, grid, args):
        g = f"[{', '.join(grid)}]" if grid else ""
        gt = f"[{', '.join(['index'] * len(grid))}]" if grid else ""
        self.lines.append(f"  kernel.launch @{kern}{g}({', '.join(a for a, _ in args)}) : {gt}({', '.join(t for _, t in args)})")

    def text(self):
        return f"check.case public @{self.name} {{\n" + "\n".join(self.lines + self.checks) + "\n  check.return\n}\n"


def _packet(rng, spans, tokens, eos, mixed, mrope, flags=None, lengths=None, count=None):
    """(metadata i32 array, records [spans x 9], token ids [tokens * stride], m table or None)."""
    lengths = lengths if lengths is not None else rng.integers(1, 5, spans)
    rec = rng.integers(0, 1000, (spans, 9)).astype(np.int32)
    cur = 0
    for s in range(spans):
        rec[s, 0] = lengths[s]
        rec[s, 4] = cur
        cur += int(lengths[s])
        rec[s, 3] = (s * 3 + 1) % 7
        rec[s, 6] = flags[s] if flags is not None else int(rng.integers(0, 8))
    assert cur <= tokens, (cur, tokens)
    count = spans if count is None else count
    head = np.array([count, eos, int(mixed), 0], np.int32)
    tab = rng.integers(-50, 50, count).astype(np.int32) if mrope else None
    meta = np.concatenate([head, rec.reshape(-1)] + ([tab] if mrope else []))
    stride = 8 if mixed else 1
    ids = rng.integers(0, 50, tokens * stride).astype(np.int32)
    return meta, rec, ids, tab


def token_info_case(save, rng):
    T, S = 13, 4
    meta, rec, _, _ = _packet(rng, S, T, 2, False, False, lengths=np.array([3, 1, 4, 2]), count=3)
    count = min(3, S)
    info = np.zeros((T, 4), np.int32)
    for t in range(T):
        r, p, l_, f = -1, 0, 0, -1
        for s in range(count):
            if rec[s, 4] <= t < rec[s, 4] + rec[s, 0]:
                r, p, l_, f = rec[s, 3], rec[s, 1] + t - rec[s, 4], 1, s
        info[t] = (r, p, l_, f)
    c = _Case("spec_token_info", save)
    args = [c.lit("t", T), c.lit("s", S), c.inp("meta", meta), c.out("info", info)]
    c.launch("g_token_info", ["%t"], args)
    return c.text()


def spec_rows_case(save, rng):
    S, R = 5, 9
    flags = np.array([1 | 4, 1, 0, 1 | 4, 1 | 4], np.int32)
    meta, rec, _, _ = _packet(rng, S, 40, 3, False, False, flags=flags, lengths=np.array([3, 2, 4, 4, 2]))
    out = np.full(4 + 9 * R, SENT, np.int32)
    o = out[4:].reshape(R, 9)
    rr = 0
    for s in range(S):
        n = (rec[s, 0] if flags[s] & 4 else 1) if flags[s] & 1 else 0
        for j in range(n):
            if rr < R:
                o[rr] = rec[s]
                o[rr, 0] = j + 1 if flags[s] & 4 else rec[s, 0]
                o[rr, 6] = 1
            rr += 1
    out[:4] = (min(rr, R), 3, 0, 0)
    c = _Case("spec_rows", save)
    args = [c.lit("s", S), c.lit("r", R), c.inp("meta", meta), c.out("rows", out)]
    c.launch("g_spec_rows", [], args)
    return c.text()


def spec_accept_case(save, rng, mrope, mixed):
    S, T, eos = 6, 24, 9
    lengths = np.array([4, 2, 3, 4, 1, 3])
    flags = np.array([4 | (8 << 8), 1, 4 | (2 << 8), 4 | (8 << 8), 0, 1 | 4 | (8 << 8)], np.int32)
    meta, rec, ids, tab = _packet(rng, S, T, eos, mixed, mrope, flags=flags, lengths=lengths)
    stride = 8 if mixed else 1
    R = 12
    pred = rng.integers(0, 50, R).astype(np.int32)
    # make the drafts of span 0 match for two tokens, span 3 hit EOS on its second prediction
    pred[0] = ids[(rec[0, 4] + 1) * stride]; pred[1] = ids[(rec[0, 4] + 2) * stride]
    r3 = 4 + 1 + 3   # rows used by spans 0 (4), 1 (1), 2 (3)
    pred[r3] = ids[(rec[3, 4] + 1) * stride]; pred[r3 + 1] = eos
    res = np.full((S, 10), SENT, np.int32)
    com = np.full(len(meta), SENT, np.int32)
    rep = np.full(len(meta), SENT, np.int32)
    cr, rp = com[4:4 + 9 * S].reshape(S, 9), rep[4:4 + 9 * S].reshape(S, 9)
    r = k = 0
    for s in range(S):
        f = int(flags[s]); length = int(rec[s, 0]); cursor = int(rec[s, 4])
        res[s, 2:10] = 0
        if f & 4:
            width, credit, cnt, live = length, (f >> 8) & 255, 0, True
            for j in range(width):
                if live and j < credit and r + j < R:
                    p = int(pred[r + j]); res[s, 2 + j] = p; cnt += 1
                    match = j < width - 1 and p == ids[(cursor + j + 1) * stride]
                    live = bool(match and p != eos)
                else:
                    live = False
            consumed = emitted = cnt; used = width
        else:
            if f & 1 and r < R:
                res[s, 2] = pred[r]; emitted, used = 1, 1
            else:
                emitted, used = 0, 0
            consumed = length
        res[s, 0], res[s, 1] = consumed, emitted
        cr[s] = rec[s]; cr[s, 0] = consumed; cr[s, 6] = f & 1
        if f & 4:
            rp[k] = rec[s]; rp[k, 0] = consumed; rp[k, 6] = 0; k += 1
        if mrope:
            com[4 + 9 * S + s] = tab[s]
        r += used
    com[:4] = (S, eos, int(mixed), 0)
    rep[:4] = (k, eos, int(mixed), 0)
    c = _Case(f"spec_accept{'_mrope' if mrope else ''}{'_mixed' if mixed else ''}", save)
    args = [c.lit("s", S), c.lit("r", R), c.lit("t", T), c.inp("meta", meta), c.inp("ids", ids), c.inp("pred", pred),
            c.out("res", res), c.out("com", com), c.out("rep", rep)]
    c.launch("g_spec_accept", [], args)
    return c.text()


def mtp_prepare_case(save, rng, mrope, mixed):
    S, T, D, eos = 6, 30, 4, 5
    flags = np.array([2, 0, 2 | 1, 2, 2, 2], np.int32)   # five proposing spans for four drafts: the last is dropped
    meta, rec, ids, tab = _packet(rng, S, T, eos, mixed, mrope, flags=flags, lengths=np.array([3, 2, 4, 1, 5, 2]))
    stride = 8 if mixed else 1
    draft = np.full(4 + 9 * D + (D if mrope else 0), SENT, np.int32)
    dr = draft[4:4 + 9 * D].reshape(D, 9)
    dids = np.full(D, SENT, np.int32); slots = np.full(D, SENT, np.int32)
    k = 0
    for s in range(S):
        if flags[s] & 2 and k < D:
            dr[k] = rec[s]; dr[k, 0] = 1; dr[k, 2] = 0; dr[k, 4] = k; dr[k, 6] = 1
            dids[k] = ids[rec[s, 4] * stride]; slots[k] = rec[s, 4]
            if mrope:
                draft[4 + 9 * D + k] = tab[s]
            k += 1
    draft[:4] = (k, eos, 0, 0)
    c = _Case(f"mtp_prepare{'_mrope' if mrope else ''}{'_mixed' if mixed else ''}", save)
    args = [c.lit("s", S), c.lit("d", D), c.lit("t", T), c.inp("meta", meta), c.inp("ids", ids), c.out("draft", draft),
            c.out("dids", dids), c.out("slots", slots)]
    c.launch("g_mtp_prepare", [], args)
    return c.text()


def mtp_gather_case(save, rng, H):
    D, rows = 4, 6
    draft = np.full(4 + 9 * D, 0, np.int32); draft[0] = 3
    dr = draft[4:].reshape(D, 9)
    dr[:, 3] = [5, 0, 2, 1]
    carry = rng.standard_normal((rows, H)).astype(np.float32)
    hid = np.zeros((D, H), np.float32)
    for k_ in range(3):
        hid[k_] = carry[dr[k_, 3]]
    c = _Case("mtp_gather", save)
    args = [c.lit("d", D), c.lit("o", 1), c.inp("draft", draft), c.inp("carry", carry), c.out("hid", hid)]
    c.launch("g_mtp_gather", ["%d"], args)
    return c.text()


def mtp_advance_case(save, rng, mixed):
    D, T, rnd = 4, 24, 1
    meta = np.array([0, 0, int(mixed), 0], np.int32)
    stride = 8 if mixed else 1
    draft = rng.integers(0, 100, 4 + 9 * D).astype(np.int32); draft[0] = 3
    slots = np.array([0, 5, 11, 2], np.int32)
    prop = rng.integers(0, 50, D).astype(np.int32)
    ids = rng.integers(0, 50, T * stride).astype(np.int32)
    dids0 = np.full(D, SENT, np.int32)
    ids_e, dids_e, draft_e = ids.copy(), dids0.copy(), draft.copy()
    for k_ in range(min(3, D)):
        ids_e[(slots[k_] + rnd + 1) * stride] = prop[k_]
        dids_e[k_] = prop[k_]
        draft_e[4 + 9 * k_ + 1] += 1
    c = _Case(f"mtp_advance{'_mixed' if mixed else ''}", save)
    args = [c.lit("rnd", rnd), c.lit("d", D), c.lit("t", T), c.inp("meta", meta), c.out("draft", draft_e, init=draft),
            c.inp("slots", slots), c.inp("prop", prop), c.out("dids", dids_e, init=dids0), c.out("ids", ids_e, init=ids)]
    c.launch("g_mtp_advance", [], args)
    return c.text()


def mtp_pair_save_cases(save, rng, H, pair, keep):
    T, S, rows = 9, 3, 5
    meta, rec, _, _ = _packet(rng, S, T, 1, False, False, lengths=np.array([3, 2, 3]))
    hidden = rng.standard_normal((T, H)).astype(np.float32)
    carry = rng.standard_normal((rows, H)).astype(np.float32)
    out = []
    if pair:
        info = np.zeros((T, 4), np.int32)
        for s in range(S):
            for t in range(rec[s, 4], rec[s, 4] + rec[s, 0]):
                info[t] = (rec[s, 3], 0, 1, s)
        info[8, 2] = 0          # a dead row
        prev = np.zeros((T, H), np.float32)
        for t in range(T):
            if info[t, 2]:
                s = info[t, 3]
                prev[t] = carry[info[t, 0]] if t == rec[s, 4] else hidden[max(t, 1) - 1]
        c = _Case("mtp_pair", save)
        args = [c.lit("t", T), c.lit("s", S), c.lit("o", 1), c.inp("meta", meta), c.inp("info", info), c.inp("hid", hidden),
                c.inp("carry", carry), c.out("prev", prev)]
        c.launch("g_mtp_pair", ["%t"], args)
        out.append(c.text())
    if keep:
        meta2 = meta.copy(); meta2[0] = 2           # the third span is past the live count
        rec2 = rec.copy(); rec2[1, 0] = 0             # an empty span keeps its row's carry
        meta2[4:4 + 9 * S] = rec2.reshape(-1)
        carry_e = carry.copy()
        for s in range(2):
            if rec2[s, 0] > 0:
                carry_e[rec2[s, 3]] = hidden[rec2[s, 4] + rec2[s, 0] - 1]
        c = _Case("mtp_save", save)
        args = [c.lit("t", T), c.lit("s", S), c.lit("o", 1), c.inp("meta", meta2), c.inp("hid", hidden), c.out("carry", carry_e, init=carry)]
        c.launch("g_mtp_save", ["%s"], args)
        out.append(c.text())
    return out


def mtp_rows_cases(save, rng, lib):
    T = 3
    out = []
    for name in ("g_mtp_copy", "g_mtp_concat", "g_spec_rmsnorm"):
        H = _width(lib, name)
        if not H:
            continue
        a = rng.standard_normal((T, H)).astype(np.float32) * np.array([[1e-2], [1.0], [40.0]], np.float32)
        b = rng.standard_normal((T, H)).astype(np.float32)
        c = _Case(name[2:], save)
        if name == "g_mtp_copy":
            c.launch(name, ["%t"], [c.lit("t", T), c.inp("a", a), c.out("o", a.copy())])
        elif name == "g_mtp_concat":
            c.launch(name, ["%t"], [c.lit("t", T), c.inp("a", a), c.inp("b", b), c.out("o", np.stack([a, b], 1).astype(np.float32))])
        else:
            eps = _eps(lib, name)
            w = (1 + 0.1 * rng.standard_normal(H)).astype(np.float32)
            a64 = a.astype(np.float64)
            y = (a64 / np.sqrt((a64 * a64).mean(-1, keepdims=True) + eps) * w).astype(np.float32)
            c.launch(name, ["%t"], [c.lit("t", T), c.inp("a", a), c.inp("w", w), c.out("o", y, close="1.0e-5")])
        out.append(c.text())
    return out


def cases(lib, save, rng, ks=""):
    """The cases of the bookkeeping kernels LIB defines; widths and eps are read off each kernel."""
    global KS
    KS = ks
    if not lib:
        return []
    out = []
    has = lambda k: _def(lib, k) is not None   # noqa: E731
    if has("g_token_info"):
        out.append(token_info_case(save, rng))
    if has("g_spec_rows"):
        out.append(spec_rows_case(save, rng))
    if has("g_spec_accept"):
        mrope = "%m_tv" in _def(lib, "g_spec_accept")   # the M-RoPE offset table is copied
        out += [spec_accept_case(save, rng, mrope, False), spec_accept_case(save, rng, mrope, True)]
    if has("g_mtp_prepare"):
        mrope = "%m_tv" in _def(lib, "g_mtp_prepare")
        out += [mtp_prepare_case(save, rng, mrope, False), mtp_prepare_case(save, rng, mrope, True)]
    if _width(lib, "g_mtp_gather"):
        out.append(mtp_gather_case(save, rng, _width(lib, "g_mtp_gather")))
    if has("g_mtp_advance"):
        out += [mtp_advance_case(save, rng, False), mtp_advance_case(save, rng, True)]
    pw, sw = _width(lib, "g_mtp_pair"), _width(lib, "g_mtp_save")
    if pw or sw:
        out += mtp_pair_save_cases(save, rng, pw or sw, bool(pw), bool(sw) and sw == (pw or sw))
    out += mtp_rows_cases(save, rng, lib)
    return out
