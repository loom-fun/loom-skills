#!/usr/bin/env python3
"""CPU-only compact-cache check adapter. No checkpoint access or GPU execution.

Actual emitted kernels are extracted recursively and only their symbols renamed.
Independent explicit owner tables and full-history F64 attention supply the
oracle. Fixtures are small; GPU-only arena allocations are initialized by checks.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse
from dataclasses import dataclass, asdict, replace
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "fleet"))
from loomtext import split_blocks

CONTEXT, WINDOW, RING = 65536, 512, 2048
LOCAL_OWNERS = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]
GLOBAL_OWNERS = [5, 11, 17, 23]
WRITER = 4096
SLOT = 40 * 2048 * 1024
EPS = 9.999999974752427e-7


@dataclass(frozen=True)
class Case:
    name: str
    start: int
    tokens: int
    owner: int = 22
    dim: int = 256
    store: bool = False
    stale: bool = False
    bad: str = ""
    write_kv: bool = True
    store_lv: bool = False


CASES = [Case("reset", 0, 32, 0, stale=True), Case("edge511", 511, 32),
         Case("edge512", 512, 1), Case("edge1023", 1023, 2),
         Case("overlap_store", 1025, 1024, store=True, store_lv=True), Case("masked_history", 1537, 32),
         Case("wrap_two", 2047, 2), Case("native64k", 65024, 512, store=True),
         Case("shared_readonly", 2047, 32, store=True, write_kv=False),
         Case("global_store", 1023, 32, 23, 512, store=True, store_lv=True)]
NEGATIVES = [Case("bad_ring512", 1025, 1024, bad="ring512"),
             Case("bad_ring1024", 1025, 1024, bad="ring1024"),
             Case("bad_v_stride", 2047, 2, bad="v_stride"),
             Case("bad_fork_copy", 2047, 2, bad="fork_copy")]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tensor(a):
    return "tensor<" + "x".join(map(str, a.shape)) + "x" + {"float16": "f16", "float32": "f32", "int32": "i32", "int64": "i64"}[str(a.dtype)] + ">"


def extract(text, names, namespace, bad=""):
    blocks = {s: (k, body) for k, s, _, body in split_blocks(text)}
    selected = {}
    def visit(name):
        if name in selected:
            return
        kind, body = blocks[name]
        if kind not in ("kernel.def", "func.def"):
            raise ValueError(f"unexpected dependency {name}: {kind}")
        selected[name] = body
        for dep in re.findall(r"func.call @([\w.]+)", body):
            visit(dep)
    for name in names:
        visit(name)
    records, output, changed = {}, [], 0
    for name, original in selected.items():
        body = original
        # Deliberately broken address variants remain otherwise actual kernels.
        if bad in ("ring512", "ring1024"):
            ring = int(bad[4:])
            body = body.replace("%ring = index.constant 2048 : index", f"%ring = index.constant {ring} : index")
            body = re.sub(r"(%(?:plane_stride|mt_plane_stride|v_plane_delta) = index.constant )2097152( : offset)", rf"\g<1>{ring * 1024}\2", body)
        elif bad == "v_stride":
            body = re.sub(r"(%(?:plane_stride|mt_plane_stride|v_plane_delta) = index.constant )2097152( : offset)", rf"\g<1>{CONTEXT * 2048}\2", body)
        changed += body != original
        renamed = re.sub(r"@([\w.]+)", lambda m: "@" + namespace + m[1] if m[1] in selected else m[0], body)
        renamed = re.sub(r'export\("([\w.]+)"\)', lambda m: f'export("{namespace}{m[1]}")', renamed)
        # Exact de-namespacing proves no accidental numerical edits.
        restored = renamed.replace("@" + namespace, "@").replace('export("' + namespace, 'export("')
        assert restored == body
        records[name] = {"original_sha256": digest(original.encode()), "emitted_sha256": digest(renamed.encode()), "negative_mutation": bad or None}
        output.append(renamed)
    if bad in ('ring512', 'ring1024', 'v_stride') and not changed:
        raise ValueError('negative mutation did not change actual emitted address code')
    return "\n\n".join(output), records


def addresses(case, positions, pages, compact, capacity=RING, writer=WRITER):
    """Independent explicit physical packing; does not import compact_kv.py."""
    result = np.empty((len(positions), 2), np.int64)
    if case.dim == 512:
        # Legacy owner5 and compact owner23 have identical D512 arithmetic.
        planes = [10, 11] if not compact else [6, 7]
        physical = pages[np.asarray(positions) // 16] * 16 + np.asarray(positions) % 16
        for kv in range(2):
            result[:, kv] = (planes[kv] * CONTEXT + physical) * 2048
    elif not compact:
        physical = pages[np.asarray(positions) // 16] * 16 + np.asarray(positions) % 16
        result[:, 0] = physical * 2048
        result[:, 1] = (CONTEXT + physical) * 2048
    else:
        ordinal = LOCAL_OWNERS.index(case.owner)
        for kv in range(2):
            result[:, kv] = writer + ((ordinal * 2 + kv) * capacity + np.asarray(positions) % capacity) * 1024
    return result


def norm(x, weights):
    x = x.astype(np.float64)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * weights


def rope(x, positions):
    # Independent E4B Neox definition: local theta1e4/full half, global theta1e6
    # with only the first64 frequency pairs active (remaining pairs identity).
    dim = x.shape[-1]
    theta = np.asarray(positions)[:, None, None] / ((10000.0 if dim == 256 else 1000000.0) ** (2 * np.arange(dim // 2) / dim))[None, None, :]
    if dim == 512:
        theta[..., 64:] = 0
    a, b = x[..., :dim // 2], x[..., dim // 2:]
    return np.concatenate((a * np.cos(theta) - b * np.sin(theta), a * np.sin(theta) + b * np.cos(theta)), axis=-1)


def reference(case, q, history, positions):
    t, d = case.tokens, case.dim
    result = np.empty((t, 8, d), np.float32)
    qp = case.start + np.arange(t)
    mask = positions[None, :] <= qp[:, None]
    if d == 256:
        mask &= positions[None, :] > qp[:, None] - WINDOW
    # One head at a time: largest temporary is1024x1550 F64 (<13MiB).
    for h in range(8):
        kh = h // 4
        score = q[:, h].astype(np.float64) @ history[:, 0, kh].astype(np.float64).T
        score[~mask] = -np.inf
        prob = np.exp(score - np.max(score, axis=1, keepdims=True))
        prob /= np.sum(prob, axis=1, keepdims=True)
        result[:, h] = (prob @ history[:, 1, kh].astype(np.float64)).astype(np.float32)
    return result.reshape(t, 8 * d)


def fixture(case):
    seed = int.from_bytes(hashlib.sha256(case.name.removeprefix("control_").encode()).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    low = max(0, case.start - 526) if case.dim == 256 else 0
    positions = np.arange(low, case.start + case.tokens, dtype=np.int64)
    n, d, t = len(positions), case.dim, case.tokens
    history = np.empty((n, 2, 2, d), np.float16)
    history[:, 0] = rng.normal(0, .125, (n, 2, d)).astype(np.float16)
    history[:, 1] = norm(rng.normal(.5, .35, (n, 2, d)).astype(np.float32), 1).astype(np.float16)
    q = rng.normal(0, .1, (t, 8, d)).astype(np.float32)
    raw_k = rng.normal(0, 1, (t, 2, d)).astype(np.float32)
    raw_v = rng.normal(.5, .35, (t, 2, d)).astype(np.float32)
    qn = np.linspace(.08, .12, d).astype(np.float32)
    kn = np.linspace(.10, .15, d).astype(np.float32)
    if case.store:
        q_golden, k_golden = norm(q, qn), norm(raw_k, kn)
        q_golden = rope(q_golden, case.start + np.arange(t))
        k_golden = rope(k_golden, case.start + np.arange(t))
        if case.write_kv:
            history[-t:, 0] = k_golden.astype(np.float16)
            history[-t:, 1] = norm(raw_v, 1).astype(np.float16)
    else:
        q_golden = q.astype(np.float64)
    expected = reference(case, q_golden, history, positions)
    expected_flash = reference(case, q_golden.astype(np.float16), history, positions)
    pages = np.arange(CONTEXT // 16, dtype=np.int64); rng.shuffle(pages)
    info = np.zeros((t, 4), np.int32); info[:, 1] = case.start + np.arange(t); info[:, 2] = 1
    origin = np.zeros(10 + len(pages), np.int32)
    origin[:10].view(np.int64)[4] = 40
    origin[10:] = pages.astype(np.int32)
    # Deliberate missing-fork-copy control has distinct valid reader/writer slots.
    writer = WRITER + SLOT if case.bad == "fork_copy" else WRITER
    compact_origin = origin.copy()
    compact_origin[:10].view(np.int64)[0:2] = [WRITER, writer]
    capacity = int(case.bad[4:]) if case.bad.startswith("ring") else RING
    a_old = addresses(case, positions, pages, False)
    a_new = addresses(case, positions, pages, True, capacity, writer)
    if case.bad == "fork_copy":
        old_history = positions < case.start
        a_new[old_history] = addresses(case, positions[old_history], pages, True, capacity, WRITER)
    # Safe negatives: simulate completed chronological writes on CPU, then scatter
    # each destination once. Never send aliased writes to the GPU packer.
    last = {}
    for i, address in enumerate(a_new[:, 0]):
        last[int(address)] = i
    keep = np.array(sorted(last.values()), dtype=np.int64)
    pack_old, pack_new = np.arange(n), keep
    if case.store and case.write_kv:
        pack_old = pack_old[positions[pack_old] < case.start]
        pack_new = pack_new[positions[pack_new] < case.start]
    arrays = dict(q=q.reshape(t, -1), k=raw_k.reshape(t, -1), v=raw_v.reshape(t, -1), qn=qn, kn=kn,
                  info=info, old_origin=origin, new_origin=compact_origin, expected=expected, expected_flash=expected_flash,
                  old_addresses=a_old[pack_old], new_addresses=a_new[pack_new],
                  old_values=history[pack_old], new_values=history[pack_new],
                  old_current=a_old[-t:], new_current=addresses(case, positions[-t:], pages, True, capacity, writer),
                  stored_expected=history[-t:])
    old_bytes = (12 if d == 512 else 2) * CONTEXT * 2048
    new_bytes = 8 * CONTEXT * 2048 if d == 512 else writer + SLOT
    if case.bad == "v_stride":
        new_bytes += CONTEXT * 2048
    for prefix, limit in (("old", old_bytes), ("new", new_bytes)):
        a = arrays[prefix + "_addresses"]
        assert len(np.unique(a)) == a.size, "GPU scatter addresses must be unique"
        assert a.min(initial=0) >= 0 and a.max(initial=0) + 2 * d * 2 <= limit
    return arrays, old_bytes, new_bytes


def transfer(dim, gather=False):
    name = f"adapter_{'gather' if gather else 'scatter'}{dim}"
    load = f"%value = view.load %cache[0] : view<1xf16> -> f16\n  view.store %value, %data[%entry, %which, %head, %dim] : f16, view<[%n]x2x2x{dim}xf16>" if gather else f"%value = view.load %data[%entry, %which, %head, %dim] : view<[%n]x2x2x{dim}xf16> -> f16\n  view.store %value, %cache[0] : f16, view<1xf16>"
    return f"""kernel.def export("{name}") @{name}(%count: index) {{
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %cd = index.constant {dim} : index
  kernel.launch.config workgroups(%count, %c4, %c1) workgroup_size(%cd, %c1, %c1) : index
}} launch(%n_arg: index, %addresses: buffer, %values: buffer, %arena: buffer) {{
  %n = index.assume %n_arg [range(%n_arg, 1, 65536)] : index
  %ei = kernel.workgroup.id<x> : index
  %entry = index.assume %ei [lt(%ei, %n)] : index
  %gi = kernel.workgroup.id<y> : index
  %group = index.assume %gi [range(%gi, 0, 3)] : index
  %di = kernel.workitem.id<x> : index
  %dim = index.assume %di [range(%di, 0, {dim - 1})] : index
  %c2 = index.constant 2 : index
  %cd = index.constant {dim} : index
  %two = index.constant 2 : offset
  %zero = index.constant 0 : offset
  %which = index.div %group, %c2 : index
  %head = index.rem %group, %c2 : index
  %address_view = buffer.view %addresses[%zero] : buffer -> view<[%n]x2xi64>
  %base64 = view.load %address_view[%entry, %which] : view<[%n]x2xi64> -> i64
  %base_b = scalar.assume %base64 [range(%base64, 0, 2147483647)] : i64
  %base = index.cast %base_b : i64 to offset
  %element = index.madd %head, %cd, %dim : index
  %delta = index.scale %element, %two : index, offset -> offset
  %address = index.add %base, %delta : offset
  %cache = buffer.view %arena[%address] : buffer -> view<1xf16>
  %data = buffer.view %values[%zero] : buffer -> view<[%n]x2x2x{dim}xf16>
  {load}
  kernel.return
}}
"""


BITS = """kernel.def export("adapter_bits") @adapter_bits(%n: index) {
  %c1 = index.constant 1 : index
  %c255 = index.constant 255 : index
  %c256 = index.constant 256 : index
  %pad = index.add %n, %c255 : index
  %groups = index.div %pad, %c256 : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%n_arg: index, %input: buffer, %output: buffer) {
  %n = index.assume %n_arg [range(%n_arg, 1, 8388608)] : index
  %group = kernel.workgroup.id<x> : index
  %lane = kernel.workitem.id<x> : index
  %c256 = index.constant 256 : index
  %iv = index.madd %group, %c256, %lane : index
  %live = index.cmp ult, %iv, %n : index
  %zero = index.constant 0 : offset
  scf.if %live {
    %i = index.assume %iv [lt(%iv, %n)] : index
    %x = buffer.view %input[%zero] : buffer -> view<[%n]xf32>
    %y = buffer.view %output[%zero] : buffer -> view<[%n]xi32>
    %value = view.load %x[%i] : view<[%n]xf32> -> f32
    %bits = scalar.bitcast %value : f32 to i32
    view.store %bits, %y[%i] : i32, view<[%n]xi32>
  }
  kernel.return
}
"""


def kernel_names(dim):
    flash = "g_attention_flash2" + ("_d512" if dim == 512 else "")
    return [f"g_qk_rope_store_d{dim}", f"gd{dim}_attention_split", f"g16d{dim}_attention_split",
            f"gd{dim}_attention_combine", f"g16d{dim}_attention_combine",
            f"g16d{dim}_attention_split_listed", f"g16d{dim}_attention_combine_listed",
            f"g_qk_rope_store_d{dim}_lv", "g_attention_fallback_list", flash, flash + "_partial"]


def variants(case):
    result = ['decode', 'decode16']
    if case.name in ('edge512', 'global_store'):
        result.append('listed')
    if case.tokens >= 16 and case.tokens % 16 == 0 and not case.bad:
        result.append('flash')
    return result


def check_case(case, arrays, old_bytes, new_bytes, namespace="compact__"):
    t, d = case.tokens, case.dim
    qtype = f"tensor<{t}x{8 * d}xf32>"
    info = tensor(arrays['info']); orig = tensor(arrays['old_origin'])
    acc = f"tensor<{t * 8 * 64 * d}xf32>"; ml = f"tensor<{t * 8 * 64 * 2}xf32>"
    arena_types = {'old':f'tensor<{old_bytes // 2}xf16>', 'new':f'tensor<{new_bytes // 2}xf16>'}
    lines = [f"check.case public @{case.name} {{", f"  %t = check.literal value({t}) : index",
             f"  %ctx = check.literal value({CONTEXT}) : index", f"  %pool = check.literal value({CONTEXT}) : index",
             f"  %sliding = check.literal value({int(d == 256)}) : i32", "  %rope = check.literal value(1) : i32",
             f"  %store = check.literal value({int(case.write_kv)}) : i32", f"  %bits_n = check.literal value({t * 8 * d}) : index"]
    for key in ['info','expected','expected_flash','k','v','qn','kn','stored_expected']:
        lines.append(f'  %{key} = check.file.read.npy path("{case.name}_{key}.npy") : {tensor(arrays[key])}')
    arms = [('old',5 if d == 512 else 0,'legacy__'),('new',case.owner,namespace)]
    # Negative samples contain ONLY the deliberately broken candidate. The
    # identically seeded legacy/correct candidate pair is a separate positive
    # control, so a baseline error cannot satisfy an expected failure.
    for arm, owner, prefix in (arms[1:] if case.bad else arms):
        ar = arena_types[arm]; n = len(arrays[arm+'_values'])
        lines += [f"  %{arm}_layer = check.literal value({owner}) : index",
                  f'  %{arm}_q = check.file.read.npy path("{case.name}_q.npy") : {qtype}',
                  f'  %{arm}_origin = check.file.read.npy path("{case.name}_{arm}_origin.npy") : {orig}',
                  f'  %{arm}_addresses = check.file.read.npy path("{case.name}_{arm}_addresses.npy") : {tensor(arrays[arm+"_addresses"])}',
                  f'  %{arm}_values = check.file.read.npy path("{case.name}_{arm}_values.npy") : {tensor(arrays[arm+"_values"])}',
                  f'  %{arm}_current = check.file.read.npy path("{case.name}_{arm}_current.npy") : {tensor(arrays[arm+"_current"])}',
                  f"  %{arm}_arena = check.generate.fill value({-3.0 if arm == 'old' and case.stale else 7.0 if case.stale else 0.0}) : {ar}"]
        if n:
            lines += [f"  %{arm}_n = check.literal value({n}) : index",
                      f"  kernel.launch @adapter_scatter{d}[%{arm}_n](%{arm}_n, %{arm}_addresses, %{arm}_values, %{arm}_arena) : [index](index, {tensor(arrays[arm+'_addresses'])}, {tensor(arrays[arm+'_values'])}, {ar})"]
        if case.store:
            store_suffix = '_lv' if case.store_lv else ''
            lines += [f"  kernel.launch @{prefix}g_qk_rope_store_d{d}{store_suffix}[%t](%t, %{arm}_layer, %rope, %store, %ctx, %pool, %{arm}_q, %k, %v, %qn, %kn, %info, %{arm}_origin, %{arm}_arena) : [index](index, index, i32, i32, index, index, {qtype}, {tensor(arrays['k'])}, {tensor(arrays['v'])}, {tensor(arrays['qn'])}, {tensor(arrays['kn'])}, {info}, {orig}, {ar})",
                      f"  %{arm}_stored = check.generate.fill value(0.0) : {tensor(arrays['stored_expected'])}",
                      f"  kernel.launch @adapter_gather{d}[%t](%t, %{arm}_current, %{arm}_stored, %{arm}_arena) : [index](index, {tensor(arrays[arm+'_current'])}, {tensor(arrays['stored_expected'])}, {ar})",
                      f"  check.expect.approximate actual(%{arm}_stored) expected(%stored_expected) atol(3.0e-3) rtol(3.0e-3) nan(different) : {tensor(arrays['stored_expected'])}, {tensor(arrays['stored_expected'])}"]
        for variant in variants(case):
            tag = arm + '_' + variant
            lines.append(f"  %{tag}_out = check.generate.fill value(0.0) : {qtype}")
            if variant.startswith('decode') or variant == 'listed':
                kn = f"{'g' if variant == 'decode' else 'g16'}d{d}_attention_split"
                combine = f"{'g' if variant == 'decode' else 'g16'}d{d}_attention_combine"
                launch, launch_type, arg, arg_type = '%t', 'index', '', ''
                if variant == 'listed':
                    kn += '_listed'; combine += '_listed'
                    lines += [f"  %{tag}_list = check.generate.iota offset({t - 1}) step(-1) : tensor<{t}xi32>",
                              f"  %{tag}_count = check.generate.fill value({t}) : tensor<1xi32>"]
                    launch, launch_type = f'%{tag}_count', 'tensor<1xi32>'
                    arg, arg_type = f'%{tag}_list, ', f'tensor<{t}xi32>, '
                lines += [f"  %{tag}_acc = check.generate.fill value(0.0) : {acc}", f"  %{tag}_ml = check.generate.fill value(0.0) : {ml}",
                          f"  kernel.launch @{prefix}{kn}[{launch}](%t, {arg}%{arm}_layer, %sliding, %ctx, %pool, %{arm}_q, %info, %{arm}_origin, %{arm}_arena, %{tag}_acc, %{tag}_ml) : [{launch_type}](index, {arg_type}index, i32, index, index, {qtype}, {info}, {orig}, {ar}, {acc}, {ml})",
                          f"  kernel.launch @{prefix}{combine}[{launch}](%t, {arg}%sliding, %info, %{tag}_acc, %{tag}_ml, %{arm}_q, %{tag}_out) : [{launch_type}](index, {arg_type}i32, {info}, {acc}, {ml}, {qtype}, {qtype})"]
            else:
                flash = 'g_attention_flash2' + ('_d512' if d == 512 else '')
                lines += [f"  %{tag}_tiles = check.generate.fill value(1073741823) : tensor<{t}xi32>", f"  %{tag}_list = check.generate.fill value(0) : tensor<{t}xi32>", f"  %{tag}_count = check.generate.fill value(0) : tensor<64xi32>",
                          f"  kernel.launch @{prefix}g_attention_fallback_list[%t](%t, %info, %{tag}_tiles, %{tag}_list, %{tag}_count) : [index](index, {info}, tensor<{t}xi32>, tensor<{t}xi32>, tensor<64xi32>)"]
                for suffix in ['', '_partial']:
                    lines.append(f"  kernel.launch @{prefix}{flash}{suffix}[%t](%t, %{arm}_layer, %sliding, %ctx, %pool, %{arm}_q, %{arm}_q, %info, %{tag}_tiles, %{arm}_origin, %{arm}_arena, %{tag}_out) : [index](index, index, i32, index, index, {qtype}, {qtype}, {info}, tensor<{t}xi32>, {orig}, {ar}, {qtype})")
            expected_name = 'expected_flash' if variant == 'flash' else 'expected'
            lines.append(f"  check.expect.approximate actual(%{tag}_out) expected(%{expected_name}) atol(3.0e-3) rtol(3.0e-3) nan(different) : {qtype}, {qtype}")
            if not case.bad:
                lines += [f"  %{tag}_bits = check.generate.fill value(0) : tensor<{t * 8 * d}xi32>",
                          f"  kernel.launch @adapter_bits[%bits_n](%bits_n, %{tag}_out, %{tag}_bits) : [index](index, {qtype}, tensor<{t * 8 * d}xi32>)"]
    if not case.bad:
        for variant in variants(case):
            lines.append(f"  check.expect.equal actual(%new_{variant}_bits) expected(%old_{variant}_bits) : tensor<{t * 8 * d}xi32>")
        if case.store:
            lines.append(f"  check.expect.approximate actual(%new_stored) expected(%old_stored) atol(0.0) rtol(0.0) nan(different) : {tensor(arrays['stored_expected'])}, {tensor(arrays['stored_expected'])}")
    return '\n'.join(lines + ['  check.return', '}'])


def generate(legacy, candidate, out):
    out.mkdir(parents=True, exist_ok=False)
    old, new = legacy.read_text(), candidate.read_text()
    roots = sorted(set(kernel_names(256) + kernel_names(512)))
    old_lib, old_records = extract(old, roots, 'legacy__')
    new_lib, new_records = extract(new, roots, 'compact__')
    pins_path = Path(__file__).with_name('kernel_pins.json')
    pins = json.loads(pins_path.read_text())
    for arm, records_ in [('legacy', old_records), ('compact', new_records)]:
        if {name: r['original_sha256'] for name, r in records_.items()} != pins[arm]:
            raise ValueError(f'{arm}: actual generated kernel/dependency differs from reviewed freeze')
    common = '\n'.join(transfer(d, g) for d in (256,512) for g in (False,True)) + BITS
    sources = {'positive.loom': [old_lib,new_lib,common], 'negative.loom':[old_lib,common]}
    records = {'legacy':old_records,'compact':new_records}
    for bad in ('ring512','ring1024','v_stride','fork_copy'):
        library, rec = extract(new, kernel_names(256)[1:5], bad + '__', bad)
        sources['negative.loom'].append(library);records[bad]=rec
    cases = []
    positive_cases = CASES + [replace(c, name='control_'+c.name, bad='') for c in NEGATIVES]
    for case in positive_cases + NEGATIVES:
        arrays, ob, nb = fixture(case)
        for name, a in arrays.items():
            np.save(out / f'{case.name}_{name}.npy', a, allow_pickle=False)
        namespace = case.bad + '__' if case.bad else 'compact__'
        sources['negative.loom' if case.bad else 'positive.loom'].append(check_case(case, arrays, ob, nb, namespace))
        cases.append({**asdict(case),'legacy_arena_bytes':ob,'compact_arena_bytes':nb,
                      'fixture_bytes':sum(a.nbytes for a in arrays.values()),
                      'unique_initialized_scatter':True,'expected':'strict EXPECT/006' if case.bad else 'PASS'})
    for name, chunks in sources.items():
        (out/name).write_text('\n\n'.join(chunks)+'\n')
    hashes = {p.name:digest(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()}
    manifest = dict(legacy_source_sha256=digest(legacy.read_bytes()),compact_source_sha256=digest(candidate.read_bytes()),
                    kernel_pins_sha256=digest(pins_path.read_bytes()),
                    cases=cases,kernels=records,files=hashes,positive_cases=[c.name for c in positive_cases],
                    positive_expectations={c.name: 3 * len(variants(c)) + 3 * int(c.store) for c in positive_cases},
                    negative_specs=['@'+c.name+'=EXPECT/006' for c in NEGATIVES],GPU_executed=False,
                    limitations=['Synthetic norms/QKV only; model A/B must use pinned actual checkpoint',
                                 'No numerical gate or tolerance has been widened from maintained attention checks'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--legacy',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();m=generate(a.legacy,a.candidate,a.out)
    print(json.dumps({'positive':len(m['positive_cases']),'negative':len(NEGATIVES),'fixture_bytes':sum(c['fixture_bytes'] for c in m['cases'])}))
