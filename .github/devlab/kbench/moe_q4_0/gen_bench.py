#!/usr/bin/env python3
"""Independent Q4_0 expert numerics on Gemma dimensions, no model download.

Generates decode and gathered/grouped prefill cases using F64 NumPy matmuls
and a scalar GGML block decoder, plus sentinel-filled output buffers. Source
GELU transforms are extracted from the current Gemma kit: test what ships.
Usage: gen_bench.py CHECKOUT OUT strix|r9700
"""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np


def dequant_reference(raw, shape):
    """GGML Q4_0 byte layout, independent of the generated Loom helper."""
    blocks = np.asarray(raw, np.uint8).reshape(-1, 18)
    scale = blocks[:, :2].copy().view('<f2').reshape(-1).astype(np.float64)
    payload = blocks[:, 2:]
    values = np.concatenate((payload & 15, payload >> 4), axis=1).astype(np.int16) - 8
    return (scale[:, None] * values).reshape(shape)


def weights(rng, shape):
    blocks = int(np.prod(shape)) // 32
    raw = np.empty((blocks, 18), np.uint8)
    # Deliberately include both scale signs and all code extremes; a nibble
    # swap, unsigned code interpretation or abs(scale) cannot pass.
    scale = (rng.uniform(.0005, .003, blocks) * rng.choice([-1, 1], blocks)).astype('<f2')
    raw[:, :2] = scale.view(np.uint8).reshape(-1, 2)
    raw[:, 2:] = rng.integers(0, 256, (blocks, 16), dtype=np.uint8)
    return raw.reshape(-1).view(np.int8), dequant_reference(raw, shape)


def gelu(x):
    return .5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + .044715 * x**3)))


def load_gelu(kit):
    tree = ast.parse((kit / 'gen_llm.py').read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'gelu_scalar', 'gelu_vectors'}]
    scope = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), '<kit GELU>', 'exec'), scope)
    return scope['gelu_scalar'], scope['gelu_vectors']


def dequant_bits_inputs():
    """Every finite F16 scale, with all 16 codes in each distinct half-block."""
    scale_bits = np.arange(65536, dtype=np.uint16)
    scale_bits = scale_bits[(scale_bits & 0x7c00) != 0x7c00]
    raw = np.empty((len(scale_bits), 18), np.uint8)
    raw[:, :2] = scale_bits.astype('<u2').view(np.uint8).reshape(-1, 2)
    codes = np.arange(16, dtype=np.uint8)
    raw[:, 2:] = codes | ((15 - codes) << 4)
    # F64 represents each finite F16 scale times an integer exactly. Round only
    # once to F16, retaining signed zero, subnormals and overflow infinities.
    with np.errstate(over='ignore'):
        expected = dequant_reference(raw, (len(scale_bits), 32)).astype('<f2')
    return raw.reshape(-1).view(np.int8), expected.reshape(-1).view(np.int16)


def dequant_bits_case(out, target):
    """Actual old/new snippets on GPU, compared as integer bits to CPU truth."""
    import gen_gemm
    import moe_q4_0
    raw, expected = dequant_bits_inputs()
    n = expected.size
    halves = n // 16
    assert halves % 64 == 0
    np.save(out / 'dequant_bits_weights.npy', raw)
    np.save(out / 'dequant_bits_expected.npy', expected)
    gpu = 'gfx12-generic' if target == 'r9700' else 'gfx11-generic'
    kernels = [f'amdgpu.target<{gpu}> @q40_bit_target {{subgroup_size = 32}}\n']
    for name, snippet in [('old', gen_gemm.dequant16('q4_0', 'w', None, 'deq')),
                          ('new', moe_q4_0.dequant16('w', 'deq'))]:
        kernels.append(f'''kernel.def target(@q40_bit_target) @k_q40_bits_{name}() {{
  %one = index.constant 1 : index
  %groups = index.constant {halves // 64} : index
  %threads = index.constant 64 : index
  kernel.launch.config workgroups(%groups, %one, %one) workgroup_size(%threads, %one, %one) : index
}} launch(%w: buffer, %y: buffer) {{
  %c0 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %block_bytes = index.constant 18 : offset
  %origin = index.constant 0 : offset
  %group = kernel.workgroup.id<x> : index
  %thread = kernel.workitem.id<x> : index
  %flat = index.madd %group, %c64, %thread : index
  %half_index = index.assume %flat [range(%flat, 0, {halves - 1})] : index
  %block = index.div %half_index, %c2 : index
  %half = index.rem %half_index, %c2 : index
  %w_row_origin = index.scale %block, %block_bytes : index, offset -> offset
  %w_col = index.mul %half, %c16 : index
{snippet}
  %bits = vector.bitcast %deq : vector<16xf16> to vector<16xi16>
  %yv = buffer.view %y[%origin] : buffer -> view<{n}xi16>
  %yi = index.mul %half_index, %c16 : index
  vector.store %bits, %yv[%yi] : vector<16xi16>, view<{n}xi16>
  kernel.return
}}
''')
    ty, wt = f'tensor<{n}xi16>', f'tensor<{raw.size}xi8>'
    case = f'''check.case public @dequant_all_finite_scale_bits {{
  %w = check.file.read.npy path("dequant_bits_weights.npy") : {wt}
  %expected = check.file.read.npy path("dequant_bits_expected.npy") : {ty}
  %old = check.generate.fill value(32767) : {ty}
  %new = check.generate.fill value(32767) : {ty}
  kernel.launch @k_q40_bits_old(%w, %old) : ({wt}, {ty})
  kernel.launch @k_q40_bits_new(%w, %new) : ({wt}, {ty})
  check.expect.equal actual(%old) expected(%expected) : {ty}
  check.expect.equal actual(%new) expected(%expected) : {ty}
  check.expect.equal actual(%new) expected(%old) : {ty}
  check.return
}}
'''
    return '\n'.join(kernels), case, {'case': 'dequant_all_finite_scale_bits',
        'comparison': 'exact_i16_bits', 'finite_scales': len(raw) // 18,
        'codes_per_half': 16, 'halves': 2, 'output_elements_per_path': n,
        'sentinel_bits': 32767}


def generate(root, out, target):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    st = root / 'skills/loom-port-llm/starter'
    sys.path.insert(0, str(st))
    import gen_gemm
    import gen_moe
    kit = out / 'kit'
    if not kit.exists():
        subprocess.run(['bash', str(root / 'skills/loom-port-llm/reference/apply_example.sh'), 'gemma4', str(kit)], check=True)
    gs, gv = load_gelu(kit)
    os.environ['LOOM_RDNA4'] = '1'
    H, F, E, U, R = 2816, 704, 8, 8, 2
    rng = np.random.default_rng(490)
    wg, gu = weights(rng, (E, 2 * F, H))
    wd, down = weights(rng, (E, H, F))
    kernels = []
    gu_mv = gen_moe.expert_mv('q4_0', H, F, E, U, R, 'gu', fused=True)
    gu_mv = gu_mv.replace('    %silu = scalar.siluf<afn> %t0 : f32\n', gs('silu', 't0', '    ') + '\n')
    kernels += [gu_mv, gen_moe.expert_mv('q4_0', F, H, E, U, R, 'dn')]
    kernels += [gv(gen_moe.grouped_gemm('q4_0', 'swiglu', F, H, E, U, gen_gemm.gemm, fused=True)),
                gen_moe.grouped_gemm('q4_0', 'store', H, F, E, U, gen_gemm.gemm)]
    kernels += [gen_moe.wscale(E, U), gen_moe.combine(H, U)]
    helper = (st / 'quant_values.loom').read_text().split('func.def inline @g_q4_0_values', 1)[1]
    helper = 'func.def inline @g_q4_0_values' + helper.split('\n}', 1)[0] + '\n}\n'
    store = (st / 'gen_llm.py').read_text().split('func.def inline @g_store_row', 1)[1]
    store = 'func.def inline @g_store_row' + store.split('\n}', 1)[0] + '\n}\n'
    pre = 'amdgpu.target<gfx12-generic> @rdna4 {subgroup_size = 32}\n' + helper + store + '\n'.join(kernels)
    pre = pre.replace('g_moe_', 'k_moe_')
    np.save(out / 'wg.npy', wg); np.save(out / 'wd.npy', wd)
    cases = []
    records = []

    def case(name, data, body, yref, atol=2e-5, rtol=2e-4):
        yref = yref.astype(np.float32)
        # A missing write must fail against sentinel 42, independent of input.
        assert np.max(np.abs(yref[yref != 42])) < 10
        data = dict(data, expected=yref)
        reads = []
        types = {}
        for key, a in data.items():
            a = np.asarray(a)
            dtype = {'float32':'f32', 'float16':'f16', 'int32':'i32', 'int8':'i8'}[str(a.dtype)]
            ty = 'tensor<' + 'x'.join(map(str, a.shape)) + 'x' + dtype + '>'
            types[key] = ty
            np.save(out / f'{name}_{key}.npy', a)
            reads.append(f'  %{key} = check.file.read.npy path("{name}_{key}.npy") : {ty}')
        yt = types['expected']
        text = f'check.case public @{name} {{\n' + '\n'.join(reads)
        text += f'\n  %wg = check.file.read.npy path("wg.npy") : tensor<{wg.size}xi8>\n  %wd = check.file.read.npy path("wd.npy") : tensor<{wd.size}xi8>\n'
        text += f'  %y = check.generate.fill value(42.0) : {yt}\n'
        text += body(types, yt)
        text += f'  check.expect.approximate actual(%y) expected(%expected) atol({atol}) rtol({rtol}) nan(different) : {yt}, {yt}\n  check.return\n}}\n'
        cases.append(text)
        records.append({'case':name, 'atol':atol, 'rtol':rtol, 'output_elements':int(yref.size), 'sentinel':42.0})

    for t in [1, 4]:
        ids = np.stack([rng.permutation(E) for _ in range(t)]).astype(np.int32)
        x = rng.standard_normal((t, H)).astype(np.float32)
        act = np.empty((t*U, F), np.float64)
        for a,e in enumerate(ids.flat):
            pair = gu[e] @ x[a // U].astype(np.float64)
            act[a] = gelu(pair[:F]) * pair[F:]
        def body(ty, yt):
            return (f'  %n = check.literal value({F}) : index\n  %a = check.literal value({t*U}) : index\n'
                    f'  kernel.launch @k_moe_gu_q4_0_fused_k{H}_r{R}[%n, %a](%n, %a, %ids, %x, %wg, %wg, %y) : [index, index](index, index, {ty["ids"]}, {ty["x"]}, tensor<{wg.size}xi8>, tensor<{wg.size}xi8>, {yt})\n')
        case(f'decode_gu_t{t}', {'ids':ids,'x':x}, body, act)
        xdn = (rng.standard_normal((t*U,F)) * .1).astype(np.float32)
        ref = np.stack([down[e] @ xdn[a].astype(np.float64) for a,e in enumerate(ids.flat)])
        def body(ty, yt):
            return (f'  %n = check.literal value({H}) : index\n  %a = check.literal value({t*U}) : index\n'
                    f'  kernel.launch @k_moe_dn_q4_0_k{F}_r{R}[%n, %a](%n, %a, %ids, %x, %wd, %y) : [index, index](index, index, {ty["ids"]}, {ty["x"]}, tensor<{wd.size}xi8>, {yt})\n')
        case(f'decode_dn_t{t}', {'ids':ids,'x':xdn}, body, ref)

    suf = '_rdna4' if target == 'r9700' else ''
    # One full tile, tails crossing 16/32-row wave boundaries, and a dead tile.
    for t in [63, 65]:
        counts = [64, 17, 1, 0]
        ti = np.array([[7,64],[0,17],[3,1],[-1,0]],np.int32)
        rm = np.full(256,-1,np.int32)
        for j,n in enumerate(counts):
            rm[j*64:j*64+n] = rng.choice(t*U,n,replace=False)
        x = rng.standard_normal((t,H)).astype(np.float16)
        s = np.ones((t,2),np.float32)
        s[:,1] = rng.choice([.5,1,2],t)
        ref = np.full((256,F),42,np.float64)
        for j,n in enumerate(counts[:-1]):
            ref[j*64:(j+1)*64]=0
            eid=ti[j,0]
            for q in range(n):
                src=rm[j*64+q]//U
                pair=gu[eid].astype(np.float16).astype(np.float64) @ x[src].astype(np.float64) * s[src,1]
                ref[j*64+q]=gelu(pair[:F])*pair[F:]
        def body(ty,yt):
            return (f'  %rows = check.literal value(256) : index\n  %h = check.literal value({H}) : index\n  %f = check.literal value({F}) : index\n  %t = check.literal value({t}) : index\n'
                    f'  kernel.launch @k_moe_ggu_q4_0_fused_k{H}{suf}[%rows, %f](%rows, %h, %f, %t, %ti, %rm, %x, %s, %wg, %wg, %y) : [index, index](index, index, index, index, {ty["ti"]}, {ty["rm"]}, {ty["x"]}, {ty["s"]}, tensor<{wg.size}xi8>, tensor<{wg.size}xi8>, {yt})\n')
        case(f'grouped_gu_t{t}',dict(ti=ti,rm=rm,x=x,s=s),body,ref,5e-5,5e-4)
        xd = (rng.standard_normal((256,F))*.1).astype(np.float16)
        sd = np.ones((256,2),np.float32)
        sd[:,1] = rng.choice([.5,1,2],256)
        refdn = np.full((256,H),42,np.float64)
        for j,n in enumerate(counts[:-1]):
            xd[j*64+n:(j+1)*64]=0
            refdn[j*64:(j+1)*64] = xd[j*64:(j+1)*64].astype(np.float64) @ down[ti[j,0]].astype(np.float16).astype(np.float64).T * sd[j*64:(j+1)*64,1:]
        def body(ty,yt):
            return (f'  %rows = check.literal value(256) : index\n  %h = check.literal value({H}) : index\n  %f = check.literal value({F}) : index\n'
                    f'  kernel.launch @k_moe_gdn_q4_0_k{F}{suf}[%rows, %h](%rows, %f, %h, %rows, %ti, %rm, %x, %s, %wd, %y) : [index, index](index, index, index, index, {ty["ti"]}, {ty["rm"]}, {ty["x"]}, {ty["s"]}, tensor<{wd.size}xi8>, {yt})\n')
        case(f'grouped_dn_t{t}',dict(ti=ti,rm=rm,x=xd,s=sd),body,refdn)

    t=4
    ids = np.stack([rng.permutation(E) for _ in range(t)]).astype(np.int32)
    scales = np.linspace(.25,1.5,E,dtype=np.float32)
    wts = rng.uniform(.01,.2,(t,U)).astype(np.float32)
    ys = rng.standard_normal((t*U,H)).astype(np.float32)*.1
    ref = np.sum(ys.reshape(t,U,H).astype(np.float64)*(wts*scales[ids])[:,:,None],axis=1)
    def body(ty,yt):
        return (f'  %t = check.literal value({t}) : index\n'
                f'  kernel.launch @k_moe_wscale[%t](%t, %ids, %wts, %scales) : [index](index, {ty["ids"]}, {ty["wts"]}, {ty["scales"]})\n'
                f'  kernel.launch @k_moe_combine[%t](%t, %ys, %wts, %y) : [index](index, {ty["ys"]}, {ty["wts"]}, {yt})\n')
    case('expert_down_scale_combine',dict(ids=ids,scales=scales,wts=wts,ys=ys),body,ref)
    bit_kernels, bit_case, bit_record = dequant_bits_case(out, target)
    pre += bit_kernels
    cases.append(bit_case)
    records.append(bit_record)
    text=pre+'\n'.join(cases)
    # Explicit expected-failure controls share the same immutable references:
    # no-op output and swapped nibble halves both must reach EXPECT/006.
    negative = cases[0].replace('@decode_gu_t1', '@negative_no_write')
    negative = '\n'.join(line for line in negative.split('\n') if 'kernel.launch' not in line)
    swapped = pre.replace('%lower = vector.andi %payload, %mask', '%lower_wrong = vector.andi %payload, %mask')
    swapped = swapped.replace('%upper = vector.andi %shifted, %mask', '%lower = vector.andi %shifted, %mask')
    swapped = swapped.replace('%lo = vector.uitofp %lower', '%upper = vector.andi %payload, %mask : vector<16xi8>\n  %lo = vector.uitofp %lower')
    # The unused lower_wrong is intentional: preserve the original helper shape.
    (out/'negative_no_write.loom').write_text(pre + negative)
    (out/'negative_nibbles.loom').write_text(swapped + cases[0].replace('@decode_gu_t1', '@negative_nibble_order'))
    (out/'test.loom').write_text(text)
    # Timings are independent cases; raw profile rows preserve case identity.
    (out/'bench.loom').write_text(text+'\n'+ '\n'.join(f'check.benchmark<@{r["case"]}> @k_case_{r["case"]}' for r in records[:8]))
    (out/'cases.json').write_text(json.dumps({'target':target,'geometry':dict(H=H,F=F,E=E,U=U),'cases':records},indent=2)+'\n')
    return records


if __name__ == '__main__':
    generate(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else 'strix')
