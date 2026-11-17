#!/usr/bin/env python3
"""Generate full producer-chain A/B and exact Q8 ABI checks, no weights needed."""
import re
import sys
from pathlib import Path
import numpy as np

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(ROOT / 'skills/loom-port-llm/starter'))
import gen_norm_q8
SOURCE = (ROOT / 'skills/loom-port-llm/starter/gen_llm.py').read_text()


def sources(hidden):
    start = SOURCE.index('kernel.def export("g_post_add_norm")')
    stop = SOURCE.index('\n"""', start)
    norm = eval('f\'\'\'' + SOURCE[start:stop] + '\'\'\'',
                dict(HG=hidden // 32, H=hidden, POST_NORM=False, POST_EPS=1e-6, EPS=1e-6))
    start = SOURCE.index('kernel.def export("g_quant_q8")')
    quant = SOURCE[start:SOURCE.index('\n"""', start)]
    return norm + "\n", quant + "\n"


def launches(h, t, fused, suffix=''):
    n = h * t
    types = f'index, tensor<{n}xf32>, tensor<{h}xf32>, tensor<{n}xf32>, tensor<{h}xf32>, tensor<{n}xf32>'
    if fused:
        return f'  kernel.launch @k_fused[%t](%t, %y, %wp, %x{suffix}, %wn, %h{suffix}, %q{suffix}) : [index]({types}, tensor<{n * 5 // 4}xi8>)\n'
    return (f'  kernel.launch @k_norm[%t](%t, %y, %wp, %x{suffix}, %wn, %h{suffix}) : [index]({types})\n'
            f'  kernel.launch @k_quant[%t, %width](%t, %width, %h{suffix}, %q{suffix}) : [index, index](index, index, tensor<{n}xf32>, tensor<{n * 5 // 4}xi8>)\n')


def module(hidden):
    norm, quant = sources(hidden)
    return (norm + quant + gen_norm_q8.kernel(norm, quant, hidden)).replace('@g_post_add_norm_q8', '@k_fused').replace('export("g_post_add_norm_q8")', 'export("k_fused")').replace('@g_post_add_norm', '@k_norm').replace('export("g_post_add_norm")', 'export("k_norm")').replace('@g_quant_q8', '@k_quant').replace('export("g_quant_q8")', 'export("k_quant")')


def case(h, rows, fused, test=False, zeros=False):
    rng = np.random.default_rng(h + rows)
    n = h * rows
    qn = n * 5 // 4
    name = f'h{h}_t{rows}' + ('_zero' if zeros else '')
    OUT.mkdir(parents=True, exist_ok=True)
    for k, shape in [('y', (rows, h)), ('x', (rows, h)), ('wn', (h,))]:
        data = np.zeros(shape, np.float32) if zeros and k != 'wn' else rng.uniform(-1, 1, shape).astype(np.float32)
        np.save(OUT / (name + '_' + k + '.npy'), data)
    x, y, wn = [np.load(OUT / (name + '_' + k + '.npy')) for k in ('x', 'y', 'wn')]
    residual = x + y
    expected = residual / np.sqrt(np.mean(residual.astype(np.float64)**2, axis=1, keepdims=True) + 1e-6) * wn
    np.save(OUT / (name + '_expected.npy'), expected.astype(np.float32))
    text = f'''check.case public @{name} {{
  %t = check.literal value({rows}) : index
  %width = check.literal value({h}) : index
  %y = check.file.read.npy path("{name}_y.npy") : tensor<{n}xf32>
  %x = check.file.read.npy path("{name}_x.npy") : tensor<{n}xf32>
  %wn = check.file.read.npy path("{name}_wn.npy") : tensor<{h}xf32>
  %wp = check.generate.fill value(1.0) : tensor<{h}xf32>
  %h = check.generate.fill value(0.0) : tensor<{n}xf32>
  %q = check.generate.fill value(0) : tensor<{qn}xi8>
'''
    # File reads use flattened data because the kernel's ABI is raw buffers.
    for k in ('x', 'y', 'expected'):
        a = np.load(OUT / (name + '_' + k + '.npy'))
        np.save(OUT / (name + '_' + k + '.npy'), a.reshape(-1))
    if test:
        text += f'''  %x2 = check.file.read.npy path("{name}_x.npy") : tensor<{n}xf32>
  %h2 = check.generate.fill value(0.0) : tensor<{n}xf32>
  %q2 = check.generate.fill value(0) : tensor<{qn}xi8>
  %expected = check.file.read.npy path("{name}_expected.npy") : tensor<{n}xf32>
'''
        text += launches(h, rows, False) + launches(h, rows, True, '2')
        text += f'''  check.expect.equal actual(%x2) expected(%x) : tensor<{n}xf32>
  check.expect.equal actual(%h2) expected(%h) : tensor<{n}xf32>
  check.expect.equal actual(%q2) expected(%q) : tensor<{qn}xi8>
  check.expect.approximate actual(%h2) expected(%expected) atol(2.0e-6) rtol(2.0e-6) nan(different) : tensor<{n}xf32>, tensor<{n}xf32>
'''
    else:
        text += launches(h, rows, fused)
    return text + '  check.return\n}\n' + ('' if test else f'check.benchmark<@{name}> @{name}_bm\n')


def main():
    # Width-specific norms have equal names, so use separate bench modules and
    # correctness files; no linker redefinitions or timing across devices.
    for h in (1024, 2048, 2560, 5120):
        kernels = module(h)
        for fused in (False, True):
            s = kernels + '\n'.join(case(h, t, fused) for t in (1, 4, 63))
            (OUT / f'bench_h{h}_{"fused" if fused else "base"}.loom').write_text(s)
        tests = '\n'.join(case(h, t, True, test=True, zeros=z) for t, z in ((1,False),(4,False),(63,False),(1,True)))
        (OUT / f'test_h{h}.loom').write_text(kernels + tests)


if __name__ == '__main__':
    main()
