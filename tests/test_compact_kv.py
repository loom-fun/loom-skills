"""CPU guards and independent execution of emitted compact cache addresses.

The optional generated-program proof reads header-only build output, never model
weights. It interprets the emitted address IR instead of calling a Python copy of
the layout formula. Float64 attention over F16 synthetic K/V is an independent
oracle; this does not substitute for GPU kernel/model numerical validation.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "skills/loom-port-llm/starter"
spec = importlib.util.spec_from_file_location("compact_kv", STARTER / "compact_kv.py")
compact = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compact
spec.loader.exec_module(compact)


def fixture():
    cfg = json.loads((ROOT / "fleet/unsloth-gemma-4-E4B-it-loom/llm_config.json").read_text())
    for k in ("vision", "assistant", "drafter", "prefill_capacity_device", "prefill_capacity_r9700"):
        cfg.pop(k, None)
    return cfg


def address_function(text, name):
    m = re.search(rf"(?ms)^func.def inline @{name}\(.*?^}}", text)
    if not m:
        raise AssertionError(f"missing emitted function {name}")
    return m.group()


def interpret(function, args, memory):
    """Small independent evaluator for the emitted straight-line address IR.

    Byte-addressed memory is populated as SDK records/page maps by the caller.
    Unknown instructions fail, so this cannot silently ignore changed addressing.
    """
    names = re.findall(r"(%\w+): (?:buffer|index)", function.splitlines()[0])
    env = dict(zip(names, args))
    for line in function.splitlines()[1:]:
        line = line.strip()
        if not line or line == "}" or line.startswith("//"):
            continue
        if line.startswith("func.return "):
            return env[line.split()[1]]
        dest, rhs = line.split(" = ", 1)
        vals = re.findall(r"%\w+", rhs)
        if m := re.match(r"(?:index|scalar).constant (-?\d+)", rhs):
            env[dest] = int(m[1])
        elif re.match(r"(?:index|scalar).(?:cast|assume) ", rhs):
            env[dest] = env[vals[0]]
        elif m := re.match(r"index.(add|sub|mul|div|rem|madd|scale) ", rhs):
            v = [env[x] for x in vals]
            env[dest] = {"add": lambda: v[0] + v[1], "sub": lambda: v[0] - v[1],
                         "mul": lambda: v[0] * v[1], "div": lambda: v[0] // v[1],
                         "rem": lambda: v[0] % v[1], "madd": lambda: v[0] * v[1] + v[2],
                         "scale": lambda: v[0] * v[1]}[m[1]]()
        elif rhs.startswith("buffer.view "):
            element = int(re.search(r"xi(\d+)>$", rhs)[1]) // 8
            env[dest] = (env[vals[0]] + env[vals[1]], element)
        elif m := re.match(r"view.load (%\w+)\[(%\w+|\d+)\]", rhs):
            base, width = env[m[1]]
            index = env[m[2]] if m[2].startswith("%") else int(m[2])
            env[dest] = memory[base + width * index]
        else:
            raise AssertionError(f"unsupported emitted address instruction {line}")
    raise AssertionError("no function return")


def origins(pool=65536, writer=None):
    # Nonzero origin and shuffled page map catch accidental direct logical-KV
    # addressing, wrong origin-record field and omitted private-region origin.
    rng = random.Random(81)
    pages = list(range(pool // 16))
    rng.shuffle(pages)
    writer = writer if writer is not None else 8 * pool * 2048 + 4608
    memory = {0: writer, 8: writer, 16: 0, 24: 0, 32: 40}
    memory.update({40 + i * 4: p for i, p in enumerate(pages)})
    return memory, pages, writer


def f16(value):
    return struct.unpack("e", struct.pack("e", value))[0]


def payload(position, dim, value=False):
    # Hash-like finite F16 inputs, independent of the address layout.
    seed = (position * 1103515245 + dim * 12345 + (7919 if value else 0)) & 65535
    return f16((seed - 32768) / 32768)


def attention(query, keys, values):
    scores = [math.fsum(x * y for x, y in zip(query, key)) for key in keys]
    maximum = max(scores)
    exp = [math.exp(s - maximum) for s in scores]
    total = math.fsum(exp)
    return [math.fsum(w * value[d] for w, value in zip(exp, values)) / total for d in range(len(query))]


class CompactTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"GEMMA4_COMPACT_KV": "1", "LOOM_RDNA4": "0", "LLM_DRAFT": "null"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def args(self, **kw):
        base = dict(rows=1, prefill=1024, context=65536, pool=None, prefill_r9700=None, prefill_device=[])
        base.update(kw)
        return SimpleNamespace(**base)

    def test_build_rejects_unsupported_before_checkpoint_access(self):
        cfg = fixture()
        self.assertEqual(compact.for_build(cfg, self.args(), None).slot_bytes, 83886080)
        for k, v in (("rows", 2), ("prefill", 1025), ("context", 65537), ("context", 1025), ("context", None), ("pool", 131072), ("prefill_r9700", 2048)):
            with self.subTest(k=k, v=v), self.assertRaises(ValueError):
                compact.for_build(cfg, self.args(**{k: v}), None)
        with self.assertRaises(ValueError):
            compact.for_build(cfg, self.args(), {"kind": "ngram"})
        for k, v in (("head_dim_swa", 128), ("kv_layers", 42), ("vision", {"mmproj": "never-read"}),
                     ("gguf", "some-other-Q4_K_M.gguf"), ("global_layers", [5, 11, 17, 24])):
            with self.subTest(k=k), self.assertRaises(ValueError):
                compact.validate_config(dict(cfg, **{k: v}))

    def test_default_is_exact_identity_even_for_unrelated_models(self):
        with patch.dict(os.environ, {"GEMMA4_COMPACT_KV": "0"}):
            text = "unrelated model source\n"
            self.assertEqual(compact.specialize(text, {}), text)
            self.assertIsNone(compact.for_build({}, None, None))

    def test_runtime_guards_cover_prepare_shapes_and_resize(self):
        for name in ("prepare", "storage"):
            text = compact.template(name, (STARTER / f"templates/{name}.loom").read_text())
            self.assertIn("%compact_rows_ok = scalar.cmpi eq, %rows, %compact_one", text)
            self.assertIn("%compact_pins_ok = scalar.cmpi eq, %pins, %compact_zero", text)
            self.assertIn('func.decl import("input") @require(', text)
            self.assertNotIn('@compact_kv_require', text)
            if name == "prepare":
                self.assertLess(text.index("%compact_shape_chunk_ok"), text.index("%stage = func.call @stage"))
            else:
                self.assertLess(text.index("%compact_rows_ok"), text.index("%bounded_rows = scalar.assume"))

    def test_unknown_stride_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unhandled direct local"):
            compact._local_kernel("%new_stride = index.scale %pool, %new_bytes : index, offset -> offset")
        renamed = """%base = func.call @g_compact_local(%origins, %row, %pos, %blocks, %plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
%renamed_stride = index.constant 2048 : offset
%new_address = index.add %base, %renamed_stride : offset
%cache = buffer.view %arena[%new_address] : buffer -> view<2x256xf16>
"""
        with self.assertRaisesRegex(ValueError, "unrecognized arena address recipe"):
            compact._validate_local_addresses(renamed)

    def test_arena_user_without_helper_is_rejected(self):
        source = """kernel.def export("gd256_attention_split") @gd256_attention_split() {
  %kb_off = index.constant 0 : offset
  %cache = buffer.view %arena[%kb_off] : buffer -> view<2x256xf16>
}
"""
        with self.assertRaisesRegex(ValueError, "unaudited arena user"):
            compact.specialize(source, fixture())

    def test_local_flash2s_v_plane_and_decode_offsets(self):
        sample = """%pos_bytes = index.constant 2048 : offset
%v_plane_delta = index.scale %plane_bytes_i, %pos_bytes : index, offset -> offset
%kd15 = index.constant 30720 : offset
%mkd7 = index.constant 14336 : offset
"""
        got = compact._local_kernel(sample)
        self.assertIn("%v_plane_delta = index.constant 2097152 : offset", got)
        self.assertIn("%kd15 = index.constant 15360 : offset", got)
        self.assertIn("%mkd7 = index.constant 7168 : offset", got)
        with self.assertRaises(ValueError):
            compact._local_kernel(sample.replace("30720", "123"))


def prove_generated(legacy, candidate):
    old, new = Path(legacy).read_text(), Path(candidate).read_text()
    # Review regression: remove the sole helper call from the REAL generated
    # decode kernel. Its direct V/kd offsets must not bypass the block audit.
    real_kernel = next(b for b in re.split(r"(?=^(?:kernel|func|command)\.def)", old, flags=re.M)
                       if re.match(r'kernel.def export\("gd256_attention_split"\)', b))
    mutated, count = re.subn(r"func.call @g_kv_offset\([^\n]+", "index.constant 0 : offset", real_kernel)
    assert count == 1
    with patch.dict(os.environ, {"GEMMA4_COMPACT_KV": "1", "LOOM_RDNA4": "0", "LLM_DRAFT": "null"}):
        assert compact.specialize(old, fixture()) == new, "defensive audit changed generated program"
        try:
            compact.specialize(old.replace(real_kernel, mutated), fixture())
        except ValueError as error:
            assert "unaudited arena user" in str(error)
        else:
            raise AssertionError("real no-helper cache kernel escaped specialization")
    local = address_function(new, "g_compact_local")
    global_ = address_function(new, "g_compact_global")
    legacy_fn = address_function(old, "g_kv_offset")
    # The generation change must leave every floating-point/reduction/load/store
    # instruction and every command/launch byte unchanged; only integer address
    # functions and stride expressions may differ.
    numeric = re.compile(r"(?m)^.*(?:scalar\.(?:f\w+|[a-z]+f)|vector\.|view\.(?:load|store)|kernel\.launch|command\.).*$")
    old_without_helper = old.replace(legacy_fn, "")
    new_without_helpers = new.replace(local, "").replace(global_, "")
    assert numeric.findall(old_without_helper) == numeric.findall(new_without_helpers), "numerical/launch instruction changed"
    memory, pages, writer = origins()
    globals_ = [5, 11, 17, 23]
    locals_ = [i for i in range(24) if i not in globals_]
    for layer in range(42):
        global_layer = layer in [5, 11, 17, 23, 29, 35, 41]
        owner = layer if layer < 24 else 23 if global_layer else 22
        assert re.search(rf"%layer_{layer} = index.constant {owner} : index", new)
        assert re.search(rf"%store_{layer} = scalar.constant {int(layer < 24)} : i32", new)
    checks = 0
    # Explicit owner tables are independent of the emitted plane arithmetic.
    for owners, fn, stride in ((globals_, global_, 2048), (locals_, local, 1024)):
        for owner_ordinal, owner in enumerate(owners):
            for kv in (0, 1):
                for position in [0, 1, 15, 16, 511, 512, 1023, 1024, 1535, 1536, 2047, 2048, 2049, 65534, 65535]:
                    args = [0, 0, position, 4096, owner * 2 + kv, 65536]
                    got = interpret(fn, args, memory)
                    if owners is globals_:
                        physical_position = pages[position // 16] * 16 + position % 16
                        want = ((owner_ordinal * 2 + kv) * 65536 + physical_position) * stride
                    else:
                        want = writer + ((owner_ordinal * 2 + kv) * 2048 + position % 2048) * stride
                    assert got == want, (owner, position, kv, got, want)
                    # Every lane of both heads remains inside this position.
                    assert got + 2 * 2 * (512 if owners is globals_ else 256) <= want + stride
                    assert interpret(legacy_fn, args, memory) == ((owner * 2 + kv) * 65536 + pages[position // 16] * 16 + position % 16) * 2048
                    checks += 1
    blocks = re.split(r"(?=^(?:kernel|func|command)\.def)", new, flags=re.M)
    audit = []
    for b in blocks:
        if "func.call @g_compact_local(" not in b:
            continue
        name = b.splitlines()[0]
        assert "index.scale %pool," not in b
        for _, n, v in re.findall(r"%(m?kd)(\d+) = index.constant (\d+) : offset", b):
            assert int(v) == 1024 * int(n)
        for v in re.findall(r"%(?:plane_stride|mt_plane_stride|v_plane_delta) = index.constant (\d+) : offset", b):
            assert int(v) == 2048 * 1024
        audit.append(name)
    assert audit and "func.call @g_kv_offset(" not in new

    # Numerically compare retained F16 values read through EMITTED IR against
    # full history. Irregular chunk starts exercise both ring sides, shared owner
    # 22 and full 64k positions. Every chunk writes all K/V before any query.
    cases = [(0, 1024), (511, 1024), (1025, 1024), (1537, 1024), (2047, 2), (65024, 512)]
    numerical = query_checks = 0
    for start, count in cases:
        ring_memory = {}
        lo = max(0, start - 526)  # includes every masked 16-aligned history lane
        end = start + count
        # Store all full-history data needed for this epoch, oldest first, then
        # simulate arbitrary producer write order for the new chunk.
        writes = list(range(lo, start))
        chunk = list(range(start, end)); random.Random(start).shuffle(chunk)
        writes += chunk
        for pos in writes:
            for kv in (0, 1):
                address = interpret(local, [0, 0, pos, 4096, 44 + kv, 65536], memory)
                ring_memory[address] = (pos, [payload(pos, d, kv == 1) for d in range(8)])
        read_addresses = {p: interpret(local, [0, 0, p, 4096, 44, 65536], memory) for p in range(lo, end)}
        for qpos in range(start, end):
            qlo = max(0, qpos - 511)
            for p in range(qlo, qpos + 1):
                # Use sparse boundary queries for floating point work; identity
                # on every query separately catches earlier-Q overwrite.
                address = read_addresses[p]
                assert ring_memory[address][0] == p
            query_checks += 1
        for qpos in sorted(set([start, min(start + 1, end - 1), end - 1])):
            positions = list(range(max(0, qpos - 511), qpos + 1))
            query = [payload(qpos, d + 17) for d in range(8)]
            keys = [ring_memory[interpret(local, [0, 0, p, 4096, 44, 65536], memory)][1] for p in positions]
            values = [ring_memory[interpret(local, [0, 0, p, 4096, 45, 65536], memory)][1] for p in positions]
            actual = attention(query, keys, values)
            expected = attention(query, [[payload(p, d) for d in range(8)] for p in positions],
                                 [[payload(p, d, True) for d in range(8)] for p in positions])
            assert actual == expected
            numerical += 1
        # Historical masked lanes must preserve finite old values as well.
        for p in range(max(0, start - 511) // 16 * 16, start):
            address = interpret(local, [0, 0, p, 4096, 44, 65536], memory)
            assert ring_memory[address][0] == p

    negatives = {}
    for capacity in (512, 1024):
        broken = local.replace("%ring = index.constant 2048", f"%ring = index.constant {capacity}")
        seen = {}
        for p in range(513, 2048):
            seen[interpret(broken, [0, 0, p, 4096, 44, 65536], memory)] = p
        addr = interpret(broken, [0, 0, 513, 4096, 44, 65536], memory)
        negatives[f"undersized_ring_{capacity}"] = seen[addr] != 513
    for capacity, first, end, label in ((512, 1536, 2049, "two_query_overwrite"), (1536, 512, 2049, "masked_historical_lane")):
        broken = local.replace("%ring = index.constant 2048", f"%ring = index.constant {capacity}")
        seen = {interpret(broken, [0, 0, p, 4096, 44, 65536], memory): p for p in range(first, end)}
        negatives[label] = seen[interpret(broken, [0, 0, first, 4096, 44, 65536], memory)] != first
    k = interpret(local, [0, 0, 2047, 4096, 44, 65536], memory)
    v = interpret(local, [0, 0, 2047, 4096, 45, 65536], memory)
    negatives["legacy_v_stride"] = k + 65536 * 2048 != v
    other, _, _ = origins(writer=writer + 90 * 2**20)
    negatives["missing_fork_history_copy"] = interpret(local, [0, 0, 2047, 4096, 44, 65536], other) != k
    assert all(negatives.values())
    return dict(status="PASS", real_no_helper_mutation_rejected=True,
                defensive_audit_generated_bytes_unchanged=True,
                address_cases=checks, layer_owner_and_store_checks=42,
                numerical_and_launch_source_identical=True, local_kernel_audit=audit, query_identity_checks=query_checks,
                f64_attention_cases=numerical, negative_controls=negatives,
                limitation="CPU emitted-address and synthetic F16/f64 oracle only; no GPU kernel or model execution")


if __name__ == "__main__":
    if "--legacy" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--legacy", required=True)
        ap.add_argument("--candidate", required=True)
        ap.add_argument("--output", required=True)
        args = ap.parse_args()
        result = prove_generated(args.legacy, args.candidate)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    else:
        unittest.main()
