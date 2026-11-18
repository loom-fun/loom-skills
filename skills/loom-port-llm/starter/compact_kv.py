"""Experimental, default-off Gemma-4 E4B text cache layout.

Only addresses/storage change. The generated kernel arithmetic and launch order
stay intact. Local rings are SDK recurrent slots, not pool-indexed cache planes.
R=2048 preserves a 1024-token epoch plus 511 history and 15 masked aligned loads.
This first prototype deliberately rejects forks, caches, vision and extra rows.
"""
import os
import re
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    ring: int = 2048
    local_bytes: int = 1024
    global_bytes: int = 2048
    local_planes: int = 40
    global_planes: int = 8
    max_chunk: int = 1024
    max_context: int = 65536

    @property
    def slot_bytes(self):
        return self.local_planes * self.ring * self.local_bytes

    @property
    def pool_position_bytes(self):
        return self.global_planes * self.global_bytes


LAYOUT = Layout()


def enabled():
    value = os.environ.get("GEMMA4_COMPACT_KV", "0")
    if value not in ("0", "1"):
        raise ValueError("GEMMA4_COMPACT_KV must be 0 or 1")
    return value == "1"


def validate_config(cfg):
    """Fail before opening checkpoint files or generating incompatible kernels."""
    expected = dict(arch="gemma4", hidden=2560, layers=42, heads=8,
                    kv_heads=2, head_dim=512, head_dim_swa=256, kv_layers=24,
                    kv_shared_from=24, sliding_window=512, vocab=262144)
    errors = [f"{k}={cfg.get(k)!r}, expected {v!r}" for k, v in expected.items() if cfg.get(k) != v]
    globals_ = [5, 11, 17, 23, 29, 35, 41]
    if cfg.get("global_layers") != globals_ or cfg.get("sliding_layers") != [i for i in range(42) if i not in globals_]:
        errors.append("unexpected local/global layer ownership")
    if cfg.get("kv_heads_global", 2) != 2:
        errors.append("global KV heads must be 2")
    if os.path.basename(cfg.get("gguf", "")) != "gemma-4-E4B-it-Q4_K_M.gguf":
        errors.append("only the exact E4B Q4_K_M GGUF configuration is supported")
    if any(cfg.get(k) for k in ("vision", "recurrent_layers", "shortconv_layers", "moe", "mtp")):
        errors.append("text-only dense model required")
    if errors:
        raise ValueError("compact E4B KV: " + "; ".join(errors))


def for_build(cfg, args, draft):
    if not enabled():
        return None
    validate_config(cfg)
    if draft or args.rows != 1 or not 1 <= args.prefill <= LAYOUT.max_chunk:
        raise ValueError("compact E4B KV requires --draft none, --rows 1, --prefill 1..1024")
    if args.context is None or not 1024 <= args.context <= LAYOUT.max_context or args.context % 1024:
        raise ValueError("compact E4B KV requires explicit --context 1024..65536, a multiple of 1024")
    if args.pool is not None and args.pool != args.context:
        raise ValueError("compact E4B KV: SDK fixed pool equals context; --pool must match --context")
    args.pool = args.context
    alternatives = [args.prefill_r9700, cfg.get("prefill_capacity_r9700"), *list((cfg.get("prefill_capacity_device") or {}).values())]
    alternatives += [int(x.partition("=")[2]) for x in args.prefill_device]
    if any(x and not 1 <= x <= LAYOUT.max_chunk for x in alternatives):
        raise ValueError("compact E4B KV: every preset prefill must be <=1024")
    if os.environ.get("LOOM_RDNA4", "1") != "0":
        raise ValueError("compact E4B KV first prototype requires target.rdna4_twins=false (LOOM_RDNA4=0)")
    return LAYOUT


def _once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"compact E4B KV: expected one source anchor {old!r}")
    return text.replace(old, new, 1)


def _helpers(text):
    pattern = r"(?ms)^func.def inline @g_kv_offset\(.*?^}\n"
    match = re.search(pattern, text)
    if not match:
        raise ValueError("compact E4B KV: missing paged address helper")
    original = match.group()
    global_ = original.replace("@g_kv_offset(", "@g_compact_global(")
    global_ = _once(global_, "  %plane_slot = index.madd %plane, %pool, %slot : index", """  %c2_compact = index.constant 2 : index
  %c12_compact = index.constant 12 : index
  %group_compact = index.div %plane, %c12_compact : index
  %which_compact = index.rem %plane, %c2_compact : index
  %plane_compact = index.madd %group_compact, %c2_compact, %which_compact : index
  %plane_slot = index.madd %plane_compact, %pool, %slot : index""")
    local = f"""
// SDK origin record[1] is the writable recurrent-slot origin in the arena.
// Pins=0 and rows=1 are required by prepare AND storage (including resize).
func.def inline @g_compact_local(%origins: buffer, %row: index, %position: index, %blocks: index, %plane: index, %pool: index) -> (offset) {{
  %record_stride = index.constant 40 : offset
  %record_offset = index.scale %row, %record_stride : index, offset -> offset
  %records = buffer.view %origins[%record_offset] : buffer -> view<5xi64>
  %writer64 = view.load %records[1] : view<5xi64> -> i64
  %writer_bound = scalar.assume %writer64 [range(%writer64, 0, 9223372036854775807)] : i64
  %writer = index.cast %writer_bound : i64 to offset
  %c2 = index.constant 2 : index
  %c12 = index.constant 12 : index
  %ring = index.constant {LAYOUT.ring} : index
  %position_bytes = index.constant {LAYOUT.local_bytes} : offset
  %global_before = index.div %plane, %c12 : index
  %removed = index.mul %global_before, %c2 : index
  %local_plane = index.sub %plane, %removed : index
  %within = index.rem %position, %ring : index
  %slot = index.madd %local_plane, %ring, %within : index
  %bytes = index.scale %slot, %position_bytes : index, offset -> offset
  %offset = index.add %writer, %bytes : offset
  func.return %offset : offset
}}
"""
    return text[:match.start()] + global_ + local + text[match.end():]


def _local_kernel(block):
    """Rewrite only the audited address recipes; fail on unknown pool strides.

    Decode uses page-aligned kd offsets; flash2 maps each position separately;
    flash2s uses the explicit V-plane delta. A 16-aligned block cannot cross R.
    """
    block = block.replace("func.call @g_kv_offset(", "func.call @g_compact_local(")
    # Direct K->V plane offsets (rope/store, decode, multi-query, flash2s).
    for stride, pos in (("plane_stride", "position_bytes_v"), ("mt_plane_stride", "mt_pbytes")):
        old = f"%{stride} = index.scale %pool, %{pos} : index, offset -> offset"
        block = block.replace(old, f"%{stride} = index.constant {LAYOUT.ring * LAYOUT.local_bytes} : offset")
        block = block.replace(f"%{pos} = index.constant 2048 : offset", f"%{pos} = index.constant 1024 : offset")
    block = block.replace("%v_plane_delta = index.scale %plane_bytes_i, %pos_bytes : index, offset -> offset",
                          f"%v_plane_delta = index.constant {LAYOUT.ring * LAYOUT.local_bytes} : offset")
    block = block.replace("%pos_bytes = index.constant 2048 : offset", "%pos_bytes = index.constant 1024 : offset")

    def dense(m):
        name, i, value = m.groups()
        if int(value) != int(i) * LAYOUT.global_bytes:
            raise ValueError(f"compact E4B KV: changed key offset recipe {m.group()}")
        return f"%{name}{i} = index.constant {int(i) * LAYOUT.local_bytes} : offset"
    block = re.sub(r"%(m?kd)(\d+) = index.constant (\d+) : offset", dense, block)
    # Segment-address code is not selected by the current E4B path. Reject any
    # future unhandled plane/position arithmetic rather than emitting mixed KV.
    if re.search(r"index.scale %pool,|%plane_stride = index.scale|%mt_plane_stride = index.scale|%v_plane_delta = index.scale", block):
        raise ValueError("compact E4B KV: unhandled direct local plane stride")
    return block


def _validate_local_addresses(block):
    """Every arena view must have an audited cache address, not just a name.

    This closes the failure mode where a new generator recipe uses another SSA
    name for a padded position stride and escapes the known-name replacements.
    """
    definitions = dict(re.findall(r"(?m)^\s*(%\w+) = (.*)$", block))
    views = re.findall(r"buffer.view %arena\[(%\w+)\] : buffer -> view<([^>]+)>", block)
    if not views:
        raise ValueError("compact E4B KV: local kernel has no audited arena views")

    def trace(value, depth=0):
        if depth > 5 or value not in definitions:
            raise ValueError(f"compact E4B KV: unrecognized arena address {value}")
        rhs = definitions[value]
        if rhs.startswith("func.call @g_compact_local("):
            return 0, 0
        if m := re.fullmatch(r"index.add (%\w+), (%\w+) : offset", rhs):
            positions, planes = trace(m[1], depth + 1)
            delta = definitions.get(m[2], "")
            if n := re.fullmatch(r"%m?kd(\d+)", m[2]):
                count = int(n[1])
                if count <= 15 and delta == f"index.constant {count * LAYOUT.local_bytes} : offset":
                    return positions + count, planes
            if m[2] in ("%plane_stride", "%mt_plane_stride", "%v_plane_delta") and delta == f"index.constant {LAYOUT.ring * LAYOUT.local_bytes} : offset":
                return positions, planes + 1
        raise ValueError(f"compact E4B KV: unrecognized arena address recipe {value} = {rhs}")

    for address, shape in views:
        if shape not in ("2x256xf16", "2x32x8xf16"):
            raise ValueError(f"compact E4B KV: unrecognized local arena view {shape}")
        positions, planes = trace(address)
        if positions > 15 or planes > 1:
            raise ValueError("compact E4B KV: local access exceeds aligned key block or V plane")


def specialize(text, cfg):
    if not enabled():
        return text
    validate_config(cfg)
    if json.loads(os.environ.get("LLM_DRAFT") or "null"):
        raise ValueError("compact E4B KV does not support drafting")
    if os.environ.get("LOOM_RDNA4", "1") != "0":
        raise ValueError("compact E4B KV requires LOOM_RDNA4=0")
    # Existing decode recipes depend on <=16, page-aligned contiguous keys.
    for key, default in (("ATTN_UNROLL", "16"), ("FLASH2_KT", "16")):
        if int(os.environ.get(key, default)) != 16:
            raise ValueError(f"compact E4B KV requires {key}=16")
    blocks = re.split(r"(?=^(?:kernel|func|command)\.def)", text, flags=re.M)
    local_count = global_count = 0
    for i, block in enumerate(blocks):
        if "func.call @g_kv_offset" not in block:
            continue
        header = block.splitlines()[0]
        if not header.startswith("kernel.def"):
            raise ValueError(f"compact E4B KV: unclassified address caller {header}")
        if "d512" in header:
            blocks[i] = block.replace("func.call @g_kv_offset(", "func.call @g_compact_global(")
            global_count += 1
        elif "d256" in header or "g_attention_flash2" in header:
            blocks[i] = _local_kernel(block)
            _validate_local_addresses(blocks[i])
            local_count += 1
        else:
            raise ValueError(f"compact E4B KV: unclassified kernel {header}")
    if not local_count or not global_count:
        raise ValueError("compact E4B KV: both attention geometries required")
    text = _helpers("".join(blocks))
    if "@g_kv_offset" in text:
        raise ValueError("compact E4B KV: unhandled legacy helper reference")
    return text


def _guard(rows, pins, chunk, context, prefix, module):
    """VM-side guard; emitted both at prepare and every storage/resize call."""
    p = "%" + prefix
    return f"""  {p}_one = scalar.constant 1 : i64
  {p}_zero = scalar.constant 0 : i64
  {p}_chunk = scalar.constant {LAYOUT.max_chunk} : i64
  {p}_context = scalar.constant {LAYOUT.max_context} : i64
  {p}_rows_ok = scalar.cmpi eq, %{rows}, {p}_one : i64
  {p}_pins_ok = scalar.cmpi eq, %{pins}, {p}_zero : i64
  {p}_chunk_ok = scalar.cmpi ule, %{chunk}, {p}_chunk : i64
  {p}_context_ok = scalar.cmpi ule, %{context}, {p}_context : i64
  {p}_a = scalar.andi {p}_rows_ok, {p}_pins_ok : i1
  {p}_b = scalar.andi {p}_chunk_ok, {p}_context_ok : i1
  {p}_ok = scalar.andi {p}_a, {p}_b : i1
  {p}_ok32 = scalar.extui {p}_ok : i1 to i32
  {p}_error = global.load @compact_kv_{module}_unsupported : buffer
  func.call @require({p}_ok32, {p}_error) : (i32, buffer)
"""


def template(name, text):
    if name not in ("prepare", "storage"):
        return text
    diagnostic = "compact E4B KV requires rows=1, pins=0 (no caches/checkpoints), prefill<=1024, context<=65536"
    if 'func.decl import("input") @require(' not in text:
        text = 'func.decl import("input") @require(%condition: i32, %diagnostic: buffer)\n' + text
    text = f'global.rodata.def @compact_kv_{name}_unsupported = bytes("{diagnostic.encode().hex()}")\n\n' + text
    signature = re.search(rf"(?m)^func.def public @{('prepare' if name == 'prepare' else 'llm_storage')}\(.*\{{\n", text)
    if not signature:
        raise ValueError("compact E4B KV: missing runtime guard insertion point")
    at = signature.end()
    text = text[:at] + _guard("rows", "pins", "prefill" if name == "prepare" else "maximum", "context", "compact", name) + text[at:]
    if name == "prepare":
        # SDK compiles shapes with tokens=spans even rows=1. Preserve that set,
        # but reject a caller-provided epoch larger than the ring proof permits.
        anchor = "    %tag = index.cast %i : index to i64"
        text = _once(text, anchor, _guard("rows", "pins", "tokens", "context", "compact_shape", name) + anchor)
    return text


def card_execution(card):
    for preset in card["presets"].values():
        ex = preset["execution"]
        if ex["rows"] != 1 or ex["prefill_capacity"] > LAYOUT.max_chunk or ex["context_capacity"] > LAYOUT.max_context:
            raise ValueError("compact E4B KV: incompatible execution preset")
        ex.update(max_sessions=1, session_cache_capacity=0, prefix_cache_capacity=0,
                  cache_target_bytes=0, checkpoint_capacity=0)
