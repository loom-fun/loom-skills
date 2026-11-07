"""Builds a loom-serve package for a dense GGUF LLM from its LLM_CONFIG json.

usage: python build.py CONFIG.json PACKAGE_DIR --tokenizer TOKENIZER.json
           [--context N] [--pool N] [--memory-gib auto] [--prefill 1024] [--rows 4] [--max-tokens 4096] [--model-id ID]

--context defaults to the model's native context (cfg context_length, up to the kernels' 262144) and
--pool (KV positions shared by all rows) to context + max(64k, context/4), both reduced to fit --memory-gib.

Writes PACKAGE_DIR/{model.json, weights.loom, tokenizer.json, program/*.loom,
parameter/<gguf> (symlink)}. Serve it with
  loom-serve --model=PACKAGE_DIR/model.json --device=amdgpu --memory_bytes=... --listen=...
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("package")
ap.add_argument("--tokenizer", required=True, help="HF tokenizer.json of the same model (download from the base repo)")
ap.add_argument("--context", type=int, default=None, help="max tokens per sequence (default: the model's native context)")
ap.add_argument("--pool", type=int, default=None, help="KV positions shared by all rows and the prefix cache (default: context + max(64k, context/4): one full-context request plus headroom for other rows' history)")
ap.add_argument("--memory-gib", default="auto",
                help="device memory budget for weights + KV pool + workspace. auto (default): enough for a KV pool of "
                     "the default pool, capped at 60%% of physical RAM (GPU memory is system RAM on Strix Halo; the "
                     "benchmarks load one model at a time); 56 when RAM can't be read")
ap.add_argument("--prefill", type=int, default=None,
                help="prefill chunk (prefill_capacity), 1024-4096; default cfg['prefill_capacity'], else 4096 for MoE "
                     "(fuller expert tiles: Flint 2k/8k/32k +22/+38/+28 %% vs 1024), else 1024")
ap.add_argument("--rows", type=int, default=4, help="initial resident rows (grows on demand); c4 benchmarks need >= 4")
ap.add_argument("--max-tokens", type=int, default=4096, help="default completion length when a request sets none")
ap.add_argument("--temperature", type=float, default=0.0, help="default sampling temperature when a request sets none (0 = greedy)")
ap.add_argument("--top-p", type=float, default=1.0, help="default top_p when a request sets none")
ap.add_argument("--top-k", type=int, default=0, help="default top_k when a request sets none (0 = up to 64 candidates)")
ap.add_argument("--drafter", default=os.environ.get("DRAFTER"), help="DFlash/DFlash2 drafter GGUF (arch dflash) for --draft dflash/auto")
ap.add_argument("--draft", default=os.environ.get("DRAFT", "auto"), choices=["auto", "mtp", "dflash", "ngram", "assist", "none"],
                help="speculative decoding drafter. auto (default): draft/tune.py's measured choice, else the GGUF's MTP/NextN "
                     "block, else a DFlash drafter, else a Gemma-4 assistant; assist: Gemma-4 assistant GGUF (cfg assistant); ngram: prompt lookup in the row's own history (any model, no drafter "
                     "weights); none: ordinary decoding (also SPEC=off). Draft settings are tuned by draft/tune.py (draft/README.md)")
ap.add_argument("--draft-depth", type=int, default=int(os.environ.get("DRAFT_DEPTH", "0")) or None,
                help="proposed tokens per verification step (1-7)")
ap.add_argument("--model-id", default=None, help="model id clients send (visible ASCII, no spaces); default: slug of cfg name")
a = ap.parse_args()

cfg = json.load(open(a.config))
if a.prefill is None:
    a.prefill = cfg.get("prefill_capacity") or (4096 if cfg.get("moe") else 1024)
chat = cfg.get("chat")
if not chat:
    sys.exit("config has no 'chat' block: fill it from the model's chat template first (chat_probe.py)")
missing = [k for k in ("generation_prompt_plain", "call_format", "tool_item_prefix", "history_think_scope") if k not in chat]
if missing:
    sys.exit(f"chat block lacks {missing}: regenerate it with inspect_gguf.py / chat_probe.py")
KERNEL_CONTEXT, KERNEL_POOL = 262144, 1048576
# Speculative decoding is on by default when the checkpoint carries a drafter (draft/README.md).
if os.environ.get("SPEC", "").lower() in ("0", "off", "no", "false", "none"):
    a.draft = "none"
a.drafter = a.drafter or (cfg.get("drafter") or {}).get("gguf")
if a.draft == "auto":   # the optimizer's measured choice (cfg["drafter"]["kind"]) wins; else MTP, else DFlash
    a.draft = (cfg.get("drafter") or {}).get("kind") or ("mtp" if cfg.get("mtp") else ("dflash" if a.drafter else
                                                        ("assist" if (cfg.get("assistant") or {}).get("gguf") else "none")))
if a.draft == "mtp" and not cfg.get("mtp"):
    sys.exit("--draft mtp: this GGUF has no NextN/MTP block (cfg.json has no 'mtp')")
if a.draft == "dflash" and not a.drafter:
    sys.exit("--draft dflash needs the drafter GGUF (--drafter PATH or cfg.json drafter.gguf)")
_tuned = cfg.get("drafter") or {}
_depth = a.draft_depth or (_tuned.get("depth") if _tuned.get("kind") == a.draft else None) or 3   # draft/tune.py's choice
DRAFT = {"kind": a.draft, "depth": max(1, min(7, _depth))} if a.draft != "none" else None
if DRAFT and DRAFT["kind"] == "dflash":
    sys.path.insert(0, os.path.join(HERE, "draft"))
    import gen_dflash  # noqa: E402
    _d = gen_dflash.config(a.drafter)
    DRAFT.update(gguf=os.path.abspath(a.drafter), layers=_d["layers"], kv_heads=_d["kv_heads"], head_dim=_d["head_dim"],
                 taps=len(_d["target_layers"]), block_size=_d["block_size"])
    # Untuned: depth 3 (at most block - 1). Verifying up to 4 tokens per span keeps the 4-token matvec, the same
    # summation as 1-token decode, so port.sh's spec smoke gate is exact; deeper verification takes the f16 WMMA
    # GEMM, whose rounding can flip near-ties (MoE routing amplifies them). draft/tune.py tries 3, 5 and 7.
    if not a.draft_depth and not (_tuned.get("kind") == "dflash" and _tuned.get("depth")):
        DRAFT["depth"] = min(3, _d["block_size"] - 1)
pos_bytes = cfg["kv_heads"] * cfg["head_dim"] * 2
kv_layers = cfg.get("kv_layers", cfg["layers"])
kv_per_position = pos_bytes * 2 * kv_layers
if DRAFT and DRAFT["kind"] == "mtp":   # the MTP block's K/V shares the pool's pages
    kv_per_position += pos_bytes * 2 * len(cfg["mtp"]["layers"])
if DRAFT and DRAFT["kind"] == "dflash":   # so does the drafter's context K/V
    kv_per_position += DRAFT["kv_heads"] * DRAFT["head_dim"] * 2 * 2 * DRAFT["layers"]
if DRAFT and DRAFT["kind"] == "ngram":    # and the n-gram drafter's token history (one i32 per position)
    kv_per_position += 4
if DRAFT and DRAFT["kind"] == "assist":   # the Gemma-4 assistant reads the target's K/V (a 4-byte placeholder plane)
    if not (cfg.get("assistant") or {}).get("gguf"):
        sys.exit("--draft assist needs cfg.json assistant.gguf (find_drafter.py records it for gemma4 targets)")
    DRAFT["gguf"] = os.path.abspath(cfg["assistant"]["gguf"])
    kv_per_position += 4
weights_bytes = os.path.getsize(os.path.abspath(cfg["gguf"]))
if DRAFT and DRAFT["kind"] in ("dflash", "assist"):
    weights_bytes += os.path.getsize(DRAFT["gguf"])
if os.path.isfile((cfg.get("vision") or {}).get("mmproj") or ""):   # the image component's projector, while it encodes
    weights_bytes += os.path.getsize(cfg["vision"]["mmproj"])
# Hybrid models keep one recurrent slot (all gated delta-net layer states) per resident row, x2 for forks.
sys.path.insert(0, HERE)
from gen_gdn import gdn_geometry  # noqa: E402
recurrent_bytes = gdn_geometry(cfg)["slot_bytes"] if cfg.get("recurrent_layers") else 0
if cfg.get("shortconv_layers"):  # LFM2 short-conv history lives in the recurrent slots too (reference/lfm2moe-example)
    from gen_sconv import sconv_geometry  # noqa: E402
    recurrent_bytes += sconv_geometry(cfg)["slot_bytes"]
def program_scratch(path, tokens, spans):
    """Bytes of scratch (buffer.alloca<global>) of the largest command program in an llm.loom, for an invocation of
    `tokens` tokens in `spans` spans. loom-serve charges the largest program's workspace against --memory_bytes at
    every admission, next to the weights and the request's KV pages. Allocations whose size this can't fold count 0."""
    config = {"tokens": tokens, "spans": spans, "epoch_tokens": tokens, "epoch_spans": spans, "width": 8,
              "context": KERNEL_CONTEXT, "pool": KERNEL_POOL}
    ops = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y, "scale": lambda x, y: x * y,
           "div": lambda x, y: x // y, "rem": lambda x, y: x % y, "min": min, "max": max,
           "uge": lambda x, y: x >= y, "ugt": lambda x, y: x > y, "ult": lambda x, y: x < y, "ule": lambda x, y: x <= y,
           "eq": lambda x, y: x == y, "ne": lambda x, y: x != y}
    expr = re.compile(r"\s+(%\w+) = (?:config\.get @llm\.(\w+)|(?:index|scalar)\.constant (-?\d+) "
                      r"|index\.(?:cmp )?(\w+),? (%\w+), (%\w+) |scf\.select (%\w+), (%\w+), (%\w+) "
                      r"|buffer\.alloca<global> align\(\d+\) (%\w+) )")
    best, total, env = 0, None, {}
    v = env.get
    for line in open(path):
        if line.startswith("command.program.def"):
            total = 0
            env.clear()
        elif line.startswith("}"):
            best, total = max(best, total or 0), None
        elif total is not None and (m := expr.match(line)):
            ssa, key, const, op, x, y, c, xt, xf, size = m.groups()
            if key:
                env[ssa] = config.get(key)
            elif const:
                env[ssa] = int(const)
            elif op:
                try:
                    env[ssa] = int(ops[op](v(x), v(y)))
                except (KeyError, TypeError, ZeroDivisionError):
                    env[ssa] = None
            elif c:
                env[ssa] = (v(xt) if v(c) else v(xf)) if v(c) is not None else max(v(xt) or 0, v(xf) or 0)
            else:
                total += v(size) or 0
    return max(best, total or 0)


pkg = os.path.abspath(a.package)
os.makedirs(os.path.join(pkg, "program"), exist_ok=True)
os.makedirs(os.path.join(pkg, "parameter"), exist_ok=True)

# 1. Kernel library + command program (first: its scratch counts in the memory budget below).
gen = os.path.join(pkg, "gen")
env = dict(os.environ, LLM_CONFIG=os.path.abspath(a.config), LLM_OUT=gen, LLM_TOKENIZER=os.path.abspath(a.tokenizer),
           LLM_DRAFT=json.dumps(DRAFT))
sys.path.insert(0, os.path.join(HERE, "draft"))
import build_draft  # noqa: E402
subprocess.run([sys.executable, os.path.join(HERE, "gen_llm.py"), os.path.abspath(a.config)], check=True, env=env, stdout=subprocess.DEVNULL)
os.replace(os.path.join(gen, "model.loom"), os.path.join(pkg, "program", "llm.loom"))
# Load-time weight policies of drafting (draft/prep_weights.py): a BF16/F16 drafter as Q8_0, the proposals' Q4_0 head.
POLICIES = []
for f, k in (("draft_weights.loom", "policy"), ("head_weights.loom", "head_policy")):
    if os.path.exists(os.path.join(pkg, f)):
        os.remove(os.path.join(pkg, f))
    if os.path.exists(os.path.join(gen, f)):
        os.replace(os.path.join(gen, f), os.path.join(pkg, f))
        DRAFT[k] = f
        POLICIES.append(f)
# The largest program's scratch at a prefill-sized invocation (a drafting hybrid's @llm_verify keeps every token's
# gated delta-net inputs for the replay: 7 GB for 30 such layers at a 4096-token prefill), plus 3 GiB for the state
# storage, slab rounding and the workspace's own rounding. A fixed 3 GiB alone refused such a card's full-context
# request (503) at the memory_bytes printed here, on a fresh server.
workspace = program_scratch(os.path.join(pkg, "program", "llm.loom"), a.prefill, a.rows) + 3 * 2**30

native = cfg.get("context_length") or 32768
if native > KERNEL_CONTEXT and not a.context:
    print(f"NOTE: native context {native} exceeds the kernels' {KERNEL_CONTEXT}; using {KERNEL_CONTEXT}")
context = a.context or min(native, KERNEL_CONTEXT)
if a.memory_gib == "auto":
    # LOOM_RESERVED_GIB: memory another resident server already holds (weaver's agent model). Parsed outside the
    # try below: a typo there would read as "RAM unknown" and fall back to 56 GiB, more than the cap without it.
    try:
        reserved = float(os.environ.get("LOOM_RESERVED_GIB") or 0)
    except ValueError:
        sys.exit(f"LOOM_RESERVED_GIB={os.environ['LOOM_RESERVED_GIB']!r} is not a number (GiB)")
    try:
        cap = 0.6 * os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") - reserved * 2**30
    except (AttributeError, ValueError, OSError):
        cap = None
    if cap is not None and cap <= 0:
        sys.exit(f"LOOM_RESERVED_GIB={reserved:g} leaves no memory under the 60%-of-RAM cap")
    want_pool = context + max(65536, context // 4)
    need = weights_bytes + workspace + 2**30 + 2 * a.rows * recurrent_bytes + want_pool * kv_per_position
    a.memory_gib = round(min(need, cap) / 2**30, 1) if cap else 56.0
    print(f"memory budget {a.memory_gib} GiB (a {want_pool}-position pool needs {need / 2**30:.0f} GiB"
          + (f"; cap 60% of RAM{f' minus LOOM_RESERVED_GIB {reserved:g}' if reserved else ''} = {cap / 2**30:.0f} GiB)"
             if cap else "; RAM unknown)"))
a.memory_gib = float(a.memory_gib)
budget = int(a.memory_gib * 2**30) - weights_bytes - workspace - 2 * a.rows * recurrent_bytes  # workspace, activations, recurrent state
if context * kv_per_position > budget:
    fit = max(4096, (budget // kv_per_position) // 1024 * 1024)
    print(f"WARNING: context {context} needs {context * kv_per_position / 2**30:.1f} GiB of KV; {a.memory_gib} GiB budget fits {fit}")
    context = fit
# One full-context request plus headroom for other rows' retained history (2x for short contexts, 1.25x at 262k:
# a 2x pool at 262k was ~40 GiB of KV for a 27B, more than most single-user machines should reserve).
pool = a.pool or min(KERNEL_POOL, max(context, min(context + max(65536, context // 4), budget // kv_per_position)))
pool = max(pool, context)
# loom-serve rejects a pool its paged cache geometry can't represent ("source cache geometry cannot
# represent this pool", e.g. 43352): keep it a multiple of 1024 positions.
pool = max(pool // 1024 * 1024, -(-context // 1024) * 1024)
if pool < min(1.25 * context, context + 65536):
    print(f"WARNING: KV pool {pool} < 1.5x context {context}: a full-context request can fail with 503 while other rows "
          f"retain history (the full-context gate did on flint). Raise --memory-gib (a pool of context + 64k needs "
          f"~{(weights_bytes + (context + 65536) * kv_per_position) / 2**30 + 6:.0f} GiB) or lower --context.")
a.context, a.pool = context, pool
model_id = a.model_id or re.sub(r"[^A-Za-z0-9._/-]+", "-", cfg.get("name", "llm")).strip("-")

# 2. Serving glue from templates.
hx = lambda s: s.encode().hex()
thinking = chat.get("thinking", "toggle" if chat.get("starts_in_reasoning") else "never")
reasoning = {"toggle": (1, 1 | 16, 4 if chat.get("thinking_default", True) else 0), "always": (1, 0, 0), "never": (0, 0, 0)}[thinking]
# Reasoning instructions per effort (chat.reasoning_instructions, Qwen3.8): advertise the template's efforts
# (loom-serve levels: none=0 minimal=1 low=2 medium=3 high=4 xhigh=5 max=6; none = thinking off).
EFFORT_LEVELS = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5, "max": 6}
instructions = chat.get("reasoning_instructions") if thinking != "never" else None
if instructions:
    efforts = (1 if thinking == "toggle" else 0) | sum(1 << EFFORT_LEVELS[e] for e in instructions if e in EFFORT_LEVELS)
    default_effort = EFFORT_LEVELS.get(chat.get("reasoning_effort_default"), 4) if chat.get("thinking_default", True) else 0
    reasoning = (1, efforts, default_effort)


def sub(text, old, new):
    """TEXT with its one occurrence of OLD replaced by NEW (the control.loom patches below)."""
    assert text.count(old) == 1, (old[:120], text.count(old))
    return text.replace(old, new)


def patch_reasoning_instructions(text):
    """control.loom with the reasoning-instruction prefix (chat_probe.reasoning_instructions): chat_begin opens
    the system block with the request effort's instructions when thinking is on, merging the first system
    message into that block. Without chat.reasoning_instructions control.loom is unchanged."""
    names = list(instructions)
    rodata = "".join(f'global.rodata.def @reasoning_name_{i} = bytes("{hx(e)}")\n\n'
                     f'global.rodata.def @reasoning_text_{i} = bytes("{hx(instructions[e])}")\n\n' for i, e in enumerate(names))
    select = "".join(f"""    %name_{i} = global.load @reasoning_name_{i} : buffer
    %match_{i} = func.call @equal(%name, %name_{i}) : (buffer, buffer) -> (i1)
    %text_{i} = global.load @reasoning_text_{i} : buffer
    %selected_{i} = scf.select %match_{i}, %text_{i}, %selected_{i - 1 if i else 'default'} : buffer
""" for i in range(len(names)))
    code = f"""
// Reasoning instructions (the template's text per reasoning_effort): selected by the request's
// reasoning_effort, else chat_template_kwargs.reasoning_effort, else the template's default effort.
global.rodata.def @reasoning_separator = bytes("{hx(chat["reasoning_separator"])}")

global.rodata.def @reasoning_default = bytes("{hx(instructions[chat["reasoning_effort_default"]])}")

{rodata}func.def @reasoning_instructions(%request: buffer, %thinking: i1) -> (buffer) {{
  %empty = global.load @empty : buffer
  %result = scf.if %thinking -> (buffer) {{
    %zero64 = scalar.constant 0 : i64
    %false = scalar.constant false : i1
    %string_type = scalar.constant 0 : i64
    %request_bytes = buffer.length %request
    %request_length = index.cast %request_bytes : offset to i64
    %effort_key = global.load @effort_key : buffer
    %kwargs_key = global.load @kwargs_key : buffer
    %fk, %bk, %lk, %tk = func.call @json_lookup(%request, %zero64, %request_length, %kwargs_key) : (buffer, i64, i64, buffer) -> (i1, i64, i64, i64)
    %f1, %b1, %l1, %t1 = scf.if %fk -> (i1, i64, i64, i64) {{
      %f, %b, %l, %t = func.call @json_lookup(%request, %bk, %lk, %effort_key) : (buffer, i64, i64, buffer) -> (i1, i64, i64, i64)
      scf.yield %f, %b, %l, %t : i1, i64, i64, i64
    }} else {{
      scf.yield %false, %zero64, %zero64, %zero64 : i1, i64, i64, i64
    }}
    %kwargs_string = scalar.cmpi eq, %t1, %string_type : i64
    %use_kwargs = scalar.andi %f1, %kwargs_string : i1
    %f2, %b2, %l2, %t2 = func.call @json_lookup(%request, %zero64, %request_length, %effort_key) : (buffer, i64, i64, buffer) -> (i1, i64, i64, i64)
    %top_string = scalar.cmpi eq, %t2, %string_type : i64
    %use_top = scalar.andi %f2, %top_string : i1
    %found = scalar.ori %use_top, %use_kwargs : i1
    %begin = scf.select %use_top, %b2, %b1 : i64
    %length = scf.select %use_top, %l2, %l1 : i64
    %name = scf.if %found -> (buffer) {{
      %value = func.call @json_key(%request, %begin, %length) : (buffer, i64, i64) -> (buffer)
      scf.yield %value : buffer
    }} else {{
      scf.yield %empty : buffer
    }}
    %selected_default = global.load @reasoning_default : buffer
{select}    scf.yield %selected_{len(names) - 1} : buffer
  }} else {{
    scf.yield %empty : buffer
  }}
  func.return %result : buffer
}}

// INSTRUCTIONS + separator + BODY, or whichever of the two is not empty.
func.def @with_instructions(%instructions: buffer, %body: buffer) -> (buffer) {{
  %origin = index.constant 0 : offset
  %empty = global.load @empty : buffer
  %instructions_length = buffer.length %instructions
  %body_length = buffer.length %body
  %has_instructions = index.cmp ne, %instructions_length, %origin : offset
  %has_body = index.cmp ne, %body_length, %origin : offset
  %both = scalar.andi %has_instructions, %has_body : i1
  %separator_value = global.load @reasoning_separator : buffer
  %separator = scf.select %both, %separator_value, %empty : buffer
  %value = func.call @join4(%instructions, %separator, %body, %empty) : (buffer, buffer, buffer, buffer) -> (buffer)
  func.return %value : buffer
}}

"""
    anchor = "// State bits carried through chat_message (opaque to the host):\n"
    text = sub(text, anchor, code.lstrip("\n") + anchor)
    text = sub(text, "  %thinking = scf.select %toggle, %think3, %think_default : i1\n",
               """  %thinking = scf.select %toggle, %think3, %think_default : i1
  %instructions = func.call @reasoning_instructions(%request, %thinking) : (buffer, i1) -> (buffer)
  %instructions_length = buffer.length %instructions
  %has_instructions = index.cmp ne, %instructions_length, %origin : offset
""")
    if chat.get("reasoning_after_lead"):   # appended to the system lead (patch_system_lead)
        return text
    text = sub(text, "    %block = func.call @join4(%system_prefix, %merged, %suffix, %empty) : (buffer, buffer, buffer, buffer) -> (buffer)\n",
               """    %instructed = func.call @with_instructions(%instructions, %merged) : (buffer, buffer) -> (buffer)
    %block = func.call @join4(%system_prefix, %instructed, %suffix, %empty) : (buffer, buffer, buffer, buffer) -> (buffer)
""")
    text = sub(text, """    %block = scf.if %use_default -> (buffer) {
      %value = func.call @join4(%system_prefix, %default_system, %suffix, %empty) : (buffer, buffer, buffer, buffer) -> (buffer)
      scf.yield %value : buffer
    } else {
""", """    // With instructions the block also takes the first system message (chat_message then skips it).
    %plain_body = scf.select %use_default, %default_system, %empty : buffer
    %rest = scf.select %has_instructions, %system_body, %plain_body : buffer
    %instructed = func.call @with_instructions(%instructions, %rest) : (buffer, buffer) -> (buffer)
    %instructed_length = buffer.length %instructed
    %has_block = index.cmp ne, %instructed_length, %origin : offset
    %block = scf.if %has_block -> (buffer) {
      %value = func.call @join4(%system_prefix, %instructed, %suffix, %empty) : (buffer, buffer, buffer, buffer) -> (buffer)
      scf.yield %value : buffer
    } else {
""")
    return sub(text, "  %consumed = scalar.andi %has_tools, %first_system : i1\n",
               """  %merges_system = scalar.ori %has_tools, %has_instructions : i1
  %consumed = scalar.andi %merges_system, %first_system : i1
""")
def patch_system_lead(text):
    """control.loom for a template whose first system block always opens with its default system prompt
    (chat_probe.system_lead, Spark-X2.5): chat_begin renders system_prefix + lead [+ reasoning instructions] + [tools
    block + tools_system_separator + system message | system_lead_separator + system message] + suffix for every
    request, and the first system message always merges into it. Other packages' control.loom is unchanged."""
    assert not chat["tools_after_system"] and not chat.get("default_system"), "system_lead: tools before the system message only"
    text = sub(text, 'global.rodata.def @default_system = bytes("@@HEX_DEFAULT_SYSTEM@@")\n',
               'global.rodata.def @default_system = bytes("@@HEX_DEFAULT_SYSTEM@@")\n\n'
               f'global.rodata.def @system_lead = bytes("{hx(chat["system_lead"])}")\n\n'
               f'global.rodata.def @system_lead_separator = bytes("{hx(chat["system_lead_separator"])}")\n')
    lead = ("  %lead_base = global.load @system_lead : buffer\n" +
            ("  %lead = func.call @join(%lead_base, %instructions) : (buffer, buffer) -> (buffer)\n" if instructions else
             "  %lead = func.call @join(%lead_base, %empty) : (buffer, buffer) -> (buffer)\n"))
    text = sub(text, "  %rendered = scf.if %has_tools -> (buffer) {\n", lead + "  %rendered = scf.if %has_tools -> (buffer) {\n")
    text = sub(text, "    %block = func.call @join4(%system_prefix, %merged, %suffix, %empty) : (buffer, buffer, buffer, buffer) -> (buffer)\n",
               "    %block = func.call @join4(%system_prefix, %lead, %merged, %suffix) : (buffer, buffer, buffer, buffer) -> (buffer)\n")
    text = sub(text, """    %block = scf.if %use_default -> (buffer) {
      %value = func.call @join4(%system_prefix, %default_system, %suffix, %empty) : (buffer, buffer, buffer, buffer) -> (buffer)
      scf.yield %value : buffer
    } else {
      scf.yield %empty : buffer
    }
""", """    %lead_body_length = buffer.length %system_body
    %lead_has_body = index.cmp ne, %lead_body_length, %origin : offset
    %lead_separator_value = global.load @system_lead_separator : buffer
    %lead_separator = scf.select %lead_has_body, %lead_separator_value, %empty : buffer
    %lead_text = func.call @join(%lead, %lead_separator) : (buffer, buffer) -> (buffer)
    %block = func.call @join4(%system_prefix, %lead_text, %system_body, %suffix) : (buffer, buffer, buffer, buffer) -> (buffer)
""")
    return sub(text, "  %consumed = scalar.andi %has_tools, %first_system : i1\n", "  %consumed = scalar.ori %first_system, %false : i1\n")


def patch_history_close(text):
    """control.loom for templates that render an assistant turn without reasoning as think_close alone (history_think
    "close", Spark-X2.5: '<|Bot|></think>answer'). Other packages' control.loom is unchanged."""
    text = sub(text, "        %wanted = scalar.ori %always, %has_thought : i1\n", "        %wanted = scalar.constant true : i1\n")
    return sub(text, "          %open = global.load @think_open : buffer\n",
               "          %open_value = global.load @think_open : buffer\n          %open = scf.select %has_thought, %open_value, %empty : buffer\n")


def patch_tools_compact(text):
    """control.loom for templates that print each tool definition as compact JSON (tojson with separators ',' ':';
    chat.tools_compact, Spark-X2.6): the host json codec compacts the request's tool JSON. Other packages unchanged."""
    old = "    %printed = func.call @tool_printed(%value) : (buffer) -> (buffer)\n"
    assert text.count(old) == 1
    return text.replace(old, "    %printed_raw = func.call @tool_printed(%value) : (buffer) -> (buffer)\n"
                              "    %printed = func.call @compact(%printed_raw) : (buffer) -> (buffer)\n")


def patch_role_suffixes(text):
    """control.loom for templates that close each role with its own tag (chat.system_suffix / chat.assistant_suffix,
    Laguna: '</system>\n', '</assistant>\n'; chat.suffix closes user messages): system blocks (chat_begin and later
    system messages) and assistant turns (chat_message, complete_text) take their own suffix. Without those fields
    control.loom is unchanged."""
    text = sub(text, 'global.rodata.def @message_suffix = bytes("@@HEX_SUFFIX@@")\n',
               'global.rodata.def @message_suffix = bytes("@@HEX_SUFFIX@@")\n\n'
               f'global.rodata.def @system_suffix = bytes("{hx(chat.get('system_suffix', chat['suffix']))}")\n\n'
               f'global.rodata.def @assistant_suffix = bytes("{hx(chat.get('assistant_suffix', chat['suffix']))}")\n')
    # chat_begin: the system block
    text = sub(text, "  %system_prefix = global.load @system_prefix : buffer\n  %suffix = global.load @message_suffix : buffer\n  %rendered",
               "  %system_prefix = global.load @system_prefix : buffer\n  %suffix = global.load @system_suffix : buffer\n  %rendered")
    # chat_message: assistant turns, and system vs user messages
    text = sub(text, "        %message = func.call @join4(%head, %calls, %close, %suffix)",
               "        %assistant_suffix = global.load @assistant_suffix : buffer\n"
               "        %message = func.call @join4(%head, %calls, %close, %assistant_suffix)")
    text = sub(text, "        %message = func.call @join4(%prefix, %text, %suffix, %empty)",
               "        %system_suffix = global.load @system_suffix : buffer\n"
               "        %role_suffix = scf.select %user, %suffix, %system_suffix : buffer\n"
               "        %message = func.call @join4(%prefix, %text, %role_suffix, %empty)")
    # complete_text: the generated assistant turn
    return sub(text, "  %separator = scf.select %separate, %separator_value, %empty : buffer\n  %suffix = global.load @message_suffix : buffer\n",
               "  %separator = scf.select %separate, %separator_value, %empty : buffer\n  %suffix = global.load @assistant_suffix : buffer\n")


def patch_history_think_off(text):
    """control.loom for templates that show reasoning in history only with thinking on and render every assistant turn
    as think_close alone with it off (chat.history_think_off "close", Laguna: '<assistant></think>answer'). Without
    the field control.loom is unchanged."""
    text = sub(text, "        %text_length = buffer.length %text\n",
               """        // Thinking off (state bit 16 clear): think_close alone, the reasoning dropped (chat.history_think_off).
        %thinking_mask = scalar.constant 16 : i64
        %thinking_bit = scalar.andi %state, %thinking_mask : i64
        %thinking_off = scalar.cmpi eq, %thinking_bit, %zero64 : i64
        %close_only = global.load @think_close : buffer
        %think_shown = scf.select %thinking_off, %close_only, %think : buffer
        %text_length = buffer.length %text
""")
    return sub(text, "func.call @join4(%prefix, %think, %text, %separator)", "func.call @join4(%prefix, %think_shown, %text, %separator)")


def patch_default_system_tools(text):
    """control.loom for a template that swaps its default system prompt when tools are present
    (chat.default_system_tools, its separator included; Nanbeige): with tools and no system message, chat_begin prints
    that text beside the tools instead of default_system + separator. Without the field control.loom is unchanged."""
    text = sub(text, 'global.rodata.def @default_system = bytes("@@HEX_DEFAULT_SYSTEM@@")\n',
               'global.rodata.def @default_system = bytes("@@HEX_DEFAULT_SYSTEM@@")\n\n'
               f'global.rodata.def @default_system_tools = bytes("{hx(chat["default_system_tools"])}")\n')
    text = sub(text, "    %separator = scf.select %has_body, %separator_value, %empty : buffer\n",
               """    %body_separator = scf.select %has_body, %separator_value, %empty : buffer
    // No system message: the template's tools-time default system prompt, separator included.
    %default_tools = global.load @default_system_tools : buffer
    %separator = scf.select %not_first_system, %empty, %body_separator : buffer
    %tools_body = scf.select %not_first_system, %default_tools, %system_body : buffer
""")
    text = sub(text, "func.call @join4(%system_body, %separator, %tools_text, %empty)", "func.call @join4(%tools_body, %separator, %tools_text, %empty)")
    return sub(text, "func.call @join4(%tools_text, %separator, %system_body, %empty)", "func.call @join4(%tools_text, %separator, %tools_body, %empty)")


def patch_history_think_before(text):
    """control.loom for a template that keeps an empty think block in assistant turns before the last user query
    (chat.history_think_before "empty"; MiniCPM-V): chat_message renders THINK_OPEN + THINK_CLOSE there instead of
    nothing. Without the field control.loom is unchanged."""
    text = sub(text, "        %think = scf.if %show -> (buffer) {\n",
               """        // Before the last user query: an empty think block (chat.history_think_before).
        %early = scalar.cmpi eq, %in_scope, %false : i1
        %show_block = scalar.ori %show, %early : i1
        %kept_thought = scf.select %in_scope, %thought, %empty : buffer
        %think = scf.if %show_block -> (buffer) {
""")
    return sub(text, "func.call @join4(%open, %thought, %close, %empty)", "func.call @join4(%open, %kept_thought, %close, %empty)")


JSON_CHAT_KEYS = ("system_preamble", "tools_in_user", "tools_indent", "tool_content_json", "calls_drop_content")


def patch_json_chat(text):
    """control.loom for Llama-3-style templates (chat_probe: system_preamble, tools_in_user, tools_indent,
    tool_content_json, calls_drop_content, call_format "json"): the always-printed system block with its preamble,
    the tools opening the first user message (re-indented like tojson(indent=N)), tool results as JSON strings,
    no content beside calls, and bare JSON calls (templates/control_json.loom). Each piece only when its field is
    present; without them control.loom is unchanged."""
    preamble, in_user = "system_preamble" in chat, chat.get("tools_in_user", False)
    assert in_user <= preamble, "tools_in_user needs the always-printed system block (system_preamble)"
    assert not (preamble and instructions), "system_preamble with reasoning instructions is not modeled"
    if preamble:
        assert in_user or chat["call_format"] == "none", "a preamble with tools in the system block is not modeled"
        text = sub(text, 'global.rodata.def @default_system = bytes("@@HEX_DEFAULT_SYSTEM@@")\n',
                   'global.rodata.def @default_system = bytes("@@HEX_DEFAULT_SYSTEM@@")\n\n'
                   f'global.rodata.def @system_preamble = bytes("{hx(chat["system_preamble"])}")\n\n'
                   f'global.rodata.def @system_preamble_tools = bytes("{hx(chat.get("system_preamble_tools", chat["system_preamble"]))}")\n')
        begin = text.index("  %rendered = scf.if %has_tools -> (buffer) {\n")
        end = text.index("  %bos = global.load @bos : buffer\n", begin)
        text = text[:begin] + f"""  // The system block is always printed, opening with the template's preamble (chat.system_preamble; with tools
  // chat.system_preamble_tools), then the first system message's text.
  %preamble_plain = global.load @system_preamble : buffer
  %preamble_tools = global.load @system_preamble_tools : buffer
  %preamble = scf.select %has_tools, %preamble_tools, %preamble_plain : buffer
  %system_block = func.call @join4(%system_prefix, %preamble, %system_body, %suffix) : (buffer, buffer, buffer, buffer) -> (buffer)
  %tools_in_user = scalar.constant {B(in_user)} : i1
  %open_user = scalar.andi %has_tools, %tools_in_user : i1
  %rendered = scf.if %open_user -> (buffer) {{
    // The tools open the first user message (chat.tools_in_user); chat_message completes it (state bit 32).
    %tools_text = func.call @tools_body(%tools, %tool_fields, %tool_count) : (buffer, buffer, i64) -> (buffer)
    %user_prefix = global.load @user_prefix : buffer
    %value = func.call @join4(%system_block, %user_prefix, %tools_text, %empty) : (buffer, buffer, buffer, buffer) -> (buffer)
    scf.yield %value : buffer
  }} else {{
    scf.yield %system_block : buffer
  }}
""" + text[end:]
        text = sub(text, "  %consumed = scalar.andi %has_tools, %first_system : i1\n", "  %consumed = scalar.ori %first_system, %false : i1\n")
        text = sub(text, "  %flags = scalar.ori %consumed_bit, %thinking_bit : i64\n", """  %pending_value = scalar.constant 32 : i64
  %pending_bit = scf.select %open_user, %pending_value, %zero64 : i64
  %flags_base = scalar.ori %consumed_bit, %thinking_bit : i64
  %flags = scalar.ori %flags_base, %pending_bit : i64
""")
    if in_user:   # chat_message: the first message after the system block completes the user turn chat_begin opened
        text = sub(text, "  %fragment = func.call @join(%before, %body) : (buffer, buffer) -> (buffer)\n", """  %pending_value = scalar.constant 32 : i64
  %pending_bit = scalar.andi %state, %pending_value : i64
  %pending = scalar.cmpi ne, %pending_bit, %zero64 : i64
  %not_skipped = scalar.cmpi eq, %skip, %false : i1
  %completes = scalar.andi %pending, %not_skipped : i1
  %completion = func.call @join(%text, %suffix) : (buffer, buffer) -> (buffer)
  %shown_body = scf.select %completes, %completion, %body : buffer
  %fragment = func.call @join(%before, %shown_body) : (buffer, buffer) -> (buffer)
""")
        text = sub(text, "  %kept = scalar.andi %state, %keep_mask : i64\n", """  %pending_mask = scalar.constant -36 : i64
  %mask = scf.select %completes, %pending_mask, %keep_mask : i64
  %kept = scalar.andi %state, %mask : i64
""")
    if chat.get("tools_indent"):   # tools printed like tojson(indent=N): compacted, then re-indented
        text = sub(text, "  func.return %result : buffer\n}\n\nfunc.def @tools_body(", """  %compacted = func.call @compact(%result) : (buffer) -> (buffer)
  %indented = func.call @json_indent(%compacted) : (buffer) -> (buffer)
  func.return %indented : buffer
}

func.def @tools_body(""")
    if chat.get("tool_content_json"):   # tool results printed as JSON strings (tojson), untrimmed
        text = sub(text, "      %value = func.call @join4(%open, %item_prefix, %text, %item_suffix) : (buffer, buffer, buffer, buffer) -> (buffer)\n",
                   """      %tool_json = func.call @json_quote(%raw_text) : (buffer) -> (buffer)
      %value = func.call @join4(%open, %item_prefix, %tool_json, %item_suffix) : (buffer, buffer, buffer, buffer) -> (buffer)
""")
    if chat.get("calls_drop_content"):
        # An assistant turn with calls renders no content in history (chat_message). complete_text still stores the
        # turn as generated (content kept), so the retained row matches it; the next prompt, which drops the content,
        # reuses that row up to the first changed byte.
        text = sub(text, "        %head = func.call @join4(%prefix, %think, %text, %separator) : (buffer, buffer, buffer, buffer) -> (buffer)\n",
                   """        %shown_text = scf.select %has_calls, %empty, %text : buffer
        %head = func.call @join4(%prefix, %think, %shown_text, %separator) : (buffer, buffer, buffer, buffer) -> (buffer)
""")
    if chat["call_format"] == "json":
        # control_json.loom's render_tool / parse_tools / output_next replace the generic ones. complete_text keeps the
        # call-prefix token (Llama 3.1's <|python_tag|>) the model wrote before its call, so the retained row matches.
        assert thinking == "never", "call_format json with reasoning is not modeled (control_json.loom's output_next)"
        for fn in ("render_tool", "parse_tools", "output_next"):
            text = sub(text, f"func.def public @{fn}(", f"func.def @generic_{fn}(")
        text = sub(text, "  %head = func.call @join4(%prefix, %content, %separator, %tools) : (buffer, buffer, buffer, buffer) -> (buffer)\n",
                   """  %call_prefix = global.load @json_call_prefix : buffer
  %call_scan_end = buffer.length %response
  %prefix_at = func.call @find(%response, %c0, %call_scan_end, %call_prefix) : (buffer, offset, offset, buffer) -> (offset)
  %prefix_written = index.cmp ult, %prefix_at, %call_scan_end : offset
  %prefix_used = scalar.andi %prefix_written, %has_calls : i1
  %prefix_text = scf.select %prefix_used, %call_prefix, %empty : buffer
  %prefixed_tools = func.call @join(%prefix_text, %tools) : (buffer, buffer) -> (buffer)
  %head = func.call @join4(%prefix, %content, %separator, %prefixed_tools) : (buffer, buffer, buffer, buffer) -> (buffer)
""")
    return text + chat_variants(open(os.path.join(HERE, "templates", "control_json.loom"), encoding="utf-8").read()) \
        .replace("@@HEX_JSON_CALL_PREFIX@@", hx(chat.get("call_prefix_token", ""))) \
        .replace("@@TOOLS_INDENT@@", str(chat.get("tools_indent") or 4)) \
        .replace("@@HEX_JSON_CALL_MIDDLE@@", hx('", "' + chat.get("call_arguments_key", "arguments") + '": ')) \
        .replace("@@HEX_JSON_ARGUMENTS_FIELD@@", hx('"' + chat.get("call_arguments_key", "arguments") + '"'))


def patch_no_tools(text):
    """control.loom for a chat-only block (chat_probe.CHAT_ONLY, call_format "none": the template ignores tools):
    chat_begin renders a request's tools as the template does (not at all) and parse_tools never parses a call.
    Other packages' control.loom is unchanged."""
    for count in ("%tool_count", "%schema_count"):
        text = sub(text, f"  %has_tools = scalar.cmpi ne, {count}, %zero64 : i64\n", "  %has_tools = scalar.constant false : i1\n")
    return text


def chat_variants(text):
    """A custom chat file's variants for what the template's derived chat block says: the lines between
    '// @@IF_CHAT:KEY@@' and '// @@END_CHAT@@' (or '// @@ELSE_CHAT@@', which starts the other branch) are kept
    when chat[KEY] is true, or with '// @@IF_CHAT:KEY!=OTHER@@' when the two fields differ. The marker lines go,
    so a block that needs no variant compiles to the same program as before the variant was added."""
    out, keep = [], [True]
    for line in text.splitlines(keepends=True):
        m = re.match(r"\s*// @@(IF_CHAT:([\w!=]+)|ELSE_CHAT|END_CHAT)@@\s*$", line)
        if not m:
            out += [line] if all(keep) else []
        elif m.group(2):
            a, _, b = m.group(2).partition("!=")
            keep.append(chat.get(a) != chat.get(b) if b else bool(chat.get(a)))
        elif m.group(1) == "ELSE_CHAT":
            keep[-1] = not keep[-1]
        else:
            keep.pop()
    assert len(keep) == 1, "unbalanced // @@IF_CHAT@@ markers"
    return "".join(out)


B = lambda v: "true" if v else "false"
CALL_FORMATS = {"none": 0, "hermes": 1, "qwen_xml": 2, "glm": 3, "pythonic": 4, "json": 5}   # none: chat-only (patch_no_tools); json: patch_json_chat
# A chat format the generic control functions can't express (Glimmer's ATEM channels, Gemma 4): the kit
# ships chat/<call_format>.loom, whose public chat functions replace the generic ones (renamed
# @generic_<fn>, still callable), and optionally chat/<call_format>_rodata.py (a rodata() returning its
# string constants). In it, @@HEX_CHAT:<key>@@ is the hex of chat[<key>]; chat.base_call_format (default
# hermes) is the tool-call format the generic functions use underneath. See reference/glimmer-example.
# Its lines may hold variants for what the derived chat block says (chat_variants: // @@IF_CHAT:KEY@@ markers).
custom_chat = os.path.join(HERE, "chat", chat["call_format"] + ".loom")
custom_chat = custom_chat if chat["call_format"] not in CALL_FORMATS and os.path.exists(custom_chat) else None
subs = {
    "@@POS_BYTES@@": str(pos_bytes), "@@PLANES@@": str(2 * kv_layers), "@@BLOCK_BYTES@@": str(16 * pos_bytes),
    "@@RECURRENT_BYTES@@": str(recurrent_bytes or 4),
    "@@MAX_CONTEXT@@": str(a.context), "@@TERMINAL@@": chat["terminal"], "@@HEX_TERMINAL@@": hx(chat["terminal"]),
    "@@HEX_UNSUPPORTED@@": hx(f"llm requires pooled epochs without MTP, at most 128 shapes and a context of at most {a.context}"),
    "@@HEX_NAME@@": hx(model_id),
    "@@HEX_SYSTEM_PREFIX@@": hx(chat["system_prefix"]), "@@HEX_USER_PREFIX@@": hx(chat["user_prefix"]),
    "@@HEX_ASSISTANT_PREFIX@@": hx(chat["assistant_prefix"]), "@@HEX_SUFFIX@@": hx(chat["suffix"]),
    "@@HEX_THINK_OPEN@@": hx(chat.get("think_open", "")), "@@HEX_THINK_CLOSE@@": hx(chat.get("think_close", "")),
    "@@HEX_GENERATION_PROMPT@@": hx(chat["generation_prompt"]),
    "@@HEX_GENERATION_PROMPT_PLAIN@@": hx(chat["generation_prompt_plain"]),
    "@@START_STATE@@": "1" if chat.get("starts_in_reasoning") else "0",
    "@@HEX_THINK_END@@": hx(chat.get("think_end", "</think>")),
    "@@HEX_DEFAULT_SYSTEM@@": hx(chat.get("default_system", "")), "@@HEX_BOS@@": hx(chat.get("bos", "")),
    "@@HEX_TOOLS_PREFIX@@": hx(chat["tools_prefix"]), "@@HEX_TOOLS_SUFFIX@@": hx(chat["tools_suffix"]),
    "@@HEX_TOOL_LINE_PREFIX@@": hx(chat.get("tool_line_prefix", "\n")),
    "@@HEX_TOOL_SEPARATOR@@": hx(chat.get("tool_separator", "")),
    "@@HEX_CALLS_OPEN@@": hx(chat.get("calls_open", "")), "@@HEX_CALLS_CLOSE@@": hx(chat.get("calls_close", "")),
    "@@HEX_CALL_END@@": hx(chat.get("call_end", "")), "@@HEX_STOP_AFTER_CALLS@@": hx(chat.get("stop_after_calls", "")), "@@HEX_STOP_AFTER_ANSWER@@": hx(chat.get("stop_after_answer", "")),
    "@@HEX_TOOLS_SYSTEM_SEPARATOR@@": hx(chat["tools_system_separator"]),
    "@@HEX_TOOL_TURN_PREFIX@@": hx(chat["tool_turn_prefix"]), "@@HEX_TOOL_ITEM_PREFIX@@": hx(chat["tool_item_prefix"]),
    "@@HEX_TOOL_ITEM_SUFFIX@@": hx(chat["tool_item_suffix"]), "@@HEX_TOOL_TURN_SUFFIX@@": hx(chat["tool_turn_suffix"]),
    "@@HEX_CONTENT_CALL_SEPARATOR@@": hx(chat["content_call_separator"]), "@@HEX_CALL_SEPARATOR@@": hx(chat["call_separator"]),
    "@@HEX_CALL_MARKER@@": hx(chat.get("call_marker", "<tool_call>")),
    "@@DEFAULT_TEMPERATURE@@": repr(float(a.temperature)), "@@DEFAULT_TOP_P@@": repr(float(a.top_p)), "@@DEFAULT_TOP_K@@": str(a.top_k),
    "@@CALL_FORMAT@@": str(CALL_FORMATS[chat.get("base_call_format", "hermes") if custom_chat else chat["call_format"]]),
    "@@TOOLS_AFTER_SYSTEM@@": B(chat["tools_after_system"]), "@@TOOLS_FUNCTION_ONLY@@": B(chat["tools_function_only"]),
    "@@TRIM_CONTENT@@": B(chat["trim_content"]),
    "@@HISTORY_THINK_ALWAYS@@": B(chat["history_think"] == "always"),
    "@@HISTORY_THINK_ALL@@": B(chat["history_think_scope"] == "all"),
    "@@THINK_TOGGLE@@": B(thinking == "toggle"), "@@THINK_DEFAULT@@": B(chat.get("thinking_default", thinking != "never")),
    "@@REASONING_SUPPORTED@@": str(reasoning[0]), "@@REASONING_EFFORTS@@": str(reasoning[1]), "@@REASONING_DEFAULT@@": str(reasoning[2]),
    "@@CHAT_COMMENT@@": json.dumps({k: v for k, v in chat.items() if not k.startswith("tools_")})[:400].replace("\n", " "),
}
for name in ("prepare", "control", "storage", "service"):
    text = open(os.path.join(HERE, "templates", name + ".loom"), encoding="utf-8").read()
    if name == "control" and instructions:
        text = patch_reasoning_instructions(text)
    if name == "control" and chat.get("system_lead"):
        text = patch_system_lead(text)
    if name == "control" and chat.get("history_think") == "close":
        text = patch_history_close(text)
    if name == "control" and chat.get("tools_compact"):
        text = patch_tools_compact(text)
    if name == "control" and chat.get("history_think_before") == "empty":
        text = patch_history_think_before(text)
    if name == "control" and "default_system_tools" in chat:
        text = patch_default_system_tools(text)
    if name == "control" and (chat["call_format"] == "json" or any(k in chat for k in JSON_CHAT_KEYS)):
        text = patch_json_chat(text)
    if name == "control" and chat["call_format"] == "none":
        text = patch_no_tools(text)
    if name == "control" and ("system_suffix" in chat or "assistant_suffix" in chat):
        text = patch_role_suffixes(text)
    if name == "control" and chat.get("history_think_off") == "close":
        text = patch_history_think_off(text)
    if name == "control" and custom_chat:
        for fn in ("chat_begin", "chat_message", "chat_end", "render_tool", "parse_tools", "output_next", "complete_text"):
            text = text.replace(f"func.def public @{fn}(", f"func.def @generic_{fn}(")
        rodata_py = custom_chat[:-len(".loom")] + "_rodata.py"
        if os.path.exists(rodata_py):
            sys.path.insert(0, os.path.dirname(rodata_py))
            text += "\n" + __import__(os.path.basename(rodata_py)[:-3]).rodata()
        text += "\n" + chat_variants(open(custom_chat, encoding="utf-8").read())
        text = re.sub(r"@@HEX_CHAT:(\w+)@@", lambda m: hx(chat.get(m.group(1), "")), text)
    if name == "storage" and (cfg.get("vision") or {}).get("mrope"):   # M-RoPE span offsets after the plan
        sys.path.insert(0, os.path.join(HERE, "vision"))
        import build_vision
        text = build_vision.mrope_storage(text)
    if name == "control" and cfg.get("vision"):   # image input (vision/README.md)
        sys.path.insert(0, os.path.join(HERE, "vision"))
        import build_vision, encoder as vision_encoder
        vision_cfg = vision_encoder.config(cfg["vision"]["mmproj"])
        text = build_vision.patch_control(text, custom_chat, cfg["vision"], vision_cfg)
    if DRAFT:   # speculative decoding glue (draft/build_draft.py)
        text = {"prepare": lambda t: build_draft.patch_prepare(t, DRAFT, cfg), "storage": lambda t: build_draft.patch_storage(t, cfg, DRAFT),
                "control": lambda t: build_draft.patch_control(t, cfg, DRAFT)}.get(name, lambda t: t)(text)
    for k, v in subs.items():
        text = text.replace(k, v)
    assert "@@" not in text, (name, [l for l in text.splitlines() if "@@" in l])
    open(os.path.join(pkg, "program", name + ".loom"), "w", encoding="utf-8", newline="\n").write(text)
open(os.path.join(pkg, "weights.loom"), "w", encoding="utf-8", newline="\n").write(open(os.path.join(HERE, "templates", "weights.loom"), encoding="utf-8").read())

# 3. Tokenizer. loom-serve strips *special* added tokens from decoded text; any
# structural token the control program must see in the output (channel tags,
# e.g. Glimmer's <|start|>/<|message|>) is listed in chat.nonspecial_tokens.
tok = json.load(open(a.tokenizer, encoding="utf-8"))
keep = set(chat.get("nonspecial_tokens", []))
for t in tok.get("added_tokens", []):
    if t["content"] in keep:
        t["special"] = False
from runtime_tokenizer import rewrite_tokenizer  # noqa: E402  forms the runtime loads, same tokenization
rewrite_tokenizer(tok)
json.dump(tok, open(os.path.join(pkg, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False)

# 4. Weights (symlink; hard link on Windows without Developer Mode) and the package card.
gguf_name = os.path.basename(cfg["gguf"])
link = os.path.join(pkg, "parameter", gguf_name)
if os.path.lexists(link):
    os.remove(link)
try:
    os.symlink(os.path.abspath(cfg["gguf"]), link)
except OSError:  # Windows needs Developer Mode (or admin) for symlinks
    try:
        os.link(os.path.realpath(cfg["gguf"]), link)
    except OSError as e:
        sys.exit(f"cannot link {cfg['gguf']} into {link} ({e}): enable Windows Developer Mode, or put the package on the GGUF's drive")
card = {
    "schema_version": 1,
    "model": {"id": model_id, "name": f"{cfg.get('name', 'llm')} (Loom port, GGUF)"},
    "programs": {"source": {
        "inputs": [{"path": "program/prepare.loom", "data": ["weights.loom"] + POLICIES},
                   "program/control.loom", "program/storage.loom", "program/service.loom"],
        "libraries": ["program/llm.loom"]}},
    "services": {"chat": {"kind": "text", "version": 1}},
    "variants": {"gguf": {"services": {"chat": {
        "programs": ["source"], "config": {"llm.card": True},
        "parameters": {"model": {"files": [{"path": f"parameter/{gguf_name}", "format": "gguf"}]}},
        "tokenizers": {"text": "tokenizer.json"}}}}},
    "presets": {"chat": {"service": "chat", "variant": "gguf", "programs": "source", "execution": {
        "rows": a.rows, "context_capacity": a.context, "pool_capacity": a.pool,
        "prefill_capacity": a.prefill, "max_tokens": a.max_tokens}}},
}
if DRAFT:
    ex = card["presets"]["chat"]["execution"]
    card["presets"]["chat"]["execution"] = build_draft.card_execution(ex, DRAFT)
    card = build_draft.card_drafter(card, pkg, DRAFT)
if cfg.get("vision"):
    card = build_vision.write_component(pkg, card, cfg["vision"], vision_cfg)
json.dump(card, open(os.path.join(pkg, "model.json"), "w"), indent=1)
# The benchmark scripts read the chat format and model id from here.
bench = {k: v for k, v in cfg.items() if k != "tensors"}
bench["model_id"] = card["model"]["id"]
bench["draft"] = DRAFT
# What generated the programs (kit_source.py): the loom-skills commit of this kit and the Loom SDK. The model card
# shows them; a kit made by apply_example.sh carries the commit in KIT_SOURCE.
from kit_source import generator_stamp  # noqa: E402
bench["generator"] = generator_stamp()
json.dump(bench, open(os.path.join(pkg, "llm_config.json"), "w"), indent=1)
kv = a.pool * kv_per_position
need = weights_bytes + kv + workspace + 2 * a.rows * recurrent_bytes
print("package:", os.path.join(pkg, "model.json"), f"(model id {model_id!r})")
print("speculative decoding:", f"{DRAFT['kind']} drafter, depth {DRAFT['depth']}" if DRAFT else "off")
print(f"context {a.context} tokens (native {native}), KV pool {a.pool} positions = {kv / 2**30:.1f} GiB, weights {weights_bytes / 2**30:.1f} GiB"
      f", workspace {workspace / 2**30:.1f} GiB"
      + (f", recurrent state {recurrent_bytes / 2**20:.1f} MiB per row" if recurrent_bytes else ""))
print(f"serve with --memory_bytes={need // 10**9 + 1}000000000"
      " (too small shows up as 503 'retained state leaves insufficient completion capacity')")
