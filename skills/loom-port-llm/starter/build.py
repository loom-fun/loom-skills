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
# Hybrid models keep one recurrent slot (all gated delta-net layer states) per resident row, x2 for forks.
sys.path.insert(0, HERE)
from gen_gdn import gdn_geometry  # noqa: E402
recurrent_bytes = gdn_geometry(cfg)["slot_bytes"] if cfg.get("recurrent_layers") else 0
if cfg.get("shortconv_layers"):  # LFM2 short-conv history lives in the recurrent slots too (reference/lfm2moe-example)
    from gen_sconv import sconv_geometry  # noqa: E402
    recurrent_bytes += sconv_geometry(cfg)["slot_bytes"]
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
    need = weights_bytes + 4 * 2**30 + 2 * a.rows * recurrent_bytes + want_pool * kv_per_position
    a.memory_gib = round(min(need, cap) / 2**30, 1) if cap else 56.0
    print(f"memory budget {a.memory_gib} GiB (a {want_pool}-position pool needs {need / 2**30:.0f} GiB"
          + (f"; cap 60% of RAM{f' minus LOOM_RESERVED_GIB {reserved:g}' if reserved else ''} = {cap / 2**30:.0f} GiB)"
             if cap else "; RAM unknown)"))
a.memory_gib = float(a.memory_gib)
budget = int(a.memory_gib * 2**30) - weights_bytes - 3 * 2**30 - 2 * a.rows * recurrent_bytes  # workspace, activations, recurrent state
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
pkg = os.path.abspath(a.package)
os.makedirs(os.path.join(pkg, "program"), exist_ok=True)
os.makedirs(os.path.join(pkg, "parameter"), exist_ok=True)

# 1. Kernel library + command program.
gen = os.path.join(pkg, "gen")
env = dict(os.environ, LLM_CONFIG=os.path.abspath(a.config), LLM_OUT=gen, LLM_TOKENIZER=os.path.abspath(a.tokenizer),
           LLM_DRAFT=json.dumps(DRAFT))
sys.path.insert(0, os.path.join(HERE, "draft"))
import build_draft  # noqa: E402
subprocess.run([sys.executable, os.path.join(HERE, "gen_llm.py"), os.path.abspath(a.config)], check=True, env=env, stdout=subprocess.DEVNULL)
os.replace(os.path.join(gen, "model.loom"), os.path.join(pkg, "program", "llm.loom"))

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


def patch_reasoning_instructions(text):
    """control.loom with the reasoning-instruction prefix (chat_probe.reasoning_instructions): chat_begin opens
    the system block with the request effort's instructions when thinking is on, merging the first system
    message into that block. Without chat.reasoning_instructions control.loom is unchanged."""
    def sub(t, old, new):
        assert t.count(old) == 1, (old[:120], t.count(old))
        return t.replace(old, new)
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
B = lambda v: "true" if v else "false"
CALL_FORMATS = {"hermes": 1, "qwen_xml": 2, "glm": 3, "pythonic": 4}
# A chat format the generic control functions can't express (Glimmer's ATEM channels, Gemma 4): the kit
# ships chat/<call_format>.loom, whose public chat functions replace the generic ones (renamed
# @generic_<fn>, still callable), and optionally chat/<call_format>_rodata.py (a rodata() returning its
# string constants). In it, @@HEX_CHAT:<key>@@ is the hex of chat[<key>]; chat.base_call_format (default
# hermes) is the tool-call format the generic functions use underneath. See reference/glimmer-example.
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
    if name == "control" and custom_chat:
        for fn in ("chat_begin", "chat_message", "chat_end", "render_tool", "parse_tools", "output_next", "complete_text"):
            text = text.replace(f"func.def public @{fn}(", f"func.def @generic_{fn}(")
        rodata_py = custom_chat[:-len(".loom")] + "_rodata.py"
        if os.path.exists(rodata_py):
            sys.path.insert(0, os.path.dirname(rodata_py))
            text += "\n" + __import__(os.path.basename(rodata_py)[:-3]).rodata()
        text += "\n" + open(custom_chat, encoding="utf-8").read()
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
        "inputs": [{"path": "program/prepare.loom", "data": ["weights.loom"]}, "program/control.loom", "program/storage.loom", "program/service.loom"],
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
json.dump(bench, open(os.path.join(pkg, "llm_config.json"), "w"), indent=1)
kv = a.pool * kv_per_position
need = weights_bytes + kv + 3 * 2**30 + 2 * a.rows * recurrent_bytes
print("package:", os.path.join(pkg, "model.json"), f"(model id {model_id!r})")
print("speculative decoding:", f"{DRAFT['kind']} drafter, depth {DRAFT['depth']}" if DRAFT else "off")
print(f"context {a.context} tokens (native {native}), KV pool {a.pool} positions = {kv / 2**30:.1f} GiB, weights {weights_bytes / 2**30:.1f} GiB"
      + (f", recurrent state {recurrent_bytes / 2**20:.1f} MiB per row" if recurrent_bytes else ""))
print(f"serve with --memory_bytes={need // 10**9 + 1}000000000"
      " (too small shows up as 503 'retained state leaves insufficient completion capacity')")
