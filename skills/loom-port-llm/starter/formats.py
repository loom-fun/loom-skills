"""Weight formats the starter's kernels decode: the one list gen_llm.py (block sizes), inspect_gguf.py (its
"no kernels for" check) and pick_gguf.py (which files a port can use) share. Add a format here when gen_llm.py
gets its dequant helper. Standard library only (pick_gguf.py runs before anything else is set up)."""

FMT = {  # bytes per superblock of SB[fmt] values
    "q4_k": 144, "q5_k": 176, "q6_k": 210, "q8_0": 34, "f32": 128,
    # low-bit formats of unsloth UD / bartowski quants (helpers in quant_values.loom, appended when used)
    "q4_0": 18, "iq4_nl": 18, "iq4_xs": 136, "q3_k": 110, "iq3_s": 110,
    "q5_0": 22,  # projections whose rows aren't a multiple of 256 in llama.cpp Q4_K_M quants: gen_llm.Q5_0, emitted when used
    "f16": 64,   # small F16 projections of unsloth UD quants (Qwen3.5 ssm_alpha/beta): @g_f16_values, emitted when used
    "bf16": 64,  # BF16 matrices (vendor drafters published only in BF16, used as-is): @g_bf16_values, emitted when used
    "mxfp4": 17,  # OCP MX FP4 (gpt-oss experts): E8M0 scale byte + 16 bytes of E2M1 nibbles: gen_llm.MXFP4, emitted when used
    # OCP MX FP4, split planes (Quark / vLLM safetensors): 16 bytes of E2M1 nibbles per 32 values in one tensor (byte j = values 2j
    # low nibble, 2j+1 high) and the E8M0 scales, one byte per 32 values, in a second tensor: gen_llm.MXFP4S, emitted when used. The
    # kernels take two operands (nibbles, scales); the block size below counts the nibble plane only.
    "mxfp4s": 16,
    "ptq1_0": 28,  # PrismML ternary (Ternary Bonsai): 128 trits + F16 d per 28 bytes: ternary.py helpers, emitted when used
}
SB = {"q4_k": 256, "q5_k": 256, "q6_k": 256, "q8_0": 32, "f32": 32,
      "q4_0": 32, "iq4_nl": 32, "iq4_xs": 256, "q3_k": 256, "iq3_s": 256, "q5_0": 32, "f16": 32, "bf16": 32, "mxfp4": 32, "mxfp4s": 32, "ptq1_0": 128}
LOWBIT = {"q4_0", "iq4_nl", "iq4_xs", "q3_k", "iq3_s"}
# Formats whose decode matvecs and head take int8 activations (ACT_Q8): gen_llm.py has their @g_<fmt>_icodes.
import os as _os
# MXFP4 split planes (choice decode.mxfp4s.act_q8, env MXFP4S_Q8): opt-in per model, gen_mxfp4s.mxfp4s_icodes.
# Q4_0 (choice decode.q4_0.act_q8, env Q4_0_Q8): opt-in by scope; @g_q4_0_icodes (codes 0..15, minimum 8 d), as llama.cpp's
# vec_dot_q4_0_q8_1. Dense matvecs, the head and the MoE expert matvecs (MOE_Q8) of Q4_0 weights take the int8 path.
_OFF = ("", "0", "false", "no", "off")
ICODES = frozenset({"q8_0", "q4_k", "q5_k", "q6_k", "q5_0", "ptq1_0"} | ({"mxfp4s"} if _os.environ.get("MXFP4S_Q8", "0").lower() not in _OFF else set())
                   | ({"q4_0"} if _os.environ.get("Q4_0_Q8", "0").lower() not in _OFF else set())
                   | ({"iq4_xs", "iq4_nl"} if _os.environ.get("IQ4_Q8", "0").lower() not in _OFF else set()))
# IQ4_XS / IQ4_NL (choice decode.iq4.act_q8, env IQ4_Q8): opt-in by scope; @g_iq4_<fmt>_icodes (signed kvalues codes, one scale per
# 32 values, no minimum), as llama.cpp's vec_dot_iq4_xs_q8_1 / vec_dot_iq4_nl_q8_1. Dense matvecs take the int8 path.
# Formats whose int8 dot is codes x one scale (no minimum, no second scale): q8_0-like in gen_llm.iq_decode / iq_dot.
ONE_SCALE = frozenset({"q8_0", "mxfp4s", "iq4_xs", "iq4_nl", "ptq1_0"})
# Formats stored as two tensors (weight plane + scale plane): a call site passes two views, the second named <weight>_sc.
SPLIT = frozenset({"mxfp4s"})
SCALE_DIV = {"mxfp4s": 16}   # nibble-plane bytes per scale-plane byte
# GGUF tensor type names (gguf.GGMLQuantizationType, lower case) the kit decodes.
DECODES = frozenset(FMT)


def split_parts(path):
    """The files of a split GGUF (llama.cpp gguf-split: NAME-00001-of-00003.gguf ...), given the first part, in order;
    [path] for a single-file GGUF. The parts sit side by side (port.sh links them all into WORKDIR/gguf)."""
    import re
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path)
    if not m:
        return [path]
    n = int(m.group(2))
    return [f"{path[:m.start()]}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]


def memory_cap():
    """(bytes, description, reserved bytes): the GPU memory a package may plan for (build.py's budget, pick_gguf.py's
    fit). LOOM_MEMORY_CAP_GIB when set (a runner whose GPU budget is known, e.g. DevLab's 100 GiB GTT), else 60 % of
    RAM; minus LOOM_RESERVED_GIB (memory another resident server already holds). bytes is None when RAM is unknown.
    A malformed value exits: read as "unknown" it would fall back to a default above the cap it meant to set."""
    import os
    import sys

    def gib(name):
        v = os.environ.get(name)
        try:
            return float(v) if v else None
        except ValueError:
            sys.exit(f"{name}={v!r} is not a number (GiB)")
    reserved, override = (gib("LOOM_RESERVED_GIB") or 0) * 2**30, gib("LOOM_MEMORY_CAP_GIB")
    if override is not None:
        cap, what = override * 2**30, f"LOOM_MEMORY_CAP_GIB {override:g}"
    else:
        try:
            cap, what = 0.6 * os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), "60% of RAM"
        except (AttributeError, ValueError, OSError):
            return None, "RAM unknown", reserved
    if reserved:
        what += f" minus LOOM_RESERVED_GIB {reserved / 2**30:g}"
    return cap - reserved, what, reserved


if __name__ == "__main__":
    import os
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "size":   # formats.py size FIRST_PART: bytes of the (split) GGUF
        print(sum(os.path.getsize(p) for p in split_parts(sys.argv[2])))
    else:
        sys.exit("usage: formats.py size MODEL.gguf")
