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
}
SB = {"q4_k": 256, "q5_k": 256, "q6_k": 256, "q8_0": 32, "f32": 32,
      "q4_0": 32, "iq4_nl": 32, "iq4_xs": 256, "q3_k": 256, "iq3_s": 256, "q5_0": 32, "f16": 32, "bf16": 32}
LOWBIT = {"q4_0", "iq4_nl", "iq4_xs", "q3_k", "iq3_s"}
# Formats whose decode matvecs and head take int8 activations (ACT_Q8): gen_llm.py has their @g_<fmt>_icodes.
ICODES = frozenset({"q8_0", "q4_k", "q5_k", "q6_k", "q5_0"})
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
