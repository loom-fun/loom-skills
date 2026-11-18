#!/usr/bin/env python3
"""THE choice registry: every knob that changes what the generators emit (programs or model.json), declared once.

A *choice* is a model-affecting generator setting. Generators read each as an environment variable (env stays the internal
transport: build.py resolves the choices of a model, exports the ones that differ from the default, and starts gen_llm.py).
`legacy` is the behaviour of the generators today; introducing the registry changes no program. A new choice is opt-in by
scope: its legacy value is the old behaviour, and only a `scope.json` rule (skills/loom-kb/**/scope.json, resolve.py) or a
model's own `llm_config.json["choices"]` turns it on, for the models and devices that rule names.

Everything the generators read from the environment is in exactly one of two places:
  CHOICES      model-affecting (changes generated programs / model.json), keyed by dotted name by concern
  NOT_CHOICES  {ENV: reason}: paths, test selection, tuning harness, debugging, transport between generators, machine facts
tests/test_choices.py greps the generators and fails when an environment variable is in neither, or when a legacy value
disagrees with the generator's default. A new knob is registered here in the same PR that adds it (kind architecture:
choices.py is a protected core file).

CHOICE(key, type, legacy, env=..., kernels=[...], rows=[...], devices=None, per_target=False, from_pr=None, doc=...)
  key         dotted name by concern: decode.attn.kv_once.d256, prefill.gemm.m256 ...
  type        "bool" | "int" | "str"
  legacy      today's default in the generators, typed (None: the variable is unset and the generator decides)
  env         the environment variable the generators read
  kernels     kernel families the choice can change (for the fleet diff and the scope lint)
  rows        card rows it can move (decode, decode_2k, decode_32k, prefill_1k..prefill_32k, streams4)
  devices     device families the choice can change (None: every device). A key that only changes a gfx12 twin says ["rdna4"]
  per_target  True when one dual-target build can emit different values per device (a twin K / K_rdna4)
  from_pr     the PR (or file) the choice came from
  enc         ("true text", "false text") env spelling of a bool (default "1", "0"); None as false text: variable unset
  derive      f(cfg, values, ctx) -> the model's effective default (cfg-driven flags such as rdna4_f16_ffn, gemm_n_fastest)
  kits        kit examples that carry the knob (None: the plain starter and every kit)
  kit_legacy  {kit example: default} where a reference kit's patched generator defaults differently
  dynamic     the generator computes its default (legacy None): the unset case is "whatever the generator picks"
  experiment  an A/B switch kept for experiments; a scope may set it but no model ships it by default
  execution   a serving setting of one (model, device) preset in model.json (exec.*): no environment variable, no generator, no
              program changes; resolve.device_execution() reads it per device, make_info.py build writes the device preset from it
"""
import re

TYPES = ("bool", "int", "str")
ROWS_DECODE = ["decode", "decode_2k", "decode_32k"]
ROWS_PREFILL = ["prefill_1k", "prefill_2k", "prefill_8k", "prefill_16k", "prefill_32k"]
ROWS_STREAMS = ["streams4"]
ROWS_ALL = ROWS_DECODE + ROWS_PREFILL + ROWS_STREAMS

CHOICES = {}
BY_ENV = {}
NOT_CHOICES = {}
FALSE_WORDS = ("", "0", "false", "no", "off", "none")


class Choice:
    def __init__(self, key, type, legacy, env, kernels=(), rows=(), devices=None, per_target=False, from_pr=None, doc="",
                 enc=("1", "0"), derive=None, kits=None, kit_legacy=None, dynamic=False, experiment=False,
                 generator_defaults=None, execution=False):
        assert type in TYPES, (key, type)
        assert key not in CHOICES and (env is None or env not in BY_ENV), f"duplicate choice {key} / {env}"
        assert (env is None) == bool(execution), f"{key}: an execution choice has no environment variable (and only those)"
        assert re.fullmatch(r"[a-z0-9_]+(\.[a-z0-9_]+)+", key), key
        self.key, self.type, self.legacy, self.env = key, type, legacy, env
        self.kernels, self.rows, self.devices = list(kernels), list(rows), list(devices) if devices else None
        self.per_target, self.from_pr, self.doc, self.enc = per_target, from_pr, doc, enc
        self.derive, self.kits, self.kit_legacy = derive, kits, dict(kit_legacy or {})
        self.dynamic, self.experiment, self.execution = dynamic, experiment, execution
        # strings (None: no default) the generators pass as the default of this variable; tests compare legacy to them
        self.generator_defaults = generator_defaults
        self.check(legacy) if legacy is not None else None
        CHOICES[key] = self
        if env is not None:
            BY_ENV[env] = self

    # -- values -------------------------------------------------------------------------------------------------
    def check(self, value):
        """VALUE as a typed value of this choice (ValueError otherwise)."""
        if self.type == "bool":
            if isinstance(value, bool):
                return value
        elif self.type == "int":
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        elif isinstance(value, str):
            return value
        raise ValueError(f"{self.key}: {value!r} is not a {self.type}")

    def to_env(self, value):
        """The environment spelling of VALUE (None: the variable is unset)."""
        if value is None:
            return None
        if self.type == "bool":
            return self.enc[0] if value else self.enc[1]
        return str(value)

    def default(self, cfg=None, values=None, ctx=None, kit=None):
        """The effective default for a model: derive() of a cfg-driven flag, else the kit's, else legacy."""
        if self.derive is not None and cfg is not None:
            v = self.derive(cfg, values if values is not None else {}, ctx or {})
            if v is not None:
                return v
        return self.kit_legacy.get(kit, self.legacy)

    @property
    def legacy_env(self):
        return self.to_env(self.legacy)

    def as_dict(self):
        return {"key": self.key, "env": self.env, "type": self.type, "legacy": self.legacy, "kernels": self.kernels,
                "rows": self.rows, "devices": self.devices, "per_target": self.per_target, "from_pr": self.from_pr,
                "kits": self.kits, "dynamic": self.dynamic, "experiment": self.experiment, "execution": self.execution,
                "doc": self.doc}


def CHOICE(key, type, legacy, env, **kw):
    return Choice(key, type, legacy, env, **kw)


def EXEC(key, type, doc, rows=(), from_pr=None):
    """An execution choice (see the module docstring): a per-(model, device) serving setting; legacy None = the preset keeps what the
    package's default preset (or the device's fit) says."""
    return Choice(key, type, None, None, rows=rows, execution=True, from_pr=from_pr, doc=doc)


def NOT(reason, *envs):
    for e in envs:
        assert e not in NOT_CHOICES and e not in BY_ENV, e
        NOT_CHOICES[e] = reason


def _twins(cfg, values, ctx):
    """A build for gfx11 only (no rdna4 device) emits no twins."""
    return "rdna4" in ctx["families"] if "families" in ctx else None


def _f16o(cfg, values, ctx):
    """rdna4_f16_ffn (llm_config) turns the gfx12 f16-output swiglu stack on, unless the twins are off."""
    return bool(cfg.get("rdna4_f16_ffn")) and values.get("target.rdna4_twins", True)


def _trim(cfg, values, ctx):
    return True if values.get("prefill.gemm.f16o_swiglu") else None


def _flash_b(cfg, values, ctx):
    return "2" if values.get("prefill.gemm.f16o_swiglu") else None


def _n_fastest(cfg, values, ctx):
    return bool(cfg["gemm_n_fastest"]) if "gemm_n_fastest" in cfg else None


def _mtp_rows(cfg, values, ctx):
    return 8 if ctx.get("draft_kind") == "mtp" else None


def _mtp_cols(cfg, values, ctx):
    return 4 if ctx.get("draft_kind") == "mtp" else None


# ======================================================================================================================
# Targets
# ======================================================================================================================
CHOICE("target.rdna4_twins", "bool", True, "LOOM_RDNA4", kernels=["*_rdna4"], devices=["rdna4"], per_target=False, derive=_twins,
       from_pr="dual_target.py", enc=("1", "0"),
       doc="Emit the gfx12 twins K_rdna4 and the dual-target selection (LOOM_RDNA4=0: gfx11 programs only, byte-identical to the "
           "pre-dual generator).")
CHOICE("target.dual.select", "str", "launch", "LOOM_DUAL_SELECT", kernels=[], rows=ROWS_ALL, devices=["rdna3"], per_target=False,
       from_pr="dual_target.py (refactor, dualprog)", generator_defaults=("launch",),
       doc="Where a dual-target package selects between K and K_rdna4: launch = one template.apply per kernel launch (today); "
           "program = one template.apply per command program that launches a twin (the gfx11 provider is the plain gfx11-only body, as "
           "in a single-target build). No kernel text changes; only the command programs.")

# ======================================================================================================================
# Execution settings of one (model, device) preset (model.json): resolved per device, never part of a program
# ======================================================================================================================
EXEC("exec.prefill_capacity", "int", "Prefill chunk (tokens) of the device's preset; a larger chunk takes more scratch beside the weights "
     "(r9700 fit: 4096 for hidden <= 2048, 1024 for the 27B hybrids).", rows=ROWS_PREFILL, from_pr="#609 #617 #619 #630")
EXEC("exec.context_capacity", "int", "Context tokens one request holds in the device's preset (below the native context where the "
     "memory pool cannot hold the native KV).", rows=ROWS_ALL)
EXEC("exec.pool_capacity", "int", "KV pool positions the device's preset commits (fixed state backing commits it eagerly). hrx-loom 0.2.0 rejects the card field "
     "(the pool is the --memory_bytes budget; --pool_backing=fixed makes it context_capacity): kept only so v0.1-era scope files stay valid, never written to a card.", rows=ROWS_ALL)
EXEC("exec.rows", "int", "Concurrent rows (streams) of the device's preset.", rows=ROWS_STREAMS)
EXEC("exec.draft_depth", "int", "Draft depth of the device's preset (0: undrafted).", rows=ROWS_DECODE)
EXEC("exec.cache_target_bytes", "int", "Sizing target (bytes) of loom-serve's automatic prefix/session caches in the device's preset (hrx-loom 0.2: default 2 GiB, "
     "allocated eagerly by --pool_backing=fixed; 0 disables them). A model whose fixed-backing fit leaves less than that beside weights, KV and workspace "
     "(R9700 Flint, Muse-Glimmer-30B: RESOURCE_EXHAUSTED at the first request, image/draft components unavailable) sets a smaller one.", rows=ROWS_ALL)
EXEC("exec.context_cap", "int", "A measured limit on the context the device serves (the retained prefix state of the 8k/16k rows leaves "
     "no room for a longer request); lowers the fit's served context, which sets context_capacity and pool_capacity.", rows=ROWS_ALL)

# ======================================================================================================================
# Decode attention (attn_rules.py, gen_llm.py)
# ======================================================================================================================
_DA = dict(kernels=["g_attention_split", "g_attention_combine"], rows=ROWS_DECODE + ROWS_STREAMS)
CHOICE("decode.attn.full_blocks", "bool", False, "ATTN_FULL_BLOCKS", per_target=True, devices=["rdna4"],
       kernels=["g_attention_split_rdna4"], rows=ROWS_DECODE + ROWS_STREAMS, from_pr="1145",
       doc="Ordinary paged hd64 decode: bypass per-key masks on fully visible blocks; unsigned edge predicates preserve arithmetic.")
CHOICE("decode.attn.unroll", "int", 16, "ATTN_UNROLL", doc="Keys per decode-attention block.", **_DA)
CHOICE("decode.attn.butterfly", "bool", True, "ATTN_BFLY", doc="A block's key scores in one butterfly all-reduce.", **_DA)
CHOICE("decode.attn.paged", "bool", True, "ATTN_PAGED", doc="16-aligned key blocks, one page lookup each.", **_DA)
CHOICE("decode.attn.ilp", "int", 1, "ATTN_ILP", doc="Independent softmax chains per block (2, 4: no gain on Qwen3.8-27B at 32k).", **_DA)
CHOICE("decode.attn.kv_once", "bool", True, "ATTN_GQA", rows=["decode_32k", "decode_2k"],
       kernels=["g_attention_split"], from_pr="gqa_split",
       doc="Decode split kernel (non-drafting, head_dim 128): K/V loaded once for the G query heads of a KV head; the per-head "
           "body stays under decode.attn.kv_once.from_keys visible keys. Qwen3-4B decode_32k 24.5 -> 28.3 tok/s.")
CHOICE("decode.attn.kv_once.from_keys", "int", 4096, "ATTN_GQA_FROM", rows=["decode_32k"], kernels=["g_attention_split"],
       doc="Visible keys from which the K/V-once body is taken.")
CHOICE("decode.attn.kv_once.drafted", "bool", True, "ATTN_GQA_DRAFT", rows=["decode_32k"], kernels=["g_attention_split"],
       doc="Drafting packages carry the K/V-once body too (Flint 50.1 -> 57-61, Llama-3.1-8B 24.9 -> 29.7 at decode_32k). "
           "False: non-drafting packages only. Not with the segment path.")
CHOICE("decode.attn.kv_once.d256", "bool", True, "ATTN_GQA_D256", per_target=True, rows=["decode_32k"], kernels=["g_attention_split"],
       doc="head_dim 256 (qwen35 hybrids, 252 VGPRs) carries the K/V-once body in drafting packages too.")
CHOICE("decode.attn.kv_once.block_keys", "int", 8, "ATTN_GQA_U", rows=["decode_32k"], kernels=["g_attention_split"],
       doc="Keys per block of the K/V-once body (Qwen3-30B-A3B kv4 35.2 -> 36.5 tok/s vs 16).")
CHOICE("decode.attn.span.tokens", "int", 4, "ATTN_MT", rows=["decode_32k", "decode_2k"], kernels=["g_attention_split"],
       doc="Verify-span tokens per wave (gen_llm.multi_token_split).")
CHOICE("decode.attn.span.from_pos", "int", 1024, "ATTN_MT_FROM", rows=["decode_32k", "decode_2k"],
       kernels=["g_attention_split"], doc="Position from which the multi-token span path is taken.")
CHOICE("decode.attn.span.min_kv_heads", "int", 8, "ATTN_MT_MIN_KV", rows=["decode_32k", "decode_2k"],
       kernels=["g_attention_split"],
       doc="KV heads from which the span path is taken (8 KV heads +8..2x, 4 KV heads 0/-15 %, <= 2 KV heads -6..-27 %).")
CHOICE("decode.attn.segment", "bool", True, "SEG_ATTN", rows=["decode_32k"], kernels=["g_attention_split", "g_attention_combine"],
       doc="Drafting builds with <= 2 KV heads and head_dim 64/128 decode long spans in segments (False: split kernel everywhere).")
CHOICE("decode.attn.segment.min_ctx", "int", 8192, "SEG_MIN", rows=["decode_32k"], kernels=["g_attention_split"],
       doc="Visible keys from which the segment path applies.")
CHOICE("decode.attn.segment.max_tokens", "int", None, "SEG_TS_MAX", rows=["decode_32k"], kernels=["g_attention_split"], dynamic=True,
       doc="Longest segment (tokens read together); unset: 3 at head_dim 128, else 64.")

# ======================================================================================================================
# Decode matvecs and the head (gen_llm.py, mv_rows.py)
# ======================================================================================================================
_MV = dict(kernels=["g_mv*"], rows=ROWS_DECODE + ROWS_STREAMS)
CHOICE("decode.mv.act_q8", "bool", True, "ACT_Q8", kernels=["g_quant_q8", "g_mv*_q8", "g_head_mv*"], rows=ROWS_ALL,
       doc="Dense decode matvecs read int8 activations (integer dot4), like llama.cpp mul_mat_vec_q.")
CHOICE("decode.mv.butterfly_perm", "bool", True, "MV_BFLY_PERM", doc="Butterfly permutation in the int8 matvec reduction.", **_MV)
CHOICE("decode.mv.head_chunk", "int", 16, "MV_CHUNK", kernels=["g_head_mv*"], rows=ROWS_DECODE + ROWS_STREAMS,
       doc="Chunks of the int8 head matvec (8 measured no faster).")
CHOICE("decode.head.butterfly_perm", "bool", True, "HEAD_BFLY_PERM", kernels=["g_head_mv*"], rows=ROWS_DECODE + ROWS_STREAMS,
       doc="Butterfly permutation in the head matvec.")
CHOICE("decode.mv.adapt", "bool", True, "MV_ADAPT", doc="Halve R while a matvec grid is under-filled (False: R=16).", **_MV)
CHOICE("decode.mv.wg1", "int", 4096, "MV_WG1", doc="Workgroup target of 1-token matvecs.", **_MV)
CHOICE("decode.mv.wg4", "int", 128, "MV_WG4", doc="Workgroup target of 4-token matvecs. Setting it makes 1- and 4-token rounding "
       "differ (consistency.py fails): record it in the report.", **_MV)
CHOICE("decode.mv.wg_long", "int", 640, "MV_WG_LONG", doc="Workgroup target of 1-token matvecs with K >= 8192.", **_MV)
CHOICE("decode.mv.q8_r16", "bool", True, "MV_Q8_R16", doc="Int8 multi-token matvecs with 16 rows when n % 16 == 0.", **_MV)
CHOICE("decode.mv.q8_r32", "bool", True, "MV_Q8_R32", doc="Int8 16-token matvecs with 32 rows (pair kernels / K >= 8192).", **_MV)
CHOICE("decode.spec.gdn_copy_split", "int", 1, "GDN_COPY_SPLIT", kernels=["g_gdn_capture*", "g_gdn_restore*"], rows=ROWS_DECODE, devices=["rdna4"], per_target=True,
       doc="Workgroups per token of the gated delta-net capture / restore copies of a drafted step (gfx12 twin; 1 = the one-workgroup-per-token "
           "kernel on both targets). R9700 Qwen3.8-27B MTP 3: 8 = -0.8 ms of a 40 ms step.")
CHOICE("decode.spec.gdn_fold", "bool", False, "GDN_FOLD", enc=("1", None),
       kernels=["g_gdn_conv_bank", "g_gdn_recurrence_rec*", "g_gdn_fold*", "g_gdn_capture*", "g_gdn_restore*"], rows=ROWS_DECODE, from_pr="mxfp4st-ogdn",
       doc="Drafted verification of a gated delta-net hybrid without capture / restore / replay: the verification records each speculative "
           "step's fold factors (raw conv input + key, update scalar per value column, decay per head) and, after acceptance, one launch "
           "folds the accepted steps into every layer's state and one commits the conv histories (gen_gdn.fold_kernels). Published state "
           "bit-identical to the replay's; the verify scratch holds 8 rows per span instead of every epoch token's replay inputs.")
CHOICE("decode.spec.gdn_fold_defer", "bool", False, "GDN_FOLD_DEFER", enc=("1", None),
       kernels=["g_gdn_recurrence_rec*", "g_gdn_fold*", "g_gdn_pend*", "g_gdn_flush"], rows=ROWS_DECODE, from_pr="mxfp4st-defold",
       doc="With decode.spec.gdn_fold: the fold of a drafted verification is deferred. The accepted steps' factors become the writer "
           "slot's pending fold (each slot layer holds 8 steps of keys + dk + decay); the next verification's recurrence applies them in "
           "registers and writes the folded state back (one state read per drafted step instead of two), and every other reader (advance "
           "epochs, publishing spans, forks) applies them first (g_gdn_flush + g_gdn_pend_clear at the program start). Bit-identical to "
           "the eager fold (gen_gdn.defer_kernels).")
CHOICE("decode.mv.pair_proj", "str", "qg", "PAIR_PROJ", kernels=["g_mv*_pair"], rows=ROWS_DECODE + ROWS_STREAMS,
       doc="Paired projections: qg (q + gate), kv (k + v), 1 (both), 0 (off). k + v measured no faster end to end.")

# ======================================================================================================================
# Prefill: GEMMs, attention, row scales
# ======================================================================================================================
_PG = dict(kernels=["g_gemm_*"], rows=ROWS_PREFILL)
CHOICE("prefill.gemm.m256", "bool", True, "GEMM_M256", doc="256-row GEMM kernels (_m256) beside the 64/128-row ones.", **_PG)
CHOICE("prefill.gemm.q4_0_native", "bool", False, "GEMM_Q4_0_NATIVE", kernels=["g_gemm_q4_0_*", "g_moe_*q4_0*"], rows=ROWS_PREFILL,
       from_pr="gemma-qat-perf",
       doc="Dense Q4_0 GEMMs stage their 16-value half-block straight to F16 (moe_q4_0.dequant16: the values of @g_q4_0_values + "
           "fptrunc) and get the 128/256-row kernels; off: the shared value helper and the 64-row tile at every size (quant/q4_0 node).")
CHOICE("prefill.gemm.n_fastest", "bool", True, "GEMM_N_FASTEST", derive=_n_fastest, from_pr="gemm_n_fastest",
       doc="Output-column tiles fastest. cfg gemm_n_fastest is the model's own value (Glimmer-30B +4/+7/+6 % at 2k/8k/32k; "
           "Flint -4 % at 8k).", **_PG)
CHOICE("prefill.gemm.skip_padding", "bool", True, "GEMM_SKIP_PADDING",
       doc="A row tile whose rows all have scale 0 (padding rows) is skipped (gen_llm launches the GEMMs with it on).",
       generator_defaults=("1", "0"), **_PG)
CHOICE("prefill.gemm.f16o_swiglu", "bool", False, "GEMM_F16O", kernels=["g_gemm_*swiglu*_rdna4"], rows=ROWS_PREFILL,
       devices=["rdna4"], derive=_f16o, enc=("swiglu", "0"), from_pr="#656", dynamic=False,
       doc="gfx12 f16-output swiglu group with its row scale (the R9700 prefill stack of #656). Opt-in per model: llm_config "
           "rdna4_f16_ffn is its default there. Also turns on prefill.attn.flash2_trim and prefill.attn.flash2_b2 (cfg-driven).",
       generator_defaults=("",))
CHOICE("decode.mxfp4s.act_q8", "bool", False, "MXFP4S_Q8", kernels=["g_mv_mxfp4s_*_q8*", "g_mxfp4s_icodes"], rows=ROWS_DECODE + ROWS_STREAMS, from_pr="mxfp4st-kern",
       doc="MXFP4 split-plane dense matvecs take int8 activations (ACT_Q8's dot4 path, as Q8_0 / the K-quants) instead of f32 conversions and fused multiply-adds: the 4-token verify "
           "matvec of a drafted decode is ALU-bound on the f32 path (the GGUF MXFP4 Qwen3.8-27B measured 47 tok/s drafted vs 74 with int8 activations). Opt-in per model.")
CHOICE("decode.q4_0.act_q8", "bool", False, "Q4_0_Q8", kernels=["g_mv_q4_0_*", "g_head_mv*", "g_moe_*q4_0*", "g_quant_q8"],
       rows=ROWS_DECODE + ROWS_STREAMS, from_pr="gemma-qat-perf",
       doc="Q4_0 dense matvecs, head and expert matvecs take int8 activations (ACT_Q8's dot4 path with @g_q4_0_icodes, as llama.cpp's "
           "vec_dot_q4_0_q8_1) instead of f32 values and a per-row subgroup reduction per token, which is ALU-bound at 4 tokens "
           "(quant/q4_0 node).")
CHOICE("decode.verify.fuse", "str", None, "VERIFY_FUSE", kernels=["g_pmv_*"], rows=ROWS_DECODE, from_pr="mxfp4st-ostep",
       doc="Drafted verify epochs of 2-4 tokens on the int8 matvec path run fused layers (starter gen_pstep.py; the layer as generated stays "
           "for every other epoch, every value is the launch chain's: kernel tests compare with equality). swq: gate + up + g_silu_mul + "
           "g_quant_q8 of the FFN in one launch (32-row workgroups whose epilogue writes the down projection's int8 input; kbench pstep_r9700 "
           "43811425655: FFN block 262 -> 251 us; in the model, R9700 run 43812485595: 157.3 us vs 2 x 78.9, decode tie). swqn: swq and every g_post_add_norm + g_quant_q8(h) pair as one g_post_add_norm_q8 launch (7.2 us vs 2.9 + 2.7: a loss). full: every projection takes its producer as a prologue (rmsnorm / f32 / silu + quant) and "
           "the out / down projections accumulate into the residual (7 launches -> 2 per FFN, but 261 -> 335+ us: the prologue is paid per workgroup).")
CHOICE("decode.verify.persist", "int", 0, "MV_PERSIST", kernels=["g_pmv_*"], rows=ROWS_DECODE, from_pr="mxfp4st-ostep",
       doc="decode.verify.fuse's matvecs as persistent row loops: min(G, row groups) workgroups each walk row groups id, id + G, ... "
           "(the prologue runs once per workgroup instead of once per row group); 0: one workgroup per row group.")
CHOICE("prefill.gemm.mxfp4s_f8", "bool", False, "GEMM_MXFP4S_F8", kernels=["g_gemm_mxfp4s_*_f8_*", "g_row_scale_f16*_f8_*", "g_post_add_norm_rs*_f8_*"], rows=ROWS_PREFILL,
       devices=["rdna4"], from_pr="mxfp4st-kern",
       doc="gfx12: MXFP4 split-plane (mxfp4s) prefill GEMMs on fp8 WMMA (v_wmma_f32_16x16x16_fp8_bf8): the nibble and scale planes are read as stored, weights decoded to bf8 "
           "exactly in LDS, activations fp8 e4m3 with a per-row power-of-two scale written by the row-scale producers' _f8_rdna4 twins (lossy activations, gated against the "
           "golden reference). Opt-in per model; every block exponent of the checkpoint must be in 107..132 (scan_mxfp4s_exponents.py). The gfx11 programs are unchanged.")
CHOICE("prefill.gemm.mxfp4s_f8_kernel", "str", None, "GEMM_MXFP4S_F8_KERNEL", kernels=["g_gemm_mxfp4s_*_f8_*"], rows=ROWS_PREFILL, devices=["rdna4"], from_pr="mxfp4st-ogemm",
       doc="The design of prefill.gemm.mxfp4s_f8's GEMM kernels (gen_mxfp4s.F8_KERNELS; None = v1, the #879 kernel). lvs: the store GEMMs load their operands lane-vector "
           "(one 16-byte load per lane covers two K substeps: activations global_load_b128 with immediate offsets, weights ds_load_b128; both operands in the same permuted K "
           "order) and issue the next K pair's loads before this pair's WMMAs; the SwiGLU pair stays v1. lv: lane-vector without the pipelining (SwiGLU at 64-row waves); "
           "ws: wave-specialized (4 producer waves decode the next step into a second LDS buffer). kbench gemm_mxfp4s (R9700): lvs store -5..-6 % at 1024 rows, lv +3..7 %, ws +8..19 %.")
CHOICE("prefill.attn.no_flash2", "bool", False, "NO_FLASH2", kernels=["*_attention_flash2*"], rows=ROWS_PREFILL, enc=("1", "0"),
       doc="No prefill flash attention (every token takes the listed split-K path).")
CHOICE("prefill.gemm.off", "bool", False, "NO_GEMM", kernels=["g_gemm_*"], rows=ROWS_PREFILL, enc=("1", None), experiment=True,
       doc="Debug/A-B: prefill epochs never take the GEMM path (the variable's presence turns it off).")
CHOICE("prefill.last_layer_rows", "bool", False, "LAST_OUT", rows=ROWS_PREFILL,
       doc="The last layer's post-attention work runs only on the rows the head reads (llama.cpp inp_out_ids). Opt-in: "
           "Qwen3.8-27B +1 %, Glimmer-30B -3.5 %. Off for MTP / assistant drafters.")
CHOICE("prefill.row_scale_live", "bool", True, "ROW_SCALE_LIVE", kernels=["g_row_scale_f16*"], rows=ROWS_PREFILL,
       doc="Live row scale (padding rows scale 0).")
CHOICE("prefill.post_add_row_scale", "bool", True, "POST_ADD_ROW_SCALE", kernels=["g_post_add_norm*"], rows=ROWS_PREFILL,
       doc="Fused post-add + row scale.")
CHOICE("prefill.live_row_skip", "bool", True, "LIVE_ROW_SKIP", rows=ROWS_PREFILL,
       doc="Rows with scale 0 skipped in the row-local post-attention kernels.")

CHOICE("prefill.prune_shared", "bool", False, "PRUNE_SHARED", kits=["gemma4"], rows=ROWS_PREFILL,
       kernels=["g_token_info_tail", "g_attention*", "g_gemm*"], from_pr=554,
       doc="Gemma4 shared-KV layers retain only the last live row per span in prefill; verify stays complete.")

# ----- prefill flash attention (gen_flash2.py, gen_flash2s.py, mv_rows.py) -----------------------------------------------
_FL = dict(kernels=["*_attention_flash2*"], rows=ROWS_PREFILL)
_FLR4 = dict(kernels=["*_attention_flash2*_rdna4"], rows=ROWS_PREFILL, devices=["rdna4"])
CHOICE("prefill.attn.flash2_split", "bool", True, "FLASH2_SPLIT", doc="head_dim 256/512 without sinks takes gen_flash2s.py's "
       "split-dims kernel (False: the generic one).", **_FL)
CHOICE("prefill.attn.flash2_kt", "int", 32, "FLASH2_KT", kit_legacy={"gemma4": 16}, doc="Keys per prefill tile (gemma4 kit: 16, "
       "head_dim 256 + window spills at 32).", **_FL)
CHOICE("prefill.attn.flash2_ts", "int", None, "FLASH2_TS", dynamic=True, doc="Forces one query-tile kernel (default: 2 tiles when "
       "head_dim <= 128 and G <= 2).", **_FL)
CHOICE("prefill.attn.flash2_ts_depth", "int", None, "FLASH2_TS_DEPTH", dynamic=True,
       doc="Position from which two 16-token tiles per workgroup run (0: off; default 16384 at head_dim 128, 32768 at 64).", **_FL)
CHOICE("prefill.attn.flash2_do", "int", None, "FLASH2_DO", kits=["gemma4"], dynamic=True, doc="Output dims per pass (gemma4 kit, "
       "FLASH512_DO=256: two output passes).", **_FL)
CHOICE("prefill.attn.flash512_do", "int", None, "FLASH512_DO", kits=["gemma4"], dynamic=True, doc="gemma4 kit: FLASH512_DO=256 runs the "
       "head_dim 512 global layers through gen_flash2.py's two output passes.", **_FL)
CHOICE("prefill.attn.flash2_edge_zero", "bool", False, "FLASH2_EDGE_ZERO", doc="Zero-fill the edge tile instead of masking.", **_FL)
CHOICE("prefill.attn.flash2_peel", "bool", True, "FLASH2_PEEL", per_target=True, from_pr="#771",
       doc="The masked path only in the last key tile (flash2_peel.py). Per target: the gfx11 kernel and its gfx12 twin may differ "
       "(FLASH2_PEEL, FLASH2_PEEL__RDNA4).", **_FL)
CHOICE("prefill.attn.flash2_entry", "str", None, "FLASH2_ENTRY", dynamic=True, experiment=True,
       doc="plain | bcast: the listed entry broadcast from lane 0 (default bcast when NF > 8).", **_FL)
CHOICE("prefill.attn.flash2_swa_twin", "str", None, "FLASH2_SWA_TWIN", dynamic=True,
       doc="s: the gfx12 sliding-layer twin is gen_flash2s.py's split-dims kernel (gemma4 kit: on when FLASH2_TRIM is 1).", **_FLR4)
CHOICE("prefill.attn.flash2_trim", "bool", False, "FLASH2_TRIM", derive=_trim, from_pr="#656",
       doc="gfx12 flash twin trims (flash2_trim.py) as one switch; the per-trim switches below default to it.", **_FLR4)
CHOICE("prefill.attn.flash2_b2", "str", None, "FLASH2_B", derive=_flash_b, from_pr="#656", generator_defaults=("2", None),
       doc="2: key-split two-heads-per-wave global-layer flash twin on gfx12 (flash2_b2.py): head_dim 512 / 8 query heads per KV head (gemma4 E2B) or head_dim 256 / 6 "
           "(Qwen3.5-class 27B, three head pairs per KV head). 3: three heads per wave at head_dim 256 / 6 (256 VGPRs: measured slower than 2).", **_FLR4)
CHOICE("prefill.attn.flash2_b2_fm", "bool", None, "FLASH2_B2_FM", from_pr="mxfp4st-oattn2",
       doc="With prefill.attn.flash2_b2 = 2 at head_dim 256 / 6 query heads per KV head: the whole-tile key-split loop uses a fixed softmax reference "
           "(the max of the first and the diagonal page; no running max, no per-block max exchange or O rescale, one barrier per 64 keys; when a "
           "lane has a later score above its reference + 15 in log2 units the workgroup reruns the loops with that lane's true max and every other "
           "lane's reference unchanged, so a token's bits depend on that token only), V^T prefetch, exp2 and an unmasked loop below the diagonal; "
           "whole- and partial-tile kernels alike (flash2_b2.py fmax).", **_FLR4)
CHOICE("prefill.attn.rope_qg", "bool", False, "ROPE_QG", kernels=["g_qk_rope_store*", "g_split_qgate"], rows=ROWS_ALL,
       doc="Gated attention with a fused q|gate projection (qwen35 family): the norm + RoPE kernel reads the query heads from the projection's rows and copies the gate half "
           "(g_qk_rope_store_qg), so g_split_qgate and a pass over q disappear. Full-attention layers only; no projection biases, no sliding layers.")
for _k, _env, _what in (("sx", "FLASH2_SX", "S partials exchanged as the lanes' own vectors"),
                        ("lazy", "FLASH2_LAZY", "O^T rescale only when a column's max changed"),
                        ("kvh", "FLASH2_KVH", "loop-invariant half of the paged-KV lookup hoisted"),
                        ("vtf", "FLASH2_VTF", "flat-index V^T stores"),
                        ("hot", "FLASH2_HOT", "the hot copy skips the validity / mask selects")):
    CHOICE(f"prefill.attn.flash2_trim.{_k}", "bool", None, _env, dynamic=True, doc=f"gfx12 twin trim, defaults to FLASH2_TRIM: {_what}.",
           **_FLR4)
CHOICE("prefill.attn.flash2_vp", "bool", False, "FLASH2_VP", experiment=True, doc="Two keys per staging lane: one 32-bit V^T "
       "store per dim pair (flash2_vpack.py); gfx12 twin only.", **_FLR4)
CHOICE("prefill.attn.flash2_pf", "bool", False, "FLASH2_PF", experiment=True, doc="The next K/V tile's loads in flight during this "
       "tile's math (flash2_prefetch.py); gfx12 twin only.", **_FLR4)
CHOICE("prefill.attn.flash2_kt256", "int", None, "FLASH2_KT256", experiment=True, doc="Experiment: 16-key tiles at head_dim 256.", **_FL)
CHOICE("prefill.attn.flash2_ng512", "int", None, "FLASH2_NG512", experiment=True,
       doc="Experiment: query-head groups per KV head at head_dim 512.", **_FL)
CHOICE("prefill.attn.flash512", "bool", True, "FLASH512", kits=["gemma4"], kernels=["*_attention_flash2*"], rows=ROWS_PREFILL,
       doc="gemma4 kit: head_dim 512 global layers use flash2 (False: the split-K path).")
CHOICE("prefill.ple.gemm", "bool", True, "PLE_GEMM", kits=["gemma4"], kernels=["g_gemm_*"], rows=ROWS_PREFILL,
       doc="gemma4 kit: the per-layer-embedding projection takes the GEMM at prefill (False: matvec).")
CHOICE("prefill.gemm.min_tokens", "int", 64, "GEMM_MIN_TOKENS", kits=["lfm2moe"], kernels=["g_gemm_*"], rows=ROWS_PREFILL,
       doc="lfm2moe kit: epoch tokens from which the GEMM path is taken.")

# ======================================================================================================================
# MoE (gen_moe.py, gen_llm.py; lfm2moe and flint-moe reference kits)
# ======================================================================================================================
_MO = dict(kernels=["g_moe_*"], rows=ROWS_DECODE + ROWS_PREFILL + ROWS_STREAMS)
CHOICE("moe.decode.rows_per_wg", "int", None, "MOE_R", dynamic=True, kit_legacy={"lfm2moe": 8},
       doc="Rows of one expert matvec per workgroup; default aims at ~3000 workgroups within 2..8 (Qwen3-30B-A3B decodes "
           "10 % faster at R=2 than R=8).", **_MO)
CHOICE("moe.decode.act_q8", "bool", True, "MOE_Q8", doc="Decode expert matvecs read int8 activations (with decode.mv.act_q8).", **_MO)
CHOICE("moe.route.unroll", "int", 4, "MOE_ROUTE_U", doc="Experts per wave scored together in the router.", **_MO)
CHOICE("moe.route.tokens_per_wg", "int", 0, "ROUTE_TR", doc="Router tokens per workgroup (0: the largest of 1/2/4/8 whose LDS fits 48 KiB).",
       **_MO)
CHOICE("moe.fold", "bool", True, "MOE_FOLD", doc="Fold the routing weights into the expert matvec.", **_MO)
CHOICE("moe.prefill.skip_empty", "bool", True, "MOE_SKIP_EMPTY", doc="Waves of empty expert tiles exit early.", **_MO)
CHOICE("moe.prefill.prefetch", "bool", True, "MOE_PREFETCH", doc="Stage prefetch in the grouped expert GEMMs.", **_MO)
CHOICE("moe.prefill.n_fastest", "str", "swiglu,store", "MOE_N_FASTEST", kit_legacy={"lfm2moe": "1"},
       doc="Grouped-GEMM modes with output-column tiles fastest (Qwen3.6-35B-A3B prefill +74 % at 1k; lfm2moe +24-30 %). The "
           "lfm2moe kit reads 1/0 and MOE_N_FASTEST_MODES.", **_MO)
CHOICE("moe.prefill.n_fastest_modes", "str", "gu,dn", "MOE_N_FASTEST_MODES", kits=["lfm2moe"], doc="lfm2moe kit: the modes MOE_N_FASTEST "
       "applies to.", **_MO)
CHOICE("moe.route.par", "bool", True, "MOE_ROUTE_PAR", kits=["lfm2moe"], doc="lfm2moe kit: parallel router.", **_MO)
CHOICE("moe.route.waves", "int", 32, "MOE_ROUTE_WAVES", kits=["lfm2moe"], doc="lfm2moe kit: router waves.", **_MO)
CHOICE("moe.flint.fmt", "str", "q4_k", "FLINT_MOE_FMT", kits=["flint-moe"], doc="loom-optimize flint-moe example: expert format.", **_MO)
CHOICE("moe.flint.tile", "int", 32, "MOE_TILE", kits=["flint-moe"], doc="flint-moe example: rows per tile.", **_MO)
CHOICE("moe.flint.down_groups", "int", 8, "DOWN_GROUPS", kits=["flint-moe"], doc="flint-moe example: down-projection groups.", **_MO)

# ======================================================================================================================
# Hybrid gated delta-net (gen_gdn.py)
# ======================================================================================================================
_GD = dict(kernels=["g_gdn_*"], rows=ROWS_DECODE + ROWS_PREFILL + ROWS_STREAMS)
CHOICE("hybrid.gdn.channel_blocks", "int", None, "GDN_CB", dynamic=True, from_pr="gdn_cb",
       doc="Channel blocks per token/span in the history commit (default (C + 255) // 256; 1: the original serial grid).", **_GD)
CHOICE("hybrid.gdn.conv_unroll", "int", 8, "GDN_CONV_UNROLL", doc="Conv unroll.", **_GD)
CHOICE("hybrid.gdn.prefetch", "bool", True, "GDN_PREFETCH", doc="Recurrence prefetch.", **_GD)
CHOICE("hybrid.gdn.row_lanes", "int", 16, "GDN_ROW_LANES", derive=_mtp_rows, from_pr="#350",
       doc="Lanes per value column of the row-split recurrence (MTP-drafted hybrids: 8, one form for every span).", **_GD)
CHOICE("hybrid.gdn.cols", "int", 2, "GDN_COLS", derive=_mtp_cols, from_pr="#350",
       doc="Value-column width of the row-split recurrence (MTP-drafted hybrids: 4).", **_GD)
CHOICE("hybrid.gdn.chunk", "bool", False, "GDN_CHUNK", devices=["rdna4"], rows=ROWS_PREFILL, kernels=["g_gdn_chunk_rdna4", "g_gdn_recurrence*"],
       from_pr="q38r9700 gdnpf",
       doc="gfx12 chunked (WY) gated delta-net prefill (gen_gdn_chunk.py): spans of 64+ tokens run g_gdn_chunk_rdna4 (f16 WMMA, f32 state), the "
           "sequential kernel's gfx12 twin the shorter ones; the gfx11 kernels do not change (128 x 128 heads only).")
CHOICE("prefill.gdn.fused2", "bool", False, "GDN_FUSED2", devices=["rdna4"], rows=ROWS_PREFILL, kernels=["g_gdn_fa_rdna4", "g_gdn_fb_rdna4"],
       from_pr="mxfp4st olayer",
       doc="gfx12 fused gated delta-net prefill in two launches (gen_gdn_fused.py): g_gdn_fa_rdna4 (conv + SiLU, q/k norms, gates, the WY solve "
           "and W = T beta e^gc K, U = T beta V, P per (value head, 32-token chunk), fully parallel) and g_gdn_fb_rdna4 (the serial state pass "
           "d = U - W S, O = e^gc Q S + P d, S update, norm-gate epilogue, conv history commit) replace conv, commit, qk_norm, the chunk kernel and "
           "norm_gate in GEMM-shape epochs; gfx11 and short epochs keep the chain (128 x 128 heads, conv 4).")
CHOICE("prefill.gdn.fused2_min", "int", 64, "GDN_FUSED2_MIN", devices=["rdna4"], rows=ROWS_PREFILL, kernels=["g_gdn_fa_rdna4", "g_gdn_fb_rdna4"],
       from_pr="mxfp4st olayer2",
       doc="With prefill.gdn.fused2: the smallest epoch (rows) that takes the fused pair: 64 = every GEMM-shape epoch; 128 = 64-row epochs (prompts and "
           "tails under 64 tokens) keep the chain, whose spans under 64 tokens run the exact sequential recurrence.")
CHOICE("prefill.gdn.ab_f8", "bool", False, "GDN_AB_F8", devices=["rdna4"], rows=ROWS_PREFILL, kernels=["g_ab_gemm_mxfp4s_store_pair_f8_m256_rdna4"],
       from_pr="mxfp4st olayer2",
       doc="MXFP4 split-plane hybrids with prefill.gdn.ab_q8 and the fp8 GEMMs (gen_mxfp4s.ab_f8): in GEMM-shape epochs whose rows are a multiple of 32 "
           "the 48-row GDN alpha / beta projections run as ONE paired skinny fp8 WMMA GEMM (32 x 48 tiles, both weights, K step 256) on the fp8 rows the "
           "producer twin already wrote for the qkv GEMM (%h8), not g_quant_q8 + two int8 16-token matvecs; gfx11 keeps the int8 launches.")
CHOICE("prefill.gdn.ab_q8", "bool", False, "GDN_AB_Q8", devices=["rdna4"], rows=ROWS_PREFILL, kernels=["g_quant_q8", "g_mv_mxfp4s_k5120_t16_q8"],
       from_pr="mxfp4st olayer",
       doc="MXFP4 split-plane hybrids, GEMM-shape epochs: the 48-row GDN alpha / beta projections run as the 16-token int8 matvec on h quantized "
           "once to int8, not as the f32 4-token matvec over h (gen_llm.ab_q8).")

# ======================================================================================================================
# Speculative decoding (gen_llm.py, draft/)
# ======================================================================================================================
_SP = dict(kernels=["g_spec_*", "g_dfl_*", "g_ng_*", "g_mtp_*"], rows=ROWS_DECODE + ROWS_STREAMS)
CHOICE("draft.verify.mm_from", "int", None, "SPEC_MM_FROM", dynamic=True,
       doc="Tokens from which a verify epoch takes the f16 WMMA GEMM (default 17 with int8 matvecs, else 5).", **_SP)
CHOICE("draft.verify.exact", "bool", False, "SPEC_EXACT", enc=("1", "0"),
       doc="Verify with the 4-token matvec at every width (decode's rounding) instead of the GEMM from 5 tokens.", **_SP)
CHOICE("draft.warm.mm_from", "int", 5, "DRAFT_MM_FROM", doc="The drafter's catch-up program takes the GEMM from this many tokens.", **_SP)
CHOICE("draft.propose.head", "str", "q4_0", "PROPOSE_HEAD", doc="q4_0: the proposals' head pass reads a load-time Q4_0 copy of the "
       "target's head (off: the head itself; q6_k: with PROPOSE_HEAD_ROWS the subset's rows stay Q6_K).", **_SP)
CHOICE("draft.propose.head_rows", "int", 0, "PROPOSE_HEAD_ROWS", kernels=["g_prep_*", "g_head_mv*_p40", "g_head_select_p"], rows=ROWS_DECODE + ROWS_STREAMS,
       doc="MTP proposals over a reduced draft vocabulary (0: off, the target's own head): the proposals' head pass reads a load-time Q4_0 "
       "copy of a Q6_K or BF16 head's rows [0, N) plus every added-token row (draft/prep_weights.py); verification keeps the head, so only acceptance moves.", **{})
CHOICE("draft.propose.block", "str", None, "MTP_BLOCK_PREP", kernels=["g_prep_*", "g_mv_*_q8*"], rows=ROWS_DECODE + ROWS_STREAMS,
       doc="MTP proposals of a safetensors checkpoint whose NextN block is BF16: the block's matrices (eh_proj, attention, FFN) are read from a "
       "load-time Q8_0 or Q4_0 copy (q8_0 | q4_0; empty: as stored). The block feeds proposals only (verification never reads it), so only "
       "acceptance moves (draft/prep_weights.py, weights.loom).", **{})
CHOICE("draft.propose.head_pieces", "bool", False, "PROPOSE_HEAD_PIECES", enc=("1", "0"), kernels=["g_prep_*", "g_head_mv*_p40_pc*"], rows=ROWS_DECODE + ROWS_STREAMS,
       doc="The proposal head's copy (draft.propose.head_rows, BF16 head) as several load-time prepared tensors of decode.head.piece_rows rows "
       "(draft/prep_weights.piece_entries): hrx-loom stages a prepared tensor's input whole through four lanes, so one N-row copy costs "
       "4 x N x 2H bytes of device budget while loading, pieces 4 x piece x 2H; one head launch per piece (g_head_mv*_pc<chunks>).", **{})
CHOICE("decode.head.verify", "str", None, "VERIFY_HEAD_PREP", kernels=["g_prep_*", "g_head_mv*_q8_pc*"], rows=ROWS_DECODE + ROWS_STREAMS,
       doc="q8_0: the verified head of a safetensors checkpoint with an untied BF16 head reads a load-time Q8_0 copy (int8 activations, "
       "decode.head.piece_rows-row prepared tensors) instead of the BF16 tensor: about half the head bytes per step, and the BF16 tensor is "
       "no longer resident. CHANGES the verified output (lossy weights): opt-in, gated (golden vs the reference, near-tie statistics).", **{})
CHOICE("decode.head.piece_rows", "int", 16384, "HEAD_PIECE_ROWS", kernels=["g_head_mv*_pc*"], rows=ROWS_DECODE + ROWS_STREAMS,
       doc="Rows per prepared piece of a pieced head (draft.propose.head_pieces, decode.head.verify); a multiple of 128.", **{})
CHOICE("draft.dflash.splits", "int", 32, "DFL_SPLITS", kernels=["g_dfl_attention", "g_dfl_ctx_split"], rows=ROWS_DECODE,
       doc="DFlash context-attention splits.")
CHOICE("draft.dflash.prep_weights", "bool", True, "DRAFTER_PREP", kernels=["g_dfl_*"], rows=ROWS_DECODE + ROWS_STREAMS,
       doc="A BF16/F16 drafter is used as Q8_0 at load (draft_weights.loom); False: as is.")
CHOICE("draft.ngram.max", "int", 4, "NGRAM_MAX", kernels=["g_ng_propose"], rows=ROWS_DECODE + ROWS_STREAMS,
       doc="Longest suffix compared by the n-gram drafter.")
CHOICE("draft.ngram.window", "int", 32768, "NGRAM_WINDOW", kernels=["g_ng_propose"], rows=ROWS_DECODE + ROWS_STREAMS,
       doc="History positions scanned per proposal.")

CHOICE("sampling.final_softcap", "bool", False, "FINAL_SOFTCAP", kernels=["g_head_select", "g_logits_export"], rows=[], from_pr="gemma-qat-clean",
       doc="The final logit softcap of a config that names it final_softcap (gemma4: 30) is applied where logit_softcap is: the top-64 sampling "
           "candidates (g_head_select) and the exported logits rows (g_logits_export) see cap * tanh(z / cap), as llama.cpp's gemma4 graph. "
           "Greedy tokens are unchanged (tanh is monotonic); without it temperature sampling and host logits use the uncapped, sharper "
           "distribution (logits gate: log-probs about 2x llama.cpp's, e.g. -19.6 vs -9.1).")
CHOICE("feature.logits_export", "bool", True, "LLM_LOGITS", kernels=["g_logits_export"], rows=[], from_pr="loom-skills#736",
       doc="The package carries the opt-in logits readback (g_logits_export, a card-config-guarded tail of @llm_advance, a guarded larger "
           "outputs allocation). Off by default at load (card config llm.logits); False drops the code from the package.")

# ======================================================================================================================
# Not choices: everything else the generators and their tools read from the environment
# ======================================================================================================================
NOT("kbench experiment options of the gfx12 flash kernels (flash_r9700 suite; the shipped variants are choices)", "FLASH2_B2_OPTS", "FLASH2_G6_OPTS")
NOT("input path or file selection (not a model-affecting knob)", "CATALOG", "LOOM_SKILLS", "LOOM_KB_ROOT", "LOOM_ROOT", "VENV", "HF", "HF_HOME",
    "HF_HUB_CACHE", "LD_LIBRARY_PATH", "LOOM_SERVE_STATE", "VISION_TEST_DIR", "FLASH2_OUT", "LLM_OUT")
NOT("transport between a generator and its subprocesses (set by the caller from cfg / the build)", "LLM_CONFIG", "LLM_DRAFT",
    "LLM_TOKENIZER", "GEMM_FORMATS", "GEMM_F16O_ON", "GEMM_KMETA", "GEMM_PREFIX", "FLASH2_TARGET", "FLASH2_ROLE", "FLASH2_ARCH",
    "FLASH2_NAME", "FLASH2_KV_FN", "FLASH2_SCALE", "FLASH2_POS_BYTES")
NOT("kernel-test selection (which cases make_kernel_tests.py writes), not a model choice", "FLASH_TESTS", "GEMM_TESTS", "MOE_TESTS",
    "MOE_GEMM_TESTS", "DRAFTER_TESTS", "KTEST_GEMM_ALL", "KTEST_GEMM_ERR", "KTEST_HEADS_SFX", "KTEST_LLM", "PREFILL_CONT",
    "DEVICE_ARCH", "VISION_TEST_RDNA4")
NOT("draft/tune.py harness or its recorded result (the tuned drafting config is llm_config drafter.*, the optimizer's choice)",
    "TUNE_BASELINE", "TUNE_CTX_MAX", "TUNE_KIND", "TUNE_LONG", "TUNE_NGRAM", "TUNE_PRIOR", "TUNE_REQUIRE_DRAFT", "TUNE_UNGATED",
    "SPEC_OCC_MIN", "SPEC_CTX_MAX", "MEMORY_GIB")
NOT("build input with a command-line twin: the drafting config of a package is llm_config drafter.* and the stamped draft block",
    "DRAFT", "DRAFTER", "DRAFT_DEPTH", "SPEC", "DRAFTER_REQUANT")
NOT("machine memory facts used to size context and pool in model.json (build.py memory_cap), not a model choice",
    "LOOM_MEMORY_CAP_GIB", "LOOM_RESERVED_GIB")
NOT("checkpoint selection (the quantization is never a knob: plain Q4_K_M as on the cards)", "QUANT_TIER")
NOT("credential of the Hugging Face range reads (inspect_safetensors.py), never part of a build", "HF_TOKEN")
NOT("catalog build switch: hand-written catalog packages are exempt from resolution (build.py hands over before it)", "VISION")
NOT("provenance of a snapshot build (the stamp records it), not a model choice", "LOOM_SKILLS_COMMIT")
NOT("profiling ablation or debug switch, never part of a published build", "FLASH2_ABL", "DBG_NO_FLASH", "DBG_NO_GEMM")


def registered_env():
    """Every environment variable name the registry knows (choices and not-choices)."""
    return set(BY_ENV) | set(NOT_CHOICES)


def choice_for_env(env):
    return BY_ENV.get(env)


if __name__ == "__main__":
    import json
    import sys
    if "--list" in sys.argv:
        for c in CHOICES.values():
            print(f"{c.key:48} {c.env or '-':22} {c.type:5} {c.legacy!r}")
    else:
        json.dump({"choices": [c.as_dict() for c in CHOICES.values()], "not_choices": NOT_CHOICES}, sys.stdout, indent=1)
        print()
