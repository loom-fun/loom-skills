"""Shared pieces of the sampled-testing tools (sample_plan, verdict, baselines, promote_check): the card rows, the per-row bands
(skills/loom-kb/method/methodology.json when it exists, else the calibrated defaults of plan Appendix A section 5), kernel -> row
and kernel -> family rules, stratum signatures, the impact adapter. Stdlib only.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

ROWS = ["decode", "decode_2k", "decode_32k", "prefill_1k", "prefill_2k", "prefill_8k", "prefill_16k", "prefill_32k", "streams4"]
DECODE_ROWS = ["decode", "decode_2k", "decode_32k"]
PREFILL_ROWS = ["prefill_1k", "prefill_2k", "prefill_8k", "prefill_16k", "prefill_32k"]
# perf.py without --long measures SHORT_ROWS, with --long LONG_ROWS; streams4 is the c4 step (devlab_perf.sh)
SHORT_ROWS = ["decode", "decode_2k", "prefill_1k", "prefill_2k"]
LONG_ROWS = ["decode_32k", "prefill_8k", "prefill_16k", "prefill_32k"]
sys.path.insert(0, os.path.join(ROOT, "skills", "loom-kb"))
import registry  # noqa: E402

# card GPU id -> target family of the programs ("gfx11", "gfx12"): from the device registry (a device row adds its entry here)
GPUS = {k: registry.family_of(k)["gfx_generic"].split("-")[0] for k in registry.devices()}
FAMILY_OF = {v: k for k, v in GPUS.items()}

# plan Appendix A section 5 (calibrated on the 63 phase-3 A/B files): band = PASS inside 1 +- band, reject = T1 threshold
DEFAULT_BANDS = {
    "prefill_8k": (0.02, 0.95), "prefill_16k": (0.02, 0.95), "prefill_32k": (0.02, 0.95),
    "prefill_1k": (0.03, 0.94), "prefill_2k": (0.03, 0.94),
    "decode": (0.03, 0.92), "decode_32k": (0.03, 0.92),
    "decode_2k": (0.04, 0.92), "streams4": (0.04, 0.92),
}
DEFAULT_BIMODAL = ["Qwen2.5-0.5B", "Qwen3-0.6B"]
DEFAULT_RATCHET = {"confirming_runs": 2}


def load_methodology(path=None):
    """{'bands': {row: (band, reject)}, 'bimodal': [name substrings], 'ratchet': {...}, 'source': path|'defaults'}.
    Reads skills/loom-kb/method/methodology.json (engineer "regs") when it exists; any key it lacks falls back to the defaults.
    Accepted shapes for the per-row table: rows[row] = {band, reject|t1_reject}, or bands[row] = {...} / number."""
    path = path or os.path.join(ROOT, "skills", "loom-kb", "method", "methodology.json")
    bands = dict(DEFAULT_BANDS)
    out = {"bands": bands, "bimodal": list(DEFAULT_BIMODAL), "ratchet": dict(DEFAULT_RATCHET), "source": "defaults"}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        d = json.load(fh)
    try:
        out["source"] = os.path.relpath(path, ROOT)
    except ValueError:      # another drive (Windows)
        out["source"] = path
    table = d.get("rows") or d.get("bands") or {}
    for r, e in (table.items() if isinstance(table, dict) else []):
        if r in bands:
            if isinstance(e, dict):
                b = e.get("band", bands[r][0])
                rj = e.get("reject", e.get("t1_reject", bands[r][1]))
            else:
                b, rj = float(e), bands[r][1]
            bands[r] = (float(b), float(rj))
    bm = d.get("bimodal_models") or d.get("bimodal")
    if isinstance(bm, list):
        out["bimodal"] = [x if isinstance(x, str) else x.get("model", "") for x in bm]
    rt = d.get("ratchet")
    if isinstance(rt, dict):
        out["ratchet"].update(rt)
        if "improvements_confirming_runs" in rt:
            out["ratchet"]["confirming_runs"] = rt["improvements_confirming_runs"]
    return out


def is_bimodal(slug, meth):
    return any(b and b.lower() in slug.lower() for b in meth["bimodal"])


# ---- kernels -> families and rows ----------------------------------------------------------------------------------------
TEXT = re.compile(r"^(ch_|h_|chat_|parse_tools|render_tool|py_literal|complete_text|content_until|output_next|history_|unsupported|"
                  r"vision_|g_vision)")
# (pattern, family, rows) first match wins; the plan's mapping (Appendix A section 1): prefill GEMM and flash2 -> prefill rows,
# decode matvec and attention -> decode rows, verify-span kernels -> drafted decode rows, MoE and GDN and control -> all rows.
RULES = [
    (re.compile(r"attention_split"), "attn_decode", DECODE_ROWS + ["streams4"]),
    (re.compile(r"attention_flash2|^g_tile_|^g_block_|^g_pair_|^g_run_|attention_fallback"), "attn_prefill", PREFILL_ROWS),
    (re.compile(r"^llm_(verify|.*propose|.*warm)"), "spec", DECODE_ROWS),
    (re.compile(r"^g_gdn_"), "gdn", ROWS),
    (re.compile(r"^g_moe_"), "moe", ROWS),
    (re.compile(r"^g_gemm"), "gemm", PREFILL_ROWS),
    (re.compile(r"matvec|^g_mv"), "matvec", DECODE_ROWS + ["streams4"]),
    (re.compile(r"^g_kv_offset|^g_qk_rope"), "kv", ROWS),
]
FORMATS = re.compile(r"(q[2-8]_[01ks]|q[2-8]_k|mxfp4|bf16|f16|iq\d_\w+)")


def kernel_family_rows(name):
    """(family, rows) of a changed block symbol (a kernel, func or command program). Unknown blocks are 'program' (all rows):
    a changed command program or scheduler can move any row. Text-only blocks (chat template, tool parsing) have no perf rows."""
    if TEXT.match(name):
        return "text", []
    for pat, fam, rows in RULES:
        if pat.search(name):
            return fam, list(rows)
    return "program", list(ROWS)


def kernel_formats(name):
    m = FORMATS.findall(name)
    return sorted(set(m))


# ---- stratum signature -----------------------------------------------------------------------------------------------------
def kv_bucket(kv):
    return "<=2" if kv <= 2 else ("4" if kv <= 4 else ">=8")


def signature_from_config(cfg):
    """The model-side fields of the stratum signature (plan Appendix A section 2), read from llm_config.json."""
    heads, kv = int(cfg.get("heads") or 0), int(cfg.get("kv_heads") or 0)
    dr = (cfg.get("draft") or {}).get("kind") or "none"
    return {"arch": cfg.get("arch"), "head_dim": cfg.get("head_dim"), "G": (heads // kv) if kv else None, "kv_bucket": kv_bucket(kv) if kv else None,
            "draft": dr, "moe": "moe" if cfg.get("moe") else "dense", "vision": bool(cfg.get("vision")),
            "hidden": cfg.get("hidden"), "kv_heads": kv, "layers": cfg.get("layers"), "context": cfg.get("context_length")}


def stratum_key(sig, families, formats, gpu):
    return "|".join([f"arch={sig.get('arch')}", f"hd={sig.get('head_dim')}", f"G={sig.get('G')}", f"kv={sig.get('kv_bucket')}",
                     f"draft={sig.get('draft')}", sig.get("moe") or "dense", "fam=" + "+".join(sorted(families)) if families else "fam=-",
                     "fmt=" + "+".join(sorted(formats)) if formats else "fmt=-", f"gpu={gpu}"])


# ---- impact adapter ---------------------------------------------------------------------------------------------------------
def _stratum_to_signature(st):
    """diff_fleet.py's stratum block -> this module's signature fields (its kv_bucket reads 'kv<=2' | 'kv4' | 'kv>=8')."""
    if not st:
        return None
    kb = str(st.get("kv_bucket") or "").replace("kv", "")
    g = st.get("G")
    return {"arch": st.get("arch"), "head_dim": st.get("head_dim"), "G": int(g) if isinstance(g, (int, float)) and g == int(g) else g,
            "kv_bucket": kb or None, "draft": st.get("draft") or "none", "moe": "moe" if st.get("topology") == "moe" else st.get("moe", "dense"),
            "vision": bool(st.get("vision")), "kv_heads": st.get("kv_heads"), "hidden": st.get("hidden"), "layers": st.get("layers"),
            "context": st.get("context")}


def normalize_impact(raw):
    """Reduce an impact report to {slug: {'kernels': [{'name','targets','family','rows','formats'}], 'signature': {...}|None}}.

    Accepted: the report of fleet/diff_fleet.py ({"models": {slug: {"blocks": [{"symbol", "target": gfx11|gfx12|both, "rows"}],
    "model_json": [...], "stratum": {...}}}}) and the simpler shape {"models": {slug: {"kernels": [{"name", "target"|"targets", "rows"?}]}}}
    (replay_impact.py, tests). A kernel given as a bare string is a name with target 'both'. A report row list wins over the rules
    below; text-only blocks (chat template, tool parsing) never carry perf rows. A model.json change counts as a block on both targets
    that can move every row. Family and formats are derived from the symbol."""
    models = raw.get("models") if isinstance(raw, dict) else None
    if isinstance(models, list):
        models = {m.get("slug") or m.get("model"): m for m in models}
    out = {}
    for slug, m in (models or {}).items():
        if m.get("error"):
            continue
        ks = []
        items = list(m.get("blocks") or m.get("kernels") or m.get("changed_kernels") or [])
        if m.get("model_json"):
            items.append({"name": "model.json", "target": "both", "family": "config", "rows": list(ROWS)})
        for k in items:
            if isinstance(k, str):
                k = {"name": k}
            name = k.get("name") or k.get("symbol")
            tg = k.get("target") or k.get("targets") or "both"
            tg = tg if isinstance(tg, list) else [tg]
            tg = sorted({x for t in tg for x in (["gfx11", "gfx12"] if t not in ("gfx11", "gfx12") else [t])})
            fam, rows = kernel_family_rows(name)
            if k.get("family"):
                fam = k["family"]
            if k.get("rows") is not None and fam != "text":
                rows = list(k["rows"])
            ks.append({"name": name, "targets": tg, "family": fam, "rows": rows, "formats": kernel_formats(name) if fam in ("gemm", "matvec", "moe") else []})
        out[slug] = {"kernels": ks, "signature": _stratum_to_signature(m.get("stratum")) or m.get("signature")}
    return out
