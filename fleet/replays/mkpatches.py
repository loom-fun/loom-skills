#!/usr/bin/env python3
"""Regenerates the patches under fleet/replays/<name>/ (base.patch, A.patch, B.patch) from the edit functions below.

usage: mkpatches.py [NAME ...] [--work DIR]

Each patch is `git diff` of a scratch worktree at origin/refactor (+ the patches the replay's base chain applies first).
The edits turn a shipped feature OFF (base: the pre-PR state; the kit code stays, only its default and the registry legacy
value change) or ON again the two ways the replays compare: A = a changed default (kit default + registry legacy value),
B = an opt-in scope rule (scope.json) at the measured scope.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
STARTER = "skills/loom-port-llm/starter"
KB = "skills/loom-kb"


class T:
    def __init__(self, root):
        self.root = root

    def sub(self, rel, old, new, count=1):
        p = os.path.join(self.root, rel)
        s = open(p, encoding="utf-8").read()
        assert s.count(old) == count, f"{rel}: {old!r} x{s.count(old)} (want {count})"
        open(p, "w", encoding="utf-8").write(s.replace(old, new))

    def scope(self, node, rules, evidence_free=False):
        """Append rules [{when, set, evidence}] to the choices of KB node `node` (a scope.json)."""
        rel = f"{KB}/{node}/scope.json"
        p = os.path.join(self.root, rel)
        j = json.load(open(p, encoding="utf-8"))
        j["choices"] = j.get("choices", []) + rules
        open(p, "w", encoding="utf-8").write(json.dumps(j, indent=1) + "\n")

    def clear_scope(self, node):
        rel = f"{KB}/{node}/scope.json"
        p = os.path.join(self.root, rel)
        j = json.load(open(p, encoding="utf-8"))
        j["choices"] = []
        open(p, "w", encoding="utf-8").write(json.dumps(j, indent=1) + "\n")

    def legacy(self, key, old, new):
        self.sub(f"{KB}/choices.py", f'CHOICE("{key}", "bool", {old},', f'CHOICE("{key}", "bool", {new},')

    def default(self, rel, env, old, new):
        self.sub(f"{STARTER}/{rel}", f'"{env}", "{old}"', f'"{env}", "{new}"')


def rule(when, key_vals, evidence):
    return {"when": when, "set": key_vals, "evidence": evidence}


# ---- kv-once family (#357, #451, D7) --------------------------------------------------------------------------------------------
def kv_base(t):
    t.default("attn_rules.py", "ATTN_GQA", "1", "0")
    t.default("attn_rules.py", "ATTN_GQA_DRAFT", "1", "0")
    t.default("attn_rules.py", "ATTN_GQA_D256", "1", "0")
    t.legacy("decode.attn.kv_once", "True", "False")
    t.legacy("decode.attn.kv_once.drafted", "True", "False")
    t.legacy("decode.attn.kv_once.d256", "True", "False")
    t.clear_scope("arch/llm/moe/qwen35moe")   # the D7 rule is a later change


def r357_A(t):
    t.default("attn_rules.py", "ATTN_GQA", "0", "1")
    t.legacy("decode.attn.kv_once", "False", "True")


def r357_B(t):
    ev = "PR #357: Qwen3-4B decode_32k 24.5 -> 28.3 tok/s (DevLab same runner), decode / 2k / 4 streams within noise"
    for node in ("arch/llm/dense/qwen3", "arch/llm/dense/qwen3vl"):
        t.scope(node, [rule({"not_facets": ["drafted"], "head_dim": 128}, {"decode.attn.kv_once": True}, ev)])


def r451_A(t):
    t.default("attn_rules.py", "ATTN_GQA_DRAFT", "0", "1")
    t.default("attn_rules.py", "ATTN_GQA_D256", "0", "1")
    t.legacy("decode.attn.kv_once.drafted", "False", "True")
    t.legacy("decode.attn.kv_once.d256", "False", "True")


def r451_B(t):
    ev = "PR #451 run 43311788249: same-runner decode_32k vs main"
    t.scope("arch/llm/dense/qwen2vl", [rule({"facets": ["drafted"]}, {"decode.attn.kv_once.drafted": True}, ev + ": Qwen2.5-VL-7B +19 %")])
    t.scope("arch/llm/dense/llama", [rule({"facets": ["drafted"]}, {"decode.attn.kv_once.drafted": True}, ev + ": Llama-3.1-8B +20 %")])
    t.scope("arch/llm/moe/flint", [rule({"facets": ["drafted"]}, {"decode.attn.kv_once.drafted": True}, ev + ": Flint +5 %")])
    t.scope("arch/llm/dense/qwen3", [rule({"facets": ["drafted"]}, {"decode.attn.kv_once.drafted": True}, ev + ": Qwen3-4B DFlash +6 %")])
    t.scope("arch/llm/hybrid/qwen35", [rule({"facets": ["drafted"]}, {"decode.attn.kv_once.drafted": True, "decode.attn.kv_once.d256": True},
                                            ev + ": Qwen3.5-4B +9 %, Qwen3.5-9B +4 % (head_dim 256)")])


def d7_A(t):
    t.default("attn_rules.py", "ATTN_GQA_D256", "1", "0")
    t.legacy("decode.attn.kv_once.d256", "True", "False")


def d7_B(t):
    t.scope("arch/llm/moe/qwen35moe", [rule({"facets": ["drafted"]}, {"decode.attn.kv_once.d256": False},
        "Drafted qwen35moe (head_dim 256, 2 KV heads): the K/V-once decode body (#451) cost decode_32k 0.74-0.78x on Strix (runs 43603564334, "
        "43608671297, 43611547482). Rule off: Strix decode_32k vs loom-fun 1.00/1.04/0.98/1.04, R9700 vs staging main 1.22-1.41 (run 43631671557, PR #715).")])


# ---- RDNA4 twins (#573 GEMM twins, #574 flash twins): the default of LOOM_RDNA4 at every site that reads it -----------------
TWIN_SITES = [("dual_target.py", 1), ("gen_gemm_rdna4.py", 1), ("gen_flash2.py", 1), ("gen_flash2s.py", 1), ("gen_llm.py", 1), ("kb_choices.py", 1)]


def twins_base(t):
    for f, n in TWIN_SITES:
        t.sub(f"{STARTER}/{f}", '"LOOM_RDNA4", "1"', '"LOOM_RDNA4", "0"', n)
    t.legacy("target.rdna4_twins", "True", "False")


def twins_A(t):
    for f, n in TWIN_SITES:
        t.sub(f"{STARTER}/{f}", '"LOOM_RDNA4", "0"', '"LOOM_RDNA4", "1"', n)
    t.legacy("target.rdna4_twins", "False", "True")


def twins_B(t):
    p = os.path.join(t.root, KB, "device/rdna4/scope.json")
    j = {"scope": "device/rdna4", "match": {}, "owner": "kb",
         "choices": [rule({}, {"target.rdna4_twins": True}, "PR #573 / #574: gfx12 twins K_rdna4 of every prefill GEMM and flash kernel; "
                          "gfx11 text unchanged (LOOM_RDNA4=0 byte-identical)")]}
    open(p, "w", encoding="utf-8").write(json.dumps(j, indent=1) + "\n")


# ---- E2B R9700 prefill stack (#656, #665): llm_config rdna4_f16_ffn vs a scope rule ---------------------------------------
E2B = "fleet/unsloth-gemma-4-E2B-it-loom/llm_config.json"


def e2b_base(t):
    t.sub(E2B, ' "model_id": "google/gemma-4-E2B-it",\n "rdna4_f16_ffn": true\n}', ' "model_id": "google/gemma-4-E2B-it"\n}')


def e2b_A(t):
    t.sub(E2B, ' "model_id": "google/gemma-4-E2B-it"\n}', ' "model_id": "google/gemma-4-E2B-it",\n "rdna4_f16_ffn": true\n}')


def e2b_B(t):
    t.scope("arch/llm/dense/gemma4", [rule({"heads": 8, "kv_heads": 1, "devices": ["rdna4"]}, {"prefill.gemm.f16o_swiglu": True},
        "PR #656 / #665 run 43576014176 / 43596459543: E2B (head_dim 512, 8 heads on 1 KV head) prefill +3..+19 % on the R9700 (f16-output swiglu "
        "group, flash twin trims, two-heads-per-wave d512 flash)")])


# ---- flash2 masked path only in the last key tile (#480) ----------------------------------------------------------------------
def peel_base(t):
    t.sub(f"{STARTER}/gen_flash2.py", '"FLASH2_PEEL", "1"', '"FLASH2_PEEL", "0"')
    t.sub(f"{STARTER}/gen_flash2s.py", '"FLASH2_PEEL", "1"', '"FLASH2_PEEL", "0"')
    t.legacy("prefill.attn.flash2_peel", "True", "False")


def peel_A(t):
    t.sub(f"{STARTER}/gen_flash2.py", '"FLASH2_PEEL", "0"', '"FLASH2_PEEL", "1"')
    t.sub(f"{STARTER}/gen_flash2s.py", '"FLASH2_PEEL", "0"', '"FLASH2_PEEL", "1"')
    t.legacy("prefill.attn.flash2_peel", "False", "True")


def peel_B(t):
    ev = "PR #480: Spark-X2.6 prefill +5..14 % vs main (devlab-perf same runner); recovers the #193/#213 loss of the six blocked cards"
    t.scope("arch/llm/dense/gemma4", [rule({"sliding_window": [None, 512]}, {"prefill.attn.flash2_peel": True}, ev + " (E2B, E4B, Bolt-E4B)")])
    t.scope("arch/llm/dense/spark2_5", [rule({}, {"prefill.attn.flash2_peel": True}, ev + " (Spark-X2.5, X2.6)")])
    t.scope("arch/llm/dense/lfm3", [rule({}, {"prefill.attn.flash2_peel": True}, ev + " (LFM3-1.7B)")])


REPLAYS_MORE = {
    "r573": {"base": twins_base, "A": twins_A, "B": twins_B},
    "r656": {"base": e2b_base, "A": e2b_A, "B": e2b_B},
    "r480": {"base": peel_base, "A": peel_A, "B": peel_B},
}


REPLAYS = {
    "r357": {"base": kv_base, "A": r357_A, "B": r357_B},
    "r451": {"base_from": "r357:A", "A": r451_A, "B": r451_B},
    "d7": {"base_from": "r451:A", "A": d7_A, "B": d7_B},
}
REPLAYS.update(REPLAYS_MORE)


def sh(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"{cmd}\n{p.stdout}{p.stderr}")
    return p.stdout


def chain(name, v):
    spec = REPLAYS[name]
    out = chain(*spec["base_from"].split(":")) if "base_from" in spec else []
    if "base_from" not in spec and "base" in spec:
        out = [os.path.join(HERE, name, "base.patch")]
    return out + [os.path.join(HERE, name, v[0] + ".patch")]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    work = "/home/jfowers/lt/refactor/eng/replay/work/mk"
    sh(["git", "fetch", "-q", "origin"], REPO)
    for name in (args or list(REPLAYS)):
        spec = REPLAYS[name]
        os.makedirs(os.path.join(HERE, name), exist_ok=True)
        pre = (chain(*spec["base_from"].split(":")) if "base_from" in spec else [])
        for v in [k for k in ("base", "A", "B") if k in spec]:
            if os.path.isdir(work):
                subprocess.run(["git", "worktree", "remove", "--force", work], cwd=REPO, capture_output=True)
            sh(["git", "worktree", "add", "--detach", work, "origin/refactor"], REPO)
            files = list(pre)
            if v != "base" and "base" in spec:
                files.append(os.path.join(HERE, name, "base.patch"))
            for p in files:
                sh(["git", "apply", "--whitespace=nowarn", p], work)
            if files:
                sh(["git", "add", "-A"], work)
                sh(["git", "-c", "user.name=replay", "-c", "user.email=replay@example.invalid", "commit", "-q", "-m", "chain"], work)
            spec[v](T(work))
            sh(["git", "add", "-N", "."], work)
            diff = sh(["git", "diff"], work)
            open(os.path.join(HERE, name, v + ".patch"), "w").write(diff)
            print(name, v, len(diff.splitlines()), "diff lines")
        subprocess.run(["git", "worktree", "remove", "--force", work], cwd=REPO, capture_output=True)


if __name__ == "__main__":
    main()
