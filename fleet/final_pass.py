#!/usr/bin/env python3
"""The final pass of a campaign that changed every package's programs without changing what its default path does (#736 logits readback):
rebuild all packages at one ref, prove each GPU's measured programs equivalent, upload the whole package, carry the card's performance rows
over (methodology rule 14) and add the functional logits gates. Resumable per step; every step takes --slugs and --work.

  final_pass.py prove    --ref REF --work W [--slugs A,B | --all] [--jobs N]
        builds the fleet at REF (build_fleet.py, programs only), downloads each staging head, finds for each GPU the staging revision whose programs
        the card's rows measured (programs_sha256 of the card's gates / re-measure block; the head, else the repo history), runs stage_equiv.py of
        that revision's program against the new build and checks that nothing else differs (storage/prepare/control only inside the logits
        blocks, model.json and llm_config.json only in logits/resolved/generator). W/proofs.json: slug -> gpu -> {equivalent, ...}; a model
        or GPU that is NOT equivalent is listed (W/not_equivalent.md): it needs a measurement, it is never carried over.
  final_pass.py package  --work W [--slugs ...]       W/pkg/SLUG = the staging head with programs, weight policies, model.json ("llm.logits": false) and
                                                      llm_config.json ("resolved" block, generator stamp = REF) of the new build
  final_pass.py card     --work W [--state S.json] [--slugs ...]
        per model: make_info.py equivalent per GPU (the measured rows stand), the logits/gufo gates of the model's regen.py logits-mode job when the
        state has one, build --from-info (every GPU's published performance kept), render_card.py, render --check; W/pkg/SLUG/info.json + README.md
  final_pass.py upload   --work W [--slugs ...] [--what package|card] [--dry-run]
        publish_package.py upload to loom-fun-staging only. package = the whole package with the CURRENT card text (the logits jobs then serve that
        revision: regen.py plan --mode logits); card = README/info/REPORT only, parent = the revision the card was built on.
Never touches loom-fun/*. GPU work only through regen.py (DevLab).
"""
import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CARD = os.path.join(ROOT, "skills", "loom-benchmark", "scripts", "card")
sys.path.insert(0, CARD)
sys.path.insert(0, HERE)
import card_lib  # noqa: E402

ORG = "loom-fun-staging"
PY = sys.executable
LOOMBIN = os.environ.get("REGEN_LOOMBIN", "/home/jfowers/lt/loom010/.venv/bin")
LOGITS_BLOCK = re.compile(r"^[ \t]*// logits:begin\n.*?^[ \t]*// logits:end\n\n?", re.M | re.S)
# the only line of the new programs outside a logits block that the capability changes: control.loom stores top_k masked with the flag bit
ALLOWED_RESIDUAL = ("view.store %value, %view[2] : i32, view<4xi32>",)
PROGRAM_FILES = ("control", "prepare", "storage", "service")


def sh(cmd, cwd=None, check=True, env=None):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=cwd, env=env)
    if check and r.returncode:
        sys.exit(f"FAILED: {' '.join(map(str, cmd))[:300]}\n{(r.stdout + r.stderr)[-1500:]}")
    return r


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(o, p):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=1)


def fleet_slugs():
    return sorted(jload(os.path.join(HERE, "manifest.json"))["models"])


def slugs_of(a):
    if a.slugs:
        return [x for x in a.slugs.split(",") if x]
    if a.cmd == "prove" and not a.all:
        sys.exit("prove: give --slugs A,B or --all")
    return fleet_slugs()


# ---------------------------------------------------------------- Hugging Face (read only here)

def hf_head(repo):
    from huggingface_hub import HfApi
    return HfApi().model_info(repo).sha


def hf_snapshot(repo, rev, out, patterns=None):
    from huggingface_hub import snapshot_download
    snapshot_download(repo, revision=rev, local_dir=out, allow_patterns=patterns)
    shutil.rmtree(os.path.join(out, ".cache"), ignore_errors=True)


def hf_commits(repo):
    from huggingface_hub import HfApi
    return [c.commit_id for c in HfApi().list_repo_commits(repo)]


# ---------------------------------------------------------------- pure helpers (tested)

def strip_logits(text):
    """a program without its `// logits:begin ... // logits:end` blocks"""
    return LOGITS_BLOCK.sub("", text)


def residual(base_text, new_text):
    """the lines that differ between BASE and NEW once NEW's logits blocks are removed (unified diff body, no context)"""
    a = base_text.splitlines()
    b = strip_logits(new_text).splitlines()
    return [x for x in difflib.unified_diff(a, b, lineterm="", n=0) if x[:1] in "+-" and not x.startswith(("+++", "---"))]


def residual_ok(lines):
    """True when every residual line is the allowed top_k store (removed from base, replaced inside a logits block)"""
    return all(x[1:].strip() in ALLOWED_RESIDUAL for x in lines)


def measured_programs(info, gpu):
    """programs_sha256 of the package whose rows the card shows on GPU: the re-measure block's (Strix: the published table re-measured on that
    package), else the gates' (every gate of a card measures one package; several = the newest date), None without performance on the GPU."""
    dev = (info.get("devices") or {}).get(gpu) or {}
    if not dev.get("performance"):
        return None
    rm = dev.get("remeasure") or {}
    if rm.get("programs_sha256"):
        return rm["programs_sha256"]
    gs = [g for g in (dev.get("gates") or {}).values() if g.get("programs_sha256")]
    if not gs:
        return None
    return max(gs, key=lambda g: g.get("date") or "")["programs_sha256"]


def logits_keys(cfg, path=()):
    """{path: value} of every "llm.logits" key of a model.json"""
    out = {}
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k == "llm.logits":
                out[path] = v
            else:
                out.update(logits_keys(v, path + (k,)))
    return out


def service_configs(cfg):
    """{(variant, service): config dict} of a model.json"""
    return {(v, s): (x or {}).get("config") or {} for v, vd in (cfg.get("variants") or {}).items() for s, x in (vd.get("services") or {}).items()}


def patch_model_json(old, new):
    """OLD (the staging package's model.json: HF origins, the device presets) with the new build's "llm.logits" declaration; (patched, problems):
    every service config of the patched model.json must equal the new build's, so nothing else of the serving settings moved."""
    out = json.loads(json.dumps(old))
    problems = []
    for path, val in logits_keys(new).items():
        d = out
        for k in path:
            d = d.setdefault(k, {}) if isinstance(d, dict) else d
        if isinstance(d, dict):
            d["llm.logits"] = val
    a, b = service_configs(out), service_configs(new)
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            diff = sorted(x for x in set(a.get(k, {})) | set(b.get(k, {})) if a.get(k, {}).get(x) != b.get(k, {}).get(x))
            problems.append(f"model.json {k[0]}/{k[1]} config differs: {diff[:6]}")
    return out, problems


def merge_llm_config(old, new, ref):
    """OLD (staging's llm_config.json) with the new build's "resolved" block and the generator stamped with REF; (merged, problems): anything else
    that differs between the two is a problem (a changed chat/serve setting is not a carried-over package)."""
    out = json.loads(json.dumps(old))
    diff = sorted(k for k in set(old) | set(new) if k not in ("generator", "resolved") and old.get(k) != new.get(k))
    if "resolved" in new:
        out["resolved"] = new["resolved"]
    if "generator" in out:
        out["generator"] = dict(out["generator"], skills_commit=ref, dirty=False)
    return out, ([f"llm_config.json differs in {diff[:8]}"] if diff else [])


# ---------------------------------------------------------------- prove

def build_new(a, W, slugs):
    out = os.path.join(W, "new")
    done = set()
    s = os.path.join(out, "summary.json")
    ref_sha = sh(["git", "rev-parse", a.ref], cwd=ROOT).stdout.strip()
    if os.path.exists(s) and jload(s).get("ref") == ref_sha and not jload(s).get("failures"):
        done = {x for x in slugs if os.path.exists(os.path.join(out, x, "program", "llm.loom"))}
    todo = [x for x in slugs if x not in done]
    if todo:   # one build at a time: build_fleet's own jobs are the parallelism (default: a few, memory is the limit)
        r = sh([PY, os.path.join(HERE, "build_fleet.py"), "--ref", a.ref, "--out", out, "--slugs", ",".join(todo), "--jobs", str(a.jobs)], check=False)
        print(r.stdout[-600:])
        if r.returncode:
            print("build failures:", jload(s).get("failures"), file=sys.stderr)
    return ref_sha


def find_base(slug, info, gpu, want, W, head_rev, head_prog):
    """directory with program/ of the staging revision whose programs_sha256 == want, else None"""
    snap = os.path.join(W, "snap", slug)
    if head_prog == want:
        return snap, head_rev
    for rev in hf_commits(f"{ORG}/{slug}")[:25]:
        d = os.path.join(W, "hist", slug, rev[:12])
        if not os.path.isdir(os.path.join(d, "program")):
            hf_snapshot(f"{ORG}/{slug}", rev, d, ["program/*.loom"])
        if card_lib.programs_sha256(d) == want:
            return d, rev
    return None, None


def cmd_prove(a):
    W = os.path.abspath(a.work)
    slugs = slugs_of(a)
    ref = build_new(a, W, slugs)
    proofs = jload(os.path.join(W, "proofs.json")) if os.path.exists(os.path.join(W, "proofs.json")) else {}
    for slug in slugs:
        new = os.path.join(W, "new", slug)
        if not os.path.exists(os.path.join(new, "model.json")):
            proofs[slug] = {"error": "the build at the ref failed: " + (open(os.path.join(new, "build.log")).read()[-300:] if os.path.exists(os.path.join(new, "build.log")) else "no log")}
            continue
        snap = os.path.join(W, "snap", slug)
        head = hf_head(f"{ORG}/{slug}")
        hp = os.path.join(W, "heads.json")
        heads = jload(hp) if os.path.exists(hp) else {}
        if not os.path.exists(os.path.join(snap, "info.json")) or heads.get(slug) != head:
            shutil.rmtree(snap, ignore_errors=True)
            hf_snapshot(f"{ORG}/{slug}", head, snap)
            heads[slug] = head
            jdump(heads, hp)
        info = jload(os.path.join(snap, "info.json"))
        head_prog, new_prog = card_lib.programs_sha256(snap), card_lib.programs_sha256(new)
        res = {"ref": ref, "head": head, "head_programs": head_prog, "new_programs": new_prog, "gpus": {}, "problems": []}
        # whole-package checks that do not depend on the GPU
        old_mj, new_mj = jload(os.path.join(snap, "model.json")), jload(os.path.join(new, "model.json"))
        _, p1 = patch_model_json(old_mj, new_mj)
        _, p2 = merge_llm_config(jload(os.path.join(snap, "llm_config.json")), jload(os.path.join(new, "llm_config.json")), ref)
        res["problems"] += p1 + p2
        for f in sorted(x for x in os.listdir(new) if x.endswith(".loom")):   # weight policies
            if not os.path.exists(os.path.join(snap, f)) or open(os.path.join(snap, f), "rb").read() != open(os.path.join(new, f), "rb").read():
                res["problems"].append(f"{f} differs")
        for gpu in sorted(info.get("devices") or {}):
            want = measured_programs(info, gpu)
            if not want:
                res["gpus"][gpu] = {"applicable": False}
                continue
            g = res["gpus"][gpu] = {"applicable": True, "measured_programs": want, "problems": list(res["problems"])}
            base, rev = find_base(slug, info, gpu, want, W, head, head_prog)
            if not base:
                g["problems"].append("the revision the card's rows measured is not in the staging history")
                g["equivalent"] = False
                continue
            g["base_revision"] = rev
            if want == new_prog:
                g.update(equivalent=True, identical=True)
                continue
            for pf in PROGRAM_FILES:   # the other programs: differences outside the logits blocks
                bp, np_ = os.path.join(base, "program", pf + ".loom"), os.path.join(new, "program", pf + ".loom")
                if os.path.exists(bp) and os.path.exists(np_):
                    r = residual(open(bp).read(), open(np_).read())
                    if not residual_ok(r):
                        g["problems"].append(f"program/{pf}.loom differs outside the logits blocks ({len(r)} lines, e.g. {r[0][:80]})")
                elif os.path.exists(bp) != os.path.exists(np_):
                    g["problems"].append(f"program/{pf}.loom only on one side")
            bdir = os.path.join(W, "base", f"{gpu}", slug)
            shutil.rmtree(bdir, ignore_errors=True)
            os.makedirs(bdir)
            os.symlink(os.path.join(base, "program"), os.path.join(bdir, "program"))
            js = os.path.join(W, "equiv", f"{slug}.{gpu}.json")
            os.makedirs(os.path.dirname(js), exist_ok=True)
            r = sh([PY, os.path.join(HERE, "stage_equiv.py"), os.path.join(W, "base", gpu), os.path.join(W, "new"), "--slugs", slug, "--json", js], check=False)
            if not os.path.exists(js):
                g["problems"].append("stage_equiv failed: " + (r.stdout + r.stderr)[-200:])
                g["equivalent"] = False
                continue
            e = jload(js)["packages"][0]
            g["stage_status"], g["stage_problems"] = e["status"], e["problems"]
            if e["status"] != "PASS":
                g["problems"].append(f"stage_equiv {e['status']}: {'; '.join(e['problems'])[:200]}")
            g["equivalent"] = e["status"] == "PASS" and not g["problems"]
        proofs[slug] = res
        jdump(proofs, os.path.join(W, "proofs.json"))
    bad = []
    for s, r in sorted(proofs.items()):
        if r.get("error"):
            bad.append(f"- {s}: {r['error']}")
        for gpu, g in (r.get("gpus") or {}).items():
            if g.get("applicable") and not g.get("equivalent"):
                bad.append(f"- {s} {gpu}: {'; '.join(g['problems'])[:300]}")
    jdump(proofs, os.path.join(W, "proofs.json"))
    open(os.path.join(W, "not_equivalent.md"), "w").write("\n".join(bad) + ("\n" if bad else ""))
    n_eq = sum(1 for r in proofs.values() for g in (r.get("gpus") or {}).values() if g.get("equivalent"))
    print(f"proofs: {len(proofs)} models, {n_eq} (model, GPU) equivalent, {len(bad)} NOT equivalent / failed")
    print("\n".join(bad))
    jdump({"ref": ref}, os.path.join(W, "ref.json"))


# ---------------------------------------------------------------- package

def cmd_package(a):
    W = os.path.abspath(a.work)
    proofs = jload(os.path.join(W, "proofs.json"))
    ref = jload(os.path.join(W, "ref.json"))["ref"]
    for slug in slugs_of(a):
        r = proofs.get(slug) or {}
        if r.get("error") or not r.get("gpus"):
            print(f"{slug}: no proof, skipped")
            continue
        if not any(g.get("equivalent") for g in r["gpus"].values() if g.get("applicable")) and any(g.get("applicable") for g in r["gpus"].values()):
            print(f"{slug}: not equivalent on any GPU, skipped (see not_equivalent.md)")
            continue
        snap, new, out = (os.path.join(W, d, slug) for d in ("snap", "new", "pkg"))
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(snap, out)
        for f in os.listdir(os.path.join(new, "program")):
            shutil.copy(os.path.join(new, "program", f), os.path.join(out, "program", f))
        for f in (x for x in os.listdir(new) if x.endswith(".loom")):
            shutil.copy(os.path.join(new, f), os.path.join(out, f))
        mj, _ = patch_model_json(jload(os.path.join(snap, "model.json")), jload(os.path.join(new, "model.json")))
        jdump(mj, os.path.join(out, "model.json"))
        cfg, _ = merge_llm_config(jload(os.path.join(snap, "llm_config.json")), jload(os.path.join(new, "llm_config.json")), ref)
        jdump(cfg, os.path.join(out, "llm_config.json"))
        print(f"{slug}: package assembled ({card_lib.programs_sha256(out)[:12]})")


# ---------------------------------------------------------------- card

def tool(*args, cwd=None):
    r = sh([PY, *args], cwd=cwd, check=False)
    if r.returncode:
        raise RuntimeError(f"{' '.join(map(str, args))[:200]}\n{(r.stdout + r.stderr)[-800:]}")
    return r.stdout


def job_of(state, slug, gpu):
    for n, e in (state or {}).get("jobs", {}).items():
        if e["slug"] == slug and e["device"] == gpu and e["stage"] in ("collected", "ingested", "published") and e.get("decision") == "promote":
            return n, e
    return None, None


def changed_paths(a, b, prefix=""):
    """dotted paths where two JSON values differ (cap 40): what a rebuilt card changed"""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += changed_paths(a.get(k), b.get(k), f"{prefix}.{k}" if prefix else k)
        return out[:40]
    return [] if a == b else [prefix]


def build_card(W, slug, ref, state, proof_url=None):
    pkg, snap = os.path.join(W, "pkg", slug), os.path.join(W, "snap", slug)
    cd = os.path.join(W, "cards", slug)
    shutil.rmtree(cd, ignore_errors=True)
    os.makedirs(cd)
    info = jload(os.path.join(snap, "info.json"))
    proofs = jload(os.path.join(W, "proofs.json"))[slug]
    ev, notes = [], []
    gpus = sorted(info["devices"], key=lambda g: list(card_lib.registry.devices()).index(g) if g in card_lib.registry.devices() else 99)
    for gpu in gpus:
        g = proofs["gpus"].get(gpu) or {}
        if not g.get("applicable"):
            continue
        if g.get("equivalent") and not g.get("identical"):
            pj = os.path.join(cd, f"proof_{gpu}.json")
            jdump({"equivalent": True, "tool": "fleet/stage_equiv.py", "compared_programs": g["measured_programs"],
                   "summary": (f"default-path stages identical after canonicalization (stage_equiv.py at {ref[:10]}, measured revision "
                               f"{g['base_revision'][:10]}); storage/prepare/control differ only inside the logits blocks"), "url": proof_url}, pj)
            tool(f"{CARD}/make_info.py", "equivalent", "--gpu", gpu, "--proof", pj, "--package", pkg, "--source-kind", "local", "--device", "none",
                 "--hrx-loom", "none", "-o", os.path.join(cd, f"ev_eq_{gpu}.json"))
            ev.append(os.path.join(cd, f"ev_eq_{gpu}.json"))
        name, e = job_of(state, slug, gpu)
        if e:
            d = e["dir"]
            tool(f"{CARD}/pull_devlab.py", "--identity", d, "--gpu", gpu, "--device-out", os.path.join(cd, f"dev_{gpu}.json"), "--hrx-out", os.path.join(cd, f"hrx_{gpu}.json"))
            files = [f for f in (f"{d}/logits_staging.json", f"{d}/conformance_staging.json", f"{d}/ktest_staging.log") if os.path.exists(f)]
            tool(f"{CARD}/make_info.py", "gates", "--gpu", gpu, "--time", (e.get("completed") or "")[:19], *files, "--package", pkg, "--source-kind", "devlab",
                 "--run", e["run"], "--url", e["job_url"], "--device", os.path.join(cd, f"dev_{gpu}.json"), "--hrx-loom", os.path.join(cd, f"hrx_{gpu}.json"),
                 "-o", os.path.join(cd, f"ev_gates_{gpu}.json"))
            ev.append(os.path.join(cd, f"ev_gates_{gpu}.json"))
        else:
            notes.append(f"{gpu}: no logits job")
    cmd = [f"{CARD}/make_info.py", "build", pkg, "--org", ORG, "--gpus", ",".join(gpus), "--fit", "auto", "--memory-bytes", str(info["card"]["memory_bytes"]),
           "--from-info", os.path.join(snap, "info.json"), "--generator-commit", ref]
    for gpu in gpus:
        if (info["devices"][gpu] or {}).get("performance"):
            cmd += ["--perf-from", f"{gpu}={os.path.join(snap, 'info.json')}"]
    tool(*cmd, *(["--evidence", *ev] if ev else []), "-o", os.path.join(cd, "info_new.json"))
    shutil.copy(os.path.join(cd, "info_new.json"), os.path.join(pkg, "info.json"))
    tool(f"{CARD}/render_card.py", pkg)
    tool(f"{CARD}/render_card.py", "--check", pkg, os.path.join(pkg, "README.md"))
    changed = changed_paths(info, jload(os.path.join(pkg, "info.json")))
    return notes, changed


def cmd_card(a):
    W = os.path.abspath(a.work)
    ref = jload(os.path.join(W, "ref.json"))["ref"]
    state = jload(a.state) if a.state else None
    for slug in slugs_of(a):
        if not os.path.isdir(os.path.join(W, "pkg", slug)):
            print(f"{slug}: no package, skipped")
            continue
        try:
            notes, changed = build_card(W, slug, ref, state, a.proof_url)
        except (RuntimeError, SystemExit) as ex:
            print(f"{slug}: CARD FAILED {str(ex)[-400:]}")
            continue
        print(f"{slug}: card built; {'; '.join(notes) or 'logits gates in'}; info.json changed: {', '.join(changed[:12])}{' ...' if len(changed) > 12 else ''}")


# ---------------------------------------------------------------- upload

def cmd_upload(a):
    W = os.path.abspath(a.work)
    heads = jload(os.path.join(W, "heads.json"))
    env = dict(os.environ, PATH=LOOMBIN + os.pathsep + os.environ["PATH"])
    ref = jload(os.path.join(W, "ref.json"))["ref"]
    for slug in slugs_of(a):
        pkg = os.path.join(W, "pkg", slug)
        if not os.path.isdir(pkg):
            print(f"{slug}: no package, skipped")
            continue
        parent = hf_head(f"{ORG}/{slug}") if a.what == "card" else heads[slug]
        if a.what == "package" and hf_head(f"{ORG}/{slug}") != heads[slug]:
            print(f"{slug}: the staging repo moved since the snapshot ({heads[slug][:8]}): run prove again, skipped")
            continue
        msg = (f"Final pass: package rebuilt at {ref[:10]} (logits readback capability, llm_config resolved block)" if a.what == "package"
               else f"Final pass: card with the carried-over measurements (stage equivalence) and the logits gates, rendered at {ref[:10]}")
        cmd = [PY, f"{CARD}/publish_package.py", "upload", pkg, "--repo", slug, "--org", ORG, "--parent-commit", parent, "--message", msg]
        if a.what == "card":
            cmd.append("--card-only")
        if a.dry_run:
            cmd.append("--dry-run")
        r = sh(cmd, check=False, env=env)
        tail = " | ".join(x.strip() for x in (r.stdout + r.stderr).strip().splitlines()[-4:])
        print(f"{slug}: {'dry run ' if a.dry_run else ''}{a.what} upload {'OK' if not r.returncode else 'FAILED'}: {tail[:300]}")
        if not r.returncode and not a.dry_run and a.what == "package":
            heads[slug] = hf_head(f"{ORG}/{slug}")
            jdump(heads, os.path.join(W, "heads.json"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("prove", cmd_prove), ("package", cmd_package), ("card", cmd_card), ("upload", cmd_upload)):
        p = sub.add_parser(name)
        p.add_argument("--work", required=True)
        p.add_argument("--slugs")
        p.set_defaults(fn=fn)
        if name == "prove":
            p.add_argument("--ref", required=True)
            p.add_argument("--all", action="store_true")
            p.add_argument("--jobs", type=int, default=3)
        if name == "card":
            p.add_argument("--state", help="the regen.py logits-mode state with the collected jobs")
            p.add_argument("--proof-url")
        if name == "upload":
            p.add_argument("--what", choices=("package", "card"), required=True)
            p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
