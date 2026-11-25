#!/usr/bin/env python3
"""Card lint: a loom-fun card says exactly what was measured, on the SDK its Quick start installs.

usage: lint_card.py DIR [DIR ...] [--form production|staging] [--offline] [--generator SCRIPTS_DIR] [--json OUT]

DIR is a card repository's files (README.md, model.json, info.json, llm_config.json, program/ ...). Exit 1 when any card
has a problem; each problem is printed as "DIR: [check] text". Run it before publishing a card: publish_package.py runs only the
files check itself (card_files.check_package, on an upload to the production org; --card-only: the card files' problems) and its
own card-text checks, never this lint (regen.py verify runs it with --form staging on each published staging card).

Checks (each problem names its check):
  files      card_files.check_package: only card-format files; no staging, rehearsal or qualification-record text
             (--form staging: the README fetch lines and info.json repo may name the staging copy of this repo, which
             promotion rewrites; every other staging reference is a problem)
  runs       every cited GitHub Actions run and job (info.json sources, README/REPORT links) concluded success; a source
             without a run, or a run gh cannot resolve, is unknown (--offline: only conclusions info.json records)
  versions   the Quick start installs one hrx-loom version (the pin); the Versions line names it and no "unknown";
             at least one GPU is measured on it
  render     README.md (and a generated REPORT.md) is exactly what render_card.py renders from the card's files: no
             hand-written text (--generator: the card generator scripts to render with, default this checkout's)
  delta      a card measured on a new SDK carries the delta line (info.json sdk_delta, comparison "card") for every
             GPU it re-measured, and every regression beyond the band (a drafted-token acceptance drop too) has an explanation
             (else "unexplained")
  acceptance a drafting card's re-measured GPUs record the drafted-token acceptance, or why it is not reported
  admission  a chat card's re-measured GPUs record the native-context admission check
  carried    (informational) GPUs and values carried from another SDK, shown marked
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from card_files import check_package  # noqa: E402
from card_lib import (DELTA_BAND, METRIC_KEYS, card_gpus, card_metrics, carried_sdk, delta_rows, free_text_errors, gpu_view,  # noqa: E402
                      has_results, load_info, load_package, metric_applies, package_facts, requalified, sdk_version, validate)

def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


RUN = re.compile(r"https://github\.com/([\w.-]+/[\w.-]+)/actions/runs/(\d+)(?:/job/(\d+))?")
STAGING_ORG, PROD_ORG = "loom-fun-staging", "loom-fun"


class Runs:
    """GitHub Actions conclusions, cached: run and job (None when gh cannot tell)."""

    def __init__(self, offline=False):
        self.offline, self.cache = offline, {}

    def _api(self, path):
        if self.offline:
            return None
        if path not in self.cache:
            try:
                out = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=60)
                self.cache[path] = json.loads(out.stdout) if out.returncode == 0 else None
            except (OSError, ValueError, subprocess.TimeoutExpired):
                self.cache[path] = None
        return self.cache[path]

    def run(self, repo, rid):
        x = self._api(f"repos/{repo}/actions/runs/{rid}")
        return (x or {}).get("conclusion") if x else None

    def job(self, repo, jid):
        x = self._api(f"repos/{repo}/actions/jobs/{jid}")
        return (x or {}).get("conclusion") if x else None


def compose(card, form):
    """A production-form copy of a staging-form card (the staging repo named only where promotion rewrites it), or the card."""
    if form == "production":
        return card, None
    tmp = tempfile.mkdtemp(prefix="lint-card-")
    for name in os.listdir(card):
        src = os.path.join(card, name)
        if name in ("README.md", "info.json") and os.path.isfile(src):
            text = read(src)
            if name == "README.md":   # the fetch lines of this repo's staging copy
                text = re.sub(rf"(loom-fetch \S*?){STAGING_ORG}/", rf"\1{PROD_ORG}/", text)
            else:
                info = json.loads(text)
                if str(info.get("repo", "")).startswith(STAGING_ORG + "/"):
                    info["repo"] = PROD_ORG + info["repo"][len(STAGING_ORG):]
                text = json.dumps(info, indent=1)
            with open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
                f.write(text)
        else:
            os.symlink(os.path.abspath(src), os.path.join(tmp, name))
    return tmp, tmp


def cited_runs(card, info):
    """[(where, repo, run, job, recorded conclusion)] of every Actions run the card cites."""
    out = []
    for sid, src in ((info or {}).get("sources") or {}).items():
        if src.get("kind") not in ("devlab", "factory"):
            continue
        m = RUN.search(src.get("url") or "")
        if m:
            out.append((f"info.json source {sid}", m.group(1), m.group(2), m.group(3), src.get("conclusion")))
        else:
            out.append((f"info.json source {sid}", None, src.get("run"), None, src.get("conclusion")))
    for name in ("README.md", "REPORT.md"):
        p = os.path.join(card, name)
        if os.path.isfile(p):
            for m in RUN.finditer(read(p)):
                out.append((name, m.group(1), m.group(2), m.group(3), None))
    seen, uniq = set(), []
    for x in out:
        if (x[1], x[2], x[3]) not in seen or x[0].startswith("info.json"):
            seen.add((x[1], x[2], x[3]))
            uniq.append(x)
    return uniq


def run_problems(card, info, runs):
    out, done = [], set()
    for where, repo, rid, jid, recorded in cited_runs(card, info):
        if recorded and recorded != "success":
            out.append(f"{where}: cites a job that concluded {recorded}")
            continue
        if not rid or not repo:
            out.append(f"{where}: cites no GitHub Actions run (unknown evidence)")
            continue
        key = (repo, rid, jid)
        if key in done:
            continue
        done.add(key)
        if runs.offline:
            continue
        rc = runs.run(repo, rid)
        jc = runs.job(repo, jid) if jid else None
        if rc is None and jc is None:
            out.append(f"{where}: run {rid}" + (f" job {jid}" if jid else "") + " cannot be resolved (unknown evidence)")
        elif jid and jc != "success":
            out.append(f"{where}: cites job {jid} of run {rid}, which concluded {jc}")
        elif not jid and rc != "success":   # a cited successful job stands on its own (its run may hold other models' jobs: D6 a)
            out.append(f"{where}: cites run {rid}, which concluded {rc}")
    return out


PIN = re.compile(r"hrx-loom\[hf\](?:==|\s*@[^'\"]*hrx_loom-)([0-9][\w.]*?)(?:[\"'\s]|-py3)")


def version_problems(card, info, pin):
    out = []
    readme = read(os.path.join(card, "README.md"))
    qs = readme.split("## Performance")[0]
    pins = sorted(set(PIN.findall(qs)))
    if len(pins) > 1:
        out.append(f"README.md: the Quick start installs several hrx-loom versions ({', '.join(pins)})")
    if not pins and "pipx install" in qs:
        out.append("README.md: the Quick start installs hrx-loom without a version")
    vline = next((ln for ln in readme.splitlines() if ln.startswith("Versions:")), "")
    if info is not None:
        if not vline:
            out.append("README.md: no Versions line")
        if "unknown" in vline:
            out.append(f"README.md: the Versions line says unknown ({vline.strip()})")
        pv = sdk_version(pin)
        m = re.search(r"hrx-loom `([^`]+)`", vline)
        if pv and m and m.group(1) != pv:
            out.append(f"README.md: the Versions line names hrx-loom {m.group(1)}, the Quick start installs {pv}")
        if pins and pv and pins != [pv]:
            out.append(f"README.md: the Quick start installs hrx-loom {', '.join(pins)}, the package pins {pv}")
        if pv and not any(requalified(info, g, pin) for g in card_gpus(info)):
            out.append(f"info.json: no GPU has values or gates measured on hrx-loom {pv} (the pin)")
    return out


def render_problems(card, generator):
    """README.md (and REPORT.md when generated) as render_card.py renders the card's files."""
    cp = subprocess.run([sys.executable, "-I", os.path.join(generator, "render_card.py"), "--check", card,
                         os.path.join(card, "README.md")], capture_output=True, text=True, timeout=300,
                        env=dict(os.environ, LOOM_CARD_UPSTREAM="none"))
    if cp.returncode == 0:
        return []
    diff = [ln[5:] for ln in cp.stdout.splitlines() if ln.startswith("DIFF ")]
    if diff:
        return [f"{os.path.basename(p)}: not what the card generator renders from the card's files (hand-written or stale text)" for p in diff]
    return [f"render_card.py refuses the card: {(cp.stderr.strip().splitlines() or ['?'])[-1]}"]


def info_problems(card, info, facts, feats, pin):
    """delta, acceptance, admission and carried findings of a text card."""
    out = {"delta": [], "acceptance": [], "admission": [], "carried": []}
    gpus = [g for g in card_gpus(info) if has_results(info["devices"][g])]
    re_measured = [g for g in gpus if requalified(info, g, pin)]
    comps = (info.get("sdk_delta") or {}).get("comparisons") or []
    card_comp = next((c for c in comps if c["label"] == "card"), None)
    if re_measured and not card_comp:
        out["delta"].append("no delta line against the previous card (info.json sdk_delta has no card comparison)")
    for c in comps:
        rows = delta_rows(info, pin, c, (facts.get("spec") or (None, None))[1])   # the draft depth, as render_card.py compares acceptance
        for g in re_measured:
            if g in c["devices"] and g not in rows:
                out["delta"].append(f"{c['label']}: {g} has no metric re-measured on the pin to compare")
        for g, rs in rows.items():
            for r in rs:
                if r["regression"] and not r["explanation"]:
                    out["delta"].append(f"{c['label']}: {g} {r['metric']} {r['relative'] * 100:+.1f} % unexplained (beyond -{DELTA_BAND * 100:g} %)")
    for g in gpus:
        view = gpu_view(info, g)
        sessions = (view.get("performance") or {}).get("sessions", [])
        shown = card_metrics(view)
        old = sorted({carried_sdk(info, sessions[i]["source"], pin) for k, (m, i) in shown.items()
                      if m.get("loom") is not None and carried_sdk(info, sessions[i]["source"], pin)})
        old_g = sorted({carried_sdk(info, x["source"], pin) for x in view["gates"].values() if carried_sdk(info, x["source"], pin)})
        if old or old_g:
            out["carried"].append(f"{g}: " + ", ".join(
                ([f"values measured on SDK {', '.join(old)}"] if old else []) + ([f"gates on SDK {', '.join(old_g)}"] if old_g else []))
                + ("" if info["card"].get("carried") == "drop" else " (shown marked)"))
        if g not in re_measured:
            continue
        if metric_applies("spec_acceptance", facts["kind"], feats):
            m, i = shown.get("spec_acceptance", ({}, None))
            if i is None or carried_sdk(info, sessions[i]["source"], pin):
                out["acceptance"].append(f"{g}: drafted-token acceptance not re-measured on the pin (and no reason recorded)")
            elif m.get("loom") is None and not m.get("unreported"):
                out["acceptance"].append(f"{g}: drafted-token acceptance dropped without a reason")
        if facts["kind"] == "chat":
            x = info["devices"][g].get("admission")
            if not x:
                out["admission"].append(f"{g}: no native-context admission check recorded")
            elif carried_sdk(info, x["source"], pin):
                out["admission"].append(f"{g}: the admission check ran on another SDK than the pin")
            elif not x["admitted"]:
                out["admission"].append(f"{g}: the admission check refused the served context ({x.get('detail')})")
    return out


def lint(card, form="production", runs=None, generator=HERE, files=None):
    """{check: [problem]} of one card directory (empty lists: it passes). "carried" is informational. files: the
    repository's paths when the directory holds only some of them (a download without the weights)."""
    runs = runs or Runs()
    res = {k: [] for k in ("files", "runs", "versions", "render", "delta", "acceptance", "admission", "carried")}
    prod, tmp = compose(card, form)
    try:
        res["files"] = check_package(prod, files)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    info = None
    if os.path.isfile(os.path.join(card, "info.json")):
        info = load_info(os.path.join(card, "info.json"))
        errs = validate(info)
        if errs:
            res["render"].append("info.json: not valid: " + "; ".join(errs[:3]))
            return res
        res["render"] += [f"info.json: {e}" for e in free_text_errors(info)]
    res["runs"] = run_problems(card, info, runs)
    pin = None
    if info is not None:
        import render_card   # noqa: E402  (this checkout's: the pin rule and the features the card shows)
        facts = package_facts(load_package(card))
        pin = render_card.required_sdk(facts, info["versions"], info["sources"])
        feats = render_card.with_gate_features({k: dict(v) for k, v in facts["features"].items()}, facts["kind"],
                                               render_card.all_gates(info, None))
        for k, v in (info["card"].get("features") or {}).items():
            if k in feats:
                feats[k] = {"status": v["status"], "value": v.get("value")}
        for k, v in info_problems(card, info, facts, feats, pin).items():
            res[k] += v
        res["render"] += render_problems(card, generator)
    res["versions"] = version_problems(card, info, pin)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--form", choices=["production", "staging"], default="production")
    ap.add_argument("--offline", action="store_true", help="do not ask GitHub for run conclusions")
    ap.add_argument("--generator", default=HERE, help="card generator scripts directory for the render check")
    ap.add_argument("--json", help="write {dir: {check: [problems]}} here")
    ap.add_argument("--listing", help="a file of DIR's repository paths, one per line (relative to DIR): the file set to check "
                    "when DIR holds a partial download (it is not itself part of the set)")
    a = ap.parse_args(argv)
    runs, report, bad = Runs(a.offline), {}, 0
    for d in a.dirs:
        listing = os.path.join(d, a.listing) if a.listing else None
        files = [ln for ln in read(listing).split("\n") if ln] if listing and os.path.isfile(listing) else None
        res = lint(d, a.form, runs, a.generator, files)
        report[d] = res
        for k, probs in res.items():
            for p in probs:
                print(f"{d}: [{k}] {p}")
        failing = any(v for k, v in res.items() if k != "carried")
        bad |= failing
        if not failing:
            print(f"{d}: ok")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1, sort_keys=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
