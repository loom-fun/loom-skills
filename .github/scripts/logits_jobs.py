#!/usr/bin/env python3
"""DevLab jobs of the logits campaign (loom-skills#736): the default-path A/B and the host-logits checks in ONE job per model.

usage: logits_jobs.py --device DEVICE --ref REF [--bundle URL SHA256] [--no-gufo] [--no-logits] [--no-ab] [--old-org loom-fun-staging] [ab_jobs.py options] SLUG... > .github/devlab/perf.json

ab_jobs.py's job (same runner, same settings, ABBA sessions) with the NEW side built from REF (a branch or commit with the logits-capable starter:
variant "staging" = build.py of that ref with the card's own settings; "fun" = the OLD published package, default loom-fun-staging) and these steps added:
  logits:staging   host logits readback checks of the new package (devlab_perf.sh, logits_check.py)
  gufo:staging     Gufo conformance on the new package (unless --no-gufo)
  --no-ab drops the perf/c4/textmatch steps (a logits-only pilot: func, ktest, logits, gufo).
The new package's default path is judged by strix_ab_compare.py on the job exactly as for any A/B (+-1 % target: the per-row bands apply).
--bundle URL SHA256: the Gufo bundle tarball (loom-gufo-extension validation/devlab/build_bundle.sh) the job fetches (devlab_perf.sh gufo_bundle).
A model whose device preset differs from the default preset (fit row r9700_execution) is refused: a rebuild uses the default preset's settings.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ab_jobs  # noqa: E402


def build(a, ab_args):
    exp = ab_jobs.build(ab_jobs.parser(a.device).parse_args(ab_args))
    for j in exp["jobs"]:
        extra = []
        if any(k.startswith("require_preset_") for k in j):   # the device preset: only a larger prefill chunk can be rebuilt (build.py --prefill-device)
            slug = j["card"].split("/")[-1]
            fit = ab_jobs.card_lib.fit_rows(os.path.join(ab_jobs.ROOT, ab_jobs.card_lib.registry.fit_table(a.device)), a.device)   # with the scope rules' execution (exec.*) put back
            ex = (fit[slug].get(f"{ab_jobs.DEVICES[a.device]['runner']}_execution") or {})
            if set(ex) != {"prefill_capacity"}:
                sys.exit(f"{j['name']}: the device preset sets {sorted(ex)}; a --build rebuild reproduces only prefill_capacity: publish the package and measure the card")
            extra = ["--prefill-device", f"{a.device}={ex['prefill_capacity']}"]
        for v in j["variants"]:
            if v["name"] in ("staging", "stagingnd"):
                v["build"] = {}
                v["ref"] = a.ref
                if extra:
                    v["build_args"] = extra
        keep = [s for s in j["steps"] if not a.no_ab or s.split(":")[0] in ("func", "ktest")]
        keep = [s for s in keep if not s.startswith("gates:") or not a.no_ab]
        j["steps"] = keep + ([] if a.no_logits else ["logits:staging"] + ([] if a.no_gufo else ["gufo:staging"]))
        if a.bundle:
            j["bundle"] = {"url": a.bundle[0], "sha256": a.bundle[1]}
    exp["_comment"] = f"logits campaign (logits_jobs.py): new side built from {a.ref}; " + exp["_comment"]
    return exp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", required=True)
    ap.add_argument("--ref", required=True, help="branch or commit holding the logits-capable starter (the new side is built from it)")
    ap.add_argument("--bundle", nargs=2, metavar=("URL", "SHA256"))
    ap.add_argument("--no-gufo", action="store_true")
    ap.add_argument("--no-logits", action="store_true", help="neither logits nor gufo steps (the default-path A/B only)")
    ap.add_argument("--no-ab", action="store_true")
    ap.add_argument("--old-org", default="loom-fun-staging")
    a, rest = ap.parse_known_args(argv)
    json.dump(build(a, ["--old-org", a.old_org, *rest]), sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
