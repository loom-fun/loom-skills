"""Helpers for the loom-factory workflow.

  select   rank the trending GGUF chat LLMs from the approved publishers that fit a
           Strix Halo, and pick the first one that has no Loom package on the target org yet
  publish  upload a finished Loom package to the target org
  info     print the info.json that `publish` would upload for a work directory

Only `publish` needs huggingface_hub; `select` and `info` use the standard library.

info.json says how fast a package is, against what, on which device, and which gates it passed. What the
package is lives elsewhere: the checkpoint and its files in model.json.lock, the drafter in llm_config.json,
the narrative in REPORT.md. The format (schema "loom-info/1"):

  {
    "schema": "loom-info/1",                       required
    "device": {"cpu": "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S", "gpu": "AMD Radeon 8060S Graphics",
               "os": "Ubuntu 24.04"},              required, see detect_device()
    "measured": "2026-11-06",                      date of the last measurement
    "baseline": {"name": "llama.cpp", "version": "llamacpp-rocm/b1357", "args": "--spec-type ..."},
    "metrics": {"decode_tps": {"loom": 40.07, "baseline": 36.83}, ...},
    "gates": {"golden": "PASS", "consistency": "PASS", ...},
    "provenance": "https://..."                    the run, PR or anything else that produced the package
  }

A key that wasn't measured is left out, never null. Readers ignore metric and gate names they don't know,
so adding one doesn't change the schema; only a breaking change does. Metric names (tok/s unless noted):

  decode_tps, decode_tps_1k, decode_tps_2k, decode_tps_32k   single-stream decode, after a 0/1k/2k/32k prompt
  prefill_1000 ... prefill_32000                             prefill of a ~N-token document
  c4_steady                                                  aggregate decode of 4 concurrent streams
  spec_acceptance                                            fraction of drafted tokens accepted (0-1)

The factory generates info.json from the files the loom-benchmark scripts write in WORK/bench (audit.json,
perf_log.jsonl, gates.json, llamacpp_pick.json, llamacpp_spec.json, gates_spec.log), never from numbers
the agent types.
"""
import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

HF = "https://huggingface.co"
# Quant files the loom-port-llm starter has dequant helpers for, in order of preference.
QUANTS = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0"]
SHARD = re.compile(r"-\d{5}-of-\d{5}(?=\.gguf$)", re.I)
GIB = 2**30
# Hugging Face organizations whose GGUF repositories scheduled runs may convert: original model
# developers and established quantization providers (organization accounts, not individuals).
PUBLISHERS = [
    "unsloth", "ggml-org", "lmstudio-community",
    "Qwen", "google", "microsoft", "amd", "nvidia", "meta-llama", "mistralai", "deepseek-ai", "openai",
    "ibm-granite", "Intel", "tencent", "LiquidAI", "allenai", "HuggingFaceTB", "tiiuae", "CohereLabs",
    "LGAI-EXAONE", "zai-org", "openbmb", "internlm", "JetBrains",
]
# The first command on every card. loom-fetch needs the hf extra to read from Hugging Face.
PIP = 'pip install "hrx-loom[hf]"'
FETCH_NOTE = "`loom-fetch` (from the Loom SDK, installed with its `hf` extra by the first command)"

INFO_SCHEMA = "loom-info/1"
DOC_SIZES = (1000, 2000, 8000, 16000, 32000)
AMD_PCI_VENDOR = "1002"
# libdrm's PCI id -> marketing name table; the distribution's copy first, then AMD's driver packages.
AMDGPU_IDS = ["/usr/share/libdrm/amdgpu.ids", "/opt/amdgpu/share/libdrm/amdgpu.ids"]


def _clean(name):
    """Hardware names without trademark marks or padding: 'AMD Radeon(TM) 8060S  ' -> 'AMD Radeon 8060S'."""
    return " ".join(re.sub(r"\((?:TM|R|C)\)", "", name, flags=re.I).split())


def _linux_gpus():
    names = set()
    for dev in glob.glob("/sys/bus/pci/devices/*"):
        def attr(k):
            try:
                with open(os.path.join(dev, k)) as f:
                    return f.read().strip().lower().removeprefix("0x")
            except OSError:
                return ""
        if attr("vendor") != AMD_PCI_VENDOR or not attr("class").startswith("03"):  # 03xxxx: display controller
            continue
        did, rev, name = attr("device"), attr("revision"), None
        for path in AMDGPU_IDS:
            try:
                with open(path, encoding="utf-8") as f:
                    rows = [l.split(",", 2) for l in f if not l.startswith("#") and l.count(",") >= 2]
            except OSError:
                continue
            rows = [(d.strip().lower(), r.strip().lower(), n.strip()) for d, r, n in rows]
            # One device id covers several products (1586 is the 8060S, 8050S or 8040S by revision).
            name = next((n for d, r, n in rows if d == did and r == rev), None)
            if name:
                break
        names.add(name or f"AMD GPU [{AMD_PCI_VENDOR}:{did}:{rev}]")
    return names


def _windows_gpus():
    import winreg
    names = set()
    cls = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"  # display adapters
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, cls) as root:
        for i in range(winreg.QueryInfoKey(root)[0]):
            try:
                with winreg.OpenKey(root, winreg.EnumKey(root, i)) as k:
                    if f"ven_{AMD_PCI_VENDOR}" in winreg.QueryValueEx(k, "MatchingDeviceId")[0].lower():
                        names.add(winreg.QueryValueEx(k, "DriverDesc")[0])
            except OSError:
                continue
    return names


def detect_device():
    """{cpu, gpu, os} of this machine. Every value comes from the hardware or the OS release, never from
    driver, kernel or patch versions, so everyone on the same device and OS release gets the same strings:
      cpu  the CPUID brand string (/proc/cpuinfo or the Windows registry report the same one)
      gpu  the marketing name of each AMD display adapter, by PCI id and revision (libdrm amdgpu.ids on
           Linux, the driver's DriverDesc on Windows), as 'AMD ...', sorted and joined with ' + '
      os   Linux: os-release NAME and VERSION_ID ('Ubuntu 24.04', not the 24.04.3 point release);
           Windows: 'Windows 10' or 'Windows 11' by build, plus the feature update ('Windows 11 25H2'),
           not the edition or the monthly update"""
    cpu, gpus, os_name = "", set(), f"{platform.system()} {platform.release()}"
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                cpu = next((l.split(":", 1)[1] for l in f if l.startswith("model name")), "")
        except OSError:
            pass
        rel = {}
        for path in ("/etc/os-release", "/usr/lib/os-release"):
            try:
                with open(path, encoding="utf-8") as f:
                    rel = dict(l.rstrip("\n").split("=", 1) for l in f if "=" in l)
                break
            except OSError:
                continue
        rel = {k: v.strip('"') for k, v in rel.items()}
        if rel.get("NAME"):
            os_name = " ".join(filter(None, (rel["NAME"], rel.get("VERSION_ID"))))
        gpus = _linux_gpus()
    elif sys.platform == "win32":
        import winreg
        hk = winreg.HKEY_LOCAL_MACHINE
        with winreg.OpenKey(hk, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0]
        # ProductName says "Windows 10" on Windows 11 too; the build number tells them apart.
        with winreg.OpenKey(hk, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as k:
            build = int(winreg.QueryValueEx(k, "CurrentBuildNumber")[0])
            try:
                feature = winreg.QueryValueEx(k, "DisplayVersion")[0]
            except OSError:
                feature = str(build)
        os_name = f"Windows {11 if build >= 22000 else 10} {feature}"
        gpus = _windows_gpus()
    # libdrm says "Radeon 8060S Graphics" where the Windows driver says "AMD Radeon(TM) 8060S Graphics".
    gpus = {g if g.startswith("AMD ") else f"AMD {g}" for g in map(_clean, gpus)}
    device = {"cpu": _clean(cpu or platform.processor()), "gpu": " + ".join(sorted(gpus)), "os": os_name}
    return {k: v for k, v in device.items() if v}


def get(url, token=None):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        nxt = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
        return json.load(r), nxt.group(1) if nxt else None


def ranked_gguf_repos(publishers, token, per_publisher):
    """GGUF repos of `publishers`, highest Hugging Face trending score first."""
    repos = []
    for author in publishers:
        q = urllib.parse.urlencode([("author", author), ("filter", "gguf"), ("sort", "trendingScore"),
                                    ("direction", "-1"), ("limit", str(per_publisher)),
                                    ("expand[]", "trendingScore"), ("expand[]", "downloads"),
                                    ("expand[]", "gated"), ("expand[]", "private"), ("expand[]", "disabled"),
                                    ("expand[]", "gguf"), ("expand[]", "pipeline_tag")])
        repos += get(f"{HF}/api/models?{q}", token)[0]
    return sorted(repos, key=lambda m: -(m.get("trendingScore") or 0))


def pick_weights(repo, params, token):
    """The preferred quant of `repo` as (quant, [files], total_bytes), or None. Shards are summed."""
    files, url = [], f"{HF}/api/models/{repo}/tree/main?recursive=true"
    while url:
        page, url = get(url, token)
        files += [f for f in page if f.get("type") == "file" and f["path"].lower().endswith(".gguf")]
    groups = {}
    for f in files:
        name = os.path.basename(f["path"]).lower()
        # Vision projectors and speculative-decoding draft heads are not the model.
        if "mmproj" in name or "imatrix" in name or name.startswith("mtp") or "fastmtp" in name:
            continue
        groups.setdefault(SHARD.sub("", f["path"]), []).append(f)
    # Below ~3.2 bits/param the file can't hold the whole model at any of QUANTS.
    floor = 0.4 * (params or 0)
    for quant in QUANTS:
        pat = re.compile(rf"(^|[-_./]){quant}([-.]|$)", re.I)
        # Plain quants before vendor variants (e.g. unsloth's UD-Q4_K_M), then the smallest.
        hits = sorted(((not re.search(r"(^|[-_./])UD-", k, re.I), -sum(f["size"] for f in g), k, g)
                       for k, g in groups.items() if pat.search(k) and sum(f["size"] for f in g) >= floor),
                      reverse=True)
        if hits:
            _, neg, _, g = hits[0]
            return quant, sorted(f["path"] for f in g), -neg
    return None


def target_name(source, org, bases):
    """`org/<publisher>-<model>-loom`, and the different source already published there, if any.

    Different sources can map to one name (`Foo-GGUF` and `Foo`, `a-b/c` and `a/b-c`, truncation).
    Publishing would overwrite that other package, so callers skip or refuse the model instead.
    """
    owner, name = source.split("/")
    name = re.sub(r"[-_.]?gguf$", "", name, flags=re.I)
    target = f"{org}/{owner}-{name}-loom"[: len(org) + 1 + 96]
    # Hugging Face repo names are case-insensitive.
    taken = [s for repo, b in bases.items() if repo.lower() == target.lower() for s in b]
    clash = next((s for s in taken if s.lower() != source.lower()), None)
    return target, clash


def published(org, token):
    """The source GGUF repos each repo on `org` declares as base_model, by repo id."""
    bases, url = {}, f"{HF}/api/models?author={org}&limit=100&expand[]=cardData"
    while url:
        page, url = get(url, token)
        for m in page:
            base = (m.get("cardData") or {}).get("base_model") or []
            bases[m["id"]] = [base] if isinstance(base, str) else base
    return bases


def slug(repo):
    return repo.replace("/", "__")


def explicit(a, token):
    """The pick for a user-named checkpoint `org/repo` or `org/repo:path/in/repo.gguf`."""
    rid, _, path = a.checkpoint.strip().partition(":")
    q = urllib.parse.urlencode([("expand[]", "gguf"), ("expand[]", "gated"), ("expand[]", "downloads"),
                                ("expand[]", "trendingScore")])
    m, _ = get(f"{HF}/api/models/{rid}?{q}", token)
    g = m.get("gguf") or {}
    if path:
        files, url = [], f"{HF}/api/models/{rid}/tree/main?recursive=true"
        while url:
            page, url = get(url, token)
            files += [f for f in page if f.get("type") == "file"]
        group = [f for f in files if SHARD.sub("", f["path"]) == SHARD.sub("", path)]
        if not group:
            sys.exit(f"{path} not found in {rid}")
        quant = next((q for q in QUANTS if re.search(rf"(^|[-_./]){q}([-.]|$)", path, re.I)), "custom")
        w = quant, sorted(f["path"] for f in group), sum(f["size"] for f in group)
    else:
        w = pick_weights(rid, g.get("total"), token)
        if not w:
            sys.exit(f"{rid} has no {'/'.join(QUANTS)} file; name one with {rid}:<file.gguf>")
    if w[2] > a.max_gib * GIB:
        sys.exit(f"{rid} {w[0]} is {w[2] / GIB:.1f} GiB, over the {a.max_gib} GiB Strix Halo budget")
    if m.get("gated"):
        print(f"warning: {rid} is gated; the HF_LOOM_FACTORY account must have accepted its license")
    approved = rid.split("/")[0].lower() in {p.lower() for p in PUBLISHERS}
    bases = published(a.org, token)
    target, clash = target_name(rid, a.org, bases)
    if clash:
        sys.exit(f"{target} already holds the Loom package of {clash}; publishing {rid} there would replace it")
    sources = {s.lower() for b in bases.values() for s in b}
    note = "" if approved else ", publisher not on the approved list"
    return {"rank": 0, "repo": rid, "trending": m.get("trendingScore"), "downloads": m.get("downloads"),
            "architecture": g.get("architecture"),
            "params_b": round((g.get("total") or 0) / 1e9, 1), "quant": w[0], "files": w[1],
            "gib": round(w[2] / GIB, 1), "target": target,
            "status": f"todo (requested; already published, will update{note})" if rid.lower() in sources
            else f"todo (requested{note})"}


def cmd_select(a):
    token = os.environ.get("HF_TOKEN") or None
    if a.checkpoint:
        pick = explicit(a, token)
        json.dump({"org": a.org, "max_gib": a.max_gib, "top": [pick], "skipped": [], "pick": pick},
                  open(a.out, "w", encoding="utf-8"), indent=1)
        return report(a, [pick], [], pick, f"### Requested checkpoint (fits in {a.max_gib} GiB)")

    attempts = {}
    if a.attempts_file and os.path.exists(a.attempts_file):
        for line in open(a.attempts_file, encoding="utf-8"):
            m = re.match(r"loom-factory-attempt-(.+)-\d+$", line.strip())
            if m:
                attempts[m.group(1)] = attempts.get(m.group(1), 0) + 1
    bases = published(a.org, token)
    sources = {s.lower() for b in bases.values() for s in b}

    top, skipped = [], []
    for m in ranked_gguf_repos(PUBLISHERS, token, a.per_publisher):
        if len(top) >= a.top:
            break
        rid, g = m["id"], m.get("gguf") or {}
        why = None
        if m.get("private") or m.get("disabled"):
            why = "private/disabled"
        elif m.get("gated"):
            why = f"gated ({m['gated']})"
        elif not g.get("chat_template") or m.get("pipeline_tag") in ("feature-extraction", "sentence-similarity"):
            why = f"not a chat LLM (arch {g.get('architecture', '?')}, {m.get('pipeline_tag') or 'no pipeline tag'})"
        if not why:
            w = pick_weights(rid, g.get("total"), token)
            if not w:
                why = "no " + "/".join(QUANTS) + " file"
            elif w[2] > a.max_gib * GIB:
                why = f"{w[0]} is {w[2] / GIB:.1f} GiB > {a.max_gib} GiB"
        if not why:
            target, clash = target_name(rid, a.org, bases)
            if clash:
                why = f"{target} already holds the Loom package of {clash}"
        if why:
            skipped.append({"repo": rid, "trending": m.get("trendingScore"), "reason": why})
            continue
        done = rid.lower() in sources
        tries = attempts.get(slug(rid), 0)
        top.append({"rank": len(top) + 1, "repo": rid, "trending": m.get("trendingScore"),
                    "downloads": m.get("downloads"),
                    "architecture": g.get("architecture"), "params_b": round((g.get("total") or 0) / 1e9, 1),
                    "quant": w[0], "files": w[1], "gib": round(w[2] / GIB, 1), "target": target,
                    "status": "published" if done else (f"failed {tries}x, skipped" if tries >= a.max_attempts
                                                        else "todo" + (f" (failed {tries}x)" if tries else ""))})
    pick = next((t for t in top if t["status"].startswith("todo")), None)
    result = {"org": a.org, "max_gib": a.max_gib, "top": top, "skipped": skipped, "pick": pick}
    json.dump(result, open(a.out, "w", encoding="utf-8"), indent=1)
    report(a, top, skipped, pick,
           f"### Top {len(top)} GGUF chat LLMs from the {len(PUBLISHERS)} approved publishers "
           f"(by Hugging Face trending score) that fit in {a.max_gib} GiB")


def report(a, top, skipped, pick, title):
    lines = [title, "", "| # | Source repo | Trending | Downloads (30d) | Arch | Params | Quant | Size | Status |",
             "|---|---|---|---|---|---|---|---|---|"]
    for t in top:
        mark = " **← this run**" if pick and t["repo"] == pick["repo"] else ""
        lines.append(f"| {t['rank'] or '-'} | [{t['repo']}]({HF}/{t['repo']}) | {t['trending'] or 0} | "
                     f"{t['downloads'] or 0:,} | "
                     f"{t['architecture']} | {t['params_b']}B | {t['quant']} | {t['gib']} GiB | {t['status']}{mark} |")
    if skipped:
        lines += ["", f"<details><summary>{len(skipped)} higher-ranked GGUF repos skipped</summary>", "",
                  *[f"- {s['repo']} (trending {s['trending'] or 0}): {s['reason']}" for s in skipped], "", "</details>"]
    lines += ["", f"Selected: `{pick['repo']}` ({', '.join(pick['files'])}) → `{pick['target']}`" if pick
              else "Nothing to do: every model in the list is published or out of attempts."]
    summary = "\n".join(lines) + "\n"
    sys.stdout.write(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8").write(summary)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"found={'true' if pick else 'false'}\n")
            if pick:
                f.write(f"source={pick['repo']}\ntarget={pick['target']}\nslug={slug(pick['repo'])}\n"
                        f"quant={pick['quant']}\nfiles={json.dumps(pick['files'])}\ngib={pick['gib']}\n")


def check_info(info):
    """Exit unless `info` is in the INFO_SCHEMA format (see the module docstring)."""
    def need(ok, what):
        if not ok:
            sys.exit(f"info.json: {what}")

    def num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    keys = {"schema", "device", "measured", "baseline", "metrics", "gates", "provenance"}
    need(set(info) <= keys, f"unknown keys {sorted(set(info) - keys)}")
    need(info.get("schema") == INFO_SCHEMA, f"schema must be {INFO_SCHEMA!r}")
    dev = info.get("device")
    need(isinstance(dev, dict) and set(dev) <= {"cpu", "gpu", "os"} and {"cpu", "os"} <= set(dev)
         and all(isinstance(v, str) and v for v in dev.values()), "device must be {cpu, gpu?, os} strings")
    for k in ("measured", "provenance"):
        need(isinstance(info.get(k, ""), str), f"{k} must be a string")
    base = info.get("baseline", {"name": ""})
    need(isinstance(base, dict) and "name" in base and set(base) <= {"name", "version", "args"}
         and all(isinstance(v, str) for v in base.values()), "baseline must be {name, version?, args?} strings")
    metrics, gates = info.get("metrics", {}), info.get("gates", {})
    need(isinstance(metrics, dict) and isinstance(gates, dict), "metrics and gates must be objects")
    for m, v in metrics.items():
        need(isinstance(v, dict) and set(v) <= {"loom", "baseline"} and num(v.get("loom"))
             and num(v.get("baseline", 0)), f"metrics.{m} must be {{loom: number, baseline?: number}}")
    for g, v in gates.items():
        need(v in ("PASS", "FAIL"), f"gates.{g} must be PASS or FAIL, got {v!r}")


def build_info(work):
    """info.json for the work directory, in the INFO_SCHEMA format. Run it on the machine that measured."""
    bench, pkg = os.path.join(work, "bench"), os.path.join(work, "pkg")

    def load(*p):
        try:
            with open(os.path.join(*p), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def text(*p):
        try:
            with open(os.path.join(*p), encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return None

    perf = []
    for line in (text(bench, "perf_log.jsonl") or "").splitlines():
        try:
            perf.append(json.loads(line))
        except ValueError:
            pass

    def final_medians(kind):
        """{metric: median} over the perf_log.jsonl rows of `kind` that carry its last label (the final build)."""
        rows = [r for r in perf if r.get("kind") == kind]
        runs = {}
        for r in rows:
            if r.get("label") != rows[-1].get("label"):
                continue
            vals = {k: r[k] for k in ("decode_tps", "decode_tps_1k", "decode_tps_2k", "decode_tps_32k", "c4_steady")
                    if r.get(k) is not None}
            for p in r.get("prefill", []) + r.get("prefill_long", []):
                vals[f"prefill_{min(DOC_SIZES, key=lambda x: abs(x - p['prompt_tokens']))}"] = p["prefill_tps"]
            for k, v in vals.items():
                runs.setdefault(k, []).append(v)
        return {k: sorted(v)[len(v) // 2] for k, v in runs.items()}

    # audit.json's headline is what the report shows; perf_log.jsonl covers metrics the roofline doesn't track.
    audit = load(bench, "audit.json")
    headline = {r["metric"]: r for r in (audit or {}).get("headline", [])}
    loom_perf, llama_perf = final_medians("loom"), final_medians("llama")
    order = ["decode_tps", "decode_tps_1k", "decode_tps_2k", "decode_tps_32k",
             *(f"prefill_{n}" for n in DOC_SIZES), "c4_steady"]
    metrics = {}
    for m in sorted({*headline, *loom_perf}, key=lambda m: (order.index(m) if m in order else len(order), m)):
        h = headline.get(m, {})
        lv = h["loom"] if h.get("loom") is not None else loom_perf.get(m)
        bv = h["llama_cpp"] if h.get("llama_cpp") is not None else llama_perf.get(m)
        if lv is not None:
            metrics[m] = {"loom": round(lv, 2), **({"baseline": round(bv, 2)} if bv is not None else {})}
    if (load(pkg, "llm_config.json") or {}).get("draft"):
        acc = re.findall(r"acceptance (\d+(?:\.\d+)?)%", text(bench, "gates_spec.log") or "")
        if acc:
            metrics["spec_acceptance"] = {"loom": round(float(acc[-1]) / 100, 3)}

    gates = {n: "PASS" if g["pass"] else "FAIL" for n, g in ((load(bench, "gates.json") or {}).get("gates") or {}).items()}
    if audit is not None:
        gates["audit"] = "FAIL" if any(c.get("result") == "FAIL" for c in audit.get("checks", [])) else "PASS"

    top = ((load(bench, "llamacpp_pick.json") or {}).get("ranked") or [{}])[0]
    baseline = {}
    if top.get("dir"):
        # The build's install directory, without the machine-specific prefix (e.g. llamacpp-rocm/b1357).
        baseline = {"name": "llama.cpp", "version": "/".join(top["dir"].rstrip("/").split("/")[-2:])}
        args = (load(bench, "llamacpp_spec.json") or {}).get("args")
        if args:
            baseline["args"] = args

    times = [r["time"] for r in perf if r.get("time")]
    info = {
        "schema": INFO_SCHEMA,
        "device": detect_device(),
        "measured": max(times)[:10] if times else None,
        "baseline": baseline or None,
        "metrics": metrics or None,
        "gates": gates or None,
        "provenance": (load(work, "attempt.json") or {}).get("run"),
    }
    info = {k: v for k, v in info.items() if v is not None}
    check_info(info)
    return info


def cmd_info(a):
    info = build_info(a.work)
    json.dump(info, sys.stdout, indent=2)
    sys.stdout.write("\n")


def cmd_publish(a):
    from huggingface_hub import HfApi, ModelCard

    api = HfApi(token=os.environ["HF_TOKEN"])
    pkg = os.path.join(a.work, "pkg")
    for need in ("model.json", "program"):
        if not os.path.exists(os.path.join(pkg, need)):
            sys.exit(f"package incomplete: {need} missing in {pkg}")
    if not any(f.endswith(".loom") for _, _, fs in os.walk(pkg) for f in fs):
        sys.exit(f"no .loom files in {pkg}")
    src = api.model_info(a.source, expand=["cardData"])
    license_ = (src.card_data or {}).get("license") if src.card_data else None
    files = json.loads(a.files)

    def read(name, parse=False):
        p = os.path.join(a.work, name)
        if not os.path.isfile(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f) if parse else f.read()

    info = build_info(a.work)

    # loom-fetch takes the GGUF from the source repo at the revision the port was validated on.
    revision = (read("source.json", True) or {}).get("revision") or src.sha
    card_path = os.path.join(pkg, "model.json")
    with open(card_path, encoding="utf-8") as f:
        model = json.load(f)
    by_name = {os.path.basename(f): f for f in files}
    model["origins"] = {"checkpoint": {"kind": "hf", "repository": a.source, "revision": revision}}
    for variant in model.get("variants", {}).values():
        for service in variant.get("services", {}).values():
            for param in service.get("parameters", {}).values():
                for entry in param.get("files", []):
                    upstream = by_name.get(os.path.basename(entry["path"]))
                    if not upstream:
                        sys.exit(f"{entry['path']} in model.json is not a file of {a.source}; "
                                 "loom-fetch could not acquire it")
                    entry["origin"] = f"@checkpoint/{upstream}"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=1)
        f.write("\n")
    subprocess.run([a.loom_fetch, card_path, "--update-lock", "--all-variants", "--no-input", "--quiet"],
                   check=True)
    # loom-fetch picks the only preset by itself. Shared materialization fails on Windows; copy works everywhere.
    presets = list(model.get("presets", {}))
    install = model["model"]["id"].split("/")[-1]
    fetch_args = [f"{HF}/{a.target}/resolve/main/model.json",
                  *([f"--preset={presets[0]}"] if len(presets) > 1 else []), "--materialize=copy"]
    # No --memory_bytes: the default elastic pool has no limit, and any fixed value is a guess per machine.
    serve_args = ["--device=amdgpu"]
    fetch = " ".join(["loom-fetch", *fetch_args, f"--output={install}"])
    serve = " ".join(["loom-serve", f"--model={install}/model.json", *serve_args])

    card = ModelCard(f"""---
base_model: {a.source}
{f"license: {license_}" if license_ else ""}
tags: [loom, loom-serve, gguf, strix-halo]
---

# {a.source.split('/')[1]} for AMD Loom

A [Loom](https://github.com/ROCm/hrx-system) package (`.loom` programs and a `model.json` card) for the
unmodified GGUF checkpoint [{a.source}]({HF}/{a.source}) (`{a.quant}`), generated on an AMD Ryzen AI Max+ 395
"Strix Halo" by the [loom-skills](https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'loom-fun/loom-skills')})
factory workflow.

The weights are not included. {FETCH_NOTE} installs the programs
from this repository and the GGUF from [{a.source}]({HF}/{a.source}), checked against `model.json.lock`:

```bash
{PIP}
{fetch}
{serve}
```

The server listens on `127.0.0.1:8080` and answers OpenAI-style `/v1/chat/completions` requests.
""")
    api.create_repo(a.target, repo_type="model", private=a.private, exist_ok=True)
    with open(os.path.join(pkg, "README.md"), "w", encoding="utf-8") as f:
        f.write(str(card))
    # Performance numbers are not claims yet: they live only in info.json and REPORT.md, and the card must
    # not mention them.
    with open(os.path.join(pkg, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    report = read("REPORT.md")
    if report is not None:
        with open(os.path.join(pkg, "REPORT.md"), "w", encoding="utf-8") as f:
            f.write(report)
    api.upload_folder(repo_id=a.target, folder_path=pkg,
                      ignore_patterns=["parameter/*", "*.gguf", ".loom-fetch.lock", ".cache/*"],
                      delete_patterns=["bench/**"],
                      commit_message=f"Loom package for {a.source} ({a.quant})")
    print(f"published {HF}/{a.target}")
    # Run the card's commands on the published copy, unmodified apart from paths and the port.
    check = os.path.join(a.work, "fetch-check")
    shutil.rmtree(check, ignore_errors=True)
    try:
        subprocess.run([a.loom_fetch, *fetch_args, f"--output={check}", "--no-input", "--quiet"], check=True)
        print(f"verified: {fetch}")
        serve_check(a, os.path.join(check, "model.json"), serve_args)
        print(f"verified: {serve}")
    finally:
        shutil.rmtree(check, ignore_errors=True)


def serve_check(a, card, serve_args):
    """Start loom-serve on the card, wait until it is ready, and require one chat completion."""
    import socket
    import time

    bindir = os.path.dirname(a.loom_fetch)
    exe = os.path.join(bindir, "loom-serve") if bindir else "loom-serve"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log = open(os.path.join(a.work, "serve-check.log"), "w", encoding="utf-8")
    proc = subprocess.Popen([exe, f"--model={card}", *serve_args, f"--listen=127.0.0.1:{port}"],
                            stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline, models = time.time() + 1800, None
        while time.time() < deadline:
            if proc.poll() is not None:
                sys.exit(f"loom-serve exited with {proc.returncode}; see serve-check.log")
            try:
                data = json.load(urllib.request.urlopen(f"{url}/v1/status", timeout=5))["data"]
            except OSError:
                data = []
            states = [m["loom"]["registration"] for m in data]
            if any("fail" in s for s in states):
                sys.exit(f"loom-serve registration failed: {json.dumps(data)[:3000]}")
            if states and all(s == "ready" for s in states):
                models = [m["id"] for m in data]
                break
            time.sleep(5)
        if not models:
            sys.exit("loom-serve did not become ready within 30 minutes")
        body = json.dumps({"model": models[0], "messages": [{"role": "user", "content": "Say hello."}],
                           "max_tokens": 16}).encode()
        req = urllib.request.Request(f"{url}/v1/chat/completions", body, {"Content-Type": "application/json"})
        reply = json.load(urllib.request.urlopen(req, timeout=1800))
        msg = reply["choices"][0]["message"]
        print("chat check:", json.dumps(msg.get("content") or msg.get("reasoning_content"))[:200])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("select")
s.add_argument("--org", default="loom-fun")
s.add_argument("--top", type=int, default=10)
s.add_argument("--max-gib", type=float, default=48)
s.add_argument("--per-publisher", type=int, default=50, help="most-downloaded GGUF repos to consider per publisher")
s.add_argument("--max-attempts", type=int, default=2)
s.add_argument("--attempts-file", help="artifact names of previous attempts, one per line")
s.add_argument("--checkpoint", help="convert this org/repo[:file.gguf] instead of ranking")
s.add_argument("--out", default="selection.json")
p = sub.add_parser("publish")
p.add_argument("--work", required=True, help="directory with pkg/, bench/, attempt.json and REPORT.md")
p.add_argument("--source", required=True)
p.add_argument("--target", required=True)
p.add_argument("--quant", required=True)
p.add_argument("--files", required=True, help="JSON list of the GGUF file paths in the source repo")
p.add_argument("--private", action="store_true")
p.add_argument("--loom-fetch", default="loom-fetch", help="loom-fetch executable that writes model.json.lock; loom-serve is taken from the same directory")
i = sub.add_parser("info")
i.add_argument("--work", required=True, help="directory with pkg/, bench/ and attempt.json")
a = ap.parse_args()
{"select": cmd_select, "publish": cmd_publish, "info": cmd_info}[a.cmd](a)
