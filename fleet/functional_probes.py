"""Read selected release-card gates from DevLab's recorded functional probes."""
import hashlib
import json
from pathlib import Path
import re


COLD = {"cold_capacity", "cold_fixed_storage", "cold_gpu_samples", "cold_host_samples", "over_capacity"}
STRESS = {"feature_smoke", "final_strict_consistency", "final_cross_mode_determinism"}


def load(path):
    return json.loads(path.read_text())


def cold_results(directory, package, names):
    doc = load(directory / "summary.json")
    hashes = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in package.rglob("*.loom") if "gen" not in p.relative_to(package).parts}
    if not hashes or hashes != doc["program_sha256"]:
        raise ValueError("cold probe used different package files")
    model = load(package / "model.json")
    preset = doc["preset"]
    if doc["execution"] != model["presets"][preset]["execution"] or doc["model_id"] != model["model"]["id"]:
        raise ValueError("cold probe measured a different preset or model")
    results = {}
    if names & {"cold_capacity", "over_capacity"}:
        error = (doc.get("over_limit") or {}).get("error", {})
        if "over_limit" not in doc and doc.get("phase") != "over_limit":
            raise ValueError("cold probe did not reach overflow")
        overflow = error.get("code") == "OUT_OF_RANGE" and error.get("type") == "loom_request_error"
        if "over_capacity" in names:
            results["over_capacity"] = (overflow, "Structured context overflow rejection" if overflow else doc.get("error", "Unexpected overflow response"))
        if "cold_capacity" in names:
            ctx = doc["execution"]["context_capacity"]
            ok = (doc["passed"] is True and doc.get("phase") == "complete" and overflow
                  and doc["logical_full_limit_usage"]["prompt_tokens"] == ctx - 1
                  and doc["sdk_stored_full_limit_usage"]["prompt_tokens"] == ctx and doc["health_after_usage"]["completion_tokens"] == 1)
            results["cold_capacity"] = (ok, f"Completed both limits at {ctx} tokens, rejected overflow and recovered" if ok else doc.get("error", "Capacity check failed"))
    if "cold_fixed_storage" in names:
        active = doc["active"]
        fixed, workspace = active["fixed_storage_total_bytes"], active["workspace_subset_bytes"]
        results["cold_fixed_storage"] = (0 < workspace <= fixed, f"Fixed {fixed / 1e9:.3f} GB including {workspace / 1e9:.3f} GB workspace")
    if names & {"cold_host_samples", "cold_gpu_samples"}:
        mem = load(directory / "memory_summary.json")
        meta = load(directory / "monitor_metadata.json")
        phases = ("cold", "loaded", "full_context")
        if not mem["identity_ok"] or not mem["stopped_by_marker"] or mem["samples"] < 1:
            raise ValueError("incomplete cold memory observations")
        maxima = [mem["maxima_by_phase"][p] for p in phases]
        if "cold_host_samples" in names:
            hwm = max(v for phase in maxima for k, v in phase.items() if k.endswith(".VmHWM"))
            avail = min(mem["minima_by_phase"][p]["host.MemAvailable"] for p in phases)
            results["cold_host_samples"] = (hwm > 0 and avail > 0, f"Process HWM {hwm / 1e9:.3f} GB, minimum host available {avail / 1e9:.3f} GB")
        if "cold_gpu_samples" in names:
            if len(meta["cards"]) != 1 or not mem["amd_cards_observed"] or not mem["driver_used_counters_observed"]:
                raise ValueError("cannot identify the measured GPU memory counters")
            card = meta["cards"][0]["card"]
            used = max(phase[card + ".mem_info_vram_used"] for phase in maxima)
            gtt = max(phase[card + ".mem_info_gtt_used"] for phase in maxima)
            results["cold_gpu_samples"] = (max(used, gtt) > 0,
                                            f"{meta['interval_seconds'] * 1000:g} ms samples: maximum VRAM {used / 1e9:.3f} GB, GTT {gtt / 1e9:.3f} GB")
    return results


def cross_mode(artifact, variant):
    outputs = {p: set() for p in range(4)}
    for concurrency in (1, 4):
        seen = set()
        for line in (artifact / f"determinism_{variant}_c{concurrency}.jsonl").read_text().splitlines():
            sample = json.loads(line)
            pair = sample["prompt"], sample["rep"]
            if pair in seen or not isinstance(sample["text"], str) or not sample["text"]:
                raise ValueError("duplicate or empty determinism sample")
            seen.add(pair)
            outputs[pair[0]].add(sample["text"])
        if seen != {(p, r) for p in range(4) for r in range(5)}:
            raise ValueError("incomplete cross-mode determinism samples")
    changed = sum(len(texts) != 1 for texts in outputs.values())
    return not changed, f"{changed}/4 prompts differed across five repetitions at concurrency one and four"


def read_probes(artifact, package, variant, requested, steps):
    names = set(requested)
    if not names or names - (COLD | STRESS):
        raise ValueError("unrecognized or empty set of specialized gates")
    results = {}
    required = {}
    if names & COLD:
        required["coldcapacity:" + variant] = True
        results.update(cold_results(artifact / f"coldcapacity_{variant}", Path(package), names & COLD))
    if "feature_smoke" in names:
        required["func:" + variant] = True
        checks = load(artifact / f"func_{variant}.json")["checks"]
        verdicts = [checks[key]["ok"] for key in ("chat", "tool_calling")]
        if any(type(v) is not bool for v in verdicts):
            raise ValueError("missing feature smoke verdict")
        results["feature_smoke"] = (all(verdicts), f"Chat {'passed' if verdicts[0] else 'failed'}, tool calling {'passed' if verdicts[1] else 'failed'}")
    if "final_strict_consistency" in names:
        required[f"consistency:{variant}:5"] = True
        outcomes = []
        for i in range(1, 6):
            log = (artifact / f"consistency_{variant}_{i}.log").read_text()
            exact = re.search(r"identical when concurrent: (\[(?:True|False)(?:, (?:True|False)){3}\]) identical when repeated alone \(same path\): (\[(?:True|False)(?:, (?:True|False)){3}\])", log)
            verdict = re.search(r"^(PASS|FAIL)$", log, re.M)
            if not exact or not verdict:
                raise ValueError("incomplete strict consistency trial")
            outcomes.append(verdict[1] == "PASS" and "False" not in exact[0])
        results["final_strict_consistency"] = (all(outcomes), f"{sum(outcomes)}/5 exact paired, alone and concurrent trials")
    if "final_cross_mode_determinism" in names:
        for concurrency in (1, 4):
            required[f"determinism:{variant}:{concurrency}:5"] = True
        results["final_cross_mode_determinism"] = cross_mode(artifact, variant)
    if required.keys() - steps.keys():
        raise ValueError("specialized probe is missing from the runner summary")
    groups = {}
    for name, (passed, detail) in results.items():
        level = "quick" if name == "over_capacity" else "cold-load-observation" if name in COLD else "final-stress"
        groups.setdefault(level, {})[name] = {"pass": passed, "detail": detail}
    return groups
