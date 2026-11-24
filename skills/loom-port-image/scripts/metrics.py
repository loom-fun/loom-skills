#!/usr/bin/env python3
"""Quality metrics for an image port: same-noise parity against the diffusers oracle, and prompt-alignment scores with paired bootstrap intervals.

usage: metrics.py [--device cpu|cuda] [--reports DIR] parity TEST REF --same-noise
       metrics.py [--device cpu|cuda] [--reports DIR] scores SET [SET ...] [--baseline SET_NAME] [--name NAME]

A set is a directory of cases <prompt>_s<seed>_<WxH>/{image.png, meta.json, *.npy}, the layout oracle_diffusers.py
and bench_images.py write. Only cases present in every set given are scored.

  parity  TEST REF   same initial noise only (a Loom port vs the diffusers fp32 oracle, or diffusers bf16 vs fp32 for the
                     noise floor). Images: PSNR, SSIM, LPIPS (AlexNet). Every *.npy stage tensor present in both cases:
                     relative L2, max |diff|, cosine, exact-mismatch count.
  scores  SET...     CLIPScore (open_clip ViT-L-14/openai, 100*max(cos, 0)) and PickScore (yuvalkirstain/PickScore_v1,
                     logit_scale * cos) per set. With --baseline, the mean paired per-case difference against that set
                     with a 95% bootstrap interval (10,000 resamples, seed 0) and the win rate (ties count one half).
                     An interval that spans 0 is "no measurable difference". This is the only valid comparison for
                     backends that draw their own noise (stable-diffusion.cpp, thenoise): never PSNR or LPIPS there.

Reports go to REPORTS/<name>.{json,md} (default ./reports). Runs on the CPU by default so it never competes with a timed
GPU run; pass --device cuda when nothing else is running. Needs numpy, Pillow, torch, scikit-image, lpips, open_clip_torch
and transformers.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def cases(root):
    return {p.name: p for p in sorted(Path(root).iterdir()) if (p / "image.png").exists()}


def common_cases(sets):
    common = sorted(set.intersection(*(set(cases(s)) for s in sets)))
    if not common:
        raise SystemExit("no cases common to every set")
    return common


def matching_metadata(sets, names):
    fields = ("prompt", "seed", "width", "height")
    matched = {}
    for name in names:
        first = None
        for root in sets:
            path = root / name / "meta.json"
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                values = tuple(meta[key] for key in fields)
            except (KeyError, OSError, ValueError) as error:
                raise ValueError(f"{path}: missing or invalid case metadata") from error
            if first is not None and first != values:
                raise ValueError(f"{path}: prompt, seed, or image size differs for paired case {name}")
            first = values
        matched[name] = dict(zip(fields, first))
    return matched


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def tensor_stats(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    diff = a - b
    return {"rel_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(b), 1e-30)),
            "max_abs": float(np.abs(diff).max()),
            "cosine": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)),
            "exact_mismatch": int(np.count_nonzero(diff))}


def bootstrap(d, n=10000, seed=0):
    """Mean of paired differences d with a 95% percentile interval, and the win rate (ties = 1/2)."""
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n, d.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    win = float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)], "win_rate": win, "n": int(d.size)}


class Lpips:
    def __init__(self, device):
        import lpips
        import torch
        self.torch, self.device = torch, device
        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def __call__(self, a, b):
        t = lambda x: self.torch.from_numpy(x).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(self.device)
        with self.torch.inference_mode():
            return float(self.net(t(a), t(b)).item())


class ClipScorer:
    name = "open_clip ViT-L-14/openai"

    def __init__(self, device):
        import open_clip
        import torch
        self.torch, self.device = torch, device
        # The OpenAI weights were trained with QuickGELU; open_clip builds plain GELU for the bare arch unless forced.
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device, force_quick_gelu=True)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")

    def __call__(self, image, prompt):
        with self.torch.inference_mode():
            img = self.model.encode_image(self.preprocess(image)[None].to(self.device))
            txt = self.model.encode_text(self.tokenizer([prompt]).to(self.device))
            img, txt = img / img.norm(dim=-1, keepdim=True), txt / txt.norm(dim=-1, keepdim=True)
            return 100.0 * max(float((img * txt).sum()), 0.0)


class PickScorer:
    name = "yuvalkirstain/PickScore_v1"

    def __init__(self, device):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained(self.name).to(device).eval()

    def __call__(self, image, prompt):
        with self.torch.inference_mode():
            px = self.processor(images=image, return_tensors="pt").to(self.device)
            tx = self.processor(text=[prompt], padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
            img, txt = self.model.get_image_features(**px), self.model.get_text_features(**tx)
            # Newer transformers wrap the projected embeddings in a model output.
            img, txt = (getattr(x, "pooler_output", x) for x in (img, txt))
            img, txt = img / img.norm(dim=-1, keepdim=True), txt / txt.norm(dim=-1, keepdim=True)
            return float(self.model.logit_scale.exp() * (img * txt).sum())


def parity(test, ref, device):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    names = common_cases([test, ref])
    metadata = matching_metadata([test, ref], names)
    lp = Lpips(device)
    rows = []
    for name in names:
        a, b = load_rgb(test / name / "image.png"), load_rgb(ref / name / "image.png")
        row = {"case": name, "stages": {}}
        expected = (metadata[name]["height"], metadata[name]["width"], 3)
        if a.shape != expected or b.shape != expected:
            row["error"] = f"expected {expected}; got {a.shape} vs {b.shape}"
            rows.append(row)
            continue
        row["psnr"] = float(peak_signal_noise_ratio(b, a, data_range=255))
        row["ssim"] = float(structural_similarity(b, a, channel_axis=2, data_range=255))
        row["lpips"] = lp(a, b)
        for fa in sorted((test / name).glob("*.npy")):
            fb = ref / name / fa.name
            if fb.exists():
                left, right = np.load(fa), np.load(fb)
                if left.shape != right.shape:
                    raise ValueError(f"{name}/{fa.name}: stage shapes differ: {left.shape} vs {right.shape}")
                row["stages"][fa.stem] = tensor_stats(left, right)
        rows.append(row)
        print(f"{name}: psnr {row['psnr']:.2f} ssim {row['ssim']:.4f} lpips {row['lpips']:.4f}")
    ok = [r for r in rows if "error" not in r]
    summary = {k: {"mean": float(np.mean([r[k] for r in ok])), "min": float(np.min([r[k] for r in ok])),
                   "max": float(np.max([r[k] for r in ok]))} for k in ("psnr", "ssim", "lpips")} if ok else {}
    return {"kind": "parity", "test": str(test), "ref": str(ref), "n": len(rows), "summary": summary, "cases": rows}


def scores(sets, baseline, device):
    set_names = {s.name for s in sets}
    if len(set_names) != len(sets):
        raise ValueError("score set directory names must be unique")
    if baseline and baseline not in set_names:
        raise ValueError(f"--baseline {baseline} is not one of the sets")
    common = common_cases(sets)
    metadata = matching_metadata(sets, common)
    scorers = {"clip": ClipScorer(device), "pick": PickScorer(device)}
    out = {"kind": "scores", "models": {k: s.name for k, s in scorers.items()}, "cases": common, "sets": {}}
    for s in sets:
        per = {k: {} for k in scorers}
        for name in common:
            case = metadata[name]
            with Image.open(s / name / "image.png") as source:
                if source.size != (case["width"], case["height"]):
                    raise ValueError(f"{s / name}: image size differs from case metadata")
                image = source.convert("RGB")
            for k, scorer in scorers.items():
                per[k][name] = scorer(image, case["prompt"])
        entry = {"path": str(s)}
        for k in scorers:
            vals = np.array([per[k][c] for c in common])
            entry[k] = {"mean": float(vals.mean()), "std": float(vals.std()), "per_case": per[k]}
        out["sets"][s.name] = entry
        print(f"{s.name}: CLIPScore {entry['clip']['mean']:.2f} PickScore {entry['pick']['mean']:.3f} over {len(common)} cases")
    if baseline:
        out["baseline"] = baseline
        for name, entry in out["sets"].items():
            if name == baseline:
                continue
            for k in scorers:
                d = [entry[k]["per_case"][c] - out["sets"][baseline][k]["per_case"][c] for c in common]
                entry[k]["vs_baseline"] = bootstrap(d)
            b = entry["pick"]["vs_baseline"]
            print(f"{name} vs {baseline}: PickScore {b['mean']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] win {b['win_rate']:.2f}")
    return out


def markdown(report):
    if report["kind"] == "parity":
        stages = sorted({k for r in report["cases"] for k in r.get("stages", {})})
        columns = ["case", "PSNR dB", "SSIM", "LPIPS", *(f"{stage} rel-L2" for stage in stages)]
        lines = [f"# Parity: `{Path(report['test']).name}` vs `{Path(report['ref']).name}`", "",
                 "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        for r in report["cases"]:
            if "error" in r:
                lines.append(f"| {r['case']} | {r['error']} |" + " |" * (len(columns) - 2))
                continue
            st = [f"{r['stages'][s]['rel_l2']:.2e}" if s in r["stages"] else "-" for s in stages]
            cells = [r["case"], f"{r['psnr']:.2f}", f"{r['ssim']:.4f}", f"{r['lpips']:.4f}", *st]
            lines.append("| " + " | ".join(cells) + " |")
        s = report["summary"]
        if s:
            lines += ["", f"Mean PSNR {s['psnr']['mean']:.2f} dB (min {s['psnr']['min']:.2f}), SSIM {s['ssim']['mean']:.4f}, "
                          f"LPIPS {s['lpips']['mean']:.4f} (max {s['lpips']['max']:.4f})."]
        return "\n".join(lines) + "\n"
    base = report.get("baseline")
    lines = [f"# Prompt alignment ({report['models']['clip']}; {report['models']['pick']})", "",
             f"{len(report['cases'])} cases common to every set. Differences are paired per case against `{base}`, "
             "95% bootstrap interval, 10,000 resamples." if base else f"{len(report['cases'])} cases common to every set.", "",
             "| set | CLIPScore | PickScore | PickScore diff [95% CI] | win rate |", "|---|---|---|---|---|"]
    for name, v in report["sets"].items():
        b = v["pick"].get("vs_baseline")
        diff = f"{b['mean']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]" if b else ("baseline" if name == base else "-")
        lines.append(f"| {name} | {v['clip']['mean']:.2f} | {v['pick']['mean']:.3f} | {diff} | {b['win_rate']:.2f} |"
                     if b else f"| {name} | {v['clip']['mean']:.2f} | {v['pick']['mean']:.3f} | {diff} | |")
    return "\n".join(lines) + "\n"


def write(report, reports, name):
    reports.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^\w.-]+", "_", name)
    (reports / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / f"{name}.md").write_text(markdown(report), encoding="utf-8")
    print(f"wrote {reports / name}.{{json,md}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("parity")
    pp.add_argument("test", type=Path)
    pp.add_argument("ref", type=Path)
    pp.add_argument("--same-noise", required=True, action="store_true",
                    help="confirm both image sets used the same initial latent")
    ps = sub.add_parser("scores")
    ps.add_argument("sets", type=Path, nargs="+")
    ps.add_argument("--baseline", help="set directory name the paired differences are taken against")
    ps.add_argument("--name", default="scores")
    args = ap.parse_args()
    try:
        if args.cmd == "parity":
            write(parity(args.test, args.ref, args.device), args.reports, f"parity_{args.test.name}_vs_{args.ref.name}")
        else:
            write(scores(args.sets, args.baseline, args.device), args.reports, args.name)
    except ValueError as error:
        ap.error(str(error))


if __name__ == "__main__":
    main()
