#!/usr/bin/env python3
"""Diffusers reference oracle started from Loom's initial noise, so its images are pixel-comparable with the port.

usage: oracle_diffusers.py --model REPO_OR_DIR --layout {flux2,zimage} --steps N --guidance G
                           [--revision SHA] [--dtype fp32|bf16] [--prompt-file prompts.json] [--prompts fox,sign]
                           [--seeds 0,42] [--sizes 1024x1024|all] [--out DIR] [--offload-text-encoder] [--no-stages]

Run it in the reference Python (PyTorch ROCm, diffusers, transformers), never in the SDK venv. --model is a
Hugging Face repo id or a local snapshot; pin --revision to the commit the card's model.json.lock names.
Steps and guidance are the checkpoint's own (FLUX.2-klein-4B: 4 and 1.0; Z-Image-Turbo: 8 and 0.0).

Per case it writes OUT/<dtype>/<prompt>_s<seed>_<WxH>/ (the layout metrics.py reads):
  image.png     decoded RGB
  meta.json     prompt, seed, size, steps, guidance, sigmas, timesteps, timings, device, versions
  noise.npy     initial latent handed to the pipeline (philox_noise.py, same seed)
  embeds.npy    prompt embeddings
  step_<i>.npy  latents after scheduler step i
  vae_in.npy    the tensor handed to the VAE decoder
Run fp32 and bf16: bf16 against fp32 is the noise floor a BF16 Loom port is judged against.

--offload-text-encoder encodes every prompt first, then moves the text encoder to the CPU before the
transformer and VAE load; use it when fp32 does not fit in the GPU carve-out.
cuDNN/MIOpen convolution is disabled: on gfx1151 under Windows it made the VAE decode take minutes and fail
at 1024x1024, while the native PyTorch convolution path is fast.
A new architecture needs a latent layout in philox_noise.py and possibly its VAE de-normalization below.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import re
import tempfile
import time
from pathlib import Path

def save(path, t):
    import numpy as np

    np.save(path, t.detach().float().cpu().numpy())


def vae_input(pipe, layout, latent):
    # Flux2KleinPipeline returns an unpatchified, de-normalized latent for output_type="latent";
    # ZImagePipeline returns the raw latent, which the VAE expects scaled and shifted.
    latent = latent.to(pipe.vae.dtype)
    if layout == "zimage":
        return latent / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    return latent


def encode(pipe, prompt):
    kwargs = {"prompt": prompt, "device": "cuda"}
    if "do_classifier_free_guidance" in inspect.signature(pipe.encode_prompt).parameters:
        kwargs["do_classifier_free_guidance"] = False
    return pipe.encode_prompt(**kwargs)[0]


def provenance(args):
    local = Path(args.model).resolve()
    if local.is_dir():
        files = []
        for path in sorted(local.rglob("*")):
            if path.is_file():
                stat = path.stat()
                files.append((path.relative_to(local).as_posix(), stat.st_size, stat.st_mtime_ns))
        model = str(local)
        snapshot = hashlib.sha256(json.dumps(files).encode()).hexdigest()
    else:
        if not re.fullmatch(r"[0-9a-f]{40}", args.revision or ""):
            raise ValueError("a Hugging Face model requires --revision with the full commit SHA")
        model, snapshot = args.model, args.revision
    source = Path(__file__)
    return {
        "model": model, "revision": args.revision, "snapshot": snapshot, "layout": args.layout,
        "steps": args.steps, "guidance_scale": args.guidance, "dtype": args.dtype,
        "offload_text_encoder": args.offload_text_encoder, "capture_stages": not args.no_stages,
        "oracle_sha256": hashlib.sha256(source.read_bytes() + source.with_name("philox_noise.py").read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "numpy")},
    }


def stage_names(settings):
    if not settings["capture_stages"]:
        return []
    return ["noise.npy", "embeds.npy", "vae_in.npy", *(f"step_{i}.npy" for i in range(settings["steps"]))]


def complete_case(case, settings):
    try:
        previous = json.loads((case / "meta.json").read_text(encoding="utf-8"))
        return previous.get("settings") == settings and all(
            (case / name).is_file() and (case / name).stat().st_size
            for name in ("image.png", *stage_names(settings))
        )
    except (OSError, ValueError, AttributeError):
        return False


def publish_case(case, settings, meta, image, noise, emb, steps, vae_in):
    import numpy as np

    case.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".oracle-", dir=case) as tmp:
        pending = Path(tmp)
        image.save(pending / "image.png")
        if settings["capture_stages"]:
            if len(steps) != settings["steps"]:
                raise ValueError(f"expected {settings['steps']} step tensors, got {len(steps)}")
            np.save(pending / "noise.npy", noise.numpy())
            save(pending / "embeds.npy", emb[0] if isinstance(emb, (list, tuple)) else emb)
            for i, state in enumerate(steps):
                np.save(pending / f"step_{i}.npy", state[0].numpy())
            save(pending / "vae_in.npy", vae_in)
        (pending / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (case / "meta.json").unlink(missing_ok=True)
        for path in case.glob("*.npy"):
            if path.name in {"noise.npy", "embeds.npy", "vae_in.npy"} or re.fullmatch(r"step_\d+\.npy", path.name):
                path.unlink()
        for name in ("image.png", *stage_names(settings), "meta.json"):
            (pending / name).replace(case / name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--layout", choices=("flux2", "zimage"), required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--guidance", type=float, required=True)
    ap.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    ap.add_argument("--prompt-file", default=str(Path(__file__).resolve().parent.parent / "reference/prompts.json"))
    ap.add_argument("--prompts", help="comma-separated prompt ids (default: all)")
    ap.add_argument("--seeds", help="comma-separated seeds (default: the prompt file's)")
    ap.add_argument("--sizes", default="1024x1024", help="comma-separated WxH, or 'all'")
    ap.add_argument("--out", default="oracle")
    ap.add_argument("--offload-text-encoder", action="store_true")
    ap.add_argument("--no-stages", action="store_true", help="only write image.png and meta.json")
    args = ap.parse_args()
    if args.steps <= 0:
        ap.error("--steps must be positive")
    try:
        shared = provenance(args)
    except (ValueError, importlib.metadata.PackageNotFoundError) as error:
        ap.error(str(error))

    cfg = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))
    prompts = cfg["prompts"]
    if args.prompts:
        keep = set(args.prompts.split(","))
        prompts = [q for q in prompts if q["id"] in keep]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    sizes = cfg["sizes"] if args.sizes == "all" else args.sizes.split(",")
    pending = []
    for q in prompts:
        for size in sizes:
            width, height = (int(v) for v in size.lower().split("x"))
            for seed in seeds:
                case = Path(args.out) / args.dtype / f"{q['id']}_s{seed}_{width}x{height}"
                settings = {**shared, "prompt_id": q["id"], "prompt": q["text"],
                            "seed": seed, "width": width, "height": height}
                if complete_case(case, settings):
                    print(f"skip {case.name}")
                else:
                    pending.append((q, width, height, seed, case, settings))
    if not pending:
        return

    import torch
    from diffusers import DiffusionPipeline
    from philox_noise import LAYOUTS

    torch.backends.cudnn.enabled = False
    t0 = time.perf_counter()
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    pipe = DiffusionPipeline.from_pretrained(shared["model"], revision=args.revision, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    pipe.text_encoder.to("cuda")
    if not args.offload_text_encoder:
        pipe.to("cuda")
    load_s = time.perf_counter() - t0

    embeds = {}
    with torch.inference_mode():
        for q in {q["id"]: q for q, *_ in pending}.values():
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            e = encode(pipe, q["text"])
            torch.cuda.synchronize()
            embeds[q["id"]] = (e, time.perf_counter() - t1)
    if args.offload_text_encoder:
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
    device = torch.cuda.get_device_name(0)
    print(f"loaded {args.model} {args.dtype} in {load_s:.1f}s on {device}")

    for q, width, height, seed, case, settings in pending:
        noise = torch.from_numpy(LAYOUTS[args.layout](seed, height, width))
        emb, t_enc = embeds[q["id"]]
        steps = []

        def on_step(_pipe, _i, _t, kw):
            steps.append(kw["latents"].detach().float().cpu())
            return kw

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            latent = pipe(prompt_embeds=emb, height=height, width=width, num_inference_steps=args.steps,
                          guidance_scale=args.guidance, latents=noise, output_type="latent",
                          callback_on_step_end=None if args.no_stages else on_step).images
            torch.cuda.synchronize()
            t_dit = time.perf_counter() - t1
            vae_in = vae_input(pipe, args.layout, latent)
            pixels = pipe.vae.decode(vae_in, return_dict=False)[0]
            image = pipe.image_processor.postprocess(pixels, output_type="pil")[0]
            torch.cuda.synchronize()
        t_dec = time.perf_counter() - t1 - t_dit

        meta = {
            **settings, "settings": settings,
            "noise": f"loom philox4x32-10, layout {args.layout}, bf16",
            "sigmas": [float(s) for s in pipe.scheduler.sigmas.cpu()],
            "timesteps": [float(t) for t in pipe.scheduler.timesteps.cpu()],
            "seconds": {"encode": t_enc, "denoise": t_dit, "decode": t_dec, "total": t_enc + t_dit + t_dec,
                        "pipeline_load": load_s},
            "device": device, "conv_backend": "native (torch.backends.cudnn.enabled=False)",
            "versions": {**settings["versions"], "python": platform.python_version()},
        }
        publish_case(case, settings, meta, image, noise, emb, steps, vae_in)
        print(f"{case.name}: enc {t_enc:.2f}s dit {t_dit:.2f}s dec {t_dec:.2f}s")


if __name__ == "__main__":
    main()
