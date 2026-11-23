#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from diffusers import ZImagePipeline
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image

work = Path(sys.argv[1])
scripts = work / "port/skills/loom-port-image/scripts"
sys.path.insert(0, str(scripts))
from oracle_diffusers import vae_input

evidence = work / "evidence"
output = evidence / "diagnostics"
output.mkdir()
case = evidence / "oracle/bf16/fox_s0_384x384"
meta = json.loads((case / "meta.json").read_text())
input_noise = torch.from_numpy(np.load(case / "noise.npy"))
loom = np.asarray(Image.open(evidence / "benchmark/z-image/fox_s0_384x384/image.png").convert("RGB"))
reference = np.asarray(Image.open(case / "image.png").convert("RGB"))
result = {"environment": {"torch": torch.__version__, "torch_hip": torch.version.hip,
                          "device": torch.cuda.get_device_name()}, "retained_oracle": {}, "cases": {}}


def stats(value):
    arr = value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else value
    finite = np.isfinite(arr)
    data = arr[finite]
    return {"shape": list(arr.shape), "dtype": str(value.dtype),
            "finite": int(finite.sum()), "size": arr.size, "mean": float(data.mean()) if data.size else None,
            "std": float(data.std()) if data.size else None, "min": float(data.min()) if data.size else None,
            "max": float(data.max()) if data.size else None}


def scores(value, expected):
    psnr = float(peak_signal_noise_ratio(expected, value, data_range=255))
    return {"exact": bool(np.array_equal(value, expected)),
            "psnr": psnr if np.isfinite(psnr) else None,
            "ssim": float(structural_similarity(expected, value, channel_axis=2, data_range=255))}


def image_result(name, image):
    image = image.convert("RGB")
    image.save(output / f"{name}.png")
    pixels = np.asarray(image)
    # Periodic tiles have a high exact repeat ratio at their period.
    repeated = {str(n): float((pixels[n:, :] == pixels[:-n, :]).mean()) for n in (8, 16, 24, 32, 48, 64)}
    return {"stats": stats(pixels), "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "loom": scores(pixels, loom), "old_oracle": scores(pixels, reference), "repeat_rate_y": repeated}


for array in sorted(case.glob("*.npy")):
    current = np.load(array)
    result["retained_oracle"][array.stem] = stats(current)
result["retained_oracle"]["image"] = image_result("old-oracle", Image.fromarray(reference))
print(json.dumps(result["retained_oracle"], indent=2), flush=True)

pipe = ZImagePipeline.from_pretrained(meta["model"], revision=meta["revision"], torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
torch.set_num_threads(8)
retained_vae_input = torch.from_numpy(np.load(case / "vae_in.npy"))
pipe.vae.to(device="cpu", dtype=torch.float32)
with torch.inference_mode():
    cpu_decoded = pipe.vae.decode(retained_vae_input.float(), return_dict=False)[0]
    cpu_image = pipe.image_processor.postprocess(cpu_decoded, output_type="pil")[0]
result["cases"]["saved-vae-input-cpu-fp32"] = image_result("saved-vae-input-cpu-fp32", cpu_image)
print("saved-vae-input-cpu-fp32", result["cases"]["saved-vae-input-cpu-fp32"], flush=True)
(output / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
pipe.vae.to(device="cuda", dtype=torch.bfloat16)
decode = pipe.vae.decode
decode_observations = {}


def observe_decode(latents, *args, **kwargs):
    decode_observations["input"] = stats(latents)
    decode_observations["layout"] = {"stride": list(latents.stride()), "contiguous": latents.is_contiguous()}
    decode_observations["latent"] = latents.detach().float().cpu().numpy().copy()
    outputs = decode(latents, *args, **kwargs)
    decode_observations["output"] = stats(outputs[0])
    return outputs


def probe(name, *, cudnn, injected, output_type):
    torch.backends.cudnn.enabled = cudnn
    stages = {}
    decode_observations.clear()

    def callback(_pipe, i, _t, kw):
        enc = kw["prompt_embeds"][0] if isinstance(kw["prompt_embeds"], (list, tuple)) else kw["prompt_embeds"]
        stages.setdefault("embeds", stats(enc))
        stages[f"step_{i}"] = stats(kw["latents"])
        return kw

    params = ({"latents": input_noise.clone()} if injected
              else {"generator": torch.Generator(device="cuda").manual_seed(meta["seed"])})
    with torch.inference_mode():
        images = pipe(prompt=meta["prompt"], height=meta["height"], width=meta["width"], guidance_scale=meta["guidance_scale"],
                      num_inference_steps=meta["steps"], output_type=output_type, **params,
                      callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"]).images
    torch.cuda.synchronize()
    row = {"cudnn": cudnn, "vae_dtype": str(pipe.vae.dtype), "scaling_factor": pipe.vae.config.scaling_factor,
           "shift_factor": pipe.vae.config.shift_factor, "stages": stages}
    result["cases"][name] = row
    if output_type == "pil":
        row.update(image_result(name, images[0]))
        row["decode"] = {k: v for k, v in decode_observations.items() if k != "latent"}
        np.save(output / f"{name}-vae-in.npy", decode_observations["latent"])
    else:
        row["returned_latent"] = stats(images)
        np.save(output / f"{name}.npy", images.detach().float().cpu().numpy())
    print(name, {k: v for k, v in row.items() if k not in ("stages",)}, flush=True)
    (output / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    return images


pipe.vae.decode = observe_decode
probe("stock-native", cudnn=True, injected=False, output_type="pil")
probe("stock-injected", cudnn=True, injected=True, output_type="pil")
probe("native-conv-native-noise", cudnn=False, injected=False, output_type="pil")
probe("native-conv-injected-noise", cudnn=False, injected=True, output_type="pil")
latent = probe("returned-latent", cudnn=False, injected=True, output_type="latent")

for name, cudnn, dtype in (("manual-native-conv-bf16", False, torch.bfloat16),
                           ("manual-miopen-bf16", True, torch.bfloat16),
                           ("manual-native-conv-fp32", False, torch.float32)):
    torch.backends.cudnn.enabled = cudnn
    pipe.vae.to(dtype=dtype)
    decode_observations.clear()
    with torch.inference_mode():
        vi = vae_input(pipe, "zimage", latent)
        decoded = pipe.vae.decode(vi, return_dict=False)[0]
        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    torch.cuda.synchronize()
    stock = np.load(output / "native-conv-injected-noise-vae-in.npy")
    row = image_result(name, image)
    row["decode"] = {k: v for k, v in decode_observations.items() if k != "latent"}
    row["vae_input_matches_stock"] = np.array_equal(decode_observations["latent"], stock)
    row["vae_input_max_diff"] = float(np.max(np.abs(decode_observations["latent"] - stock)))
    result["cases"][name] = row
    print(name, row, flush=True)
    (output / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
