"""Compare frozen Quark device preprocessing and projection with Hugging Face."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO = "amd/Qwen3.8-27B-Quark-AWQ-MXFP4"
REVISION = "93245f7c00f1a8964b66a97c6e02e2fd09080be4"
CASES = ("chart", "chart-pre", "wide")


def image_cases(root, toolkit):
    sys.path.insert(0, str(toolkit / "skills/loom-benchmark/scripts"))
    from vision_golden import builtin_images

    fixtures = builtin_images()
    chart = fixtures["chart"][0]
    aligned_chart = chart.resize((800, 512), Image.Resampling.BICUBIC)
    for name, image in (("chart", chart), ("chart-pre", aligned_chart), ("wide", fixtures["wide"][0])):
        image.save(root / f"{name}.png")


def loom_layers(root, toolkit, package):
    sys.path.insert(0, str(toolkit / "skills/loom-port-llm/starter/vision"))
    import encoder
    import gen_component
    import preprocess
    import st_load
    import vk_common

    config = json.loads((package / "llm_config.json").read_text())
    source = config["source"]
    assert (source["repo"], source["revision"]) == (REPO, REVISION), source
    vision = config["vision"]
    c = encoder.config_st(vision)
    plan = encoder.build(c)
    actual_program = (package / "program/vision_encoder.loom").read_text()
    assert gen_component.encoder_program(c).strip() == actual_program.strip(), "encoder source differs from frozen program"
    scheme = preprocess.FAMILY[c["family"]]
    (horizontal, h_source), (vertical, v_source) = gen_component.resize_kernels(scheme["filt"])
    raw = st_load.raw_weights([str(package / "parameter/model.safetensors")], vision)
    for name in CASES:
        work = root / name
        work.mkdir()
        image = np.asarray(Image.open(root / f"{name}.png").convert("RGB"))
        in_h, in_w, channels = image.shape
        patch, merge = c["patch"], c["merge"]
        width, height = preprocess.target_size(in_w, in_h, patch, merge, *scheme["tokens"], scheme["half_even"])
        gh, gw = height // patch, width // patch
        tokens, merged = gh * gw, gh * gw // merge**2
        max_rows = 4096
        pixels = width * height * 3
        case = vk_common.Case("quark_" + name.replace("-", "_"), str(work))
        header = np.zeros(16, np.int32)
        header[:6] = [in_w, in_h, channels, header.nbytes, width, height]
        payload = np.concatenate([header.view(np.uint8), image.reshape(-1)])
        payload_arg = case.arr(payload)
        wide_arg = case.out((max_rows * width * 3,), "i8")
        args = {key: case.out((size,), "f16" if key == "x16" else "f32")
                for key, size in encoder.buffers(c, tokens, merged).items()}
        args["pix"] = case.out((pixels,), "i8")
        args["pixf"] = case.out((pixels,))
        out_w, out_h, max_h = case.lit(width), case.lit(height), case.lit(max_rows)
        case.launch(horizontal, [out_w, max_h], [out_w, max_h, payload_arg, wide_arg])
        case.launch(vertical, [out_h, out_w], [out_h, out_w, payload_arg, wide_arg, args["pix"]])
        values = {"T": tokens, "Tp": -(-tokens // 256) * 256,
                  "M": merged, "Mp": -(-merged // 64) * 64,
                  "gw": gw, "gh": gh, "img_w": width, "n_pix": pixels, "NH": c["NH"]}
        weights, literals = {}, {}

        def argument(item):
            kind, val = item
            if kind == "B":
                return args[val]
            if kind == "W":
                if val not in weights:
                    weights[val] = case.arr(np.ascontiguousarray(raw[val]))
                return weights[val]
            value = int(eval(val, {"__builtins__": {}}, values))
            if value not in literals:
                literals[value] = case.lit(value)
            return literals[value]

        for kernel, grid, inputs in plan.ops:
            case.launch(kernel, [argument(a) for a in grid], [argument(a) for a in inputs])
        for field, filename in (("pix", "device_pixels.npy"), ("out", "projector_padded.npy")):
            value, datatype = args[field]
            case.lines.append(f'  check.file.write.npy value({value}) path("{filename}") mode(always) : {datatype}')
        assert vk_common.run("\n".join((*plan.kernels.values(), h_source, v_source)), [case], str(work)), name
        data = np.load(work / "projector_padded.npy")
        projector = data[:merged * c["out_row"]].reshape(merged, c["out_row"])
        assert np.isfinite(projector).all(), name
        np.save(root / f"{name}-loom-projection.npy", projector)
        pix = np.load(work / "device_pixels.npy").view(np.uint8).reshape(height, width, 3)
        Image.fromarray(pix).save(root / f"{name}-loom-resized.png")
        del data, projector
        for sandbox_file in work.glob("*.npy"):
            sandbox_file.unlink()
    metadata = {
        "checkpoint": f"{REPO}@{REVISION}",
        "frozen_vision_source_sha256": hashlib.sha256((package / "program/vision_encoder.loom").read_bytes()).hexdigest(),
    }
    (root / "loom-source.json").write_text(json.dumps(metadata, indent=2) + "\n")


def compare_features(candidate, reference):
    assert candidate.shape == reference.shape
    a = candidate.astype(np.float64)
    b = reference.astype(np.float64)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    assert (denom > 0).all()
    cos = (a * b).sum(axis=1) / denom
    return {"shape": list(a.shape), "min_cosine": float(cos.min()), "median_cosine": float(np.median(cos)),
            "mean_absolute_error": float(np.abs(a - b).mean()), "maximum_absolute_error": float(np.abs(a - b).max())}


def reference_layers(root, package):
    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import Qwen2VLImageProcessor
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    assert transformers.__version__ == "5.3.0", transformers.__version__
    config_path = hf_hub_download(REPO, "config.json", revision=REVISION)
    checkpoint = json.loads(Path(config_path).read_text())
    processor = Qwen2VLImageProcessor.from_pretrained(REPO, revision=REVISION)
    vision_config = Qwen3_5VisionConfig(**checkpoint["vision_config"])
    vision_config._attn_implementation = "eager"
    torch.set_num_threads(min(12, os.cpu_count() or 1))
    model = Qwen3_5VisionModel(vision_config).eval()
    prefix = "model.visual."
    with safe_open(package / "parameter/model.safetensors", framework="pt", device="cpu") as weights:
        tensors = {k.removeprefix(prefix): weights.get_tensor(k) for k in weights.keys() if k.startswith(prefix)}
        model.load_state_dict(tensors, strict=True)
    del tensors
    cached_inputs = {}
    results = {"transformers": transformers.__version__, "torch": torch.__version__, "cases": {}}
    for name in CASES:
        original = Image.open(root / f"{name}.png").convert("RGB")
        resized = Image.open(root / f"{name}-loom-resized.png").convert("RGB")
        normal = processor(images=original, return_tensors="pt")
        matched = processor(images=resized, do_resize=False, return_tensors="pt")
        assert torch.equal(normal.image_grid_thw, matched.image_grid_thw), name
        difference = (normal.pixel_values - matched.pixel_values).abs()
        cached_inputs[name] = (normal, matched)
        results["cases"][name] = {"grid_thw": normal.image_grid_thw.tolist(), "resized_rgb": list(resized.size),
                                  "fraction_preprocessed_values_different": float((difference != 0).float().mean()),
                                  "max_normalized_preprocessing_delta": float(difference.max())}
    for dtype in (torch.float32, torch.bfloat16):
        model = model.to(dtype=dtype)
        for name, (normal, matched) in cached_inputs.items():
            candidate = np.load(root / f"{name}-loom-projection.npy")
            case = results["cases"][name]
            with torch.inference_mode():
                for label, feature in (("upstream_resize", normal), ("identical_loom_pixels", matched)):
                    expected = model(feature.pixel_values.to(dtype), grid_thw=feature.image_grid_thw)
                    data = expected.pooler_output.float().numpy()
                    key = f"{dtype}_{label}"
                    case[key] = compare_features(candidate, data)
                    if dtype == torch.bfloat16 and label == "identical_loom_pixels":
                        np.save(root / f"{name}-{key}.npy", data)
    (root / "projector-comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("loom", "reference"))
    parser.add_argument("output", type=Path)
    parser.add_argument("toolkit", type=Path)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    package = args.package.resolve()
    if args.phase == "loom":
        out.mkdir(parents=True, exist_ok=True)
        image_cases(out, args.toolkit.resolve())
        loom_layers(out, args.toolkit.resolve(), package)
    else:
        reference_layers(out, package)


if __name__ == "__main__":
    main()
