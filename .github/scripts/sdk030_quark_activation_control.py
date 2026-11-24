"""Read first-token scores with only Quark activation QDQ toggled."""

import argparse
import json
import sys
from pathlib import Path


CHECKPOINT = "amd/Qwen3.8-27B-Quark-AWQ-MXFP4"
REVISION = "93245f7c00f1a8964b66a97c6e02e2fd09080be4"


def set_activation_mode(model, enabled):
    from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import QuarkOCP_MX

    def unchanged(x):
        return x

    names, seen = [], set()
    for name, layer in model.named_modules():
        scheme = getattr(layer, "scheme", None) or getattr(getattr(layer, "quant_method", None), "scheme", None)
        if not isinstance(scheme, QuarkOCP_MX) or not scheme.emulate or scheme.input_dtype != "mxfp4":
            continue
        if id(scheme) in seen:
            continue
        seen.add(id(scheme))
        if enabled:
            scheme.quant_dequant_func = scheme._sdk030_original_qdq
            del scheme._sdk030_original_qdq
        else:
            scheme._sdk030_original_qdq = scheme.quant_dequant_func
            scheme.quant_dequant_func = unchanged
        names.append(name)
    assert names and all("visual" not in name for name in names), names
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("toolkit", type=Path)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.toolkit.resolve() / "skills/loom-benchmark/scripts"))
    from bench_common import Package
    from vision_golden import builtin_images, data_url, reference_body
    from vllm_shim import VllmEngine, completion_response, parse_request

    from PIL import Image

    pkg = Package(str(args.package.resolve()))
    fixtures = builtin_images()
    chart, chart_prompt = fixtures["chart"]
    inputs = {
        "chart": (chart, chart_prompt),
        "chart_pre": (chart.resize((800, 512), Image.Resampling.BICUBIC), chart_prompt),
        "wide": fixtures["wide"],
    }
    engine = VllmEngine(CHECKPOINT, 16384, REVISION, images=2,
                        extra={"enforce_eager": True, "mm_processor_cache_gb": 0})
    props = engine.props()
    assert props["version"] == "0.18.0" and props["quantization"] == "quark", props

    def measure():
        result = {}
        for name, (image, question) in inputs.items():
            request = reference_body(pkg, data_url(image), question, 32)
            request["n_probs"] = 12
            prompt, encoded, maximum, top = parse_request(request)
            ids, scores, text, count = engine.generate(prompt, encoded, maximum, top)
            result[name] = completion_response(ids, scores, text, top, count)
        return result

    before = measure()
    disabled = engine.llm.apply_model(lambda model: set_activation_mode(model, False))
    after = measure()
    restored_layers = engine.llm.apply_model(lambda model: set_activation_mode(model, True))
    restored = measure()
    assert disabled == restored_layers and len(disabled) == 1
    result = {"reference": props, "modified_text_linear_layers": disabled[0],
              "stock_qdq": before, "without_activation_qdq": after, "restored_stock_qdq": restored}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for arm, variants in (("stock", before), ("no_activation_qdq", after), ("restored", restored)):
        for name, request in variants.items():
            first = request["completion_probabilities"][0]["top_logprobs"][:5]
            print(arm, name, request["tokens_evaluated"], first)


if __name__ == "__main__":
    main()
