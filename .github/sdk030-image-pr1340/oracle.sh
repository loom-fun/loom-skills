#!/usr/bin/env bash
set -euo pipefail
exec > >(tee "$WORK/evidence/oracle.log") 2>&1
cd "$WORK"
export HF_HOME="$WORK/hf"
export TORCH_HOME="$WORK/torch-cache"
export PYTHONNOUSERSITE=1
archive=thenoise-0.15.0-rocm10.1.0-gfx1151-x64.tar.gz
tag=thenoise-0.15.0-rocm10.1.0
digest=cc467ea02582017fdf5afede2f1476abac83397b1502b1cff55abe2dc3fade84
curl -fsSL --retry 3 "https://github.com/lemonade-sdk/thenoise/releases/download/$tag/$archive" -o "$archive"
printf '%s  %s\n' "$digest" "$archive" | sha256sum -c -
mkdir "$WORK/oracle-runtime"
tar -xzf "$archive" -C "$WORK/oracle-runtime"
rm -- "$archive"
runtime="$WORK/oracle-runtime/bin/python3"
if [ ! -x "$runtime" ]; then
  runtime=$(find "$WORK/oracle-runtime/bin" -maxdepth 1 -type f -name 'python3.[0-9]*' | sort | head -1)
fi
python3 -m venv "$WORK/pip-bootstrap"
pip_packages=$("$WORK/pip-bootstrap/bin/python" -c 'from pathlib import Path; import pip; print(Path(pip.__file__).parent.parent)')
mkdir "$WORK/pip-module"
cp -a "$pip_packages/pip" "$pip_packages"/pip-*.dist-info "$WORK/pip-module/"
libdirs=$(find "$WORK/oracle-runtime/lib" -type d \
  \( -path '*/_rocm_sdk_*/lib' -o -path '*/_rocm_sdk_*/lib/llvm/lib' -o -path '*/torch/lib' \) -print | paste -sd:)
export LD_LIBRARY_PATH="$WORK/oracle-runtime/lib:$libdirs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf 'LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH" >> "$GITHUB_ENV"
printf 'HF_HOME=%s\n' "$HF_HOME" >> "$GITHUB_ENV"
printf 'TORCH_HOME=%s\n' "$TORCH_HOME" >> "$GITHUB_ENV"
PYTHONPATH="$WORK/pip-module" "$runtime" -m pip install --no-cache-dir 'diffusers==0.43.0' 'transformers==5.24.0' 'lpips==0.1.4' \
  'scikit-image==0.26.0' 'accelerate==1.16.0' 'numpy==2.5.5'
PYTHONPATH="$WORK/pip-module" "$runtime" -m pip freeze > "$WORK/evidence/oracle-dependencies.txt"
"$runtime" - "$digest" "$WORK/evidence/oracle-runtime.json" <<'PY'
import importlib.metadata
import json
from pathlib import Path
import sys
import torch

assert torch.version.hip and torch.cuda.is_available(), "The reference environment must run on ROCm"
device = torch.cuda.get_device_properties(0)
assert device.gcnArchName.split(":")[0] == "gfx1151", device
value = {"torch": torch.__version__, "hip": torch.version.hip, "device": str(device),
         "portable_bundle_sha256": sys.argv[1],
         "versions": {name: importlib.metadata.version(name) for name in
                      ("diffusers", "transformers", "scikit-image", "accelerate", "lpips")}}
Path(sys.argv[2]).write_text(json.dumps(value, indent=2) + "\n")
print(value)
PY
scripts="$WORK/port/skills/loom-port-image/scripts"
model="Tongyi-MAI/Z-Image-Turbo"
revision=f332072aa78be7aecdf3ee76d5c247082da564a6
"$runtime" "$scripts/oracle_diffusers.py" --model "$model" --revision "$revision" --layout zimage \
  --dtype bf16 --steps 8 --guidance 0 --sizes 384x384 --prompt-file "$WORK/prompts.json" \
  --offload-text-encoder --out "$WORK/evidence/oracle"
"$runtime" "$scripts/oracle_diffusers.py" --model "$model" --revision "$revision" --layout zimage \
  --dtype bf16 --steps 8 --guidance 0 --sizes 384x384 --prompt-file "$WORK/prompts.json" \
  --offload-text-encoder --out "$WORK/evidence/oracle" | tee "$WORK/evidence/oracle-reuse.log"
grep -Fx 'skip fox_s0_384x384' "$WORK/evidence/oracle-reuse.log"
"$runtime" "$scripts/metrics.py" --reports "$WORK/evidence/metrics" parity \
  "$WORK/evidence/benchmark/z-image" "$WORK/evidence/oracle/bf16" --same-noise
"$runtime" - "$WORK/evidence" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
meta = json.loads((root / "oracle/bf16/fox_s0_384x384/meta.json").read_text())
assert meta["vae_device"] == "cpu" and meta["vae_decode_dtype"] == "fp32", meta
runtime = json.loads((root / "oracle-runtime.json").read_text())
assert meta["device"] in runtime["device"], meta
metrics = json.loads((root / "metrics/parity_z-image_vs_bf16.json").read_text())
assert metrics["n"] == 1
case = metrics["cases"][0]
assert case["psnr"] >= 35 and case["ssim"] >= 0.95 and case["lpips"] <= 0.10, case
print({name: case[name] for name in ("psnr", "ssim", "lpips")})
PY
