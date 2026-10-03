# -*- coding: utf-8 -*-
"""One-shot prep-time model download via hf-mirror.com (approved list).

下载生图层所需的全部 HuggingFace 模型到 HF_HOME（默认 D:\\huggingface_cache）:
  1. runwayml/stable-diffusion-v1-5  (single-file v1-5-pruned-emaonly.safetensors)
  2. gsdf/Counterfeit-V3.0           (fp16 single file)
  3. FoivosPar/Arc2Face              (unet + arcface.onnx + ref_adapter)
  4. h94/IP-Adapter                  (models/image_encoder + ip-adapter_sd15)
  5. lllyasviel/control_v11p_sd15_lineart
  6. stablediffusionapi/anything-v5  (fp16 single file, loaded as fp32 on CPU)
  7. sczhou/CodeFormer               (codeformer.pth, GitHub Release)

GUI 运行时经典层模型（anime-face-detector / face-frontalization /
DINOv2）不需要本脚本——首次运行自动下载。

Run:  HF_ENDPOINT=https://hf-mirror.com python tools/download_models.py
"""
import os
import sys
import time
import urllib.request

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")
# hf-mirror cannot proxy the Xet CAS backend (401 from xethub.hf.co) —
# fall back to plain resolve-URL downloads through the mirror.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import hf_hub_download, snapshot_download

CACHE = os.environ["HF_HOME"]

JOBS = [
    ("SD1.5 single file", hf_hub_download, dict(
        repo_id="runwayml/stable-diffusion-v1-5",
        filename="v1-5-pruned-emaonly.safetensors")),
    ("Counterfeit-V3.0 fp16", hf_hub_download, dict(
        repo_id="gsdf/Counterfeit-V3.0",
        filename="Counterfeit-V3.0_fp16.safetensors")),
    ("Arc2Face (unet/onnx/ref_adapter)", snapshot_download, dict(
        repo_id="FoivosPar/Arc2Face",
        allow_patterns=["arc2face/*", "encoder/*", "ref_adapter/*",
                        "arcface.onnx"])),
    ("IP-Adapter adapter weights", snapshot_download, dict(
        repo_id="h94/IP-Adapter",
        allow_patterns=["models/ip-adapter_sd15*",
                        "models/image_encoder/*"])),
    ("ControlNet lineart", snapshot_download, dict(
        repo_id="lllyasviel/control_v11p_sd15_lineart",
        allow_patterns=["*.json", "*.safetensors"])),
    ("Anything-V5 anime (diffusers layout)", snapshot_download, dict(
        repo_id="stablediffusionapi/anything-v5",
        allow_patterns=["*.json", "**/*.safetensors", "tokenizer/*",
                        "scheduler/*"])),
]

for name, fn, kw in JOBS:
    t0 = time.time()
    print(f"[down] {name} ...", flush=True)
    try:
        path = fn(**kw)
        print(f"[ok] {name} -> {path} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"[FAIL] {name}: {e}", flush=True)
        sys.exit(1)

# CodeFormer 权重在 GitHub Release（不在 HF）
cf_dir = os.path.join(CACHE, "codeformer")
os.makedirs(cf_dir, exist_ok=True)
cf_path = os.path.join(cf_dir, "codeformer.pth")
if not os.path.exists(cf_path):
    url = ("https://github.com/sczhou/CodeFormer/releases/"
           "download/v0.1.0/codeformer.pth")
    print(f"[down] CodeFormer codeformer.pth ...", flush=True)
    try:
        urllib.request.urlretrieve(url, cf_path)
        print(f"[ok] CodeFormer -> {cf_path}", flush=True)
    except Exception as e:
        print(f"[FAIL] CodeFormer: {e}", flush=True)
        sys.exit(1)
else:
    print(f"[skip] CodeFormer already at {cf_path}", flush=True)

print("[done] all models", flush=True)
