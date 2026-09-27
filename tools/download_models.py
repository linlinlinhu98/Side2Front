# -*- coding: utf-8 -*-
"""One-shot prep-time model download via hf-mirror.com (approved list).

Models (total ~9GB) -> D:\\huggingface_cache:
  1. runwayml/stable-diffusion-v1-5  (single-file v1-5-pruned-emaonly.safetensors)
  2. stablediffusionapi/anything-v5  (fp16 single file, loaded as fp32 on CPU)
  3. h94/IP-Adapter                  (models/image_encoder + ip-adapter_sd15)
  4. lllyasviel/control_v11p_sd15_lineart
Run:  HF_ENDPOINT=https://hf-mirror.com python tools/download_models.py
"""
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")
# hf-mirror cannot proxy the Xet CAS backend (401 from xethub.hf.co) —
# fall back to plain resolve-URL downloads through the mirror.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import hf_hub_download, snapshot_download

JOBS = [
    ("IP-Adapter adapter weights", snapshot_download, dict(
        repo_id="h94/IP-Adapter",
        allow_patterns=["models/ip-adapter_sd15*",
                        "models/image_encoder/*"])),
    ("SD1.5 single file", hf_hub_download, dict(
        repo_id="runwayml/stable-diffusion-v1-5",
        filename="v1-5-pruned-emaonly.safetensors")),
    ("Anything-V5 anime (diffusers layout)", snapshot_download, dict(
        repo_id="stablediffusionapi/anything-v5",
        allow_patterns=["*.json", "**/*.safetensors", "tokenizer/*",
                        "scheduler/*"])),
    ("ControlNet lineart", snapshot_download, dict(
        repo_id="lllyasviel/control_v11p_sd15_lineart",
        allow_patterns=["*.json", "*.safetensors"])),
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
print("[done] all models", flush=True)
