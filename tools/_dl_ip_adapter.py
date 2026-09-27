"""Download SDXL IP-Adapter weights (user-approved 2026-09-22):
h94/IP-Adapter -> sdxl_models/ip-adapter_sdxl.bin (~2.5GB)
                + models/image_encoder/* (CLIP ViT-H, ~3.5GB)
"""
from huggingface_hub import snapshot_download

p = snapshot_download(
    "h94/IP-Adapter",
    allow_patterns=["sdxl_models/ip-adapter_sdxl.bin",
                    "models/image_encoder/*"],
)
print("downloaded to:", p, flush=True)
