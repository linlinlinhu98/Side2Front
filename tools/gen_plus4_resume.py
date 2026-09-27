"""R63 resume: the main run died (OOM, exit 4) after 2 of 6 eye
variants. Redo ONLY the remaining 4 with the inpaint pipeline
(nothing else loaded -> no OOM)."""
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from trial_inpaint_eyes import _find_file, OUT  # noqa: E402
from gen_plus4 import EYE_PROMPT, EYE_NEG, _eye_mask  # noqa: E402

TODO = ["genplus4_full_s998", "genplus4_ht_s7",
        "genplus4_ht_s42", "genplus4_ht_s998"]


def main():
    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    from diffusers import (LCMScheduler,
                           StableDiffusionInpaintPipeline)
    ipipe = StableDiffusionInpaintPipeline.from_single_file(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    ipipe.scheduler = LCMScheduler.from_config(
        ipipe.scheduler.config)
    ipipe.load_lora_weights(lcm)
    try:
        ipipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora: {e}", flush=True)
    ipipe.set_progress_bar_config(disable=True)

    for stem in TODO:
        src = os.path.join(OUT, stem + ".png")
        dst = os.path.join(OUT, stem + "_eye.png")
        if not os.path.exists(src) or os.path.exists(dst):
            continue
        g = cv2.imread(src, 0)
        H, W = g.shape
        mask = _eye_mask((H, W))
        gpil = Image.fromarray(g).convert("RGB")
        mpil = Image.fromarray(mask).convert("RGB")
        t0 = time.time()
        res = ipipe(prompt=EYE_PROMPT, negative_prompt=EYE_NEG,
                    image=gpil, mask_image=mpil,
                    height=664, width=520,
                    num_inference_steps=8, guidance_scale=2.0,
                    strength=0.75,
                    generator=torch.Generator("cpu")
                    .manual_seed(42)).images[0]
        eg = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        eg = cv2.resize(eg, (W, H), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(dst, eg)
        print(f"[resume {stem}_eye.png] {time.time() - t0:.0f}s",
              flush=True)
    print("[resume] done", flush=True)


if __name__ == "__main__":
    main()
