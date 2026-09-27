"""R50 interim: SD1.5 (Counterfeit-V3.0, cached) FULL img2img from
the R47 composite — same harness shape as gen_frontal_sdxl.py.
Runs while the Animagine XL download finishes; gives a same-hour
baseline for the user to judge the SDXL candidates against.
"""
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
from gen_frontal_sdxl import PROMPT, NEG, BASE  # noqa: E402

STRENGTHS = [0.45, 0.60, 0.75]
SEEDS = [998]


def main():
    base = cv2.imread(BASE, 0)
    assert base is not None, BASE
    H, W = base.shape
    big = cv2.resize(base, (520, 664), interpolation=cv2.INTER_CUBIC)
    pil = Image.fromarray(big).convert("RGB")

    collar = np.zeros((H, W), np.float32)
    collar[396:, :] = 1.0
    collar = cv2.GaussianBlur(collar, (0, 0), 6)

    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    from diffusers import (LCMScheduler,
                           StableDiffusionImg2ImgPipeline)
    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora: {e}", flush=True)
    pipe.set_progress_bar_config(disable=True)

    cands = []
    for s in STRENGTHS:
        for seed in SEEDS:
            t0 = time.time()
            res = pipe(prompt=PROMPT, negative_prompt=NEG,
                       image=pil, num_inference_steps=8,
                       guidance_scale=1.0, strength=s,
                       generator=torch.Generator("cpu").manual_seed(
                           seed)).images[0]
            g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
            out = (g.astype(np.float32) * (1 - collar)
                   + base.astype(np.float32) * collar)
            out = np.clip(out, 0, 255).astype(np.uint8)
            name = f"gensd_s{int(round(s * 100))}_seed{seed}.png"
            cv2.imwrite(os.path.join(OUT, name), out)
            cands.append((f"cf s={s}", out))
            print(f"[gensd {name}] {time.time() - t0:.0f}s",
                  flush=True)

    panels = [("R47 (base)", base)] + cands
    sheet = np.full((H, W * len(panels) + 10 * (len(panels) - 1)),
                    255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_gensd_sheet.png"), sheet)
    print("[gensd] sheet saved", flush=True)


if __name__ == "__main__":
    main()
