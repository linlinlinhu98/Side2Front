"""R64 style pass (user: 画风跟原图不像):

Analysis (crop-matched, _style_analysis.png): the original is
LOW-contrast SOFT tonal shading (hoodie std 72, dark 22%), fine
hatching, delicate thin lines; Counterfeit's native look is
high-contrast hard black outlines, white-dominant (std 45,
dark 13%) — a digital illustration, not a pencil drawing.

Now that CFG works (guidance 2.0), the digital-style traits can
be NEGATIVELY prompted (thick outlines/cel shading/harsh
contrast/digital painting/anime screenshot) — impossible before
R63. Combined with:
- IP-Adapter Plus scale raised 0.7 -> 1.0 (push the original's
  texture features harder)
- prompt rewritten around the measured style: soft shading,
  delicate thin lines, fine hatching, smooth tonal gradient

Input: genplus4_ht_s7_eye.png (best zipper+narrow eyes).
2 strengths x 3 seeds = 6 cands, ~100s each (CFG).
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

PROMPT = ("1boy, mature, handsome, calm expression, pencil "
          "sketch, hand-drawn, monochrome, greyscale, soft "
          "shading, delicate thin lines, fine hatching, smooth "
          "tonal gradient, light touch, waist up, front view, "
          "looking at viewer, black hair, narrow eyes, hood "
          "down, hoodie, zipper, blush, white background")
NEG = ("thick outlines, bold lines, harsh contrast, cel "
       "shading, digital painting, anime screenshot, screentone, "
       "shiny eyes, detailed eyes, lowres, bad anatomy, text, "
       "watermark, blurry, color, colored, photo, realistic, "
       "3d, shota, childish, cute, round eyes, big eyes, "
       "turtleneck, drawstrings, fat, bulky, dark background")

SEEDS = [7, 42, 998]
STRENGTHS = [0.40, 0.50]
IP_SCALE = 1.0
GUIDANCE = 2.0


def main():
    src = cv2.imread(os.path.join(OUT, "genplus4_ht_s7_eye.png"),
                     0)
    assert src is not None
    H, W = src.shape
    pil = Image.fromarray(src).convert("RGB")

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    ip_head = Image.fromarray(rgb[:400, :])
    ip_torso = Image.fromarray(rgb[280:, :])

    from diffusers import (LCMScheduler,
                           StableDiffusionImg2ImgPipeline)
    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora: {e}", flush=True)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(IP_SCALE)

    with torch.no_grad():
        emb, unc = pipe.encode_image(
            [ip_head, ip_torso], "cpu", 1,
            output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    cands = []
    for st in STRENGTHS:
        for seed in SEEDS:
            t0 = time.time()
            res = pipe(prompt=PROMPT, negative_prompt=NEG,
                       image=pil, ip_adapter_image_embeds=[emb],
                       height=664, width=520,
                       num_inference_steps=8,
                       guidance_scale=GUIDANCE, strength=st,
                       generator=torch.Generator("cpu")
                       .manual_seed(seed)).images[0]
            g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
            name = f"genstyle_s{int(st * 100)}_s{seed}.png"
            cv2.imwrite(os.path.join(OUT, name), g)
            cands.append((f"st{st} s{seed}", g))
            print(f"[genstyle {name}] {time.time() - t0:.0f}s",
                  flush=True)

    ph = 660
    n = len(cands)
    sheet = np.full((ph, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in cands:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genstyle_sheet.png"), sheet)
    print("[genstyle] sheet saved", flush=True)


if __name__ == "__main__":
    main()
