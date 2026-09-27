"""R50: FULL GENERATION route (user 2026-09-22: 重新转生成路线,
换更好的模型 + 编写 harness; R47 保留为保底).

Animagine XL 3.1 (user-approved download) img2img FROM the R47
composite — its composition is already correct (frontal layout,
remapped original hair, approved features), so moderate strengths
repaint STYLE (soft unified pencil, classmate look) without
moving the layout. LCM-LoRA SDXL, 8 steps, CPU.

Harness: sweep STRENGTHS x SEEDS in ONE run -> individual PNGs +
a labeled contact sheet for the user to pick (生成的图片让我过目).

The approved ORIGINAL collar/torso (y>396) is pasted back onto
every candidate through a soft feather: the model redraws the
head/neck/collar-rolls coherently, the approved fabric stays
bit-exact.
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

BASE = os.path.join(OUT, "eyes_r47_sd_s998.png")

PROMPT = ("1boy, solo, monochrome, greyscale, pencil (medium), "
          "traditional media, sketch, portrait, front view, looking "
          "at viewer, short hair, bangs, hair between eyes, hood "
          "down, hoodie, zipper, pale skin, simple background, "
          "white background, masterpiece, best quality, very "
          "aesthetic")
NEG = ("lowres, bad anatomy, bad hands, text, error, missing "
       "fingers, extra digit, fewer digits, cropped, worst quality, "
       "low quality, normal quality, jpeg artifacts, signature, "
       "watermark, username, blurry, artist name, color, colored, "
       "photo, realistic, 3d, long hair, hat, turtleneck, dark "
       "background, grey background")

# SDXL ~1MP, closest multiple-of-8 to the 520x660 canvas ratio
GEN_W, GEN_H = 888, 1128
STRENGTHS = [0.50, 0.65]
SEEDS = [998]

# R51 (user: 肩膀还是太窄了，不像正常人 + 衣服部分作为参考):
# PROPORTION-FIX the base before generation — head shrinks x0.92
# about the center, torso widens x1.25 (mirror-route legacy made
# head:shoulders ~1:1.33; the classmate ref is ~1:2). Smooth
# transition across the neck so there is no shear line. The model
# then repaints the anatomy coherently from this corrected layout;
# NO collar paste-back (pasting would lock the narrow shoulders).
HEAD_K, BODY_K, SPLIT = 0.92, 1.25, 395.0


def _proportion_fix(img):
    H, W = img.shape
    yy = np.arange(H, dtype=np.float32)[:, None]
    k = BODY_K + (HEAD_K - BODY_K) * np.clip((SPLIT + 12.0 - yy)
                                             / 24.0, 0, 1)
    xs = np.arange(W, dtype=np.float32)[None, :]
    map_x = (256.0 + (xs - 256.0) / k).astype(np.float32)
    map_y = np.tile(yy, (1, W)).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def main():
    base = cv2.imread(BASE, 0)
    assert base is not None, BASE
    H, W = base.shape
    base = _proportion_fix(base)
    cv2.imwrite(os.path.join(OUT, "_genxl_wide_base.png"), base)
    big = cv2.resize(base, (GEN_W, GEN_H), interpolation=cv2.INTER_CUBIC)
    pil = Image.fromarray(big).convert("RGB")

    # no paste-back: the shoulders themselves must change
    collar = np.zeros((H, W), np.float32)

    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdxl"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--cagliostrolab--animagine-xl-3.1"),
        "animagine-xl-3.1.safetensors")

    from diffusers import (LCMScheduler,
                           StableDiffusionXLImg2ImgPipeline)
    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora: {e}", flush=True)
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
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
            name = f"genxlw_s{int(round(s * 100))}_seed{seed}.png"
            cv2.imwrite(os.path.join(OUT, name), out)
            cands.append((f"s={s} seed={seed}", out))
            print(f"[genxl {name}] {time.time() - t0:.0f}s",
                  flush=True)

    # R51 track 2: the ORIGINAL side profile as the direct img2img
    # input (user: 把原图作为数据生成) at high strength so the model
    # re-composes into the prompted front view.
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"), 0)
    assert orig is not None
    obig = cv2.resize(orig, (GEN_W, GEN_H),
                      interpolation=cv2.INTER_CUBIC)
    opil = Image.fromarray(obig).convert("RGB")
    t0 = time.time()
    res = pipe(prompt=PROMPT, negative_prompt=NEG,
               image=opil, num_inference_steps=8,
               guidance_scale=1.0, strength=0.80,
               generator=torch.Generator("cpu").manual_seed(
                   998)).images[0]
    g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
    name = "genxlo_s80_seed998.png"
    cv2.imwrite(os.path.join(OUT, name), g)
    cands.append(("orig s=0.80", g))
    print(f"[genxl {name}] {time.time() - t0:.0f}s", flush=True)

    # labeled contact sheet: widened base first, then candidates
    panels = [("wide base", base)] + cands
    ph = 660
    scaled = []
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        scaled.append(p)
    sheet = np.full((ph, W * len(scaled) + 10 * (len(scaled) - 1)),
                    255, np.uint8)
    x = 0
    for p in scaled:
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genxlw_sheet.png"), sheet)
    print("[genxl] sheet saved", flush=True)


if __name__ == "__main__":
    main()
