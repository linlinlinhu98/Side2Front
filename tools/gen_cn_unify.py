"""R58 (user: 我是要你生成, 拼接永远融合不了):

FULLY GENERATIVE, line-locked redraw:
  base = refine3_s7 (s7 head + original-stroke torso + blush),
  self-mirror symmetrized below the chin (neck/clothes axis was
  off — user: 脖子和衣服不在一条线上)
  -> Counterfeit-V3.0 + ControlNet LINEART (the original's own
     clothes/blush/stroke lines pin the composition)
  + IP-Adapter (original = identity/style)
  + LCM 8 steps, strength 0.6, FULL-frame img2img:
  the model redraws EVERYTHING as one drawing (no paste seam can
  exist — there is no paste), while the ControlNet keeps the
  original's clothes and hatching.

SD1.5 @ 520x664 ~40s/image -> 4 seeds. All models cached+approved.
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

PROMPT = ("1boy, solo, monochrome, greyscale, pencil (medium), "
          "traditional media, sketch, waist up, front view, "
          "looking at viewer, arms at sides, relaxed shoulders, "
          "broad shoulders, black hair, short hair, narrow eyes, "
          "bangs, hood down, hoodie, high collar, zipper, long "
          "sleeves, blush, pale skin, simple background, white "
          "background, masterpiece, best quality")
NEG = ("lowres, bad anatomy, text, error, cropped, worst "
       "quality, low quality, normal quality, jpeg artifacts, "
       "signature, watermark, username, blurry, artist name, "
       "color, colored, photo, realistic, 3d, long hair, hat, "
       "turtleneck, drawstrings, hunched shoulders, arms "
       "crossed, hands on chest, dark background, grey "
       "background")

SEEDS = [998, 7, 42, 123]
STRENGTH = 0.60
CN_SCALE = 0.9
SYM_Y = 412


def _symmetrize(img):
    """min-blend the torso with its own mirror below SYM_Y so the
    collar/fold mass is centered on the neck axis"""
    mir = img[:, ::-1]
    sym = np.minimum(img, mir)
    w = np.zeros(img.shape, np.float32)
    w[SYM_Y:, :] = 1.0
    w = cv2.GaussianBlur(w, (0, 0), 6)
    return (img.astype(np.float32) * (1 - w)
            + sym.astype(np.float32) * w).astype(np.uint8)


def _lineart(img):
    g = cv2.GaussianBlur(img, (0, 0), 3)
    hp = np.clip(g.astype(np.float32) - img.astype(np.float32),
                 0, None)
    return 255 - np.clip(hp * 2.0, 0, 255).astype(np.uint8)


def main():
    base = cv2.imread(os.path.join(OUT, "refine3_s7.png"), 0)
    assert base is not None
    base = _symmetrize(base)
    cv2.imwrite(os.path.join(OUT, "_cn_base_sym.png"), base)
    lines = _lineart(base)
    cv2.imwrite(os.path.join(OUT, "_cn_lines.png"), lines)
    H, W = base.shape

    pil = Image.fromarray(base).convert("RGB")
    ctrl = Image.fromarray(lines).convert("RGB")
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    ip_rgb = Image.fromarray(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))

    from diffusers import (ControlNetModel, LCMScheduler,
                           StableDiffusionControlNetImg2ImgPipeline)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)
    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    pipe = StableDiffusionControlNetImg2ImgPipeline \
        .from_single_file(ckpt, controlnet=cn,
                          torch_dtype=torch.float32,
                          safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora: {e}", flush=True)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.5)

    cands = [("sym base", base)]
    for seed in SEEDS:
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=pil, control_image=ctrl,
                   controlnet_conditioning_scale=CN_SCALE,
                   ip_adapter_image=ip_rgb,
                   height=664, width=520,
                   num_inference_steps=8, guidance_scale=1.0,
                   strength=STRENGTH,
                   generator=torch.Generator("cpu").manual_seed(
                       seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"gencn_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cands.append((f"seed {seed}", g))
        print(f"[gencn {name}] {time.time() - t0:.0f}s", flush=True)

    ph = 660
    sheet = np.full((ph, W * len(cands) + 10 * (len(cands) - 1)),
                    255, np.uint8)
    x = 0
    for label, im in cands:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_gencn_sheet.png"), sheet)
    print("[gencn] sheet saved", flush=True)


if __name__ == "__main__":
    main()
