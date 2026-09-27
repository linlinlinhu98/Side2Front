"""R60 (user on genslim_torso_s7: 还行, 但衣服不像, 画风太幼态,
完全不像原图的男生; 你有没有把原图全部的数据、特征传入进去):

Honest answer to the question: the BASE ip-adapter squeezed the
whole original into ONE global vector — clothes texture, hatch
strokes and facial features were lost. R60 fixes that:

- IP-Adapter PLUS (16 patch tokens: face/cloth/strokes each keep
  their own feature region) instead of the 1-token base adapter.
- MULTI-CROP reference: face crop + torso crop of the original are
  fed as TWO reference images -> 32 tokens = nearly the whole
  original's data goes in (user: 把原图全部的数据传进去).
- De-childify the prompt (user: 太幼态): mature handsome teen,
  sharp narrow eyes, calm expression, defined jawline; negatives
  shota/childish/cute/round eyes/big eyes.
- Same approved s7 geometry + zipper guide as R59.

Counterfeit+LCM, ~40s/img, 6 candidates (2 ref modes x 3 seeds).
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
from gen_cn_unify import _lineart  # noqa: E402

PROMPT = ("1boy, solo, mature, handsome, calm expression, "
          "monochrome, greyscale, pencil (medium), traditional "
          "media, sketch, waist up, front view, looking at viewer, "
          "arms at sides, relaxed shoulders, black hair, short "
          "hair, narrow eyes, sharp eyes, defined jawline, "
          "bangs, hood down, loose hoodie, high collar, zipper, "
          "long sleeves, blush, pale skin, simple background, "
          "white background, masterpiece, best quality")
NEG = ("lowres, bad anatomy, text, error, cropped, worst "
       "quality, low quality, normal quality, jpeg artifacts, "
       "signature, watermark, username, blurry, artist name, "
       "color, colored, photo, realistic, 3d, long hair, hat, "
       "shota, child, childish, cute, chibi, round eyes, big "
       "eyes, turtleneck, drawstrings, hunched shoulders, arms "
       "crossed, hands on chest, fat, bulky, thick neck, dark "
       "background, grey background")

SEEDS = [7, 42, 998]
STRENGTH = 0.55
CN_SCALE = 0.65
IP_SCALE = 0.7


def _guide(base):
    """lineart of s7 with the drawstrings erased and a zipper
    hint drawn at the chest center (same as R59)"""
    lines = _lineart(base)
    lines[415:480, 208:242] = 255      # left drawstring
    lines[415:480, 278:308] = 255      # right drawstring
    cv2.line(lines, (256, 425), (256, 535), 60, 2, cv2.LINE_AA)
    cv2.circle(lines, (256, 427), 5, 60, 1, cv2.LINE_AA)
    return lines


def main():
    base = cv2.imread(os.path.join(OUT, "genxlp3_s7.png"), 0)
    assert base is not None
    H, W = base.shape
    lines = _guide(base)
    pil = Image.fromarray(base).convert("RGB")
    ctrl = Image.fromarray(lines).convert("RGB")

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    ip_full = Image.fromarray(rgb)
    # ALL the original's data, in two crops: head + torso
    ip_head = Image.fromarray(rgb[:400, :])
    ip_torso = Image.fromarray(rgb[280:, :])

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
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(IP_SCALE)

    variants = [("plus", [ip_full]),
                ("plusht", [ip_head, ip_torso])]
    cands = []
    for vname, ipimgs in variants:
        for seed in SEEDS:
            t0 = time.time()
            try:
                res = pipe(prompt=PROMPT, negative_prompt=NEG,
                           image=pil, control_image=ctrl,
                           controlnet_conditioning_scale=CN_SCALE,
                           ip_adapter_image=ipimgs,
                           height=664, width=520,
                           num_inference_steps=8, guidance_scale=1.0,
                           strength=STRENGTH,
                           generator=torch.Generator("cpu")
                           .manual_seed(seed)).images[0]
            except Exception as e:
                print(f"[genplus {vname} s{seed}] FAILED: {e}",
                      flush=True)
                if vname == "plusht":
                    print("[genplus] multi-image ip unsupported; "
                          "falling back to full-image ref",
                          flush=True)
                    variants[1] = ("plusht", [ip_full])
                continue
            g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
            name = f"genplus_{vname}_s{seed}.png"
            cv2.imwrite(os.path.join(OUT, name), g)
            cands.append((f"{vname} s{seed}", g))
            print(f"[genplus {name}] {time.time() - t0:.0f}s",
                  flush=True)

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
    cv2.imwrite(os.path.join(OUT, "_genplus_sheet.png"), sheet)
    print("[genplus] sheet saved", flush=True)


if __name__ == "__main__":
    main()
