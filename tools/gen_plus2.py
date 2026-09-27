"""R61 (user: torso s7 还行, 但衣服不像/太幼态/不像原图的男生;
把原图全部数据传进去):

Fixes on top of R60 (IP-Adapter Plus):
- MULTI-CROP reference done right: encode head crop + torso crop
  of the original together -> [1, 2, 257, 1280] embeds; the
  Resampler projects each into 16 tokens -> 32 tokens of the
  original's data reach the UNet (R60's list form was rejected by
  the pipeline's length check; the projection layer itself
  accepts num_images=2).
- Prompt trimmed under 77 tokens — R60's prompt was TRUNCATED and
  'blush, white background' never reached the model.
- Keep the de-childify wording (mature/handsome/sharp narrow
  eyes/defined jawline; negatives shota/cute/round/big eyes).

Counterfeit+LCM ~40-90s/img, 2 ref modes x 3 seeds = 6 cands.
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
          "hair, narrow eyes, sharp eyes, defined jawline, bangs, "
          "hood down, loose hoodie, high collar, zipper, long "
          "sleeves, blush, white background")
NEG = ("lowres, bad anatomy, text, error, cropped, worst "
       "quality, low quality, jpeg artifacts, signature, "
       "watermark, blurry, color, colored, photo, realistic, 3d, "
       "long hair, hat, shota, child, childish, cute, chibi, "
       "round eyes, big eyes, turtleneck, drawstrings, hunched "
       "shoulders, arms crossed, hands on chest, fat, bulky, "
       "thick neck, dark background, grey background")

SEEDS = [7, 42, 998]
STRENGTH = 0.55
CN_SCALE = 0.65
IP_SCALE = 0.7


def _guide(base):
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

    # precompute embeds: full image, and head+torso (num_images=2)
    with torch.no_grad():
        emb_full, _ = pipe.encode_image(
            ip_full, "cpu", 1, output_hidden_states=True)
        emb_full = emb_full.unsqueeze(0)               # [1,1,257,1280]
        emb_ht, _ = pipe.encode_image(
            [ip_head, ip_torso], "cpu", 1,
            output_hidden_states=True)
        emb_ht = emb_ht.unsqueeze(0)                   # [1,2,257,1280]
    print(f"[genplus2] emb_full {tuple(emb_full.shape)} "
          f"emb_ht {tuple(emb_ht.shape)}", flush=True)

    variants = [("full", emb_full), ("ht", emb_ht)]
    cands = []
    for vname, emb in variants:
        for seed in SEEDS:
            t0 = time.time()
            res = pipe(prompt=PROMPT, negative_prompt=NEG,
                       image=pil, control_image=ctrl,
                       controlnet_conditioning_scale=CN_SCALE,
                       ip_adapter_image_embeds=[emb],
                       height=664, width=520,
                       num_inference_steps=8, guidance_scale=1.0,
                       strength=STRENGTH,
                       generator=torch.Generator("cpu")
                       .manual_seed(seed)).images[0]
            g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
            name = f"genplus2_{vname}_s{seed}.png"
            cv2.imwrite(os.path.join(OUT, name), g)
            cands.append((f"{vname} s{seed}", g))
            print(f"[genplus2 {name}] {time.time() - t0:.0f}s",
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
    cv2.imwrite(os.path.join(OUT, "_genplus2_sheet.png"), sheet)
    print("[genplus2] sheet saved", flush=True)


if __name__ == "__main__":
    main()
