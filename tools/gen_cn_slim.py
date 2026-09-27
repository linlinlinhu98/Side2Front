"""R59 (user: 还是没拼接上/脖子和衣服不在一条线上/人物很肥大;
你就不能把衣服转化成数据传进去吗):

- Back to the SLIM s7 geometry (the transplanted fold-mass torso
  was what looked fat; the side-view hood bunch belongs at the
  SIDES of the neck in a frontal view, not across the chest).
- CLOTHES AS DATA (user's own words): the original (and a torso
  crop) go through IP-Adapter -> image-feature data fed into the
  UNet; the model learns the fabric/hatching from it.
- ControlNet lineart only locks the POSE; the drawstring marks
  are erased from the GUIDE and a zipper hint is drawn in, so the
  model draws the original's zipper placket instead.
- Fully generated, zero paste. Counterfeit+LCM, 40s/img, 6 cands.
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

PROMPT = ("1boy, solo, monochrome, greyscale, pencil (medium), "
          "traditional media, sketch, waist up, front view, "
          "looking at viewer, arms at sides, relaxed shoulders, "
          "black hair, short hair, narrow eyes, "
          "bangs, hood down, loose hoodie, high collar, zipper, "
          "long "
          "sleeves, blush, pale skin, simple background, white "
          "background, masterpiece, best quality")
NEG = ("lowres, bad anatomy, text, error, cropped, worst "
       "quality, low quality, normal quality, jpeg artifacts, "
       "signature, watermark, username, blurry, artist name, "
       "color, colored, photo, realistic, 3d, long hair, hat, "
       "turtleneck, drawstrings, hunched shoulders, arms "
       "crossed, hands on chest, fat, bulky, thick neck, dark "
       "background, grey background")

SEEDS = [998, 7, 42]
STRENGTH = 0.55
CN_SCALE = 0.65


def _guide(base):
    """lineart of s7 with the drawstrings erased and a zipper
    hint drawn at the chest center"""
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
    cv2.imwrite(os.path.join(OUT, "_cnslim_guide.png"), lines)
    pil = Image.fromarray(base).convert("RGB")
    ctrl = Image.fromarray(lines).convert("RGB")

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    ip_full = Image.fromarray(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    torso = orig[280:, :]
    ip_torso = Image.fromarray(cv2.cvtColor(torso,
                                            cv2.COLOR_BGR2RGB))

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
    pipe.set_ip_adapter_scale(0.6)

    cands = []
    for ipname, ipimg in (("full", ip_full), ("torso", ip_torso)):
        for seed in SEEDS:
            t0 = time.time()
            res = pipe(prompt=PROMPT, negative_prompt=NEG,
                       image=pil, control_image=ctrl,
                       controlnet_conditioning_scale=CN_SCALE,
                       ip_adapter_image=ipimg,
                       height=664, width=520,
                       num_inference_steps=8, guidance_scale=1.0,
                       strength=STRENGTH,
                       generator=torch.Generator("cpu").manual_seed(
                           seed)).images[0]
            g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
            name = f"genslim_{ipname}_s{seed}.png"
            cv2.imwrite(os.path.join(OUT, name), g)
            cands.append((f"{ipname} s{seed}", g))
            print(f"[genslim {name}] {time.time() - t0:.0f}s",
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
    cv2.imwrite(os.path.join(OUT, "_genslim_sheet.png"), sheet)
    print("[genslim] sheet saved", flush=True)


if __name__ == "__main__":
    main()
