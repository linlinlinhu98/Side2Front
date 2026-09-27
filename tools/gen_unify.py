"""R57 (user: 我是要你生成, 拼接融合不了; 要和原图一模一样的
衣服/腮红/笔触):

GENERATIVE UNIFICATION at LOW img2img strength: the base
(refine3_s7.png) already carries the original's clothes, hatch
blush and strokes; Animagine repaints only ~30-40%, which fuses
the neck/collar junction and the texture boundary coherently —
a GENERATED blend, not a paste — while the original pixels mostly
survive. IP-Adapter (original) anchors identity/style.

Outputs gen_uni_s30.png / gen_uni_s40.png + sheet + compares.
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

GEN_W, GEN_H = 888, 1128

PROMPT = ("1boy, solo, monochrome, greyscale, pencil (medium), "
          "traditional media, sketch, waist up, front view, "
          "straight-on, looking at viewer, arms at sides, relaxed "
          "shoulders, broad shoulders, black hair, short hair, "
          "narrow eyes, bangs, hood down, hoodie, high collar, "
          "zipper, long sleeves, blush, pale skin, simple "
          "background, white background, masterpiece, best "
          "quality, very aesthetic")
NEG = ("lowres, bad anatomy, bad hands, text, error, cropped, "
       "worst quality, low quality, normal quality, jpeg "
       "artifacts, signature, watermark, username, blurry, "
       "artist name, color, colored, photo, realistic, 3d, "
       "long hair, hat, turtleneck, drawstrings, hunched "
       "shoulders, arms crossed, hands on chest, dark "
       "background, grey background")

STRENGTHS = [0.30, 0.40]
SEED = 998


def main():
    base = cv2.imread(os.path.join(OUT, "refine3_s7.png"), 0)
    assert base is not None
    H, W = base.shape
    big = cv2.resize(base, (GEN_W, GEN_H), interpolation=cv2.INTER_CUBIC)
    pil = Image.fromarray(big).convert("RGB")

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    ph, pw = int(orig.shape[0] * 0.55), int(orig.shape[1] * 0.25)
    orig_pad = cv2.copyMakeBorder(orig, ph, ph, pw, pw,
                                  cv2.BORDER_CONSTANT,
                                  value=(255, 255, 255))
    ip_pil = Image.fromarray(cv2.cvtColor(orig_pad,
                                          cv2.COLOR_BGR2RGB))

    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdxl"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--cagliostrolab--animagine-xl-3.1"),
        "animagine-xl-3.1.safetensors")

    from diffusers import (LCMScheduler,
                           StableDiffusionXLImg2ImgPipeline)
    from transformers import CLIPImageProcessor, CLIPVisionModel
    enc = CLIPVisionModel.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder",
        torch_dtype=torch.float32)
    pv = CLIPImageProcessor()(images=ip_pil,
                              return_tensors="pt").pixel_values
    with torch.no_grad():
        ip_emb = enc(pv).pooler_output.unsqueeze(1)

    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        ckpt, image_encoder=enc, torch_dtype=torch.float32,
        safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora: {e}", flush=True)
    pipe.vae.enable_slicing()
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                         weight_name="ip-adapter_sdxl.bin")
    pipe.set_ip_adapter_scale(0.5)
    pipe.set_progress_bar_config(disable=True)

    cands = [("refine3 base", base)]
    for s in STRENGTHS:
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=pil, ip_adapter_image_embeds=[ip_emb],
                   num_inference_steps=10, guidance_scale=1.0,
                   strength=s,
                   generator=torch.Generator("cpu").manual_seed(
                       SEED)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
        name = f"gen_uni_s{int(s * 100)}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cands.append((f"unify s={s}", g))
        print(f"[unify {name}] {time.time() - t0:.0f}s", flush=True)

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
    cv2.imwrite(os.path.join(OUT, "_gen_uni_sheet.png"), sheet)
    print("[unify] sheet saved", flush=True)


if __name__ == "__main__":
    main()
