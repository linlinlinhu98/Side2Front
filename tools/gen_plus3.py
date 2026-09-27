"""R62 (user: 衣服不像/太幼态; R61 diagnosis):

1. DRAWSTRINGS COME FROM THE INIT IMAGE: R59-R61 erased them from
   the ControlNet GUIDE only, but genxlp3_s7.png itself still has
   the two drawstrings and img2img@0.55 keeps them. Now whiten
   the drawstring rectangles in the BASE IMAGE too, so the model
   must invent the chest center -> zipper placket (guide already
   has the zipper hint).
2. EYE REPAINT STAGE (still 100% generation, zero paste): each
   candidate gets a masked Counterfeit inpaint over the eye band
   with 'narrow eyes, sharp eyes, calm expression' — the model
   repaints the eyes inside the mask and fuses at the boundary
   via denoising (same technique as the approved R47 eye strip).

Outputs genplus3_{full,ht}_s{7,42,998}.png + *_eye.png variants
+ sheets.
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

EYE_PROMPT = ("narrow eyes, sharp eyes, calm expression, mature, "
              "handsome, monochrome, greyscale, pencil (medium), "
              "traditional media, sketch, looking at viewer, "
              "best quality")
EYE_NEG = ("round eyes, big eyes, cute, shota, childish, lowres, "
           "bad anatomy, blurry, color, colored, watermark")

# drawstring rectangles on the s7 canvas (same coords as guide)
DS_RECTS = [(415, 480, 208, 242), (415, 480, 278, 308)]
# eye band on generated candidates (measured on genplus2)
EYE_BAND = (272, 336, 198, 316)          # y0, y1, x0, x1

SEEDS = [7, 42, 998]
STRENGTH = 0.55
CN_SCALE = 0.65
IP_SCALE = 0.7


def _guide(base):
    lines = _lineart(base)
    cv2.line(lines, (256, 425), (256, 535), 60, 2, cv2.LINE_AA)
    cv2.circle(lines, (256, 427), 5, 60, 1, cv2.LINE_AA)
    return lines


def _eye_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    y0, y1, x0, x1 = EYE_BAND
    cv2.ellipse(m, ((x0 + x1) // 2, (y0 + y1) // 2),
                ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360, 255,
                -1)
    return cv2.GaussianBlur(m, (0, 0), 6)


def main():
    base = cv2.imread(os.path.join(OUT, "genxlp3_s7.png"), 0)
    assert base is not None
    H, W = base.shape
    for y0, y1, x0, x1 in DS_RECTS:
        base[y0:y1, x0:x1] = 255
    cv2.imwrite(os.path.join(OUT, "_s7_nostr.png"), base)
    lines = _guide(base)
    cv2.imwrite(os.path.join(OUT, "_cnslim_guide3.png"), lines)
    pil = Image.fromarray(base).convert("RGB")
    ctrl = Image.fromarray(lines).convert("RGB")

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    ip_full = Image.fromarray(rgb)
    ip_head = Image.fromarray(rgb[:400, :])
    ip_torso = Image.fromarray(rgb[280:, :])

    from diffusers import (ControlNetModel, LCMScheduler,
                           StableDiffusionControlNetImg2ImgPipeline,
                           StableDiffusionInpaintPipeline)
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

    with torch.no_grad():
        emb_full, _ = pipe.encode_image(
            ip_full, "cpu", 1, output_hidden_states=True)
        emb_full = emb_full.unsqueeze(0)
        emb_ht, _ = pipe.encode_image(
            [ip_head, ip_torso], "cpu", 1,
            output_hidden_states=True)
        emb_ht = emb_ht.unsqueeze(0)

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
            name = f"genplus3_{vname}_s{seed}.png"
            cv2.imwrite(os.path.join(OUT, name), g)
            cands.append((f"{vname} s{seed}", g, name))
            print(f"[genplus3 {name}] {time.time() - t0:.0f}s",
                  flush=True)

    del pipe, cn
    # stage 2: generative eye repaint
    ipipe = StableDiffusionInpaintPipeline.from_single_file(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    ipipe.scheduler = LCMScheduler.from_config(ipipe.scheduler.config)
    ipipe.load_lora_weights(lcm)
    try:
        ipipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora2: {e}", flush=True)
    ipipe.set_progress_bar_config(disable=True)

    mask = _eye_mask((H, W))
    cv2.imwrite(os.path.join(OUT, "_eye_mask3.png"), mask)
    mpil = Image.fromarray(mask).convert("RGB")
    eyes = []
    for label, g, name in cands:
        t0 = time.time()
        gpil = Image.fromarray(g).convert("RGB")
        res = ipipe(prompt=EYE_PROMPT, negative_prompt=EYE_NEG,
                    image=gpil, mask_image=mpil,
                    height=664, width=520,
                    num_inference_steps=8, guidance_scale=1.0,
                    strength=0.75,
                    generator=torch.Generator("cpu")
                    .manual_seed(SEEDS[1])).images[0]
        eg = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        eg = cv2.resize(eg, (W, H), interpolation=cv2.INTER_CUBIC)
        ename = name.replace(".png", "_eye.png")
        cv2.imwrite(os.path.join(OUT, ename), eg)
        eyes.append((label + " eye", eg))
        print(f"[genplus3 {ename}] {time.time() - t0:.0f}s",
              flush=True)

    for tag, panels in (("", [c[:2] for c in cands]),
                        ("_eye", eyes)):
        n = len(panels)
        sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
        x = 0
        for label, im in panels:
            p = im.copy()
            cv2.putText(p, label, (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2,
                        cv2.LINE_AA)
            sheet[:, x:x + W] = p
            x += W + 10
        cv2.imwrite(os.path.join(OUT, f"_genplus3{tag}_sheet.png"),
                    sheet)
    print("[genplus3] sheets saved", flush=True)


if __name__ == "__main__":
    main()
