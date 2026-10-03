"""R65: final route with the style LoRA (user-approved training).

R63 recipe (zipper + narrow eyes via CFG 2.0 + whitened
drawstring base + IP-Adapter Plus head+torso) + the freshly
trained style LoRA (trigger 's2fstyle') so the strokes/tonality
come from the ORIGINAL's own hand, then the eye-repaint stage.

Stage 1: ControlNet img2img (s7 whitened base + zipper guide)
         + LoRA + trigger in prompt, guidance 2.0, IP scale 0.7
Stage 2: generative eye-band repaint (same as R63)

Merged from gen_lora_style.py + gen_plus4.py (R63): gen_plus4
was the no-LoRA predecessor of this exact route; its shared
helpers (PROMPT/NEG/EYE_*/DS_RECTS/EYE_BAND/_guide/_eye_mask)
and constants are identical, so only its superseded main()
(no style LoRA) was dropped. Outputs
genlora_{full,ht}_s{7,42,998}.png + *_eye.png + sheets.
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
_HFC = os.environ.get("HF_HOME", r"D:\huggingface_cache")
from trial_inpaint_eyes import _find_file, OUT  # noqa: E402
from gen_anime_common import LORA_DIR, TRIGGER, _lineart  # noqa: E402

PROMPT = ("1boy, mature, handsome, calm expression, "
          "monochrome, greyscale, pencil (medium), sketch, "
          "waist up, front view, looking at viewer, "
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
GUIDANCE = 2.0


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


def _apply_lora(unet):
    from peft import PeftModel
    return PeftModel.from_pretrained(unet, LORA_DIR)


def main():
    base = cv2.imread(os.path.join(OUT, "genxlp3_s7.png"), 0)
    assert base is not None
    H, W = base.shape
    for y0, y1, x0, x1 in DS_RECTS:
        base[y0:y1, x0:x1] = 255
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
                           StableDiffusionControlNetImg2ImgPipeline,
                           StableDiffusionInpaintPipeline)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)
    lcm = _find_file(os.path.join(
        _HFC, "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        _HFC, "models--gsdf--Counterfeit-V3.0"),
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
    pipe.unet = _apply_lora(pipe.unet)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(IP_SCALE)

    with torch.no_grad():
        emb_full, unc_full = pipe.encode_image(
            ip_full, "cpu", 1, output_hidden_states=True)
        emb_full = torch.cat([unc_full.unsqueeze(0),
                              emb_full.unsqueeze(0)], dim=0)
        emb_ht, unc_ht = pipe.encode_image(
            [ip_head, ip_torso], "cpu", 1,
            output_hidden_states=True)
        emb_ht = torch.cat([unc_ht.unsqueeze(0),
                            emb_ht.unsqueeze(0)], dim=0)

    prompt = f"{TRIGGER}, {PROMPT}"
    variants = [("full", emb_full), ("ht", emb_ht)]
    cands = []
    for vname, emb in variants:
        for seed in SEEDS:
            t0 = time.time()
            res = pipe(prompt=prompt, negative_prompt=NEG,
                       image=pil, control_image=ctrl,
                       controlnet_conditioning_scale=CN_SCALE,
                       ip_adapter_image_embeds=[emb],
                       height=664, width=520,
                       num_inference_steps=8,
                       guidance_scale=GUIDANCE,
                       strength=STRENGTH,
                       generator=torch.Generator("cpu")
                       .manual_seed(seed)).images[0]
            g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
            name = f"genlora_{vname}_s{seed}.png"
            cv2.imwrite(os.path.join(OUT, name), g)
            cands.append((f"{vname} s{seed}", g, name))
            print(f"[genlora {name}] {time.time() - t0:.0f}s",
                  flush=True)

    del pipe, cn
    ipipe = StableDiffusionInpaintPipeline.from_single_file(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    ipipe.scheduler = LCMScheduler.from_config(
        ipipe.scheduler.config)
    ipipe.load_lora_weights(lcm)
    try:
        ipipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora2: {e}", flush=True)
    ipipe.unet = _apply_lora(ipipe.unet)
    ipipe.set_progress_bar_config(disable=True)

    mask = _eye_mask((H, W))
    mpil = Image.fromarray(mask).convert("RGB")
    eyes = []
    for label, g, name in cands:
        t0 = time.time()
        gpil = Image.fromarray(g).convert("RGB")
        res = ipipe(prompt=f"{TRIGGER}, {EYE_PROMPT}",
                    negative_prompt=EYE_NEG,
                    image=gpil, mask_image=mpil,
                    height=664, width=520,
                    num_inference_steps=8, guidance_scale=GUIDANCE,
                    strength=0.75,
                    generator=torch.Generator("cpu")
                    .manual_seed(42)).images[0]
        eg = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        eg = cv2.resize(eg, (W, H), interpolation=cv2.INTER_CUBIC)
        ename = name.replace(".png", "_eye.png")
        cv2.imwrite(os.path.join(OUT, ename), eg)
        eyes.append((label + " eye", eg))
        print(f"[genlora {ename}] {time.time() - t0:.0f}s",
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
        cv2.imwrite(os.path.join(OUT, f"_genlora{tag}_sheet.png"),
                    sheet)
    print("[genlora] sheets saved", flush=True)


if __name__ == "__main__":
    main()
