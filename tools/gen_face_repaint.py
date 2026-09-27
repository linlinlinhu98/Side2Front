"""R66: full-FACE generative repaint (user: 面部完全不像原图那个人;
画风还不完全像).

Precise features measured from the original (_orig_study.png):
- thick straight low-set eyebrows; narrow elongated single-eyelid
  eyes, small dark iris, calm slightly weary look
- small straight nose; small closed lips, corners slightly down
- soft oval face; shading built from PARALLEL HATCHING strokes,
  pencil-dark-gray lines (not pure black), near-white smooth skin

Whole face oval is repainted by the model (same denoise-fusion
technique as the approved eye stage — zero paste), with:
- s2fstyle LoRA ON (original's strokes)
- IP-Adapter Plus fed the original HEAD crop only, scale 0.9
- identity prompt written from the measurements above
- 'parallel hatching' added to push the stroke shading

Input: genlora_ht_s7_eye.png (best R65). 6 seeds.
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

LORA_DIR = os.path.join(ROOT, "assets", "lora", "s2fstyle")
TRIGGER = "s2fstyle"

FACE_PROMPT = (f"{TRIGGER}, 1boy face, thick straight eyebrows, "
               "narrow eyes, single eyelid, small dark iris, "
               "calm expression, weary eyes, small nose, small "
               "closed lips, oval face, monochrome, pencil "
               "sketch, soft shading, parallel hatching, "
               "delicate thin lines, looking at viewer")
FACE_NEG = ("round eyes, big eyes, double eyelid, shiny eyes, "
            "thick lips, smile, open mouth, shota, cute, "
            "childish, thick outlines, cel shading, digital "
            "painting, color, colored, photo, lowres, blurry, "
            "bad anatomy, watermark")

# face ellipse on genlora_ht_s7_eye.png (measured)
FACE = (258, 338, 82, 100)        # cx, cy, ax, ay
SEEDS = [7, 42, 998, 123, 555, 2024]
IP_SCALE = 1.0
GUIDANCE = 2.0
STRENGTH = 0.80


def _face_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cx, cy, ax, ay = FACE
    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def main():
    base = cv2.imread(os.path.join(OUT, "genlora_ht_s7_eye.png"),
                      0)
    assert base is not None
    H, W = base.shape
    mask = _face_mask((H, W))
    cv2.imwrite(os.path.join(OUT, "_face_mask.png"), mask)
    bpil = Image.fromarray(base).convert("RGB")
    mpil = Image.fromarray(mask).convert("RGB")

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    # 整张原图 + 头部 + 躯干 = 3 x 16 tokens (user: 直接把整张
    # 原图输入进去)
    ip_full = Image.fromarray(rgb)
    ip_head = Image.fromarray(rgb[:400, :])
    ip_torso = Image.fromarray(rgb[280:, :])

    from diffusers import (LCMScheduler,
                           StableDiffusionInpaintPipeline)
    from peft import PeftModel
    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    pipe = StableDiffusionInpaintPipeline.from_single_file(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora: {e}", flush=True)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(IP_SCALE)

    with torch.no_grad():
        emb, unc = pipe.encode_image(
            [ip_full, ip_head, ip_torso], "cpu", 1,
            output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)
    print(f"[genface2] ip embeds {tuple(emb.shape)} "
          f"(3 refs x 16 tokens)", flush=True)

    cands = [("base", base)]
    for seed in SEEDS:
        t0 = time.time()
        res = pipe(prompt=FACE_PROMPT, negative_prompt=FACE_NEG,
                   image=bpil, mask_image=mpil,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8,
                   guidance_scale=GUIDANCE, strength=STRENGTH,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genface2_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cands.append((f"s{seed}", g))
        print(f"[genface2 {name}] {time.time() - t0:.0f}s",
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
    cv2.imwrite(os.path.join(OUT, "_genface2_sheet.png"), sheet)
    print("[genface2] sheet saved", flush=True)


if __name__ == "__main__":
    main()
