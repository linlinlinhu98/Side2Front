"""R71: clothes-region ENHANCEMENT repaint (R70 failed: whitened
torso + hand-drawn guide -> white blob, structure lost).

Fix:
- init = base UNCHANGED (no whitening), strength 0.55 only
- guide = base lineart for the head + the R63 base's OWN torso
  lineart (genplus4_ht_s7 = historically best clothes: zipper
  teeth + pull + hood lining), plus hand-added full-length
  zipper line with 9px teeth and the RING PULL at the collar
- torso mask; s2fstyle LoRA x0.7; IP-Adapter Plus [full+torso]
  @0.9; ControlNet 0.8
4 seeds.
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

LORA_DIR = os.path.join(ROOT, "assets", "lora", "s2fstyle")
TRIGGER = "s2fstyle"
LORA_SCALE = 0.7

PROMPT = (f"{TRIGGER}, monochrome, pencil sketch, soft shading, "
          "parallel hatching, delicate thin lines, zip-up "
          "hoodie, full-length zipper, zipper teeth, ring "
          "zipper pull, hood down, thick collar rolls, dark "
          "hood lining, loose fit, long sleeves, fabric "
          "folds, white background")
NEG = ("drawstrings, pullover, turtleneck, buttons, thick "
       "outlines, cel shading, digital painting, color, "
       "colored, photo, lowres, blurry, bad anatomy, "
       "watermark, dark background")

SEEDS = [7, 42, 998, 2024]
STRENGTH = 0.55
CN_SCALE = 0.8
IP_SCALE = 0.9
GUIDANCE = 2.0
NECK_Y = 425                      # mask / repaint starts here


def _clothes_guide(base, r63):
    """head lineart from base + torso lineart from the R63 base
    (best historical clothes) + hand-added full-length zipper
    with visible teeth and the ring pull at the collar."""
    g = _lineart(base)
    g63 = _lineart(r63)
    blend = 12
    g[NECK_Y + blend:, :] = g63[NECK_Y + blend:, :]
    band = g[NECK_Y:NECK_Y + blend, :].astype(np.float32)
    band63 = g63[NECK_Y:NECK_Y + blend, :].astype(np.float32)
    w = np.linspace(0, 1, blend)[:, None]
    g[NECK_Y:NECK_Y + blend, :] = (
        band * (1 - w) + band63 * w).astype(np.uint8)
    cx = 256
    # full-length zipper: center line + teeth ticks (9px pitch,
    # 7px long so they survive the 64px latent grid)
    cv2.line(g, (cx, NECK_Y + 20), (cx, 656), 45, 2, cv2.LINE_AA)
    for y in range(NECK_Y + 26, 650, 9):
        cv2.line(g, (cx - 5, y + 3), (cx + 5, y - 3), 80, 1,
                 cv2.LINE_AA)
    # RING pull + slider at the collar
    cv2.circle(g, (cx, NECK_Y + 34), 8, 35, 2, cv2.LINE_AA)
    cv2.rectangle(g, (cx - 4, NECK_Y + 14), (cx + 4, NECK_Y + 27),
                  35, 1)
    return g


def _torso_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[NECK_Y:, :] = 255
    # keep the neck itself: carve a small notch under the chin
    cv2.ellipse(m, (256, NECK_Y), (60, 26), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def _scale_lora(unet, factor):
    for m in unet.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict) and "default" in sc:
            sc["default"] = sc["default"] * factor


def main():
    base = cv2.imread(os.path.join(OUT, "genr69_s2024_eye.png"),
                      0)
    r63 = cv2.imread(os.path.join(OUT, "genplus4_ht_s7.png"), 0)
    assert base is not None and r63 is not None
    H, W = base.shape
    init = base.copy()
    guide = _clothes_guide(base, r63)
    mask = _torso_mask((H, W))
    cv2.imwrite(os.path.join(OUT, "_cloth_guide.png"), guide)
    cv2.imwrite(os.path.join(OUT, "_cloth_init.png"), init)
    bpil = Image.fromarray(init).convert("RGB")
    mpil = Image.fromarray(mask).convert("RGB")
    ctrl = Image.fromarray(guide).convert("RGB")

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[280:, :])]

    from diffusers import (ControlNetModel, LCMScheduler,
                           StableDiffusionControlNetInpaintPipeline)
    from peft import PeftModel
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

    pipe = StableDiffusionControlNetInpaintPipeline \
        .from_single_file(ckpt, controlnet=cn,
                          torch_dtype=torch.float32,
                          safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora: {e}", flush=True)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
    _scale_lora(pipe.unet, LORA_SCALE)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(IP_SCALE)

    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    cands = [("base", base)]
    for seed in SEEDS:
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=bpil, mask_image=mpil,
                   control_image=ctrl,
                   controlnet_conditioning_scale=CN_SCALE,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8,
                   guidance_scale=GUIDANCE, strength=STRENGTH,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"gencloth2_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cands.append((f"s{seed}", g))
        print(f"[gencloth {name}] {time.time() - t0:.0f}s",
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
    cv2.imwrite(os.path.join(OUT, "_gencloth2_sheet.png"), sheet)
    print("[gencloth2] sheet saved", flush=True)


if __name__ == "__main__":
    main()
