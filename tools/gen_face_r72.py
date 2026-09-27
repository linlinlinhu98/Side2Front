"""R72: face-feature repaint with a hand-drawn CORRECTED guide
(user on base: 脸略倾斜/眼睛斜视/神情不像; 衣服以 base 为准).

Diagnosis from _baseface_grid.png + _face_study2.png:
- base eyes are upturned fox eyes (outer corners sweep up) with
  slanted brows -> sly look; original = HORIZONTAL almond eyes,
  upper lid pressing the iris (weary), straight low brows
- nose = two dots, mouth = cat dot; original = small straight
  nose line + small thin closed lips, corners slightly down
- eye line tilted ~3 deg, chin pointy/off-center

Guide: base lineart, features whitened, redrawn frontal &
symmetric (both eyes y=300, spacing 72px, almond 40x12, iris
centered; straight brows y~278; nose line to (258,333); mouth
y=358; rounded chin U). ControlNet 0.85 locks it; LoRA 0.7;
IP-Adapter Plus [full+head] @0.9. Face-oval mask only - hair,
clothes untouched. 6 seeds.
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

PROMPT = (f"{TRIGGER}, 1boy face, monochrome, pencil sketch, "
          "soft shading, delicate thin lines, almond eyes, "
          "relaxed half-closed eyes, looking at viewer, "
          "symmetrical face, straight eyebrows, small straight "
          "nose, small closed lips, thin lips, soft oval face, "
          "rounded chin, calm gentle expression, weary, blush, "
          "white background")
NEG = ("upturned eyes, slanted eyes, fox eyes, sharp eyes, "
       "big round eyes, shiny eyes, cat mouth, fang, smirk, "
       "smug, seductive, angry, pointy chin, tilted head, "
       "asymmetrical face, thick eyebrows, double eyelid, "
       "open mouth, smile, color, colored, photo, lowres, "
       "blurry, bad anatomy, watermark")

SEEDS = [7, 42, 998, 123, 555, 2024]
STRENGTH = 0.80
CN_SCALE = 0.85
IP_SCALE = 0.9
GUIDANCE = 2.0
CX = 258                          # face symmetry axis
FACE = (258, 332, 92, 110)        # repaint mask ellipse


def _face_guide(base):
    g = _lineart(base)
    # whiten the old feature regions (interior only, keep jaw)
    g[266:318, 182:338] = 255     # brows + eyes
    g[312:352, 232:288] = 255     # nose
    g[350:378, 224:296] = 255     # mouth + chin interior
    # straight horizontal brows: medium, pencil-gray (not black
    # bars)
    cv2.line(g, (198, 280), (242, 278), 85, 2, cv2.LINE_AA)
    cv2.line(g, (274, 278), (318, 280), 85, 2, cv2.LINE_AA)
    # narrow almond eyes: iris mostly COVERED by the upper lid
    # (weary), flatter wide arcs, same height, gaze centered
    for cx in (222, 294):
        cv2.circle(g, (cx, 305), 6, 100, -1, cv2.LINE_AA)
        # upper lid: low flat arc covering the iris top ~40%
        cv2.ellipse(g, (cx, 312), (21, 14), 0, 205, 335, 50, 2,
                    cv2.LINE_AA)
        cv2.line(g, (cx - 21, 300), (cx + 21, 300), 50, 1,
                 cv2.LINE_AA)
        # white-out whatever iris pokes above the lid line
        cv2.rectangle(g, (cx - 20, 294), (cx + 20, 299), 255, -1)
        cv2.line(g, (cx - 21, 300), (cx + 21, 300), 50, 1,
                 cv2.LINE_AA)
        # faint thin lower lid
        cv2.ellipse(g, (cx, 307), (15, 8), 0, 35, 145, 130, 1,
                    cv2.LINE_AA)
    # small straight nose
    cv2.line(g, (CX, 318), (CX, 331), 110, 1, cv2.LINE_AA)
    cv2.ellipse(g, (CX, 334), (5, 3), 0, 20, 160, 100, 1,
                cv2.LINE_AA)
    # small thin closed lips: single soft short stroke
    cv2.line(g, (245, 358), (271, 358), 95, 1, cv2.LINE_AA)
    cv2.line(g, (250, 363), (266, 363), 150, 1, cv2.LINE_AA)
    # subtle chin curve (shallow, light)
    cv2.ellipse(g, (CX, 330), (40, 36), 0, 30, 150, 95, 1,
                cv2.LINE_AA)
    return g


def _face_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cx, cy, ax, ay = FACE
    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def _scale_lora(unet, factor):
    for m in unet.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict) and "default" in sc:
            sc["default"] = sc["default"] * factor


def main():
    base = cv2.imread(os.path.join(OUT, "genr69_s2024_eye.png"),
                      0)
    assert base is not None
    H, W = base.shape
    guide = _face_guide(base)
    mask = _face_mask((H, W))
    cv2.imwrite(os.path.join(OUT, "_face_guide2.png"), guide)
    bpil = Image.fromarray(base).convert("RGB")
    mpil = Image.fromarray(mask).convert("RGB")
    ctrl = Image.fromarray(guide).convert("RGB")

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :])]

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
        name = f"genface4_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cands.append((f"s{seed}", g))
        print(f"[r72 {name}] {time.time() - t0:.0f}s", flush=True)

    n = len(cands)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in cands:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genface4_sheet.png"), sheet)
    # face zoom sheet for judging features
    crops = [im[220:470, 140:380] for _, im in cands]
    ch, cw = crops[0].shape
    zs = np.full((ch, cw * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), c in zip(cands, crops):
        p = c.copy()
        cv2.putText(p, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, 0, 2, cv2.LINE_AA)
        zs[:, x:x + cw] = p
        x += cw + 10
    cv2.imwrite(os.path.join(OUT, "_genface4_faces.png"), zs)
    print("[r72] sheets saved", flush=True)


if __name__ == "__main__":
    main()
