"""R87 (user: 衣服不还是没什么变化吗/你有没有仔细看原图):

Tonal proof: ORIG hoodie body mean=224 std=53 dark=14-16%;
OURS mean=252 std=24 dark=1-2%. The original's fabric has
broad GRAY TONE + dense hatching over the whole surface; ours
is white paper with outlines. Hatch BANDS on white were
decoration - the body itself needs tone.

R87 (torso only, masked):
1. TONAL INIT: extract the ORIGINAL torso's low-frequency
   tonal field (heavy blur), symmetrize it, blend 55% into
   the init - the denoise starts from real fabric tone.
2. DENSE GUIDE: full fold pattern (armpit sweeps, collar
   folds, sleeve creases, hem arcs) + broad hatch AREAS
   (under hood mass, side body, fold valleys, sleeve
   undersides, spacing 7px) + bold dark accents at the main
   fold edges (the original's style).
3. strength 0.62, LoRA 1.0, CN 0.85, IP torso 1.0,
   + 'tonal shading, gray fabric' prompt.

Base: genr86_m42_cloth (user-approved face). 2 seeds.
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
from gen_r83 import LORA_DIR, TRIGGER, CX, _scale_lora  # noqa: E402
from gen_r84 import _hatch  # noqa: E402

CLOTH_PROMPT = (f"{TRIGGER}, zip-up hoodie, full-length "
                "zipper, ring zipper pull, hood down, dense "
                "parallel hatching, fine pencil hatching, "
                "tonal shading, gray fabric, fabric folds, "
                "sketch texture, monochrome, pencil sketch, "
                "white background")
CLOTH_NEG = ("smooth, clean, digital painting, flat, pure "
             "white fabric, drawstrings, pullover, thick "
             "outlines, cel shading, color, colored, photo, "
             "lowres, blurry, watermark")

SEEDS = [7, 42]
STRENGTH = 0.62
CN_SCALE = 0.85
NECK_Y = 410


def _torso_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[NECK_Y:, :] = 255
    cv2.ellipse(m, (CX, NECK_Y), (60, 26), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def _tonal_init(base, orig):
    """blend the original torso's low-freq tonal field
    (symmetrized) into the init at 55%"""
    tone = orig[420:660, :].astype(np.float32)
    tone = cv2.GaussianBlur(tone, (0, 0), 22)
    tone = np.minimum(tone, tone[:, ::-1])      # symmetrize
    init = base.astype(np.float32).copy()
    region = init[420:660, :]
    target = 255.0 - (255.0 - tone) * 0.55
    region = np.minimum(region, target)
    init[420:660, :] = region
    return init.astype(np.uint8)


def _cloth_guide(base):
    g = _lineart(base)
    # ---- fold line pattern (original's: long sweeps) ----
    folds = [
        # armpit sweeps down-in to hem
        [(150, 470), (170, 520), (185, 575), (195, 645)],
        [(370, 470), (350, 520), (335, 575), (325, 645)],
        [(175, 460), (200, 515), (218, 580), (228, 645)],
        [(345, 460), (320, 515), (302, 580), (292, 645)],
        # center fold beside zipper
        [(228, 460), (224, 540), (222, 645)],
        [(288, 460), (292, 540), (294, 645)],
    ]
    for pts in folds:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 115, 1, cv2.LINE_AA)
    # sleeve creases
    for x0, y0, x1, y1 in ((110, 520, 145, 545), (100, 560, 138,
                           585), (410, 520, 375, 545),
                           (420, 560, 382, 585)):
        cv2.line(g, (x0, y0), (x1, y1), 120, 1, cv2.LINE_AA)
    # hem fold arcs
    for cx in (170, 258, 346):
        cv2.ellipse(g, (cx, 660), (46, 26), 0, 200, 340, 120, 1,
                    cv2.LINE_AA)
    # bold dark accents at the main fold edges (orig style)
    for pts in ([(150, 470), (170, 520), (185, 575)],
                [(370, 470), (350, 520), (335, 575)]):
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 60, 2, cv2.LINE_AA)
    # ---- broad hatch AREAS ----
    zones = [
        [(150, 415), (366, 415), (360, 460), (156, 460)],   # under hood mass
        [(88, 470), (150, 470), (138, 650), (76, 650)],     # left side body
        [(366, 470), (428, 470), (440, 650), (378, 650)],   # right side body
        [(196, 540), (224, 540), (220, 645), (192, 645)],   # left fold valley
        [(292, 540), (320, 540), (324, 645), (296, 645)],   # right fold valley
        [(84, 505), (116, 500), (106, 590), (74, 596)],     # left sleeve under
        [(400, 500), (432, 505), (442, 596), (410, 590)],   # right sleeve under
    ]
    for poly in zones:
        m = np.zeros_like(g)
        cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
        tmp = np.full_like(g, 255)
        _hatch(tmp, poly, spacing=7, val=130)
        g[m > 0] = tmp[m > 0]
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr86_m42_cloth.png"),
                      0)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"), 0)
    assert base is not None and orig is not None
    H, W = base.shape
    init = _tonal_init(base, orig)
    guide = _cloth_guide(base)
    mask = _torso_mask((H, W))
    cv2.imwrite(os.path.join(OUT, "_r87_init.png"), init)
    cv2.imwrite(os.path.join(OUT, "_r87_guide.png"), guide)

    origc = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                       cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(origc, cv2.COLOR_BGR2RGB)
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
    pipe.set_progress_bar_config(disable=True)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(1.0)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    cands = [("base", base)]
    for seed in SEEDS:
        t0 = time.time()
        res = pipe(prompt=CLOTH_PROMPT, negative_prompt=CLOTH_NEG,
                   image=Image.fromarray(init).convert("RGB"),
                   mask_image=Image.fromarray(mask).convert("RGB"),
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=CN_SCALE,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8, guidance_scale=3.0,
                   strength=STRENGTH,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr87_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cands.append((f"s{seed}", g))
        print(f"[r87 {name}] {time.time() - t0:.0f}s", flush=True)

    n = len(cands)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in cands:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr87_sheet.png"), sheet)
    cz = [cv2.resize(im[415:660, 50:470], None, fx=1.5, fy=1.5,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in cands]
    ch, cw = cz[0].shape
    csheet = np.full((ch, cw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(cands, cz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        csheet[:, x:x + cw] = t
        x += cw + 8
    cv2.imwrite(os.path.join(OUT, "_genr87_cloths.png"), csheet)
    print("[r87] sheets saved", flush=True)


if __name__ == "__main__":
    main()
