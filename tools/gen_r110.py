"""R110: FULL clean regeneration (user: patches hit the
ceiling; target = the AI-reference level in ONE consistent
hand).

Everything learned in 30 rounds baked into ONE guide + ONE
generation (single VAE pass, unified style):
- guide: current best image's lineart for the HAIR only
  (keep the approved black-hair strokes); face at R92's
  measured proportions; neck with side lines + trapezius;
  open collar V (ZERO hood lines at the neck front); ring
  pull + zipper teeth; both straps; slim side seams; light
  folds
- init: white canvas + hair-region dark blob (x0.4 of the
  current hair silhouette) + subtle neck skin shade ->
  black hair and skin neck FORCED by seeding, not prayer
- report's verbatim style prompt; DPM-26, guidance 3.5,
  LoRA 0.9, IP [full+head+torso] @0.7, strength 0.85,
  CN 1.0
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
from gen_r83 import LORA_DIR, TRIGGER, CX, _scale_lora  # noqa: E402
from gen_r101 import _load_dpm  # noqa: E402
from gen_r102 import STEPS, UNIFY_PROMPT, UNIFY_NEG  # noqa: E402

PROMPT = (f"{TRIGGER}, 1boy, teenage boy, thin narrow "
          "face, sharp jawline, small chin, short messy "
          "spiky dark hair, black hair, long almond-shaped "
          "eyes, thin lips, neutral calm expression, "
          "zip-up hoodie, zipper pull tab, backpack two "
          "shoulder straps, bare neck, hand drawn, line "
          "weight variation, subtle grey shading, faint "
          "blush, monochrome, pencil sketch, white "
          "background, front view")
NEG = ("chibi, big round eyes, wide fat face, thick "
       "lips, sad pout, cloak, rolled collar, scarf, "
       "turtleneck, uniform rigid outlines, repetitive "
       "parallel hatching inside hair, heavy black "
       "fill, light hair, colored, 3d render, "
       "deformed, disfigured, watermark")


def _guide(base):
    """keep current HAIR lineart; redraw everything else to
    spec"""
    g = _lineart(base)
    # wipe face + neck + torso, keep hair (y<245 region)
    g[245:660, :] = 255
    # also wipe the inner-face bang strands overlapping the
    # brow zone? keep them (hair identity)
    # ---- face (R92 measured) ----
    jaw = [(206, 292), (212, 322), (222, 350), (236, 372),
           (248, 384), (258, 388)]
    for sgn in (-1, 1):
        pts = [(CX + sgn * (CX - x), y) for x, y in jaw]
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 60, 2, cv2.LINE_AA)
    for sgn in (-1, 1):                     # brows
        xs = [CX + sgn * d for d in range(28, 89, 10)]
        ys = [253, 252, 251, 251, 252, 253, 253]
        ths = [2, 2, 2, 2, 1, 1]
        for i in range(len(xs) - 1):
            cv2.line(g, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                     90, ths[i], cv2.LINE_AA)
    for cx in (196, 320):                   # long narrow eyes
        cv2.circle(g, (cx, 276), 5, 100, -1, cv2.LINE_AA)
        cv2.rectangle(g, (cx - 35, 255), (cx + 35, 271), 255,
                      -1)
        up = [(cx - 34, 273), (cx - 14, 270), (cx + 14, 270),
              (cx + 34, 273)]
        for i in range(3):
            cv2.line(g, up[i], up[i + 1], 45, 3, cv2.LINE_AA)
        sgn = -1 if cx < CX else 1
        cv2.line(g, (cx + sgn * 34, 273),
                 (cx + sgn * 39, 271), 45, 2, cv2.LINE_AA)
        lo = [(cx - 34, 273), (cx - 14, 279), (cx + 14, 279),
              (cx + 34, 273)]
        for i in range(3):
            cv2.line(g, lo[i], lo[i + 1], 95, 1, cv2.LINE_AA)
    cv2.ellipse(g, (CX - 7, 308), (2, 1), 0, 0, 360, 120, 1,
                cv2.LINE_AA)
    cv2.ellipse(g, (CX + 7, 308), (2, 1), 0, 0, 360, 120, 1,
                cv2.LINE_AA)
    cv2.line(g, (CX - 2, 286), (CX - 3, 300), 165, 1,
             cv2.LINE_AA)
    cv2.line(g, (CX - 14, 316), (CX + 14, 316), 65, 1,
             cv2.LINE_AA)
    cv2.ellipse(g, (CX, 323), (10, 4), 0, 25, 155, 175, 1,
                cv2.LINE_AA)
    for sgn in (-1, 1):                     # blush hatch
        for k in range(5):
            x0 = CX + sgn * (52 + k * 7)
            y0 = 296 + k * 3
            cv2.line(g, (x0, y0), (x0 - 6, y0 + 10), 150, 1,
                     cv2.LINE_AA)
    for sgn in (-1, 1):                     # ears
        ex = CX + sgn * 88
        cv2.ellipse(g, (ex, 292), (9, 14), 0, 310, 150, 105,
                    1, cv2.LINE_AA)
    # ---- neck + trapezius ----
    cv2.line(g, (CX - 26, 390), (CX - 28, 436), 85, 2,
             cv2.LINE_AA)
    cv2.line(g, (CX + 26, 390), (CX + 28, 436), 85, 2,
             cv2.LINE_AA)
    cv2.line(g, (CX - 28, 436), (158, 466), 110, 1,
             cv2.LINE_AA)
    cv2.line(g, (CX + 28, 436), (358, 466), 110, 1,
             cv2.LINE_AA)
    # ---- open collar V + zipper ----
    cv2.line(g, (CX - 28, 436), (CX - 6, 464), 70, 2,
             cv2.LINE_AA)
    cv2.line(g, (CX + 28, 436), (CX + 6, 464), 70, 2,
             cv2.LINE_AA)
    cv2.circle(g, (CX, 470), 6, 45, 2, cv2.LINE_AA)
    cv2.rectangle(g, (CX - 3, 458), (CX + 3, 465), 45, 1)
    cv2.line(g, (CX, 476), (CX, 656), 55, 2, cv2.LINE_AA)
    for y in range(482, 650, 9):
        cv2.line(g, (CX - 4, y + 2), (CX + 4, y - 2), 95, 1,
                 cv2.LINE_AA)
    # hood mass ONLY behind the nape (small arc, outside the
    # neck front zone)
    cv2.ellipse(g, (CX, 424), (84, 18), 0, 200, 340, 90, 1,
                cv2.LINE_AA)
    # ---- WIDE side seams + straps (shoulder:head ~1.9) ----
    cv2.line(g, (CX - 28, 436), (110, 462), 80, 2, cv2.LINE_AA)
    cv2.line(g, (CX + 28, 436), (410, 462), 80, 2, cv2.LINE_AA)
    cv2.line(g, (110, 462), (95, 560), 70, 2, cv2.LINE_AA)
    cv2.line(g, (95, 560), (85, 656), 70, 2, cv2.LINE_AA)
    cv2.line(g, (410, 462), (425, 560), 70, 2, cv2.LINE_AA)
    cv2.line(g, (425, 560), (435, 656), 70, 2, cv2.LINE_AA)
    for sgn in (-1, 1):
        xs = CX + sgn * 128
        for off in (-9, 9):
            cv2.line(g, (xs + off, 458),
                     (xs + sgn * 20 + off, 656), 55, 2,
                     cv2.LINE_AA)
        cv2.rectangle(g, (xs + sgn * 8 - 6, 516),
                      (xs + sgn * 8 + 6, 525), 55, 1)
    folds = [
        [(185, 490), (205, 545), (218, 648)],
        [(330, 490), (310, 545), (297, 648)],
        [(233, 498), (229, 558), (227, 648)],
        [(283, 498), (287, 558), (289, 648)],
    ]
    for pts in folds:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 125, 1, cv2.LINE_AA)
    return g


def _init(base):
    """white canvas + hair dark blob + neck skin shade"""
    H, W = base.shape
    init = np.full((H, W), 255, np.uint8)
    hair_dark = np.zeros((H, W), np.uint8)
    hair_dark[:245, :] = (base[:245, :] < 170) \
        .astype(np.uint8) * 255
    hair_dark = cv2.dilate(hair_dark, np.ones((5, 5), np.uint8),
                           iterations=2)
    a = cv2.GaussianBlur(hair_dark, (0, 0), 6) \
        .astype(np.float32) / 255
    init = (init * (1 - a) + 255 * 0.4 * a).astype(np.uint8)
    # neck skin
    ys = np.arange(390, 442, dtype=np.float32)[:, None]
    grad = 246 - (ys - 390) / 52 * 8
    init[390:442, 224:292] = np.clip(grad, 0, 255)
    return init


def main():
    base = cv2.imread(os.path.join(OUT, "genr108_s123.png"), 0)
    assert base is not None
    H, W = base.shape
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[280:, :])]
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    pipe = _load_dpm(None, ckpt,
                     "lllyasviel/control_v11p_sd15_lineart")
    _scale_lora(pipe.unet, 0.9)
    pipe.set_ip_adapter_scale(0.7)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    guide = _guide(base)
    cv2.imwrite(os.path.join(OUT, "_r110_guide.png"), guide)
    init = _init(base)
    cv2.imwrite(os.path.join(OUT, "_r110_init.png"), init)
    full_mask = Image.fromarray(
        np.full((H, W), 255, np.uint8)).convert("RGB")
    panels = [("guide", guide)]
    for seed in (555, 777, 2024, 3141):
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=Image.fromarray(init).convert("RGB"),
                   mask_image=full_mask,
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.85,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr110_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"s{seed}", g))
        print(f"[r108 {name}] {time.time() - t0:.0f}s",
              flush=True)

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr110_sheet.png"), sheet)
    print("[r108] sheets saved", flush=True)


if __name__ == "__main__":
    main()
