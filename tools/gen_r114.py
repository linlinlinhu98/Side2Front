"""R112 (user: R111不如base(genr110_s998); 肩宽再宽一点/
嘴唇再明显一点/头发收一点不要有乱线/仔细看原图):

Full regeneration (one VAE pass, proven > patches) of the
R110 recipe with three fixes baked into the guide/init:
1. SHOULDERS wider: side seams x75/445 (ratio ~1.9)
2. LIPS more pronounced: lip line 36px darker + bigger
   lower-lip tonal blob (15x7, 225, sigma 5) in the init
3. TIDY HAIR: a clean outer silhouette boundary drawn in
   the guide; ALL flyaway strands outside it erased;
   'tidy short hair' / 'flyaway, spiky, messy' negatives
Base for hair lineart + silhouette: genr110_s998.
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
from gen_r102 import STEPS  # noqa: E402

PROMPT = (f"{TRIGGER}, 1boy, teenage boy, thin narrow "
          "face, sharp jawline, small chin, neat smooth "
          "short hair, black hair, hair swept to the "
          "side, long almond-shaped eyes, calm narrow eyes, detailed nose, "
          "nostrils, defined lips, neutral calm "
          "expression, zip-up hoodie, broad shoulders, "
          "zipper pull tab, backpack two shoulder "
          "straps, bare neck, hand drawn, line weight "
          "variation, subtle grey shading, faint blush, "
          "monochrome, pencil sketch, white background, "
          "front view")
NEG = ("chibi, big round eyes, wide fat face, thick "
       "lips, sad pout, cloak, rolled collar, scarf, "
       "turtleneck, flyaway strands, spiky hair, messy "
       "hair, wild hair, narrow shoulders, uniform "
       "rigid outlines, light hair, colored, 3d render, "
       "deformed, disfigured, watermark")


def _guide(base):
    g = _lineart(base)
    # ---- CALM hair: wipe the whole hair zone, redraw smooth
    # strand groups (original = neat short hair, ~1 tuft) ----
    g[:245, :] = 255
    # smooth crown silhouette
    cv2.ellipse(g, (CX, 205), (165, 172), 0, 200, 340, 60, 2,
                cv2.LINE_AA)
    # organized crown spikes (reference style, not flyaways)
    for x0, y0, x1, y1 in ((240, 42, 226, 24), (272, 40, 276,
                           20), (306, 42, 322, 28)):
        cv2.line(g, (x0, y0), (x1, y1), 60, 2, cv2.LINE_AA)
    # big smooth strand groups following the head curve
    groups = [
        [(258, 42), (200, 62), (150, 105), (120, 165)],
        [(258, 42), (316, 62), (366, 105), (396, 165)],
        [(235, 48), (190, 82), (152, 135), (132, 195)],
        [(281, 48), (326, 82), (364, 135), (384, 195)],
        [(212, 56), (172, 100), (146, 158), (136, 214)],
        [(304, 56), (344, 100), (370, 158), (380, 214)],
    ]
    for pts in groups:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 80, 2, cv2.LINE_AA)
    # bangs swept to viewer's LEFT, 3 soft pointed tips
    bangs = [
        [(290, 78), (250, 105), (212, 150), (192, 200)],
        [(268, 84), (226, 112), (192, 158), (176, 208)],
        [(312, 84), (336, 120), (356, 165), (366, 210)],
    ]
    for pts in bangs:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 75, 2, cv2.LINE_AA)
    # wipe face + neck + torso, redraw to spec
    g[245:660, :] = 255
    jaw = [(206, 292), (212, 322), (222, 350), (236, 372),
           (248, 384), (258, 388)]
    for sgn in (-1, 1):
        pts = [(CX + sgn * (CX - x), y) for x, y in jaw]
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 60, 2, cv2.LINE_AA)
    for sgn in (-1, 1):
        xs = [CX + sgn * d for d in range(28, 89, 10)]
        ys = [253, 252, 251, 251, 252, 253, 253]
        ths = [2, 2, 2, 2, 1, 1]
        for i in range(len(xs) - 1):
            cv2.line(g, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                     90, ths[i], cv2.LINE_AA)
    for cx in (196, 320):
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
    # realistic nose (reference spec): bridge side lines,
    # tip, ala wings connecting, nostril dots
    cv2.line(g, (CX - 4, 288), (CX - 6, 310), 130, 1,
             cv2.LINE_AA)
    cv2.line(g, (CX + 3, 290), (CX + 4, 308), 155, 1,
             cv2.LINE_AA)
    cv2.ellipse(g, (CX, 312), (5, 4), 0, 0, 180, 110, 1,
                cv2.LINE_AA)
    for sgn in (-1, 1):
        nx = CX + sgn * 8
        cv2.ellipse(g, (nx, 315), (5, 5), 0,
                    290 if sgn < 0 else 70,
                    60 if sgn < 0 else 230, 95, 1, cv2.LINE_AA)
        cv2.ellipse(g, (nx, 318), (2, 1), 0, 0, 360, 90, 2,
                    cv2.LINE_AA)
    # DEFINED lips (reference spec): dark level line, wider
    cv2.line(g, (CX - 18, 316), (CX + 18, 316), 50, 2,
             cv2.LINE_AA)
    cv2.ellipse(g, (CX, 324), (12, 5), 0, 25, 155, 165, 1,
                cv2.LINE_AA)
    for sgn in (-1, 1):
        for k in range(5):
            x0 = CX + sgn * (52 + k * 7)
            y0 = 296 + k * 3
            cv2.line(g, (x0, y0), (x0 - 6, y0 + 10), 150, 1,
                     cv2.LINE_AA)
    # detailed ears (reference spec): outer C + inner Y
    for sgn in (-1, 1):
        ex = CX + sgn * 88
        cv2.ellipse(g, (ex, 292), (10, 16), 0, 300, 160, 90,
                    2, cv2.LINE_AA)
        cv2.ellipse(g, (ex + sgn * 1, 294), (4, 8), 0, 280,
                    90, 120, 1, cv2.LINE_AA)
        cv2.line(g, (ex - sgn * 2, 296), (ex + sgn * 3, 301),
                 130, 1, cv2.LINE_AA)
    # neck
    cv2.line(g, (CX - 26, 390), (CX - 28, 436), 85, 2,
             cv2.LINE_AA)
    cv2.line(g, (CX + 26, 390), (CX + 28, 436), 85, 2,
             cv2.LINE_AA)
    # open collar V + zipper
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
    cv2.ellipse(g, (CX, 424), (84, 18), 0, 200, 340, 90, 1,
                cv2.LINE_AA)
    # WIDER side seams + straps
    cv2.line(g, (CX - 28, 436), (70, 462), 80, 2, cv2.LINE_AA)
    cv2.line(g, (CX + 28, 436), (446, 462), 80, 2, cv2.LINE_AA)
    cv2.line(g, (70, 462), (63, 560), 70, 2, cv2.LINE_AA)
    cv2.line(g, (63, 560), (58, 656), 70, 2, cv2.LINE_AA)
    cv2.line(g, (446, 462), (453, 560), 70, 2, cv2.LINE_AA)
    cv2.line(g, (453, 560), (458, 656), 70, 2, cv2.LINE_AA)
    for sgn in (-1, 1):
        xs = CX + sgn * 145
        for off in (-9, 9):
            cv2.line(g, (xs + off, 458),
                     (xs + sgn * 18 + off, 656), 55, 2,
                     cv2.LINE_AA)
        cv2.rectangle(g, (xs + sgn * 8 - 6, 516),
                      (xs + sgn * 8 + 6, 525), 55, 1)
    folds = [
        [(160, 490), (190, 548), (206, 648)],
        [(356, 490), (326, 548), (310, 648)],
        [(233, 498), (229, 558), (227, 648)],
        [(283, 498), (287, 558), (289, 648)],
    ]
    for pts in folds:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 125, 1, cv2.LINE_AA)
    return g


def _init(base):
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
    ys = np.arange(390, 442, dtype=np.float32)[:, None]
    grad = 246 - (ys - 390) / 52 * 8
    init[390:442, 224:292] = np.clip(grad, 0, 255)
    # bigger lower-lip tonal blob
    blob = np.zeros_like(init)
    cv2.ellipse(blob, (CX, 324), (15, 7), 0, 0, 360, 255, -1)
    a2 = cv2.GaussianBlur(blob, (0, 0), 5) \
        .astype(np.float32) / 255
    init = (init * (1 - a2) + 225 * a2).astype(np.uint8)
    return init


def main():
    base = cv2.imread(os.path.join(OUT, "genr112_s998.png"),
                      0)
    assert base is not None
    H, W = base.shape
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refc = cv2.imread(os.path.join(
        ROOT, "result", "8aa591a6c9e652b79697d55eeee769ec.jpg"),
        cv2.IMREAD_COLOR)
    refrgb = cv2.cvtColor(refc, cv2.COLOR_BGR2RGB)
    # user mandate: 头发/嘴唇/鼻子/眼睛/耳朵可以模仿这张图
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(refrgb)]
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    pipe = _load_dpm(None, ckpt,
                     "lllyasviel/control_v11p_sd15_lineart")
    _scale_lora(pipe.unet, 0.9)
    pipe.set_ip_adapter_scale(0.85)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    guide = _guide(base)
    cv2.imwrite(os.path.join(OUT, "_r114_guide.png"), guide)
    init = _init(base)
    full_mask = Image.fromarray(
        np.full((H, W), 255, np.uint8)).convert("RGB")
    panels = [("guide", guide)]
    for seed in (555, 777, 2024, 3141, 666, 888):
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
        name = f"genr114_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"s{seed}", g))
        print(f"[r112 {name}] {time.time() - t0:.0f}s",
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
    cv2.imwrite(os.path.join(OUT, "_genr114_sheet.png"), sheet)
    print("[r112] sheets saved", flush=True)


if __name__ == "__main__":
    main()
