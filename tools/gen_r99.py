"""R99 (independent analysis on R98: eyes still round-big,
mouth still thick/sad, zipper pull lost in texture passes,
cheek blush hatching completely missing, brows uniform):

Consolidated into TWO passes (limit VAE wear) on genr98_c7_h
(the analysis's and our best):

A. FACE pass (1 VAE trip): comprehensive guide -
   - eyes LONGER (68px) and NARROWER (lid covers ~55%) with
     a 2px outer-corner upturn = long thin almond
   - thin lips, corners +2px up (not sad)
   - CHEEK BLUSH: diagonal hatch strokes drawn into the
     guide (the original's signature diagonal-hatch blush)
   - keep R92 slim jaw, R95-style tapered brows, faint nose
   prompt: long thin almond eyes, thin lips, blush
B. COLLAR pass: restore the RING PULL + zipper teeth at the
   collar center (lost in texture passes).
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
from gen_r91 import _load  # noqa: E402

FACE_PROMPT = (f"{TRIGGER}, 1boy face, front view, facing "
               "forward, looking at viewer, symmetrical "
               "face, slim oval face, narrow jaw, "
               "monochrome, pencil sketch, soft shading, "
               "long thin almond eyes, single eyelid, "
               "half-closed eyes, slightly upturned eyes, "
               "straight tapered eyebrows, detailed nose, "
               "nostrils, thin lips, neutral mouth, calm "
               "expression, blush, hatched blush, white "
               "background")
FACE_NEG = ("round face, wide face, baby face, shota, "
            "cute, big round eyes, shiny eyes, double "
            "eyelid, thick lips, sad, pout, frown, "
            "downturned mouth, dot nose, thick blocky "
            "eyebrows, 3/4 view, side view, looking "
            "away, tilted head, asymmetrical face, open "
            "mouth, smile, color, colored, photo, "
            "lowres, blurry, bad anatomy, watermark")

COLLAR_PROMPT = (f"{TRIGGER}, zip-up hoodie, metal zipper, "
                 "ring zipper pull, zipper teeth, hood "
                 "down, collar, pencil sketch, "
                 "monochrome, soft shading, white "
                 "background")
COLLAR_NEG = ("pullover, no zipper, buttons, drawstrings, "
              "color, colored, lowres, blurry, watermark")

FACE_SEEDS = [7, 42, 998, 123]
MASK_ELL = (258, 310, 116, 105)


def _face_guide(base):
    g = _lineart(base)
    y0, y1 = 240, 395
    right = g[y0:y1, CX:CX + 112]
    g[y0:y1, CX - 112:CX] = right[:, ::-1]
    g[240:290, 150:366] = 255
    g[285:395, 200:316] = 255
    # R92's slim jaw (keep)
    jaw = [(206, 292), (212, 322), (222, 350), (236, 372),
           (248, 384), (258, 388)]
    for sgn in (-1, 1):
        pts = [(CX + sgn * (CX - x), y) for x, y in jaw]
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 60, 2, cv2.LINE_AA)
    # tapered brows (approved)
    for sgn in (-1, 1):
        xs = [CX + sgn * d for d in range(28, 89, 10)]
        ys = [253, 252, 251, 251, 252, 253, 253]
        ths = [2, 2, 2, 2, 1, 1]
        for i in range(len(xs) - 1):
            cv2.line(g, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                     90, ths[i], cv2.LINE_AA)
    # LONG THIN almond eyes (68px, lid covers ~55%, 2px
    # outer upturn)
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
    # faint nose (approved R89 spec)
    cv2.ellipse(g, (CX - 7, 308), (2, 1), 0, 0, 360, 120, 1,
                cv2.LINE_AA)
    cv2.ellipse(g, (CX + 7, 308), (2, 1), 0, 0, 360, 120, 1,
                cv2.LINE_AA)
    cv2.line(g, (CX - 2, 286), (CX - 3, 300), 165, 1,
             cv2.LINE_AA)
    # thin lips, corners +2px up
    pts = [(CX - 14, 314), (CX, 316), (CX + 14, 314)]
    cv2.line(g, pts[0], pts[1], 70, 1, cv2.LINE_AA)
    cv2.line(g, pts[1], pts[2], 70, 1, cv2.LINE_AA)
    cv2.ellipse(g, (CX, 322), (9, 3), 0, 25, 155, 185, 1,
                cv2.LINE_AA)
    # CHEEK BLUSH: diagonal hatch (original's signature)
    for sgn in (-1, 1):
        for k in range(5):
            x0 = CX + sgn * (52 + k * 7)
            y0 = 296 + k * 3
            cv2.line(g, (x0, y0), (x0 - 6, y0 + 10), 150, 1,
                     cv2.LINE_AA)
    return g


def _collar_guide(base):
    g = _lineart(base)
    g[425:490, 200:316] = 255
    # RING pull + slider at the collar center
    cv2.circle(g, (CX, 452), 6, 45, 2, cv2.LINE_AA)
    cv2.rectangle(g, (CX - 3, 440), (CX + 3, 447), 45, 1)
    # zipper + teeth down the chest
    cv2.line(g, (CX, 458), (CX, 656), 55, 2, cv2.LINE_AA)
    for y in range(464, 650, 9):
        cv2.line(g, (CX - 4, y + 2), (CX + 4, y - 2), 95, 1,
                 cv2.LINE_AA)
    return g


def _collar_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[425:660, :] = 255
    cv2.ellipse(m, (CX, 425), (60, 20), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def _face_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cx, cy, ax, ay = MASK_ELL
    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 10)


def main():
    base = cv2.imread(os.path.join(OUT, "genr98_c7_h.png"), 0)
    assert base is not None
    H, W = base.shape
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[170:300, 100:320]),
            Image.fromarray(rgb[280:, :])]
    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    pipe = _load(lcm, ckpt, "lllyasviel/control_v11p_sd15_lineart")
    _scale_lora(pipe.unet, 0.8)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    # ---- stage A: comprehensive face ----
    guide = _face_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r99_guide.png"), guide)
    mask = _face_mask((H, W))
    init = base.copy()
    cx, cy, ax, ay = MASK_ELL
    cv2.ellipse(init, (cx, cy), (ax - 6, ay - 6), 0, 0, 360,
                255, -1)
    bpil = Image.fromarray(init).convert("RGB")
    mpil = Image.fromarray(mask).convert("RGB")
    ctrl = Image.fromarray(guide).convert("RGB")
    faces = []
    for seed in FACE_SEEDS:
        t0 = time.time()
        res = pipe(prompt=FACE_PROMPT, negative_prompt=FACE_NEG,
                   image=bpil, mask_image=mpil,
                   control_image=ctrl,
                   controlnet_conditioning_scale=0.95,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8,
                   guidance_scale=3.0, strength=0.85,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr99_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        faces.append((seed, g))
        print(f"[r99 {name}] {time.time() - t0:.0f}s", flush=True)

    # ---- stage B: collar zipper restore on each ----
    cmask = _collar_mask((H, W))
    panels = [("base", base)]
    for seed, fimg in faces:
        cguide = _collar_guide(fimg)
        t0 = time.time()
        res = pipe(prompt=COLLAR_PROMPT,
                   negative_prompt=COLLAR_NEG,
                   image=Image.fromarray(fimg).convert("RGB"),
                   mask_image=Image.fromarray(cmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(cguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.9,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8,
                   guidance_scale=3.0, strength=0.55,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr99_s{seed}_z.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        print(f"[r99 {name}] {time.time() - t0:.0f}s", flush=True)
        panels.append((f"s{seed}", fimg))
        panels.append((f"s{seed}z", g))

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr99_sheet.png"), sheet)
    fz = [cv2.resize(im[235:400, 130:390], None, fx=2.0, fy=2.0,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    fh, fw = fz[0].shape
    fsheet = np.full((fh, fw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, fz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        fsheet[:, x:x + fw] = t
        x += fw + 8
    cv2.imwrite(os.path.join(OUT, "_genr99_faces.png"), fsheet)
    print("[r99] sheets saved", flush=True)


if __name__ == "__main__":
    main()
