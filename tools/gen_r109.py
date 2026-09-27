"""R109 (user: 注意原图宽高比例/肩宽要合适/现在脸大身子小):

Measured: references' shoulder:head ratio ~2.1-2.3; ours
(R108 slimmed torso) = 1.45 => big head small body.

Torso-WIDENING pass on genr108_s123 (face untouched):
- guide: keep everything above y400; redraw the torso with
  side seams x95/425 at the shoulder (y455) tapering to
  x85/435 at the hem => shoulder ~330px wider -> ratio ~1.9
- straps move out to x135/385; zipper/folds stay
- init: y400-660 whitened; CN 1.0, DPM-26, strength 0.80,
  LoRA 1.0, IP [full+torso] @0.8, 'broad shoulders' prompt
3 seeds.
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

PROMPT = (f"{TRIGGER}, 1boy, zip-up hoodie, broad "
          "shoulders, teenage boy, hood down, metal "
          "zipper, ring zipper pull, backpack straps, "
          "fabric folds, subtle grey shading, "
          "monochrome, pencil sketch, hand drawn, white "
          "background")
NEG = ("narrow shoulders, child body, cloak, cape, "
       "scarf, turtleneck, drawstrings, pullover, "
       "color, colored, photo, lowres, blurry, "
       "watermark")


def _guide(base):
    g = _lineart(base)
    g[400:660, :] = 255
    # neck + collar V (keep position from the R108 spec)
    cv2.line(g, (CX - 26, 394), (CX - 28, 436), 85, 2,
             cv2.LINE_AA)
    cv2.line(g, (CX + 26, 394), (CX + 28, 436), 85, 2,
             cv2.LINE_AA)
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
    # hood edges out to WIDE shoulders
    cv2.line(g, (CX - 28, 436), (110, 462), 80, 2, cv2.LINE_AA)
    cv2.line(g, (CX + 28, 436), (410, 462), 80, 2, cv2.LINE_AA)
    # WIDE side seams: shoulder x95/425 -> hem x85/435
    cv2.line(g, (110, 462), (95, 560), 70, 2, cv2.LINE_AA)
    cv2.line(g, (95, 560), (85, 656), 70, 2, cv2.LINE_AA)
    cv2.line(g, (410, 462), (425, 560), 70, 2, cv2.LINE_AA)
    cv2.line(g, (425, 560), (435, 656), 70, 2, cv2.LINE_AA)
    # straps moved out
    for sgn in (-1, 1):
        xs = CX + sgn * 128
        for off in (-9, 9):
            cv2.line(g, (xs + off, 458),
                     (xs + sgn * 20 + off, 656), 55, 2,
                     cv2.LINE_AA)
        cv2.rectangle(g, (xs + sgn * 8 - 6, 516),
                      (xs + sgn * 8 + 6, 525), 55, 1)
    # light folds
    folds = [
        [(170, 490), (195, 548), (210, 648)],
        [(345, 490), (320, 548), (305, 648)],
        [(233, 498), (229, 558), (227, 648)],
        [(283, 498), (287, 558), (289, 648)],
    ]
    for pts in folds:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 125, 1, cv2.LINE_AA)
    return g


def _mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[400:, :] = 255
    cv2.ellipse(m, (CX, 400), (64, 18), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def main():
    base = cv2.imread(os.path.join(OUT, "genr108_s123.png"),
                      0)
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
    _scale_lora(pipe.unet, 1.0)
    pipe.set_ip_adapter_scale(0.8)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    guide = _guide(base)
    cv2.imwrite(os.path.join(OUT, "_r109_guide.png"), guide)
    mask = _mask((H, W))
    init = base.copy()
    init[400:, :] = 255
    panels = [("base", base)]
    for seed in (7, 42, 998):
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=Image.fromarray(init).convert("RGB"),
                   mask_image=Image.fromarray(mask).convert("RGB"),
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.80,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr109_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"s{seed}", g))
        print(f"[r109 {name}] {time.time() - t0:.0f}s",
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
    cv2.imwrite(os.path.join(OUT, "_genr109_sheet.png"), sheet)
    print("[r109] sheets saved", flush=True)


if __name__ == "__main__":
    main()
