"""R105 (user: 领子哪里降了/画风越来越诡异/仔细看衣服所有
部分):

Full-image truth (_now_vs_orig.png): the collar renders as a
bulky white SCARF-ROLL around the neck (not a hood resting
on shoulders); the torso is a giant cloak; straps weak;
hair halo. Patching made it worse - one STRUCTURAL redraw:

Guide (full canvas): keep face/hair lineart above y400;
whiten y400-660 and redraw the WHOLE lower structure:
- SLIM side seams (hem x~115/400, was ~60/460)
- hood edges: two diagonal lines from the neck base OUT to
  the shoulders (the real hood-down shape), NO front roll
- low flat hood mound BEHIND the neck (peeks 15px)
- zipper + ring pull at y470
- backpack straps
- light fold lines
Init: y400-660 whitened (kill the scarf roll). CN 1.0,
DPM-26, LoRA 1.0, IP [full+torso] @0.9, strength 0.78.
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

PROMPT = (f"{TRIGGER}, 1boy, zip-up hoodie, slim fit, "
          "hood down resting on shoulders, metal zipper, "
          "ring zipper pull, backpack straps, fabric "
          "folds, subtle grey shading, monochrome, pencil "
          "sketch, hand drawn, soft shading, white "
          "background")
NEG = ("cloak, cape, oversized, scarf, turtleneck, high "
       "collar, drawstrings, pullover, color, colored, "
       "photo, lowres, blurry, watermark")


def _guide(base):
    g = _lineart(base)
    g[398:660, :] = 255
    # SLIM side seams (shoulder ~x150/365 at y455 -> hem
    # x~115/400 at y656)
    cv2.line(g, (150, 455), (115, 656), 70, 2, cv2.LINE_AA)
    cv2.line(g, (365, 455), (400, 656), 70, 2, cv2.LINE_AA)
    # neck base -> shoulders: two diagonal hood edges
    cv2.line(g, (CX - 26, 440), (150, 470), 75, 2, cv2.LINE_AA)
    cv2.line(g, (CX + 26, 440), (365, 470), 75, 2, cv2.LINE_AA)
    # low flat hood mound behind the neck (peeks ~15px)
    cv2.ellipse(g, (CX, 428), (82, 20), 0, 195, 345, 85, 2,
                cv2.LINE_AA)
    # V opening + ring pull + zipper
    cv2.line(g, (CX - 24, 444), (CX - 5, 466), 70, 2,
             cv2.LINE_AA)
    cv2.line(g, (CX + 24, 444), (CX + 5, 466), 70, 2,
             cv2.LINE_AA)
    cv2.circle(g, (CX, 472), 6, 45, 2, cv2.LINE_AA)
    cv2.rectangle(g, (CX - 3, 460), (CX + 3, 467), 45, 1)
    cv2.line(g, (CX, 478), (CX, 656), 55, 2, cv2.LINE_AA)
    for y in range(484, 650, 9):
        cv2.line(g, (CX - 4, y + 2), (CX + 4, y - 2), 95, 1,
                 cv2.LINE_AA)
    # backpack straps (inside the side seams)
    for sgn in (-1, 1):
        xs = CX + sgn * 96
        for off in (-9, 9):
            cv2.line(g, (xs + off, 452),
                     (xs + sgn * 36 + off, 656), 55, 2,
                     cv2.LINE_AA)
        cv2.rectangle(g, (xs + sgn * 12 - 6, 508),
                      (xs + sgn * 12 + 6, 517), 55, 1)
    # light folds
    folds = [
        [(185, 480), (205, 540), (218, 648)],
        [(330, 480), (310, 540), (297, 648)],
        [(233, 490), (229, 556), (227, 648)],
        [(283, 490), (287, 556), (289, 648)],
    ]
    for pts in folds:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 125, 1, cv2.LINE_AA)
    return g


def _mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[398:, :] = 255
    cv2.ellipse(m, (CX, 398), (64, 18), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def main():
    base = cv2.imread(os.path.join(OUT, "genr104_c7_m.png"),
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
    pipe.set_ip_adapter_scale(0.9)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    guide = _guide(base)
    cv2.imwrite(os.path.join(OUT, "_r105_guide.png"), guide)
    mask = _mask((H, W))
    init = base.copy()
    init[398:, :] = 255
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
                   guidance_scale=3.5, strength=0.78,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr105_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"s{seed}", g))
        print(f"[r105 {name}] {time.time() - t0:.0f}s",
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
    cv2.imwrite(os.path.join(OUT, "_genr105_sheet.png"), sheet)
    cz = [cv2.resize(im[390:660, 60:460], None, fx=1.5, fy=1.5,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    ch, cw = cz[0].shape
    csheet = np.full((ch, cw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, cz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        csheet[:, x:x + cw] = t
        x += cw + 8
    cv2.imwrite(os.path.join(OUT, "_genr105_clothes.png"),
                csheet)
    print("[r105] sheets saved", flush=True)


if __name__ == "__main__":
    main()
