"""R107 (user: 脖子哪里露出了, 就这么简单的一个点你要修改
多少次):

Neck truth (_neck_truth.png): the 40px gap below the chin
IS there (spec-correct) but it's PURE WHITE - no neck side
lines, no under-jaw shadow, no skin tone => reads as a
floating head, not a neck.

Definitive neck repaint (small zone y388-452):
- guide: TWO neck side lines (jaw -> collar V) + collar V
  + ring pull below (kept)
- init: whitened + subtle skin tone (light vertical
  gradient 246->238, slightly darker under the jaw) so the
  zone reads as SKIN not paper
- prompt: neck, bare neck, under-jaw shadow, skin
DPM-26, CN 0.9, strength 0.70, LoRA 0.6, 2 seeds.
Base: genr106_s998.
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

NECK_PROMPT = (f"{TRIGGER}, neck, bare neck, under-jaw "
               "shadow, skin, skin shading, pencil sketch, "
               "soft shading, monochrome, white background")
NECK_NEG = ("collar, turtleneck, scarf, high collar, "
            "color, colored, lowres, blurry, watermark")


def _neck_guide(base):
    g = _lineart(base)
    g[388:452, 180:336] = 255
    # two neck side lines, jaw -> collar V
    cv2.line(g, (CX - 26, 390), (CX - 28, 436), 85, 2,
             cv2.LINE_AA)
    cv2.line(g, (CX + 26, 390), (CX + 28, 436), 85, 2,
             cv2.LINE_AA)
    # collar V edges + pull
    cv2.line(g, (CX - 28, 436), (CX - 6, 462), 70, 2,
             cv2.LINE_AA)
    cv2.line(g, (CX + 28, 436), (CX + 6, 462), 70, 2,
             cv2.LINE_AA)
    cv2.circle(g, (CX, 468), 6, 45, 2, cv2.LINE_AA)
    return g


def _neck_init(base):
    init = base.copy()
    init[388:452, 180:336] = 255
    # subtle skin gradient + under-jaw shadow
    ys = np.arange(388, 452, dtype=np.float32)[:, None]
    grad = 246 - (ys - 388) / 64 * 8          # 246 -> 238
    init[388:452, 180:336] = np.clip(grad, 0, 255)
    shadow = np.zeros_like(init)
    cv2.ellipse(shadow, (CX, 396), (30, 8), 0, 0, 360, 255,
                -1)
    a = cv2.GaussianBlur(shadow, (0, 0), 4) \
        .astype(np.float32) / 255
    init = (init * (1 - a) + 226 * a).astype(np.uint8)
    return init


def _neck_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CX, 420), (80, 36), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def main():
    base = cv2.imread(os.path.join(OUT, "genr106_s998.png"),
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
    _scale_lora(pipe.unet, 0.6)
    pipe.set_ip_adapter_scale(0.8)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    guide = _neck_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r107_nguide.png"), guide)
    init = _neck_init(base)
    cv2.imwrite(os.path.join(OUT, "_r107_ninit.png"), init)
    mask = _neck_mask((H, W))
    panels = [("base", base)]
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=NECK_PROMPT, negative_prompt=NECK_NEG,
                   image=Image.fromarray(init).convert("RGB"),
                   mask_image=Image.fromarray(mask).convert("RGB"),
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.9,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.70,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr107_n{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"n{seed}", g))
        print(f"[r107 {name}] {time.time() - t0:.0f}s",
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
    cv2.imwrite(os.path.join(OUT, "_genr107_sheet.png"), sheet)
    nz = [cv2.resize(im[370:520, 120:400], None, fx=2.6, fy=2.6,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    nh, nw = nz[0].shape
    nsheet = np.full((nh, nw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, nz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        nsheet[:, x:x + nw] = t
        x += nw + 8
    cv2.imwrite(os.path.join(OUT, "_genr107_necks.png"), nsheet)
    print("[r107] sheets saved", flush=True)


if __name__ == "__main__":
    main()
