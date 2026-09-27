"""R135 (fix #4 of 4: 背包带子有3条以上没有擦干净):

Current (genr134_s42): 1 diagonal cross-body band + 1 left
vertical strap + a squiggle at the left collar = reads as
3+. Target: EXACTLY 2 vertical shoulder straps.
- whiten the diagonal band AND the left-collar squiggle in
  init
- guide: two vertical straps (x140/x375) with dark seeds
  (value 100) + thin zipper; nothing else
DPM-26, CN 1.0, strength 0.70, LoRA 0.8. 2 seeds.
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

PROMPT = (f"{TRIGGER}, zip-up hoodie, two vertical "
          "backpack shoulder straps, thin zipper, "
          "clean fabric, fabric folds, monochrome, "
          "pencil sketch, soft shading, white "
          "background")
NEG = ("diagonal strap, crossbody, extra straps, "
       "many straps, drawstrings, squiggle, color, "
       "colored, lowres, blurry, watermark")


def _torso_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[425:, :] = 255
    cv2.ellipse(m, (CX, 425), (70, 16), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def _init(base):
    init = base.astype(np.float32).copy()
    # whiten the diagonal band
    band = np.zeros_like(init)
    cv2.line(band, (350, 445), (175, 660), 255, 80,
             cv2.LINE_AA)
    band = cv2.GaussianBlur(band, (0, 0), 5)
    init[band > 0] = 255
    # whiten the left-collar squiggle
    init[410:470, 165:230] = 255
    # dark seeds for the two straps
    band2 = np.zeros_like(init)
    for xs in (140, 375):
        cv2.line(band2, (xs, 445), (xs, 660), 255, 22,
                 cv2.LINE_AA)
    a = cv2.GaussianBlur(band2, (0, 0), 4) / 255.0
    dark = init * (1 - a) + 100 * a
    return np.minimum(init, dark).astype(np.uint8)


def _guide(base):
    g = _lineart(base)
    band = np.zeros_like(g)
    cv2.line(band, (350, 445), (175, 660), 255, 80,
             cv2.LINE_AA)
    band = cv2.GaussianBlur(band, (0, 0), 5)
    g[band > 0] = 255
    g[410:470, 165:230] = 255
    # exactly TWO vertical straps
    for xs in (140, 375):
        for off in (-9, 9):
            cv2.line(g, (xs + off, 445), (xs + off, 656), 50,
                     2, cv2.LINE_AA)
        cv2.rectangle(g, (xs - 6, 520), (xs + 6, 529), 50, 1)
    # thin zipper
    cv2.line(g, (CX, 470), (CX, 656), 60, 2, cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr134_s42.png"), 0)
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
    _scale_lora(pipe.unet, 0.8)
    pipe.set_ip_adapter_scale(0.8)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    mask = _torso_mask((H, W))
    init = _init(base)
    guide = _guide(base)
    cv2.imwrite(os.path.join(OUT, "_r135_guide.png"), guide)
    panels = [("base", base)]
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=Image.fromarray(init).convert("RGB"),
                   mask_image=Image.fromarray(mask)
                   .convert("RGB"),
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.70,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr135_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"s{seed}", g))
        print(f"[r135 {name}] {time.time() - t0:.0f}s",
              flush=True)

    n = len(panels)
    cz = [cv2.resize(im[420:660, 30:490], None, fx=1.6, fy=1.6,
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
    cv2.imwrite(os.path.join(OUT, "_genr135_straps.png"), csheet)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr135_sheet.png"), sheet)
    print("[r135] sheets saved", flush=True)


if __name__ == "__main__":
    main()
