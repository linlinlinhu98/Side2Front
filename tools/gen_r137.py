"""R137 (user: 衣服纹理还是太杂乱 + 领子还有空白块 + 自己
看没看到问题):

Measured (_cloth_vs.png): ORIG clothes = edge density 12.8%,
mean 228, dark 11% (bright, few thin lines); OURS = 15.8%,
mean 192, dark 31% (dark, bold tangled lines). The tonal
field over-darkened and the model renders bold lines by
default.

LIGHTEN everything on genr135_s7:
- init: NO darkening field; straps seeded at 130 (not 90)
- guide: only 4 LIGHT fold lines (155 gray, 1px); NO hatch
- prompt: delicate thin lines, minimal lines, mostly white
  fabric, light pencil shading
- negatives: thick/bold/heavy/dark/dense lines
LoRA 1.0, CN 0.65, strength 0.45, DPM-26. 2 seeds.
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

PROMPT = (f"{TRIGGER}, zip-up hoodie, delicate thin "
          "lines, minimal lines, mostly white fabric, "
          "light pencil shading, soft shading, two "
          "shoulder straps, thin zipper, "
          "monochrome, pencil sketch, white "
          "background")
NEG = ("thick outlines, bold lines, heavy lines, "
       "dark lines, dense lines, many lines, "
       "chaotic lines, messy lines, drawstrings, "
       "color, colored, lowres, blurry, watermark")


def _mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[440:660, :] = 255
    cv2.ellipse(m, (CX, 440), (70, 16), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def _init(base):
    init = base.astype(np.float32).copy()
    # straps seeded at 130 (textured gray, not solid black)
    band = np.zeros_like(init)
    for xs in (140, 375):
        cv2.line(band, (xs, 445), (xs, 660), 255, 22,
                 cv2.LINE_AA)
    a = cv2.GaussianBlur(band, (0, 0), 4) / 255.0
    dark = init * (1 - a) + 130 * a
    # VERY light overall shading under the collar (kill the
    # white patch with tone, not lines)
    shade = np.zeros_like(init)
    cv2.ellipse(shade, (CX, 460), (110, 34), 0, 0, 360, 255,
                -1)
    a2 = cv2.GaussianBlur(shade, (0, 0), 14) / 255.0
    toned = init * (1 - a2) + 228 * a2
    init = np.minimum(init, np.minimum(dark, toned))
    return np.clip(init, 0, 255).astype(np.uint8)


def _guide(base):
    g = _lineart(base)
    # erase heavy lines in the chest, keep only light folds
    g[470:660, 100:420] = 255
    folds = [
        [(150, 485), (190, 545), (205, 648)],
        [(210, 478), (228, 545), (238, 648)],
        [(306, 478), (288, 545), (278, 648)],
        [(366, 485), (326, 545), (311, 648)],
    ]
    for pts in folds:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 155, 1, cv2.LINE_AA)
    # keep the 2 straps (light edges) + thin zipper
    for xs in (140, 375):
        for off in (-9, 9):
            cv2.line(g, (xs + off, 445), (xs + off, 656), 90,
                     1, cv2.LINE_AA)
    cv2.line(g, (CX, 470), (CX, 656), 90, 1, cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr135_s7.png"), 0)
    assert base is not None
    H, W = base.shape
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[280:, :])]
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    pipe = _load_dpm(None, ckpt,
                     "lllyasviel/control_v11p_sd15_lineart")
    _scale_lora(pipe.unet, 1.0)
    pipe.set_ip_adapter_scale(0.85)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    mask = _mask((H, W))
    init = _init(base)
    guide = _guide(base)
    cv2.imwrite(os.path.join(OUT, "_r137_guide.png"), guide)
    panels = [("base", base)]
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=Image.fromarray(init).convert("RGB"),
                   mask_image=Image.fromarray(mask)
                   .convert("RGB"),
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.65,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.45,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr137_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"s{seed}", g))
        print(f"[r137 {name}] {time.time() - t0:.0f}s",
              flush=True)

    n = len(panels)
    cz = [cv2.resize(im[420:660, 40:480], None, fx=1.5, fy=1.5,
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
    cv2.imwrite(os.path.join(OUT, "_genr137_cloth.png"), csheet)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr137_sheet.png"), sheet)
    print("[r137] sheets saved", flush=True)


if __name__ == "__main__":
    main()
