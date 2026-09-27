"""R138 (final cloth polish; R137 self-check: interior
lighter but straps render SOLID BLACK and 6-8 crossing bold
diagonals = the 'messy' the user sees; original = white
fabric + textured gray strap + ~2 thin folds):

- init: whiten the chest interior (x120-400, y470-660)
  EXCEPT the 2 straps + zipper column - ALL old crossing
  lines die
- guide: ONLY light side seams + 2 straps (seed 150 gray,
  textured) + thin zipper + THREE light fold curves (150)
- strength 0.65 (clears old lines), LoRA 1.0, CN 0.75
- prompt: minimal lines, delicate thin lines, mostly white
  fabric, textured gray straps
DPM-26, 2 seeds. Base: genr137_s42 (lighter interior).
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

PROMPT = (f"{TRIGGER}, zip-up hoodie, minimal lines, "
          "delicate thin lines, mostly white fabric, "
          "textured gray straps, two shoulder "
          "straps, thin zipper, soft shading, "
          "monochrome, pencil sketch, white "
          "background")
NEG = ("thick outlines, bold lines, heavy lines, "
       "dark lines, dense lines, crossing lines, "
       "chaotic, messy, solid black straps, "
       "drawstrings, color, colored, lowres, "
       "blurry, watermark")


def _mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[470:660, 120:400] = 255
    return cv2.GaussianBlur(m, (0, 0), 8)


def _init(base):
    init = base.copy()
    # whiten the chest interior EXCEPT straps + zipper
    wipe = np.zeros_like(init)
    wipe[470:660, 120:400] = 255
    keep = np.zeros_like(init)
    for xs in (140, 375):
        cv2.line(keep, (xs, 445), (xs, 660), 255, 26,
                 cv2.LINE_AA)
    cv2.line(keep, (CX, 460), (CX, 660), 255, 14, cv2.LINE_AA)
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8),
                      iterations=2)
    init[(wipe > 0) & (keep == 0)] = 255
    # seed straps at 150 (textured gray)
    init = init.astype(np.float32)
    band = np.zeros_like(init)
    for xs in (140, 375):
        cv2.line(band, (xs, 445), (xs, 660), 255, 22,
                 cv2.LINE_AA)
    a = cv2.GaussianBlur(band, (0, 0), 4) / 255.0
    dark = init * (1 - a) + 150 * a
    init = np.minimum(init, dark)
    return np.clip(init, 0, 255).astype(np.uint8)


def _guide(base):
    g = _lineart(base)
    g[470:660, 120:400] = 255
    # light side seams
    cv2.line(g, (125, 480), (112, 656), 130, 1, cv2.LINE_AA)
    cv2.line(g, (391, 480), (404, 656), 130, 1, cv2.LINE_AA)
    # 2 straps (light edges)
    for xs in (140, 375):
        for off in (-9, 9):
            cv2.line(g, (xs + off, 445), (xs + off, 656), 100,
                     1, cv2.LINE_AA)
    # thin zipper
    cv2.line(g, (CX, 460), (CX, 656), 90, 1, cv2.LINE_AA)
    # ONLY three light fold curves
    folds = [
        [(165, 490), (200, 550), (214, 648)],
        [(258, 485), (258, 560), (258, 648)],
        [(351, 490), (316, 550), (302, 648)],
    ]
    for pts in folds:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 150, 1, cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr137_s42.png"),
                      0)
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
    cv2.imwrite(os.path.join(OUT, "_r138_guide.png"), guide)
    panels = [("base", base)]
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=Image.fromarray(init).convert("RGB"),
                   mask_image=Image.fromarray(mask)
                   .convert("RGB"),
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.75,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.65,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr138_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"s{seed}", g))
        print(f"[r138 {name}] {time.time() - t0:.0f}s",
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
    cv2.imwrite(os.path.join(OUT, "_genr138_cloth.png"), csheet)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr138_sheet.png"), sheet)
    print("[r138] sheets saved", flush=True)


if __name__ == "__main__":
    main()
