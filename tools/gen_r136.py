"""R136 (user: s7定底; 衣服纹理杂乱不像原图 + 领子处还是
有空白块; 好好修复衣服, 画风不要变, 要模仿原图):

The ORIGINAL's clothes recipe (from _cloth_study): soft gray
TONAL base (mean 224, not 252), only 3-5 LONG ELEGANT fold
sweeps (not a dozen chaotic lines), hatching ONLY in the
deep valleys, soft shading under the collar V (kills the
white patch).

One careful torso pass on genr135_s7:
- init: tonal-seeded with the ORIGINAL torso's low-freq
  tonal field (symmetrized, 55%) - proven R87/R88 method
- guide: 5 long elegant sweeps + collar under-shade lines +
  2 hatch valleys; keep the 2 straps + thin zipper
- LoRA 1.0 (trained on the original's clothes), CN 0.85,
  strength 0.55, DPM-26. 2 seeds.
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
from gen_r84 import _hatch  # noqa: E402

PROMPT = (f"{TRIGGER}, zip-up hoodie, soft fabric, long "
          "elegant folds, subtle grey shading, "
          "parallel hatching in shadow, two "
          "shoulder straps, thin zipper, "
          "monochrome, pencil sketch, white "
          "background")
NEG = ("chaotic lines, messy lines, dense lines, "
       "empty white, blank fabric, extra straps, "
       "drawstrings, color, colored, lowres, "
       "blurry, watermark")


def _mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[440:660, :] = 255
    cv2.ellipse(m, (CX, 440), (70, 16), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def _tonal_init(base, orig):
    H = base.shape[0]
    tone_full = orig.astype(np.float32)
    tone_full = cv2.GaussianBlur(tone_full, (0, 0), 28)
    tone_full = np.minimum(tone_full, tone_full[:, ::-1])
    init = base.astype(np.float32).copy()
    target = 255.0 - np.clip(255.0 - tone_full, 0, 80) * 0.55
    w = np.zeros((H, 1), np.float32)
    y0, y1 = 415, 470
    w[y0:y1] = np.linspace(0, 1, y1 - y0)[:, None]
    w[y1:] = 1.0
    blended = init * (1 - w) + np.minimum(init, target) * w
    return np.clip(blended, 0, 255).astype(np.uint8)


def _guide(base):
    g = _lineart(base)
    # 5 long elegant sweeps (original style)
    sweeps = [
        [(150, 480), (185, 540), (205, 648)],
        [(200, 475), (222, 540), (234, 648)],
        [(258, 480), (258, 560), (258, 648)],
        [(316, 475), (294, 540), (282, 648)],
        [(366, 480), (331, 540), (311, 648)],
    ]
    for pts in sweeps:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 120, 1, cv2.LINE_AA)
    # soft under-collar shading lines (kill the white patch)
    for sgn in (-1, 1):
        for k in range(4):
            x0 = CX + sgn * (30 + k * 14)
            cv2.line(g, (x0, 446), (x0 + sgn * 10, 462), 145,
                     1, cv2.LINE_AA)
    # hatch ONLY the two deep valleys
    zones = [
        [(196, 550), (220, 550), (216, 645), (192, 645)],
        [(296, 550), (320, 550), (324, 645), (300, 645)],
    ]
    for poly in zones:
        m = np.zeros_like(g)
        cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
        tmp = np.full_like(g, 255)
        _hatch(tmp, poly, spacing=8, val=140)
        g[m > 0] = tmp[m > 0]
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr135_s7.png"), 0)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"), 0)
    assert base is not None and orig is not None
    H, W = base.shape
    origc = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                       cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(origc, cv2.COLOR_BGR2RGB)
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
    init = _tonal_init(base, orig)
    guide = _guide(base)
    cv2.imwrite(os.path.join(OUT, "_r136b_guide.png"), guide)
    panels = [("base", base)]
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=Image.fromarray(init).convert("RGB"),
                   mask_image=Image.fromarray(mask)
                   .convert("RGB"),
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.85,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.55,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr136b_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"s{seed}", g))
        print(f"[r136 {name}] {time.time() - t0:.0f}s",
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
    cv2.imwrite(os.path.join(OUT, "_genr136b_cloth.png"), csheet)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr136b_sheet.png"), sheet)
    print("[r136] sheets saved", flush=True)


if __name__ == "__main__":
    main()
