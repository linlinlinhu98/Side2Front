"""R134 (fix #3 of 4: 领子中有白色横条 = the big EMPTY white
expanse on the chest below the collar V):

Add the original's LONG fold sweeps + light hatch valleys to
the chest (R112's proven fold system), mask y465-660, init
UNMODIFIED (straps/zipper stay), gentle strength 0.50 so it
only ADDS folds without changing structure.
Base: genr133_s7 (gaze+mouth fixed). DPM-26, CN 1.0,
LoRA 1.0. 2 seeds.
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

PROMPT = (f"{TRIGGER}, zip-up hoodie, fabric folds, "
          "long fold lines, parallel hatching, tonal "
          "shading, two shoulder straps, thin "
          "zipper, monochrome, pencil sketch, soft "
          "shading, white background")
NEG = ("empty white, blank fabric, flat, extra "
       "straps, diagonal strap, drawstrings, "
       "color, colored, lowres, blurry, watermark")


def _mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[465:660, :] = 255
    cv2.ellipse(m, (CX, 465), (70, 16), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def _guide(base):
    g = _lineart(base)
    sweeps = [
        [(150, 480), (180, 530), (200, 590), (210, 648)],
        [(180, 475), (205, 525), (222, 585), (232, 648)],
        [(366, 480), (336, 530), (316, 590), (306, 648)],
        [(336, 475), (311, 525), (294, 585), (284, 648)],
        [(120, 500), (140, 555), (152, 648)],
        [(396, 500), (376, 555), (364, 648)],
    ]
    for pts in sweeps:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 115, 1, cv2.LINE_AA)
    zones = [
        [(190, 545), (216, 545), (212, 645), (186, 645)],
        [(300, 545), (326, 545), (330, 645), (304, 645)],
    ]
    for poly in zones:
        m = np.zeros_like(g)
        cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
        tmp = np.full_like(g, 255)
        _hatch(tmp, poly, spacing=8, val=140)
        g[m > 0] = tmp[m > 0]
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr133_s7.png"), 0)
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

    mask = _mask((H, W))
    guide = _guide(base)
    panels = [("base", base)]
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=Image.fromarray(base).convert("RGB"),
                   mask_image=Image.fromarray(mask)
                   .convert("RGB"),
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.50,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr134_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"s{seed}", g))
        print(f"[r134 {name}] {time.time() - t0:.0f}s",
              flush=True)

    n = len(panels)
    cz = [cv2.resize(im[430:660, 40:480], None, fx=1.5, fy=1.5,
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
    cv2.imwrite(os.path.join(OUT, "_genr134_chest.png"), csheet)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr134_sheet.png"), sheet)
    print("[r134] sheets saved", flush=True)


if __name__ == "__main__":
    main()
