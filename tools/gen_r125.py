"""R125 (user: m7 还算可以, 但是背包带子仍然过多):

Count check (_count_straps.png): the 'extra straps' are the
zipper's TWO heavy tape-edge lines + flanking fold lines in
the center chest (3-4 heavy verticals reading as straps).

Fix on genr124_m7 (user's acceptable base):
- center chest zone (x180-340, y470-660) whitened in init
  AND guide - all fold/tape lines gone
- redraw ONLY a THIN zipper (2px line + small teeth ticks)
- the 2 shoulder straps are outside the zone, untouched
'two shoulder straps, thin zipper, clean fabric' /
'extra straps, fold lines, heavy lines' negatives.
DPM-26, CN 1.0, strength 0.7. 2 seeds.
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

PROMPT = (f"{TRIGGER}, zip-up hoodie, two shoulder "
          "straps, thin zipper, zipper teeth, clean "
          "fabric, subtle folds, monochrome, pencil "
          "sketch, soft shading, white background")
NEG = ("extra straps, many straps, fold lines, heavy "
       "lines, diagonal strap, drawstrings, color, "
       "colored, lowres, blurry, watermark")


def _mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[470:660, 180:340] = 255
    return cv2.GaussianBlur(m, (0, 0), 8)


def _guide(base):
    g = _lineart(base)
    g[470:660, 180:340] = 255
    # THIN zipper only
    cv2.line(g, (CX, 470), (CX, 656), 60, 2, cv2.LINE_AA)
    for y in range(478, 650, 9):
        cv2.line(g, (CX - 4, y + 2), (CX + 4, y - 2), 95, 1,
                 cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr124_m7.png"), 0)
    assert base is not None
    H, W = base.shape
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :])]
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

    mask = _mask((H, W))
    guide = _guide(base)
    cv2.imwrite(os.path.join(OUT, "_r125_guide.png"), guide)
    init = base.copy()
    init[470:660, 180:340] = 255
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
        name = f"genr125_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"s{seed}", g))
        print(f"[r125 {name}] {time.time() - t0:.0f}s",
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
    cv2.imwrite(os.path.join(OUT, "_genr125_sheet.png"), sheet)
    cz = [cv2.resize(im[430:660, 40:480], None, fx=1.8, fy=1.8,
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
    cv2.imwrite(os.path.join(OUT, "_genr125_straps.png"), csheet)
    print("[r125] sheets saved", flush=True)


if __name__ == "__main__":
    main()
