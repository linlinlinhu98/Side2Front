"""R126B (user correction: 不是说领子, 是说书包背带 - the
strap GHOSTS: diagonal fold lines radiating from the collar
across the chest (y420-470, the zone R125 missed) still read
as extra straps):

CHEST MINIMALISM: erase EVERYTHING below the collar except
[2 vertical straps + thin zipper + collar V edges] - in
both init and guide. No diagonals, no curves, no folds.
Plus the (correct) parts of R126:
A. FACE: level gaze (iris in lower half) + FULL lip set
   (upper-lip outline 110 + straight line + lower-lip
   outline).

Base: genr125_s7. DPM-26, 2+2 seeds.
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
from gen_r126 import (FACE_PROMPT, FACE_NEG, _face_guide,
                      _face_mask)  # noqa: E402

CHEST_PROMPT = (f"{TRIGGER}, zip-up hoodie, two vertical "
                "backpack shoulder straps, thin zipper, "
                "clean fabric, minimal lines, "
                "monochrome, pencil sketch, soft "
                "shading, white background")
CHEST_NEG = ("extra straps, diagonal lines, fold "
             "lines, wrinkles, messy lines, heavy "
             "lines, drawstrings, color, colored, "
             "lowres, blurry, watermark")

STRAP_X = (132, 158)
STRAP2_X = (362, 388)
ZIP_X = (248, 268)


def _keep_mask(shape):
    """regions to KEEP: 2 strap bands + zipper column"""
    H, W = shape
    k = np.zeros((H, W), np.uint8)
    k[:, STRAP_X[0]:STRAP_X[1]] = 255
    k[:, STRAP2_X[0]:STRAP2_X[1]] = 255
    k[:, ZIP_X[0]:ZIP_X[1]] = 255
    return k


def _chest_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[430:660, :] = 255
    cv2.ellipse(m, (CX, 430), (70, 18), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def _minimal_guide(base):
    g = _lineart(base)
    keep = _keep_mask(g.shape)
    g[430:660, :][keep[430:660, :] == 0] = 255
    # re-draw the two straps clean (dark edges) + thin zipper
    for xs in (145, 375):
        for off in (-10, 10):
            cv2.line(g, (xs + off, 440), (xs + off, 656), 50,
                     2, cv2.LINE_AA)
        cv2.rectangle(g, (xs - 7, 520), (xs + 7, 530), 50, 1)
    cv2.line(g, (CX, 440), (CX, 656), 60, 2, cv2.LINE_AA)
    for y in range(448, 650, 9):
        cv2.line(g, (CX - 4, y + 2), (CX + 4, y - 2), 95, 1,
                 cv2.LINE_AA)
    # crisp collar V
    cv2.line(g, (CX - 30, 432), (CX - 6, 466), 50, 2,
             cv2.LINE_AA)
    cv2.line(g, (CX + 30, 432), (CX + 6, 466), 50, 2,
             cv2.LINE_AA)
    return g


def _chest_init(base):
    init = base.copy()
    keep = _keep_mask(base.shape)
    wipe = (keep == 0)
    wipe[:430, :] = False
    init[wipe] = 255
    return init


def main():
    base = cv2.imread(os.path.join(OUT, "genr125_s7.png"), 0)
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

    # ---- A: face (level gaze + full lips) ----
    fguide = _face_guide(base)
    fmask = _face_mask((H, W))
    finit = base.copy()
    finit[246:342, 150:366] = 255
    faces = []
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=FACE_PROMPT, negative_prompt=FACE_NEG,
                   image=Image.fromarray(finit).convert("RGB"),
                   mask_image=Image.fromarray(fmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(fguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.75,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr126b_f{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        faces.append((seed, g))
        print(f"[r126b {name}] {time.time() - t0:.0f}s",
              flush=True)

    # ---- B: chest minimalism on each ----
    cmask = _chest_mask((H, W))
    cguide = _minimal_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r126b_cguide.png"), cguide)
    panels = [("base", base)]
    for seed, fimg in faces:
        cinit = _chest_init(fimg)
        t0 = time.time()
        res = pipe(prompt=CHEST_PROMPT,
                   negative_prompt=CHEST_NEG,
                   image=Image.fromarray(cinit).convert("RGB"),
                   mask_image=Image.fromarray(cmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(cguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.75,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr126b_f{seed}_c.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        print(f"[r126b {name}] {time.time() - t0:.0f}s",
              flush=True)
        panels.append((f"f{seed}", fimg))
        panels.append((f"f{seed}c", g))

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr126b_sheet.png"), sheet)
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
    cv2.imwrite(os.path.join(OUT, "_genr126b_straps.png"),
                csheet)
    mz = [cv2.resize(im[255:345, 155:365], None, fx=3.4, fy=3.4,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    mh, mw = mz[0].shape
    msheet = np.full((mh, mw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, mz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        msheet[:, x:x + mw] = t
        x += mw + 8
    cv2.imwrite(os.path.join(OUT, "_genr126b_face.png"), msheet)
    print("[r126b] sheets saved", flush=True)


if __name__ == "__main__":
    main()
