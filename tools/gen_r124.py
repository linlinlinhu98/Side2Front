"""R124 (user's diagnosis: 上嘴唇不明显所以显得像弯嘴;
背包带子太多了删到只有两条):

A. MOUTH: straight level lip line + a VERY light nearly-
   flat UPPER-LIP top edge above it (value 150, shallow) -
   the upper lip is what makes it read as 'lips' not 'an
   arc'. No lower-lip blob.
B. STRAPS: exactly TWO (one per shoulder) - each strap =
   ONE band (two edges 16px apart + dark seed inside,
   value 90); nothing else vertical near them.

Base: genr123_s42 (fixes-complete candidate). DPM-26.
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

MOUTH_PROMPT = (f"{TRIGGER}, closed lips, upper lip, "
                "straight level lip line, natural lips, "
                "neutral mouth, pencil sketch, soft "
                "shading, monochrome, white background")
MOUTH_NEG = ("curved mouth, arc mouth, smile, frown, "
             "no lips, line mouth, open mouth, color, "
             "colored, lowres, blurry, watermark")

TORSO_PROMPT = (f"{TRIGGER}, zip-up hoodie, backpack, "
                "two shoulder straps, one strap per "
                "shoulder, vertical straps, metal "
                "zipper, clear collar, fabric folds, "
                "monochrome, pencil sketch, soft "
                "shading, white background")
TORSO_NEG = ("many straps, extra straps, diagonal "
             "strap, crossbody, drawstrings, color, "
             "colored, lowres, blurry, watermark")


def _mouth_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CX, 318), (44, 20), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 6)


def _mouth_guide(base):
    g = _lineart(base)
    g[304:338, 216:302] = 255
    # straight level lip line
    cv2.line(g, (CX - 22, 321), (CX + 22, 321), 50, 2,
             cv2.LINE_AA)
    # VERY light nearly-flat upper-lip top edge above
    pts = [(CX - 15, 317), (CX - 6, 315), (CX, 316),
           (CX + 6, 315), (CX + 15, 317)]
    for i in range(len(pts) - 1):
        cv2.line(g, pts[i], pts[i + 1], 150, 1, cv2.LINE_AA)
    return g


def _torso_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[425:, :] = 255
    cv2.ellipse(m, (CX, 425), (64, 18), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def _torso_init(base):
    init = base.astype(np.float32).copy()
    band = np.zeros_like(init)
    for xs in (140, 375):
        cv2.line(band, (xs, 450), (xs, 660), 255, 16,
                 cv2.LINE_AA)
    a = cv2.GaussianBlur(band, (0, 0), 4) / 255.0
    dark = init * (1 - a) + 90 * a
    return np.minimum(init, dark).astype(np.uint8)


def _torso_guide(base):
    g = _lineart(base)
    # erase old strap edge lines in the strap zones
    for xs in (140, 375):
        g[445:660, xs - 26:xs + 26] = 255
    cv2.line(g, (CX - 30, 432), (CX - 6, 466), 45, 2,
             cv2.LINE_AA)
    cv2.line(g, (CX + 30, 432), (CX + 6, 466), 45, 2,
             cv2.LINE_AA)
    cv2.circle(g, (CX, 472), 6, 40, 2, cv2.LINE_AA)
    cv2.rectangle(g, (CX - 3, 460), (CX + 3, 467), 40, 1)
    cv2.line(g, (CX, 478), (CX, 656), 50, 2, cv2.LINE_AA)
    # exactly TWO single-band straps
    for xs in (140, 375):
        for off in (-8, 8):
            cv2.line(g, (xs + off, 450), (xs + off, 656), 45,
                     2, cv2.LINE_AA)
        cv2.rectangle(g, (xs - 6, 520), (xs + 6, 529), 45, 1)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr123_s42.png"),
                      0)
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

    # ---- A: mouth with upper lip ----
    mmask = _mouth_mask((H, W))
    mguide = _mouth_guide(base)
    mouths = []
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=MOUTH_PROMPT,
                   negative_prompt=MOUTH_NEG,
                   image=Image.fromarray(base).convert("RGB"),
                   mask_image=Image.fromarray(mmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(mguide)
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
        name = f"genr124_m{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        mouths.append((seed, g))
        print(f"[r124 {name}] {time.time() - t0:.0f}s",
              flush=True)

    # ---- B: exactly two straps on each ----
    tmask = _torso_mask((H, W))
    panels = [("base", base)]
    for seed, mimg in mouths:
        t0 = time.time()
        res = pipe(prompt=TORSO_PROMPT,
                   negative_prompt=TORSO_NEG,
                   image=Image.fromarray(_torso_init(mimg))
                   .convert("RGB"),
                   mask_image=Image.fromarray(tmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(
                       _torso_guide(mimg)).convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.70,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr124_m{seed}_t.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        print(f"[r124 {name}] {time.time() - t0:.0f}s",
              flush=True)
        panels.append((f"m{seed}", mimg))
        panels.append((f"m{seed}t", g))

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr124_sheet.png"), sheet)
    mz = [cv2.resize(im[305:340, 215:305], None, fx=8, fy=8,
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
    cv2.imwrite(os.path.join(OUT, "_genr124_mouths.png"), msheet)
    cz = [cv2.resize(im[415:600, 50:470], None, fx=1.6, fy=1.6,
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
    cv2.imwrite(os.path.join(OUT, "_genr124_straps.png"), csheet)
    print("[r124] sheets saved", flush=True)


if __name__ == "__main__":
    main()
