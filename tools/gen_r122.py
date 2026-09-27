"""R122 (user angry: R121 mouth = seagull frown from MY OWN
3-point kinked guide; right collar = white blob from a
washed-out strap seed. genr119_s7 base is better on both):

A. MOUTH: ONE perfectly horizontal straight line in the
   guide (NO arcs, NO corner kinks, NO polylines), value
   65 th 1.5; 'straight level neutral mouth' / 'curved,
   arc, smile, frown' negatives. LoRA 0.3, CN 1.0, str 0.70.
B. STRAPS symmetric: BOTH bands seeded at value 90 (R121's
   120 washed out), strength 0.75, CN 1.0, crisp dark
   collar V edges in the guide.

Base: genr119_s7 (user's chosen best). DPM-26.
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

MOUTH_PROMPT = (f"{TRIGGER}, straight mouth, level mouth, "
                "neutral mouth, thin lips, closed lips, "
                "pencil sketch, soft shading, "
                "monochrome, white background")
MOUTH_NEG = ("curved mouth, arc, smile, frown, upturned "
             "corners, downturned corners, sad, open "
             "mouth, color, colored, lowres, blurry, "
             "watermark")

TORSO_PROMPT = (f"{TRIGGER}, zip-up hoodie, two black "
                "backpack shoulder straps, vertical "
                "straps, clear collar, crisp lines, "
                "metal zipper, fabric folds, "
                "monochrome, pencil sketch, soft "
                "shading, white background")
TORSO_NEG = ("diagonal strap, crossbody, one strap, "
             "asymmetric, blurry collar, white "
             "smudge, drawstrings, color, colored, "
             "lowres, blurry, watermark")


def _mouth_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CX, 318), (42, 20), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 6)


def _mouth_guide(base):
    g = _lineart(base)
    g[304:338, 216:302] = 255
    # ONE straight horizontal line - nothing else
    cv2.line(g, (CX - 16, 321), (CX + 16, 321), 65, 1,
             cv2.LINE_AA)
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
        cv2.line(band, (xs, 450), (xs, 660), 255, 26,
                 cv2.LINE_AA)
    a = cv2.GaussianBlur(band, (0, 0), 5) / 255.0
    dark = init * (1 - a) + 90 * a
    return np.minimum(init, dark).astype(np.uint8)


def _torso_guide(base):
    g = _lineart(base)
    cv2.line(g, (CX - 30, 432), (CX - 6, 466), 45, 2,
             cv2.LINE_AA)
    cv2.line(g, (CX + 30, 432), (CX + 6, 466), 45, 2,
             cv2.LINE_AA)
    cv2.circle(g, (CX, 472), 6, 40, 2, cv2.LINE_AA)
    cv2.rectangle(g, (CX - 3, 460), (CX + 3, 467), 40, 1)
    cv2.line(g, (CX, 478), (CX, 656), 50, 2, cv2.LINE_AA)
    for xs in (140, 375):
        for off in (-10, 10):
            cv2.line(g, (xs + off, 450), (xs + off, 656), 45,
                     2, cv2.LINE_AA)
        cv2.rectangle(g, (xs - 7, 520), (xs + 7, 530), 45, 1)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr119_s7.png"), 0)
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

    # ---- A: straight mouth ----
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
        name = f"genr122_m{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        mouths.append((seed, g))
        print(f"[r122 {name}] {time.time() - t0:.0f}s",
              flush=True)

    # ---- B: symmetric dark straps on each ----
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
                   guidance_scale=3.5, strength=0.75,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr122_m{seed}_t.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        print(f"[r122 {name}] {time.time() - t0:.0f}s",
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
    cv2.imwrite(os.path.join(OUT, "_genr122_sheet.png"), sheet)
    mz = [cv2.resize(im[300:345, 210:310], None, fx=6, fy=6,
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
    cv2.imwrite(os.path.join(OUT, "_genr122_mouths.png"), msheet)
    cz = [cv2.resize(im[415:560, 60:460], None, fx=1.8, fy=1.8,
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
    cv2.imwrite(os.path.join(OUT, "_genr122_collars.png"),
                csheet)
    print("[r122] sheets saved", flush=True)


if __name__ == "__main__":
    main()
