"""R111: final push past 0.70 (seed lottery capped at
0.696/0.677 - refine the face on genr110_s998, the best
WIDE-shoulder package):

A. EYE narrowing (R99 proven recipe, edit mode): long thin
   almond guide (68px, lid covers ~55%, 2px outer upturn),
   eye-band mask, strength 0.70
B. MOUTH soft (R96/R104 proven recipe): faint tonal blob
   (232) + clean-mouth prompt + upturned-corner guide,
   strength 0.65
Both DPM-26, on genr110_s998. 2+2 seeds.
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
from gen_r96 import _lip_guide, _lip_init  # noqa: E402
from gen_r83 import MOUTH_PROMPT, MOUTH_NEG  # noqa: E402

EYE_PROMPT = (f"{TRIGGER}, long thin almond eyes, single "
              "eyelid, monolid, half-closed eyes, "
              "slightly upturned eyes, calm expression, "
              "looking at viewer, pencil sketch, "
              "monochrome, white background")
EYE_NEG = ("round eyes, big eyes, double eyelid, shiny "
           "eyes, fox eyes, sad, color, colored, lowres, "
           "blurry, watermark")


def _eye_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CX, 276), (100, 28), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 7)


def _eye_guide(base):
    g = _lineart(base)
    g[250:298, 150:366] = 255
    # keep brows (redraw approved tapered)
    for sgn in (-1, 1):
        xs = [CX + sgn * d for d in range(28, 89, 10)]
        ys = [253, 252, 251, 251, 252, 253, 253]
        ths = [2, 2, 2, 2, 1, 1]
        for i in range(len(xs) - 1):
            cv2.line(g, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                     90, ths[i], cv2.LINE_AA)
    for cx in (196, 320):
        cv2.circle(g, (cx, 276), 5, 100, -1, cv2.LINE_AA)
        cv2.rectangle(g, (cx - 35, 260), (cx + 35, 271), 255,
                      -1)
        up = [(cx - 34, 273), (cx - 14, 270), (cx + 14, 270),
              (cx + 34, 273)]
        for i in range(3):
            cv2.line(g, up[i], up[i + 1], 45, 3, cv2.LINE_AA)
        sgn = -1 if cx < CX else 1
        cv2.line(g, (cx + sgn * 34, 273),
                 (cx + sgn * 39, 271), 45, 2, cv2.LINE_AA)
        lo = [(cx - 34, 273), (cx - 14, 279), (cx + 14, 279),
              (cx + 34, 273)]
        for i in range(3):
            cv2.line(g, lo[i], lo[i + 1], 95, 1, cv2.LINE_AA)
    return g


def _mouth_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CX, 317), (40, 20), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 6)


def main():
    base = cv2.imread(os.path.join(OUT, "genr110_s998.png"),
                      0)
    assert base is not None
    H, W = base.shape
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[170:300, 100:320])]
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

    # ---- A: eyes ----
    emask = _eye_mask((H, W))
    eguide = _eye_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r111_eguide.png"), eguide)
    einit = base.copy()
    einit[250:298, 150:366] = 255
    eyes = []
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=EYE_PROMPT, negative_prompt=EYE_NEG,
                   image=Image.fromarray(einit).convert("RGB"),
                   mask_image=Image.fromarray(emask)
                   .convert("RGB"),
                   control_image=Image.fromarray(eguide)
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
        name = f"genr111_e{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        eyes.append((seed, g))
        print(f"[r111 {name}] {time.time() - t0:.0f}s",
              flush=True)

    # ---- B: mouth on each ----
    mmask = _mouth_mask((H, W))
    panels = [("base", base)]
    for seed, eimg in eyes:
        t0 = time.time()
        res = pipe(prompt=MOUTH_PROMPT,
                   negative_prompt=MOUTH_NEG,
                   image=Image.fromarray(_lip_init(eimg))
                   .convert("RGB"),
                   mask_image=Image.fromarray(mmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(
                       _lip_guide(eimg)).convert("RGB"),
                   controlnet_conditioning_scale=0.8,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.65,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr111_e{seed}_m.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        print(f"[r111 {name}] {time.time() - t0:.0f}s",
              flush=True)
        panels.append((f"e{seed}", eimg))
        panels.append((f"e{seed}m", g))

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr111_sheet.png"), sheet)
    ez = [cv2.resize(im[246:305, 150:370], None, fx=3.0, fy=3.0,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    eh, ew = ez[0].shape
    esheet = np.full((eh, ew * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, ez):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        esheet[:, x:x + ew] = t
        x += ew + 8
    cv2.imwrite(os.path.join(OUT, "_genr111_eyes.png"), esheet)
    mz = [cv2.resize(im[305:348, 218:302], None, fx=6, fy=6,
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
    cv2.imwrite(os.path.join(OUT, "_genr111_mouths.png"), msheet)
    print("[r111] sheets saved", flush=True)


if __name__ == "__main__":
    main()
