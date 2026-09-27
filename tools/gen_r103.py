"""R103 (user: 领子高度不行/头发也不是黑色/这么明确的指令
模型就是画不出来):

Stop ASKING the model - FORCE it via pixel-level seeding
(the tonal-init trick that worked for clothes tone & straps):

A. BLACK HAIR: multiply the hair region of the init x0.45
   (original hair mean ~110, ours ~200) + 'black hair,
   dark hair' prompt + 'light/gray hair' negatives.
   DPM-26, strength 0.65.
B. COLLAR ERADICATION: whiten the ENTIRE collar zone
   (y390-475) in both init AND guide - ControlNet gets zero
   high-collar lines, the model inherits zero high-collar
   pixels. Redraw only: bare neck to y465, small hood humps
   behind the neck base at y475-495, deep V to y500, zipper
   from y505. CN 1.0, strength 0.85.
Base: genr102_c42_u.
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
from gen_r97 import _hair_mask  # noqa: E402
from gen_r101 import _load_dpm  # noqa: E402
from gen_r102 import (COLLAR_PROMPT, COLLAR_NEG, STEPS,
                      _collar_mask)  # noqa: E402

HAIR_PROMPT = (f"{TRIGGER}, black hair, dark hair, short "
               "dark hair, fine light strand lines on "
               "dark hair, spiky dark hair, pencil "
               "sketch, monochrome, white background")
HAIR_NEG = ("light hair, gray hair, white hair, blonde, "
            "color, colored, photo, lowres, blurry, "
            "watermark")


def _hair_init(base, mask):
    """darken the hair region x0.45, feathered at the edge"""
    init = base.astype(np.float32)
    a = (mask.astype(np.float32) / 255.0)[:, :]
    dark = init * 0.45
    out = init * (1 - a) + dark * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _collar_guide(base):
    g = _lineart(base)
    g[388:478, :] = 255          # eradicate ALL collar lines
    # bare neck to y465
    cv2.line(g, (CX - 22, 392), (CX - 23, 464), 95, 1,
             cv2.LINE_AA)
    cv2.line(g, (CX + 22, 392), (CX + 23, 464), 95, 1,
             cv2.LINE_AA)
    # small hood humps behind the neck base only
    for sgn in (-1, 1):
        cv2.ellipse(g, (CX + sgn * 64, 482), (40, 16), 0,
                    180, 330, 85, 2, cv2.LINE_AA)
    # deep V opening + zipper pull low
    cv2.line(g, (CX - 26, 468), (CX - 6, 500), 70, 2,
             cv2.LINE_AA)
    cv2.line(g, (CX + 26, 468), (CX + 6, 500), 70, 2,
             cv2.LINE_AA)
    cv2.circle(g, (CX, 506), 6, 45, 2, cv2.LINE_AA)
    cv2.rectangle(g, (CX - 3, 494), (CX + 3, 501), 45, 1)
    cv2.line(g, (CX, 512), (CX, 656), 55, 2, cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr102_c42_u.png"),
                      0)
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

    # ---- A: black hair (tonal seeding) ----
    hmask = _hair_mask((H, W))
    hinit = _hair_init(base, hmask)
    cv2.imwrite(os.path.join(OUT, "_r103_hinit.png"), hinit)
    hairs = []
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=HAIR_PROMPT, negative_prompt=HAIR_NEG,
                   image=Image.fromarray(hinit).convert("RGB"),
                   mask_image=Image.fromarray(hmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(
                       _lineart(base)).convert("RGB"),
                   controlnet_conditioning_scale=0.8,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.65,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr103_h{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        hairs.append((seed, g))
        print(f"[r103 {name}] {time.time() - t0:.0f}s",
              flush=True)

    # ---- B: collar eradication on each ----
    cguide = _collar_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r103_cguide.png"), cguide)
    cmask = _collar_mask((H, W))
    panels = [("base", base)]
    for seed, himg in hairs:
        cinit = himg.copy()
        cinit[388:478, :] = 255
        t0 = time.time()
        res = pipe(prompt=COLLAR_PROMPT,
                   negative_prompt=COLLAR_NEG,
                   image=Image.fromarray(cinit).convert("RGB"),
                   mask_image=Image.fromarray(cmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(cguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.85,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr103_h{seed}_c.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        print(f"[r103 {name}] {time.time() - t0:.0f}s",
              flush=True)
        panels.append((f"h{seed}", himg))
        panels.append((f"h{seed}c", g))

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr103_sheet.png"), sheet)
    hz = [cv2.resize(im[30:280, 70:450], None, fx=1.2, fy=1.2,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    hh, hw = hz[0].shape
    hsheet = np.full((hh, hw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, hz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        hsheet[:, x:x + hw] = t
        x += hw + 8
    cv2.imwrite(os.path.join(OUT, "_genr103_hairs.png"), hsheet)
    cz = [cv2.resize(im[380:530, 120:400], None, fx=2.0, fy=2.0,
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
    cv2.imwrite(os.path.join(OUT, "_genr103_collars.png"),
                csheet)
    print("[r103] sheets saved", flush=True)


if __name__ == "__main__":
    main()
