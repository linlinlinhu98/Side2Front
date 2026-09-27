"""R120B: strap fix STRENGTHENED (R120 failed: diagonal strap
survived strength 0.6; lip pass made mouth worse - lips left
untouched this time).

Whiten band 90px wide + strength 0.80 + CN 1.0 + vertical-
straps-only guide. Base: genr119_s7 (user's pick). 2 seeds.
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

STRAP_PROMPT = (f"{TRIGGER}, zip-up hoodie, backpack, two "
                "vertical shoulder straps, shoulder "
                "straps, metal zipper, fabric folds, "
                "monochrome, pencil sketch, soft "
                "shading, white background")
STRAP_NEG = ("diagonal strap, crossbody strap, messenger "
             "bag, satchel, drawstrings, color, colored, "
             "lowres, blurry, watermark")


def _band_mask():
    band = np.zeros((664, 520), np.uint8)
    cv2.line(band, (350, 445), (140, 664), 255, 90,
             cv2.LINE_AA)
    return cv2.GaussianBlur(band, (0, 0), 4)


def _torso_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[430:, :] = 255
    cv2.ellipse(m, (CX, 430), (64, 18), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


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

    band = _band_mask()[:H, :W]
    init = base.copy()
    init[band > 0] = 255
    g = _lineart(base)
    g[band > 0] = 255
    for xs in (140, 375):
        for off in (-9, 9):
            cv2.line(g, (xs + off, 455), (xs + off, 656), 55,
                     2, cv2.LINE_AA)
        cv2.rectangle(g, (xs - 6, 520), (xs + 6, 529), 55, 1)
    cv2.imwrite(os.path.join(OUT, "_r120b_sguide.png"), g)
    tmask = _torso_mask((H, W))

    panels = [("base", base)]
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=STRAP_PROMPT,
                   negative_prompt=STRAP_NEG,
                   image=Image.fromarray(init).convert("RGB"),
                   mask_image=Image.fromarray(tmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(g).convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.80,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        gg = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        gg = cv2.resize(gg, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr120b_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), gg)
        panels.append((f"s{seed}", gg))
        print(f"[r120b {name}] {time.time() - t0:.0f}s",
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
    cv2.imwrite(os.path.join(OUT, "_genr120b_sheet.png"), sheet)
    print("[r120b] sheet saved", flush=True)


if __name__ == "__main__":
    main()
