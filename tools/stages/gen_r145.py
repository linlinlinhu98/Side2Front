"""R145 (user's verdict: genr119_s7 is the base - its lines
are CLEAN; fix ONLY the right diagonal strap, keep
everything else EXACTLY):

Diagonal band: (430,430) -> (200,660).
- whiten the band in init; guide erases it and draws a
  VERTICAL strap at x=375 (mirror of the left strap at
  x=140, with buckle at y=520)
- masked pass on the strap zone only, composite back only
  that zone (line clarity elsewhere bit-identical,
  verified)
- run similarity vs original after (user mandate)
DPM-26, LoRA 0.8, CN 1.0, strength 0.70. 2 seeds.
"""
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from trial_inpaint_eyes import _find_file, OUT  # noqa: E402
from gen_anime_common import CX, LORA_DIR, STEPS, TRIGGER, _lineart, _load_dpm, _scale_lora  # noqa: E402


def _seeds():
    return tuple(int(x) for x in
                 os.environ.get("S2F_SEEDS", "7,42").split(",")
                 if x)

PROMPT = (f"{TRIGGER}, zip-up hoodie, two vertical "
          "backpack shoulder straps, black strap, "
          "textured strap, fabric folds, thin "
          "zipper, monochrome, pencil sketch, "
          "white background")
NEG = ("diagonal strap, crossbody strap, messenger "
       "bag, extra straps, drawstrings, color, "
       "colored, lowres, blurry, watermark")

DIAG = ((430, 430), (200, 660))


def _mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[420:660, 150:470] = 255
    return cv2.GaussianBlur(m, (0, 0), 8)


def _composite(base, result, mask):
    a = mask.astype(np.float32) / 255.0
    out = base.astype(np.float32) * (1 - a) \
        + result.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _band(img_shape):
    band = np.zeros(img_shape, np.uint8)
    cv2.line(band, DIAG[0], DIAG[1], 255, 52, cv2.LINE_AA)
    return cv2.GaussianBlur(band, (0, 0), 4)


def _init(base, band):
    init = base.astype(np.float32).copy()
    init[band > 0] = 255
    # dark seed for the new vertical strap
    seed = np.zeros_like(init)
    cv2.line(seed, (375, 435), (375, 660), 255, 22,
             cv2.LINE_AA)
    a = cv2.GaussianBlur(seed, (0, 0), 4) / 255.0
    dark = init * (1 - a) + 120 * a
    return np.clip(np.minimum(init, dark), 0, 255) \
        .astype(np.uint8)


def _guide(base, band):
    g = _lineart(base)
    g[band > 0] = 255
    # vertical strap at x=375 (mirror of the left one)
    for off in (-10, 10):
        cv2.line(g, (375 + off, 435), (375 + off, 656), 50,
                 2, cv2.LINE_AA)
    cv2.rectangle(g, (368, 516), (382, 526), 50, 1)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr119_s7.png"),
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

    mask = _mask((H, W))
    band = _band((H, W))
    init = _init(base, band)
    guide = _guide(base, band)
    cv2.imwrite(os.path.join(OUT, "_r145_guide.png"), guide)
    for seed in _seeds():
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
        final = _composite(base, g, mask)
        name = f"genr145_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), final)
        unchanged = (final[mask == 0] == base[mask == 0]).mean()
        print(f"[r145 {name}] {time.time() - t0:.0f}s "
              f"unchanged_outside={unchanged:.3f}", flush=True)

    panels = [("base", base)]
    for seed in _seeds():
        im = cv2.imread(os.path.join(OUT, f"genr145_s{seed}.png"), 0)
        panels.append((f"s{seed}", im))
    n = len(panels)
    cz = [cv2.resize(im[420:660, 150:470], None, fx=1.6, fy=1.6,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    ch, cw = cz[0].shape
    sheet = np.full((ch, cw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, cz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        csheet = sheet
        csheet[:, x:x + cw] = t
        x += cw + 8
    cv2.imwrite(os.path.join(OUT, "_genr145_straps.png"), sheet)
    print("[r145] sheets saved", flush=True)


if __name__ == "__main__":
    main()
