"""R154 (user rule: both straps must be symmetric and the
SAME width; original has no buckle hardware. R153 removed
the blob + widened the top, but the right strap is still
not uniform: dome top 63px vs upper body 37px):

Full right-strap uniform redraw: whiten the whole strap
zone (dome + varying body), redraw ONE uniform strap:
- edges at x=386 / x=434 (center 410, width 48 = left
  strap's target width), straight vertical y448-655
- top cap at y446 (strap starts at the shoulder line)
- diagonal hatching every 12px, NO buckle
- init: whiten zone + tonal seed (value 120) the uniform
  band
- composite back only the zone
Base: genr153_s42. DPM-26, LoRA 1.0, CN 1.0,
strength 0.70. 2 seeds.
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

PROMPT = (f"{TRIGGER}, backpack shoulder strap, textured "
          "strap, parallel hatching, pencil texture, dark "
          "strap, fabric strap, uniform width, pencil "
          "sketch, monochrome, white background")
NEG = ("tapering, narrow strap, wide bulge, buckle, ring, "
       "hardware, scribble, color, colored, lowres, "
       "blurry, watermark")

ZX0, ZX1, ZY0, ZY1 = 358, 448, 428, 660
SC = 410          # strap center
SW = 24           # half-width -> 48px total


def _mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[ZY0:ZY1, ZX0:ZX1] = 255
    return cv2.GaussianBlur(m, (0, 0), 6)


def _composite(base, result, mask):
    a = mask.astype(np.float32) / 255.0
    out = base.astype(np.float32) * (1 - a) \
        + result.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _init(base):
    init = base.astype(np.float32).copy()
    init[ZY0 + 4:ZY1, ZX0:ZX1] = 255
    seed = np.zeros_like(init)
    cv2.line(seed, (SC, 450), (SC, 658), 255, 44,
             cv2.LINE_AA)
    a = cv2.GaussianBlur(seed, (0, 0), 4) / 255.0
    dark = init * (1 - a) + 120 * a
    return np.clip(np.minimum(init, dark), 0, 255) \
        .astype(np.uint8)


def _guide(base):
    g = _lineart(base)
    g[ZY0 + 4:ZY1, ZX0:ZX1] = 255
    # uniform strap edges
    for off in (-SW, SW):
        cv2.line(g, (SC + off, 448), (SC + off, 656), 55,
                 2, cv2.LINE_AA)
    # top cap at the shoulder line
    cv2.line(g, (SC - SW, 447), (SC + SW, 447), 55, 2,
             cv2.LINE_AA)
    # diagonal hatching
    for y in range(458, 652, 12):
        cv2.line(g, (SC - SW + 6, y + 6), (SC + SW - 6,
                 y - 6), 110, 1, cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr153_s42.png"),
                      0)
    assert base is not None
    H, W = base.shape
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
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
    init = _init(base)
    guide = _guide(base)
    cv2.imwrite(os.path.join(OUT, "_r154_guide.png"), guide)

    def _span(im, y):
        xs = [x for x in range(350, 460) if im[y, x] < 170]
        return (min(xs), max(xs)) if xs else None

    print("[r154] before spans: " + " ".join(
        f"y{y}={_span(base, y)}" for y in (460, 470, 520,
                                           570, 620)),
          flush=True)
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
        name = f"genr154_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), final)
        unchanged = (final[mask == 0] == base[mask == 0]).mean()
        print(f"[r154 {name}] {time.time() - t0:.0f}s "
              f"unchanged_outside={unchanged:.3f} spans: "
              + " ".join(f"y{y}={_span(final, y)}"
                         for y in (460, 470, 520, 570, 620)),
              flush=True)

    panels = [("base", base)]
    for seed in _seeds():
        im = cv2.imread(os.path.join(OUT, f"genr154_s{seed}.png"), 0)
        panels.append((f"s{seed}", im))
    n = len(panels)
    cz = [cv2.resize(im[420:660, 340:460], None, fx=2.6, fy=2.6,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    ch, cw = cz[0].shape
    sheet = np.full((ch, cw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, cz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + cw] = t
        x += cw + 8
    cv2.imwrite(os.path.join(OUT, "_genr154_rstrap.png"), sheet)
    print("[r154] sheets saved", flush=True)


if __name__ == "__main__":
    main()
