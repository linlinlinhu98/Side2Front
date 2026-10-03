"""R151 (user review fix #3: the zipper below the ring pull
(y>530) degenerates into wavy wandering double lines, and a
stray arc sweeps down-left from below the ring to the lower
chest; the original's zipper teeth are regular and run the
full length):

Zipper-column pass BELOW the ring pull (the ring itself is
good - protected above y530):
- guide: whiten the column x205-285/y535-658 (erases the
  wavy lines AND the arc), redraw a STRAIGHT zipper at
  x=258 with ladder-style alternating teeth every 8px
- init: whiten the same column so the old wavy lines
  cannot survive img2img
- composite back only the zone (ring pull, collar, face,
  straps bit-identical, verified)
Base: genr150_s42 (strap shifted). DPM-26, LoRA 0.8,
CN 1.0, strength 0.70. 2 seeds.
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
from gen_anime_common import CX, LORA_DIR, STEPS, TRIGGER, _lineart, _load_dpm, _scale_lora  # noqa: E402


def _seeds():
    return tuple(int(x) for x in
                 os.environ.get("S2F_SEEDS", "7,42").split(",")
                 if x)

PROMPT = (f"{TRIGGER}, full-length zipper, metal zipper "
          "teeth, straight zipper, zipper teeth row, "
          "zip-up hoodie, clean fabric, pencil sketch, "
          "monochrome, white background")
NEG = ("wavy lines, messy lines, scribble, drawstrings, "
       "cord, buttons, hidden zipper, short zipper, "
       "color, colored, lowres, blurry, watermark")

ZX0, ZX1, ZY0, ZY1 = 205, 290, 530, 660


def _mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[ZY0:ZY1, 212:ZX1] = 255
    return cv2.GaussianBlur(m, (0, 0), 6)


def _composite(base, result, mask):
    a = mask.astype(np.float32) / 255.0
    out = base.astype(np.float32) * (1 - a) \
        + result.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _guide(base):
    g = _lineart(base)
    g[ZY0 + 4:ZY1, ZX0:ZX1] = 255
    # straight zipper spine
    cv2.line(g, (CX, ZY0 + 6), (CX, 656), 55, 2, cv2.LINE_AA)
    # ladder-style alternating teeth
    left = True
    for y in range(ZY0 + 12, 654, 8):
        if left:
            cv2.line(g, (CX - 5, y), (CX, y), 90, 1,
                     cv2.LINE_AA)
        else:
            cv2.line(g, (CX, y), (CX + 5, y), 90, 1,
                     cv2.LINE_AA)
        left = not left
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr150_s42.png"),
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
    _scale_lora(pipe.unet, 0.8)
    pipe.set_ip_adapter_scale(0.8)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    mask = _mask((H, W))
    guide = _guide(base)
    init = base.copy()
    init[ZY0 + 4:ZY1, ZX0:ZX1] = 255
    cv2.imwrite(os.path.join(OUT, "_r151_guide.png"), guide)

    def _stats(im):
        zip_dark = int((im[545:655, 248:268] < 150).sum())
        arc_dark = int((im[555:650, 208:248] < 150).sum())
        return zip_dark, arc_dark

    z0, a0 = _stats(base)
    print(f"[r151] before: zip_col_dark={z0} arc_dark={a0}",
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
        name = f"genr151_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), final)
        unchanged = (final[mask == 0] == base[mask == 0]).mean()
        z, a = _stats(final)
        print(f"[r151 {name}] {time.time() - t0:.0f}s "
              f"unchanged_outside={unchanged:.3f} "
              f"zip_dark={z} arc_dark={a}", flush=True)

    panels = [("base", base)]
    for seed in _seeds():
        im = cv2.imread(os.path.join(OUT, f"genr151_s{seed}.png"), 0)
        panels.append((f"s{seed}", im))
    n = len(panels)
    cz = [cv2.resize(im[480:660, 190:310], None, fx=3.0, fy=3.0,
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
    cv2.imwrite(os.path.join(OUT, "_genr151_zipper.png"), sheet)
    print("[r151] sheets saved", flush=True)


if __name__ == "__main__":
    main()
