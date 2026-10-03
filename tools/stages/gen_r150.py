"""R150 (user review fix #2: right strap sits at x=359, only
101px from the zipper center; left strap is at x=108, i.e.
150px out - move the right strap RIGHT to x~408 so the pair
is symmetric about the zipper/body center):

Single masked pass over the right-chest strap zone:
- guide: whiten the OLD strap column (x344-400) AND the NEW
  strap column (x394-422) so no fold lines cross the new
  strap, then draw strap edges at x=408 +/-13, diagonal
  hatching every 12px (same as R148/left strap texture),
  and the slider buckle at the same height as before (y520)
- init: whiten the old strap column, dark-seed (value 120)
  the new strap column (tonal seeding, R145 pattern)
- composite back only the zone (everything else
  bit-identical, verified)
Base: genr149_s7 (cord removed). DPM-26, LoRA 1.0, CN 1.0,
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
          "strap, fabric strap, zip-up hoodie, pencil "
          "sketch, monochrome, white background")
NEG = ("diagonal strap, crossbody strap, messenger bag, "
       "extra straps, drawstrings, smooth, flat, color, "
       "colored, lowres, blurry, watermark")

OLD_X0, OLD_X1 = 344, 400      # old strap column to erase
NEW_SX = 408                   # new strap center
NEW_X0, NEW_X1 = NEW_SX - 14, NEW_SX + 14
ZY0, ZY1 = 430, 660            # zone vertical extent


def _mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[ZY0:ZY1, 336:430] = 255
    return cv2.GaussianBlur(m, (0, 0), 8)


def _composite(base, result, mask):
    a = mask.astype(np.float32) / 255.0
    out = base.astype(np.float32) * (1 - a) \
        + result.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _init(base):
    init = base.astype(np.float32).copy()
    # erase old strap
    init[ZY0 + 5:ZY1, OLD_X0:OLD_X1] = 255
    # dark seed for the new strap (tonal seeding)
    seed = np.zeros_like(init)
    cv2.line(seed, (NEW_SX, 435), (NEW_SX, 660), 255, 26,
             cv2.LINE_AA)
    a = cv2.GaussianBlur(seed, (0, 0), 4) / 255.0
    dark = init * (1 - a) + 120 * a
    return np.clip(np.minimum(init, dark), 0, 255) \
        .astype(np.uint8)


def _guide(base):
    g = _lineart(base)
    # erase old strap and clear the new strap's column
    g[ZY0 + 2:ZY1, OLD_X0:OLD_X1] = 255
    g[ZY0 + 2:ZY1, NEW_X0:NEW_X1] = 255
    # new strap edges
    for off in (-13, 13):
        cv2.line(g, (NEW_SX + off, 435), (NEW_SX + off, 656),
                 50, 2, cv2.LINE_AA)
    # diagonal hatching inside (match left strap texture)
    for y in range(448, 650, 12):
        cv2.line(g, (NEW_SX - 11, y + 6), (NEW_SX + 11, y - 6),
                 110, 1, cv2.LINE_AA)
    # slider buckle at the same height as before
    cv2.rectangle(g, (NEW_SX - 8, 516), (NEW_SX + 8, 528),
                  50, 2)
    cv2.rectangle(g, (NEW_SX - 5, 519), (NEW_SX + 5, 525),
                  50, 1)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr149_s7.png"), 0)
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
    cv2.imwrite(os.path.join(OUT, "_r150_guide.png"), guide)

    def _peak_x(im, x0, x1):
        band = im[460:540, x0:x1]
        return x0 + int(np.argmax((255 - band).sum(axis=0)))

    old_dark_before = int((base[460:640, 348:396] < 150).sum())
    print(f"[r150] left peak={_peak_x(base, 60, 170)} "
          f"right peak(before)={_peak_x(base, 330, 430)} "
          f"old-col dark={old_dark_before}", flush=True)
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
        name = f"genr150_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), final)
        unchanged = (final[mask == 0] == base[mask == 0]).mean()
        old_dark = int((final[460:640, 348:396] < 150).sum())
        print(f"[r150 {name}] {time.time() - t0:.0f}s "
              f"unchanged_outside={unchanged:.3f} "
              f"new_peak={_peak_x(final, 330, 450)} "
              f"old_col_dark={old_dark}", flush=True)

    panels = [("base", base)]
    for seed in _seeds():
        im = cv2.imread(os.path.join(OUT, f"genr150_s{seed}.png"), 0)
        panels.append((f"s{seed}", im))
    n = len(panels)
    cz = [cv2.resize(im[420:660, 300:470], None, fx=2.2, fy=2.2,
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
    cv2.imwrite(os.path.join(OUT, "_genr150_strap.png"), sheet)
    print("[r150] sheets saved", flush=True)


if __name__ == "__main__":
    main()
