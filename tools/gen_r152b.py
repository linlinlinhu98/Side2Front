"""R152B (R152 failed - block whitening ate the strap top
and turned the shoulder to mush; switch to the R149-proven
THIN-LINE removal: mask only the messy lines' own dark
pixels, never touch the strap column x375-420):

Two thin-line masks in one pass:
- zone A (shoulder-cap arc cluster): dark px (<150) in
  x420-498/y415-490, dilated+feathered; guide whitens them
  and draws ONE clean shoulder line (420,448)->(490,472)
- zone B (collar-edge knot): dark px in x328-354/y418-444;
  removed, the collar line bridges the gap from context
- init whitens the same pixels; composite back only the
  thin mask (strap body + texture bit-identical, verified)
Base: genr151_s42. DPM-26, LoRA 0.8, CN 1.0,
strength 0.70. 2 seeds.
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


def _seeds():
    return tuple(int(x) for x in
                 os.environ.get("S2F_SEEDS", "7,42").split(",")
                 if x)

PROMPT = (f"{TRIGGER}, clean shoulder seam, smooth "
          "shoulder line, hoodie fabric, pencil sketch, "
          "monochrome, white background")
NEG = ("messy lines, scratchy, doubled lines, scribble, "
       "extra lines, tangled, knot, color, colored, "
       "lowres, blurry, watermark")

Z_A = (420, 498, 415, 490)   # arc cluster (right of strap)
Z_B = (328, 354, 418, 444)   # collar knot


def _thin_mask(base):
    m = np.zeros(base.shape, np.uint8)
    for x0, x1, y0, y1 in (Z_A, Z_B):
        z = base[y0:y1, x0:x1]
        core = (z < 150).astype(np.uint8) * 255
        core = cv2.dilate(core, np.ones((7, 7), np.uint8))
        m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], core)
    return cv2.GaussianBlur(m, (0, 0), 5)


def _composite(base, result, mask):
    a = mask.astype(np.float32) / 255.0
    out = base.astype(np.float32) * (1 - a) \
        + result.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    base = cv2.imread(os.path.join(OUT, "genr151_s42.png"),
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

    mask = _thin_mask(base)
    guide = _lineart(base)
    guide[mask > 60] = 255
    # one clean shoulder line right of the strap
    cv2.line(guide, (420, 448), (490, 472), 55, 2,
             cv2.LINE_AA)
    init = base.copy()
    init[mask > 120] = 255
    cv2.imwrite(os.path.join(OUT, "_r152b_guide.png"), guide)
    cv2.imwrite(os.path.join(OUT, "_r152b_mask.png"), mask)

    def _dark(im, x0, x1, y0, y1):
        return int((im[y0:y1, x0:x1] < 150).sum())

    print(f"[r152b] before: arc={_dark(base, 420, 498, 415, 490)} "
          f"knot={_dark(base, 328, 354, 418, 444)} "
          f"strap={_dark(base, 380, 415, 455, 510)}", flush=True)
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
        name = f"genr152b_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), final)
        unchanged = (final[mask == 0] == base[mask == 0]).mean()
        print(f"[r152b {name}] {time.time() - t0:.0f}s "
              f"unchanged_outside={unchanged:.3f} "
              f"arc={_dark(final, 420, 498, 415, 490)} "
              f"knot={_dark(final, 328, 354, 418, 444)} "
              f"strap={_dark(final, 380, 415, 455, 510)}",
              flush=True)

    panels = [("base", base)]
    for seed in _seeds():
        im = cv2.imread(os.path.join(OUT, f"genr152b_s{seed}.png"), 0)
        panels.append((f"s{seed}", im))
    n = len(panels)
    cz = [cv2.resize(im[390:515, 280:510], None, fx=2.4, fy=2.4,
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
    cv2.imwrite(os.path.join(OUT, "_genr152b_collar.png"), sheet)
    print("[r152b] sheets saved", flush=True)


if __name__ == "__main__":
    main()
