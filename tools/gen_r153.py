"""R153 (user review: (a) small black blob just above-left of
the right strap top at x364-378/y427-437, (b) right strap
top is too narrow - 28px at y470 vs body ~50px and left
strap ~50px; original has NO buckle hardware on straps, so
straps are plain uniform bands):

Right strap top pass (one zone, two related fixes):
- guide: whiten x352-445/y412-500 (blob + narrow taper +
  the fold line that ends in the blob), redraw strap edges
  as a gentle taper into the existing body: left edge
  (384,445)->(390,500), right edge (430,445)->(426,500)
- init: whiten the zone, tonal-seed (value 120) the wider
  top band so the strap renders uniform
- composite back only the zone (strap body below y500,
  collar, face bit-identical, verified)
Base: genr152b_s42. DPM-26, LoRA 1.0, CN 1.0,
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

PROMPT = (f"{TRIGGER}, backpack shoulder strap, textured "
          "strap, parallel hatching, pencil texture, dark "
          "strap, fabric strap, clean shoulder, pencil "
          "sketch, monochrome, white background")
NEG = ("blob, knot, ink blot, tapering strap, narrow "
       "strap, scribble, color, colored, lowres, blurry, "
       "watermark")

ZX0, ZX1, ZY0, ZY1 = 352, 445, 412, 500


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
    # tonal seed: wider top band matching the body
    seed = np.zeros_like(init)
    cv2.line(seed, (407, 445), (407, 500), 255, 44,
             cv2.LINE_AA)
    a = cv2.GaussianBlur(seed, (0, 0), 4) / 255.0
    dark = init * (1 - a) + 120 * a
    return np.clip(np.minimum(init, dark), 0, 255) \
        .astype(np.uint8)


def _guide(base):
    g = _lineart(base)
    g[ZY0 + 4:ZY1, ZX0:ZX1] = 255
    # strap edges: gentle taper into the existing body
    cv2.line(g, (384, 445), (390, 500), 55, 2, cv2.LINE_AA)
    cv2.line(g, (430, 445), (426, 500), 55, 2, cv2.LINE_AA)
    # strap top cap at the shoulder line
    cv2.line(g, (386, 443), (428, 443), 55, 2, cv2.LINE_AA)
    # hatching continues from the body upward
    for y in range(452, 498, 12):
        cv2.line(g, (392, y + 6), (424, y - 6), 110, 1,
                 cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr152b_s42.png"),
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
    cv2.imwrite(os.path.join(OUT, "_r153_guide.png"), guide)

    def _blob(im):
        return int((im[423:441, 360:382] < 150).sum())

    def _span(im, y):
        xs = [x for x in range(350, 450) if im[y, x] < 170]
        return (min(xs), max(xs)) if xs else None

    print(f"[r153] blob before={_blob(base)} "
          f"span470={_span(base, 470)}", flush=True)
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
        name = f"genr153_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), final)
        unchanged = (final[mask == 0] == base[mask == 0]).mean()
        print(f"[r153 {name}] {time.time() - t0:.0f}s "
              f"unchanged_outside={unchanged:.3f} "
              f"blob={_blob(final)} "
              f"span460={_span(final, 460)} "
              f"span470={_span(final, 470)} "
              f"span520={_span(final, 520)}", flush=True)

    panels = [("base", base)]
    for seed in _seeds():
        im = cv2.imread(os.path.join(OUT, f"genr153_s{seed}.png"), 0)
        panels.append((f"s{seed}", im))
    n = len(panels)
    cz = [cv2.resize(im[400:540, 330:460], None, fx=3.0, fy=3.0,
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
    cv2.imwrite(os.path.join(OUT, "_genr153_rtop.png"), sheet)
    print("[r153] sheets saved", flush=True)


if __name__ == "__main__":
    main()
