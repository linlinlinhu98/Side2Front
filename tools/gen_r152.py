"""R152 (user review fix #4: right collar + shoulder lines
are messy - (a) 3+ overlapping swooping arcs on the
shoulder cap x405-495/y420-485 from the R150 mask-feather
boundary misalignment, (b) an isolated dark knot on the
collar edge x333-345/y424-436, (c) doubled/ragged lines
where the strap top meets the shoulder x375-428/y412-450;
the LEFT side is clean - single collar line, one shoulder
line, strap emerging cleanly):

Right collar/shoulder cleanup pass:
- guide: whiten the three messy zones, redraw ONE clean
  shoulder line ((360,436)->(492,470)), a smooth collar
  edge segment ((318,420)->(362,438)), and a tidy strap
  top cap at y~438
- init: whiten the same zones
- composite back only the zone (strap body below y452,
  collar V, neck, face all bit-identical, verified)
Base: genr151_s42 (zipper fixed). DPM-26, LoRA 0.6,
CN 1.0, strength 0.65. 2 seeds.
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

PROMPT = (f"{TRIGGER}, clean shoulder seam, smooth "
          "shoulder line, hoodie collar, backpack strap, "
          "clean fabric, pencil sketch, monochrome, "
          "white background")
NEG = ("messy lines, scratchy, doubled lines, scribble, "
       "extra lines, tangled, color, colored, lowres, "
       "blurry, watermark")

# messy zones (x0, x1, y0, y1)
Z_ARC = (398, 500, 412, 495)     # shoulder-cap arc cluster
Z_KNOT = (322, 358, 416, 446)    # collar-edge knot
Z_TOP = (372, 430, 408, 452)     # strap-top tangle


def _mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[405:500, 315:505] = 255
    return cv2.GaussianBlur(m, (0, 0), 6)


def _whiten(g):
    for x0, x1, y0, y1 in (Z_ARC, Z_KNOT, Z_TOP):
        g[y0:y1, x0:x1] = 255
    return g


def _guide(base):
    g = _whiten(_lineart(base))
    # one clean shoulder line
    cv2.line(g, (360, 436), (492, 470), 55, 2, cv2.LINE_AA)
    # smooth collar edge segment
    cv2.line(g, (318, 420), (362, 438), 55, 2, cv2.LINE_AA)
    # strap top cap
    cv2.line(g, (388, 440), (388, 452), 55, 2, cv2.LINE_AA)
    cv2.line(g, (414, 440), (414, 452), 55, 2, cv2.LINE_AA)
    cv2.line(g, (388, 439), (414, 439), 55, 2, cv2.LINE_AA)
    return g


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
    _scale_lora(pipe.unet, 0.6)
    pipe.set_ip_adapter_scale(0.8)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    mask = _mask((H, W))
    guide = _guide(base)
    init = _whiten(base.copy())
    cv2.imwrite(os.path.join(OUT, "_r152_guide.png"), guide)

    def _mess(im):
        return int((im[418:490, 400:498] < 150).sum())

    print(f"[r152] arc-zone dark before={_mess(base)}",
          flush=True)
    for seed in (7, 42):
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
                   guidance_scale=3.5, strength=0.65,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        final = _composite(base, g, mask)
        name = f"genr152_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), final)
        unchanged = (final[mask == 0] == base[mask == 0]).mean()
        print(f"[r152 {name}] {time.time() - t0:.0f}s "
              f"unchanged_outside={unchanged:.3f} "
              f"arc_dark={_mess(final)}", flush=True)

    panels = [("base", base)]
    for seed in (7, 42):
        im = cv2.imread(os.path.join(OUT, f"genr152_s{seed}.png"), 0)
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
    cv2.imwrite(os.path.join(OUT, "_genr152_collar.png"), sheet)
    print("[r152] sheets saved", flush=True)


if __name__ == "__main__":
    main()
