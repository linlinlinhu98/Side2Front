"""R146 (clothes review fix #1: the odd ring+chain at the
LEFT strap top (x~140,y445) - the zipper pull ring
duplicated at the wrong place; should be a square strap
buckle or nothing):

Small zone pass: whiten the ring+chain in init AND guide,
redraw a simple SQUARE strap slider buckle there instead.
Composite back only that zone. Base: genr145_s42.
DPM-26, LoRA 0.8, CN 1.0, strength 0.60. 2 seeds.
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

PROMPT = (f"{TRIGGER}, backpack strap, strap slider "
          "buckle, rectangular buckle, fabric strap, "
          "pencil sketch, monochrome, white background")
NEG = ("ring, circle, chain, zipper pull, round "
       "buckle, color, colored, lowres, blurry, "
       "watermark")

ZONE = (105, 180, 425, 480)     # x0, x1, y0, y1 (the ring area)


def _mask(shape):
    H, W = shape
    x0, x1, y0, y1 = ZONE
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, ((x0 + x1) // 2, (y0 + y1) // 2),
                ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360,
                255, -1)
    return cv2.GaussianBlur(m, (0, 0), 6)


def _composite(base, result, mask):
    a = mask.astype(np.float32) / 255.0
    out = base.astype(np.float32) * (1 - a) \
        + result.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _guide(base):
    g = _lineart(base)
    x0, x1, y0, y1 = ZONE
    g[y0:y1, x0:x1] = 255
    # the strap continues through
    cv2.line(g, (140, y0), (140, y1), 70, 2, cv2.LINE_AA)
    # simple SQUARE slider buckle
    cv2.rectangle(g, (132, 445), (148, 462), 60, 2)
    cv2.rectangle(g, (136, 449), (144, 458), 60, 1)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr145_s42.png"),
                      0)
    assert base is not None
    H, W = base.shape
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :])]
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
    x0, x1, y0, y1 = ZONE
    init[y0 + 10:y1 - 5, x0 + 10:x1 - 10] = 255
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
                   guidance_scale=3.5, strength=0.60,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        final = _composite(base, g, mask)
        name = f"genr146_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), final)
        unchanged = (final[mask == 0] == base[mask == 0]).mean()
        print(f"[r146 {name}] {time.time() - t0:.0f}s "
              f"unchanged_outside={unchanged:.3f}", flush=True)

    panels = [("base", base)]
    for seed in _seeds():
        im = cv2.imread(os.path.join(OUT, f"genr146_s{seed}.png"), 0)
        panels.append((f"s{seed}", im))
    n = len(panels)
    cz = [cv2.resize(im[420:490, 90:200], None, fx=4, fy=4,
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
    cv2.imwrite(os.path.join(OUT, "_genr146_buckle.png"), sheet)
    print("[r146] sheets saved", flush=True)


if __name__ == "__main__":
    main()
