"""R142 (fix #2: 两边腮红不对称 - right cheek has diagonal
hatch blush, left cheek is bare):

Protocol (user: one thing at a time, everything else fixed):
tiny zone pass on the LEFT cheek only, guide = diagonal
hatch lines mirroring the right cheek's pattern, then
composite ONLY that zone back (rest bit-identical).
Base: genr141_s7 (hair restored). DPM-26, LoRA 0.8,
strength 0.55. 2 seeds.
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

PROMPT = (f"{TRIGGER}, blush, cheek blush, diagonal "
          "hatching, pencil sketch, soft shading, "
          "monochrome, white background")
NEG = ("no blush, smooth skin, color, colored, "
       "lowres, blurry, watermark")

# left cheek zone (viewer-left = his right cheek)
ZONE = (170, 235, 285, 325)     # x0, x1, y0, y1


def _mask(shape):
    H, W = shape
    x0, x1, y0, y1 = ZONE
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, ((x0 + x1) // 2, (y0 + y1) // 2),
                ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360,
                255, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def _composite(base, result, mask):
    a = mask.astype(np.float32) / 255.0
    out = base.astype(np.float32) * (1 - a) \
        + result.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _guide(base):
    g = _lineart(base)
    x0, x1, y0, y1 = ZONE
    g[y0:y1, x0:x1] = 255
    # diagonal hatch mirroring the right cheek's pattern
    for k in range(7):
        x = x0 + 18 + k * 13
        cv2.line(g, (x, y0 + 12), (x - 10, y1 - 10), 130, 1,
                 cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr141_s7.png"),
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
    pipe.set_ip_adapter_scale(0.7)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    mask = _mask((H, W))
    guide = _guide(base)
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=Image.fromarray(base).convert("RGB"),
                   mask_image=Image.fromarray(mask)
                   .convert("RGB"),
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.85,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.55,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        final = _composite(base, g, mask)
        name = f"genr142_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), final)
        unchanged = (final[mask == 0] == base[mask == 0]).mean()
        print(f"[r142 {name}] {time.time() - t0:.0f}s "
              f"unchanged_outside={unchanged:.3f}", flush=True)

    panels = [("base", base)]
    for seed in (7, 42):
        im = cv2.imread(os.path.join(OUT, f"genr142_s{seed}.png"), 0)
        panels.append((f"s{seed}", im))
    n = len(panels)
    cz = [cv2.resize(im[220:360, 130:390], None, fx=2.4, fy=2.4,
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
    cv2.imwrite(os.path.join(OUT, "_genr142_blush.png"), sheet)
    print("[r142] sheets saved", flush=True)


if __name__ == "__main__":
    main()
