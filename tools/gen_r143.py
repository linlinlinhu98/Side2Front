"""R143 (fix #3 per user: 嘴部不要下弯, 要么就把嘴唇搞得
明显一点):

R133's proven structure (upper-lip M shading + thick
STRAIGHT level line) on genr142_s42, mouth zone only,
NO whiten (edit existing), composite back only the mouth
zone (everything else bit-identical, verified).
DPM-26, LoRA 0.3, CN 1.0, strength 0.60. 2 seeds.
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

MOUTH_PROMPT = (f"{TRIGGER}, closed lips, defined lips, "
                "upper lip, straight level lip line, "
                "natural lips, neutral mouth, pencil "
                "sketch, soft shading, monochrome, "
                "white background")
MOUTH_NEG = ("curved mouth, arc mouth, frown, sad, "
             "downturned corners, open mouth, thin "
             "lips, line mouth, color, colored, "
             "lowres, blurry, watermark")

ZONE = (205, 315, 298, 342)     # x0, x1, y0, y1


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
    cx = (x0 + x1) // 2
    g[y0:y1, x0:x1] = 255
    # upper-lip M hint (light)
    pts = [(cx - 15, 317), (cx - 6, 315), (cx, 316),
           (cx + 6, 315), (cx + 15, 317)]
    for i in range(len(pts) - 1):
        cv2.line(g, pts[i], pts[i + 1], 140, 1, cv2.LINE_AA)
    # THICK STRAIGHT level line
    cv2.line(g, (cx - 24, 321), (cx + 24, 321), 45, 3,
             cv2.LINE_AA)
    # faint lower-lip volume
    cv2.ellipse(g, (cx, 328), (10, 4), 0, 25, 155, 165, 1,
                cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr142_s42.png"),
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
    _scale_lora(pipe.unet, 0.3)
    pipe.set_ip_adapter_scale(0.8)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    mask = _mask((H, W))
    guide = _guide(base)
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=MOUTH_PROMPT,
                   negative_prompt=MOUTH_NEG,
                   image=Image.fromarray(base).convert("RGB"),
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
        name = f"genr143_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), final)
        unchanged = (final[mask == 0] == base[mask == 0]).mean()
        print(f"[r143 {name}] {time.time() - t0:.0f}s "
              f"unchanged_outside={unchanged:.3f}", flush=True)

    panels = [("base", base)]
    for seed in (7, 42):
        im = cv2.imread(os.path.join(OUT, f"genr143_s{seed}.png"), 0)
        panels.append((f"s{seed}", im))
    n = len(panels)
    mz = [cv2.resize(im[295:345, 205:315], None, fx=7, fy=7,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    mh, mw = mz[0].shape
    sheet = np.full((mh, mw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, mz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + mw] = t
        x += mw + 8
    cv2.imwrite(os.path.join(OUT, "_genr143_mouths.png"), sheet)
    print("[r143] sheets saved", flush=True)


if __name__ == "__main__":
    main()
