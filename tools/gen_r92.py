"""R92 (user: 脸的长度也不对/看原图各部分比例/直接提取原图
的掩码):

Feature-position map extracted from the original (fractions of
brow->chin): eye 0.25 / nose 0.50 / mouth 0.55 / below-mouth
vs above-mouth 0.81. Classmate: 0.17 / 0.42 / 0.47 / 0.85.
OURS (R90): 0.37 / 0.62 / 0.79 / 0.27 - features crammed LOW,
chin stubby (28px vs original's 103px): baby proportions.

R92 corrects the geometry (brow 250, chin 388, span 138):
- eyes up 11px (center y273)
- nose up 10px (nostrils y308)
- mouth up 25px (lip line y315)
- chin extended down 20px (y388) with a longer slimmer jaw
- below/above ratio 0.27 -> ~0.68 (much closer to 0.81)
Mask extended to cover the new chin. All approved feature
styles kept (monolid eyes / tapered brows / faint nose guide
/ free lips via prompt). 4 seeds on genr90_c7.
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
from gen_r91 import (FACE_PROMPT, FACE_NEG, FACE_SEEDS,
                     FACE_STRENGTH, CN_SCALE, GUIDANCE,
                     _face_mask, _load)  # noqa: E402

MASK_ELL = (258, 310, 116, 105)


def _face_guide(base):
    g = _lineart(base)
    y0, y1 = 240, 395
    right = g[y0:y1, CX:CX + 112]
    g[y0:y1, CX - 112:CX] = right[:, ::-1]
    g[240:290, 150:366] = 255     # brows + eyes
    g[285:395, 200:316] = 255     # nose + mouth + jaw + chin
    # LONGER SLIMMER jaw down to the extended chin (y388)
    jaw = [(206, 292), (212, 322), (222, 350), (236, 372),
           (248, 384), (258, 388)]
    for sgn in (-1, 1):
        pts = [(CX + sgn * (CX - x), y) for x, y in jaw]
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 60, 2, cv2.LINE_AA)
    # tapered flat brows (up 3px)
    for sgn in (-1, 1):
        xs = [CX + sgn * d for d in range(28, 89, 10)]
        ys = [253, 252, 251, 251, 252, 253, 253]
        ths = [2, 2, 2, 2, 1, 1]
        for i in range(len(xs) - 1):
            cv2.line(g, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                     90, ths[i], cv2.LINE_AA)
    # closed almond monolid eyes (up 11px: center 273)
    for cx in (196, 320):
        cv2.circle(g, (cx, 276), 5, 100, -1, cv2.LINE_AA)
        cv2.rectangle(g, (cx - 32, 255), (cx + 32, 271), 255,
                      -1)
        up = [(cx - 31, 273), (cx - 12, 270), (cx + 12, 270),
              (cx + 31, 273)]
        for i in range(3):
            cv2.line(g, up[i], up[i + 1], 45, 3, cv2.LINE_AA)
        lo = [(cx - 31, 273), (cx - 12, 280), (cx + 12, 280),
              (cx + 31, 273)]
        for i in range(3):
            cv2.line(g, lo[i], lo[i + 1], 90, 1, cv2.LINE_AA)
    # nose (up 10px)
    cv2.ellipse(g, (CX - 7, 308), (2, 1), 0, 0, 360, 120, 1,
                cv2.LINE_AA)
    cv2.ellipse(g, (CX + 7, 308), (2, 1), 0, 0, 360, 120, 1,
                cv2.LINE_AA)
    cv2.line(g, (CX - 2, 286), (CX - 3, 300), 165, 1,
             cv2.LINE_AA)
    # mouth (up 25px: line at 315)
    cv2.line(g, (CX - 14, 315), (CX + 14, 315), 65, 1,
             cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr90_c7.png"), 0)
    assert base is not None
    H, W = base.shape
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[170:300, 100:320])]

    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    pipe = _load(lcm, ckpt, "lllyasviel/control_v11p_sd15_lineart")
    _scale_lora(pipe.unet, 0.8)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)
    guide = _face_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r92_guide.png"), guide)
    H2, W2 = H, W
    m = np.zeros((H2, W2), np.uint8)
    cx, cy, ax, ay = MASK_ELL
    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(m, (0, 0), 10)
    init = base.copy()
    cv2.ellipse(init, (cx, cy), (ax - 6, ay - 6), 0, 0, 360,
                255, -1)
    bpil = Image.fromarray(init).convert("RGB")
    mpil = Image.fromarray(mask).convert("RGB")
    ctrl = Image.fromarray(guide).convert("RGB")

    cands = [("base", base)]
    for seed in FACE_SEEDS:
        t0 = time.time()
        res = pipe(prompt=FACE_PROMPT, negative_prompt=FACE_NEG,
                   image=bpil, mask_image=mpil,
                   control_image=ctrl,
                   controlnet_conditioning_scale=CN_SCALE,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8,
                   guidance_scale=GUIDANCE,
                   strength=FACE_STRENGTH,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr92_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cands.append((f"s{seed}", g))
        print(f"[r92 {name}] {time.time() - t0:.0f}s", flush=True)

    n = len(cands)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in cands:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr92_sheet.png"), sheet)
    crops = [im[230:520, 110:410] for _, im in cands]
    ch, cw = crops[0].shape
    zs = np.full((ch, cw * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), c in zip(cands, crops):
        p = c.copy()
        cv2.putText(p, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, 0, 2, cv2.LINE_AA)
        zs[:, x:x + cw] = p
        x += cw + 10
    cv2.imwrite(os.path.join(OUT, "_genr92_faces.png"), zs)
    print("[r92] sheets saved", flush=True)


if __name__ == "__main__":
    main()
