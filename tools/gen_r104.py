"""R104 (user: 领子哪里降了/颜色和嘴还是不像/有没有把原图
所有数据传给模型):

Collar truth (_collar_truth.png): R103 overshot DOWN
(y470-490 = a chest stripe, 80-100px below chin) and left
side fragments at y390-430. Original: collar/hood mass top
edge 40-60px below the chin (y430-450), bulk at the neck
base sides, V opens ~y450, pull ~y470.

A. COLLAR REPOSITION: eradication extended to y388-465 (kill
   side fragments), then draw: hood bulk humps at the neck
   base SIDES (y420-460, x120-200 and x316-392), V opening
   from y450, ring pull at y470, zipper from y476.
B. MOUTH (tonal, classmate spec): soft full lower-lip
   shadow blob (larger, softer, 218) blended into the init
   + clean-mouth prompt (suppresses smudge) + faint
   lip-line guide.

Base: genr103_h7_c (black hair approved). DPM-26.
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
from gen_r102 import COLLAR_PROMPT, COLLAR_NEG, STEPS  # noqa: E402
from gen_r93 import _mouth_mask  # noqa: E402

MOUTH_PROMPT = (f"{TRIGGER}, closed lips, neutral mouth, "
                "soft lower lip shading, small mouth, "
                "pencil sketch, soft shading, monochrome, "
                "white background")
MOUTH_NEG = ("frown, sad, downturned mouth, open mouth, "
             "smile, smirk, line mouth, upper lip shadow, "
             "philtrum, color, colored, lowres, blurry, "
             "watermark")


def _collar_guide(base):
    g = _lineart(base)
    g[388:466, :] = 255          # eradicate incl. sides
    # hood bulk humps at the neck base SIDES
    for sgn in (-1, 1):
        pts = [(CX + sgn * 30, 436), (CX + sgn * 66, 446),
               (CX + sgn * 96, 460)]
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 80, 2, cv2.LINE_AA)
        pts2 = [(x, y + 10) for x, y in pts]
        for i in range(len(pts2) - 1):
            cv2.line(g, pts2[i], pts2[i + 1], 55, 2,
                     cv2.LINE_AA)
    # hood mass arc behind the neck
    cv2.ellipse(g, (CX, 434), (78, 24), 0, 190, 350, 90, 1,
                cv2.LINE_AA)
    # V opening from y450 + ring pull at y470
    cv2.line(g, (CX - 26, 452), (CX - 6, 466), 70, 2,
             cv2.LINE_AA)
    cv2.line(g, (CX + 26, 452), (CX + 6, 466), 70, 2,
             cv2.LINE_AA)
    cv2.circle(g, (CX, 474), 6, 45, 2, cv2.LINE_AA)
    cv2.rectangle(g, (CX - 3, 462), (CX + 3, 469), 45, 1)
    cv2.line(g, (CX, 480), (CX, 656), 55, 2, cv2.LINE_AA)
    return g


def _collar_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[388:500, :] = 255
    cv2.ellipse(m, (CX, 388), (62, 16), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def _mouth_init(base):
    init = base.copy()
    init[300:338, 218:300] = 255
    blob = np.zeros_like(init)
    cv2.ellipse(blob, (CX, 328), (15, 7), 0, 0, 360, 255, -1)
    a = cv2.GaussianBlur(blob, (0, 0), 6).astype(np.float32) / 255
    init = (init * (1 - a) + 218 * a).astype(np.uint8)
    return init


def _mouth_guide(base):
    g = _lineart(base)
    g[306:340, 216:302] = 255
    cv2.line(g, (CX - 14, 321), (CX + 14, 321), 75, 1,
             cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr103_h7_c.png"),
                      0)
    assert base is not None
    H, W = base.shape
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[280:, :])]
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

    # ---- A: collar reposition ----
    cguide = _collar_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r104_cguide.png"), cguide)
    cmask = _collar_mask((H, W))
    collars = []
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=COLLAR_PROMPT,
                   negative_prompt=COLLAR_NEG,
                   image=Image.fromarray(base).convert("RGB"),
                   mask_image=Image.fromarray(cmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(cguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.85,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr104_c{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        collars.append((seed, g))
        print(f"[r104 {name}] {time.time() - t0:.0f}s",
              flush=True)

    # ---- B: tonal mouth on each ----
    mmask = _mouth_mask((H, W))
    panels = [("base", base)]
    for seed, cimg in collars:
        minit = _mouth_init(cimg)
        mguide = _mouth_guide(cimg)
        t0 = time.time()
        res = pipe(prompt=MOUTH_PROMPT, negative_prompt=MOUTH_NEG,
                   image=Image.fromarray(minit).convert("RGB"),
                   mask_image=Image.fromarray(mmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(mguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.8,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.65,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr104_c{seed}_m.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        print(f"[r104 {name}] {time.time() - t0:.0f}s",
              flush=True)
        panels.append((f"c{seed}", cimg))
        panels.append((f"c{seed}m", g))

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr104_sheet.png"), sheet)
    cz = [cv2.resize(im[380:530, 110:410], None, fx=1.8, fy=1.8,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    ch, cw = cz[0].shape
    csheet = np.full((ch, cw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, cz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        csheet[:, x:x + cw] = t
        x += cw + 8
    cv2.imwrite(os.path.join(OUT, "_genr104_collars.png"),
                csheet)
    mz = [cv2.resize(im[300:344, 214:304], None, fx=6, fy=6,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    mh, mw = mz[0].shape
    msheet = np.full((mh, mw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, mz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        msheet[:, x:x + mw] = t
        x += mw + 8
    cv2.imwrite(os.path.join(OUT, "_genr104_mouths.png"), msheet)
    print("[r104] sheets saved", flush=True)


if __name__ == "__main__":
    main()
