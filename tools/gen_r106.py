"""R106 (user: 领子应该降低露出一些脖子, 参照原图高度和
比例; R105: torso slimmed but collar STILL a neck-roll):

Root cause of the eternal neck-roll: in the 40px neck space
the model's 'hood down' prior always fills it with a rolled
collar. Every guide that includes ANY hood-mound arc near
the neck gets merged into the roll.

R106 radical-minimal guide: the neck area is COMPLETELY
BARE (whitened in init AND guide, zero hood lines within
30px of the neck); the collar exists ONLY as an open V at
the chest:
- two open-collar edges from the neck base (y436) down-out
  to the shoulders (x150/365 at y480)
- the hood mass hinted ONLY at the nape OUTSIDE those
  edges (two short lines at x150-190/330-370, y465-480)
- ring pull y472, zipper from y478, slim seams, straps,
  folds (same as R105 which slimmed the torso fine)
CN 1.0, DPM-26, strength 0.80, 'open collar, bare neck'
prompt, 'rolled collar, scarf, turtleneck' negatives.
Base: best of R105 (slim torso) or fallback genr104_c7_m.
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

PROMPT = (f"{TRIGGER}, 1boy, zip-up hoodie, open collar, "
          "bare neck, hood down behind shoulders, metal "
          "zipper, ring zipper pull, backpack straps, "
          "slim fit, fabric folds, subtle grey shading, "
          "monochrome, pencil sketch, hand drawn, white "
          "background")
NEG = ("rolled collar, scarf, turtleneck, high collar, "
       "hood up, cloak, cape, drawstrings, pullover, "
       "color, colored, photo, lowres, blurry, watermark")


def _guide(base):
    g = _lineart(base)
    g[398:660, :] = 255
    # bare neck sides ONLY (no hood lines near the neck)
    cv2.line(g, (CX - 22, 394), (CX - 23, 434), 95, 1,
             cv2.LINE_AA)
    cv2.line(g, (CX + 22, 394), (CX + 23, 434), 95, 1,
             cv2.LINE_AA)
    # open-collar V edges: neck base -> out to shoulders
    cv2.line(g, (CX - 24, 436), (150, 480), 70, 2, cv2.LINE_AA)
    cv2.line(g, (CX + 24, 436), (365, 480), 70, 2, cv2.LINE_AA)
    # hood mass hinted ONLY outside the edges, at the nape
    cv2.line(g, (150, 480), (190, 470), 85, 2, cv2.LINE_AA)
    cv2.line(g, (365, 480), (325, 470), 85, 2, cv2.LINE_AA)
    # slim side seams (R105 torso was approved)
    cv2.line(g, (150, 480), (115, 656), 70, 2, cv2.LINE_AA)
    cv2.line(g, (365, 480), (400, 656), 70, 2, cv2.LINE_AA)
    # ring pull + zipper
    cv2.circle(g, (CX, 472), 6, 45, 2, cv2.LINE_AA)
    cv2.rectangle(g, (CX - 3, 460), (CX + 3, 467), 45, 1)
    cv2.line(g, (CX, 478), (CX, 656), 55, 2, cv2.LINE_AA)
    for y in range(484, 650, 9):
        cv2.line(g, (CX - 4, y + 2), (CX + 4, y - 2), 95, 1,
                 cv2.LINE_AA)
    # straps
    for sgn in (-1, 1):
        xs = CX + sgn * 96
        for off in (-9, 9):
            cv2.line(g, (xs + off, 470),
                     (xs + sgn * 36 + off, 656), 55, 2,
                     cv2.LINE_AA)
    # light folds
    folds = [
        [(185, 495), (205, 548), (218, 648)],
        [(330, 495), (310, 548), (297, 648)],
        [(233, 500), (229, 560), (227, 648)],
        [(283, 500), (287, 560), (289, 648)],
    ]
    for pts in folds:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 125, 1, cv2.LINE_AA)
    return g


def _mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[398:, :] = 255
    cv2.ellipse(m, (CX, 398), (64, 18), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def main():
    base_name = sys.argv[1] if len(sys.argv) > 1 else \
        "genr105_s7.png"
    base = cv2.imread(os.path.join(OUT, base_name), 0)
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
    _scale_lora(pipe.unet, 1.0)
    pipe.set_ip_adapter_scale(0.9)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    guide = _guide(base)
    cv2.imwrite(os.path.join(OUT, "_r106_guide.png"), guide)
    mask = _mask((H, W))
    init = base.copy()
    init[398:, :] = 255
    panels = [("base", base)]
    for seed in (7, 42, 998):
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=Image.fromarray(init).convert("RGB"),
                   mask_image=Image.fromarray(mask).convert("RGB"),
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.80,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr106_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"s{seed}", g))
        print(f"[r106 {name}] {time.time() - t0:.0f}s",
              flush=True)

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr106_sheet.png"), sheet)
    print("[r106] sheets saved", flush=True)


if __name__ == "__main__":
    main()
