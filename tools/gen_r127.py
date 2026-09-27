"""R127 (user's one-at-a-time #1: 领子中有白色横条 = the
empty white band in the collar V, collateral of R126B's
over-erased minimal chest):

Collar-V soft-fold pass on genr126b_f42_c: add a few LIGHT
fold lines radiating from the collar V (value 130, thin,
like the original's soft hoodie folds) so the V region is
fabric, not a white hole. Small mask (V region only),
strength 0.60, CN 0.85, LoRA 0.8, DPM-26. 2 seeds.
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

PROMPT = (f"{TRIGGER}, zip-up hoodie collar, subtle "
          "fabric folds, soft shading, clear collar, "
          "metal zipper pull, monochrome, pencil "
          "sketch, white background")
NEG = ("empty white, blank, smudge, heavy lines, "
       "scarf, turtleneck, color, colored, lowres, "
       "blurry, watermark")


def _mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CX, 452), (105, 30), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def _guide(base):
    g = _lineart(base)
    # soft fold lines radiating from the collar V bottom
    folds = [
        [(CX - 10, 466), (CX - 60, 478), (CX - 120, 492)],
        [(CX + 10, 466), (CX + 60, 478), (CX + 120, 492)],
        [(CX - 22, 458), (CX - 82, 470), (CX - 150, 484)],
        [(CX + 22, 458), (CX + 82, 470), (CX + 150, 484)],
    ]
    for pts in folds:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 130, 1, cv2.LINE_AA)
    # light shading under the collar V
    for k in range(4):
        x0 = CX - 50 + k * 12
        cv2.line(g, (x0, 470), (x0 - 8, 482), 155, 1,
                 cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr126b_f42_c.png"),
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
    panels = [("base", base)]
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
                   guidance_scale=3.5, strength=0.60,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr127_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"s{seed}", g))
        print(f"[r127 {name}] {time.time() - t0:.0f}s",
              flush=True)

    n = len(panels)
    cz = [cv2.resize(im[400:500, 100:420], None, fx=2.4, fy=2.4,
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
    cv2.imwrite(os.path.join(OUT, "_genr127_collar.png"), csheet)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr127_sheet.png"), sheet)
    print("[r127] sheets saved", flush=True)


if __name__ == "__main__":
    main()
