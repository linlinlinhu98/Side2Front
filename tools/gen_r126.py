"""R126 (user: 领子还没完全擦掉/嘴唇依然不明显/眼睛还是
看着上面没有平视):

Three final fixes on genr125_s7 (2 straps + thin zipper
already in place):
A. FACE: level gaze (iris in the LOWER half of the eye,
   upper lid covers 50%) + FULL lip set (upper-lip
   outline at value 110 - drawn solid this time, straight
   lip line, lower-lip outline) like the reference.
B. COLLAR: erase the whole zone (init + guide), redraw
   ONLY crisp collar V lines + ring pull + thin hood
   edges - zero fuzzy remnants.

DPM-26, CN 1.0, 2+2 seeds.
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

FACE_PROMPT = (f"{TRIGGER}, 1boy face, level gaze, "
               "looking straight ahead, calm eyes, "
               "half-closed eyes, single eyelid, "
               "defined lips, upper lip, lower lip, "
               "thin lips, neutral mouth, pencil "
               "sketch, soft shading, monochrome, "
               "white background")
FACE_NEG = ("looking up, upturned eyes, big shiny "
            "eyes, excited, round eyes, curved "
            "mouth, smile, frown, thick lips, open "
            "mouth, color, colored, lowres, blurry, "
            "watermark")

COLLAR_PROMPT = (f"{TRIGGER}, zip-up hoodie collar, "
                 "crisp clean lines, clear collar, "
                 "metal zipper pull, no shading, "
                 "monochrome, pencil sketch, white "
                 "background")
COLLAR_NEG = ("fuzzy, blurry, smudge, gray marks, "
              "messy lines, scarf, turtleneck, "
              "color, colored, lowres, watermark")


def _face_guide(base):
    g = _lineart(base)
    g[246:342, 150:366] = 255
    for sgn in (-1, 1):
        xs = [CX + sgn * d for d in range(28, 89, 10)]
        ys = [253, 252, 251, 251, 252, 253, 253]
        ths = [2, 2, 2, 2, 1, 1]
        for i in range(len(xs) - 1):
            cv2.line(g, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                     90, ths[i], cv2.LINE_AA)
    # LEVEL gaze: iris in the LOWER half of the eye
    for cx in (196, 320):
        cv2.circle(g, (cx, 281), 5, 100, -1, cv2.LINE_AA)
        cv2.rectangle(g, (cx - 34, 258), (cx + 34, 276), 255,
                      -1)
        up = [(cx - 32, 277), (cx - 14, 274), (cx + 14, 274),
              (cx + 32, 277)]
        for i in range(3):
            cv2.line(g, up[i], up[i + 1], 45, 3, cv2.LINE_AA)
        lo = [(cx - 32, 277), (cx - 14, 284), (cx + 14, 284),
              (cx + 32, 277)]
        for i in range(3):
            cv2.line(g, lo[i], lo[i + 1], 95, 1, cv2.LINE_AA)
    # nose (keep)
    cv2.ellipse(g, (CX - 7, 308), (2, 1), 0, 0, 360, 120, 1,
                cv2.LINE_AA)
    cv2.ellipse(g, (CX + 7, 308), (2, 1), 0, 0, 360, 120, 1,
                cv2.LINE_AA)
    cv2.line(g, (CX - 2, 292), (CX - 3, 303), 150, 1,
             cv2.LINE_AA)
    # FULL lip set: upper-lip outline (solid, 110) +
    # straight lip line + lower-lip outline
    pts = [(CX - 16, 316), (CX - 6, 314), (CX, 315),
           (CX + 6, 314), (CX + 16, 316)]
    for i in range(len(pts) - 1):
        cv2.line(g, pts[i], pts[i + 1], 110, 1, cv2.LINE_AA)
    cv2.line(g, (CX - 20, 321), (CX + 20, 321), 55, 2,
             cv2.LINE_AA)
    cv2.ellipse(g, (CX, 329), (11, 5), 0, 25, 155, 150, 1,
                cv2.LINE_AA)
    return g


def _face_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CX, 297), (110, 62), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 10)


def _collar_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[415:485, 150:370] = 255
    return cv2.GaussianBlur(m, (0, 0), 8)


def _collar_guide(base):
    g = _lineart(base)
    g[415:485, 150:370] = 255
    # ONLY crisp collar V + pull + thin hood edges
    cv2.line(g, (CX - 30, 432), (CX - 6, 466), 50, 2,
             cv2.LINE_AA)
    cv2.line(g, (CX + 30, 432), (CX + 6, 466), 50, 2,
             cv2.LINE_AA)
    cv2.circle(g, (CX, 472), 6, 40, 2, cv2.LINE_AA)
    cv2.rectangle(g, (CX - 3, 460), (CX + 3, 467), 40, 1)
    cv2.line(g, (CX - 30, 432), (160, 462), 90, 1, cv2.LINE_AA)
    cv2.line(g, (CX + 30, 432), (356, 462), 90, 1, cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr125_s7.png"), 0)
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

    # ---- A: face (level gaze + full lips) ----
    fguide = _face_guide(base)
    fmask = _face_mask((H, W))
    finit = base.copy()
    finit[246:342, 150:366] = 255
    faces = []
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=FACE_PROMPT, negative_prompt=FACE_NEG,
                   image=Image.fromarray(finit).convert("RGB"),
                   mask_image=Image.fromarray(fmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(fguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.75,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr126_f{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        faces.append((seed, g))
        print(f"[r126 {name}] {time.time() - t0:.0f}s",
              flush=True)

    # ---- B: collar erase-and-clean on each ----
    cmask = _collar_mask((H, W))
    cguide = _collar_guide(base)
    panels = [("base", base)]
    for seed, fimg in faces:
        cinit = fimg.copy()
        cinit[415:485, 150:370] = 255
        t0 = time.time()
        res = pipe(prompt=COLLAR_PROMPT,
                   negative_prompt=COLLAR_NEG,
                   image=Image.fromarray(cinit).convert("RGB"),
                   mask_image=Image.fromarray(cmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(cguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.70,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr126_f{seed}_c.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        print(f"[r126 {name}] {time.time() - t0:.0f}s",
              flush=True)
        panels.append((f"f{seed}", fimg))
        panels.append((f"f{seed}c", g))

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr126_sheet.png"), sheet)
    ez = [cv2.resize(im[255:300, 155:365], None, fx=3.0, fy=3.0,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    eh, ew = ez[0].shape
    esheet = np.full((eh, ew * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, ez):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        esheet[:, x:x + ew] = t
        x += ew + 8
    cv2.imwrite(os.path.join(OUT, "_genr126_gaze.png"), esheet)
    mz = [cv2.resize(im[305:340, 215:305], None, fx=8, fy=8,
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
    cv2.imwrite(os.path.join(OUT, "_genr126_mouths.png"), msheet)
    cz = [cv2.resize(im[410:500, 130:390], None, fx=2.6, fy=2.6,
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
    cv2.imwrite(os.path.join(OUT, "_genr126_collars.png"),
                csheet)
    print("[r126] sheets saved", flush=True)


if __name__ == "__main__":
    main()
