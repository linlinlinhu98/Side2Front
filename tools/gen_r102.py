"""R102 (user: 领子还是太高/衣服纹理很奇怪/每次都忘记处理
一部分/参考报告建议):

Two-step per the report's route:
A. STRUCTURAL collar fix (masked, DPM-26): neck FULLY bare
   to y460, hood mass tucked BEHIND the neck (small humps at
   the neck base sides, not wrapping the front), deep V
   opening to y490, zipper pull at y495.
B. ONE unifying full-figure img2img (DPM-26, strength 0.38,
   IP 0.68) with the REPORT'S VERBATIM prompt/negative - the
   model unifies straps/folds/shading/line weight naturally
   (killing the plastic 'sticker bands' from my tonal
   seeding).
Then verify EVERY part against a checklist sheet.
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

COLLAR_PROMPT = (f"{TRIGGER}, zip-up hoodie, deep v "
                 "neckline, low collar, hood behind neck, "
                 "hood resting on shoulders, neck visible, "
                 "metal zipper, ring zipper pull, pencil "
                 "sketch, soft shading, monochrome, white "
                 "background")
COLLAR_NEG = ("high collar, turtleneck, collar covering "
              "neck, snorkel hood, pullover, drawstrings, "
              "color, colored, lowres, blurry, watermark")

# the report's verbatim prompt
UNIFY_PROMPT = ("masterpiece, best quality, sketch drawing, "
                "monochrome, 1boy, teenage boy, thin narrow "
                "face, sharp jawline, small chin, short "
                "messy spiky dark hair, long almond-shaped "
                "eyes, thin lips, neutral calm expression, "
                "hoodie with front zipper, zipper pull "
                "tab, backpack two shoulder straps, hand "
                "drawn, line weight variation, thick "
                "outline for silhouette, thin internal "
                "lines, subtle grey shading, faint blush "
                "on cheek, plain white background, front "
                "view")
UNIFY_NEG = ("chibi, big round eyes, wide fat face, thick "
             "lips, sad pout, cloak, uniform rigid "
             "outlines, repetitive parallel hatching "
             "inside hair, heavy black fill, colored, "
             "3d render, deformed, disfigured")

STEPS = 26


def _collar_guide(base):
    g = _lineart(base)
    g[392:520, 120:396] = 255
    # bare neck to y460
    cv2.line(g, (CX - 22, 392), (CX - 23, 458), 95, 1,
             cv2.LINE_AA)
    cv2.line(g, (CX + 22, 392), (CX + 23, 458), 95, 1,
             cv2.LINE_AA)
    # small hood humps BEHIND the neck base (sides only)
    for sgn in (-1, 1):
        cv2.ellipse(g, (CX + sgn * 66, 468), (42, 20), 0,
                    180, 330, 85, 2, cv2.LINE_AA)
    # deep V opening
    cv2.line(g, (CX - 26, 462), (CX - 6, 496), 70, 2,
             cv2.LINE_AA)
    cv2.line(g, (CX + 26, 462), (CX + 6, 496), 70, 2,
             cv2.LINE_AA)
    # zipper pull at the V bottom
    cv2.circle(g, (CX, 502), 6, 45, 2, cv2.LINE_AA)
    cv2.rectangle(g, (CX - 3, 490), (CX + 3, 497), 45, 1)
    cv2.line(g, (CX, 508), (CX, 656), 55, 2, cv2.LINE_AA)
    return g


def _collar_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[392:525, :] = 255
    cv2.ellipse(m, (CX, 392), (62, 16), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def main():
    base = cv2.imread(os.path.join(OUT, "genr101_final.png"),
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
    pipe.set_ip_adapter_scale(0.68)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    # ---- A: collar structure ----
    cguide = _collar_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r102_cguide.png"), cguide)
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
                   controlnet_conditioning_scale=0.9,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.75,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr102_c{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        collars.append((seed, g))
        print(f"[r102 {name}] {time.time() - t0:.0f}s",
              flush=True)

    # ---- B: ONE unifying full pass (report's prompt) ----
    full_mask = Image.fromarray(
        np.full((H, W), 255, np.uint8)).convert("RGB")
    panels = [("base", base)]
    for seed, cimg in collars:
        t0 = time.time()
        res = pipe(prompt=UNIFY_PROMPT, negative_prompt=UNIFY_NEG,
                   image=Image.fromarray(cimg).convert("RGB"),
                   mask_image=full_mask,
                   control_image=Image.fromarray(
                       _lineart(cimg)).convert("RGB"),
                   controlnet_conditioning_scale=0.7,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.38,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr102_c{seed}_u.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        print(f"[r102 {name}] {time.time() - t0:.0f}s",
              flush=True)
        panels.append((f"c{seed}", cimg))
        panels.append((f"c{seed}u", g))

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr102_sheet.png"), sheet)

    # ---- EVERY-PART checklist sheet (orig-specific crops) ----
    # (label, orig crop y0y1x0x1, ours crop y0y1x0x1)
    parts = [
        ("eyes",  (175, 250, 105, 230), (255, 296, 150, 366)),
        ("brows", (178, 210, 110, 220), (244, 268, 160, 356)),
        ("nose",  (245, 330, 30, 150),  (285, 325, 225, 295)),
        ("mouth", (280, 330, 40, 135),  (305, 335, 220, 300)),
        ("chin",  (330, 410, 45, 180),  (350, 400, 190, 330)),
        ("ears",  (240, 330, 225, 320), (260, 325, 130, 390)),
        ("hair",  (55, 260, 30, 360),   (30, 230, 90, 430)),
        ("collar",(390, 500, 0, 520),   (395, 520, 130, 390)),
        ("zipper",(430, 660, 80, 220),  (440, 560, 220, 300)),
        ("straps",(420, 560, 300, 470), (430, 600, 60, 460)),
        ("folds", (450, 660, 100, 400), (500, 660, 60, 460)),
    ]
    orig_img = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                          0)
    rows = []
    for label, oc, uc in parts:
        tiles = [cv2.resize(orig_img[oc[0]:oc[1], oc[2]:oc[3]],
                            (216, 100),
                            interpolation=cv2.INTER_CUBIC)]
        for _, im in panels:
            tiles.append(cv2.resize(
                im[uc[0]:uc[1], uc[2]:uc[3]], (216, 100),
                interpolation=cv2.INTER_CUBIC))
        r = np.full((100, 216 * len(tiles)
                     + 6 * (len(tiles) - 1)), 255, np.uint8)
        x = 0
        for t in tiles:
            r[:, x:x + 216] = t
            x += 216 + 6
        rows.append((label, r))
    Wc = max(r.shape[1] for _, r in rows)
    chk = np.full((len(rows) * 128, Wc), 255, np.uint8)
    y = 0
    for label, r in rows:
        cv2.putText(chk, label, (6, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2,
                    cv2.LINE_AA)
        chk[y + 26:y + 26 + 100, :r.shape[1]] = r
        y += 128
    cv2.imwrite(os.path.join(OUT, "_genr102_checklist.png"),
                chk)
    print("[r102] sheets saved", flush=True)


if __name__ == "__main__":
    main()
