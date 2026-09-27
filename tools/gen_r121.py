"""R121: FINAL fix list (user: 修完这几个问题就可以了):
1. mouth level (corners +2px up, kill the ∩)
2. gaze calm and LEVEL, not looking up (iris low in the
   eye, lid covers half)
3. straps SYMMETRIC: both sides seeded DARK (value 120)
   in the init + two vertical straps in the guide
4. collar crisp: clean dark V edges, no white blur zone

Base: genr120b_s42 (dual-threshold-passing, mostly-vertical
straps). Two passes (face / torso), DPM-26.
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

FACE_PROMPT = (f"{TRIGGER}, 1boy face, calm relaxed gaze, "
               "looking straight, half-closed eyes, "
               "narrow eyes, single eyelid, level "
               "mouth, neutral expression, thin lips, "
               "pencil sketch, soft shading, "
               "monochrome, white background")
FACE_NEG = ("looking up, big shiny eyes, excited, "
            "round eyes, frown, sad, downturned "
            "mouth, smile, open mouth, color, "
            "colored, lowres, blurry, watermark")

TORSO_PROMPT = (f"{TRIGGER}, zip-up hoodie, two black "
                "backpack shoulder straps, vertical "
                "straps, clear collar, crisp lines, "
                "metal zipper, fabric folds, "
                "monochrome, pencil sketch, soft "
                "shading, white background")
TORSO_NEG = ("diagonal strap, crossbody, one strap, "
             "asymmetric, blurry collar, white "
             "smudge, drawstrings, color, colored, "
             "lowres, blurry, watermark")


def _face_guide(base):
    g = _lineart(base)
    g[246:340, 150:366] = 255
    # calm eyes: iris LOW in the eye, lid covers ~50%
    for sgn in (-1, 1):
        xs = [CX + sgn * d for d in range(28, 89, 10)]
        ys = [253, 252, 251, 251, 252, 253, 253]
        ths = [2, 2, 2, 2, 1, 1]
        for i in range(len(xs) - 1):
            cv2.line(g, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                     90, ths[i], cv2.LINE_AA)
    for cx in (196, 320):
        cv2.circle(g, (cx, 280), 5, 100, -1, cv2.LINE_AA)
        cv2.rectangle(g, (cx - 34, 260), (cx + 34, 275), 255,
                      -1)
        up = [(cx - 32, 276), (cx - 14, 273), (cx + 14, 273),
              (cx + 32, 276)]
        for i in range(3):
            cv2.line(g, up[i], up[i + 1], 45, 3, cv2.LINE_AA)
        lo = [(cx - 32, 276), (cx - 14, 282), (cx + 14, 282),
              (cx + 32, 276)]
        for i in range(3):
            cv2.line(g, lo[i], lo[i + 1], 95, 1, cv2.LINE_AA)
    # nose (keep)
    cv2.ellipse(g, (CX - 7, 308), (2, 1), 0, 0, 360, 120, 1,
                cv2.LINE_AA)
    cv2.ellipse(g, (CX + 7, 308), (2, 1), 0, 0, 360, 120, 1,
                cv2.LINE_AA)
    cv2.line(g, (CX - 2, 292), (CX - 3, 303), 150, 1,
             cv2.LINE_AA)
    # LEVEL mouth: corners +2px up
    pts = [(CX - 16, 319), (CX, 321), (CX + 16, 319)]
    cv2.line(g, pts[0], pts[1], 60, 2, cv2.LINE_AA)
    cv2.line(g, pts[1], pts[2], 60, 2, cv2.LINE_AA)
    cv2.ellipse(g, (CX, 327), (10, 4), 0, 25, 155, 170, 1,
                cv2.LINE_AA)
    return g


def _torso_init(base):
    """seed BOTH strap bands dark"""
    init = base.astype(np.float32).copy()
    band = np.zeros_like(init)
    for xs in (140, 375):
        cv2.line(band, (xs, 450), (xs, 660), 255, 26,
                 cv2.LINE_AA)
    a = cv2.GaussianBlur(band, (0, 0), 5) / 255.0
    dark = init * (1 - a) + 120 * a
    return np.minimum(init, dark).astype(np.uint8)


def _torso_guide(base):
    g = _lineart(base)
    # crisp collar V edges (dark)
    cv2.line(g, (CX - 30, 432), (CX - 6, 466), 45, 2,
             cv2.LINE_AA)
    cv2.line(g, (CX + 30, 432), (CX + 6, 466), 45, 2,
             cv2.LINE_AA)
    cv2.circle(g, (CX, 472), 6, 40, 2, cv2.LINE_AA)
    cv2.rectangle(g, (CX - 3, 460), (CX + 3, 467), 40, 1)
    cv2.line(g, (CX, 478), (CX, 656), 50, 2, cv2.LINE_AA)
    # two vertical straps (dark edges)
    for xs in (140, 375):
        for off in (-10, 10):
            cv2.line(g, (xs + off, 450), (xs + off, 656), 45,
                     2, cv2.LINE_AA)
        cv2.rectangle(g, (xs - 7, 520), (xs + 7, 530), 45, 1)
    return g


def _face_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CX, 295), (110, 60), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 10)


def _torso_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[425:, :] = 255
    cv2.ellipse(m, (CX, 425), (64, 18), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def main():
    base = cv2.imread(os.path.join(OUT, "genr120b_s42.png"),
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

    # ---- A: face (level gaze + level mouth) ----
    fguide = _face_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r121_fguide.png"), fguide)
    fmask = _face_mask((H, W))
    finit = base.copy()
    finit[246:340, 150:366] = 255
    faces = []
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=FACE_PROMPT, negative_prompt=FACE_NEG,
                   image=Image.fromarray(finit).convert("RGB"),
                   mask_image=Image.fromarray(fmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(fguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.95,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.75,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr121_f{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        faces.append((seed, g))
        print(f"[r121 {name}] {time.time() - t0:.0f}s",
              flush=True)

    # ---- B: torso (symmetric straps + crisp collar) ----
    tmask = _torso_mask((H, W))
    panels = [("base", base)]
    for seed, fimg in faces:
        t0 = time.time()
        res = pipe(prompt=TORSO_PROMPT,
                   negative_prompt=TORSO_NEG,
                   image=Image.fromarray(_torso_init(fimg))
                   .convert("RGB"),
                   mask_image=Image.fromarray(tmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(
                       _torso_guide(fimg)).convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.65,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr121_f{seed}_t.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        print(f"[r121 {name}] {time.time() - t0:.0f}s",
              flush=True)
        panels.append((f"f{seed}", fimg))
        panels.append((f"f{seed}t", g))

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr121_sheet.png"), sheet)
    ez = [cv2.resize(im[250:350, 150:370], None, fx=2.6, fy=2.6,
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
    cv2.imwrite(os.path.join(OUT, "_genr121_eyes.png"), esheet)
    print("[r121] sheets saved", flush=True)


if __name__ == "__main__":
    main()
