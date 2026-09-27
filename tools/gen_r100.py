"""R100 (user on R99: 脸太尖了/耳朵画的不好/衣服上的褶皱呢):

Verified in _r100_check.png:
- chin = sharp heavy-outlined V; original = soft rounded jaw
  curve with a LIGHT line
- ears = angular hook (right) / dark wedge (left); need
  simple soft C-curve ears
- torso folds were SMOOTHED OUT by the R99 collar pass; the
  straps render as flat wide seatbelts

R100 on genr99_s7_z (keep its eyes/zipper/straps):
A. face-zone pass: soft ROUND chin (shallow U, light line) +
   clean soft ears (outer C + inner arc, symmetric)
B. torso pass: R87/R88 fold system back (radiating sweeps +
   hatch valleys + tonal init) WITH the straps and zipper
   drawn into the guide so they survive
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
from gen_r84 import _hatch  # noqa: E402
from gen_r87 import _torso_mask  # noqa: E402
from gen_r88 import _tonal_init  # noqa: E402
from gen_r91 import _load  # noqa: E402
from gen_r98 import CLOTH_PROMPT  # noqa: E402
from gen_r97 import CLOTH_NEG  # noqa: E402
from gen_r99 import FACE_PROMPT, FACE_NEG  # noqa: E402

FACE_SEEDS = [7, 42]
MASK_ELL = (258, 318, 118, 108)


def _face_guide(base):
    g = _lineart(base)
    y0, y1 = 240, 395
    right = g[y0:y1, CX:CX + 112]
    g[y0:y1, CX - 112:CX] = right[:, ::-1]
    g[240:290, 150:366] = 255
    g[285:395, 195:321] = 255
    # SOFT ROUND chin: shallow U, light line (not a V)
    chin = [(212, 300), (218, 330), (228, 356), (240, 376),
            (250, 386)]
    for sgn in (-1, 1):
        pts = [(CX + sgn * (CX - x), y) for x, y in chin]
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 85, 1, cv2.LINE_AA)
    cv2.ellipse(g, (CX, 384), (12, 6), 0, 10, 170, 85, 1,
                cv2.LINE_AA)
    # clean soft ears
    for sgn in (-1, 1):
        ex = CX + sgn * 92
        cv2.ellipse(g, (ex, 288), (13, 21), 0, 300, 170, 85,
                    2, cv2.LINE_AA)
        cv2.ellipse(g, (ex + sgn * 2, 290), (6, 10), 0, 300,
                    120, 125, 1, cv2.LINE_AA)
    # approved features (R99 spec)
    for sgn in (-1, 1):
        xs = [CX + sgn * d for d in range(28, 89, 10)]
        ys = [253, 252, 251, 251, 252, 253, 253]
        ths = [2, 2, 2, 2, 1, 1]
        for i in range(len(xs) - 1):
            cv2.line(g, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                     90, ths[i], cv2.LINE_AA)
    for cx in (196, 320):
        cv2.circle(g, (cx, 276), 5, 100, -1, cv2.LINE_AA)
        cv2.rectangle(g, (cx - 35, 255), (cx + 35, 271), 255,
                      -1)
        up = [(cx - 34, 273), (cx - 14, 270), (cx + 14, 270),
              (cx + 34, 273)]
        for i in range(3):
            cv2.line(g, up[i], up[i + 1], 45, 3, cv2.LINE_AA)
        sgn = -1 if cx < CX else 1
        cv2.line(g, (cx + sgn * 34, 273),
                 (cx + sgn * 39, 271), 45, 2, cv2.LINE_AA)
        lo = [(cx - 34, 273), (cx - 14, 279), (cx + 14, 279),
              (cx + 34, 273)]
        for i in range(3):
            cv2.line(g, lo[i], lo[i + 1], 95, 1, cv2.LINE_AA)
    cv2.ellipse(g, (CX - 7, 308), (2, 1), 0, 0, 360, 120, 1,
                cv2.LINE_AA)
    cv2.ellipse(g, (CX + 7, 308), (2, 1), 0, 0, 360, 120, 1,
                cv2.LINE_AA)
    cv2.line(g, (CX - 2, 286), (CX - 3, 300), 165, 1,
             cv2.LINE_AA)
    pts = [(CX - 14, 314), (CX, 316), (CX + 14, 314)]
    cv2.line(g, pts[0], pts[1], 70, 1, cv2.LINE_AA)
    cv2.line(g, pts[1], pts[2], 70, 1, cv2.LINE_AA)
    cv2.ellipse(g, (CX, 322), (9, 3), 0, 25, 155, 185, 1,
                cv2.LINE_AA)
    for sgn in (-1, 1):
        for k in range(5):
            x0 = CX + sgn * (52 + k * 7)
            y0 = 296 + k * 3
            cv2.line(g, (x0, y0), (x0 - 6, y0 + 10), 150, 1,
                     cv2.LINE_AA)
    return g


def _cloth_guide(base):
    g = _lineart(base)
    # straps (keep, slightly narrower)
    for sgn in (-1, 1):
        xs = CX + sgn * 104
        for off in (-8, 8):
            cv2.line(g, (xs + off, 436),
                     (xs + sgn * 40 + off, 660), 50, 2,
                     cv2.LINE_AA)
    # zipper + pull (keep)
    cv2.circle(g, (CX, 452), 6, 45, 2, cv2.LINE_AA)
    cv2.rectangle(g, (CX - 3, 440), (CX + 3, 447), 45, 1)
    cv2.line(g, (CX, 458), (CX, 656), 55, 2, cv2.LINE_AA)
    for y in range(464, 650, 9):
        cv2.line(g, (CX - 4, y + 2), (CX + 4, y - 2), 95, 1,
                 cv2.LINE_AA)
    # radiating folds (R88 system)
    sweeps = [
        [(158, 460), (190, 512), (214, 578), (226, 648)],
        [(186, 466), (212, 520), (230, 582), (238, 648)],
        [(358, 460), (326, 512), (302, 578), (290, 648)],
        [(330, 466), (304, 520), (286, 582), (278, 648)],
        [(232, 475), (228, 548), (226, 648)],
        [(284, 475), (288, 548), (290, 648)],
    ]
    for pts in sweeps:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 115, 1, cv2.LINE_AA)
    for x0, y0, x1, y1 in ((112, 520, 148, 545), (102, 560, 140,
                           585), (408, 520, 372, 545),
                           (418, 560, 380, 585)):
        cv2.line(g, (x0, y0), (x1, y1), 120, 1, cv2.LINE_AA)
    for cx in (170, 258, 346):
        cv2.ellipse(g, (cx, 660), (46, 26), 0, 200, 340, 120,
                    1, cv2.LINE_AA)
    zones = [
        [(198, 545), (224, 545), (220, 645), (194, 645)],
        [(292, 545), (318, 545), (322, 645), (296, 645)],
    ]
    for poly in zones:
        m = np.zeros_like(g)
        cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
        tmp = np.full_like(g, 255)
        _hatch(tmp, poly, spacing=7, val=140)
        g[m > 0] = tmp[m > 0]
    return g


def _face_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cx, cy, ax, ay = MASK_ELL
    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 10)


def main():
    base = cv2.imread(os.path.join(OUT, "genr99_s7_z.png"), 0)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"), 0)
    assert base is not None and orig is not None
    H, W = base.shape
    origc = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                       cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(origc, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[170:300, 100:320]),
            Image.fromarray(rgb[280:, :])]
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

    # ---- stage A: soft chin + ears ----
    guide = _face_guide(base)
    mask = _face_mask((H, W))
    init = base.copy()
    cx, cy, ax, ay = MASK_ELL
    cv2.ellipse(init, (cx, cy), (ax - 6, ay - 6), 0, 0, 360,
                255, -1)
    bpil = Image.fromarray(init).convert("RGB")
    faces = []
    for seed in FACE_SEEDS:
        t0 = time.time()
        res = pipe(prompt=FACE_PROMPT, negative_prompt=FACE_NEG,
                   image=bpil,
                   mask_image=Image.fromarray(mask).convert("RGB"),
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.95,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8,
                   guidance_scale=3.0, strength=0.85,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr100_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        faces.append((seed, g))
        print(f"[r100 {name}] {time.time() - t0:.0f}s", flush=True)

    # ---- stage B: folds + straps + zipper ----
    _scale_lora(pipe.unet, 1.0 / 0.8)
    tmask = _torso_mask((H, W))
    panels = [("base", base)]
    for seed, fimg in faces:
        tinit = _tonal_init(fimg, orig)
        tguide = _cloth_guide(fimg)
        t0 = time.time()
        res = pipe(prompt=CLOTH_PROMPT, negative_prompt=CLOTH_NEG,
                   image=Image.fromarray(tinit).convert("RGB"),
                   mask_image=Image.fromarray(tmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(tguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.9,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8,
                   guidance_scale=3.0, strength=0.55,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr100_s{seed}_c.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        print(f"[r100 {name}] {time.time() - t0:.0f}s", flush=True)
        panels.append((f"s{seed}", fimg))
        panels.append((f"s{seed}c", g))

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr100_sheet.png"), sheet)
    ez = [cv2.resize(im[255:330, 130:390], None, fx=2.4, fy=2.4,
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
    cv2.imwrite(os.path.join(OUT, "_genr100_ears.png"), esheet)
    cz = [cv2.resize(im[430:660, 40:480], None, fx=1.4, fy=1.4,
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
    cv2.imwrite(os.path.join(OUT, "_genr100_folds.png"), csheet)
    print("[r100] sheets saved", flush=True)


if __name__ == "__main__":
    main()
