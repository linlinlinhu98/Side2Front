"""R98 (R97 failed: 'big hair strands' -> solid black helmet;
straps too weak; eye pass smudged):

Corrected approach:
A. STRAPS via tonal init (proven in the clothes round): paint
   two soft dark bands (value 120) into the init + strap edge
   guide lines + 'black backpack straps' prompt. strength 0.5.
B. HAIR: NO repaint-from-scratch. LOW-strength (0.45) EDIT on
   genr96_s7's existing pencil hair - the guide only ADDS big
   clump separation lines over the existing fine hatching
   (init untouched), prompt 'hair clumps, strand separation'.
2+2 seeds on genr96_s7.
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
from gen_r87 import _torso_mask  # noqa: E402
from gen_r91 import _load  # noqa: E402
from gen_r97 import CLOTH_NEG, _hair_mask  # noqa: E402

CLOTH_PROMPT = (f"{TRIGGER}, zip-up hoodie, full-length "
                "zipper, hood down, black backpack straps, "
                "shoulder straps, fabric folds, tonal "
                "shading, parallel hatching, monochrome, "
                "pencil sketch, soft shading, white "
                "background")

HAIR_PROMPT = (f"{TRIGGER}, short hair, hair clumps, hair "
               "strand separation, spiky crown, bangs swept "
               "to the side, pencil sketch hatching, fine "
               "pencil strokes, monochrome, soft shading, "
               "white background")
HAIR_NEG = ("solid black hair, helmet hair, wig, smooth "
            "hair, anime hair, color, colored, photo, "
            "lowres, blurry, watermark")


def _strap_init(base):
    """soft dark strap bands blended into the init"""
    init = base.astype(np.float32).copy()
    band = np.zeros_like(init)
    for sgn in (-1, 1):
        xs = CX + sgn * 104
        pts = np.array([(xs - 10, 436), (xs + 10, 436),
                        (xs + sgn * 40 + 10, 660),
                        (xs + sgn * 40 - 10, 660)], np.int32)
        cv2.fillPoly(band, [pts], 255)
    a = cv2.GaussianBlur(band, (0, 0), 6) / 255.0
    dark = init * (1 - a) + 120 * a
    return np.minimum(init, dark).astype(np.uint8)


def _strap_guide(base):
    g = _lineart(base)
    for sgn in (-1, 1):
        xs = CX + sgn * 104
        for off in (-10, 10):
            cv2.line(g, (xs + off, 436),
                     (xs + sgn * 40 + off, 660), 45, 2,
                     cv2.LINE_AA)
        cv2.rectangle(g, (xs + sgn * 14 - 6, 505),
                      (xs + sgn * 14 + 6, 515), 50, 1)
    cv2.line(g, (CX, 458), (CX, 656), 60, 2, cv2.LINE_AA)
    return g


def _hair_guide(base):
    """existing hair lineart + ADDED clump separation lines"""
    g = _lineart(base)
    seps = [
        [(258, 48), (204, 76), (156, 126), (126, 196)],
        [(258, 48), (312, 76), (360, 126), (390, 196)],
        [(240, 54), (196, 96), (158, 156), (136, 232)],
        [(276, 54), (320, 96), (358, 156), (380, 232)],
        [(290, 96), (248, 126), (208, 168), (184, 214)],
        [(268, 100), (224, 132), (190, 174), (168, 220)],
    ]
    for pts in seps:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 65, 2, cv2.LINE_AA)
    # crown spikes
    for x0, y0, x1, y1 in ((220, 44, 202, 28), (258, 42, 262,
                           24), (298, 46, 318, 32)):
        cv2.line(g, (x0, y0), (x1, y1), 55, 2, cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr96_s7.png"), 0)
    assert base is not None
    H, W = base.shape
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[280:, :])]
    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    pipe = _load(lcm, ckpt, "lllyasviel/control_v11p_sd15_lineart")
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    # ---- stage A: straps (tonal init) ----
    _scale_lora(pipe.unet, 1.0)
    sinit = _strap_init(base)
    sguide = _strap_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r98_sinit.png"), sinit)
    tmask = _torso_mask((H, W))
    cloths = []
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=CLOTH_PROMPT, negative_prompt=CLOTH_NEG,
                   image=Image.fromarray(sinit).convert("RGB"),
                   mask_image=Image.fromarray(tmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(sguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.9,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8, guidance_scale=3.0,
                   strength=0.50,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr98_c{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cloths.append((seed, g))
        print(f"[r98 {name}] {time.time() - t0:.0f}s", flush=True)

    # ---- stage B: hair clump edit (low strength) ----
    _scale_lora(pipe.unet, 0.8 / 1.0)
    hmask = _hair_mask((H, W))
    panels = [("base", base)]
    for seed, cimg in cloths:
        hguide = _hair_guide(cimg)
        t0 = time.time()
        res = pipe(prompt=HAIR_PROMPT, negative_prompt=HAIR_NEG,
                   image=Image.fromarray(cimg).convert("RGB"),
                   mask_image=Image.fromarray(hmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(hguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.85,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8, guidance_scale=3.0,
                   strength=0.45,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr98_c{seed}_h.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        print(f"[r98 {name}] {time.time() - t0:.0f}s", flush=True)
        panels.append((f"c{seed}", cimg))
        panels.append((f"c{seed}h", g))

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr98_sheet.png"), sheet)
    # strap zoom + hair zoom
    sz = [cv2.resize(im[420:660, 40:480], None, fx=1.3, fy=1.3,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    sh, sw = sz[0].shape
    ssheet = np.full((sh, sw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, sz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        ssheet[:, x:x + sw] = t
        x += sw + 8
    cv2.imwrite(os.path.join(OUT, "_genr98_straps.png"), ssheet)
    hz = [cv2.resize(im[30:280, 80:440], None, fx=1.2, fy=1.2,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    hh, hw = hz[0].shape
    hsheet = np.full((hh, hw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, hz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        hsheet[:, x:x + hw] = t
        x += hw + 8
    cv2.imwrite(os.path.join(OUT, "_genr98_hairs.png"), hsheet)
    print("[r98] sheets saved", flush=True)


if __name__ == "__main__":
    main()
