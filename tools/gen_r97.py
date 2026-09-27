"""R97 (independent analysis + verification _verify3.png):

1. BACKPACK STRAPS confirmed on the original (dark straps with
   buckle detail) and in the classmate's frontal (two straps)
   - we NEVER had them. Identity-critical (schoolboy with a
   backpack). Add to the clothes guide: two bold strap bands
   from the shoulder points down the torso sides.
2. HAIR: ours = fine mechanical parallel hatching (wig cap);
   original = BIG strand clumps with dark gaps, spiky crown
   tufts, bangs swept to the left with pointed tips. Hair-
   region repaint with a strand-clump guide.
3. EYES: original's lash has a subtle 2-3px upturn at the
   outer corner - I flattened it to death. Eye micro-pass.

Base: genr96_s7. Stages: A clothes+straps (2), B hair (2),
C eye flick (1 on the best).
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
from gen_r91 import _load  # noqa: E402

CLOTH_PROMPT = (f"{TRIGGER}, zip-up hoodie, full-length "
                "zipper, hood down, backpack straps, "
                "shoulder straps, dense parallel hatching, "
                "fabric folds, tonal shading, monochrome, "
                "pencil sketch, soft shading, white "
                "background")
CLOTH_NEG = ("smooth, clean, digital painting, flat, "
             "drawstrings, pullover, color, colored, photo, "
             "lowres, blurry, watermark")

HAIR_PROMPT = (f"{TRIGGER}, short messy hair, spiky hair, "
               "big hair strands, thick hair locks, bangs "
               "swept to the side, fluffy hair, pencil "
               "sketch, soft shading, monochrome, white "
               "background")
HAIR_NEG = ("fine hatching, parallel lines, helmet hair, "
            "neat hair, curly hair, long hair, color, "
            "colored, photo, lowres, blurry, watermark")

EYE_PROMPT = (f"{TRIGGER}, long narrow eyes, single eyelid, "
              "half-closed eyes, slightly upturned outer "
              "corners, calm expression, pencil sketch, "
              "monochrome, white background")
EYE_NEG = ("round eyes, big eyes, double eyelid, droopy "
           "eyes, color, colored, lowres, blurry, "
           "watermark")


def _cloth_guide(base):
    g = _lineart(base)
    # BACKPACK STRAPS: bold paired edges, shoulder -> side
    for sgn in (-1, 1):
        xs = CX + sgn * 104       # shoulder top x
        xe = CX + sgn * 158       # hem exit x
        pts = [(xs, 438), (xs + sgn * 16, 500),
               (xs + sgn * 36, 570), (xe, 650)]
        for off in (-9, 9):
            p2 = [(x + off, y) for x, y in pts]
            for i in range(len(p2) - 1):
                cv2.line(g, p2[i], p2[i + 1], 55, 2,
                         cv2.LINE_AA)
        # buckle hint
        cv2.rectangle(g, (xs + sgn * 12 - 6, 505),
                      (xs + sgn * 12 + 6, 514), 60, 1)
    # zipper + folds kept light
    cv2.line(g, (CX, 458), (CX, 656), 60, 2, cv2.LINE_AA)
    sweeps = [
        [(200, 470), (206, 545), (210, 648)],
        [(316, 470), (310, 545), (306, 648)],
    ]
    for pts in sweeps:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 120, 1, cv2.LINE_AA)
    return g


def _hair_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (258, 185), (155, 155), 0, 0, 360, 255, -1)
    f = np.zeros((H, W), np.uint8)
    cv2.ellipse(f, (258, 310), (104, 82), 0, 0, 360, 255, -1)
    m[f > 0] = 0
    m[:28, :] = 0
    m[400:, :] = 0
    return cv2.GaussianBlur(m, (0, 0), 10)


def _hair_guide(base):
    g = _lineart(base)
    # keep the outer silhouette, wipe the interior hatching
    sil = g.copy()
    g[40:330, 90:430] = 255
    # redraw: BIG strand clumps
    strands = [
        # crown -> sides big sweeps
        [(258, 46), (200, 70), (150, 120), (118, 190)],
        [(258, 46), (316, 70), (366, 120), (398, 190)],
        [(240, 50), (190, 90), (150, 150), (130, 230)],
        [(276, 50), (326, 90), (366, 150), (386, 230)],
        # bangs clumps swept to viewer's LEFT, pointed tips
        [(292, 92), (252, 120), (212, 160), (186, 210)],
        [(270, 96), (228, 126), (190, 168), (168, 216)],
        [(248, 100), (206, 134), (176, 176), (158, 224)],
        [(312, 100), (340, 140), (360, 190), (368, 236)],
        [(336, 108), (368, 150), (390, 200), (398, 244)],
    ]
    for pts in strands:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 70, 2, cv2.LINE_AA)
    # crown spikes sticking out
    for x0, y0, x1, y1 in ((220, 44, 200, 26), (258, 42, 262,
                           22), (300, 46, 322, 30), (170, 66,
                           140, 44), (350, 70, 384, 52)):
        cv2.line(g, (x0, y0), (x1, y1), 60, 2, cv2.LINE_AA)
    # restore the silhouette outline on top
    edge = cv2.Canny(sil, 60, 150)
    g[edge > 0] = sil[edge > 0]
    return g


def _eye_guide(base):
    g = _lineart(base)
    g[258:296, 158:358] = 255
    for cx in (196, 320):
        cv2.circle(g, (cx, 276), 5, 100, -1, cv2.LINE_AA)
        cv2.rectangle(g, (cx - 32, 258), (cx + 32, 271), 255,
                      -1)
        sgn = -1 if cx < CX else 1
        up = [(cx - 31, 273), (cx - 12, 270), (cx + 12, 270),
              (cx + 31, 273)]
        for i in range(3):
            cv2.line(g, up[i], up[i + 1], 45, 3, cv2.LINE_AA)
        # subtle 3px outer-corner upturn
        cv2.line(g, (cx + sgn * 31, 273),
                 (cx + sgn * 36, 270), 45, 2, cv2.LINE_AA)
        lo = [(cx - 31, 273), (cx - 12, 280), (cx + 12, 280),
              (cx + 31, 273)]
        for i in range(3):
            cv2.line(g, lo[i], lo[i + 1], 90, 1, cv2.LINE_AA)
    return g


def _eye_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (258, 276), (100, 26), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 6)


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

    # ---- stage A: clothes + backpack straps ----
    _scale_lora(pipe.unet, 1.0)
    tguide = _cloth_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r97_cguide.png"), tguide)
    tmask = _torso_mask((H, W))
    cloths = []
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=CLOTH_PROMPT, negative_prompt=CLOTH_NEG,
                   image=Image.fromarray(base).convert("RGB"),
                   mask_image=Image.fromarray(tmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(tguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.9,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8, guidance_scale=3.0,
                   strength=0.55,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr97_c{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cloths.append((seed, g))
        print(f"[r97 {name}] {time.time() - t0:.0f}s", flush=True)

    # ---- stage B: hair strand clumps on each ----
    _scale_lora(pipe.unet, 0.8 / 1.0)
    hmask = _hair_mask((H, W))
    hairs = []
    for seed, cimg in cloths:
        hguide = _hair_guide(cimg)
        cv2.imwrite(os.path.join(OUT, f"_r97_hguide{seed}.png"),
                    hguide)
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
                   strength=0.75,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr97_c{seed}_h.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        hairs.append((f"c{seed}h", g))
        print(f"[r97 {name}] {time.time() - t0:.0f}s", flush=True)

    # ---- stage C: eye flick on the first hair result ----
    esrc = hairs[0][1]
    eguide = _eye_guide(esrc)
    emask = _eye_mask((H, W))
    einit = esrc.copy()
    einit[256:296, 158:358] = 255
    t0 = time.time()
    res = pipe(prompt=EYE_PROMPT, negative_prompt=EYE_NEG,
               image=Image.fromarray(einit).convert("RGB"),
               mask_image=Image.fromarray(emask).convert("RGB"),
               control_image=Image.fromarray(eguide)
               .convert("RGB"),
               controlnet_conditioning_scale=0.9,
               ip_adapter_image_embeds=[emb],
               height=664, width=520,
               num_inference_steps=8, guidance_scale=3.0,
               strength=0.65,
               generator=torch.Generator("cpu")
               .manual_seed(7)).images[0]
    g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(OUT, "genr97_final.png"), g)
    print(f"[r97 final] {time.time() - t0:.0f}s", flush=True)

    panels = [("base", base)] + \
             [(f"c{s}", im) for s, im in cloths] + hairs + \
             [("final", g)]
    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr97_sheet.png"), sheet)
    print("[r97] sheets saved", flush=True)


if __name__ == "__main__":
    main()
