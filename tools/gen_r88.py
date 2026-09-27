"""R88 (user: 衣服不像/鼻子不像/保持原图比例):

3-way study (_nose_cloth_study.png):
NOSE: orig = structured real nose (bridge + tip + CONNECTED
ala wings, ~55px = 1/6 of face); classmate = soft tone nose;
OURS = two floating commas, 24px, half the size. Fix: bigger
connected nose - bridge side-shadow line y288-318, tip arc,
ala curves CONNECTING tip to wings, nostril dots inside,
soft under-nose shadow. Nose-zone micro pass.
CLOTHES: R87 put the hood-mass tone blob ON the shoulders +
a pale band under the collar. Original folds = LONG SWEEPS
radiating from the shoulder points down-in. Fix: tonal field
clipped to y>460 with 35% cap (no shoulder blobs), fold
guide redrawn as the original's radiating sweeps, hatch only
side panels + fold valleys, lighter (145).

Stage A: clothes on genr86_m42_cloth (2 seeds).
Stage B: nose micro-pass on each (2 seeds).
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
from gen_r87 import (CLOTH_PROMPT, CLOTH_NEG, _torso_mask,
                     NECK_Y)  # noqa: E402

NOSE_PROMPT = (f"{TRIGGER}, detailed nose, nose bridge, "
               "nostrils, nose wings, realistic nose, small "
               "nose, pencil sketch, soft shading, "
               "monochrome, white background")
NOSE_NEG = ("dot nose, simple nose, line nose, big nose, "
            "long nose, color, colored, lowres, blurry, "
            "watermark")

STRENGTH = 0.60
CN_SCALE = 0.85


def _tonal_init(base, orig):
    tone = orig[460:660, :].astype(np.float32)
    tone = cv2.GaussianBlur(tone, (0, 0), 35)
    tone = np.minimum(tone, tone[:, ::-1])
    init = base.astype(np.float32).copy()
    region = init[460:660, :]
    target = 255.0 - np.clip(255.0 - tone, 0, 90) * 0.55
    region = np.minimum(region, target)
    init[460:660, :] = region
    return init.astype(np.uint8)


def _cloth_guide(base):
    g = _lineart(base)
    # radiating long sweeps from the shoulder points
    sweeps = [
        [(132, 448), (160, 505), (182, 570), (196, 648)],
        [(156, 452), (192, 510), (216, 575), (228, 648)],
        [(384, 448), (356, 505), (334, 570), (320, 648)],
        [(360, 452), (324, 510), (300, 575), (288, 648)],
        # center folds beside the zipper
        [(230, 470), (226, 545), (224, 648)],
        [(286, 470), (290, 545), (292, 648)],
    ]
    for pts in sweeps:
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 115, 1, cv2.LINE_AA)
    # sleeve creases
    for x0, y0, x1, y1 in ((112, 520, 148, 545), (102, 560, 140,
                           585), (408, 520, 372, 545),
                           (418, 560, 380, 585)):
        cv2.line(g, (x0, y0), (x1, y1), 120, 1, cv2.LINE_AA)
    # hem fold arcs
    for cx in (170, 258, 346):
        cv2.ellipse(g, (cx, 660), (46, 26), 0, 200, 340, 120, 1,
                    cv2.LINE_AA)
    # hatch: side panels + fold valleys only, light
    zones = [
        [(92, 480), (140, 478), (130, 645), (80, 645)],
        [(376, 478), (424, 480), (436, 645), (386, 645)],
        [(198, 545), (224, 545), (220, 645), (194, 645)],
        [(292, 545), (318, 545), (322, 645), (296, 645)],
    ]
    for poly in zones:
        m = np.zeros_like(g)
        cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
        tmp = np.full_like(g, 255)
        _hatch(tmp, poly, spacing=7, val=145)
        g[m > 0] = tmp[m > 0]
    return g


def _nose_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CX, 310), (34, 32), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 6)


def _nose_guide(base):
    g = _lineart(base)
    g[282:340, 222:296] = 255
    # bridge side-shadow line (left side), longer
    cv2.line(g, (CX - 3, 288), (CX - 4, 312), 125, 1,
             cv2.LINE_AA)
    # tip arc
    cv2.ellipse(g, (CX, 316), (6, 4), 0, 0, 180, 105, 1,
                cv2.LINE_AA)
    # ala wings CONNECTING tip to nostrils
    cv2.ellipse(g, (CX - 8, 316), (7, 6), 0, 300, 70, 90, 1,
                cv2.LINE_AA)
    cv2.ellipse(g, (CX + 8, 316), (7, 6), 0, 110, 240, 90, 1,
                cv2.LINE_AA)
    # nostril dots inside the wings
    cv2.ellipse(g, (CX - 8, 322), (2, 1), 0, 0, 360, 80, 2,
                cv2.LINE_AA)
    cv2.ellipse(g, (CX + 8, 322), (2, 1), 0, 0, 360, 80, 2,
                cv2.LINE_AA)
    # soft under-nose shadow
    cv2.ellipse(g, (CX, 330), (10, 3), 0, 25, 155, 180, 1,
                cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr86_m42_cloth.png"),
                      0)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"), 0)
    assert base is not None and orig is not None
    H, W = base.shape

    origc = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                       cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(origc, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[280:, :])]

    from diffusers import (ControlNetModel, LCMScheduler,
                           StableDiffusionControlNetInpaintPipeline)
    from peft import PeftModel
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)
    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    pipe = StableDiffusionControlNetInpaintPipeline \
        .from_single_file(ckpt, controlnet=cn,
                          torch_dtype=torch.float32,
                          safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora: {e}", flush=True)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(1.0)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    # ---- stage A: clothes (fixed tonal + radiating folds) ----
    init = _tonal_init(base, orig)
    guide = _cloth_guide(base)
    mask = _torso_mask((H, W))
    cv2.imwrite(os.path.join(OUT, "_r88_init.png"), init)
    cv2.imwrite(os.path.join(OUT, "_r88_guide.png"), guide)
    cloths = []
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=CLOTH_PROMPT, negative_prompt=CLOTH_NEG,
                   image=Image.fromarray(init).convert("RGB"),
                   mask_image=Image.fromarray(mask).convert("RGB"),
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=CN_SCALE,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8, guidance_scale=3.0,
                   strength=STRENGTH,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr88_c{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cloths.append((seed, g))
        print(f"[r88 {name}] {time.time() - t0:.0f}s", flush=True)

    # ---- stage B: nose micro-pass on each ----
    nmask = _nose_mask((H, W))
    panels = [("base", base)]
    for seed, cimg in cloths:
        nguide = _nose_guide(cimg)
        ninit = cimg.copy()
        ninit[284:338, 224:294] = 255
        t0 = time.time()
        res = pipe(prompt=NOSE_PROMPT, negative_prompt=NOSE_NEG,
                   image=Image.fromarray(ninit).convert("RGB"),
                   mask_image=Image.fromarray(nmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(nguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8, guidance_scale=3.5,
                   strength=0.75,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr88_c{seed}_nose.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        print(f"[r88 {name}] {time.time() - t0:.0f}s", flush=True)
        panels.append((f"c{seed}", cimg))
        panels.append((f"c{seed}+nose", g))

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr88_sheet.png"), sheet)
    nz = [cv2.resize(im[280:345, 214:304], None, fx=4, fy=4,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    nh, nw = nz[0].shape
    nsheet = np.full((nh, nw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, nz):
        t = t.copy()
        cv2.putText(t, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, 0, 2, cv2.LINE_AA)
        nsheet[:, x:x + nw] = t
        x += nw + 8
    cv2.imwrite(os.path.join(OUT, "_genr88_noses.png"), nsheet)
    cz = [cv2.resize(im[415:660, 50:470], None, fx=1.4, fy=1.4,
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
    cv2.imwrite(os.path.join(OUT, "_genr88_cloths.png"), csheet)
    print("[r88] sheets saved", flush=True)


if __name__ == "__main__":
    main()
