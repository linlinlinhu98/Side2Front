"""R90 (user: 不如base, 在 base 上再优化嘴唇和衣服, 仔细看
原图; base = genr86_m42_cloth):

LIPS (grid study _orig_mouth_grid.png): orig = protruding
upper lip + DARK LEVEL 33px lip line + FULL lower lip (soft
shadow MASS below, not a thin arc) + lip-chin groove. Our
thin line + faint arc reads as 'line mouth'. Fix: FREE
inpaint (no CN, like the winning R89 nose) with a full-lips
prompt.

CLOTHES (deep study _cloth_deep.png): missing structures vs
original:
1. hood mass behind the neck - thick rolls over BOTH
   shoulders with a BOLD dark inner edge
2. ring pull at the CENTER collar (ours drifted to the side
   seam)
3. bold dark accents at the main fold edges (ours all thin)
Guide redraws all three + keeps R88 radiating folds/hatch.
2+3 seeds.
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
from gen_r87 import (CLOTH_PROMPT, CLOTH_NEG, _torso_mask)  # noqa: E402
from gen_r88 import _tonal_init  # noqa: E402

LIP_PROMPT = (f"{TRIGGER}, full lips, closed mouth, soft "
              "lower lip shadow, realistic lips, neutral "
              "expression, small mouth, pencil sketch, soft "
              "shading, monochrome, white background")
LIP_NEG = ("line mouth, thin lips, frown, downturned mouth, "
           "sad, smile, open mouth, smirk, color, colored, "
           "lowres, blurry, watermark")


def _mouth_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CX, 340), (40, 22), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 6)


def _cloth_guide(base):
    g = _lineart(base)
    # 1) hood mass over BOTH shoulders: thick rolls with a
    # BOLD dark inner edge
    for sgn in (-1, 1):
        x0 = CX + sgn * 26
        x1 = CX + sgn * 108
        pts = [(x0 + (x1 - x0) * t / 24,
                int(424 + 30 * (t / 24) ** 1.5))
               for t in range(25)]
        pts = [(int(x), int(y)) for x, y in pts]
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 70, 2, cv2.LINE_AA)
        pts2 = [(x, y + 10) for x, y in pts[:-5]]
        for i in range(len(pts2) - 1):
            cv2.line(g, pts2[i], pts2[i + 1], 45, 3, cv2.LINE_AA)
    # hood mass arc behind the neck
    cv2.ellipse(g, (CX, 420), (86, 26), 0, 185, 355, 75, 2,
                cv2.LINE_AA)
    # 2) RING pull at the CENTER collar + slider
    cv2.circle(g, (CX, 452), 6, 45, 2, cv2.LINE_AA)
    cv2.rectangle(g, (CX - 3, 440), (CX + 3, 447), 45, 1)
    # 3) full zipper + teeth
    cv2.line(g, (CX, 458), (CX, 656), 55, 2, cv2.LINE_AA)
    for y in range(464, 650, 9):
        cv2.line(g, (CX - 4, y + 2), (CX + 4, y - 2), 95, 1,
                 cv2.LINE_AA)
    # 4) radiating long folds (R88) with bold accents at the
    # two main armpit folds
    sweeps = [
        [(132, 448), (160, 505), (182, 570), (196, 648)],
        [(156, 452), (192, 510), (216, 575), (228, 648)],
        [(384, 448), (356, 505), (334, 570), (320, 648)],
        [(360, 452), (324, 510), (300, 575), (288, 648)],
        [(230, 470), (226, 545), (224, 648)],
        [(286, 470), (290, 545), (292, 648)],
    ]
    for k, pts in enumerate(sweeps):
        bold = (k in (0, 2))
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1],
                     70 if bold else 115, 2 if bold else 1,
                     cv2.LINE_AA)
    # 5) hatch side panels + valleys
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
        _hatch(tmp, poly, spacing=7, val=140)
        g[m > 0] = tmp[m > 0]
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
                           StableDiffusionControlNetInpaintPipeline,
                           StableDiffusionInpaintPipeline)
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
        print(f"[warn] fuse: {e}", flush=True)
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

    # ---- stage A: clothes (hood mass + ring pull + bold) ----
    init = _tonal_init(base, orig)
    guide = _cloth_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r90_guide.png"), guide)
    tmask = _torso_mask((H, W))
    cloths = []
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=CLOTH_PROMPT, negative_prompt=CLOTH_NEG,
                   image=Image.fromarray(init).convert("RGB"),
                   mask_image=Image.fromarray(tmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.9,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8, guidance_scale=3.0,
                   strength=0.6,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr90_c{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cloths.append((seed, g))
        print(f"[r90 {name}] {time.time() - t0:.0f}s", flush=True)
    del pipe, cn
    import gc
    gc.collect()

    # ---- stage B: lips, FREE inpaint on each clothes seed ----
    pipe2 = StableDiffusionInpaintPipeline.from_single_file(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    pipe2.scheduler = LCMScheduler.from_config(
        pipe2.scheduler.config)
    pipe2.load_lora_weights(lcm)
    try:
        pipe2.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse2: {e}", flush=True)
    pipe2.unet = PeftModel.from_pretrained(pipe2.unet, LORA_DIR)
    _scale_lora(pipe2.unet, 0.4)
    pipe2.set_progress_bar_config(disable=True)
    pipe2.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe2.set_ip_adapter_scale(0.9)
    with torch.no_grad():
        emb2, unc2 = pipe2.encode_image(
            refs[:2], "cpu", 1, output_hidden_states=True)
        emb2 = torch.cat([unc2.unsqueeze(0), emb2.unsqueeze(0)],
                         dim=0)

    mmask = _mouth_mask((H, W))
    mpil = Image.fromarray(mmask).convert("RGB")
    panels = [("base", base)]
    for seed, cimg in cloths:
        minit = cimg.copy()
        minit[324:360, 222:296] = 255
        lname = None
        for lseed in (7,):
            t0 = time.time()
            res = pipe2(prompt=LIP_PROMPT, negative_prompt=LIP_NEG,
                        image=Image.fromarray(minit)
                        .convert("RGB"),
                        mask_image=mpil,
                        ip_adapter_image_embeds=[emb2],
                        height=664, width=520,
                        num_inference_steps=8,
                        guidance_scale=3.0, strength=0.7,
                        generator=torch.Generator("cpu")
                        .manual_seed(lseed)).images[0]
            g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
            g = cv2.resize(g, (W, H),
                           interpolation=cv2.INTER_CUBIC)
            lname = f"genr90_c{seed}_lip{lseed}.png"
            cv2.imwrite(os.path.join(OUT, lname), g)
            print(f"[r90 {lname}] {time.time() - t0:.0f}s",
                  flush=True)
        panels.append((f"c{seed}", cimg))
        panels.append((f"c{seed}+lip", g))

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr90_sheet.png"), sheet)
    mz = [cv2.resize(im[322:362, 216:302], None, fx=5, fy=5,
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
    cv2.imwrite(os.path.join(OUT, "_genr90_mouths.png"), msheet)
    cz = [cv2.resize(im[400:600, 60:460], None, fx=1.6, fy=1.6,
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
    cv2.imwrite(os.path.join(OUT, "_genr90_cloths.png"), csheet)
    print("[r90] sheets saved", flush=True)


if __name__ == "__main__":
    main()
