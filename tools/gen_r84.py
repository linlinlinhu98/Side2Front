"""R84 (user: 嘴平直了一点但还是不像 - 原图有嘴唇(体积);
衣服纹理也不像):

Mouth: overshot to a 'line mouth'. Classmate/original spec =
flat lip line + SOFT LOWER-LIP VOLUME (light shading below,
center-weighted, NOT at the corners). Guide: keep the level
line, add 2 very light arcs below (lower-lip edge + under
shadow). LoRA 0.3, CN 1.0.

Clothes: hand-DRAW the hatching into the ControlNet guide
(spacing 8-10px so it survives the latent grid): collar-roll
shadow, zipper placket sides, side seams, chest fold valleys.
Strength 0.5, LoRA 1.0, IP torso 1.0, 'dense parallel
hatching' prompt.

Base: genr83_m7_cloth. 2+2 seeds.
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
from gen_r83 import (CLOTH_PROMPT, CLOTH_NEG, LORA_DIR,
                     TRIGGER, CX, _scale_lora, _mouth_mask,
                     _torso_mask)  # noqa: E402

MOUTH_PROMPT = (f"{TRIGGER}, closed lips, full lower lip, "
                "lip volume, realistic mouth, neutral "
                "mouth, small mouth, monochrome, pencil "
                "sketch, soft shading, white background")
MOUTH_NEG = ("frown, downturned mouth, sad, pout, line "
             "mouth, open mouth, smile, smirk, upper lip "
             "shadow, philtrum, color, colored, lowres, "
             "blurry, watermark")


def _mouth_guide(base):
    g = _lineart(base)
    g[326:354, 222:296] = 255
    # level lip line
    cv2.line(g, (CX - 14, 340), (CX + 14, 340), 55, 2,
             cv2.LINE_AA)
    # soft lower-lip volume: two very light arcs BELOW center
    cv2.ellipse(g, (CX, 349), (12, 5), 0, 20, 160, 175, 1,
                cv2.LINE_AA)
    cv2.ellipse(g, (CX, 353), (8, 3), 0, 25, 155, 190, 1,
                cv2.LINE_AA)
    return g


def _hatch(g, poly, spacing=9, val=135, ang=(4, -3)):
    """diagonal hatch inside a quad-ish polygon"""
    m = np.zeros_like(g)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
    x0 = min(p[0] for p in poly) - 20
    x1 = max(p[0] for p in poly) + 20
    y0 = min(p[1] for p in poly) - 20
    y1 = max(p[1] for p in poly) + 20
    dx, dy = ang
    ln = int(np.hypot(dx, dy))
    dx, dy = dx / ln, dy / ln
    nx, ny = -dy, dx
    d = -200
    while d < 200:
        p0 = (int((x0 + x1) / 2 + nx * d - dx * 300),
              int((y0 + y1) / 2 + ny * d - dy * 300))
        p1 = (int((x0 + x1) / 2 + nx * d + dx * 300),
              int((y0 + y1) / 2 + ny * d + dy * 300))
        cv2.line(g, p0, p1, val, 1, cv2.LINE_AA)
        d += spacing
    g[m == 0] = 255 if False else g[m == 0]
    # restore: keep hatch only inside polygon
    return m


def _cloth_guide(base):
    g = _lineart(base)
    zones = []
    # collar-roll shadows
    zones.append([(165, 425), (250, 430), (246, 458),
                  (168, 452)])
    zones.append([(266, 430), (350, 425), (348, 452),
                  (270, 458)])
    # zipper placket sides
    zones.append([(243, 445), (252, 445), (252, 645),
                  (243, 645)])
    zones.append([(264, 445), (273, 445), (273, 645),
                  (264, 645)])
    # side seams
    zones.append([(96, 480), (112, 480), (104, 650),
                  (88, 650)])
    zones.append([(404, 480), (420, 480), (428, 650),
                  (412, 650)])
    # chest fold valleys
    zones.append([(175, 520), (240, 528), (236, 548),
                  (172, 540)])
    zones.append([(280, 545), (345, 536), (348, 556),
                  (284, 565)])
    for poly in zones:
        m = np.zeros_like(g)
        cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
        tmp = np.full_like(g, 255)
        _hatch(tmp, poly, spacing=9, val=135)
        g[m > 0] = tmp[m > 0]
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr83_m7_cloth.png"),
                      0)
    assert base is not None
    H, W = base.shape

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
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

    # ---- stage A: lips with volume (LoRA 0.3, CN 1.0) ----
    _scale_lora(pipe.unet, 0.3)
    mmask = _mouth_mask((H, W))
    mguide = _mouth_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r84_mguide.png"), mguide)
    minit = base.copy()
    minit[324:356, 218:300] = 255
    mouths = []
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=MOUTH_PROMPT, negative_prompt=MOUTH_NEG,
                   image=Image.fromarray(minit).convert("RGB"),
                   mask_image=Image.fromarray(mmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(mguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8, guidance_scale=3.5,
                   strength=0.75,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr84_m{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        mouths.append((seed, g))
        print(f"[r84 {name}] {time.time() - t0:.0f}s", flush=True)

    # ---- stage B: hatched clothes (LoRA 1.0) ----
    _scale_lora(pipe.unet, 1.0 / 0.3)
    tmask = _torso_mask((H, W))
    finals = []
    for seed, mimg in mouths:
        tguide = _cloth_guide(mimg)
        cv2.imwrite(os.path.join(OUT, f"_r84_cguide{seed}.png"),
                    tguide)
        t0 = time.time()
        res = pipe(prompt=CLOTH_PROMPT, negative_prompt=CLOTH_NEG,
                   image=Image.fromarray(mimg).convert("RGB"),
                   mask_image=Image.fromarray(tmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(tguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.9,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8, guidance_scale=3.0,
                   strength=0.5,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr84_m{seed}_cloth.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        finals.append((f"m{seed}+cloth", g))
        print(f"[r84 {name}] {time.time() - t0:.0f}s", flush=True)

    panels = [("base", base)] + [(f"m{s}", m) for s, m in mouths] \
             + finals
    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr84_sheet.png"), sheet)
    mz = [cv2.resize(im[322:360, 214:304], None, fx=5, fy=5,
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
    cv2.imwrite(os.path.join(OUT, "_genr84_mouths.png"), msheet)
    cz = [cv2.resize(im[420:640, 70:450], None, fx=1.5, fy=1.5,
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
    cv2.imwrite(os.path.join(OUT, "_genr84_cloths.png"), csheet)
    print("[r84] sheets saved", flush=True)


if __name__ == "__main__":
    main()
