"""R93 (user picked genr92_s42 '还行', but: 嘴巴依旧向下弯看不
出嘴唇/眉毛不像; 仔细看原图每个像素):

BROW spec measured on _brow_grid.png (6x grid):
- 48px long on a 60px eye (ratio 0.8), SHORT relative to eye
- pressure profile: thin inner start -> thickest at 1/3 ->
  long taper to a fine outer point; top edge nearly straight,
  bottom edge follows the eye
- dark GRAY with soft pencil edges, sits CLOSE above the eye
  (inner gap ~3px)
- ours: uniform dark bars, too long, floating too high
Brow micro-pass: brow-band mask, guide redrawn to the
measured profile, LoRA 0.6, CN 0.9, strength 0.70.

LIPS: the free-inpaint that worked in R90 (full lips, no CN)
at the R92 mouth position (y315).
Base: genr92_s42. 2+2 seeds.
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
from gen_r90 import LIP_PROMPT, LIP_NEG  # noqa: E402

BROW_PROMPT = (f"{TRIGGER}, natural eyebrows, tapered "
               "eyebrows, straight eyebrows, pencil texture "
               "eyebrows, monochrome, pencil sketch, white "
               "background")
BROW_NEG = ("thick eyebrows, blocky eyebrows, arched "
            "eyebrows, angry eyebrows, color, colored, "
            "lowres, blurry, watermark")


def _brow_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (196, 258), (52, 16), 0, 0, 360, 255, -1)
    cv2.ellipse(m, (320, 258), (52, 16), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 5)


def _brow_guide(base):
    g = _lineart(base)
    g[246:270, 158:358] = 255
    # measured profile: thin inner -> thick at 1/3 -> long
    # taper to a fine outer point, outer end 2px lower
    for sgn in (-1, 1):
        xi = CX + sgn * 30          # inner end x
        xo = CX + sgn * 86          # outer tip x
        segs = [(0.00, 1, 85), (0.18, 3, 75), (0.45, 2, 85),
                (0.70, 1, 95)]
        for f0, th, val in segs:
            f1 = f0 + 0.25
            x0 = int(xi + (xo - xi) * f0)
            x1 = int(xi + (xo - xi) * f1)
            y0 = int(261 + 2 * f0 + (1 if 0.2 < f0 < 0.7
                                     else 0))
            y1 = int(261 + 2 * f1 + (1 if 0.2 < f1 < 0.7
                                     else 0))
            cv2.line(g, (x0, y0), (x1, y1), val, th,
                     cv2.LINE_AA)
    return g


def _mouth_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CX, 317), (40, 20), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 6)


def main():
    base = cv2.imread(os.path.join(OUT, "genr92_s42.png"), 0)
    assert base is not None
    H, W = base.shape
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[170:300, 100:320])]

    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    # ---- stage A: brows (CN guide) ----
    from diffusers import (ControlNetModel, LCMScheduler,
                           StableDiffusionControlNetInpaintPipeline,
                           StableDiffusionInpaintPipeline)
    from peft import PeftModel
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetInpaintPipeline \
        .from_single_file(ckpt, controlnet=cn,
                          torch_dtype=torch.float32,
                          safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception:
        pass
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
    _scale_lora(pipe.unet, 0.6)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.9)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    bmask = _brow_mask((H, W))
    bguide = _brow_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r93_bguide.png"), bguide)
    binit = base.copy()
    binit[246:270, 158:358] = 255
    brows = []
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=BROW_PROMPT, negative_prompt=BROW_NEG,
                   image=Image.fromarray(binit).convert("RGB"),
                   mask_image=Image.fromarray(bmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(bguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.9,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8, guidance_scale=3.0,
                   strength=0.70,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr93_b{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        brows.append((seed, g))
        print(f"[r93 {name}] {time.time() - t0:.0f}s", flush=True)
    del pipe, cn
    import gc
    gc.collect()

    # ---- stage B: lips (free inpaint, R90 method) ----
    pipe2 = StableDiffusionInpaintPipeline.from_single_file(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    pipe2.scheduler = LCMScheduler.from_config(
        pipe2.scheduler.config)
    pipe2.load_lora_weights(lcm)
    try:
        pipe2.fuse_lora()
    except Exception:
        pass
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
    panels = [("base", base)]
    for seed, bimg in brows:
        minit = bimg.copy()
        minit[300:336, 222:296] = 255
        t0 = time.time()
        res = pipe2(prompt=LIP_PROMPT, negative_prompt=LIP_NEG,
                    image=Image.fromarray(minit).convert("RGB"),
                    mask_image=Image.fromarray(mmask)
                    .convert("RGB"),
                    ip_adapter_image_embeds=[emb2],
                    height=664, width=520,
                    num_inference_steps=8, guidance_scale=3.0,
                    strength=0.7,
                    generator=torch.Generator("cpu")
                    .manual_seed(7)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr93_b{seed}_lip.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        print(f"[r93 {name}] {time.time() - t0:.0f}s", flush=True)
        panels.append((f"b{seed}", bimg))
        panels.append((f"b{seed}+lip", g))

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr93_sheet.png"), sheet)
    # brow zoom + mouth zoom
    bz = [cv2.resize(im[244:296, 150:366], None, fx=3.2, fy=3.2,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    bh, bw = bz[0].shape
    bsheet = np.full((bh, bw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, bz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        bsheet[:, x:x + bw] = t
        x += bw + 8
    cv2.imwrite(os.path.join(OUT, "_genr93_brows.png"), bsheet)
    mz = [cv2.resize(im[296:336, 216:302], None, fx=5, fy=5,
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
    cv2.imwrite(os.path.join(OUT, "_genr93_mouths.png"), msheet)
    print("[r93] sheets saved", flush=True)


if __name__ == "__main__":
    main()
