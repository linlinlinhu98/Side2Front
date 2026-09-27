"""R94 (user on s42: 嘴巴依旧向下弯看不出嘴唇/眉毛不像):

TWO literal pixel-transfer fixes:

1. BROWS: extract the ORIGINAL'S OWN brow stroke from the
   profile (it's fully visible!), threshold it, mirror for
   the other side, scale into the guide = the model renders
   the original's exact brow shape. (传递原图像素, literally)
2. LIPS: tone-built like the classmate's - paint a SOFT GRAY
   lower-lip blob + faint upper-lip hint into the init (tone,
   not lines), lip-line guide with corners 1px UPTURNED to
   counteract the model's ∩ prior. CN 0.6 only to hold the
   position.

Base: genr93_b42_lip (flattest mouth so far). 2 seeds each.
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
from gen_r93 import (BROW_PROMPT, BROW_NEG, _brow_mask,
                     _mouth_mask)  # noqa: E402


def _extract_brow(orig):
    """the original's own brow stroke, thresholded, trimmed"""
    roi = orig[180:200, 118:176]
    _, bw = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                          np.ones((2, 2), np.uint8))
    return bw


def _brow_guide(base, brow):
    g = _lineart(base)
    g[246:270, 158:358] = 255
    bh, bw = brow.shape
    # scale the extracted stroke to ~56px wide
    sc = 56 / bw
    b = cv2.resize(brow, None, fx=sc, fy=sc,
                   interpolation=cv2.INTER_CUBIC)
    b = (b > 100).astype(np.uint8) * 255
    bh2, bw2 = b.shape
    for sgn in (-1, 1):
        bb = b[:, ::-1] if sgn > 0 else b
        x_tip = CX + sgn * 30        # inner end
        y_top = 263 - bh2 // 2
        if sgn < 0:
            x0 = x_tip - bw2
            reg = g[y_top:y_top + bh2, x0:x_tip]
            reg[bb > 0] = 80
        else:
            x0 = x_tip
            reg = g[y_top:y_top + bh2, x0:x0 + bw2]
            reg[bb > 0] = 80
    return g


def _lip_init(base):
    """whitened mouth zone + soft tonal lip volume (alpha
    blends, not lines)"""
    init = base.copy()
    init[300:338, 218:300] = 255
    blob = np.zeros_like(init)
    cv2.ellipse(blob, (CX, 330), (16, 7), 0, 0, 360, 255, -1)
    a = cv2.GaussianBlur(blob, (0, 0), 4).astype(np.float32) / 255
    init = (init * (1 - a) + 218 * a).astype(np.uint8)
    up = np.zeros_like(init)
    cv2.ellipse(up, (CX, 322), (10, 4), 0, 0, 360, 255, -1)
    a2 = cv2.GaussianBlur(up, (0, 0), 3).astype(np.float32) / 255
    init = (init * (1 - a2) + 232 * a2).astype(np.uint8)
    return init


def _lip_guide(base):
    g = _lineart(base)
    g[306:340, 216:302] = 255
    # corners 1px UPTURNED to counteract the ∩ prior
    pts = [(CX - 15, 323), (CX, 325), (CX + 15, 323)]
    cv2.line(g, pts[0], pts[1], 70, 1, cv2.LINE_AA)
    cv2.line(g, pts[1], pts[2], 70, 1, cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr93_b42_lip.png"),
                      0)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"), 0)
    assert base is not None and orig is not None
    H, W = base.shape
    origc = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                       cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(origc, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[170:300, 100:320])]

    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    from diffusers import (ControlNetModel, LCMScheduler,
                           StableDiffusionControlNetInpaintPipeline)
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

    # ---- stage A: pixel-transferred brows ----
    brow = _extract_brow(orig)
    cv2.imwrite(os.path.join(OUT, "_r94_brow_extract.png"),
                255 - brow)
    bguide = _brow_guide(base, brow)
    cv2.imwrite(os.path.join(OUT, "_r94_bguide.png"), bguide)
    bmask = _brow_mask((H, W))
    _scale_lora(pipe.unet, 0.6)
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
        name = f"genr94_b{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        brows.append((seed, g))
        print(f"[r94 {name}] {time.time() - t0:.0f}s", flush=True)

    # ---- stage B: tone-built lips ----
    _scale_lora(pipe.unet, 0.4 / 0.6)
    mmask = _mouth_mask((H, W))
    panels = [("base", base)]
    for seed, bimg in brows:
        linit = _lip_init(bimg)
        lguide = _lip_guide(bimg)
        cv2.imwrite(os.path.join(OUT, f"_r94_linit{seed}.png"),
                    linit)
        t0 = time.time()
        res = pipe(prompt=LIP_PROMPT, negative_prompt=LIP_NEG,
                   image=Image.fromarray(linit).convert("RGB"),
                   mask_image=Image.fromarray(mmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(lguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.6,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8, guidance_scale=3.0,
                   strength=0.70,
                   generator=torch.Generator("cpu")
                   .manual_seed(7)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr94_b{seed}_lip.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        print(f"[r94 {name}] {time.time() - t0:.0f}s", flush=True)
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
    cv2.imwrite(os.path.join(OUT, "_genr94_sheet.png"), sheet)
    bz = [cv2.resize(im[244:272, 160:352], None, fx=4.5, fy=4.5,
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
    cv2.imwrite(os.path.join(OUT, "_genr94_brows.png"), bsheet)
    mz = [cv2.resize(im[305:348, 218:302], None, fx=6, fy=6,
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
    cv2.imwrite(os.path.join(OUT, "_genr94_mouths.png"), msheet)
    print("[r94] sheets saved", flush=True)


if __name__ == "__main__":
    main()
