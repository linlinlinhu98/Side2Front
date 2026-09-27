"""R95 (R94 failed: brow extraction caught the eyelash +
misplaced onto the eyes; lip tone-blob became a gray smudge):

Fixes:
1. BROW pixel-transfer v2: extraction tightened (y180-193,
   thr 180, largest component only -> the full 50px brow
   stroke, no lash). Placement: stroke bottom at y=262 (8px
   above the lash line), thick end at the INNER side.
2. LIPS v2: NO tone blob. Guide = lip line with corners +2px
   UPTURNED (counter the model's ∩ prior) + one very faint
   lower-lip arc (185). CN 0.8, strength 0.65.

Base: genr93_b42_lip. 2+2 seeds.
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
    roi = orig[180:193, 116:176]
    _, bw = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY_INV)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw)
    if n > 2:
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        bw = (lab == big).astype(np.uint8) * 255
    return bw


def _brow_guide(base, brow):
    g = _lineart(base)
    g[246:272, 158:358] = 255
    bh, bw = brow.shape
    sc = 56 / bw
    b = cv2.resize(brow, None, fx=sc, fy=sc,
                   interpolation=cv2.INTER_CUBIC)
    b = (b > 100).astype(np.uint8) * 255
    bh2, bw2 = b.shape
    y0 = 262 - bh2                       # stroke bottom at 262
    for sgn in (-1, 1):
        bb = b if sgn > 0 else b[:, ::-1]   # thick end inner
        x_inner = CX + sgn * 30
        if sgn < 0:
            x0 = x_inner - bw2
            reg = g[y0:y0 + bh2, x0:x_inner]
            reg[bb > 0] = 85
        else:
            reg = g[y0:y0 + bh2, x_inner:x_inner + bw2]
            reg[bb > 0] = 85
    return g


def _lip_guide(base):
    g = _lineart(base)
    g[306:340, 216:302] = 255
    # corners +2px upturned
    pts = [(CX - 15, 321), (CX, 323), (CX + 15, 321)]
    cv2.line(g, pts[0], pts[1], 70, 1, cv2.LINE_AA)
    cv2.line(g, pts[1], pts[2], 70, 1, cv2.LINE_AA)
    # very faint lower-lip volume arc
    cv2.ellipse(g, (CX, 330), (10, 4), 0, 25, 155, 185, 1,
                cv2.LINE_AA)
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

    # ---- stage A: pixel-transferred brows v2 ----
    brow = _extract_brow(orig)
    bguide = _brow_guide(base, brow)
    cv2.imwrite(os.path.join(OUT, "_r95_bguide.png"), bguide)
    bmask = _brow_mask((H, W))
    _scale_lora(pipe.unet, 0.6)
    binit = base.copy()
    binit[246:272, 158:358] = 255
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
        name = f"genr95_b{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        brows.append((seed, g))
        print(f"[r95 {name}] {time.time() - t0:.0f}s", flush=True)

    # ---- stage B: lips v2 ----
    _scale_lora(pipe.unet, 0.4 / 0.6)
    mmask = _mouth_mask((H, W))
    panels = [("base", base)]
    for seed, bimg in brows:
        lguide = _lip_guide(bimg)
        linit = bimg.copy()
        linit[300:338, 218:300] = 255
        t0 = time.time()
        res = pipe(prompt=LIP_PROMPT, negative_prompt=LIP_NEG,
                   image=Image.fromarray(linit).convert("RGB"),
                   mask_image=Image.fromarray(mmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(lguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.8,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8, guidance_scale=3.0,
                   strength=0.65,
                   generator=torch.Generator("cpu")
                   .manual_seed(7)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr95_b{seed}_lip.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        print(f"[r95 {name}] {time.time() - t0:.0f}s", flush=True)
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
    cv2.imwrite(os.path.join(OUT, "_genr95_sheet.png"), sheet)
    bz = [cv2.resize(im[244:274, 155:360], None, fx=3.4, fy=3.4,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    bh, bw2 = bz[0].shape
    bsheet = np.full((bh, bw2 * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, bz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        bsheet[:, x:x + bw2] = t
        x += bw2 + 8
    cv2.imwrite(os.path.join(OUT, "_genr95_brows.png"), bsheet)
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
    cv2.imwrite(os.path.join(OUT, "_genr95_mouths.png"), msheet)
    print("[r95] sheets saved", flush=True)


if __name__ == "__main__":
    main()
