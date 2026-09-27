"""R74 (user: 头部也小了要像原图占背景的比例; 身体每个部位
比例要合适):

Measured proportions (original vs our base):
- head height / canvas: 320px (48%) vs 250px (38%)
- shoulder width / canvas: ~400px (77%) vs 330px (63%)
- shoulder / head width: ~1.55 vs 1.83

A uniform zoom cannot fix the head:shoulder RATIO, so:
Stage 0: piecewise composition warp - head band x1.30, torso
  band x1.12, smooth neck transition (same remap applied to
  base + face guide + face mask so everything stays aligned)
Stage A: img2img fusion pass (strength 0.45, LoRA 0.7, IP
  [full+head+torso] @0.8, R69 clothes-lock prompt) - the model
  re-renders everything at the new composition, erasing warp
  artifacts. 1 seed.
Stage B: R73 symmetrized-face repaint (CN guide 0.90,
  strength 0.85) on the fused zoomed image. 4 seeds.
"""
import gc
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
from gen_face_r73 import (_face_guide, _face_mask, PROMPT,
                          NEG, LORA_DIR, TRIGGER, LORA_SCALE,
                          CN_SCALE, IP_SCALE, GUIDANCE)  # noqa: E402
from gen_final_r69 import PROMPT as FUSE_PROMPT  # noqa: E402
from gen_final_r69 import NEG as FUSE_NEG  # noqa: E402

CX = 258
HEAD_SC = 1.28
BODY_SC = 1.12
# vertical anchors src_y -> dst_y (head 326px = 49% of canvas
# like the original's 48%, ~5% breathing room above the crown)
V_ANCH = [(115, 36), (370, 362), (430, 420), (660, 660)]
FACE_SEEDS = [7, 42, 998, 123]
FUSE_SEED = 42
FUSE_STRENGTH = 0.45
FUSE_IP = 0.8
FACE_STRENGTH = 0.85


def _warp_maps(H, W):
    dst_y = np.arange(H, dtype=np.float32)
    ay = np.array([d for _, d in V_ANCH], np.float32)
    by = np.array([s for s, _ in V_ANCH], np.float32)
    src_y = np.interp(dst_y, ay, by)
    top = dst_y < ay[0]
    src_y[top] = by[0] - (ay[0] - dst_y[top]) / HEAD_SC
    # per-row horizontal scale: head rows HEAD_SC, body BODY_SC
    sc = np.where(dst_y <= 362, HEAD_SC,
                  np.where(dst_y >= 420, BODY_SC,
                           HEAD_SC + (BODY_SC - HEAD_SC)
                           * (dst_y - 362) / 58.0))
    dst_x = np.arange(W, dtype=np.float32)
    src_x = CX + (dst_x[None, :] - CX) / sc[:, None]
    src_yy = np.repeat(src_y[:, None], W, axis=1)
    return src_x.astype(np.float32), src_yy.astype(np.float32)


def _warp(img, H, W, border=255):
    mx, my = _warp_maps(H, W)
    return cv2.remap(img, mx, my, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=border)


def _scale_lora(unet, factor):
    for m in unet.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict) and "default" in sc:
            sc["default"] = sc["default"] * factor


def main():
    base = cv2.imread(os.path.join(OUT, "genr69_s2024_eye.png"),
                      0)
    assert base is not None
    H, W = base.shape
    zbase = _warp(base, H, W)
    guide = _warp(_face_guide(base), H, W)
    mask = _warp(_face_mask((H, W)), H, W, border=0)
    mask = cv2.GaussianBlur(mask, (0, 0), 6)
    cv2.imwrite(os.path.join(OUT, "_r74_zbase.png"), zbase)
    cv2.imwrite(os.path.join(OUT, "_r74_guide.png"), guide)
    cv2.imwrite(os.path.join(OUT, "_r74_mask.png"), mask)

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs3 = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
             Image.fromarray(rgb[280:, :])]
    refs2 = refs3[:2]

    from diffusers import (ControlNetModel, LCMScheduler,
                           StableDiffusionImg2ImgPipeline,
                           StableDiffusionControlNetInpaintPipeline)
    from peft import PeftModel
    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    # ---- Stage A: fusion img2img at the new composition ----
    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora: {e}", flush=True)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
    _scale_lora(pipe.unet, LORA_SCALE)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(FUSE_IP)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs3, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)
    t0 = time.time()
    res = pipe(prompt=FUSE_PROMPT, negative_prompt=FUSE_NEG,
               image=Image.fromarray(zbase).convert("RGB"),
               ip_adapter_image_embeds=[emb],
               height=664, width=520, num_inference_steps=8,
               guidance_scale=GUIDANCE, strength=FUSE_STRENGTH,
               generator=torch.Generator("cpu")
               .manual_seed(FUSE_SEED)).images[0]
    fused = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
    fused = cv2.resize(fused, (W, H), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(OUT, "genr74_fused.png"), fused)
    print(f"[r74 fused] {time.time() - t0:.0f}s", flush=True)
    del pipe, emb
    gc.collect()

    # ---- Stage B: symmetrized face repaint on fused ----
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
    except Exception as e:
        print(f"[warn] fuse_lora2: {e}", flush=True)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
    _scale_lora(pipe.unet, LORA_SCALE)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(IP_SCALE)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs2, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    bpil = Image.fromarray(fused).convert("RGB")
    mpil = Image.fromarray(mask).convert("RGB")
    ctrl = Image.fromarray(guide).convert("RGB")
    cands = [("fused", fused)]
    for seed in FACE_SEEDS:
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=bpil, mask_image=mpil,
                   control_image=ctrl,
                   controlnet_conditioning_scale=CN_SCALE,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8,
                   guidance_scale=GUIDANCE,
                   strength=FACE_STRENGTH,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr74_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cands.append((f"s{seed}", g))
        print(f"[r74 {name}] {time.time() - t0:.0f}s", flush=True)

    n = len(cands)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in cands:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr74_sheet.png"), sheet)
    print("[r74] sheet saved", flush=True)


if __name__ == "__main__":
    main()
