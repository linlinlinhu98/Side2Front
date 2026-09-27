"""R77 (user on R76: 脸和五官是正了, 但是头部有些倾斜, 表情
还是不对, 不太像原图):

Two fixes:
1. HEAD LEAN: hair-axis measurement on _r76_straight shows the
   crown sits ~12px right of the lower face (~4 deg lean).
   Rotate the head band around the chin pivot (258, 370) so
   the crown moves LEFT ~19px at y=100 - implemented as a
   per-row horizontal shift dx(y) = -ROT*(370-y) with the
   y380-430 ramp (small-angle rotation; chin does not move).
2. EXPRESSION (original = calm/listless/flat):
   - brows made PERFECTLY FLAT (R76's inner-end-low arc read
     as worried/stern)
   - eyes more closed: lid line lowered to cover ~45% of the
     iris, iris r7 (R76 was ~20% covered -> too open/earnest)
   - mouth unchanged (neutral, approved)

Base: _r76_straight.png (head axis already aligned). 4 seeds.
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
from gen_face_r75 import (PROMPT, NEG, LORA_DIR, TRIGGER,
                          LORA_SCALE, SEEDS, STRENGTH, CN_SCALE,
                          IP_SCALE, GUIDANCE, CX,
                          MASK_ELL, _mask, _scale_lora)  # noqa: E402

ROT = 0.070                       # rad (~4 deg), crown moves LEFT
RAMP = (380, 430)
PIVOT_Y = 370


def _rotate_head(img):
    H, W = img.shape
    dst_x = np.arange(W, dtype=np.float32)[None, :]
    ys = np.arange(H, dtype=np.float32)[:, None]
    # mx = dst + dx samples from the right -> content moves
    # LEFT for dx > 0 (verified empirically in R76)
    dx = ROT * (PIVOT_Y - ys)
    dx = np.where(ys <= RAMP[0], dx,
                  np.where(ys >= RAMP[1], 0.0,
                           dx * (RAMP[1] - ys)
                           / (RAMP[1] - RAMP[0])))
    mx = (dst_x + dx).astype(np.float32)
    my = np.repeat(ys, W, axis=1).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=255)


def _guide(base):
    g = _lineart(base)
    y0, y1 = 245, 362
    right = g[y0:y1, CX:CX + 108]
    g[y0:y1, CX - 108:CX] = right[:, ::-1]
    g[246:298, 148:368] = 255     # brows + eyes
    g[290:332, 228:292] = 255     # nose
    g[330:364, 222:298] = 255     # mouth
    # FLAT relaxed tapered brows (no inner-end drop)
    for sgn in (-1, 1):
        xs = [CX + sgn * d for d in range(30, 101, 10)]
        ys = [253, 253, 252, 252, 252, 253, 253, 254]
        ths = [2, 3, 3, 2, 2, 1, 1]
        for i in range(len(xs) - 1):
            cv2.line(g, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                     85, ths[i], cv2.LINE_AA)
    # half-closed flat eyes: lid covers ~45% of iris
    for cx in (196, 320):
        cv2.circle(g, (cx, 284), 7, 95, -1, cv2.LINE_AA)
        cv2.rectangle(g, (cx - 32, 266), (cx + 32, 280), 255,
                      -1)
        pts = [(cx - 31, 281), (cx - 12, 278), (cx + 12, 278),
               (cx + 31, 281)]
        for i in range(3):
            cv2.line(g, pts[i], pts[i + 1], 45, 3, cv2.LINE_AA)
        cv2.ellipse(g, (cx, 289), (23, 8), 0, 30, 150, 135, 1,
                    cv2.LINE_AA)
    # nose (approved)
    cv2.line(g, (CX, 296), (CX, 311), 135, 1, cv2.LINE_AA)
    cv2.ellipse(g, (CX, 314), (5, 4), 0, 10, 170, 110, 1,
                cv2.LINE_AA)
    for sgn in (-1, 1):
        nx = CX + sgn * 8
        cv2.ellipse(g, (nx, 320), (3, 2), 0,
                    210 if sgn < 0 else 150,
                    340 if sgn < 0 else 380, 85, 1, cv2.LINE_AA)
    # neutral mouth (approved)
    cv2.line(g, (CX - 15, 339), (CX + 15, 339), 90, 1,
             cv2.LINE_AA)
    cv2.ellipse(g, (CX, 345), (9, 4), 0, 25, 155, 150, 1,
                cv2.LINE_AA)
    cv2.ellipse(g, (CX, 354), (7, 3), 0, 30, 150, 170, 1,
                cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "_r76_straight.png"), 0)
    assert base is not None
    H, W = base.shape
    straight = _rotate_head(base)
    guide = _guide(straight)
    mask = _mask((H, W))
    init = straight.copy()
    cx, cy, ax, ay = MASK_ELL
    cv2.ellipse(init, (cx, cy), (ax - 6, ay - 6), 0, 0, 360,
                255, -1)
    cv2.imwrite(os.path.join(OUT, "_r77_straight.png"), straight)
    cv2.imwrite(os.path.join(OUT, "_r77_guide.png"), guide)
    bpil = Image.fromarray(init).convert("RGB")
    mpil = Image.fromarray(mask).convert("RGB")
    ctrl = Image.fromarray(guide).convert("RGB")

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[170:300, 100:320]),
            Image.fromarray(rgb[240:420, 20:220])]

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
    _scale_lora(pipe.unet, LORA_SCALE)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(IP_SCALE)

    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    cands = [("rot", straight)]
    for seed in SEEDS:
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=bpil, mask_image=mpil,
                   control_image=ctrl,
                   controlnet_conditioning_scale=CN_SCALE,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8,
                   guidance_scale=GUIDANCE, strength=STRENGTH,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr77_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cands.append((f"s{seed}", g))
        print(f"[r77 {name}] {time.time() - t0:.0f}s", flush=True)

    n = len(cands)
    crops = [im[240:520, 110:410] for _, im in cands]
    ch, cw = crops[0].shape
    zs = np.full((ch, cw * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), c in zip(cands, crops):
        p = c.copy()
        cv2.putText(p, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, 0, 2, cv2.LINE_AA)
        zs[:, x:x + cw] = p
        x += cw + 10
    cv2.imwrite(os.path.join(OUT, "_genr77_faces.png"), zs)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in cands:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr77_sheet.png"), sheet)
    print("[r77] sheets saved", flush=True)


if __name__ == "__main__":
    main()
