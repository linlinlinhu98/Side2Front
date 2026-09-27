"""R73 (user on R72: 脸太小/眉毛不像/脸还是倾斜/不要看旁边
只看中间/脸不要往旁边转):

Root cause of R72's residual 3/4 look: only the inner features
were redrawn - the base head itself is a 3/4 head (one ear,
asymmetric jaw, one-sided cheek shading) and strength 0.80 kept
20% of it.

R73:
- SYMMETRIZE the head: mirror the right half of the face/jaw
  lineart onto the left (frontal by construction)
- features redrawn ~15% LARGER, to the measured original spec:
  long narrow almond eyes (44px, lid covers iris top), LONG
  tapered pencil brows (52px, thick inner -> point outer),
  defined small nose line, wider thin mouth
- bigger face mask (incl. jaw), strength 0.85, CN 0.90 so the
  old 3/4 shading is fully overwritten
- prompt: front view + facing forward; negatives: 3/4 view,
  side view, looking away
4 seeds (memory). Base: genr69_s2024_eye (approved clothes).
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

LORA_DIR = os.path.join(ROOT, "assets", "lora", "s2fstyle")
TRIGGER = "s2fstyle"
LORA_SCALE = 0.7

PROMPT = (f"{TRIGGER}, 1boy face, front view, facing forward, "
          "looking at viewer, symmetrical face, monochrome, "
          "pencil sketch, soft shading, delicate thin lines, "
          "long narrow almond eyes, relaxed half-closed eyes, "
          "straight tapered eyebrows, small straight nose, "
          "small closed lips, thin lips, soft oval face, "
          "rounded chin, calm gentle expression, weary, "
          "blush, white background")
NEG = ("3/4 view, side view, looking away, looking to the "
       "side, tilted head, asymmetrical face, upturned eyes, "
       "fox eyes, big round eyes, shiny eyes, thick blocky "
       "eyebrows, cat mouth, smirk, smug, pointy chin, small "
       "eyes, baby face, shota, cute, open mouth, smile, "
       "color, colored, photo, lowres, blurry, bad anatomy, "
       "watermark")

SEEDS = [7, 42, 998, 123]
STRENGTH = 0.85
CN_SCALE = 0.90
IP_SCALE = 0.9
GUIDANCE = 2.0
CX = 258
FACE = (258, 328, 100, 110)       # repaint mask ellipse


def _face_guide(base):
    g = _lineart(base)
    # 1) symmetrize the face/jaw: mirror right half -> left
    y0, y1 = 258, 378
    right = g[y0:y1, CX:CX + 84]
    g[y0:y1, CX - 84:CX] = right[:, ::-1]
    # 2) whiten the (now symmetric) old features
    g[268:322, 184:334] = 255     # brows + eyes
    g[312:352, 232:288] = 255     # nose
    g[350:380, 222:298] = 255     # mouth + chin interior
    # 3) LONG tapered brows: thick inner third -> pointed tip
    for sgn in (-1, 1):
        xi = CX + sgn * 62        # inner end x (196 / 320)
        xo = CX + sgn * 10        # outer end x (248 / 268)...
        # inner -> outer: y 282 -> 279 (nearly straight)
        pts = [(CX + sgn * d,
                282 - round(3 * d / 52)) for d in
               range(10, 63, 4)]
        for i, (x, y) in enumerate(pts[:-1]):
            th = 3 if i < 4 else (2 if i < 9 else 1)
            cv2.line(g, (x, y), pts[i + 1], 85, th, cv2.LINE_AA)
    # 4) long narrow almond eyes, centers y=304, 44px wide
    for cx in (216, 300):
        cv2.circle(g, (cx, 307), 7, 100, -1, cv2.LINE_AA)
        cv2.ellipse(g, (cx, 314), (22, 14), 0, 205, 335, 50, 2,
                    cv2.LINE_AA)
        cv2.rectangle(g, (cx - 21, 294), (cx + 21, 301), 255,
                      -1)
        cv2.line(g, (cx - 22, 302), (cx + 22, 302), 50, 1,
                 cv2.LINE_AA)
        cv2.ellipse(g, (cx, 309), (16, 8), 0, 35, 145, 130, 1,
                    cv2.LINE_AA)
    # 5) defined small straight nose
    cv2.line(g, (CX, 315), (CX, 333), 105, 1, cv2.LINE_AA)
    cv2.ellipse(g, (CX, 336), (5, 3), 0, 20, 160, 100, 1,
                cv2.LINE_AA)
    # 6) wider thin closed lips, corners slightly down
    cv2.line(g, (241, 361), (275, 361), 95, 1, cv2.LINE_AA)
    cv2.line(g, (241, 361), (238, 363), 95, 1, cv2.LINE_AA)
    cv2.line(g, (275, 361), (278, 363), 95, 1, cv2.LINE_AA)
    cv2.line(g, (248, 366), (268, 366), 150, 1, cv2.LINE_AA)
    # 7) soft symmetric chin
    cv2.ellipse(g, (CX, 330), (42, 38), 0, 25, 155, 95, 1,
                cv2.LINE_AA)
    return g


def _face_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cx, cy, ax, ay = FACE
    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 10)


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
    guide = _face_guide(base)
    mask = _face_mask((H, W))
    cv2.imwrite(os.path.join(OUT, "_face_guide3.png"), guide)
    bpil = Image.fromarray(base).convert("RGB")
    mpil = Image.fromarray(mask).convert("RGB")
    ctrl = Image.fromarray(guide).convert("RGB")

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :])]

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

    cands = [("base", base)]
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
        name = f"genface5_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cands.append((f"s{seed}", g))
        print(f"[r73 {name}] {time.time() - t0:.0f}s", flush=True)

    n = len(cands)
    crops = [im[210:480, 130:390] for _, im in cands]
    ch, cw = crops[0].shape
    zs = np.full((ch, cw * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), c in zip(cands, crops):
        p = c.copy()
        cv2.putText(p, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, 0, 2, cv2.LINE_AA)
        zs[:, x:x + cw] = p
        x += cw + 10
    cv2.imwrite(os.path.join(OUT, "_genface5_faces.png"), zs)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in cands:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genface5_sheet.png"), sheet)
    print("[r73] sheets saved", flush=True)


if __name__ == "__main__":
    main()
