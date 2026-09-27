"""R75 (user: 嘴部不像/眉毛不像/脸还是倾斜/只看中间/原图鼻子
嘴部比较写实有嘴唇有鼻孔/要传递原图每个部分每个像素):

Pixel-level spec from _feat_study.png (original, grid=10px):
- brow: 85px long, thin inner -> thicker middle -> pointed tip,
  slight arc, pencil gray (NOT a uniform bar)
- eye: 60px long, NARROW (height/len ~0.3), THICK straight
  upper lash, iris ~40% covered
- nose: realistic - bridge, small rounded tip, DRAWN NOSTRILS
- mouth: realistic - protruding upper lip, dark lip line,
  LOWER-LIP volume shadow, corners slightly down

R75 on genr74_fused (composition approved implicitly):
- guide: symmetrize face (mirror right->left), then draw the
  above spec at fused-image coords (measured _fused_grid.png)
- init face interior WHITENED -> kills the residual 3/4 tonal
  shading that kept reading as a turned/tilted face
- IP-Adapter Plus fed FOUR crops [full + head + eye/brow +
  nose/mouth] = 64 tokens: the original's own feature pixels
  condition the features being redrawn
- prompt names the realistic structures (nostrils, lips);
  negatives ban dot nose / line mouth
- LoRA x0.8 (its training data IS those realistic strokes)
4 seeds.
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
LORA_SCALE = 0.8

PROMPT = (f"{TRIGGER}, 1boy face, front view, facing forward, "
          "looking at viewer, symmetrical face, monochrome, "
          "pencil sketch, soft shading, long narrow eyes, "
          "thick upper eyelashes, relaxed half-closed eyes, "
          "long tapered eyebrows, detailed nose, nostrils, "
          "defined lips, upper lip, lower lip, closed mouth, "
          "calm gentle expression, weary, blush, white "
          "background")
NEG = ("3/4 view, side view, looking away, tilted head, "
       "asymmetrical face, dot nose, simple nose, line mouth, "
       "no lips, thin mouth, upturned eyes, fox eyes, big "
       "round eyes, shiny eyes, thick blocky eyebrows, cat "
       "mouth, smirk, shota, cute, open mouth, smile, color, "
       "colored, photo, lowres, blurry, bad anatomy, "
       "watermark")

SEEDS = [7, 42, 998, 123]
STRENGTH = 0.85
CN_SCALE = 0.90
IP_SCALE = 0.9
GUIDANCE = 2.0
CX = 258
MASK_ELL = (258, 312, 112, 96)


def _guide(fused):
    g = _lineart(fused)
    # 1) symmetrize face interior + jaw: mirror right -> left
    y0, y1 = 245, 362
    right = g[y0:y1, CX:CX + 108]
    g[y0:y1, CX - 108:CX] = right[:, ::-1]
    # 2) whiten old features
    g[246:298, 148:368] = 255     # brows + eyes
    g[290:332, 228:292] = 255     # nose
    g[330:364, 222:298] = 255     # mouth
    # 3) long tapered brows (70px, inner thin -> mid thick ->
    # pointed tip, slight arc), pencil gray
    for sgn in (-1, 1):
        xs = [CX + sgn * d for d in range(30, 101, 10)]
        ys = [257, 255, 253, 252, 251, 251, 252, 253]
        ths = [2, 3, 3, 2, 2, 1, 1]
        for i in range(len(xs) - 1):
            cv2.line(g, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                     85, ths[i], cv2.LINE_AA)
    # 4) long narrow eyes: THICK straight upper lash, iris 40%
    # covered, gaze dead center
    for cx in (196, 320):
        cv2.circle(g, (cx, 283), 8, 95, -1, cv2.LINE_AA)
        cv2.rectangle(g, (cx - 32, 266), (cx + 32, 277), 255,
                      -1)
        pts = [(cx - 31, 280), (cx - 12, 276), (cx + 12, 276),
               (cx + 31, 280)]
        for i in range(3):
            cv2.line(g, pts[i], pts[i + 1], 45, 3, cv2.LINE_AA)
        cv2.ellipse(g, (cx, 288), (24, 8), 0, 30, 150, 135, 1,
                    cv2.LINE_AA)
    # 5) realistic nose: bridge, tip, TWO NOSTRILS, ala wings
    cv2.line(g, (CX, 296), (CX, 311), 135, 1, cv2.LINE_AA)
    cv2.ellipse(g, (CX, 314), (6, 4), 0, 10, 170, 105, 1,
                cv2.LINE_AA)
    for sgn in (-1, 1):
        nx = CX + sgn * 9
        cv2.ellipse(g, (nx, 320), (4, 3), 0,
                    200 if sgn < 0 else 160,
                    340 if sgn < 0 else 380, 60, 2, cv2.LINE_AA)
        cv2.ellipse(g, (nx + sgn * 6, 317), (5, 6), 0,
                    270, 340 if sgn < 0 else 380, 115, 1,
                    cv2.LINE_AA)
    # 6) realistic mouth: lip line (corners slightly down) +
    # lower-lip volume + lip-chin groove
    pts = [(CX - 20, 340), (CX - 9, 338), (CX + 9, 338),
           (CX + 20, 340)]
    for i in range(3):
        cv2.line(g, pts[i], pts[i + 1], 70, 2, cv2.LINE_AA)
    cv2.ellipse(g, (CX, 346), (11, 5), 0, 25, 155, 145, 1,
                cv2.LINE_AA)
    cv2.ellipse(g, (CX, 356), (8, 4), 0, 30, 150, 165, 1,
                cv2.LINE_AA)
    return g


def _mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cx, cy, ax, ay = MASK_ELL
    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 12)


def _scale_lora(unet, factor):
    for m in unet.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict) and "default" in sc:
            sc["default"] = sc["default"] * factor


def main():
    fused = cv2.imread(os.path.join(OUT, "genr74_fused.png"), 0)
    assert fused is not None
    H, W = fused.shape
    guide = _guide(fused)
    mask = _mask((H, W))
    init = fused.copy()
    cx, cy, ax, ay = MASK_ELL
    cv2.ellipse(init, (cx, cy), (ax - 6, ay - 6), 0, 0, 360,
                255, -1)          # whiten face interior in init
    cv2.imwrite(os.path.join(OUT, "_r75_guide.png"), guide)
    cv2.imwrite(os.path.join(OUT, "_r75_init.png"), init)
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
    print(f"[r75] ip embeds {tuple(emb.shape)} (4 refs)",
          flush=True)

    cands = [("fused", fused)]
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
        name = f"genr75_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cands.append((f"s{seed}", g))
        print(f"[r75 {name}] {time.time() - t0:.0f}s", flush=True)

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
    cv2.imwrite(os.path.join(OUT, "_genr75_faces.png"), zs)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in cands:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr75_sheet.png"), sheet)
    print("[r75] sheets saved", flush=True)


if __name__ == "__main__":
    main()
