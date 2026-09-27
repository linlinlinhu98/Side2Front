"""R78 (user on R77: 脸和五官又向右倾斜/头部没有摆正/眉毛弧度
不对/嘴巴弯曲幅度不对/还是显得太幼):

Defects confirmed in _definitive_study.png (3-way visual):
1. TOO YOUNG = double-eyelid crease + big round fully-open
   iris + round short face. Original: MONOLID narrow eyes,
   iris half covered, slimmer longer face.
2. FROWN = rendered mouth is a ∩ arc; classmate/original spec
   = short straight dark level line. Contributor: the IP
   nose+mouth crop was a PROFILE (its lip line descends) -
   removed from refs.
3. RIGHT-TIP READ = crown tuft right of center + fuller right
   hair + diagonal bangs. Hair is OUTSIDE the face mask, so
   previous face-only rounds could never fix it.

Stage 1 (hair): mirror the fuller RIGHT hair lineart onto the
left inside a hair-only mask, whiten the tuft + diagonal bang
strokes, repaint with 'symmetrical bangs'. 1 seed.
Stage 2 (face): monolid guide (brows low/thick/flat, thick
lash covering half the iris, NO crease space), slimmer longer
chin, short straight dark mouth (no stacked arcs), IP refs
[full+head+eye/brow], monolid/neutral-mouth prompt. 4 seeds.
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
GUIDANCE = 2.0
CN_SCALE = 0.85
IP_SCALE = 0.9
CX = 258

HAIR_PROMPT = (f"{TRIGGER}, short black hair, symmetrical "
               "bangs, center part, monochrome, pencil "
               "sketch, soft shading, delicate thin lines, "
               "white background")
HAIR_NEG = ("asymmetrical hair, ahoge, tilted, side part, "
            "color, colored, photo, lowres, blurry, "
            "watermark")

FACE_PROMPT = (f"{TRIGGER}, 1boy face, front view, facing "
               "forward, looking at viewer, symmetrical "
               "face, monochrome, pencil sketch, soft "
               "shading, long narrow eyes, single eyelid, "
               "monolid, thick eyelashes, half-closed "
               "eyes, straight thick eyebrows, detailed "
               "nose, nostrils, neutral mouth, closed "
               "mouth, relaxed lips, calm expression, "
               "blush, white background")
FACE_NEG = ("3/4 view, side view, looking away, tilted "
            "head, asymmetrical face, double eyelid, "
            "eyelid crease, big round eyes, shiny eyes, "
            "dot nose, frown, downturned mouth, sad, "
            "worried, thick blocky eyebrows, cat mouth, "
            "smirk, shota, cute, baby face, small eyes, "
            "open mouth, smile, color, colored, photo, "
            "lowres, blurry, bad anatomy, watermark")

FACE_SEEDS = [7, 42, 998, 123]
FACE_STRENGTH = 0.85
HAIR_STRENGTH = 0.80
MASK_ELL = (258, 312, 112, 96)


def _hair_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (258, 190), (152, 152), 0, 0, 360, 255, -1)
    face = np.zeros((H, W), np.uint8)
    cv2.ellipse(face, (258, 312), (100, 80), 0, 0, 360, 255,
                -1)
    m[face > 0] = 0
    m[:30, :] = 0
    return cv2.GaussianBlur(m, (0, 0), 10)


def _hair_guide(base):
    g = _lineart(base)
    # mirror the fuller right half onto the left inside the
    # hair zone (y 36-330)
    right = g[36:330, CX:CX + 165]
    g[36:330, CX - 165:CX] = right[:, ::-1]
    # whiten the crown tuft + center bang diagonals so the
    # model redraws them symmetric
    g[36:78, 175:345] = 255
    g[222:262, 195:325] = 255
    return g


def _face_guide(base):
    g = _lineart(base)
    y0, y1 = 245, 366
    right = g[y0:y1, CX:CX + 108]
    g[y0:y1, CX - 108:CX] = right[:, ::-1]
    g[246:298, 148:368] = 255     # brows + eyes
    g[290:332, 228:292] = 255     # nose
    g[330:366, 222:298] = 255     # mouth + chin interior
    # FLAT THICK straight brows (original: thick, level)
    for sgn in (-1, 1):
        xi = CX + sgn * 28
        xo = CX + sgn * 94
        cv2.line(g, (xi, 256), (CX + sgn * 60, 254), 75, 3,
                 cv2.LINE_AA)
        cv2.line(g, (CX + sgn * 60, 254), (xo, 255), 75, 2,
                 cv2.LINE_AA)
    # MONOLID narrow eyes: thick lash covering half the iris,
    # no crease space (brow close above)
    for cx in (196, 320):
        cv2.circle(g, (cx, 286), 6, 100, -1, cv2.LINE_AA)
        cv2.rectangle(g, (cx - 32, 266), (cx + 32, 281), 255,
                      -1)
        pts = [(cx - 31, 283), (cx - 12, 280), (cx + 12, 280),
               (cx + 31, 283)]
        for i in range(3):
            cv2.line(g, pts[i], pts[i + 1], 45, 3, cv2.LINE_AA)
    # nose (approved): bridge, tip, small nostrils
    cv2.line(g, (CX, 296), (CX, 311), 135, 1, cv2.LINE_AA)
    cv2.ellipse(g, (CX, 314), (5, 4), 0, 10, 170, 110, 1,
                cv2.LINE_AA)
    for sgn in (-1, 1):
        nx = CX + sgn * 8
        cv2.ellipse(g, (nx, 320), (3, 2), 0,
                    210 if sgn < 0 else 150,
                    340 if sgn < 0 else 380, 85, 1, cv2.LINE_AA)
    # SHORT STRAIGHT dark level mouth (classmate/original)
    cv2.line(g, (CX - 14, 339), (CX + 14, 339), 70, 2,
             cv2.LINE_AA)
    cv2.ellipse(g, (CX, 346), (8, 3), 0, 25, 155, 155, 1,
                cv2.LINE_AA)
    # slimmer slightly longer chin
    cv2.ellipse(g, (CX, 328), (38, 42), 0, 22, 158, 95, 1,
                cv2.LINE_AA)
    return g


def _face_mask(shape):
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
    base = cv2.imread(os.path.join(OUT, "genr77_s7.png"), 0)
    assert base is not None
    H, W = base.shape

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[170:300, 100:320])]

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

    # ---- stage 1: hair straighten ----
    hguide = _hair_guide(base)
    hmask = _hair_mask((H, W))
    cv2.imwrite(os.path.join(OUT, "_r78_hguide.png"), hguide)
    cv2.imwrite(os.path.join(OUT, "_r78_hmask.png"), hmask)
    hinit = base.copy()
    hm_bin = cv2.GaussianBlur(hmask, (0, 0), 2) > 40
    hinit[hm_bin] = 255
    t0 = time.time()
    res = pipe(prompt=HAIR_PROMPT, negative_prompt=HAIR_NEG,
               image=Image.fromarray(hinit).convert("RGB"),
               mask_image=Image.fromarray(hmask).convert("RGB"),
               control_image=Image.fromarray(hguide)
               .convert("RGB"),
               controlnet_conditioning_scale=0.75,
               ip_adapter_image_embeds=[emb],
               height=664, width=520, num_inference_steps=8,
               guidance_scale=GUIDANCE, strength=HAIR_STRENGTH,
               generator=torch.Generator("cpu")
               .manual_seed(42)).images[0]
    hair = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
    hair = cv2.resize(hair, (W, H), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(OUT, "genr78_hair.png"), hair)
    print(f"[r78 hair] {time.time() - t0:.0f}s", flush=True)

    # ---- stage 2: anti-young face ----
    guide = _face_guide(hair)
    mask = _face_mask((H, W))
    init2 = hair.copy()
    cx, cy, ax, ay = MASK_ELL
    cv2.ellipse(init2, (cx, cy), (ax - 6, ay - 6), 0, 0, 360,
                255, -1)
    cv2.imwrite(os.path.join(OUT, "_r78_guide.png"), guide)
    bpil = Image.fromarray(init2).convert("RGB")
    mpil = Image.fromarray(mask).convert("RGB")
    ctrl = Image.fromarray(guide).convert("RGB")

    cands = [("hair", hair)]
    for seed in FACE_SEEDS:
        t0 = time.time()
        res = pipe(prompt=FACE_PROMPT, negative_prompt=FACE_NEG,
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
        name = f"genr78_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cands.append((f"s{seed}", g))
        print(f"[r78 {name}] {time.time() - t0:.0f}s", flush=True)

    n = len(cands)
    crops = [im[230:520, 100:420] for _, im in cands]
    ch, cw = crops[0].shape
    zs = np.full((ch, cw * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), c in zip(cands, crops):
        p = c.copy()
        cv2.putText(p, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, 0, 2, cv2.LINE_AA)
        zs[:, x:x + cw] = p
        x += cw + 10
    cv2.imwrite(os.path.join(OUT, "_genr78_faces.png"), zs)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in cands:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr78_sheet.png"), sheet)
    print("[r78] sheets saved", flush=True)


if __name__ == "__main__":
    main()
