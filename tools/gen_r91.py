"""R91 (user: 究竟看没看原图/相似度怎么到75%):

Root cause found by measurement: OUR face h/w = 0.64 (round/
wide) vs classmate's 1.54 (long oval) - the original's face is
a slim long oval. All the feature fixes sat on a wrongly
SHAPED face => persistent 'not the same person' + 'too young'.

Stage A: face repaint on genr90_c7 with a SLIMMED jaw guide
(jaw sides pulled in ~14px, chin kept at y368), all approved
feature specs unchanged (R80 eyes / tapered brows / R89-style
nose left to the model / R90 lips left to the model).
Prompt: slim oval face; neg: round face, wide face, baby face.
4 seeds.

Stage B: hair-region refresh (mask = hair only) on the best
face seed: strength 0.40, LoRA 1.0, IP head - restores the
VAE-worn hair texture (hair subscore 0.45) without touching
face/clothes. 1 seed.
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

FACE_PROMPT = (f"{TRIGGER}, 1boy face, front view, facing "
               "forward, looking at viewer, symmetrical "
               "face, slim oval face, narrow jaw, "
               "monochrome, pencil sketch, soft shading, "
               "long narrow eyes, single eyelid, half-closed "
               "eyes, straight tapered eyebrows, detailed "
               "nose, nostrils, full lips, closed mouth, "
               "neutral mouth, calm expression, blush, "
               "white background")
FACE_NEG = ("round face, wide face, baby face, shota, cute, "
            "3/4 view, side view, looking away, tilted "
            "head, asymmetrical face, double eyelid, big "
            "round eyes, shiny eyes, thick blocky "
            "eyebrows, dot nose, frown, downturned "
            "mouth, line mouth, thin lips, open mouth, "
            "smile, color, colored, photo, lowres, "
            "blurry, bad anatomy, watermark")

HAIR_PROMPT = (f"{TRIGGER}, short black hair, pencil "
               "sketch hatching, fine hair strands, "
               "monochrome, pencil sketch, soft shading, "
               "white background")
HAIR_NEG = ("smooth, digital painting, color, colored, "
            "photo, lowres, blurry, watermark")

FACE_SEEDS = [7, 42, 998, 123]
FACE_STRENGTH = 0.85
CN_SCALE = 0.95
GUIDANCE = 3.0
MASK_ELL = (258, 312, 116, 100)


def _face_guide(base):
    g = _lineart(base)
    y0, y1 = 245, 370
    right = g[y0:y1, CX:CX + 112]
    g[y0:y1, CX - 112:CX] = right[:, ::-1]
    # whiten face interior incl. jaw, then redraw slim
    g[246:300, 150:366] = 255
    g[290:370, 200:316] = 255
    # SLIM jaw: sides pulled in ~14px vs before
    jaw = [(206, 296), (214, 322), (226, 348), (242, 362),
           (258, 368)]
    for sgn in (-1, 1):
        pts = [(CX + sgn * (CX - x), y) for x, y in jaw]
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 60, 2, cv2.LINE_AA)
    # tapered flat brows (approved)
    for sgn in (-1, 1):
        xs = [CX + sgn * d for d in range(28, 89, 10)]
        ys = [256, 255, 254, 254, 255, 256, 256]
        ths = [2, 2, 2, 2, 1, 1]
        for i in range(len(xs) - 1):
            cv2.line(g, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                     90, ths[i], cv2.LINE_AA)
    # closed almond monolid eyes (approved)
    for cx in (196, 320):
        cv2.circle(g, (cx, 287), 5, 100, -1, cv2.LINE_AA)
        cv2.rectangle(g, (cx - 32, 266), (cx + 32, 282), 255,
                      -1)
        up = [(cx - 31, 284), (cx - 12, 281), (cx + 12, 281),
              (cx + 31, 284)]
        for i in range(3):
            cv2.line(g, up[i], up[i + 1], 45, 3, cv2.LINE_AA)
        lo = [(cx - 31, 284), (cx - 12, 291), (cx + 12, 291),
              (cx + 31, 284)]
        for i in range(3):
            cv2.line(g, lo[i], lo[i + 1], 90, 1, cv2.LINE_AA)
    # nose: minimal (2 faint nostril dots + bridge, R89 spec)
    cv2.ellipse(g, (CX - 7, 321), (2, 1), 0, 0, 360, 120, 1,
                cv2.LINE_AA)
    cv2.ellipse(g, (CX + 7, 321), (2, 1), 0, 0, 360, 120, 1,
                cv2.LINE_AA)
    cv2.line(g, (CX - 2, 296), (CX - 3, 310), 165, 1,
             cv2.LINE_AA)
    # mouth: single level line (R90 lips left to the model
    # via prompt)
    cv2.line(g, (CX - 14, 340), (CX + 14, 340), 65, 1,
             cv2.LINE_AA)
    return g


def _face_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cx, cy, ax, ay = MASK_ELL
    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 10)


def _hair_mask(shape, face_ell):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (258, 190), (152, 152), 0, 0, 360, 255, -1)
    f = np.zeros((H, W), np.uint8)
    cx, cy, ax, ay = face_ell
    cv2.ellipse(f, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    m[f > 0] = 0
    m[:30, :] = 0
    m[400:, :] = 0
    return cv2.GaussianBlur(m, (0, 0), 10)


def _load(lcm, ckpt, cn_path=None):
    from diffusers import (ControlNetModel, LCMScheduler,
                           StableDiffusionControlNetInpaintPipeline,
                           StableDiffusionInpaintPipeline)
    from peft import PeftModel
    if cn_path:
        cn = ControlNetModel.from_pretrained(
            cn_path, torch_dtype=torch.float32)
        pipe = StableDiffusionControlNetInpaintPipeline \
            .from_single_file(ckpt, controlnet=cn,
                              torch_dtype=torch.float32,
                              safety_checker=None)
    else:
        pipe = StableDiffusionInpaintPipeline.from_single_file(
            ckpt, torch_dtype=torch.float32, safety_checker=None)
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
    return pipe


def main():
    base = cv2.imread(os.path.join(OUT, "genr90_c7.png"), 0)
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

    # ---- stage A: slim face ----
    pipe = _load(lcm, ckpt, "lllyasviel/control_v11p_sd15_lineart")
    _scale_lora(pipe.unet, 0.8)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)
    guide = _face_guide(base)
    mask = _face_mask((H, W))
    init = base.copy()
    cx, cy, ax, ay = MASK_ELL
    cv2.ellipse(init, (cx, cy), (ax - 6, ay - 6), 0, 0, 360,
                255, -1)
    cv2.imwrite(os.path.join(OUT, "_r91_guide.png"), guide)
    bpil = Image.fromarray(init).convert("RGB")
    mpil = Image.fromarray(mask).convert("RGB")
    ctrl = Image.fromarray(guide).convert("RGB")
    cands = []
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
        name = f"genr91_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cands.append((seed, g))
        print(f"[r91 {name}] {time.time() - t0:.0f}s", flush=True)
    del pipe
    import gc
    gc.collect()

    # ---- stage B: hair refresh on the first seed's result ----
    pipe2 = _load(lcm, ckpt)
    _scale_lora(pipe2.unet, 1.0)
    with torch.no_grad():
        emb2, unc2 = pipe2.encode_image(
            refs[:2], "cpu", 1, output_hidden_states=True)
        emb2 = torch.cat([unc2.unsqueeze(0), emb2.unsqueeze(0)],
                         dim=0)
    hmask = _hair_mask((H, W), MASK_ELL)
    hmpil = Image.fromarray(hmask).convert("RGB")
    finals = []
    for seed, g in cands[:2]:
        t0 = time.time()
        res = pipe2(prompt=HAIR_PROMPT, negative_prompt=HAIR_NEG,
                    image=Image.fromarray(g).convert("RGB"),
                    mask_image=hmpil,
                    ip_adapter_image_embeds=[emb2],
                    height=664, width=520,
                    num_inference_steps=8,
                    guidance_scale=2.0, strength=0.40,
                    generator=torch.Generator("cpu")
                    .manual_seed(42)).images[0]
        h = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        h = cv2.resize(h, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr91_s{seed}_hair.png"
        cv2.imwrite(os.path.join(OUT, name), h)
        finals.append((f"s{seed}+hair", h))
        print(f"[r91 {name}] {time.time() - t0:.0f}s", flush=True)

    panels = [("base", base)] + [(f"s{s}", g) for s, g in cands] \
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
    cv2.imwrite(os.path.join(OUT, "_genr91_sheet.png"), sheet)
    crops = [im[230:520, 110:410] for _, im in panels]
    ch, cw = crops[0].shape
    zs = np.full((ch, cw * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), c in zip(panels, crops):
        p = c.copy()
        cv2.putText(p, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, 0, 2, cv2.LINE_AA)
        zs[:, x:x + cw] = p
        x += cw + 10
    cv2.imwrite(os.path.join(OUT, "_genr91_faces.png"), zs)
    print("[r91] sheets saved", flush=True)


if __name__ == "__main__":
    main()
