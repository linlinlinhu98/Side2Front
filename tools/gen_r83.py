"""R83 (user: 嘴还是向下弯/衣服纹理也不像):

6x zoom evidence (_mouth_cloth_check.png):
1. The FROWN is not the lip line (flat) but a ∩ arc the model
   adds ABOVE the lips (upper-lip/nasolabial shadow). Root
   causes: Counterfeit's 'closed mouth' prior + the s2fstyle
   LoRA which MEMORIZED the profile's descending lip line
   ('this character's mouth = downsloped') at scale 0.8.
   Fix: tiny mouth-zone repaint with LoRA 0.2 (weakens the
   profile-mouth memory), CN 1.0, 'one straight line' guide,
   frown/upper-lip-shadow negatives.
2. CLOTHES went bald: original = dense parallel hatching +
   rich folds (ring pull visible); ours = two clean lines on
   empty white after 15+ repaint rounds. Fix: torso texture
   pass, LoRA 1.0 (the hatching lives in it), CN 0.9 locks
   zipper/folds, strength 0.45.

Stage A: mouth micro-pass on genr82_s7_e7, 2 seeds -> pick
flattest. Stage B: torso texture pass on it, 2 seeds.
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
CX = 258

MOUTH_PROMPT = (f"{TRIGGER}, neutral expressionless mouth, "
                "straight closed lips, one line mouth, "
                "monochrome, pencil sketch, soft shading, "
                "white background")
MOUTH_NEG = ("frown, sad, downturned mouth, pout, upper lip "
             "shadow, philtrum, open mouth, smile, smirk, "
             "color, colored, lowres, blurry, watermark")

CLOTH_PROMPT = (f"{TRIGGER}, zip-up hoodie, full-length "
                "zipper, ring zipper pull, hood down, "
                "parallel hatching, fine pencil hatching, "
                "sketch texture, fabric folds, monochrome, "
                "pencil sketch, soft shading, white "
                "background")
CLOTH_NEG = ("smooth, clean, digital painting, flat, "
             "drawstrings, pullover, thick outlines, cel "
             "shading, color, colored, photo, lowres, "
             "blurry, watermark")


def _scale_lora(unet, factor):
    for m in unet.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict) and "default" in sc:
            sc["default"] = sc["default"] * factor


def _mouth_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CX, 338), (40, 20), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 6)


def _mouth_guide(base):
    g = _lineart(base)
    g[326:352, 222:296] = 255
    cv2.line(g, (CX - 14, 340), (CX + 14, 340), 55, 2,
             cv2.LINE_AA)
    return g


def _torso_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[410:, :] = 255
    cv2.ellipse(m, (CX, 410), (60, 26), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def main():
    base = cv2.imread(os.path.join(OUT, "genr82_s7_e7.png"), 0)
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

    # ---- stage A: mouth (LoRA 0.2, CN 1.0) ----
    from peft import PeftModel
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
    _scale_lora(pipe.unet, 0.2)
    mmask = _mouth_mask((H, W))
    mguide = _mouth_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r83_mguide.png"), mguide)
    minit = base.copy()
    minit[324:354, 218:300] = 255
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
        name = f"genr83_m{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        mouths.append((seed, g))
        print(f"[r83 {name}] {time.time() - t0:.0f}s", flush=True)

    # ---- stage B: torso texture (LoRA 1.0, CN 0.9) ----
    _scale_lora(pipe.unet, 5.0)   # 0.2 -> 1.0
    tmask = _torso_mask((H, W))
    finals = []
    for seed, mimg in mouths:
        tguide = _lineart(mimg)
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
                   strength=0.45,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr83_m{seed}_cloth.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        finals.append((f"m{seed}+cloth", g))
        print(f"[r83 {name}] {time.time() - t0:.0f}s", flush=True)

    # sheets: mouth zoom + cloth zoom + full
    panels = [("base", base)] + [(f"m{s}", m) for s, m in mouths] \
             + finals
    n = len(panels)
    W2 = 520
    sheet = np.full((660, W2 * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W2] = p
        x += W2 + 10
    cv2.imwrite(os.path.join(OUT, "_genr83_sheet.png"), sheet)
    mz = [cv2.resize(im[324:358, 218:300], None, fx=5, fy=5,
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
    cv2.imwrite(os.path.join(OUT, "_genr83_mouths.png"), msheet)
    cz = [cv2.resize(im[440:620, 80:440], None, fx=1.4, fy=1.4,
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
    cv2.imwrite(os.path.join(OUT, "_genr83_cloths.png"), csheet)
    print("[r83] sheets saved", flush=True)


if __name__ == "__main__":
    main()
