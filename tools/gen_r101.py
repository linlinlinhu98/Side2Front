"""R101 (user: 下巴稍好/领子太高/头发很奇怪/耳朵还是没改;
'参考分析报告的参考建议'):

KEY LEVER from the reports, untried until now: swap the
sampler from LCM-8 (fast, crude uniform lines) to
DPM++ 2M Karras @ 26 steps (the reports' explicit recipe).
This route pre-generates OFFLINE so runtime doesn't matter.

Three passes on genr100_s7_c, all DPM-26:
A. FACE: ears - smaller, simpler, half-hidden under hair
   (single soft C arc, light), everything else kept
B. COLLAR: rolls moved DOWN (y415-455 -> y445-475), neck
   visible, hood resting ON the shoulders (not hugging the
   chin) + zipper pull moves down accordingly
C. HAIR: pencil-texture restoration (0.45) so the hair
   matches the pencil style of face/clothes
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
from gen_r97 import _hair_mask  # noqa: E402
from gen_r99 import FACE_PROMPT, FACE_NEG  # noqa: E402

COLLAR_PROMPT = (f"{TRIGGER}, zip-up hoodie, hood down, low "
                 "collar, hoodie collar resting on "
                 "shoulders, neck visible, metal zipper "
                 "pull, ring zipper pull, fabric folds, "
                 "pencil sketch, soft shading, monochrome, "
                 "white background")
COLLAR_NEG = ("high collar, turtleneck, collar covering "
              "neck, pullover, drawstrings, color, "
              "colored, lowres, blurry, watermark")

HAIR_PROMPT = (f"{TRIGGER}, short hair, pencil sketch "
               "hatching, fine hair strands, sketch "
               "texture, hair clumps, monochrome, pencil "
               "sketch, soft shading, white background")
HAIR_NEG = ("smooth, solid black hair, helmet hair, "
            "digital painting, color, colored, photo, "
            "lowres, blurry, watermark")

STEPS = 26
GUIDANCE = 3.5


def _load_dpm(lcm_unused, ckpt, cn_path):
    from diffusers import (ControlNetModel,
                           DPMSolverMultistepScheduler,
                           StableDiffusionControlNetInpaintPipeline)
    from peft import PeftModel
    cn = ControlNetModel.from_pretrained(cn_path,
                                         torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetInpaintPipeline \
        .from_single_file(ckpt, controlnet=cn,
                          torch_dtype=torch.float32,
                          safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.9)
    return pipe


def _face_guide(base):
    g = _lineart(base)
    # ears: wipe, redraw tiny and tucked
    g[262:318, 152:210] = 255
    g[262:318, 306:364] = 255
    for sgn in (-1, 1):
        ex = CX + sgn * 88
        cv2.ellipse(g, (ex, 292), (9, 14), 0, 310, 150, 105,
                    1, cv2.LINE_AA)
    return g


def _face_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CX - 88, 292), (26, 30), 0, 0, 360, 255,
                -1)
    cv2.ellipse(m, (CX + 88, 292), (26, 30), 0, 0, 360, 255,
                -1)
    return cv2.GaussianBlur(m, (0, 0), 6)


def _collar_guide(base):
    g = _lineart(base)
    g[396:496, 128:392] = 255
    # neck visible y396-440
    cv2.line(g, (CX - 22, 392), (CX - 24, 438), 95, 1,
             cv2.LINE_AA)
    cv2.line(g, (CX + 22, 392), (CX + 24, 438), 95, 1,
             cv2.LINE_AA)
    # LOW collar rolls resting on the shoulders
    for sgn in (-1, 1):
        pts = [(CX + sgn * 24, 444), (CX + sgn * 62, 458),
               (CX + sgn * 100, 472)]
        for i in range(len(pts) - 1):
            cv2.line(g, pts[i], pts[i + 1], 75, 2, cv2.LINE_AA)
        pts2 = [(x, y + 9) for x, y in pts]
        for i in range(len(pts2) - 1):
            cv2.line(g, pts2[i], pts2[i + 1], 55, 2,
                     cv2.LINE_AA)
    # hood mass low behind the neck
    cv2.ellipse(g, (CX, 452), (80, 22), 0, 190, 350, 85, 1,
                cv2.LINE_AA)
    # zipper pull lower + zipper start
    cv2.circle(g, (CX, 478), 6, 45, 2, cv2.LINE_AA)
    cv2.rectangle(g, (CX - 3, 466), (CX + 3, 473), 45, 1)
    cv2.line(g, (CX, 484), (CX, 656), 55, 2, cv2.LINE_AA)
    return g


def _collar_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    m[394:500, :] = 255
    cv2.ellipse(m, (CX, 394), (62, 18), 0, 0, 360, 0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def main():
    base = cv2.imread(os.path.join(OUT, "genr100_s7_c.png"), 0)
    assert base is not None
    H, W = base.shape
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[280:, :])]
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    pipe = _load_dpm(None, ckpt,
                     "lllyasviel/control_v11p_sd15_lineart")
    _scale_lora(pipe.unet, 0.8)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    # ---- A: ears ----
    fguide = _face_guide(base)
    fmask = _face_mask((H, W))
    faces = []
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=FACE_PROMPT, negative_prompt=FACE_NEG,
                   image=Image.fromarray(base).convert("RGB"),
                   mask_image=Image.fromarray(fmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(fguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.9,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=GUIDANCE, strength=0.75,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr101_e{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        faces.append((seed, g))
        print(f"[r101 {name}] {time.time() - t0:.0f}s",
              flush=True)

    # ---- B: lower collar on each ----
    cguide = _collar_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r101_cguide.png"), cguide)
    cmask = _collar_mask((H, W))
    combos = []
    for seed, fimg in faces:
        t0 = time.time()
        res = pipe(prompt=COLLAR_PROMPT,
                   negative_prompt=COLLAR_NEG,
                   image=Image.fromarray(fimg).convert("RGB"),
                   mask_image=Image.fromarray(cmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(cguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.9,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=GUIDANCE, strength=0.7,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr101_e{seed}_c.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        combos.append((f"e{seed}c", g))
        print(f"[r101 {name}] {time.time() - t0:.0f}s",
              flush=True)

    # ---- C: pencil hair on the first combo ----
    hmask = _hair_mask((H, W))
    t0 = time.time()
    res = pipe(prompt=HAIR_PROMPT, negative_prompt=HAIR_NEG,
               image=Image.fromarray(combos[0][1]).convert("RGB"),
               mask_image=Image.fromarray(hmask).convert("RGB"),
               control_image=Image.fromarray(
                   _lineart(combos[0][1])).convert("RGB"),
               controlnet_conditioning_scale=0.6,
               ip_adapter_image_embeds=[emb],
               height=664, width=520,
               num_inference_steps=STEPS,
               guidance_scale=GUIDANCE, strength=0.45,
               generator=torch.Generator("cpu")
               .manual_seed(7)).images[0]
    g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(OUT, "genr101_final.png"), g)
    print(f"[r101 final] {time.time() - t0:.0f}s", flush=True)

    panels = [("base", base)] + \
             [(f"e{s}", im) for s, im in faces] + combos + \
             [("final", g)]
    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr101_sheet.png"), sheet)
    print("[r101] sheets saved", flush=True)


if __name__ == "__main__":
    main()
