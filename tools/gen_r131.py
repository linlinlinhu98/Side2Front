"""R131 (one-at-a-time #2 after restart: 眼睛看着上面没有平视):

GENTLE edit-mode eye-band pass (strength 0.55, no style
regeneration): iris placed in the LOWER half of the eye,
upper lid covers the top ~50% => level gaze. Base:
genr130_s7 (mouth fixed). DPM-26, CN 1.0, LoRA 0.3.
2 seeds.
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
from gen_r101 import _load_dpm  # noqa: E402
from gen_r102 import STEPS  # noqa: E402

EYE_PROMPT = (f"{TRIGGER}, level gaze, looking straight "
              "ahead, half-closed eyes, calm eyes, "
              "single eyelid, pencil sketch, "
              "monochrome, white background")
EYE_NEG = ("looking up, upturned eyes, big shiny "
           "eyes, excited, round eyes, double "
           "eyelid, color, colored, lowres, blurry, "
           "watermark")


def _eye_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CX, 224), (100, 24), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 7)


def _eye_guide(base):
    g = _lineart(base)
    g[196:248, 150:366] = 255
    for sgn in (-1, 1):
        xs = [CX + sgn * d for d in range(28, 89, 10)]
        ys = [201, 200, 199, 199, 200, 201, 201]
        ths = [2, 2, 2, 2, 1, 1]
        for i in range(len(xs) - 1):
            cv2.line(g, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                     90, ths[i], cv2.LINE_AA)
    for cx in (196, 320):
        cv2.circle(g, (cx, 230), 5, 100, -1, cv2.LINE_AA)
        cv2.rectangle(g, (cx - 34, 206), (cx + 34, 225), 255,
                      -1)
        up = [(cx - 32, 226), (cx - 14, 223), (cx + 14, 223),
              (cx + 32, 226)]
        for i in range(3):
            cv2.line(g, up[i], up[i + 1], 45, 3, cv2.LINE_AA)
        lo = [(cx - 32, 226), (cx - 14, 233), (cx + 14, 233),
              (cx + 32, 226)]
        for i in range(3):
            cv2.line(g, lo[i], lo[i + 1], 95, 1, cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr130_s7.png"), 0)
    assert base is not None
    H, W = base.shape
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :])]
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    pipe = _load_dpm(None, ckpt,
                     "lllyasviel/control_v11p_sd15_lineart")
    _scale_lora(pipe.unet, 0.3)
    pipe.set_ip_adapter_scale(0.8)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    mmask = _eye_mask((H, W))
    mguide = _eye_guide(base)
    einit = base.copy()
    einit[196:248, 150:366] = 255
    panels = [("base", base)]
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=EYE_PROMPT, negative_prompt=EYE_NEG,
                   image=Image.fromarray(einit).convert("RGB"),
                   mask_image=Image.fromarray(mmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(mguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.72,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr131c_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"s{seed}", g))
        print(f"[r131 {name}] {time.time() - t0:.0f}s",
              flush=True)

    n = len(panels)
    ez = [cv2.resize(im[196:252, 155:365], None, fx=3.2, fy=3.2,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    eh, ew = ez[0].shape
    esheet = np.full((eh, ew * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, ez):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        esheet[:, x:x + ew] = t
        x += ew + 8
    cv2.imwrite(os.path.join(OUT, "_genr131c_gaze.png"), esheet)
    print("[r131] sheets saved", flush=True)


if __name__ == "__main__":
    main()
