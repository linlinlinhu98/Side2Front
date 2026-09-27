"""R129 (one-at-a-time #3: 眼睛还是看着上面没有平视):

Gaze is set by WHERE the iris sits in the socket: ours sits
high (looking up). Guide: irises placed in the LOWER half
of the eye, upper lid covering the top ~50% => level,
slightly sleepy-calm gaze. 'level gaze, looking straight
ahead, half-closed eyes' prompt; 'looking up, upturned'
negative. Eye-band mask, CN 1.0, strength 0.72, LoRA 0.3,
DPM-26. Base: genr128_s7. 2 seeds.
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
              "single eyelid, narrow eyes, pencil "
              "sketch, monochrome, white background")
EYE_NEG = ("looking up, upturned eyes, big shiny "
           "eyes, excited, round eyes, double "
           "eyelid, color, colored, lowres, blurry, "
           "watermark")


def _eye_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CX, 276), (100, 26), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 7)


def _eye_guide(base):
    g = _lineart(base)
    g[252:298, 150:366] = 255
    # keep brows
    for sgn in (-1, 1):
        xs = [CX + sgn * d for d in range(28, 89, 10)]
        ys = [253, 252, 251, 251, 252, 253, 253]
        ths = [2, 2, 2, 2, 1, 1]
        for i in range(len(xs) - 1):
            cv2.line(g, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                     90, ths[i], cv2.LINE_AA)
    # iris in the LOWER half, lid covers top ~50%
    for cx in (196, 320):
        cv2.circle(g, (cx, 282), 5, 100, -1, cv2.LINE_AA)
        cv2.rectangle(g, (cx - 34, 260), (cx + 34, 277), 255,
                      -1)
        up = [(cx - 32, 278), (cx - 14, 275), (cx + 14, 275),
              (cx + 32, 278)]
        for i in range(3):
            cv2.line(g, up[i], up[i + 1], 45, 3, cv2.LINE_AA)
        lo = [(cx - 32, 278), (cx - 14, 285), (cx + 14, 285),
              (cx + 32, 278)]
        for i in range(3):
            cv2.line(g, lo[i], lo[i + 1], 95, 1, cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr128_s7.png"), 0)
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

    emask = _eye_mask((H, W))
    eguide = _eye_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r129_eguide.png"), eguide)
    einit = base.copy()
    einit[252:298, 150:366] = 255
    panels = [("base", base)]
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=EYE_PROMPT, negative_prompt=EYE_NEG,
                   image=Image.fromarray(einit).convert("RGB"),
                   mask_image=Image.fromarray(emask)
                   .convert("RGB"),
                   control_image=Image.fromarray(eguide)
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
        name = f"genr129_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"s{seed}", g))
        print(f"[r129 {name}] {time.time() - t0:.0f}s",
              flush=True)

    n = len(panels)
    ez = [cv2.resize(im[250:300, 155:365], None, fx=3.2, fy=3.2,
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
    cv2.imwrite(os.path.join(OUT, "_genr129_gaze.png"), esheet)
    print("[r129] sheets saved", flush=True)


if __name__ == "__main__":
    main()
