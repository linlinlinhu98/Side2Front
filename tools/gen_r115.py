"""R115: frayed-edge cleanup (user: 头发依旧有毛边; REF-sim
hair band 0.595 is the drag to 0.75).

Generative background cleanup: mask = BACKGROUND ONLY (figure
silhouette protected by dilation), prompt 'white background,
clean paper', strength 0.35 - erases stray gray pixels around
the hair edge without touching a single line of the figure.

Usage: python tools/gen_r115.py <base.png> [seed]
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
from gen_r83 import LORA_DIR, TRIGGER, _scale_lora  # noqa: E402
from gen_r101 import _load_dpm  # noqa: E402
from gen_r102 import STEPS  # noqa: E402

BG_PROMPT = ("white background, clean paper, monochrome, "
             "empty background")
BG_NEG = ("gray, shadow, smudge, noise, stray marks, "
          "texture, color, colored, lowres, watermark")


def _bg_mask(img):
    dark = (img < 200).astype(np.uint8) * 255
    sil = cv2.dilate(dark, np.ones((9, 9), np.uint8),
                     iterations=3)
    sil = cv2.GaussianBlur(sil, (0, 0), 6)
    m = 255 - sil
    m[m < 128] = 0
    m[m >= 128] = 255
    return cv2.GaussianBlur(m, (0, 0), 4).astype(np.uint8)


def main():
    base_name = sys.argv[1] if len(sys.argv) > 1 else \
        "genr113_s123.png"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    base = cv2.imread(os.path.join(OUT, base_name), 0)
    assert base is not None
    H, W = base.shape
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb)]
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    pipe = _load_dpm(None, ckpt,
                     "lllyasviel/control_v11p_sd15_lineart")
    _scale_lora(pipe.unet, 0.8)
    pipe.set_ip_adapter_scale(0.5)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    bmask = _bg_mask(base)
    t0 = time.time()
    res = pipe(prompt=BG_PROMPT, negative_prompt=BG_NEG,
               image=Image.fromarray(base).convert("RGB"),
               mask_image=Image.fromarray(bmask).convert("RGB"),
               control_image=Image.fromarray(_lineart(base))
               .convert("RGB"),
               controlnet_conditioning_scale=0.9,
               ip_adapter_image_embeds=[emb],
               height=664, width=520,
               num_inference_steps=STEPS,
               guidance_scale=3.0, strength=0.35,
               generator=torch.Generator("cpu")
               .manual_seed(seed)).images[0]
    g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
    name = base_name.replace(".png", "") + "_bg.png"
    cv2.imwrite(os.path.join(OUT, name), g)
    print(f"[r115 {name}] {time.time() - t0:.0f}s", flush=True)

    # edge zoom check
    z1 = cv2.resize(base[30:250, 60:460], None, fx=1.4, fy=1.4,
                    interpolation=cv2.INTER_CUBIC)
    z2 = cv2.resize(g[30:250, 60:460], None, fx=1.4, fy=1.4,
                    interpolation=cv2.INTER_CUBIC)
    cv2.putText(z1, "before", (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, 0, 2, cv2.LINE_AA)
    cv2.putText(z2, "after bg-clean", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)
    sheet = np.hstack([z1, np.full((z1.shape[0], 8), 200,
                                   np.uint8), z2])
    cv2.imwrite(os.path.join(OUT, "_r115_edge.png"), sheet)
    print("[r115] edge sheet saved", flush=True)


if __name__ == "__main__":
    main()
