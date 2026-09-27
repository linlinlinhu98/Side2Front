"""R116 (vs-REF 0.704, target 0.75; drags: hair 0.566,
eyes too big/shiny vs the reference's calm narrow gaze):

A. HAIR darkening: hair-region mask, init x0.60 (the R103
   forcing trick), 'jet black hair' prompt, strength 0.55 -
   keeps the texture, deepens the mass toward the
   reference's dense black.
B. CALM-EYE edit: eye-band mask, R111's narrow-almond guide
   (68px, lid covers ~55%, 2px upturn), strength 0.65,
   'calm relaxed narrow eyes, not shiny' prompt.

Base: genr114_s42 (best vs-REF). 2+2 seeds.
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
from gen_r101 import _load_dpm  # noqa: E402
from gen_r102 import STEPS  # noqa: E402
from gen_r111 import _eye_mask, _eye_guide  # noqa: E402

HAIR_PROMPT = (f"{TRIGGER}, jet black hair, dark hair, "
               "short dark hair, fine hair strands, "
               "pencil sketch, monochrome, white "
               "background")
HAIR_NEG = ("light hair, gray hair, white hair, messy "
            "hair, flyaway, color, colored, lowres, "
            "blurry, watermark")

EYE_PROMPT = (f"{TRIGGER}, calm eyes, relaxed eyes, "
              "narrow eyes, long almond eyes, single "
              "eyelid, quiet expression, not shiny, "
              "looking at viewer, pencil sketch, "
              "monochrome, white background")
EYE_NEG = ("big eyes, round eyes, shiny eyes, sparkling "
           "eyes, excited, double eyelid, color, "
           "colored, lowres, blurry, watermark")


def _hair_init(base, mask):
    init = base.astype(np.float32)
    a = (mask.astype(np.float32) / 255.0)
    dark = init * 0.60
    return np.clip(init * (1 - a) + dark * a, 0, 255) \
        .astype(np.uint8)


def main():
    base = cv2.imread(os.path.join(OUT, "genr114_s42.png"), 0)
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
    _scale_lora(pipe.unet, 0.8)
    pipe.set_ip_adapter_scale(0.7)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    # ---- A: hair darkening ----
    hmask = _hair_mask((H, W))
    hinit = _hair_init(base, hmask)
    hairs = []
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=HAIR_PROMPT, negative_prompt=HAIR_NEG,
                   image=Image.fromarray(hinit).convert("RGB"),
                   mask_image=Image.fromarray(hmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(
                       _lineart(base)).convert("RGB"),
                   controlnet_conditioning_scale=0.85,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.55,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr116_h{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        hairs.append((seed, g))
        print(f"[r116 {name}] {time.time() - t0:.0f}s",
              flush=True)

    # ---- B: calm eyes on each ----
    emask = _eye_mask((H, W))
    panels = [("base", base)]
    for seed, himg in hairs:
        eguide = _eye_guide(himg)
        einit = himg.copy()
        einit[250:298, 150:366] = 255
        t0 = time.time()
        res = pipe(prompt=EYE_PROMPT, negative_prompt=EYE_NEG,
                   image=Image.fromarray(einit).convert("RGB"),
                   mask_image=Image.fromarray(emask)
                   .convert("RGB"),
                   control_image=Image.fromarray(eguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.9,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.65,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr116_h{seed}_e.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        print(f"[r116 {name}] {time.time() - t0:.0f}s",
              flush=True)
        panels.append((f"h{seed}", himg))
        panels.append((f"h{seed}e", g))

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr116_sheet.png"), sheet)
    print("[r116] sheets saved", flush=True)


if __name__ == "__main__":
    main()
