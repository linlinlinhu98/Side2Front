"""R119: img2img FROM the AI reference itself (user: 可以模
仿这张图; vs-REF ceiling 0.70 across 4 one-pass recipes -
the untried lever).

Start from the reference image (its style/composition/hair
mass is already top-scoring), swap the identity to the
ORIGINAL boy via IP-Adapter [original full+head+torso]
@0.9 + s2fstyle LoRA 0.8 + original-spec prompt.
Strength 0.55 (keep the reference's structure), CN lineart
of the reference 0.7, DPM-26. 4 seeds.
Scored vs REF (construction-high) AND vs ORIGINAL (the real
identity test).
"""
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from trial_inpaint_eyes import _find_file, OUT  # noqa: E402
from gen_anime_common import CX, LORA_DIR, STEPS, TRIGGER, _lineart, _load_dpm, _scale_lora  # noqa: E402


def _seeds():
    return tuple(int(x) for x in
                 os.environ.get("S2F_SEEDS",
                                "7,42,998,123").split(",")
                 if x)

PROMPT = (f"{TRIGGER}, 1boy, teenage boy, thin narrow "
          "face, black hair, long narrow almond eyes, "
          "calm expression, single eyelid, small nose, "
          "thin lips, zip-up hoodie, backpack two "
          "shoulder straps, hand drawn, pencil sketch, "
          "line weight variation, subtle grey shading, "
          "monochrome, white background, front view")
NEG = ("chibi, big round shiny eyes, wide fat face, "
       "thick lips, sad pout, cloak, drawstrings, "
       "light hair, colored, 3d render, deformed, "
       "watermark")


def main():
    ref = cv2.imread(os.path.join(
        ROOT, "result", "8aa591a6c9e652b79697d55eeee769ec.jpg"),
        0)
    assert ref is not None
    rh, rw = ref.shape
    ref520 = cv2.resize(ref, (520, int(520 * rh / rw)),
                        interpolation=cv2.INTER_AREA)
    H = min(664, ref520.shape[0] // 8 * 8)
    ref520 = cv2.resize(ref520[:H, :], (520, 664),
                        interpolation=cv2.INTER_CUBIC)

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
    pipe.set_ip_adapter_scale(0.9)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    ctrl = Image.fromarray(_lineart(ref520)).convert("RGB")
    full_mask = Image.fromarray(
        np.full((664, 520), 255, np.uint8)).convert("RGB")
    panels = [("ref520", ref520)]
    for seed in _seeds():
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=Image.fromarray(ref520).convert("RGB"),
                   mask_image=full_mask,
                   control_image=ctrl,
                   controlnet_conditioning_scale=0.7,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.55,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (520, 660),
                       interpolation=cv2.INTER_CUBIC)
        name = f"genr119_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"s{seed}", g))
        print(f"[r119 {name}] {time.time() - t0:.0f}s",
              flush=True)

    W = 520
    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = cv2.resize(im, (W, 660),
                       interpolation=cv2.INTER_CUBIC).copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr119_sheet.png"), sheet)
    print("[r119] sheets saved", flush=True)


if __name__ == "__main__":
    main()
