"""R144 (fix #4 per user: 头发上不要有太多游离的乱毛):

Hair-region pass: erase ALL strands outside a clean
silhouette boundary in the guide (draw the boundary line),
'neat tidy hair' / 'flyaway strands, spiky, messy'
negatives. Composite back ONLY the hair zone (bit-identical
elsewhere). Base: genr143_s7 (mouth fixed, hair black from
R141 preserved by composite-back). DPM-26, LoRA 0.6,
strength 0.55. 2 seeds.
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
from gen_r97 import _hair_mask  # noqa: E402

PROMPT = (f"{TRIGGER}, short black hair, neat tidy hair, "
          "clean hairline, hair clumps, pencil sketch, "
          "monochrome, white background")
NEG = ("flyaway strands, stray hairs, spiky hair, "
       "messy hair, wild hair, light hair, color, "
       "colored, lowres, blurry, watermark")


def _composite(base, result, mask):
    a = mask.astype(np.float32) / 255.0
    out = base.astype(np.float32) * (1 - a) \
        + result.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _guide(base):
    g = _lineart(base)
    # erase everything outside the clean boundary
    bound = np.zeros_like(g)
    cv2.ellipse(bound, (CX, 205), (168, 175), 0, 0, 360,
                255, -1)
    cv2.ellipse(bound, (CX, 320), (106, 98), 0, 0, 360, 255,
                -1)
    bound = cv2.GaussianBlur(bound, (0, 0), 2)
    g[bound == 0] = 255
    # draw the boundary line itself
    cv2.ellipse(g, (CX, 205), (168, 175), 0, 200, 340, 60, 2,
                cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr143_s7.png"),
                      0)
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
    _scale_lora(pipe.unet, 0.6)
    pipe.set_ip_adapter_scale(0.7)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    hmask = _hair_mask((H, W))
    guide = _guide(base)
    cv2.imwrite(os.path.join(OUT, "_r144_guide.png"), guide)
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=Image.fromarray(base).convert("RGB"),
                   mask_image=Image.fromarray(hmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.85,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.55,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        final = _composite(base, g, hmask)
        name = f"genr144_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), final)
        unchanged = (final[hmask == 0] == base[hmask == 0]).mean()
        print(f"[r144 {name}] {time.time() - t0:.0f}s "
              f"unchanged_outside={unchanged:.3f}", flush=True)

    panels = [("base", base)]
    for seed in (7, 42):
        im = cv2.imread(os.path.join(OUT, f"genr144_s{seed}.png"), 0)
        panels.append((f"s{seed}", im))
    n = len(panels)
    hz = [cv2.resize(im[30:280, 80:440], None, fx=1.1, fy=1.1,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    hh, hw = hz[0].shape
    sheet = np.full((hh, hw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, hz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + hw] = t
        x += hw + 8
    cv2.imwrite(os.path.join(OUT, "_genr144_hair.png"), sheet)
    print("[r144] sheets saved", flush=True)


if __name__ == "__main__":
    main()
