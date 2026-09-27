"""R123: the mouth's last stand (15+ attempts all bent by
the anime arc prior; R122 m42_t has vertical straps +
crisp collar + white-blob-free right side, but a seagull
mouth):

THICK DARK perfectly-STRAIGHT line (3px, value 40, full
52px span = the width the model insists on, but dead
level) at CN 1.0 so it cannot be ignored; 'thin straight
mouth line, one line mouth, closed lips' prompt. LoRA
0.3, strength 0.70, DPM-26. On genr122_m42_t. 3 seeds.
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

MOUTH_PROMPT = (f"{TRIGGER}, thin straight mouth line, "
                "one line mouth, closed lips, level "
                "mouth, neutral mouth, pencil sketch, "
                "monochrome, white background")
MOUTH_NEG = ("curved mouth, arc mouth, smile, frown, "
             "upturned corners, downturned corners, "
             "thick lips, open mouth, color, colored, "
             "lowres, blurry, watermark")


def _mouth_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CX, 318), (44, 20), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 6)


def _mouth_guide(base):
    g = _lineart(base)
    g[304:338, 216:302] = 255
    # ONE THICK DARK STRAIGHT line, dead level, full width
    cv2.line(g, (CX - 26, 321), (CX + 26, 321), 40, 3,
             cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr122_m42_t.png"),
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
    _scale_lora(pipe.unet, 0.3)
    pipe.set_ip_adapter_scale(0.8)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    mmask = _mouth_mask((H, W))
    mguide = _mouth_guide(base)
    panels = [("base", base)]
    for seed in (7, 42, 998):
        t0 = time.time()
        res = pipe(prompt=MOUTH_PROMPT,
                   negative_prompt=MOUTH_NEG,
                   image=Image.fromarray(base).convert("RGB"),
                   mask_image=Image.fromarray(mmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(mguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.70,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr123_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"s{seed}", g))
        print(f"[r123 {name}] {time.time() - t0:.0f}s",
              flush=True)

    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr123_sheet.png"), sheet)
    mz = [cv2.resize(im[305:340, 215:305], None, fx=8, fy=8,
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
    cv2.imwrite(os.path.join(OUT, "_genr123_mouths.png"), msheet)
    print("[r123] sheets saved", flush=True)


if __name__ == "__main__":
    main()
