"""R96 (mouth saga root cause found: 'full lips / realistic
lips' prompts make the model paint big nasolabial shadow
smudges; 'one line mouth' prompts are clean but lip-less):

Combine both lessons:
- init: whitened mouth zone + a VERY faint small soft
  lower-lip tone (232, 12x5, sigma 6)
- prompt: R83's clean-mouth spec ('neutral expressionless
  mouth, straight closed lips, one line mouth') which
  suppresses heavy shading
- guide: lip line corners +2px up + faint lower-lip arc
- LoRA 0.3, CN 0.8, strength 0.60
Base: genr95_b42 (brow-fixed, lip not yet passed). 2 seeds.
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
from gen_r83 import (LORA_DIR, TRIGGER, CX, _scale_lora,
                     MOUTH_PROMPT, MOUTH_NEG)  # noqa: E402
from gen_r93 import _mouth_mask  # noqa: E402


def _lip_guide(base):
    g = _lineart(base)
    g[306:340, 216:302] = 255
    pts = [(CX - 15, 321), (CX, 323), (CX + 15, 321)]
    cv2.line(g, pts[0], pts[1], 70, 1, cv2.LINE_AA)
    cv2.line(g, pts[1], pts[2], 70, 1, cv2.LINE_AA)
    cv2.ellipse(g, (CX, 330), (10, 4), 0, 25, 155, 185, 1,
                cv2.LINE_AA)
    return g


def _lip_init(base):
    init = base.copy()
    init[300:338, 218:300] = 255
    blob = np.zeros_like(init)
    cv2.ellipse(blob, (CX, 329), (12, 5), 0, 0, 360, 255, -1)
    a = cv2.GaussianBlur(blob, (0, 0), 6).astype(np.float32) / 255
    init = (init * (1 - a) + 232 * a).astype(np.uint8)
    return init


def main():
    base = cv2.imread(os.path.join(OUT, "genr95_b42.png"), 0)
    assert base is not None
    H, W = base.shape
    origc = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                       cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(origc, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[170:300, 100:320])]

    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    from diffusers import (ControlNetModel, LCMScheduler,
                           StableDiffusionControlNetInpaintPipeline)
    from peft import PeftModel
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetInpaintPipeline \
        .from_single_file(ckpt, controlnet=cn,
                          torch_dtype=torch.float32,
                          safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception:
        pass
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
    _scale_lora(pipe.unet, 0.3)
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

    mmask = _mouth_mask((H, W))
    guide = _lip_guide(base)
    init = _lip_init(base)
    cv2.imwrite(os.path.join(OUT, "_r96_linit.png"), init)
    panels = [("base", base), ("r83_m7 ref", None)]
    ref = cv2.imread(os.path.join(OUT, "genr83_m7.png"), 0)
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=MOUTH_PROMPT, negative_prompt=MOUTH_NEG,
                   image=Image.fromarray(init).convert("RGB"),
                   mask_image=Image.fromarray(mmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.8,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8, guidance_scale=3.5,
                   strength=0.60,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr96_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        panels.append((f"s{seed}", g))
        print(f"[r96 {name}] {time.time() - t0:.0f}s", flush=True)

    gen = [p for p in panels if p[1] is not None]
    n = len(gen)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in gen:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr96_sheet.png"), sheet)
    mz = [cv2.resize(im[305:348, 218:302], None, fx=6, fy=6,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in gen]
    mh, mw = mz[0].shape
    msheet = np.full((mh, mw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(gen, mz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        msheet[:, x:x + mw] = t
        x += mw + 8
    cv2.imwrite(os.path.join(OUT, "_genr96_mouths.png"), msheet)
    print("[r96] sheets saved", flush=True)


if __name__ == "__main__":
    main()
