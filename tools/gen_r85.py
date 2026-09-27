"""R85: consolidation rebuild (similarity dropped to 0.58 from
0.62: every pipeline pass costs a full-image VAE round-trip,
blurring untouched regions; face subscore hit all-time-high
0.614 but hair/cloth degraded).

ONE low-strength full img2img fusion pass on the current best
(genr84_m42_cloth): LoRA 1.0 + IP [full+head+torso] @1.0 +
hatching prompt. Restores crispness/texture uniformly without
moving any structure. 2 seeds.
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
from gen_r83 import LORA_DIR, TRIGGER, _scale_lora  # noqa: E402

PROMPT = (f"{TRIGGER}, 1boy, monochrome, pencil sketch, soft "
          "shading, parallel hatching, fine pencil texture, "
          "delicate thin lines, waist up, front view, black "
          "hair, zip-up hoodie, full-length zipper, hood "
          "down, fabric folds, blush, white background")
NEG = ("smooth, clean, digital painting, thick outlines, "
       "cel shading, anime screenshot, drawstrings, color, "
       "colored, photo, realistic, 3d, lowres, blurry, "
       "bad anatomy, watermark, dark background")

SEEDS = [7, 42]
STRENGTH = 0.28
GUIDANCE = 2.0
IP_SCALE = 1.0


def main():
    base = cv2.imread(os.path.join(OUT, "genr84_m42_cloth.png"),
                      0)
    assert base is not None
    H, W = base.shape
    bpil = Image.fromarray(base).convert("RGB")

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[280:, :])]

    from diffusers import (LCMScheduler,
                           StableDiffusionImg2ImgPipeline)
    from peft import PeftModel
    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora: {e}", flush=True)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(IP_SCALE)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    cands = [("base", base)]
    for seed in SEEDS:
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=bpil, ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8,
                   guidance_scale=GUIDANCE, strength=STRENGTH,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr85_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cands.append((f"s{seed}", g))
        print(f"[r85 {name}] {time.time() - t0:.0f}s", flush=True)

    n = len(cands)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in cands:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr85_sheet.png"), sheet)
    print("[r85] sheet saved", flush=True)


if __name__ == "__main__":
    main()
