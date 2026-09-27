"""R69 (user on R68: 不如base/衣服还是不对/眉毛太粗/弯曲不对):

Verdict synthesis:
- R63 base (genplus4_ht_s7) = BEST CLOTHES (zipper teeth + pull,
  hood lining) but digital style.
- R65 (LoRA 1.0) = BEST STROKES but washed out the clothes'
  structure. R68's full-face repaint drifted identity (and my
  'thick eyebrows' prompt was wrong — the original's brows are
  MEDIUM straight; no brow words at all now).

R69 = both dials at once:
- style-transfer img2img FROM the R63 base (strength 0.40, proven
  to keep composition/zipper in R64)
- s2fstyle LoRA at SCALE 0.7 (softens strokes without melting the
  zipper structure)
- whole original [full+head+torso] IP-Adapter Plus @0.8 (user:
  整张原图输入)
- prompt carries clothes locks (zipper placket, hood down, dark
  hood lining); ZERO eyebrow/face-shape words — identity comes
  from the image data, not my guesses
Stage B: proven eye-only repaint (narrow calm eyes) on each.
4 seeds x 2 stages.
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
from gen_plus4 import _eye_mask, EYE_PROMPT, EYE_NEG  # noqa: E402

LORA_DIR = os.path.join(ROOT, "assets", "lora", "s2fstyle")
TRIGGER = "s2fstyle"
LORA_SCALE = 0.7

PROMPT = (f"{TRIGGER}, 1boy, monochrome, pencil sketch, soft "
          "shading, parallel hatching, delicate thin lines, "
          "waist up, front view, looking at viewer, arms at "
          "sides, black hair, narrow eyes, hood down, zip-up "
          "hoodie, zipper placket, dark hood lining, high "
          "collar, long sleeves, blush, white background")
NEG = ("thick outlines, cel shading, digital painting, anime "
       "screenshot, drawstrings, pullover, turtleneck, thick "
       "eyebrows, round eyes, big eyes, cute, shota, smile, "
       "color, colored, photo, realistic, 3d, lowres, blurry, "
       "bad anatomy, watermark, dark background")

SEEDS = [7, 42, 998, 2024]
STRENGTH = 0.40
IP_SCALE = 0.8
GUIDANCE = 2.0


def _scale_lora(unet, factor):
    n = 0
    for m in unet.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict) and "default" in sc:
            sc["default"] = sc["default"] * factor
            n += 1
    print(f"[r69] lora scaling x{factor} on {n} layers",
          flush=True)


def main():
    base = cv2.imread(os.path.join(OUT, "genplus4_ht_s7.png"), 0)
    assert base is not None
    H, W = base.shape
    bpil = Image.fromarray(base).convert("RGB")

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb),
            Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[280:, :])]

    from diffusers import (LCMScheduler,
                           StableDiffusionImg2ImgPipeline,
                           StableDiffusionInpaintPipeline)
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
    _scale_lora(pipe.unet, LORA_SCALE)
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

    cands = []
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
        name = f"genr69_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cands.append((f"s{seed}", g, name))
        print(f"[r69 {name}] {time.time() - t0:.0f}s", flush=True)

    del pipe
    ipipe = StableDiffusionInpaintPipeline.from_single_file(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    ipipe.scheduler = LCMScheduler.from_config(
        ipipe.scheduler.config)
    ipipe.load_lora_weights(lcm)
    try:
        ipipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora2: {e}", flush=True)
    ipipe.unet = PeftModel.from_pretrained(ipipe.unet, LORA_DIR)
    _scale_lora(ipipe.unet, LORA_SCALE)
    ipipe.set_progress_bar_config(disable=True)

    mask = _eye_mask((H, W))
    mpil = Image.fromarray(mask).convert("RGB")
    eyes = []
    for label, g, name in cands:
        t0 = time.time()
        gpil = Image.fromarray(g).convert("RGB")
        res = ipipe(prompt=f"{TRIGGER}, {EYE_PROMPT}",
                    negative_prompt=EYE_NEG,
                    image=gpil, mask_image=mpil,
                    height=664, width=520,
                    num_inference_steps=8, guidance_scale=GUIDANCE,
                    strength=0.75,
                    generator=torch.Generator("cpu")
                    .manual_seed(42)).images[0]
        eg = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        eg = cv2.resize(eg, (W, H), interpolation=cv2.INTER_CUBIC)
        ename = name.replace(".png", "_eye.png")
        cv2.imwrite(os.path.join(OUT, ename), eg)
        eyes.append((label + " eye", eg))
        print(f"[r69 {ename}] {time.time() - t0:.0f}s", flush=True)

    panels = [("R63 base", base)] + [c[:2] for c in cands] + eyes
    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr69_sheet.png"), sheet)
    print("[r69] sheet saved", flush=True)


if __name__ == "__main__":
    main()
