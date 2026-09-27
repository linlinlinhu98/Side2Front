"""R89: nose redo (R88's nose = triangle smudge + mustache
commas - too many stacked guide elements merged in a 34px
zone).

Classmate structure: soft shadow + two faint dots, NOTHING
hard. Two variants on genr88_c42:
- A (minimal CN): guide = ONLY 2 faint nostril dots + 1 faint
  short bridge line, CN 0.9
- B (free inpaint, no CN): prior + IP shapes the nose
LoRA 0.5, strength 0.65, 2 seeds each.
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
from gen_r88 import NOSE_NEG, _nose_mask  # noqa: E402

NOSE_PROMPT = (f"{TRIGGER}, small delicate nose, soft nose "
               "shadow, two small nostrils, subtle nose "
               "bridge, realistic nose, pencil sketch, soft "
               "shading, monochrome, white background")
NOSE_NEG2 = ("big nose, long nose, hard lines, dark lines, "
             "triangle nose, dot nose, hooked nose, color, "
             "colored, lowres, blurry, watermark")


def _nose_guide(base):
    g = _lineart(base)
    g[282:340, 222:296] = 255
    # two faint small nostril dots
    cv2.ellipse(g, (CX - 7, 321), (2, 1), 0, 0, 360, 120, 1,
                cv2.LINE_AA)
    cv2.ellipse(g, (CX + 7, 321), (2, 1), 0, 0, 360, 120, 1,
                cv2.LINE_AA)
    # one faint short bridge line
    cv2.line(g, (CX - 2, 296), (CX - 3, 310), 165, 1,
             cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr88_c42.png"), 0)
    assert base is not None
    H, W = base.shape

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :])]

    from diffusers import (ControlNetModel, LCMScheduler,
                           StableDiffusionControlNetInpaintPipeline,
                           StableDiffusionInpaintPipeline)
    from peft import PeftModel
    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    nmask = _nose_mask((H, W))
    mpil = Image.fromarray(nmask).convert("RGB")
    guide = _nose_guide(base)
    cv2.imwrite(os.path.join(OUT, "_r89_nguide.png"), guide)
    ninit = base.copy()
    ninit[284:338, 224:294] = 255
    bpil = Image.fromarray(ninit).convert("RGB")

    results = [("base", base)]

    # ---- variant A: minimal CN guide ----
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")
    pipe = StableDiffusionControlNetInpaintPipeline \
        .from_single_file(ckpt, controlnet=cn,
                          torch_dtype=torch.float32,
                          safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse: {e}", flush=True)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
    _scale_lora(pipe.unet, 0.5)
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
    ctrl = Image.fromarray(guide).convert("RGB")
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=NOSE_PROMPT, negative_prompt=NOSE_NEG2,
                   image=bpil, mask_image=mpil,
                   control_image=ctrl,
                   controlnet_conditioning_scale=0.9,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8, guidance_scale=3.0,
                   strength=0.65,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr89_a{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        results.append((f"A{seed}", g))
        print(f"[r89 {name}] {time.time() - t0:.0f}s", flush=True)
    del pipe, cn
    import gc
    gc.collect()

    # ---- variant B: free inpaint ----
    pipe2 = StableDiffusionInpaintPipeline.from_single_file(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    pipe2.scheduler = LCMScheduler.from_config(
        pipe2.scheduler.config)
    pipe2.load_lora_weights(lcm)
    try:
        pipe2.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse2: {e}", flush=True)
    pipe2.unet = PeftModel.from_pretrained(pipe2.unet, LORA_DIR)
    _scale_lora(pipe2.unet, 0.5)
    pipe2.set_progress_bar_config(disable=True)
    pipe2.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe2.set_ip_adapter_scale(0.9)
    with torch.no_grad():
        emb2, unc2 = pipe2.encode_image(refs, "cpu", 1,
                                        output_hidden_states=True)
        emb2 = torch.cat([unc2.unsqueeze(0), emb2.unsqueeze(0)],
                         dim=0)
    for seed in (7, 42):
        t0 = time.time()
        res = pipe2(prompt=NOSE_PROMPT, negative_prompt=NOSE_NEG2,
                    image=bpil, mask_image=mpil,
                    ip_adapter_image_embeds=[emb2],
                    height=664, width=520,
                    num_inference_steps=8, guidance_scale=3.0,
                    strength=0.65,
                    generator=torch.Generator("cpu")
                    .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr89_b{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        results.append((f"B{seed}", g))
        print(f"[r89 {name}] {time.time() - t0:.0f}s", flush=True)

    n = len(results)
    nz = [cv2.resize(im[282:342, 216:302], None, fx=4.5, fy=4.5,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in results]
    nh, nw = nz[0].shape
    nsheet = np.full((nh, nw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(results, nz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        nsheet[:, x:x + nw] = t
        x += nw + 8
    cv2.imwrite(os.path.join(OUT, "_genr89_noses.png"), nsheet)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in results:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr89_sheet.png"), sheet)
    print("[r89] sheets saved", flush=True)


if __name__ == "__main__":
    main()
