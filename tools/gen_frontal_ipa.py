"""R53 (user 2026-09-22: 肩膀像正常人, 手臂舒张开在两侧, 不要收拢
在胸前; 重点是从原图侧面转正面, 不要完全参考底图):

FREE-COMPOSITION generation — Animagine XL txt2img-equivalent
(img2img strength=1.0) so the pose/shoulders come from the PROMPT,
while the ORIGINAL side profile drives identity/style through
IP-Adapter (user: 把原图作为数据生成). No base-image layout.

Prompt adds: upper body, arms at sides, relaxed shoulders, broad
shoulders. Negative bans hunched/tucked poses.

IP-Adapter note (probed from the weights): ip-adapter_sdxl.bin's
image_proj is Linear(1280 -> 8192) = it consumes the ViT-H POOLED
output (1280-d), not the 1024-d WithProjection output — so we
precompute pooler_output with CLIPVisionModel and pass
ip_adapter_image_embeds directly (CFG is off at guidance 1.0, so
diffusers uses the tensor as-is).

Outputs genxlp_*.png + _genxlp_sheet.png for user review.
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

GEN_W, GEN_H = 888, 1128

PROMPT = ("1boy, solo, monochrome, greyscale, pencil (medium), "
          "traditional media, sketch, waist up, front view, "
          "straight-on, looking at viewer, arms at sides, arms "
          "down, relaxed shoulders, broad shoulders, black hair, "
          "short hair, narrow eyes, bangs, hair between eyes, hood "
          "down, hoodie, zipper, pale skin, simple background, "
          "white background, masterpiece, best quality, "
          "very aesthetic")
NEG = ("lowres, bad anatomy, bad hands, text, error, missing "
       "fingers, extra digit, fewer digits, cropped, worst quality, "
       "low quality, normal quality, jpeg artifacts, signature, "
       "watermark, username, blurry, artist name, color, colored, "
       "photo, realistic, 3d, long hair, hat, turtleneck, dark "
       "background, grey background, hunched shoulders, shrug, "
       "arms crossed, folded arms, hands on chest, leaning forward")

# (label, ip scale, seed) — all free composition from the original
# (R53.1: ip=0.85 washed out, dropped; more seeds instead)
CANDS = [("s7", 0.55, 7),
         ("s998", 0.55, 998),
         ("s42", 0.55, 42)]


def _stretch(g):
    """free txt2img comes out pale — restore pencil contrast"""
    lo, hi = np.percentile(g, [0.5, 99.0])
    return np.clip((g.astype(np.float32) - lo) * 255.0 / (hi - lo),
                   0, 255).astype(np.uint8)


def main():
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    assert orig is not None
    # R53.1: the tight head-and-neck framing came from the reference
    # filling the frame — pad it with white so CLIP encodes a WIDER
    # composition (shoulders + arms in frame)
    ph, pw = int(orig.shape[0] * 0.55), int(orig.shape[1] * 0.25)
    orig_pad = cv2.copyMakeBorder(orig, ph, ph, pw, pw,
                                  cv2.BORDER_CONSTANT,
                                  value=(255, 255, 255))
    ip_pil = Image.fromarray(cv2.cvtColor(orig_pad,
                                          cv2.COLOR_BGR2RGB))
    H, W = 660, 520

    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdxl"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--cagliostrolab--animagine-xl-3.1"),
        "animagine-xl-3.1.safetensors")

    from diffusers import (LCMScheduler,
                           StableDiffusionXLImg2ImgPipeline)
    from transformers import CLIPImageProcessor, CLIPVisionModel

    enc = CLIPVisionModel.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder",
        torch_dtype=torch.float32)
    proc = CLIPImageProcessor()
    pv = proc(images=ip_pil, return_tensors="pt").pixel_values
    with torch.no_grad():
        ip_emb = enc(pv).pooler_output.unsqueeze(1)  # (1, 1, 1280)
    print("[genxlp] ip embeds", tuple(ip_emb.shape), flush=True)

    # passing our own encoder keeps load_ip_adapter from trying to
    # download its default one (offline); embeds are precomputed so
    # the encoder is never actually called by the pipeline
    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        ckpt, image_encoder=enc, torch_dtype=torch.float32,
        safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora: {e}", flush=True)
    # NOTE: enable_attention_slicing is incompatible with
    # load_ip_adapter in this diffusers version (both swap attention
    # processors; the IP converter cannot wrap SlicedAttnProcessor).
    # SDPA keeps attention memory in check instead.
    pipe.vae.enable_slicing()
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                         weight_name="ip-adapter_sdxl.bin")
    pipe.set_progress_bar_config(disable=True)

    # white canvas: at strength=1.0 the init image only sets the
    # latent size — composition is fully free
    white = Image.fromarray(np.full((GEN_H, GEN_W, 3), 255, np.uint8))

    cands = [("original", cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY))]
    for label, ip, seed in CANDS:
        pipe.set_ip_adapter_scale(ip)
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=white, ip_adapter_image_embeds=[ip_emb],
                   num_inference_steps=8, guidance_scale=1.0,
                   strength=1.0,
                   generator=torch.Generator("cpu").manual_seed(
                       seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
        g = _stretch(g)
        name = f"genxlp3_{label}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cands.append((f"{label} ip={ip}", g))
        print(f"[genxlp {name}] {time.time() - t0:.0f}s", flush=True)

    ph = 660
    scaled = []
    for label, im in cands:
        p = cv2.resize(im, (W, ph)) if im.shape[0] != ph else im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        scaled.append(p)
    sheet = np.full((ph, W * len(scaled) + 10 * (len(scaled) - 1)),
                    255, np.uint8)
    x = 0
    for p in scaled:
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genxlp_sheet.png"), sheet)
    print("[genxlp] sheet saved", flush=True)


if __name__ == "__main__":
    main()
