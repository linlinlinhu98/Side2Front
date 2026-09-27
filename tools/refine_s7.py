"""R54 (user on genxlp3_s7: 衣服不太像, 笔触不太像, 没有腮红):

Keep s7's approved head/pose; repaint ONLY the torso (below the
chin) with the ORIGINAL's clothes as IP-Adapter reference — high
collar rolls + zipper, NO drawstrings. 12 LCM steps for firmer
hatching. Blush is added classically afterwards (diagonal pencil
hatching on the cheeks, like the original's) so the approved face
is never touched by the model.

Outputs refine_s7_a.png (strength 0.50) / _b.png (0.65) + compares.
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
from gen_frontal_ipa import _stretch  # noqa: E402

GEN_W, GEN_H = 888, 1128

PROMPT = ("1boy, solo, monochrome, greyscale, pencil (medium), "
          "traditional media, sketch, waist up, front view, "
          "straight-on, looking at viewer, arms at sides, relaxed "
          "shoulders, broad shoulders, black hair, short hair, "
          "narrow eyes, bangs, hood down, hoodie, high collar, "
          "zipper, long sleeves, loose clothes, blush, pale skin, "
          "simple background, white background, masterpiece, "
          "best quality, very aesthetic")
NEG = ("lowres, bad anatomy, bad hands, text, error, cropped, "
       "worst quality, low quality, normal quality, jpeg "
       "artifacts, signature, watermark, username, blurry, "
       "artist name, color, colored, photo, realistic, 3d, "
       "long hair, hat, turtleneck, drawstrings, hunched "
       "shoulders, arms crossed, hands on chest, dark "
       "background, grey background")

CANDS = [("a", 998), ("b", 7)]


def _blush(img):
    """classical diagonal-hatch blush on the cheeks (original has
    hatched blush; the model keeps washing it out). Cheeks measured
    on s7: eyes ~y270, face half-width ~55 around x257."""
    out = img.copy().astype(np.float32)
    H, W = img.shape
    for cx, ang in ((209, 20), (305, -20)):
        hatch = np.full((H, W), 255, np.uint8)
        for i in range(-H, W + H, 5):
            x0 = i if ang > 0 else i + H
            cv2.line(hatch, (x0, 0), (x0 + (H if ang > 0 else -H),
                                      H), 160, 1, cv2.LINE_AA)
        m = np.zeros((H, W), np.float32)
        cv2.ellipse(m, (cx, 306), (26, 14), ang, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), 6) * 0.28
        dark = np.minimum(out, hatch.astype(np.float32))
        out = out * (1 - m) + dark * m
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    base = cv2.imread(os.path.join(OUT, "genxlp3_s7.png"), 0)
    assert base is not None
    H, W = base.shape
    big = cv2.resize(base, (GEN_W, GEN_H), interpolation=cv2.INTER_CUBIC)
    pil = Image.fromarray(big).convert("RGB")

    # torso mask: below the chin, soft feather at the neck
    mask = np.zeros((H, W), np.float32)
    mask[398:, :] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), 6)
    # inpaint-pipeline mask at generation resolution (white=repaint)
    mpil = Image.fromarray(
        cv2.resize((mask * 255).astype(np.uint8), (GEN_W, GEN_H),
                   interpolation=cv2.INTER_LINEAR))

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    ph, pw = int(orig.shape[0] * 0.55), int(orig.shape[1] * 0.25)
    orig_pad = cv2.copyMakeBorder(orig, ph, ph, pw, pw,
                                  cv2.BORDER_CONSTANT,
                                  value=(255, 255, 255))
    ip_pil = Image.fromarray(cv2.cvtColor(orig_pad,
                                          cv2.COLOR_BGR2RGB))

    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdxl"),
        "pytorch_lora_weights.safetensors")
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--cagliostrolab--animagine-xl-3.1"),
        "animagine-xl-3.1.safetensors")

    from diffusers import (LCMScheduler,
                           StableDiffusionXLInpaintPipeline)
    from transformers import CLIPImageProcessor, CLIPVisionModel
    enc = CLIPVisionModel.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder",
        torch_dtype=torch.float32)
    pv = CLIPImageProcessor()(images=ip_pil,
                              return_tensors="pt").pixel_values
    with torch.no_grad():
        ip_emb = enc(pv).pooler_output.unsqueeze(1)

    # INPAINT pipeline: img2img kept the drawstring layout no matter
    # the prompt — the torso must be redrawn from noise inside the
    # mask so the clothes STYLE comes from prompt + original IP
    pipe = StableDiffusionXLInpaintPipeline.from_single_file(
        ckpt, image_encoder=enc, torch_dtype=torch.float32,
        safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora: {e}", flush=True)
    pipe.vae.enable_slicing()
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                         weight_name="ip-adapter_sdxl.bin")
    pipe.set_ip_adapter_scale(0.7)
    pipe.set_progress_bar_config(disable=True)

    cands = [("s7 (base)", base)]
    for label, seed in CANDS:
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=pil, mask_image=mpil,
                   ip_adapter_image_embeds=[ip_emb],
                   num_inference_steps=12, guidance_scale=1.0,
                   strength=0.9,
                   generator=torch.Generator("cpu").manual_seed(
                       seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
        g = _stretch(g)
        out = (g.astype(np.float32) * mask
               + base.astype(np.float32) * (1 - mask))
        out = np.clip(out, 0, 255).astype(np.uint8)
        out = _blush(out)
        name = f"refine_s7_{label}.png"
        cv2.imwrite(os.path.join(OUT, name), out)
        cands.append((f"inpaint seed={seed}", out))
        print(f"[refine {name}] {time.time() - t0:.0f}s", flush=True)

    ph = 660
    sheet = np.full((ph, W * len(cands) + 10 * (len(cands) - 1)),
                    255, np.uint8)
    x = 0
    for label, im in cands:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_refine_s7_sheet.png"), sheet)
    print("[refine] sheet saved", flush=True)


if __name__ == "__main__":
    main()
