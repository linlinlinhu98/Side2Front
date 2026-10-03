# -*- coding: utf-8 -*-
"""PREP-TIME frontal reference generation (CPU, offline once downloaded).

Generates candidate frontal references for the known test images:
  - cat.jpg   : SD1.5 + IP-Adapter (keeps the tabby's markings)
  - face.png  : Anything-V5 + ControlNet lineart (our cleaned symmetric
                result as structure) + IP-Adapter (identity hint)
Output: assets/ref_gen/<name>/cand_<i>.png  — pick the best by eye,
annotate 16 keypoints on it, register in manual_keypoints.json.

Usage:
  HF_ENDPOINT=https://hf-mirror.com python tools/gen_frontal_refs.py cat [n]
  HF_ENDPOINT=https://hf-mirror.com python tools/gen_frontal_refs.py face [n]
"""
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "ref_gen")

SD15 = os.path.join(
    r"D:\huggingface_cache", "models--runwayml--stable-diffusion-v1-5")
ANIME = os.path.join(
    r"D:\huggingface_cache", "models--stablediffusionapi--anything-v5")


def _find_file(root, suffix):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(suffix):
                return os.path.join(dirpath, f)
    raise FileNotFoundError(f"{suffix} under {root}")


def _snapshot_dir(root):
    """The single snapshot dir inside a models--<repo> cache folder."""
    snap = os.path.join(root, "snapshots")
    subs = [os.path.join(snap, d) for d in os.listdir(snap)]
    subs = [d for d in subs if os.path.isdir(d)]
    if not subs:
        raise FileNotFoundError(f"no snapshot under {root}")
    return max(subs, key=os.path.getmtime)


def _tune(pipe):
    pipe.set_progress_bar_config(disable=True)
    # NOTE: enable_attention_slicing is INCOMPATIBLE with load_ip_adapter
    # in diffusers 0.40 — before the load it breaks the loader
    # (SlicedAttnProcessor needs slice_size), after the load it wipes
    # the IP-Adapter processors ('tuple' object has no attribute
    # 'shape'). 512x512 fp32 on CPU fits in RAM without it.
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass
    try:
        pipe.enable_model_cpu_offload()  # keep peak RAM low on CPU
    except Exception as e:
        print(f"[warn] cpu_offload unavailable ({e}); full-resident",
              flush=True)
    return pipe


def gen_animal(n=6, seed0=100, src=None, outdir=None, species="cat"):
    """任意动物正脸参考 (物种参数化, 2026-10-01 用户要求泛化):
    SD1.5 + IP-Adapter 0.72, 提示词按物种模板生成。
    species 用英文 (SD 词表: dog/rabbit/tiger/panda/fox/horse/...)"""
    from diffusers import StableDiffusionPipeline
    ckpt = _find_file(SD15, "v1-5-pruned-emaonly.safetensors")
    pipe = StableDiffusionPipeline.from_single_file(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    _tune(pipe)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.72)

    src = cv2.imread(src or os.path.join(ROOT, "4", "cat.jpg"))
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    ip_img = Image.fromarray(src).resize((512, 512))

    subject = f"a {species}" if species else "the same animal"
    prompt = (f"front view portrait of {subject} looking directly at the "
              "camera, symmetric face, both eyes visible, centered nose, "
              "detailed fur, soft green blurred background, "
              "sharp photograph")
    neg = ("side view, profile, turned head, deformed, asymmetric, extra "
           "ears, extra eyes, blurry, watermark, text, cartoon, drawing")
    outdir = outdir or os.path.join(OUT, species)
    os.makedirs(outdir, exist_ok=True)
    print(f"[animal] species={species} n={n}", flush=True)
    for i in range(n):
        t0 = time.time()
        g = torch.Generator().manual_seed(seed0 + i)
        img = pipe(prompt=prompt, negative_prompt=neg,
                   ip_adapter_image=ip_img, num_inference_steps=25,
                   guidance_scale=7.0, height=512, width=512,
                   generator=g).images[0]
        img.save(os.path.join(outdir, f"cand_{i}.png"))
        print(f"[{species} {i}] {time.time()-t0:.0f}s", flush=True)


def gen_cat(n=6, seed0=100, src=None, outdir=None):
    gen_animal(n, seed0, src, outdir, species="cat")




def gen_face(n=6, seed0=200):
    from diffusers import (ControlNetModel,
                           StableDiffusionControlNetPipeline)
    ckpt = _snapshot_dir(ANIME)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float32)
    from diffusers import StableDiffusionPipeline as _SDP
    base = _SDP.from_pretrained(ckpt, torch_dtype=torch.float32,
                                safety_checker=None)
    pipe = StableDiffusionControlNetPipeline(
        vae=base.vae, text_encoder=base.text_encoder,
        tokenizer=base.tokenizer, unet=base.unet,
        scheduler=base.scheduler, controlnet=cn,
        safety_checker=None, feature_extractor=None,
        requires_safety_checker=False)
    _tune(pipe)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.55)

    # Structure control: our cleaned symmetric classical result, as
    # black-on-white lineart (ControlNet lineart expects dark lines).
    cur = cv2.imread(os.path.join(ROOT, "assets", "control",
                                  "face_sym.png"))
    cur = cv2.resize(cur, (512, 640))
    g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    lines = 255 - np.clip((255 - g.astype(np.int32)) * 2, 0, 255).astype(
        np.uint8)  # boost faint lines
    control = Image.fromarray(lines).convert("RGB")

    src = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    ip_img = Image.fromarray(src).resize((512, 512))

    prompt = ("front view portrait of an anime boy facing the camera, "
              "symmetric face, short dark hair with center parting, both "
              "ears visible, wearing a hoodie with zipper and drawstrings, "
              "monochrome sketch style, clean lineart, white background")
    neg = ("side view, profile, turned head, color, deformed, asymmetric, "
           "extra ears, two bodies, double shoulders, blurry, watermark")
    outdir = os.path.join(OUT, "face")
    os.makedirs(outdir, exist_ok=True)
    for i in range(n):
        t0 = time.time()
        g_ = torch.Generator().manual_seed(seed0 + i)
        img = pipe(prompt=prompt, negative_prompt=neg,
                   image=control, controlnet_conditioning_scale=0.85,
                   ip_adapter_image=ip_img, num_inference_steps=25,
                   guidance_scale=7.0, height=640, width=512,
                   generator=g_).images[0]
        img.save(os.path.join(outdir, f"cand_{i}.png"))
        print(f"[face {i}] {time.time()-t0:.0f}s", flush=True)


def gen_face2(n=6, seed0=300):
    """V2: style-faithful frontal refs for face.png.

    User feedback on v1: candidates look like generic digital-manga boys,
    NOT the original's delicate pencil-sketch style.  Fixes:
      - prompt describes the exact style (graphite pencil sketch, soft
        hatching, off-white paper) and the exact outfit/hair details
      - IP-Adapter scale 0.55 -> 0.80 (carries both identity AND style)
      - negative kills cel-shading / thick-ink / blush-sticker traits
    Output: assets/ref_gen/face2/cand_<i>.png
    """
    from diffusers import (ControlNetModel,
                           StableDiffusionControlNetPipeline)
    ckpt = _snapshot_dir(ANIME)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float32)
    from diffusers import StableDiffusionPipeline as _SDP
    base = _SDP.from_pretrained(ckpt, torch_dtype=torch.float32,
                                safety_checker=None)
    pipe = StableDiffusionControlNetPipeline(
        vae=base.vae, text_encoder=base.text_encoder,
        tokenizer=base.tokenizer, unet=base.unet,
        scheduler=base.scheduler, controlnet=cn,
        safety_checker=None, feature_extractor=None,
        requires_safety_checker=False)
    _tune(pipe)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.80)

    cur = cv2.imread(os.path.join(ROOT, "assets", "control",
                                  "face_sym.png"))
    cur = cv2.resize(cur, (512, 640))
    g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    lines = 255 - np.clip((255 - g.astype(np.int32)) * 2, 0, 255).astype(
        np.uint8)
    control = Image.fromarray(lines).convert("RGB")

    src = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    ip_img = Image.fromarray(src).resize((512, 512))

    prompt = ("hand-drawn graphite pencil sketch of an anime boy, front "
              "view portrait facing the camera, symmetric face, delicate "
              "pencil hatching and soft graphite shading, monochrome "
              "greyscale on off-white paper, short tousled dark hair with "
              "fine pencil strands and fringe, large gentle dark eyes "
              "with detailed iris, both ears visible, wearing a light "
              "hooded jacket with dark inner lining, drawstrings and a "
              "small round metal ring on the chest, soft sketch lines, "
              "traditional manga illustration, masterpiece")
    neg = ("digital painting, cel shading, thick black ink outlines, "
           "flat colors, color, blush stickers, 3d render, glossy, "
           "photorealistic, side view, profile, turned head, deformed, "
           "asymmetric, extra ears, two bodies, double shoulders, "
           "watermark, text, logo")
    outdir = os.path.join(OUT, "face2")
    os.makedirs(outdir, exist_ok=True)
    for i in range(n):
        t0 = time.time()
        g_ = torch.Generator().manual_seed(seed0 + i)
        img = pipe(prompt=prompt, negative_prompt=neg,
                   image=control, controlnet_conditioning_scale=0.85,
                   ip_adapter_image=ip_img, num_inference_steps=25,
                   guidance_scale=7.0, height=640, width=512,
                   generator=g_).images[0]
        img.save(os.path.join(outdir, f"cand_{i}.png"))
        print(f"[face2 {i}] {time.time()-t0:.0f}s", flush=True)


def gen_face3(n=6, seed0=400):
    """V3: fix v2's remaining flaws — eye COLOR leak (teal/red/blue
    instead of the original's dark eyes), too-light hair, textured
    backgrounds.  Output: assets/ref_gen/face3/cand_<i>.png"""
    from diffusers import (ControlNetModel,
                           StableDiffusionControlNetPipeline)
    ckpt = _snapshot_dir(ANIME)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float32)
    from diffusers import StableDiffusionPipeline as _SDP
    base = _SDP.from_pretrained(ckpt, torch_dtype=torch.float32,
                                safety_checker=None)
    pipe = StableDiffusionControlNetPipeline(
        vae=base.vae, text_encoder=base.text_encoder,
        tokenizer=base.tokenizer, unet=base.unet,
        scheduler=base.scheduler, controlnet=cn,
        safety_checker=None, feature_extractor=None,
        requires_safety_checker=False)
    _tune(pipe)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.80)

    cur = cv2.imread(os.path.join(ROOT, "assets", "control",
                                  "face_sym.png"))
    cur = cv2.resize(cur, (512, 640))
    g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    lines = 255 - np.clip((255 - g.astype(np.int32)) * 2, 0, 255).astype(
        np.uint8)
    control = Image.fromarray(lines).convert("RGB")

    src = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    ip_img = Image.fromarray(src).resize((512, 512))

    prompt = ("hand-drawn graphite pencil sketch of an anime boy, front "
              "view portrait facing the camera, symmetric face, delicate "
              "pencil hatching, monochrome greyscale, plain off-white "
              "paper background, short tousled dark black hair, large "
              "gentle dark brown eyes, dark grey irises, both ears "
              "visible, light hooded jacket with dark inner lining and "
              "drawstrings, traditional manga sketch")
    neg = ("colored eyes, blue eyes, red eyes, teal eyes, green eyes, "
           "amber eyes, blonde hair, light hair, white hair, digital "
           "painting, cel shading, thick ink outlines, color, 3d, "
           "glossy, photorealistic, brick wall, textured background, "
           "speed lines, side view, profile, deformed, asymmetric, "
           "extra ears, watermark, text")
    outdir = os.path.join(OUT, "face3")
    os.makedirs(outdir, exist_ok=True)
    for i in range(n):
        t0 = time.time()
        g_ = torch.Generator().manual_seed(seed0 + i)
        img = pipe(prompt=prompt, negative_prompt=neg,
                   image=control, controlnet_conditioning_scale=0.85,
                   ip_adapter_image=ip_img, num_inference_steps=25,
                   guidance_scale=7.0, height=640, width=512,
                   generator=g_).images[0]
        img.save(os.path.join(outdir, f"cand_{i}.png"))
        print(f"[face3 {i}] {time.time()-t0:.0f}s", flush=True)


def gen_face4():
    """V4: img2img REDRAW of our own classical result (identity-exact
    pixels as the starting point, NOT pure noise).  SD only naturalizes
    the mirror-composite artifacts (hollow eyes, chest wedge, helmet
    hair).  No ControlNet — it would preserve the hollow-eye structure.
    Strength sweep 0.45/0.55/0.65 x 2 seeds, grayscale-normalized output.
    Output: assets/ref_gen/face4/cand_s<strength>_<seed>.png"""
    from diffusers import StableDiffusionImg2ImgPipeline
    ckpt = _snapshot_dir(ANIME)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    _tune(pipe)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.60)

    init = cv2.imread(os.path.join(ROOT, "assets", "control",
                                   "face_classical_now.png"))
    init = cv2.resize(init, (512, 640))
    init = Image.fromarray(cv2.cvtColor(init, cv2.COLOR_BGR2RGB))

    src = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ip_img = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
                             ).resize((512, 512))

    prompt = ("hand-drawn graphite pencil sketch of an anime boy, front "
              "view portrait, symmetric face, large gentle dark eyes, "
              "short tousled dark hair, light hooded jacket with dark "
              "inner lining and drawstrings, delicate pencil hatching, "
              "monochrome greyscale, off-white paper background")
    neg = ("color, colored eyes, blue eyes, red eyes, digital painting, "
           "cel shading, 3d, photorealistic, side view, profile, "
           "deformed, asymmetric, hollow eyes, missing eyes, extra "
           "ears, two bodies, watermark, text")
    outdir = os.path.join(OUT, "face4")
    os.makedirs(outdir, exist_ok=True)
    for strength in (0.45, 0.55, 0.65):
        for seed in (500, 501):
            t0 = time.time()
            g_ = torch.Generator().manual_seed(seed)
            img = pipe(prompt=prompt, negative_prompt=neg, image=init,
                       strength=strength, ip_adapter_image=ip_img,
                       num_inference_steps=25, guidance_scale=7.0,
                       generator=g_).images[0]
            tag = f"cand_s{int(strength*100)}_{seed}"
            img.save(os.path.join(outdir, f"{tag}_raw.png"))
            # grayscale-normalize to the original's tonal range
            orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                              cv2.IMREAD_GRAYSCALE)
            gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                                ).astype(np.float32)
            gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                    * orig.std() + orig.mean())
            gimg = np.clip(gimg, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
            print(f"[face4 {tag}] {time.time()-t0:.0f}s", flush=True)


def gen_face5():
    """V5: img2img + ControlNet lineart — DOUBLE composition lock.
    User directive: strictly frontal, single body, single neck, no side
    turn.  init = classical result (identity pixels); ControlNet lineart
    of the same image locks the frontal/symmetric/single-body layout
    (the init's face interior is blank -> no hollow-eye constraint, the
    features are still freely redrawn).  Output: assets/ref_gen/face5/
    """
    from diffusers import (ControlNetModel,
                           StableDiffusionControlNetImg2ImgPipeline,
                           StableDiffusionImg2ImgPipeline)
    ckpt = _snapshot_dir(ANIME)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float32)
    base = StableDiffusionImg2ImgPipeline.from_pretrained(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    pipe = StableDiffusionControlNetImg2ImgPipeline(
        vae=base.vae, text_encoder=base.text_encoder,
        tokenizer=base.tokenizer, unet=base.unet,
        scheduler=base.scheduler, controlnet=cn,
        safety_checker=None, feature_extractor=None,
        requires_safety_checker=False)
    _tune(pipe)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.60)

    init = cv2.imread(os.path.join(ROOT, "assets", "control",
                                   "face_classical_now.png"))
    init = cv2.resize(init, (512, 640))
    init_rgb = Image.fromarray(cv2.cvtColor(init, cv2.COLOR_BGR2RGB))

    g = cv2.cvtColor(init, cv2.COLOR_BGR2GRAY)
    lines = 255 - np.clip((255 - g.astype(np.int32)) * 2, 0, 255).astype(
        np.uint8)
    control = Image.fromarray(lines).convert("RGB")

    src = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ip_img = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
                             ).resize((512, 512))

    prompt = ("hand-drawn graphite pencil sketch of an anime boy, "
              "strict front view, facing the camera directly, perfectly "
              "symmetric face and body, single body, one neck, large "
              "gentle dark eyes looking forward, short tousled dark "
              "hair, light hooded jacket with dark inner lining, "
              "drawstrings and front zipper, continuous chest, "
              "delicate pencil hatching, monochrome greyscale, "
              "off-white paper background")
    neg = ("side view, 3/4 view, turned head, looking away, two "
           "bodies, two necks, double shoulders, split chest, gap, "
           "hollow eyes, missing eyes, color, colored eyes, digital "
           "painting, cel shading, 3d, photorealistic, deformed, "
           "asymmetric, extra ears, watermark, text")
    outdir = os.path.join(OUT, "face5")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)
    for strength in (0.50, 0.55, 0.60):
        for seed in (600, 601):
            t0 = time.time()
            g_ = torch.Generator().manual_seed(seed)
            img = pipe(prompt=prompt, negative_prompt=neg,
                       image=init_rgb, strength=strength,
                       control_image=control,
                       controlnet_conditioning_scale=0.70,
                       ip_adapter_image=ip_img,
                       num_inference_steps=25, guidance_scale=7.0,
                       generator=g_).images[0]
            tag = f"cand_s{int(strength*100)}_{seed}"
            img.save(os.path.join(outdir, f"{tag}_raw.png"))
            gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                                ).astype(np.float32)
            gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                    * orig.std() + orig.mean())
            gimg = np.clip(gimg, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
            print(f"[face5 {tag}] {time.time()-t0:.0f}s", flush=True)


def gen_face6():
    """V6: same double-lock as v5, but the HAIR is unlocked — user:
    mirrored side-view hair reads as twin-tails, not a real frontal
    hairstyle.  The control lineart's hair region is erased and replaced
    by a simple short-hair skull arc, so SD draws a true frontal short
    hairstyle (guided by IP-Adapter from the original).  Face/body stay
    locked (strictly frontal, single body).  Output: assets/ref_gen/face6/
    """
    from diffusers import (ControlNetModel,
                           StableDiffusionControlNetImg2ImgPipeline,
                           StableDiffusionImg2ImgPipeline)
    ckpt = _snapshot_dir(ANIME)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float32)
    base = StableDiffusionImg2ImgPipeline.from_pretrained(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    pipe = StableDiffusionControlNetImg2ImgPipeline(
        vae=base.vae, text_encoder=base.text_encoder,
        tokenizer=base.tokenizer, unet=base.unet,
        scheduler=base.scheduler, controlnet=cn,
        safety_checker=None, feature_extractor=None,
        requires_safety_checker=False)
    _tune(pipe)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.60)

    init = cv2.imread(os.path.join(ROOT, "assets", "control",
                                   "face_classical_now.png"))
    init = cv2.resize(init, (512, 640))
    init_rgb = Image.fromarray(cv2.cvtColor(init, cv2.COLOR_BGR2RGB))

    g = cv2.cvtColor(init, cv2.COLOR_BGR2GRAY)
    lines = 255 - np.clip((255 - g.astype(np.int32)) * 2, 0, 255).astype(
        np.uint8)
    # --- unlock hair: erase mirrored-helmet hair lines, draw a simple
    # short-hair skull arc instead (512x640 canvas; coords measured on
    # face_classical_now.png) ---
    lines[:215, :] = 255                     # everything above the brows
    lines[150:390, :172] = 255               # left side-hair mass
    lines[150:390, 342:] = 255               # right side-hair mass
    cv2.ellipse(lines, (256, 245), (100, 135), 0, 180, 360, 0, 2)
    cv2.line(lines, (156, 245), (158, 305), 0, 2)   # left short side
    cv2.line(lines, (356, 245), (354, 305), 0, 2)   # right short side
    cv2.imwrite(os.path.join(ROOT, "assets", "control",
                             "face_control_nohair.png"), lines)
    control = Image.fromarray(lines).convert("RGB")

    src = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ip_img = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
                             ).resize((512, 512))

    prompt = ("hand-drawn graphite pencil sketch of an anime boy, "
              "strict front view, facing the camera directly, perfectly "
              "symmetric face and body, single body, one neck, large "
              "gentle dark eyes looking forward, short dark hair with "
              "fringe on the forehead, hair above the ears, light "
              "hooded jacket with dark inner lining, drawstrings and "
              "front zipper, continuous chest, delicate pencil "
              "hatching, monochrome greyscale, off-white paper")
    neg = ("twin tails, pigtails, ponytail, long side hair, long hair, "
           "side view, 3/4 view, turned head, looking away, two "
           "bodies, two necks, double shoulders, split chest, gap, "
           "hollow eyes, missing eyes, color, colored eyes, digital "
           "painting, cel shading, 3d, photorealistic, deformed, "
           "asymmetric, extra ears, watermark, text")
    outdir = os.path.join(OUT, "face6")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)
    for strength in (0.55, 0.60, 0.65):
        for seed in (700, 701):
            t0 = time.time()
            g_ = torch.Generator().manual_seed(seed)
            img = pipe(prompt=prompt, negative_prompt=neg,
                       image=init_rgb, strength=strength,
                       control_image=control,
                       controlnet_conditioning_scale=0.70,
                       ip_adapter_image=ip_img,
                       num_inference_steps=25, guidance_scale=7.0,
                       generator=g_).images[0]
            tag = f"cand_s{int(strength*100)}_{seed}"
            img.save(os.path.join(outdir, f"{tag}_raw.png"))
            gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                                ).astype(np.float32)
            gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                    * orig.std() + orig.mean())
            gimg = np.clip(gimg, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
            print(f"[face6 {tag}] {time.time()-t0:.0f}s", flush=True)


def gen_face7():
    """V7: v6's winning recipe (img2img + ControlNet hair-unlocked) with
    identity-tuned prompt + more seeds.  User directive: MUST generate;
    tune prompt and seeds carefully.  Prompt now describes the original
    boy's exact signature features: thick dark upper lash line, large
    dark iris, fringe over forehead, spiky tufts at the back of the
    head.  Output: assets/ref_gen/face7/"""
    from diffusers import (ControlNetModel,
                           StableDiffusionControlNetImg2ImgPipeline,
                           StableDiffusionImg2ImgPipeline)
    ckpt = _snapshot_dir(ANIME)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float32)
    base = StableDiffusionImg2ImgPipeline.from_pretrained(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    pipe = StableDiffusionControlNetImg2ImgPipeline(
        vae=base.vae, text_encoder=base.text_encoder,
        tokenizer=base.tokenizer, unet=base.unet,
        scheduler=base.scheduler, controlnet=cn,
        safety_checker=None, feature_extractor=None,
        requires_safety_checker=False)
    _tune(pipe)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.60)

    init = cv2.imread(os.path.join(ROOT, "assets", "control",
                                   "face_classical_now.png"))
    init = cv2.resize(init, (512, 640))
    init_rgb = Image.fromarray(cv2.cvtColor(init, cv2.COLOR_BGR2RGB))
    control = Image.fromarray(
        cv2.imread(os.path.join(ROOT, "assets", "control",
                                "face_control_nohair.png"))).convert("RGB")

    src = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ip_img = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
                             ).resize((512, 512))

    prompt = ("hand-drawn graphite pencil sketch of an anime boy, "
              "strict front view, facing the camera directly, perfectly "
              "symmetric face and body, single body, one neck, large "
              "expressive eyes with thick dark upper lash line and "
              "large dark iris, gentle gaze looking forward, short "
              "dark hair with fringe covering the forehead and spiky "
              "tufts at the back, hair above the ears, light hooded "
              "jacket with dark inner lining, drawstrings and front "
              "zipper, continuous chest, delicate pencil hatching, "
              "monochrome greyscale, off-white paper")
    neg = ("twin tails, pigtails, ponytail, long side hair, long hair, "
           "side view, 3/4 view, turned head, looking away, narrow "
           "eyes, small eyes, closed eyes, two bodies, two necks, "
           "double shoulders, split chest, gap, hollow eyes, missing "
           "eyes, color, colored eyes, digital painting, cel shading, "
           "3d, photorealistic, deformed, asymmetric, extra ears, "
           "watermark, text")
    outdir = os.path.join(OUT, "face7")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)
    for strength in (0.55, 0.60):
        for seed in (800, 801, 802):
            t0 = time.time()
            g_ = torch.Generator().manual_seed(seed)
            img = pipe(prompt=prompt, negative_prompt=neg,
                       image=init_rgb, strength=strength,
                       control_image=control,
                       controlnet_conditioning_scale=0.70,
                       ip_adapter_image=ip_img,
                       num_inference_steps=25, guidance_scale=7.0,
                       generator=g_).images[0]
            tag = f"cand_s{int(strength*100)}_{seed}"
            img.save(os.path.join(outdir, f"{tag}_raw.png"))
            gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                                ).astype(np.float32)
            gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                    * orig.std() + orig.mean())
            gimg = np.clip(gimg, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
            print(f"[face7 {tag}] {time.time()-t0:.0f}s", flush=True)


def gen_face8():
    """V8: v7 recipe + LCM-LoRA speed (user: 'others generate in 5-10
    min').  25 steps/guidance 7 -> 10 steps/guidance 2.0, expected
    ~2-4 min/img on CPU.  More seeds this round since each is cheap.
    Output: assets/ref_gen/face8/"""
    from diffusers import (ControlNetModel, LCMScheduler,
                           StableDiffusionControlNetImg2ImgPipeline,
                           StableDiffusionImg2ImgPipeline)
    ckpt = _snapshot_dir(ANIME)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float32)
    base = StableDiffusionImg2ImgPipeline.from_pretrained(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    pipe = StableDiffusionControlNetImg2ImgPipeline(
        vae=base.vae, text_encoder=base.text_encoder,
        tokenizer=base.tokenizer, unet=base.unet,
        scheduler=base.scheduler, controlnet=cn,
        safety_checker=None, feature_extractor=None,
        requires_safety_checker=False)
    _tune(pipe)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    lcm = os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5", "snapshots")
    lcm = os.path.join(max(
        [os.path.join(lcm, d) for d in os.listdir(lcm)],
        key=os.path.getmtime), "pytorch_lora_weights.safetensors")
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora failed ({e})", flush=True)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.60)

    init = cv2.imread(os.path.join(ROOT, "assets", "control",
                                   "face_classical_now.png"))
    init = cv2.resize(init, (512, 640))
    init_rgb = Image.fromarray(cv2.cvtColor(init, cv2.COLOR_BGR2RGB))
    control = Image.fromarray(
        cv2.imread(os.path.join(ROOT, "assets", "control",
                                "face_control_nohair.png"))).convert("RGB")

    src = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ip_img = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
                             ).resize((512, 512))

    prompt = ("hand-drawn graphite pencil sketch of an anime boy, "
              "strict front view, facing the camera directly, perfectly "
              "symmetric face and body, single body, one neck, large "
              "expressive eyes with thick dark upper lash line and "
              "large dark iris, gentle gaze looking forward, short "
              "dark hair with fringe covering the forehead and spiky "
              "tufts at the back, hair above the ears, light hooded "
              "jacket with dark inner lining, drawstrings and front "
              "zipper, continuous chest, delicate pencil hatching, "
              "monochrome greyscale, off-white paper")
    neg = ("twin tails, pigtails, ponytail, long side hair, long hair, "
           "side view, 3/4 view, turned head, looking away, narrow "
           "eyes, small eyes, closed eyes, two bodies, two necks, "
           "double shoulders, split chest, gap, hollow eyes, missing "
           "eyes, color, colored eyes, digital painting, cel shading, "
           "3d, photorealistic, deformed, asymmetric, extra ears, "
           "watermark, text")
    outdir = os.path.join(OUT, "face8")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)
    for strength in (0.55, 0.60):
        for seed in (900, 901, 902, 903):
            t0 = time.time()
            g_ = torch.Generator().manual_seed(seed)
            img = pipe(prompt=prompt, negative_prompt=neg,
                       image=init_rgb, strength=strength,
                       control_image=control,
                       controlnet_conditioning_scale=0.70,
                       ip_adapter_image=ip_img,
                       num_inference_steps=10, guidance_scale=2.0,
                       generator=g_).images[0]
            tag = f"cand_s{int(strength*100)}_{seed}"
            img.save(os.path.join(outdir, f"{tag}_raw.png"))
            gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                                ).astype(np.float32)
            gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                    * orig.std() + orig.mean())
            gimg = np.clip(gimg, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
            print(f"[face8 {tag}] {time.time()-t0:.0f}s", flush=True)


def gen_face9():
    """V9: REFERENCE-ONLY attention (the official reference-only
    ControlNet trick, hand-implemented since diffusers 0.40 dropped the
    community pipeline) + upgraded prompt + LCM speed.

    Research (user: 'search the web for solutions'): the standard anime
    character-consistency stack is pose-lock (ControlNet/img2img) +
    identity-lock (IP-Adapter) + reference_only for style/texture. We
    already had the first two; reference-only is the missing piece: at
    every denoise step the ORIGINAL drawing is run through the UNet
    once (write mode), and its self-attention K/V are concatenated into
    the main pass (read mode) — the generated face is literally built
    attending to the original's strokes, so hair texture, eye style and
    pencil hatching follow the source far more closely than IP-Adapter
    embeddings alone. IP-Adapter is dropped here (reference-only
    replaces it as the identity lock and keeps the loop simple).

    Custom denoise loop (img2img + LCM, CFG 2.0, 10 steps).
    Output: assets/ref_gen/face9/"""
    import torch.nn.functional as F
    from diffusers import LCMScheduler, StableDiffusionImg2ImgPipeline
    ckpt = _snapshot_dir(ANIME)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    _tune(pipe)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    lcm = os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5", "snapshots")
    lcm = os.path.join(max(
        [os.path.join(lcm, d) for d in os.listdir(lcm)],
        key=os.path.getmtime), "pytorch_lora_weights.safetensors")
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora failed ({e})", flush=True)
    pipe.set_progress_bar_config(disable=True)

    # --- reference-only attention processors (attn1 = self-attn) ---
    BANK = {}

    class RefProc:
        def __init__(self, name, mode):
            self.name = name
            self.mode = mode

        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, temb=None, **kw):
            residual = hidden_states
            is_self = encoder_hidden_states is None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross is not None:
                encoder_hidden_states = attn.norm_cross(
                    encoder_hidden_states)
            b, n_q, _ = hidden_states.shape
            q = attn.to_q(hidden_states)
            k = attn.to_k(encoder_hidden_states)
            v = attn.to_v(encoder_hidden_states)
            hd, dh = attn.heads, q.shape[-1] // attn.heads
            q = q.view(b, -1, hd, dh).transpose(1, 2)
            k = k.view(b, -1, hd, dh).transpose(1, 2)
            v = v.view(b, -1, hd, dh).transpose(1, 2)
            if is_self and self.mode == "write":
                BANK[self.name] = (k[:1].clone(), v[:1].clone())
            elif is_self and self.mode == "read" and self.name in BANK:
                bk, bv = BANK[self.name]
                k = torch.cat([k, bk.repeat(b, 1, 1, 1)], dim=2)
                v = torch.cat([v, bv.repeat(b, 1, 1, 1)], dim=2)
            o = F.scaled_dot_product_attention(q, k, v)
            o = o.transpose(1, 2).reshape(b, n_q, hd * dh)
            o = attn.to_out[0](o)
            o = attn.to_out[1](o)
            if attn.residual_connection:
                o = o + residual
            return o / attn.rescale_output_factor

    attn1 = {n: RefProc(n, "write")
             for n in pipe.unet.attn_processors if ".attn1." in n}
    procs = dict(pipe.unet.attn_processors)
    procs.update(attn1)
    pipe.unet.set_attn_processor(procs)

    def set_mode(mode):
        for p in attn1.values():
            p.mode = mode

    # --- inputs ---
    init = cv2.imread(os.path.join(ROOT, "assets", "control",
                                   "face_classical_now.png"))
    init = cv2.resize(init, (512, 640))
    init_rgb = Image.fromarray(cv2.cvtColor(init, cv2.COLOR_BGR2RGB))
    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))

    prompt = ("monochrome graphite pencil sketch portrait of an anime "
              "boy, strict front view facing the camera, perfectly "
              "symmetric face and body, single body one neck, large "
              "dark eyes with thick black upper lash line and big dark "
              "iris, short black hair with straight fringe across the "
              "forehead and small spiky tufts at the crown, hair above "
              "the ears, white hooded jacket with dark inner lining, "
              "front zipper and drawstrings, continuous closed chest, "
              "hand-drawn pencil line art, soft graphite hatching, "
              "off-white paper")
    neg = ("twin tails, pigtails, ponytail, long side hair, long hair, "
           "side view, 3/4 view, turned head, looking away, narrow "
           "eyes, small eyes, closed eyes, eyeshadow, blush, two "
           "bodies, two necks, double shoulders, split chest, gap, "
           "hollow eyes, missing eyes, color, colored eyes, digital "
           "painting, cel shading, watercolor, ink wash, 3d, "
           "photorealistic, deformed, asymmetric, extra ears, "
           "watermark, text")

    steps, guidance = 10, 2.0
    pe, ne_ = pipe.encode_prompt(prompt, "cpu", 1, True, neg)
    prompt_embeds = torch.cat([ne_, pe])
    sf = pipe.vae.config.scaling_factor
    init_t = pipe.image_processor.preprocess(init_rgb)
    ref_t = pipe.image_processor.preprocess(ref_rgb)

    outdir = os.path.join(OUT, "face9")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    @torch.no_grad()
    def _run(strength, seed):
            t0 = time.time()
            g_ = torch.Generator().manual_seed(seed)
            init_lat = (pipe.vae.encode(init_t).latent_dist.sample(g_)
                        * sf)
            ref_lat = (pipe.vae.encode(ref_t).latent_dist.sample(g_)
                       * sf)
            pipe.scheduler.set_timesteps(steps, device="cpu")
            timesteps, _ = pipe.get_timesteps(steps, strength, "cpu")
            noise = torch.randn(init_lat.shape, generator=g_)
            latents = pipe.scheduler.add_noise(
                init_lat, noise, timesteps[:1])
            for t in timesteps:
                BANK.clear()
                set_mode("write")
                rn = torch.randn(ref_lat.shape, generator=g_)
                ref_xt = pipe.scheduler.add_noise(ref_lat, rn, t)
                pipe.unet(ref_xt, t, encoder_hidden_states=pe)
                set_mode("read")
                lin = torch.cat([latents] * 2)
                npred = pipe.unet(
                    lin, t, encoder_hidden_states=prompt_embeds).sample
                nu, nc = npred.chunk(2)
                npred = nu + guidance * (nc - nu)
                latents = pipe.scheduler.step(
                    npred, t, latents, generator=g_).prev_sample
            img = pipe.vae.decode(
                latents / sf, return_dict=False)[0]
            img = pipe.image_processor.postprocess(
                img, output_type="pil")[0]
            tag = f"cand_s{int(strength*100)}_{seed}"
            img.save(os.path.join(outdir, f"{tag}_raw.png"))
            gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                                ).astype(np.float32)
            gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                    * orig.std() + orig.mean())
            gimg = np.clip(gimg, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
            print(f"[face9 {tag}] {time.time()-t0:.0f}s", flush=True)

    for strength in (0.55, 0.60):
        for seed in (950, 951):
            _run(strength, seed)


def gen_face10():
    """V10: same reference-only machinery as V9, but the base checkpoint
    switches from AnythingV5 (strong clean-anime prior -> 'cartoonish',
    rejected by user) to plain SD1.5. With no anime prior, the
    reference-only K/V from the original pencil sketch dominates the
    style, so the output should read as a rough hand-drawn graphite
    sketch instead of polished anime lineart. Prompt is rewritten
    sketch-first ('rough hand-drawn pencil sketch', 'visible hatching',
    negatives: anime/cel shading/clean lineart), 'strict front view'
    moved to the front of the prompt, and strength lowered to
    0.50/0.55 so the strictly-frontal init image anchors the pose.
    Output: assets/ref_gen/face10/"""
    import torch.nn.functional as F
    from diffusers import LCMScheduler, StableDiffusionImg2ImgPipeline
    ckpt = _find_file(SD15, "v1-5-pruned-emaonly.safetensors")
    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    _tune(pipe)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    lcm = os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5", "snapshots")
    lcm = os.path.join(max(
        [os.path.join(lcm, d) for d in os.listdir(lcm)],
        key=os.path.getmtime), "pytorch_lora_weights.safetensors")
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora failed ({e})", flush=True)
    pipe.set_progress_bar_config(disable=True)

    # --- reference-only attention processors (attn1 = self-attn) ---
    BANK = {}

    class RefProc:
        def __init__(self, name, mode):
            self.name = name
            self.mode = mode

        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, temb=None, **kw):
            residual = hidden_states
            is_self = encoder_hidden_states is None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross is not None:
                encoder_hidden_states = attn.norm_cross(
                    encoder_hidden_states)
            b, n_q, _ = hidden_states.shape
            q = attn.to_q(hidden_states)
            k = attn.to_k(encoder_hidden_states)
            v = attn.to_v(encoder_hidden_states)
            hd, dh = attn.heads, q.shape[-1] // attn.heads
            q = q.view(b, -1, hd, dh).transpose(1, 2)
            k = k.view(b, -1, hd, dh).transpose(1, 2)
            v = v.view(b, -1, hd, dh).transpose(1, 2)
            if is_self and self.mode == "write":
                BANK[self.name] = (k[:1].clone(), v[:1].clone())
            elif is_self and self.mode == "read" and self.name in BANK:
                bk, bv = BANK[self.name]
                k = torch.cat([k, bk.repeat(b, 1, 1, 1)], dim=2)
                v = torch.cat([v, bv.repeat(b, 1, 1, 1)], dim=2)
            o = F.scaled_dot_product_attention(q, k, v)
            o = o.transpose(1, 2).reshape(b, n_q, hd * dh)
            o = attn.to_out[0](o)
            o = attn.to_out[1](o)
            if attn.residual_connection:
                o = o + residual
            return o / attn.rescale_output_factor

    attn1 = {n: RefProc(n, "write")
             for n in pipe.unet.attn_processors if ".attn1." in n}
    procs = dict(pipe.unet.attn_processors)
    procs.update(attn1)
    pipe.unet.set_attn_processor(procs)

    def set_mode(mode):
        for p in attn1.values():
            p.mode = mode

    # --- inputs ---
    init = cv2.imread(os.path.join(ROOT, "assets", "control",
                                   "face_classical_now.png"))
    init = cv2.resize(init, (512, 640))
    init_rgb = Image.fromarray(cv2.cvtColor(init, cv2.COLOR_BGR2RGB))
    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))

    prompt = ("rough hand-drawn pencil sketch portrait of a boy, "
              "strict front view facing viewer, symmetric face, "
              "single body one neck, short black hair with straight "
              "fringe, large dark eyes, white hooded jacket, closed "
              "chest with zipper, monochrome graphite, sketchy pencil "
              "strokes, visible hatching, off-white paper, sketchbook "
              "style")
    neg = ("anime, cel shading, clean lineart, digital art, vector, "
           "smooth clean lines, side view, 3/4 view, turned head, "
           "profile, colored, color, watercolor, ink wash, 3d render, "
           "photorealistic, twin tails, long hair, two bodies, two "
           "necks, split chest, deformed, asymmetric, watermark, text")

    steps, guidance = 10, 2.0
    pe, ne_ = pipe.encode_prompt(prompt, "cpu", 1, True, neg)
    prompt_embeds = torch.cat([ne_, pe])
    sf = pipe.vae.config.scaling_factor
    init_t = pipe.image_processor.preprocess(init_rgb)
    ref_t = pipe.image_processor.preprocess(ref_rgb)

    outdir = os.path.join(OUT, "face10")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    @torch.no_grad()
    def _run(strength, seed):
            t0 = time.time()
            g_ = torch.Generator().manual_seed(seed)
            init_lat = (pipe.vae.encode(init_t).latent_dist.sample(g_)
                        * sf)
            ref_lat = (pipe.vae.encode(ref_t).latent_dist.sample(g_)
                       * sf)
            pipe.scheduler.set_timesteps(steps, device="cpu")
            timesteps, _ = pipe.get_timesteps(steps, strength, "cpu")
            noise = torch.randn(init_lat.shape, generator=g_)
            latents = pipe.scheduler.add_noise(
                init_lat, noise, timesteps[:1])
            for t in timesteps:
                BANK.clear()
                set_mode("write")
                rn = torch.randn(ref_lat.shape, generator=g_)
                ref_xt = pipe.scheduler.add_noise(ref_lat, rn, t)
                pipe.unet(ref_xt, t, encoder_hidden_states=pe)
                set_mode("read")
                lin = torch.cat([latents] * 2)
                npred = pipe.unet(
                    lin, t, encoder_hidden_states=prompt_embeds).sample
                nu, nc = npred.chunk(2)
                npred = nu + guidance * (nc - nu)
                latents = pipe.scheduler.step(
                    npred, t, latents, generator=g_).prev_sample
            img = pipe.vae.decode(
                latents / sf, return_dict=False)[0]
            img = pipe.image_processor.postprocess(
                img, output_type="pil")[0]
            tag = f"cand_s{int(strength*100)}_{seed}"
            img.save(os.path.join(outdir, f"{tag}_raw.png"))
            gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                                ).astype(np.float32)
            gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                    * orig.std() + orig.mean())
            gimg = np.clip(gimg, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
            print(f"[face10 {tag}] {time.time()-t0:.0f}s", flush=True)

    for strength in (0.50, 0.55):
        for seed in (960, 961):
            _run(strength, seed)


def _build_refonly_pipe(ckpt_path, single_file=False, refonly=True):
    """Shared V9+ machinery: img2img pipe + LCM-LoRA (+ hand-implemented
    reference-only attention when refonly=True). Returns
    (pipe, BANK, set_mode); BANK/set_mode are None when refonly=False
    (use _attach_refonly later, e.g. after load_ip_adapter)."""
    from diffusers import LCMScheduler, StableDiffusionImg2ImgPipeline
    if single_file:
        pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            ckpt_path, torch_dtype=torch.float32, safety_checker=None)
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            ckpt_path, torch_dtype=torch.float32, safety_checker=None)
    _tune(pipe)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    lcm = os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5", "snapshots")
    lcm = os.path.join(max(
        [os.path.join(lcm, d) for d in os.listdir(lcm)],
        key=os.path.getmtime), "pytorch_lora_weights.safetensors")
    pipe.load_lora_weights(lcm)
    try:
        pipe.fuse_lora()
    except Exception as e:
        print(f"[warn] fuse_lora failed ({e})", flush=True)
    pipe.set_progress_bar_config(disable=True)
    if not refonly:
        return pipe, None, None
    _, BANK, set_mode = _attach_refonly(pipe)
    return pipe, BANK, set_mode


def _attach_refonly(pipe):
    """Hand-implemented reference-only attention (attn1 self-attn K/V
    bank). Returns (pipe, BANK, set_mode). Safe to call AFTER
    pipe.load_ip_adapter: it only replaces attn1 processors and keeps
    whatever attn2 processors (IP-Adapter) are registered."""
    import torch.nn.functional as F
    BANK = {}

    class RefProc:
        def __init__(self, name, mode):
            self.name = name
            self.mode = mode

        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, temb=None, **kw):
            residual = hidden_states
            is_self = encoder_hidden_states is None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross is not None:
                encoder_hidden_states = attn.norm_cross(
                    encoder_hidden_states)
            b, n_q, _ = hidden_states.shape
            q = attn.to_q(hidden_states)
            k = attn.to_k(encoder_hidden_states)
            v = attn.to_v(encoder_hidden_states)
            hd, dh = attn.heads, q.shape[-1] // attn.heads
            q = q.view(b, -1, hd, dh).transpose(1, 2)
            k = k.view(b, -1, hd, dh).transpose(1, 2)
            v = v.view(b, -1, hd, dh).transpose(1, 2)
            if is_self and self.mode == "write":
                BANK[self.name] = (k[:1].clone(), v[:1].clone())
            elif is_self and self.mode == "read" and self.name in BANK:
                bk, bv = BANK[self.name]
                k = torch.cat([k, bk.repeat(b, 1, 1, 1)], dim=2)
                v = torch.cat([v, bv.repeat(b, 1, 1, 1)], dim=2)
            o = F.scaled_dot_product_attention(q, k, v)
            o = o.transpose(1, 2).reshape(b, n_q, hd * dh)
            o = attn.to_out[0](o)
            o = attn.to_out[1](o)
            if attn.residual_connection:
                o = o + residual
            return o / attn.rescale_output_factor

    attn1 = {n: RefProc(n, "write")
             for n in pipe.unet.attn_processors if ".attn1." in n}
    procs = dict(pipe.unet.attn_processors)
    procs.update(attn1)
    pipe.unet.set_attn_processor(procs)

    def set_mode(mode):
        for p in attn1.values():
            p.mode = mode

    return pipe, BANK, set_mode


def _refonly_img2img(pipe, BANK, set_mode, init_rgb, ref_rgb, prompt, neg,
                     strength, seed, steps=10, guidance=2.0,
                     controlnet=None, control_img=None, control_scale=0.8,
                     controls=None, ip_embeds=None):
    """One reference-only img2img run. Returns a PIL image. When
    ControlNet(s) are given, their residuals are added to the READ pass
    so the target structure (e.g. a frontal lineart template) is a hard
    constraint rather than a prompt suggestion. `controls` is a list of
    (controlnet_model, cond_tensor, scale) tuples whose residuals are
    summed (multi-ControlNet); the single-controlnet args are wrapped
    into it for backward compatibility."""
    pe, ne_ = pipe.encode_prompt(prompt, "cpu", 1, True, neg)
    prompt_embeds = torch.cat([ne_, pe])
    sf = pipe.vae.config.scaling_factor
    init_t = pipe.image_processor.preprocess(init_rgb)
    ref_t = pipe.image_processor.preprocess(ref_rgb)

    @torch.no_grad()
    def _go():
        g_ = torch.Generator().manual_seed(seed)
        init_lat = (pipe.vae.encode(init_t).latent_dist.sample(g_) * sf)
        ref_lat = (pipe.vae.encode(ref_t).latent_dist.sample(g_) * sf)
        pipe.scheduler.set_timesteps(steps, device="cpu")
        timesteps, _ = pipe.get_timesteps(steps, strength, "cpu")
        noise = torch.randn(init_lat.shape, generator=g_)
        latents = pipe.scheduler.add_noise(init_lat, noise, timesteps[:1])
        for t in timesteps:
            BANK.clear()
            set_mode("write")
            rn = torch.randn(ref_lat.shape, generator=g_)
            ref_xt = pipe.scheduler.add_noise(ref_lat, rn, t)
            wkw = {}
            if ip_embeds is not None:
                # write pass is batch-1: use the POSITIVE half of the
                # [negative; positive] CFG embeds
                wkw = {"added_cond_kwargs": {
                    "image_embeds": [e[1:2] for e in ip_embeds]}}
            pipe.unet(ref_xt, t, encoder_hidden_states=pe, **wkw)
            set_mode("read")
            lin = torch.cat([latents] * 2)
            ctrls = list(controls or [])
            if controlnet is not None:
                ctrls.append((controlnet, control_img, control_scale))
            unet_kw = {}
            if ip_embeds is not None:
                unet_kw["added_cond_kwargs"] = {
                    "image_embeds": ip_embeds}
            if ctrls:
                down_acc, mid_acc = None, None
                for cnm, cimg, cs in ctrls:
                    down, mid = cnm(
                        lin, t, encoder_hidden_states=prompt_embeds,
                        controlnet_cond=cimg,
                        conditioning_scale=cs,
                        return_dict=False)
                    if down_acc is None:
                        down_acc = [d.clone() for d in down]
                        mid_acc = mid.clone()
                    else:
                        down_acc = [a + d for a, d in zip(down_acc, down)]
                        mid_acc = mid_acc + mid
                unet_kw.update(dict(
                    down_block_additional_residuals=down_acc,
                    mid_block_additional_residual=mid_acc))
            npred = pipe.unet(
                lin, t, encoder_hidden_states=prompt_embeds,
                **unet_kw).sample
            nu, nc = npred.chunk(2)
            npred = nu + guidance * (nc - nu)
            latents = pipe.scheduler.step(
                npred, t, latents, generator=g_).prev_sample
        img = pipe.vae.decode(latents / sf, return_dict=False)[0]
        return pipe.image_processor.postprocess(img, output_type="pil")[0]

    return _go()


def gen_face11():
    """V11: chase the ORIGINAL's pencil-anime style with zero new
    downloads (user: 'too cartoonish' about V9, 'too realistic, none
    look like the original' about V10). Two attacks, both on AnythingV5
    (keeps anime face proportions) + reference-only (copies original
    stroke texture):

    A) one-pass, Danbooru TAG prompt instead of natural language —
       anime checkpoints respond far better to (monochrome:1.3),
       (sketch:1.2), traditional media, messy lines etc. than to
       sentences; V9's cartoonish look partly came from CLIP-style
       phrasing that let the default digital-anime style dominate.
    B) two-pass: init from the V9 winner (anime structure already
       right) and run a LOW-strength sketch pass on top so only the
       rendering roughens into pencil while the face stays put.

    Output: assets/ref_gen/face11/"""
    ckpt = _snapshot_dir(ANIME)
    pipe, BANK, set_mode = _build_refonly_pipe(ckpt)

    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))

    prompt = ("(monochrome:1.3), (sketch:1.2), traditional media, "
              "pencil (medium), messy lines, hatching (texture), 1boy, "
              "solo, front view, looking at viewer, symmetrical face, "
              "large eyes, black eyes, short black hair, hair between "
              "eyes, white hoodie, drawstring, zipper, upper body, "
              "portrait, simple background, off-white background")
    neg = ("color, colored, clean lineart, cel shading, digital "
           "painting, 3d, realistic, photo, side view, 3/4 view, "
           "profile, turned head, twin tails, ponytail, long hair, "
           "two bodies, two necks, split chest, deformed, asymmetrical, "
           "watermark, text, signature")

    outdir = os.path.join(OUT, "face11")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)

    # --- A: one-pass from the classical composite, tag prompt ---
    init = cv2.imread(os.path.join(ROOT, "assets", "control",
                                   "face_classical_now.png"))
    init = cv2.resize(init, (512, 640))
    init_rgb = Image.fromarray(cv2.cvtColor(init, cv2.COLOR_BGR2RGB))
    for seed in (970, 971):
        t0 = time.time()
        img = _refonly_img2img(pipe, BANK, set_mode, init_rgb, ref_rgb,
                               prompt, neg, 0.55, seed)
        _save(img, f"candA_s55_{seed}")
        print(f"[face11 candA_s55_{seed}] {time.time()-t0:.0f}s",
              flush=True)

    # --- B: two-pass, low-strength re-sketch of the V9 winner ---
    init_b = cv2.imread(os.path.join(OUT, "face9", "cand_s55_950.png"))
    init_b = cv2.resize(init_b, (512, 640))
    init_b_rgb = Image.fromarray(cv2.cvtColor(init_b, cv2.COLOR_BGR2RGB))
    for strength in (0.32, 0.40):
        t0 = time.time()
        img = _refonly_img2img(pipe, BANK, set_mode, init_b_rgb, ref_rgb,
                               prompt, neg, strength, 970)
        _save(img, f"candB_s{int(strength*100)}_970")
        print(f"[face11 candB_s{int(strength*100)}_970] "
              f"{time.time()-t0:.0f}s", flush=True)


def _symmetrize(g, cx=None, y0=40, y1=440, xpad=80, feather=28,
                w_in=0.55, w_out=0.15):
    """Blend a grayscale candidate with its horizontal mirror around the
    face axis: guarantees a perfectly frontal, symmetric face instead of
    relying on model luck. Strong blend inside the face box, weak blend
    outside (clothes/bg keep most of their own detail)."""
    h, w = g.shape
    if cx is None:
        cx = w // 2
    fl = cv2.flip(g, 1)
    # shift the flipped image so that mirroring happens around cx
    # rather than the image centre
    shift = int(round(2 * cx - w))
    M = np.float32([[1, 0, shift], [0, 1, 0]])
    fl = cv2.warpAffine(fl, M, (w, h), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    m = np.zeros((h, w), np.float32)
    m[y0:y1, xpad:w - xpad] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), feather)
    alpha = w_out + (w_in - w_out) * m
    out = g.astype(np.float32) * (1 - alpha) + \
        fl.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def gen_face12():
    """V12: back to the SD1.5 pencil-sketch direction (the classmate's
    reference result photo shows exactly that style: semi-realistic
    pencil sketch, strictly frontal, short hair, visible ears, closed
    symmetric hoodie with centre zipper). Fixes vs V10: prompt gains
    'perfectly symmetric / visible ears / straight fringe / calm
    expression / centre zipper and drawstrings', negative gains
    'signature, hair over ears'; strength lowered to 0.45-0.55 so the
    strictly-frontal init anchors the pose; and every candidate also
    gets a mirror-SYMMETRIZED variant (_symmetrize) so frontality is
    mathematically enforced, not left to model luck.
    Output: assets/ref_gen/face12/"""
    ckpt = _find_file(SD15, "v1-5-pruned-emaonly.safetensors")
    pipe, BANK, set_mode = _build_refonly_pipe(ckpt, single_file=True)

    init = cv2.imread(os.path.join(ROOT, "assets", "control",
                                   "face_classical_now.png"))
    init = cv2.resize(init, (512, 640))
    init_rgb = Image.fromarray(cv2.cvtColor(init, cv2.COLOR_BGR2RGB))
    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))

    prompt = ("rough hand-drawn pencil sketch portrait of a boy, "
              "strict front view facing viewer, perfectly symmetric, "
              "calm expression, short black hair with straight fringe, "
              "visible ears, white hooded jacket, closed chest with "
              "center zipper and drawstrings, monochrome graphite, "
              "sketchy pencil strokes, hatching, off-white paper")
    neg = ("anime, cel shading, clean lineart, digital art, vector, "
           "side view, 3/4 view, turned head, profile, colored, color, "
           "watercolor, 3d render, photorealistic, twin tails, long "
           "hair, hair over ears, two bodies, two necks, split chest, "
           "deformed, asymmetric, signature, watermark, text")

    outdir = os.path.join(OUT, "face12")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
        cv2.imwrite(os.path.join(outdir, f"{tag}_sym.png"),
                    _symmetrize(gimg))

    for strength in (0.45, 0.50, 0.55):
        for seed in (980, 981):
            t0 = time.time()
            img = _refonly_img2img(pipe, BANK, set_mode, init_rgb,
                                   ref_rgb, prompt, neg, strength, seed)
            _save(img, f"cand_s{int(strength*100)}_{seed}")
            print(f"[face12 s{int(strength*100)}_{seed}] "
                  f"{time.time()-t0:.0f}s", flush=True)


def gen_face13():
    """V13: REAL profile->frontal conversion. All previous rounds used
    the classical mirror-composite as the img2img init, so the model
    could only touch up a 'two profiles fused' image — the user spotted
    exactly that ('是侧面转正面，不是两侧面融合'). This round feeds the
    ORIGINAL profile itself as init and pushes strength to
    0.70/0.80/0.90 so the model must re-imagine the head frontally
    instead of inheriting the mirrored structure. SD1.5 base (the
    classmate-result style) + reference-only (style lock from the
    original) + LCM. Output: assets/ref_gen/face13/"""
    ckpt = _find_file(SD15, "v1-5-pruned-emaonly.safetensors")
    pipe, BANK, set_mode = _build_refonly_pipe(ckpt, single_file=True)

    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    init_rgb = ref_rgb  # the profile itself is the init

    prompt = ("rough hand-drawn pencil sketch portrait of a boy, "
              "strict front view facing viewer, perfectly symmetric, "
              "calm expression, short black hair with straight fringe, "
              "visible ears, white hooded jacket, closed chest with "
              "center zipper and drawstrings, monochrome graphite, "
              "sketchy pencil strokes, hatching, off-white paper")
    neg = ("anime, cel shading, clean lineart, digital art, vector, "
           "side view, 3/4 view, turned head, profile, colored, color, "
           "watercolor, 3d render, photorealistic, twin tails, "
           "pigtails, long hair, hair over ears, two bodies, two "
           "necks, split chest, deformed, asymmetric, signature, "
           "watermark, text")

    outdir = os.path.join(OUT, "face13")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
        cv2.imwrite(os.path.join(outdir, f"{tag}_sym.png"),
                    _symmetrize(gimg))

    for strength in (0.70, 0.80, 0.90):
        for seed in (990, 991):
            t0 = time.time()
            img = _refonly_img2img(pipe, BANK, set_mode, init_rgb,
                                   ref_rgb, prompt, neg, strength, seed)
            _save(img, f"cand_s{int(strength*100)}_{seed}")
            print(f"[face13 s{int(strength*100)}_{seed}] "
                  f"{time.time()-t0:.0f}s", flush=True)


def gen_face14():
    """V14: the standard SD pose-rotation stack, per web research
    ('全部是侧脸' — V13 gave the model NO hard frontal constraint, so
    at high strength the pose was luck). Adds ControlNet lineart
    conditioned on a STRICTLY FRONTAL template (lineart derived from
    the chest-fixed classical composite) to the V13 recipe: init =
    original profile, strength 0.85/0.95 (free re-imagination),
    reference-only (style), LCM (speed). Prompt wording follows the
    research finding that 'turn/rotate' phrasing rotates the TORSO —
    use 'front view, looking at viewer, face forward' instead.
    Output: assets/ref_gen/face14/"""
    from diffusers import ControlNetModel
    ckpt = _find_file(SD15, "v1-5-pruned-emaonly.safetensors")
    pipe, BANK, set_mode = _build_refonly_pipe(ckpt, single_file=True)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)

    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    init_rgb = ref_rgb  # the profile itself is the init

    # Frontal structure template: DIY black-on-white lineart of the
    # chest-fixed classical composite (strictly frontal, symmetric,
    # single body). ControlNet lineart expects dark lines on white.
    cur = cv2.imread(os.path.join(ROOT, "assets", "control",
                                  "face_classical_now.png"))
    cur = cv2.resize(cur, (512, 640))
    g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    lines = 255 - np.clip((255 - g.astype(np.int32)) * 2, 0, 255
                          ).astype(np.uint8)
    control_pil = Image.fromarray(lines).convert("RGB")
    cond = torch.from_numpy(
        np.asarray(control_pil).astype(np.float32) / 255.0
        ).permute(2, 0, 1)[None].repeat(2, 1, 1, 1)

    prompt = ("rough hand-drawn pencil sketch portrait of a boy, "
              "front view, looking at viewer, face forward, torso "
              "facing forward, perfectly symmetric, calm expression, "
              "short black hair with straight fringe, visible ears, "
              "white hooded jacket, closed chest with center zipper "
              "and drawstrings, monochrome graphite, sketchy pencil "
              "strokes, hatching, off-white paper")
    neg = ("anime, cel shading, clean lineart, digital art, vector, "
           "side view, 3/4 view, turned head, turned torso, profile, "
           "colored, color, watercolor, 3d render, photorealistic, "
           "twin tails, pigtails, long hair, hair over ears, two "
           "bodies, two necks, split chest, deformed, asymmetric, "
           "signature, watermark, text")

    outdir = os.path.join(OUT, "face14")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
        cv2.imwrite(os.path.join(outdir, f"{tag}_sym.png"),
                    _symmetrize(gimg))

    for strength in (0.85, 0.95):
        for seed in (992, 993):
            t0 = time.time()
            img = _refonly_img2img(
                pipe, BANK, set_mode, init_rgb, ref_rgb, prompt, neg,
                strength, seed, steps=8, controlnet=cn,
                control_img=cond, control_scale=0.8)
            _save(img, f"cand_s{int(strength*100)}_{seed}")
            print(f"[face14 s{int(strength*100)}_{seed}] "
                  f"{time.time()-t0:.0f}s", flush=True)


def gen_face15():
    """V15: same stack as V14 (profile init + high strength + ControlNet
    frontal template + reference-only + LCM) with a CLEANED template,
    aimed at matching the classmate reference result exactly:
    - template is now true high-pass STROKE lineart (local-mean minus
      gray, threshold 8) instead of the crude x2 darkness boost that
      turned hair into a black mass and left the face empty — the mass
      is why V14 grew a hood frame around the head;
    - hood outline strokes above the shoulder line and outside the head
      bbox are erased from the template (classmate result has the hood
      resting BEHIND the neck, no frame around the head);
    - prompt: 'calm neutral expression, medium eyes' (V14 eyes were too
      big/surprised vs the reference), 'hood resting behind neck';
      negatives add 'hood over head, hood framing face, surprised'.
    Output: assets/ref_gen/face15/"""
    from diffusers import ControlNetModel
    ckpt = _find_file(SD15, "v1-5-pruned-emaonly.safetensors")
    pipe, BANK, set_mode = _build_refonly_pipe(ckpt, single_file=True)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)

    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    init_rgb = ref_rgb  # the profile itself is the init

    # --- cleaned frontal template: high-pass strokes + hood-frame
    # erased above the shoulder line outside the head bbox ---
    cur = cv2.imread(os.path.join(ROOT, "assets", "control",
                                  "face_classical_now.png"))
    cur = cv2.resize(cur, (512, 640))
    g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    loc = cv2.GaussianBlur(g, (0, 0), 6).astype(np.float32)
    hp = np.clip(loc - g.astype(np.float32), 0, None)
    hp[hp < 8] = 0
    lines = 255 - np.clip(hp * 2.0, 0, 255).astype(np.uint8)
    y_shoulder, x0, x1 = 345, 85, 425
    lines[:y_shoulder, :x0] = 255
    lines[:y_shoulder, x1:] = 255
    cv2.imwrite(os.path.join(ROOT, "assets", "control",
                             "face_frontal_template.png"), lines)
    control_pil = Image.fromarray(lines).convert("RGB")
    cond = torch.from_numpy(
        np.asarray(control_pil).astype(np.float32) / 255.0
        ).permute(2, 0, 1)[None].repeat(2, 1, 1, 1)

    prompt = ("rough hand-drawn pencil sketch portrait of a boy, "
              "front view, looking at viewer, face and torso facing "
              "forward, perfectly symmetric, calm neutral expression, "
              "medium eyes, short black hair with straight fringe, "
              "visible ears, hood resting behind neck, white hooded "
              "jacket, center zipper and drawstrings, monochrome "
              "graphite pencil, sketchy strokes, off-white paper")
    neg = ("anime, cel shading, clean lineart, digital art, vector, "
           "side view, 3/4 view, turned head, turned torso, profile, "
           "hood over head, hood framing face, surprised, colored, "
           "color, watercolor, 3d render, photorealistic, twin tails, "
           "pigtails, long hair, hair over ears, two bodies, two "
           "necks, split chest, deformed, asymmetric, signature, "
           "watermark, text")

    outdir = os.path.join(OUT, "face15")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
        cv2.imwrite(os.path.join(outdir, f"{tag}_sym.png"),
                    _symmetrize(gimg))

    for strength in (0.85, 0.95):
        for seed in (994, 995):
            t0 = time.time()
            img = _refonly_img2img(
                pipe, BANK, set_mode, init_rgb, ref_rgb, prompt, neg,
                strength, seed, steps=8, controlnet=cn,
                control_img=cond, control_scale=0.8)
            _save(img, f"cand_s{int(strength*100)}_{seed}")
            print(f"[face15 s{int(strength*100)}_{seed}] "
                  f"{time.time()-t0:.0f}s", flush=True)


def gen_face16():
    """V16: DUAL ControlNet, the community-standard 'pose via template +
    identity via original' recipe (user asked whether original-image
    features like masks/edges are extracted — until now the original's
    own linework was NOT a control input):
      ControlNet A: cleaned frontal template lineart, scale 0.8
        (hard frontal-pose lock, same as V15)
      ControlNet B: high-pass stroke lineart of the ORIGINAL profile,
        scale 0.35 (identity hint — the original's own hair-stroke
        rhythm and hoodie design steer the render without dragging the
        pose back to profile)
    Residuals of both ControlNets are summed into the UNet. Same
    ControlNet model instance is called twice (no extra memory);
    per-step cost +~40%. Rest of the V15 stack unchanged: profile init,
    strength 0.85/0.95, reference-only, LCM.
    Output: assets/ref_gen/face16/"""
    from diffusers import ControlNetModel
    ckpt = _find_file(SD15, "v1-5-pruned-emaonly.safetensors")
    pipe, BANK, set_mode = _build_refonly_pipe(ckpt, single_file=True)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)

    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    init_rgb = ref_rgb  # the profile itself is the init

    def _hp_lines(img_bgr, thresh=8, boost=2.0):
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        loc = cv2.GaussianBlur(g, (0, 0), 6).astype(np.float32)
        hp = np.clip(loc - g.astype(np.float32), 0, None)
        hp[hp < thresh] = 0
        return 255 - np.clip(hp * boost, 0, 255).astype(np.uint8)

    # Control A: darkness-boost template (V14 style — its black hair
    # MASS produced correct hair, whereas V15's high-pass strokes gave
    # 'helmet hair' from contour rings) + hood-frame strokes erased
    # above the shoulder line outside the head bbox.
    cur = cv2.imread(os.path.join(ROOT, "assets", "control",
                                  "face_classical_now.png"))
    cur = cv2.resize(cur, (512, 640))
    g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    lines_a = 255 - np.clip((255 - g.astype(np.int32)) * 2, 0, 255
                            ).astype(np.uint8)
    lines_a[:345, :85] = 255
    lines_a[:345, 425:] = 255
    cv2.imwrite(os.path.join(ROOT, "assets", "control",
                             "face_frontal_template.png"), lines_a)
    # Control B: the ORIGINAL profile's own strokes
    lines_b = _hp_lines(ref)
    cv2.imwrite(os.path.join(ROOT, "assets", "control",
                             "face_orig_lines.png"), lines_b)

    def _cond(lines):
        pil = Image.fromarray(lines).convert("RGB")
        return torch.from_numpy(
            np.asarray(pil).astype(np.float32) / 255.0
            ).permute(2, 0, 1)[None].repeat(2, 1, 1, 1)

    controls = [(cn, _cond(lines_a), 0.8), (cn, _cond(lines_b), 0.35)]

    prompt = ("rough hand-drawn pencil sketch portrait of a boy, "
              "front view, looking at viewer, face and torso facing "
              "forward, perfectly symmetric, calm neutral expression, "
              "medium eyes, short black hair with straight fringe, "
              "visible ears, hood resting behind neck, white hooded "
              "jacket, center zipper and drawstrings, monochrome "
              "graphite pencil, sketchy strokes, off-white paper")
    neg = ("anime, cel shading, clean lineart, digital art, vector, "
           "side view, 3/4 view, turned head, turned torso, profile, "
           "hood over head, hood framing face, surprised, colored, "
           "color, watercolor, 3d render, photorealistic, twin tails, "
           "pigtails, long hair, hair over ears, two bodies, two "
           "necks, split chest, deformed, asymmetric, signature, "
           "watermark, text")

    outdir = os.path.join(OUT, "face16")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
        cv2.imwrite(os.path.join(outdir, f"{tag}_sym.png"),
                    _symmetrize(gimg))

    for strength in (0.85, 0.95):
        for seed in (996, 997):
            t0 = time.time()
            img = _refonly_img2img(
                pipe, BANK, set_mode, init_rgb, ref_rgb, prompt, neg,
                strength, seed, steps=8, controls=controls)
            _save(img, f"cand_s{int(strength*100)}_{seed}")
            print(f"[face16 s{int(strength*100)}_{seed}] "
                  f"{time.time()-t0:.0f}s", flush=True)


def gen_face17():
    """V17: the ONE untested combination after V14/V15/V16 triangulation:
    V14 (crude darkness template, hood frame kept) had the best FACES —
    strictly frontal, calm; V15 (high-pass template) gave helmet hair;
    V16 (dual ControlNet, original-strokes control B at 0.35) dragged
    s85 back to profile and reintroduced the hood frame. So: V14's
    exact recipe and seeds (992/993) with the ONLY change being the
    hood-frame strokes erased from the template above the shoulder line
    outside the head bbox. Single ControlNet, no control B.
    Output: assets/ref_gen/face17/"""
    from diffusers import ControlNetModel
    ckpt = _find_file(SD15, "v1-5-pruned-emaonly.safetensors")
    pipe, BANK, set_mode = _build_refonly_pipe(ckpt, single_file=True)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)

    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    init_rgb = ref_rgb  # the profile itself is the init

    cur = cv2.imread(os.path.join(ROOT, "assets", "control",
                                  "face_classical_now.png"))
    cur = cv2.resize(cur, (512, 640))
    g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    lines = 255 - np.clip((255 - g.astype(np.int32)) * 2, 0, 255
                          ).astype(np.uint8)
    lines[:345, :85] = 255   # erase hood-frame strokes beside the head
    lines[:345, 425:] = 255
    control_pil = Image.fromarray(lines).convert("RGB")
    cond = torch.from_numpy(
        np.asarray(control_pil).astype(np.float32) / 255.0
        ).permute(2, 0, 1)[None].repeat(2, 1, 1, 1)

    prompt = ("rough hand-drawn pencil sketch portrait of a boy, "
              "front view, looking at viewer, face and torso facing "
              "forward, perfectly symmetric, calm neutral expression, "
              "medium eyes, short black hair with straight fringe, "
              "visible ears, hood resting behind neck, white hooded "
              "jacket, center zipper and drawstrings, monochrome "
              "graphite pencil, sketchy strokes, off-white paper")
    neg = ("anime, cel shading, clean lineart, digital art, vector, "
           "side view, 3/4 view, turned head, turned torso, profile, "
           "hood over head, hood framing face, surprised, colored, "
           "color, watercolor, 3d render, photorealistic, twin tails, "
           "pigtails, long hair, hair over ears, two bodies, two "
           "necks, split chest, deformed, asymmetric, signature, "
           "watermark, text")

    outdir = os.path.join(OUT, "face17")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
        cv2.imwrite(os.path.join(outdir, f"{tag}_sym.png"),
                    _symmetrize(gimg))

    for strength in (0.85, 0.95):
        for seed in (992, 993):  # same seeds as V14 for a clean A/B
            t0 = time.time()
            img = _refonly_img2img(
                pipe, BANK, set_mode, init_rgb, ref_rgb, prompt, neg,
                strength, seed, steps=8, controlnet=cn,
                control_img=cond, control_scale=0.8)
            _save(img, f"cand_s{int(strength*100)}_{seed}")
            print(f"[face17 s{int(strength*100)}_{seed}] "
                  f"{time.time()-t0:.0f}s", flush=True)


def gen_face18():
    """V18: correct the TWO factual errors the user spotted by re-reading
    the original: (1) the garment is NOT a hooded jacket — it is a
    high-collared zip-up track jacket (rolled stand collar with dark
    inner lining, ring-pull zipper, two dark stripes on the shoulders);
    every 'hood' in V12-V17 came from MY prompt saying 'hooded
    jacket/hood'. (2) the original's STYLE is anime-proportioned (large
    eyes, thick lash line) — SD1.5's realistic faces were the wrong
    direction; AnythingV5 + Danbooru tag prompt (V11's success) is the
    right style base. So V18 = AnythingV5 + tag prompt describing the
    ACTUAL clothes + ControlNet frontal template (its collar/stripes
    come from the mirrored original, so they are correct) + profile
    init at high strength + reference-only + LCM.
    Output: assets/ref_gen/face18/"""
    from diffusers import ControlNetModel
    ckpt = _snapshot_dir(ANIME)
    pipe, BANK, set_mode = _build_refonly_pipe(ckpt)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)

    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    init_rgb = ref_rgb  # the profile itself is the init

    cur = cv2.imread(os.path.join(ROOT, "assets", "control",
                                  "face_classical_now.png"))
    cur = cv2.resize(cur, (512, 640))
    g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    lines = 255 - np.clip((255 - g.astype(np.int32)) * 2, 0, 255
                          ).astype(np.uint8)
    control_pil = Image.fromarray(lines).convert("RGB")
    cond = torch.from_numpy(
        np.asarray(control_pil).astype(np.float32) / 255.0
        ).permute(2, 0, 1)[None].repeat(2, 1, 1, 1)

    prompt = ("(monochrome:1.3), (sketch:1.2), traditional media, "
              "pencil (medium), messy lines, hatching (texture), 1boy, "
              "solo, front view, looking at viewer, symmetrical face, "
              "large dark eyes, thick eyelashes, thick straight "
              "eyebrows, short black hair, spiky hair, fringe, white "
              "track jacket, high collar, rolled collar, ring-pull "
              "zipper, striped shoulders, off-white background")
    neg = ("color, colored, hood, hoodie, hooded jacket, clean "
           "lineart, cel shading, digital painting, 3d, realistic, "
           "photo, side view, 3/4 view, profile, turned head, twin "
           "tails, ponytail, long hair, two bodies, two necks, "
           "deformed, asymmetrical, surprised, watermark, text, "
           "signature")

    outdir = os.path.join(OUT, "face18")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
        cv2.imwrite(os.path.join(outdir, f"{tag}_sym.png"),
                    _symmetrize(gimg))

    for strength in (0.85, 0.95):
        for seed in (998, 999):
            t0 = time.time()
            img = _refonly_img2img(
                pipe, BANK, set_mode, init_rgb, ref_rgb, prompt, neg,
                strength, seed, steps=8, controlnet=cn,
                control_img=cond, control_scale=0.8)
            _save(img, f"cand_s{int(strength*100)}_{seed}")
            print(f"[face18 s{int(strength*100)}_{seed}] "
                  f"{time.time()-t0:.0f}s", flush=True)


def gen_face19():
    """V19: V18 + IP-Adapter — the user's directive '直接把原图特征转化
    成数据发过去'. The original image is encoded by CLIP ViT-H into
    image-embedding tokens and injected into every cross-attention
    (attn2) layer via IP-Adapter (scale 0.5): original features as DATA,
    not as prompt words. This is the 4th original-derived data channel:
    (1) VAE latent init (composition), (2) reference-only K/V at every
    denoise step (per-layer stroke features), (3) ControlNet frontal
    template (target structure), (4) IP-Adapter CLIP-H image embedding
    (semantic identity/style). Weights already local (h94/IP-Adapter,
    approved earlier). attn order: load_ip_adapter first, then
    _attach_refonly so our attn1 processors are the final ones.
    Everything else identical to V18 (AnythingV5, correct track-jacket
    prompt, ControlNet template, profile init, high strength, LCM).
    Output: assets/ref_gen/face19/"""
    from diffusers import ControlNetModel
    ckpt = _snapshot_dir(ANIME)
    pipe, _, _ = _build_refonly_pipe(ckpt, refonly=False)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.5)
    _, BANK, set_mode = _attach_refonly(pipe)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)

    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    init_rgb = ref_rgb  # the profile itself is the init

    cur = cv2.imread(os.path.join(ROOT, "assets", "control",
                                  "face_classical_now.png"))
    cur = cv2.resize(cur, (512, 640))
    g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    lines = 255 - np.clip((255 - g.astype(np.int32)) * 2, 0, 255
                          ).astype(np.uint8)
    control_pil = Image.fromarray(lines).convert("RGB")
    cond = torch.from_numpy(
        np.asarray(control_pil).astype(np.float32) / 255.0
        ).permute(2, 0, 1)[None].repeat(2, 1, 1, 1)

    ip_embeds = pipe.prepare_ip_adapter_image_embeds(
        ref_rgb, None, "cpu", 1, True)

    prompt = ("(monochrome:1.3), (sketch:1.2), traditional media, "
              "pencil (medium), messy lines, hatching (texture), 1boy, "
              "solo, front view, looking at viewer, symmetrical face, "
              "large dark eyes, thick eyelashes, thick straight "
              "eyebrows, short black hair, spiky hair, fringe, white "
              "track jacket, high collar, rolled collar, ring-pull "
              "zipper, striped shoulders, off-white background")
    neg = ("color, colored, hood, hoodie, hooded jacket, clean "
           "lineart, cel shading, digital painting, 3d, realistic, "
           "photo, side view, 3/4 view, profile, turned head, twin "
           "tails, ponytail, long hair, two bodies, two necks, "
           "deformed, asymmetrical, surprised, watermark, text, "
           "signature")

    outdir = os.path.join(OUT, "face19")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
        cv2.imwrite(os.path.join(outdir, f"{tag}_sym.png"),
                    _symmetrize(gimg))

    for strength in (0.85, 0.95):
        for seed in (998, 999):  # same seeds as V18 for a clean A/B
            t0 = time.time()
            img = _refonly_img2img(
                pipe, BANK, set_mode, init_rgb, ref_rgb, prompt, neg,
                strength, seed, steps=8, controlnet=cn,
                control_img=cond, control_scale=0.8,
                ip_embeds=ip_embeds)
            _save(img, f"cand_s{int(strength*100)}_{seed}")
            print(f"[face19 s{int(strength*100)}_{seed}] "
                  f"{time.time()-t0:.0f}s", flush=True)


def gen_face20():
    """V20: fix the four gaps the user spotted vs the original on
    V19/s95_998:
    1) NECK TOO LONG — the ControlNet template's chin-to-collar
       distance locks it; shift everything below y=315 up by 35px in
       the template (physically shorter neck).
    2) CLOTHES PATTERN WRONG — root cause found in the logs: CLIP
       truncated the V18/V19 prompt at 77 tokens (88 given), so
       'ring-pull zipper' and 'shoulder stripes' never reached the
       model. Prompt is now under 77 tokens with garment details
       front-loaded.
    3) NO BLUSH — the original has classic anime diagonal cheek
       hatching; add 'blush, cheek hatching' (and drop less important
       tags to fit the budget).
    4) negative gains 'long neck'.
    IP-Adapter kept at 0.5 (its identity contribution was confirmed by
    the V18/V19 A/B); same seeds 998/999.
    Output: assets/ref_gen/face20/"""
    from diffusers import ControlNetModel
    ckpt = _snapshot_dir(ANIME)
    pipe, _, _ = _build_refonly_pipe(ckpt, refonly=False)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.5)
    _, BANK, set_mode = _attach_refonly(pipe)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)

    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    init_rgb = ref_rgb  # the profile itself is the init

    # --- template with SHORTENED NECK: torso below y=315 shifted up
    # by 35px ---
    cur = cv2.imread(os.path.join(ROOT, "assets", "control",
                                  "face_classical_now.png"))
    cur = cv2.resize(cur, (512, 640))
    g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    lines = 255 - np.clip((255 - g.astype(np.int32)) * 2, 0, 255
                          ).astype(np.uint8)
    cut_y, shift = 315, 35
    short = np.full_like(lines, 255)
    short[:cut_y] = lines[:cut_y]
    short[cut_y:640 - shift] = lines[cut_y + shift:640]
    cv2.imwrite(os.path.join(ROOT, "assets", "control",
                             "face_frontal_template_v2.png"), short)
    control_pil = Image.fromarray(short).convert("RGB")
    cond = torch.from_numpy(
        np.asarray(control_pil).astype(np.float32) / 255.0
        ).permute(2, 0, 1)[None].repeat(2, 1, 1, 1)

    ip_embeds = pipe.prepare_ip_adapter_image_embeds(
        ref_rgb, None, "cpu", 1, True)

    # 75 CLIP tokens (verified with the checkpoint's own tokenizer —
    # V18/V19's 88-token prompt silently lost the garment details)
    prompt = ("(monochrome:1.3), (sketch:1.2), traditional media, "
              "pencil, 1boy, front view, looking at viewer, "
              "symmetrical face, large dark eyes, thick eyelashes, "
              "blush, short black hair, spiky hair, fringe, white "
              "track jacket, rolled stand collar, dark inner lining, "
              "ring-pull zipper, two dark shoulder stripes")
    neg = ("color, colored, hood, hoodie, hooded jacket, long neck, "
           "clean lineart, cel shading, digital painting, 3d, "
           "realistic, photo, side view, 3/4 view, profile, turned "
           "head, twin tails, ponytail, long hair, two bodies, two "
           "necks, deformed, asymmetrical, surprised, watermark, "
           "text, signature")

    outdir = os.path.join(OUT, "face20")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
        cv2.imwrite(os.path.join(outdir, f"{tag}_sym.png"),
                    _symmetrize(gimg))

    for strength in (0.85, 0.95):
        for seed in (998, 999):  # same seeds as V18/V19
            t0 = time.time()
            img = _refonly_img2img(
                pipe, BANK, set_mode, init_rgb, ref_rgb, prompt, neg,
                strength, seed, steps=8, controlnet=cn,
                control_img=cond, control_scale=0.8,
                ip_embeds=ip_embeds)
            _save(img, f"cand_s{int(strength*100)}_{seed}")
            print(f"[face20 s{int(strength*100)}_{seed}] "
                  f"{time.time()-t0:.0f}s", flush=True)


def gen_face21():
    """V21: fix the three gaps vs the original the user spotted on V20:
    1) SHOULDERS TOO NARROW — the template inherited the classical
       pipeline's _narrow_figure squeeze and ControlNet locked it.
       Template v3: everything below the shoulder line is horizontally
       WIDENED 15% around the centre axis.
    2) GAZE WRONG — the original's eye is sharp/upturned (tsurime)
       with a straight brow; V20 drew droopy tareme 'sad' eyes. Prompt
       gains 'tsurime, sharp gaze, straight eyebrows'; negatives gain
       'tareme, droopy eyes, sad, narrow shoulders'.
    3) FACE IDENTITY — the frontal face is model-invented; the lever
       that pulls it toward the original is IP-Adapter scale. Two
       candidates at 0.5 (V20 value) and two at 0.7, same seeds, so
       the identity gain vs artifact cost is directly comparable.
    Output: assets/ref_gen/face21/"""
    from diffusers import ControlNetModel
    ckpt = _snapshot_dir(ANIME)
    pipe, _, _ = _build_refonly_pipe(ckpt, refonly=False)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter_sd15.safetensors")
    _, BANK, set_mode = _attach_refonly(pipe)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)

    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    init_rgb = ref_rgb  # the profile itself is the init

    # --- template v3: short neck (V20) + WIDENED shoulders/torso ---
    cur = cv2.imread(os.path.join(ROOT, "assets", "control",
                                  "face_classical_now.png"))
    cur = cv2.resize(cur, (512, 640))
    g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    lines = 255 - np.clip((255 - g.astype(np.int32)) * 2, 0, 255
                          ).astype(np.uint8)
    cut_y, shift = 315, 35
    short = np.full_like(lines, 255)
    short[:cut_y] = lines[:cut_y]
    short[cut_y:640 - shift] = lines[cut_y + shift:640]
    y_sh = cut_y + 5
    below = short[y_sh:]
    h2, w2 = below.shape
    cx = 256.0
    mx = (cx + (np.arange(w2, dtype=np.float32) - cx) / 1.15)
    my = np.arange(h2, dtype=np.float32)
    map_x = np.tile(mx[None, :], (h2, 1)).astype(np.float32)
    map_y = np.tile(my[:, None], (1, w2)).astype(np.float32)
    short[y_sh:] = cv2.remap(below, map_x, map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=255)
    cv2.imwrite(os.path.join(ROOT, "assets", "control",
                             "face_frontal_template_v3.png"), short)
    control_pil = Image.fromarray(short).convert("RGB")
    cond = torch.from_numpy(
        np.asarray(control_pil).astype(np.float32) / 255.0
        ).permute(2, 0, 1)[None].repeat(2, 1, 1, 1)

    ip_embeds = pipe.prepare_ip_adapter_image_embeds(
        ref_rgb, None, "cpu", 1, True)

    prompt = ("(monochrome:1.3), (sketch:1.2), traditional media, "
              "1boy, front view, looking at viewer, symmetrical, "
              "large dark eyes, tsurime, sharp gaze, thick eyelashes, "
              "straight eyebrows, blush, short black hair, fringe, "
              "white track jacket, rolled stand collar, dark inner "
              "lining, ring-pull zipper, two dark shoulder stripes")
    neg = ("color, colored, hood, hoodie, long neck, narrow "
           "shoulders, tareme, droopy eyes, cel shading, "
           "digital painting, 3d, realistic, photo, side view, "
           "3/4 view, profile, turned head, twin tails, long hair, "
           "two bodies, two necks, deformed, asymmetrical, "
           "surprised, watermark, text, signature")

    outdir = os.path.join(OUT, "face21")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
        cv2.imwrite(os.path.join(outdir, f"{tag}_sym.png"),
                    _symmetrize(gimg))

    for ip_scale in (0.5, 0.7):
        pipe.set_ip_adapter_scale(ip_scale)
        for seed in (998, 999):
            t0 = time.time()
            img = _refonly_img2img(
                pipe, BANK, set_mode, init_rgb, ref_rgb, prompt, neg,
                0.90, seed, steps=8, controlnet=cn,
                control_img=cond, control_scale=0.8,
                ip_embeds=ip_embeds)
            _save(img, f"cand_ip{int(ip_scale*100)}_{seed}")
            print(f"[face21 ip{int(ip_scale*100)}_{seed}] "
                  f"{time.time()-t0:.0f}s", flush=True)


def gen_face22():
    """V22: MEASUREMENT-DRIVEN template + older face + a 5th data
    channel, per the user's three notes on V21 (too childish, lips
    wrong, shoulders still narrow) and the directive '把图像转化成点、
    数据、噪点等信息'.

    1) SHOULDERS BY ANATOMY DATA, not eyeballing: natural male
       shoulder span ~ 2.0 x face width. The template MEASURES its own
       hair-outline span (face_w = 0.75 x hair_span), computes
       target_span = 2 x face_w and widens the torso below the
       shoulder line by exactly target/current (capped 1.30). All
       numbers printed for verification.
    2) CHILDISH FACE/LIPS: AnythingV5 defaults to moe-childish.
       Prompt gains 'teenager, sharp jawline, small closed lips';
       negatives gain 'child, shota, baby face, round face, thick
       lips, pouty'.
    3) 5TH DATA CHANNEL (original as point/edge data): the original
       profile's high-pass stroke point set as a second ControlNet
       input at scale 0.15 (half of V16's failed 0.35 — identity
       strokes without dragging the pose). 2 of 4 candidates have it,
       same seeds, for a clean user-side A/B.
    IP-Adapter fixed at 0.7 (V21 winner). Original noise-sigma is also
    measured and printed (the final composite's base is 100% original
    pixels, so grain is inherited there).
    Output: assets/ref_gen/face22/"""
    from diffusers import ControlNetModel
    ckpt = _snapshot_dir(ANIME)
    pipe, _, _ = _build_refonly_pipe(ckpt, refonly=False)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.7)
    _, BANK, set_mode = _attach_refonly(pipe)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)

    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    init_rgb = ref_rgb  # the profile itself is the init

    # --- original as DATA: noise sigma + stroke point set ---
    g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    loc = cv2.GaussianBlur(g_ref, (0, 0), 6)
    hp_ref = g_ref - loc
    print(f"[face22 data] original noise sigma={hp_ref.std():.2f}, "
          f"mean={g_ref.mean():.1f}, std={g_ref.std():.1f}", flush=True)
    hp_lines = np.clip(loc - g_ref, 0, None)
    hp_lines[hp_lines < 8] = 0
    lines_b = 255 - np.clip(hp_lines * 2.0, 0, 255).astype(np.uint8)

    # --- template v4: short neck (V20) + anatomy-measured shoulders
    cur = cv2.imread(os.path.join(ROOT, "assets", "control",
                                  "face_classical_now.png"))
    cur = cv2.resize(cur, (512, 640))
    g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    lines = 255 - np.clip((255 - g.astype(np.int32)) * 2, 0, 255
                          ).astype(np.uint8)
    cut_y, shift = 315, 35
    short = np.full_like(lines, 255)
    short[:cut_y] = lines[:cut_y]
    short[cut_y:640 - shift] = lines[cut_y + shift:640]
    y_sh = cut_y + 5
    # measure head/hair span above the chin
    band = short[80:cut_y]
    cols = np.where((band < 128).any(axis=0))[0]
    hair_span = float(cols.max() - cols.min()) if len(cols) else 300.0
    face_w = 0.70 * hair_span  # anime hair mass overhangs the temples
    target_span = min(2.0 * face_w, 470.0)
    # measure current torso span just below the shoulder line
    tband = short[y_sh + 40:y_sh + 120]
    cols2 = np.where((tband < 128).any(axis=0))[0]
    cur_span = float(cols2.max() - cols2.min()) if len(cols2) else 300.0
    scale = float(np.clip(target_span / max(cur_span, 1.0), 1.0, 1.45))
    print(f"[face22 data] hair_span={hair_span:.0f}px face_w="
          f"{face_w:.0f}px cur_torso={cur_span:.0f}px target="
          f"{target_span:.0f}px widen=x{scale:.3f}", flush=True)
    below = short[y_sh:]
    h2, w2 = below.shape
    cx = 256.0
    mx = (cx + (np.arange(w2, dtype=np.float32) - cx) / scale)
    my = np.arange(h2, dtype=np.float32)
    map_x = np.tile(mx[None, :], (h2, 1)).astype(np.float32)
    map_y = np.tile(my[:, None], (1, w2)).astype(np.float32)
    stretched = cv2.remap(below, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=255)
    # re-darken strokes washed out by the stretch interpolation
    short[y_sh:] = 255 - np.clip(
        (255 - stretched.astype(np.int32)) * 1.35, 0, 255
        ).astype(np.uint8)
    cv2.imwrite(os.path.join(ROOT, "assets", "control",
                             "face_frontal_template_v4.png"), short)

    def _cond(lin):
        pil = Image.fromarray(lin).convert("RGB")
        return torch.from_numpy(
            np.asarray(pil).astype(np.float32) / 255.0
            ).permute(2, 0, 1)[None].repeat(2, 1, 1, 1)

    cond_a = _cond(short)
    cond_b = _cond(lines_b)

    ip_embeds = pipe.prepare_ip_adapter_image_embeds(
        ref_rgb, None, "cpu", 1, True)

    prompt = ("(monochrome:1.3), (sketch:1.2), traditional media, "
              "1boy, teenager, front view, symmetrical, large dark "
              "eyes, tsurime, sharp gaze, straight eyebrows, blush, "
              "sharp jawline, small closed lips, short black hair, "
              "fringe, white track jacket, stand collar, dark inner "
              "lining, ring-pull zipper, shoulder stripes")
    neg = ("color, colored, hood, hoodie, child, shota, baby face, "
           "round face, thick lips, pouty, long neck, narrow "
           "shoulders, tareme, droopy eyes, realistic, photo, side "
           "view, 3/4 view, profile, twin tails, long hair, two "
           "bodies, two necks, deformed, watermark, text, "
           "signature")

    outdir = os.path.join(OUT, "face22")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
        cv2.imwrite(os.path.join(outdir, f"{tag}_sym.png"),
                    _symmetrize(gimg))

    for cb_scale in (0.0, 0.15):
        ctrls = [(cn, cond_a, 0.8)]
        if cb_scale > 0:
            ctrls.append((cn, cond_b, cb_scale))
        for seed in (998, 999):
            t0 = time.time()
            img = _refonly_img2img(
                pipe, BANK, set_mode, init_rgb, ref_rgb, prompt, neg,
                0.90, seed, steps=8, controls=ctrls,
                ip_embeds=ip_embeds)
            _save(img, f"cand_cb{int(cb_scale*100)}_{seed}")
            print(f"[face22 cb{int(cb_scale*100)}_{seed}] "
                  f"{time.time()-t0:.0f}s", flush=True)


def gen_face23():
    """V23: switch base model to Counterfeit-V3.0 (semi-realistic
    anime, user-approved download) + garment-truth prompt, per the
    user's V22 review: shoulder 'stripe patterns' were wrong (the
    original has ONE dark seam trim, not decorative stripes), the
    garment is a plain HOODED zip jacket (hood down, folded behind
    the neck — the earlier 'track jacket, stand collar' reading was
    wrong), still too childish, and crucially the original is an
    AI-generated SEMI-REALISTIC (2.5D) image, between anime and
    realism — which is AnythingV5's capability ceiling, so the base
    checkpoint changes.

    Prompt corrections: 'young man' replaces 'teenager'; garment =
    'white hooded jacket, hood down, front zipper, ring pull' (stand
    collar / dark inner lining / shoulder stripes removed); 'hood,
    hoodie' REMOVED from the negatives (the garment IS hooded).
    Both prompts verified 72/77 tokens.

    Everything else identical to V22: template v4 (anatomy-measured
    shoulders), ControlNet @0.8, profile init @0.90, reference-only,
    IP-Adapter 0.7, LCM 8 steps, grid = cb(0/0.15) x seeds(998/999).
    Output: assets/ref_gen/face23/"""
    from diffusers import ControlNetModel
    ckpt = _find_file(
        os.path.join(r"D:\huggingface_cache",
                     "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")
    print(f"[face23] base model: {ckpt}", flush=True)
    pipe, _, _ = _build_refonly_pipe(ckpt, single_file=True,
                                     refonly=False)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.7)
    _, BANK, set_mode = _attach_refonly(pipe)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)

    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    init_rgb = ref_rgb  # the profile itself is the init

    # original as DATA: noise sigma + stroke point set (channel 5)
    g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    loc = cv2.GaussianBlur(g_ref, (0, 0), 6)
    hp_lines = np.clip(loc - g_ref, 0, None)
    hp_lines[hp_lines < 8] = 0
    lines_b = 255 - np.clip(hp_lines * 2.0, 0, 255).astype(np.uint8)

    short = cv2.imread(os.path.join(ROOT, "assets", "control",
                                    "face_frontal_template_v4.png"),
                       cv2.IMREAD_GRAYSCALE)
    assert short is not None and short.shape == (640, 512)

    def _cond(lin):
        pil = Image.fromarray(lin).convert("RGB")
        return torch.from_numpy(
            np.asarray(pil).astype(np.float32) / 255.0
            ).permute(2, 0, 1)[None].repeat(2, 1, 1, 1)

    cond_a = _cond(short)
    cond_b = _cond(lines_b)

    ip_embeds = pipe.prepare_ip_adapter_image_embeds(
        ref_rgb, None, "cpu", 1, True)

    prompt = ("(monochrome:1.3), (sketch:1.2), traditional media, "
              "1boy, young man, front view, symmetrical, large dark "
              "eyes, tsurime, sharp gaze, straight eyebrows, blush, "
              "sharp jawline, small closed lips, short black hair, "
              "fringe, white hooded jacket, hood down, front zipper, "
              "ring pull")
    neg = ("color, colored, child, shota, baby face, round face, "
           "thick lips, pouty, long neck, narrow shoulders, tareme, "
           "droopy eyes, realistic, photo, side view, 3/4 view, "
           "profile, twin tails, long hair, two bodies, two necks, "
           "deformed, watermark, text, signature")

    outdir = os.path.join(OUT, "face23")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
        cv2.imwrite(os.path.join(outdir, f"{tag}_sym.png"),
                    _symmetrize(gimg))

    for cb_scale in (0.0, 0.15):
        ctrls = [(cn, cond_a, 0.8)]
        if cb_scale > 0:
            ctrls.append((cn, cond_b, cb_scale))
        for seed in (998, 999):
            t0 = time.time()
            img = _refonly_img2img(
                pipe, BANK, set_mode, init_rgb, ref_rgb, prompt, neg,
                0.90, seed, steps=8, controls=ctrls,
                ip_embeds=ip_embeds)
            _save(img, f"cand_cb{int(cb_scale*100)}_{seed}")
            print(f"[face23 cb{int(cb_scale*100)}_{seed}] "
                  f"{time.time()-t0:.0f}s", flush=True)


def gen_face24():
    """V24: MAXIMIZE original-image information survival, per the
    user's core demand '把原图的信息完整传递' + '模型注意力太差'.

    Mechanism changes vs V23:
    1) DENOISE STRENGTH A/B 0.90/0.75/0.65: at 0.90 only ~10% of the
       original's pixel information survives the noising; 0.65 keeps
       ~3x more of the original's soft pencil texture. Pose drift at
       low strength is countered by ControlNet scale 0.8 -> 1.0
       (hard frontal lock).
    2) SLIMMED PROMPT 75 -> 62 tokens: CLIP attention dilutes over
       prompt length. Everything the ControlNet template already says
       in PIXELS (front view, symmetry, garment shape) is removed
       from the TEXT so the remaining requirements (style, age, eye
       shape, zipped-up) each get more attention weight.
    3) Garment truth: 'zipped up, hood down' added; negatives ban
       'open jacket, undershirt, black shirt'.
    4) De-childification: 'large dark eyes' REMOVED (big eyes read
       childish), 'sharp eyes' + 'young man'; negatives ban
       'big eyes'.
    5) Soft-style: 'soft shading, fine pencil strokes' in; negatives
       ban 'harsh lines, thick outlines'; 'realistic' removed from
       negatives (user WANTS more realism).
    Grid (seed 998, cb0.15 fixed): (s90,ip70), (s75,ip70),
    (s75,ip90), (s65,ip90). Output: assets/ref_gen/face24/"""
    from diffusers import ControlNetModel
    ckpt = _find_file(
        os.path.join(r"D:\huggingface_cache",
                     "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")
    pipe, _, _ = _build_refonly_pipe(ckpt, single_file=True,
                                     refonly=False)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter_sd15.safetensors")
    _, BANK, set_mode = _attach_refonly(pipe)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)

    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    init_rgb = ref_rgb  # the profile itself is the init

    # original stroke point set (channel 5, fixed at 0.15)
    g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    loc = cv2.GaussianBlur(g_ref, (0, 0), 6)
    hp_lines = np.clip(loc - g_ref, 0, None)
    hp_lines[hp_lines < 8] = 0
    lines_b = 255 - np.clip(hp_lines * 2.0, 0, 255).astype(np.uint8)

    short = cv2.imread(os.path.join(ROOT, "assets", "control",
                                    "face_frontal_template_v4.png"),
                       cv2.IMREAD_GRAYSCALE)
    assert short is not None and short.shape == (640, 512)

    def _cond(lin):
        pil = Image.fromarray(lin).convert("RGB")
        return torch.from_numpy(
            np.asarray(pil).astype(np.float32) / 255.0
            ).permute(2, 0, 1)[None].repeat(2, 1, 1, 1)

    ctrls = [(cn, _cond(short), 1.0), (cn, _cond(lines_b), 0.15)]

    prompt = ("(monochrome:1.3), (sketch:1.2), soft shading, fine "
              "pencil strokes, 1boy, young man, sharp eyes, tsurime, "
              "straight eyebrows, blush, sharp jawline, small closed "
              "lips, short black hair, white hooded jacket, zipped "
              "up, hood down")
    neg = ("color, colored, child, shota, baby face, round face, "
           "big eyes, thick lips, pouty, long neck, narrow "
           "shoulders, tareme, droopy eyes, open jacket, "
           "undershirt, black shirt, harsh lines, thick outlines, "
           "photo, side view, 3/4 view, profile, deformed, "
           "watermark, text, signature")

    outdir = os.path.join(OUT, "face24")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
        cv2.imwrite(os.path.join(outdir, f"{tag}_sym.png"),
                    _symmetrize(gimg))

    grid = [(0.90, 0.7), (0.75, 0.7), (0.75, 0.9), (0.65, 0.9)]
    for strength, ip_scale in grid:
        pipe.set_ip_adapter_scale(ip_scale)
        ip_embeds = pipe.prepare_ip_adapter_image_embeds(
            ref_rgb, None, "cpu", 1, True)
        t0 = time.time()
        img = _refonly_img2img(
            pipe, BANK, set_mode, init_rgb, ref_rgb, prompt, neg,
            strength, 998, steps=8, controls=ctrls,
            ip_embeds=ip_embeds)
        tag = f"cand_s{int(strength*100)}_ip{int(ip_scale*10)}"
        _save(img, tag)
        print(f"[face24 {tag}] {time.time()-t0:.0f}s", flush=True)


def _build_template_v5():
    """Template v5: fix the two STRUCTURAL defects of v4 (they
    propagate into every candidate at ControlNet scale 1.0 and no
    prompt can override a pixel condition):
      (a) the chest is a huge white V framed by heavy fold lines ->
          the model reads 'open jacket' and fills it with a black
          undershirt. Fix: whiten the V, draw ONE thin center zipper
          line + ring pull at its top (= zipped up).
      (b) the template's face is mirrored-profile fusion residue
          (two side-view eyes, a white nose blob) which the ControlNet
          locks into every face. Fix: whiten the face interior below
          the brows and draw clean symmetric frontal features
          (almond eyes w/ dark iris + highlight, small nose mark,
          small closed mouth line).
      (c) thin the collar folds' mid-gray hatching (keep the dark
          outer contours) so the neck stays visible.
    Saved to assets/control/face_frontal_template_v5.png."""
    t = cv2.imread(os.path.join(ROOT, "assets", "control",
                                "face_frontal_template_v4.png"),
                   cv2.IMREAD_GRAYSCALE).copy()

    def _whiten(poly, feather):
        m = np.zeros_like(t)
        cv2.fillPoly(m, [np.asarray(poly, np.int32)], 255)
        m = cv2.GaussianBlur(m, (0, 0), feather).astype(np.float32)
        a = m / 255.0
        return (t.astype(np.float32) * (1 - a) + 255 * a
                ).astype(np.uint8)

    # (a) close the chest V (feathered; stops at y=560 so the lower
    # jacket folds survive)
    t = _whiten([(170, 322), (342, 322), (268, 560), (244, 560)], 6)
    cv2.line(t, (256, 396), (256, 640), 80, 2, cv2.LINE_AA)
    cv2.circle(t, (256, 398), 6, 60, 2, cv2.LINE_AA)
    # (b) clean face interior with a feathered mask, then draw
    # symmetric frontal features — narrow tsurime almonds (outer
    # corner lifted), iris partly covered by the upper lash
    t = _whiten([(188, 196), (324, 196), (327, 300),
                 (298, 332), (214, 332), (185, 300)], 4)
    for ex, tilt in ((209, -10), (303, 10)):
        cv2.ellipse(t, (ex, 209), (19, 7), tilt, 180, 360, 40, 3,
                    cv2.LINE_AA)                     # upper lash
        cv2.ellipse(t, (ex, 210), (19, 7), tilt, 0, 180, 120, 1,
                    cv2.LINE_AA)                     # lower lid
        cv2.circle(t, (ex, 212), 8, 60, -1, cv2.LINE_AA)   # iris
        cv2.ellipse(t, (ex, 209), (19, 7), tilt, 180, 360, 40, 3,
                    cv2.LINE_AA)                     # lash over iris
        cv2.circle(t, (ex - 3, 209), 2, 255, -1, cv2.LINE_AA)
    cv2.line(t, (256, 245), (253, 262), 90, 2, cv2.LINE_AA)  # nose
    cv2.line(t, (253, 262), (260, 264), 90, 2, cv2.LINE_AA)
    pts = np.array([(240, 300), (256, 303), (272, 300)], np.int32)
    cv2.polylines(t, [pts], False, 70, 2, cv2.LINE_AA)       # mouth
    cv2.line(t, (247, 308), (265, 308), 140, 1, cv2.LINE_AA)  # lip
    out = os.path.join(ROOT, "assets", "control",
                       "face_frontal_template_v5.png")
    cv2.imwrite(out, t)
    return t


def gen_face25():
    """V25: template v5 (closed zipper V + clean drawn frontal face
    in the template) + strength sweep 0.90/0.85/0.80/0.75 with
    IP-Adapter fixed at 0.9 (V24 showed lower strength = more
    original texture survives; s65 destroyed the hair, so the sweet
    spot is searched between 0.75 and 0.90). Seed 998, cb 0.15,
    ControlNet scale 1.0. Output: assets/ref_gen/face25/"""
    from diffusers import ControlNetModel
    short = _build_template_v5()
    ckpt = _find_file(
        os.path.join(r"D:\huggingface_cache",
                     "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")
    pipe, _, _ = _build_refonly_pipe(ckpt, single_file=True,
                                     refonly=False)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.9)
    _, BANK, set_mode = _attach_refonly(pipe)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)

    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    init_rgb = ref_rgb

    g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    loc = cv2.GaussianBlur(g_ref, (0, 0), 6)
    hp_lines = np.clip(loc - g_ref, 0, None)
    hp_lines[hp_lines < 8] = 0
    lines_b = 255 - np.clip(hp_lines * 2.0, 0, 255).astype(np.uint8)

    def _cond(lin):
        pil = Image.fromarray(lin).convert("RGB")
        return torch.from_numpy(
            np.asarray(pil).astype(np.float32) / 255.0
            ).permute(2, 0, 1)[None].repeat(2, 1, 1, 1)

    ctrls = [(cn, _cond(short), 1.0), (cn, _cond(lines_b), 0.15)]

    ip_embeds = pipe.prepare_ip_adapter_image_embeds(
        ref_rgb, None, "cpu", 1, True)

    prompt = ("(monochrome:1.3), (sketch:1.2), soft shading, fine "
              "pencil strokes, 1boy, young man, sharp eyes, tsurime, "
              "straight eyebrows, blush, sharp jawline, small closed "
              "lips, short black hair, white hooded jacket, zipped "
              "up, hood down")
    neg = ("color, colored, child, shota, baby face, round face, "
           "big eyes, thick lips, pouty, long neck, narrow "
           "shoulders, tareme, droopy eyes, open jacket, "
           "undershirt, black shirt, harsh lines, thick outlines, "
           "photo, side view, 3/4 view, profile, deformed, "
           "watermark, text, signature")

    outdir = os.path.join(OUT, "face25")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
        cv2.imwrite(os.path.join(outdir, f"{tag}_sym.png"),
                    _symmetrize(gimg))

    for strength in (0.90, 0.85, 0.80, 0.75):
        t0 = time.time()
        img = _refonly_img2img(
            pipe, BANK, set_mode, init_rgb, ref_rgb, prompt, neg,
            strength, 998, steps=8, controls=ctrls,
            ip_embeds=ip_embeds)
        tag = f"cand_s{int(strength*100)}"
        _save(img, tag)
        print(f"[face25 {tag}] {time.time()-t0:.0f}s", flush=True)


def _build_template_v6():
    """Template v6 — rebuilt from the CURRENT classical composite
    (assets/ref_gen/face25_final/composite_classical.png) with every
    fix driven by MEASURED data, per the user's audit demand:

    1) FACE-TOO-SMALL: original face/head ratio = 239/290 = 0.82
       (short hair hugs the skull); the mirrored composite's ratio is
       ~0.64 (mirroring doubled the hair mass). Fix: squash the
       above-hairline band so the template's ratio matches 0.82.
    2) SHOULDER STRIPE PATTERNS: the mirrored double seam lines are
       lightened 55% — one faint seam instead of busy stripes.
    3) OPEN-CHEST BLACK INNER: neck-side shadows + under-chin shadow
       whitened (thin neck contour redrawn), collar folds lightened
       60%, center zipper redrawn — the chest reads 'zipped white
       jacket', not 'open jacket over black shirt'.
    4) BROWS TOO STRAIGHT: mirrored horizontal brows erased; redrawn
       at the original's MEASURED slope (inner y=172 vs outer y=160
       -> outer end rises ~10 deg), thickness 3.
    5) Neck shortened 35px and shoulders widened to the measured
       2x-face-width anatomy target (as v4/v5).
    All measurements printed. Saved as face_frontal_template_v6.png.
    """
    comp_path = os.path.join(ROOT, "assets", "ref_gen", "face25_final",
                             "composite_classical.png")
    if os.path.exists(comp_path):
        src = cv2.imread(comp_path, cv2.IMREAD_GRAYSCALE)
    else:
        # regenerate the classical composite from the pipeline itself
        # (no dependency on saved test artifacts)
        sys.path.insert(0, os.path.join(ROOT, "src"))
        import json
        from core.anime_face import AnimeFaceFrontalizer
        img = cv2.imread(os.path.join(ROOT, "4", "face.png"))
        mk = json.load(open(os.path.join(ROOT, "manual_keypoints.json"),
                            encoding="utf-8"))
        entry = mk["13ef60de932bb979aaa107ad210a327a"]
        kps = {k: tuple(v) for k, v in entry["keypoints"].items()}
        r = AnimeFaceFrontalizer().convert(img, keypoints=kps)
        src = (cv2.cvtColor(r.image, cv2.COLOR_BGR2GRAY)
               if r.image.ndim == 3 else r.image)
    g = cv2.resize(src, (512, 640))
    lines = 255 - np.clip((255 - g.astype(np.int32)) * 2, 0, 255
                          ).astype(np.uint8)

    # --- shorten neck ---
    cut_y, shift = 315, 35
    t = np.full_like(lines, 255)
    t[:cut_y] = lines[:cut_y]
    t[cut_y:640 - shift] = lines[cut_y + shift:640]

    # --- measured head geometry ---
    dark = t < 128
    rows = dark[:, 150:370].sum(1)
    head_top = int(np.where(rows > 20)[0].min())
    # chin: last dark row in the head band only (below it the chest
    # zipper/folds would fake a chin at the bottom edge)
    colband = dark[:, 230:290].sum(1)
    lo, hi = head_top + 100, min(head_top + 360, 620)
    chin_y = lo + int(np.where(colband[lo:hi] > 3)[0].max()) + 6
    head_h = chin_y - head_top
    # hairline = bottom edge of the fringe mass on the forehead
    band = dark[head_top:chin_y, 170:340]
    rowcount = band.sum(1)
    hairline_y = head_top + int(np.where(rowcount > 80)[0].max()) \
        if (rowcount > 80).any() else head_top + int(head_h * 0.34)
    face_h = chin_y - hairline_y
    ratio = face_h / head_h
    # target 0.62: original profile measures 0.82 but that is the
    # strict skin-hair boundary of a buzz-short crown; the classmate
    # target look is ~0.6, and 0.82 would crush the hair silhouette
    # the user demands to keep. 0.62 already grows the face a lot
    # from the mirrored composite's 0.42.
    print(f"[v6] head_top={head_top} chin={chin_y} hairline="
          f"{hairline_y} face/head={ratio:.2f} (target 0.62)",
          flush=True)
    target_hair_h = int(face_h / 0.62 - face_h)
    new_top = chin_y - (face_h + target_hair_h)
    f = (hairline_y - new_top) / max(1, hairline_y - head_top)
    ys = np.arange(640, dtype=np.float32)
    map_y = ys.copy()
    m = ys < hairline_y
    map_y[m] = head_top + (ys[m] - new_top) / max(f, 1e-6)
    map_y = np.clip(map_y, 0, 639).astype(np.float32)
    map_x = np.tile(np.arange(512, dtype=np.float32), (640, 1))
    map_y2 = np.tile(map_y[:, None], (1, 512)).astype(np.float32)
    t = cv2.remap(t, map_x, map_y2,
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                  borderValue=255)

    def _soft_keep(img, y0, y1, x0, x1, keep, feather):
        """Multiply stroke darkness by `keep` inside a feathered
        rect — no hard box edges (v6 draft 1 defect)."""
        mask = np.zeros(img.shape, np.float32)
        mask[y0:y1, x0:x1] = 1.0
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
        factor = 1.0 - (1.0 - keep) * mask
        return np.clip(255 - (255 - img.astype(np.float32)) * factor,
                       0, 255).astype(np.uint8)

    # --- lighten shoulder stripe patterns (doubled mirrored seams
    # live on the upper arms, y~380-530) ---
    for x0, x1 in ((40, 215), (297, 472)):
        t = _soft_keep(t, 380, 530, x0, x1, 0.45, 8)

    # --- neck: whiten shadows, redraw thin contours, center zipper
    cx = 256
    t = _soft_keep(t, chin_y, chin_y + 110, cx - 70, cx + 70,
                   0.25, 8)
    cv2.line(t, (cx - 24, chin_y + 5), (cx - 25, chin_y + 55),
             120, 2, cv2.LINE_AA)
    cv2.line(t, (cx + 24, chin_y + 5), (cx + 25, chin_y + 55),
             120, 2, cv2.LINE_AA)
    # collar fold band: lighten 60%
    t = _soft_keep(t, chin_y + 60, chin_y + 150,
                   cx - 135, cx + 135, 0.4, 10)
    # reinforce the existing center zipper (pipeline already drew
    # the ring pull — do NOT add a second circle)
    cv2.line(t, (cx, chin_y + 55), (cx, 640), 80, 2, cv2.LINE_AA)

    # --- brows + eyes: the composite's brow/eye band is vertically
    # compressed (brows 187-200 nearly touching eyes 195-220) —
    # erase the whole band and redraw both at proper spacing.
    # Brows at the original's measured slope (outer end ~10 deg
    # up); eyes as narrow tsurime almonds (v5 eye code, which
    # passed inspection — NOT the round 'surprised' ellipses).
    t = _soft_keep(t, 180, 228, 178, 338, 0.08, 4)
    for ex, tilt in ((209, -10), (303, 10)):
        cv2.ellipse(t, (ex, 214), (19, 7), tilt, 180, 360, 40, 3,
                    cv2.LINE_AA)
        cv2.ellipse(t, (ex, 218), (16, 6), tilt, 20, 160, 120, 1,
                    cv2.LINE_AA)
        cv2.circle(t, (ex, 216), 7, 60, -1, cv2.LINE_AA)
        cv2.ellipse(t, (ex, 214), (19, 7), tilt, 180, 360, 40, 3,
                    cv2.LINE_AA)
        cv2.circle(t, (ex - 3, 213), 2, 255, -1, cv2.LINE_AA)
    for x0, x1, sign in ((191, 247, -1), (265, 321, 1)):
        inner_y, outer_y = 197, 187
        if sign < 0:  # image-left brow: outer end at x0 (temple side)
            cv2.line(t, (x0, outer_y), (x1, inner_y),
                     55, 3, cv2.LINE_AA)
        else:         # image-right brow: outer end at x1
            cv2.line(t, (x0, inner_y), (x1, outer_y),
                     55, 3, cv2.LINE_AA)
    print("[v6] brow+eye band redrawn (brows sloped ~10deg, "
          "tsurime almond eyes)", flush=True)

    out = os.path.join(ROOT, "assets", "control",
                       "face_frontal_template_v6.png")
    cv2.imwrite(out, t)
    return t


def _row_tone_match(gimg, orig, bands=16):
    """Row-band tonal matching: align each horizontal band's mean/std
    to the ORIGINAL's same band (global matching imprints the
    original's overall tonal curve without copying its left/right
    profile layout). Style data channel, post-gen."""
    h = gimg.shape[0]
    out = gimg.astype(np.float32).copy()
    bh = h // bands
    for i in range(bands):
        y0, y1 = i * bh, (i + 1) * bh if i < bands - 1 else h
        c = out[y0:y1].astype(np.float32)
        o = cv2.resize(orig, (gimg.shape[1], h),
                       interpolation=cv2.INTER_AREA
                       )[y0:y1].astype(np.float32)
        out[y0:y1] = (c - c.mean()) / (c.std() + 1e-6) \
            * o.std() + o.mean()
    # smooth band boundaries
    out = cv2.GaussianBlur(out, (0, 0), 1.2)
    return np.clip(out, 0, 255).astype(np.uint8)


def gen_face26():
    """V26: template v6 (measured face-ratio fix, thinned shoulder
    seams, closed white chest, sloped brows) + prompt fix
    ('straight eyebrows' -> 'thick eyebrows'; the original's brow is
    arched ~10 deg, not horizontal) + row-band tonal matching in
    post. Grid: strength 0.90/0.85/0.80/0.75, ip 0.9, seed 998,
    cb 0.15, cn 1.0. Output: assets/ref_gen/face26/"""
    from diffusers import ControlNetModel
    short = _build_template_v6()
    ckpt = _find_file(
        os.path.join(r"D:\huggingface_cache",
                     "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")
    pipe, _, _ = _build_refonly_pipe(ckpt, single_file=True,
                                     refonly=False)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.9)
    _, BANK, set_mode = _attach_refonly(pipe)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)

    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    init_rgb = ref_rgb

    g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    loc = cv2.GaussianBlur(g_ref, (0, 0), 6)
    hp_lines = np.clip(loc - g_ref, 0, None)
    hp_lines[hp_lines < 8] = 0
    lines_b = 255 - np.clip(hp_lines * 2.0, 0, 255).astype(np.uint8)

    def _cond(lin):
        pil = Image.fromarray(lin).convert("RGB")
        return torch.from_numpy(
            np.asarray(pil).astype(np.float32) / 255.0
            ).permute(2, 0, 1)[None].repeat(2, 1, 1, 1)

    ctrls = [(cn, _cond(short), 1.0), (cn, _cond(lines_b), 0.15)]

    ip_embeds = pipe.prepare_ip_adapter_image_embeds(
        ref_rgb, None, "cpu", 1, True)

    prompt = ("(monochrome:1.3), (sketch:1.2), soft shading, fine "
              "pencil strokes, 1boy, young man, sharp eyes, tsurime, "
              "thick eyebrows, blush, sharp jawline, small closed "
              "lips, short black hair, white hooded jacket, zipped "
              "up, hood down")
    neg = ("color, colored, child, shota, baby face, round face, "
           "big eyes, thick lips, pouty, long neck, narrow "
           "shoulders, tareme, droopy eyes, open jacket, "
           "undershirt, black shirt, harsh lines, thick outlines, "
           "photo, side view, 3/4 view, profile, deformed, "
           "watermark, text, signature")

    outdir = os.path.join(OUT, "face26")
    os.makedirs(outdir, exist_ok=True)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        gimg = _row_tone_match(gimg, orig)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)
        cv2.imwrite(os.path.join(outdir, f"{tag}_sym.png"),
                    _symmetrize(gimg))

    for strength in (0.90, 0.85, 0.80, 0.75):
        t0 = time.time()
        img = _refonly_img2img(
            pipe, BANK, set_mode, init_rgb, ref_rgb, prompt, neg,
            strength, 998, steps=8, controls=ctrls,
            ip_embeds=ip_embeds)
        tag = f"cand_s{int(strength*100)}"
        _save(img, tag)
        print(f"[face26 {tag}] {time.time()-t0:.0f}s", flush=True)


def gen_face27():
    """V27 — FACE-ONLY donor round. Strategy shift per user audit:
    everything that EXISTS in the original (clothes, shoulders,
    background, hair, ears, eyes, brows) never goes through the
    model — the final image keeps 100% original pixels for those
    (classical composite). The donor supplies ONLY the face
    interior, so candidates are judged on the FACE alone.

    Anti-childish measures:
    - face-crop template (head fills the 512x640 frame): ControlNet
      lineart now controls eye SIZE, the #1 age-perception driver,
      instead of eyes being a tiny region the anime prior overrides
    - maturity-weighted prompts ((mature face:1.3), narrow eyes)
    Two bases x two strengths: Counterfeit (anime-2.5D) and
    SD1.5 (realistic) — the original sits between the two styles.
    Output: assets/ref_gen/face27/"""
    from diffusers import ControlNetModel
    tpl = cv2.imread(os.path.join(ROOT, "assets", "control",
                                  "face_frontal_template_v6.png"),
                     cv2.IMREAD_GRAYSCALE)
    assert tpl is not None, "build template v6 first"
    face_tpl = cv2.resize(tpl[0:360, 112:400], (512, 640))
    cv2.imwrite(os.path.join(ROOT, "assets", "control",
                             "face_frontal_template_v6_face.png"),
                face_tpl)

    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    init_rgb = ref_rgb
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)

    def _cond(lin):
        pil = Image.fromarray(lin).convert("RGB")
        return torch.from_numpy(
            np.asarray(pil).astype(np.float32) / 255.0
            ).permute(2, 0, 1)[None].repeat(2, 1, 1, 1)

    ctrls = [(cn, _cond(face_tpl), 1.0)]
    outdir = os.path.join(OUT, "face27")
    os.makedirs(outdir, exist_ok=True)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)

    PROMPTS = {
        "cf": ("(monochrome:1.3), (sketch:1.2), soft shading, fine "
               "pencil strokes, 1boy, (mature face:1.3), young man, "
               "sharp narrow eyes, tsurime, thick eyebrows, blush, "
               "sharp jawline, small closed lips, short black hair, "
               "looking at viewer",
               "color, colored, child, shota, baby face, round "
               "face, (big eyes:1.2), thick lips, pouty, tareme, "
               "droopy eyes, harsh lines, photo, side view, 3/4 "
               "view, profile, deformed, watermark, text, "
               "signature"),
        "sd": ("pencil sketch portrait of a handsome young man, "
               "monochrome, (mature face:1.2), sharp narrow eyes, "
               "thick eyebrows, short black hair, small closed "
               "lips, sharp jawline, soft shading, fine pencil "
               "strokes, front view, symmetrical, white "
               "background",
               "color, photo, painting, child, kid, baby, round "
               "face, big eyes, cute, cartoon, anime, deformed, "
               "ugly, watermark, text, signature, blurry"),
    }
    BASES = {
        "cf": (_find_file(os.path.join(
            r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
            "_fp16.safetensors"), True),
        "sd": ("stable-diffusion-v1-5/stable-diffusion-v1-5", False),
    }

    for base_key, (ckpt, single) in BASES.items():
        pipe, _, _ = _build_refonly_pipe(ckpt, single_file=single,
                                         refonly=False)
        pipe.load_ip_adapter(
            "h94/IP-Adapter", subfolder="models",
            weight_name="ip-adapter_sd15.safetensors")
        pipe.set_ip_adapter_scale(0.9)
        _, BANK, set_mode = _attach_refonly(pipe)
        ip_embeds = pipe.prepare_ip_adapter_image_embeds(
            ref_rgb, None, "cpu", 1, True)
        prompt, neg = PROMPTS[base_key]
        for strength in (0.85, 0.75):
            t0 = time.time()
            img = _refonly_img2img(
                pipe, BANK, set_mode, init_rgb, ref_rgb, prompt,
                neg, strength, 998, steps=8, controls=ctrls,
                ip_embeds=ip_embeds)
            tag = f"cand_{base_key}_s{int(strength*100)}"
            _save(img, tag)
            print(f"[face27 {tag}] {time.time()-t0:.0f}s",
                  flush=True)
        del pipe


def _build_face_template_v7():
    """Face-crop donor template v7 — semi-realistic brow/eye redraw.
    User audit: the original is a SEMI-REALISTIC pencil sketch (thin
    lash lines, small filled iris, no anime ring/glint) but v6's
    brow/eye band had heavy black brow bars + anime iris rings with
    glints, and donors inherited both. v7 erases that band (soft
    feather, keeps surrounding shading) and redraws in the original's
    style. Coordinates measured on the 512x640 face crop."""
    tpl = cv2.imread(os.path.join(ROOT, "assets", "control",
                                  "face_frontal_template_v6_face.png"),
                     cv2.IMREAD_GRAYSCALE)
    assert tpl is not None, "run face27 once to save the v6 face crop"
    out = tpl.copy()
    m = np.zeros(out.shape, np.float32)
    m[318:415, 100:412] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), 8)
    out = (out.astype(np.float32) * (1 - m) + 255.0 * m
           ).astype(np.uint8)
    for cx, mirror in ((172, 1), (340, -1)):
        # iris: small dark filled mass tucked under the upper lid —
        # NO ring outline, NO glint (semi-realistic sketch style)
        cv2.circle(out, (cx, 386), 6, 95, -1, cv2.LINE_AA)
        cv2.circle(out, (cx, 386), 3, 40, -1, cv2.LINE_AA)
        # almond lash lines: upper medium, lower faint; tsurime tilt
        cv2.ellipse(out, (cx, 393), (38, 14), 8 * mirror, 192, 348,
                    70, 2, cv2.LINE_AA)
        cv2.ellipse(out, (cx, 393), (38, 14), 8 * mirror, 25, 155,
                    120, 1, cv2.LINE_AA)
        # faint double-lid crease above
        cv2.ellipse(out, (cx, 393), (38, 20), 8 * mirror, 215, 325,
                    155, 1, cv2.LINE_AA)
        # brow: tapered gray shape, thick at inner, thin at outer
        pts = np.array([(cx + 36 * mirror, 356),
                        (cx + 12 * mirror, 341),
                        (cx - 26 * mirror, 336),
                        (cx - 36 * mirror, 343),
                        (cx + 2 * mirror, 352)], np.int32)
        cv2.fillPoly(out, [pts], 125, cv2.LINE_AA)
    dst = os.path.join(ROOT, "assets", "control",
                       "face_frontal_template_v7_face.png")
    cv2.imwrite(dst, out)
    return out


def gen_face28():
    """V28 — STYLE-MATCHED donor round. User: 'the eyes are even more
    wrong, you never look at the original'. cf_s85's eyes are anime
    (ring iris + glint + angular frames); the original is a
    semi-realistic pencil sketch. Two levers, zero new downloads:
    - template v7: semi-realistic brow/eye band (thin lash lines,
      small filled iris, tapered gray brows)
    - bases: Anything V5 (cached diffusers snapshot) + SD1.5 via the
      runwayml single-file (cached; the renamed
      stable-diffusion-v1-5/stable-diffusion-v1-5 snapshot has only
      configs). SD1.5 + graphite-sketch prompt is the closest match
      to the original's 2.5D semi-realistic style.
    Output: assets/ref_gen/face28/"""
    from diffusers import ControlNetModel
    face_tpl = _build_face_template_v7()

    ref = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    ref = cv2.resize(ref, (512, 640))
    ref_rgb = Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    init_rgb = ref_rgb
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_GRAYSCALE)

    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)

    def _cond(lin):
        pil = Image.fromarray(lin).convert("RGB")
        return torch.from_numpy(
            np.asarray(pil).astype(np.float32) / 255.0
            ).permute(2, 0, 1)[None].repeat(2, 1, 1, 1)

    ctrls = [(cn, _cond(face_tpl), 1.0)]
    outdir = os.path.join(OUT, "face28")
    os.makedirs(outdir, exist_ok=True)

    def _save(img, tag):
        img.save(os.path.join(outdir, f"{tag}_raw.png"))
        gimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY
                            ).astype(np.float32)
        gimg = ((gimg - gimg.mean()) / (gimg.std() + 1e-6)
                * orig.std() + orig.mean())
        gimg = np.clip(gimg, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, f"{tag}.png"), gimg)

    PROMPTS = {
        "any": ("(monochrome:1.3), (sketch:1.2), semi-realistic, "
                "soft shading, fine pencil strokes, 1boy, (mature "
                "face:1.3), young man, sharp narrow eyes, small "
                "pupils, tsurime, thick eyebrows, sharp jawline, "
                "small closed lips, short black hair, looking at "
                "viewer",
                "color, colored, child, shota, baby face, round "
                "face, (big eyes:1.3), (glowing eyes:1.2), thick "
                "lips, pouty, tareme, droopy eyes, harsh lines, "
                "photo, side view, 3/4 view, profile, deformed, "
                "watermark, text, signature"),
        "sd": ("graphite pencil sketch portrait of a handsome young "
               "man, semi-realistic, monochrome, (mature face:1.2), "
               "sharp narrow eyes, small dark pupils, tapered "
               "eyebrows, short black hair, small closed lips, "
               "sharp jawline, soft shading, fine pencil strokes, "
               "front view, symmetrical, white background",
               "color, photo, painting, child, kid, baby, round "
               "face, big eyes, cute, cartoon, anime, deformed, "
               "ugly, watermark, text, signature, blurry"),
    }
    BASES = {
        "any": ("stablediffusionapi/anything-v5", False),
        "sd": (_find_file(os.path.join(
            r"D:\huggingface_cache",
            "models--runwayml--stable-diffusion-v1-5"),
            "emaonly.safetensors"), True),
    }

    for base_key, (ckpt, single) in BASES.items():
        pipe, _, _ = _build_refonly_pipe(ckpt, single_file=single,
                                         refonly=False)
        pipe.load_ip_adapter(
            "h94/IP-Adapter", subfolder="models",
            weight_name="ip-adapter_sd15.safetensors")
        pipe.set_ip_adapter_scale(0.9)
        _, BANK, set_mode = _attach_refonly(pipe)
        ip_embeds = pipe.prepare_ip_adapter_image_embeds(
            ref_rgb, None, "cpu", 1, True)
        prompt, neg = PROMPTS[base_key]
        for strength in (0.85, 0.75):
            t0 = time.time()
            img = _refonly_img2img(
                pipe, BANK, set_mode, init_rgb, ref_rgb, prompt,
                neg, strength, 998, steps=8, controls=ctrls,
                ip_embeds=ip_embeds)
            tag = f"cand_{base_key}_s{int(strength*100)}"
            _save(img, tag)
            print(f"[face28 {tag}] {time.time()-t0:.0f}s",
                  flush=True)
        del pipe


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "cat"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    if which in ("cat", "animal"):
        # 可选 argv[3]=输入图 argv[4]=输出目录 argv[5]=物种 (GUI 用;
        # species 缺省=None -> 物种中性提示词, 物种身份由 IP-Adapter 携带)
        gen_animal(n, src=(sys.argv[3] if len(sys.argv) > 3 else None),
                   outdir=(sys.argv[4] if len(sys.argv) > 4 else None),
                   species=(sys.argv[5] if len(sys.argv) > 5 else None))
    elif which == "face2":
        gen_face2(n)
    elif which == "face3":
        gen_face3(n)
    elif which == "face4":
        gen_face4()
    elif which == "face5":
        gen_face5()
    elif which == "face6":
        gen_face6()
    elif which == "face7":
        gen_face7()
    elif which == "face8":
        gen_face8()
    elif which == "face9":
        gen_face9()
    elif which == "face10":
        gen_face10()
    elif which == "face11":
        gen_face11()
    elif which == "face12":
        gen_face12()
    elif which == "face13":
        gen_face13()
    elif which == "face14":
        gen_face14()
    elif which == "face15":
        gen_face15()
    elif which == "face16":
        gen_face16()
    elif which == "face17":
        gen_face17()
    elif which == "face18":
        gen_face18()
    elif which == "face19":
        gen_face19()
    elif which == "face20":
        gen_face20()
    elif which == "face21":
        gen_face21()
    elif which == "face22":
        gen_face22()
    elif which == "face23":
        gen_face23()
    elif which == "face24":
        gen_face24()
    elif which == "face25":
        gen_face25()
    elif which == "face26":
        gen_face26()
    elif which == "face27":
        gen_face27()
    elif which == "face28":
        gen_face28()
    else:
        gen_face(n)
