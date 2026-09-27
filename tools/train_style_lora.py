"""Train a STYLE LoRA on the single original 4/face.png (user:
你究竟有没有仔细分析原图，提取原图所有特征 -> now we literally
fine-tune on it).

Uses ONLY already-approved packages: torch + diffusers + peft.
No kohya download needed.

Method:
- Counterfeit-V3.0 (cached) SD1.5 fp32; VAE + text encoder frozen,
  UNet gets a peft LoRA (rank 8) on all attention projections.
- Dataset: 14 augmentations of the original (mirror, shifts,
  scales, gamma) resized to 448x448; caption is a fixed trigger
  token string so the STYLE binds to it.
- Latents + text embeds precomputed once -> the loop only runs
  the UNet (with gradient checkpointing) so it fits in RAM.
- ~250 AdamW steps, lr 1e-4, then save the peft adapter to
  assets/lora/s2fstyle/.

Inference: apply PeftModel to pipe.unet and put the trigger word
in the prompt (see gen_lora_style.py).
"""
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from trial_inpaint_eyes import _find_file  # noqa: E402

SAVE_DIR = os.path.join(ROOT, "assets", "lora", "s2fstyle")
TRIGGER = "s2fstyle"
CAPTION = (f"{TRIGGER}, monochrome pencil sketch, soft shading, "
           "delicate thin lines, fine hatching")
RES = 448
STEPS = 250
LR = 1e-4
RANK = 8


def _augment(orig):
    """14 style-preserving variants of the original"""
    outs = []
    variants = [orig, orig[:, ::-1]]
    H, W = orig.shape[:2]
    for dx, dy in ((-12, 0), (12, 0), (0, -12), (0, 12),
                   (-8, -8), (8, 8)):
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        variants.append(cv2.warpAffine(orig, m, (W, H),
                                       borderValue=(255, 255,
                                                    255)))
    for s in (0.94, 1.06):
        m = np.float32([[s, 0, W * (1 - s) / 2],
                        [0, s, H * (1 - s) / 2]])
        variants.append(cv2.warpAffine(orig, m, (W, H),
                                       borderValue=(255, 255,
                                                    255)))
    for g in (0.9, 1.1):
        lut = np.clip((np.arange(256) / 255.0) ** g * 255,
                      0, 255).astype(np.uint8)
        variants.append(cv2.LUT(orig, lut))
    for im in variants:
        outs.append(cv2.resize(im, (RES, RES),
                               interpolation=cv2.INTER_AREA))
    return outs


def main():
    from diffusers import (DDPMScheduler,
                           StableDiffusionPipeline)
    from peft import LoraConfig, get_peft_model

    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")
    pipe = StableDiffusionPipeline.from_single_file(
        ckpt, torch_dtype=torch.float32, safety_checker=None)
    pipe.set_progress_bar_config(disable=True)
    vae, unet, te = pipe.vae, pipe.unet, pipe.text_encoder
    vae.requires_grad_(False).eval()
    te.requires_grad_(False).eval()
    sched = DDPMScheduler.from_config(pipe.scheduler.config)

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    imgs = _augment(orig)
    print(f"[lora] dataset: {len(imgs)} images", flush=True)

    with torch.no_grad():
        latents = []
        for im in imgs:
            t = torch.from_numpy(
                cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) \
                .permute(2, 0, 1).float() / 127.5 - 1
            lat = vae.encode(t.unsqueeze(0)).latent_dist.mean \
                * 0.18215
            latents.append(lat)
        latents = torch.cat(latents)                    # [N,4,56,56]
        tok = pipe.tokenizer(
            [CAPTION], padding="max_length", truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt")
        text_emb = te(tok.input_ids)[0]                 # [1,77,768]
    del vae, te, pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    cfg = LoraConfig(r=RANK, lora_alpha=RANK, init_lora_weights="gaussian",
                     target_modules=["to_q", "to_k", "to_v",
                                     "to_out.0"])
    unet = get_peft_model(unet, cfg)
    unet.enable_gradient_checkpointing()
    unet.train()
    opt = torch.optim.AdamW(unet.parameters(), lr=LR)
    n = latents.shape[0]
    gen = torch.Generator().manual_seed(0)

    t_start = time.time()
    for step in range(STEPS):
        i = step % n
        lat = latents[i:i + 1]
        noise = torch.randn(lat.shape, generator=gen)
        t = torch.randint(0, sched.config.num_train_timesteps,
                          (1,), generator=gen)
        noisy = sched.add_noise(lat, noise, t)
        pred = unet(noisy, t, encoder_hidden_states=text_emb,
                    return_dict=True).sample
        loss = F.mse_loss(pred.float(), noise.float())
        loss.backward()
        if (step + 1) % 2 == 0 or step == STEPS - 1:
            opt.step()
            opt.zero_grad(set_to_none=True)
        if (step + 1) % 25 == 0:
            el = time.time() - t_start
            print(f"[lora] step {step + 1}/{STEPS} "
                  f"loss {loss.item():.4f} "
                  f"{el / (step + 1):.1f}s/it eta "
                  f"{(STEPS - step - 1) * el / (step + 1) / 60:.0f}min",
                  flush=True)

    os.makedirs(SAVE_DIR, exist_ok=True)
    unet.save_pretrained(SAVE_DIR)
    print(f"[lora] saved to {SAVE_DIR}", flush=True)


if __name__ == "__main__":
    main()
