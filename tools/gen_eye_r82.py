"""R80: eye+brow precision micro-pass on R79 seeds (user
pattern: eyes still too open/young, brows too dark).

Eye-band-only repaint (proven R69 technique, edits not
regenerates): closed-almond monolid guide (iris 60% covered),
brows redrawn LIGHTER (value 105 vs 75), strength 0.70 so the
surrounding face stays put. On genr79_s7 + genr79_s42, 2
seeds each.
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
from gen_face_r78 import (FACE_PROMPT, FACE_NEG, LORA_DIR,
                          LORA_SCALE, CX, _scale_lora)  # noqa: E402

GUIDANCE = 3.0
CN_SCALE = 0.95
IP_SCALE = 0.9
STRENGTH = 0.70
BASES = ["genr81_s7.png", "genr81_s42.png"]
SEEDS = [7, 42]
BAND = (258, 279, 100, 40)        # eye+brow band ellipse


def _band_mask(shape):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cx, cy, ax, ay = BAND
    cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)


def _eye_guide(base):
    g = _lineart(base)
    cx0, cy0, ax0, ay0 = BAND
    g[cy0 - ay0:cy0 + ay0, cx0 - ax0:cx0 + ax0] = 255
    # LIGHTER flat brows
    for sgn in (-1, 1):
        cv2.line(g, (CX + sgn * 28, 257), (CX + sgn * 60, 255),
                 105, 3, cv2.LINE_AA)
        cv2.line(g, (CX + sgn * 60, 255), (CX + sgn * 92, 256),
                 105, 2, cv2.LINE_AA)
    # closed almond monolid eyes
    for cx in (196, 320):
        cv2.circle(g, (cx, 287), 5, 100, -1, cv2.LINE_AA)
        cv2.rectangle(g, (cx - 32, 270), (cx + 32, 282), 255,
                      -1)
        up = [(cx - 31, 284), (cx - 12, 281), (cx + 12, 281),
              (cx + 31, 284)]
        for i in range(3):
            cv2.line(g, up[i], up[i + 1], 45, 3, cv2.LINE_AA)
        lo = [(cx - 31, 284), (cx - 12, 291), (cx + 12, 291),
              (cx + 31, 284)]
        for i in range(3):
            cv2.line(g, lo[i], lo[i + 1], 90, 1, cv2.LINE_AA)
    return g


def main():
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[170:300, 100:320])]

    from diffusers import (ControlNetModel, LCMScheduler,
                           StableDiffusionControlNetInpaintPipeline)
    from peft import PeftModel
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)
    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
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

    H, W = 660, 520
    mask = _band_mask((H, W))
    mpil = Image.fromarray(mask).convert("RGB")
    cands = []
    for bname in BASES:
        base = cv2.imread(os.path.join(OUT, bname), 0)
        assert base is not None
        guide = _eye_guide(base)
        cv2.imwrite(os.path.join(
            OUT, f"_r80_guide_{bname.split('_')[1]}.png"), guide)
        bpil = Image.fromarray(base).convert("RGB")
        ctrl = Image.fromarray(guide).convert("RGB")
        cands.append((bname[3:9], base))
        for seed in SEEDS:
            t0 = time.time()
            res = pipe(prompt=FACE_PROMPT,
                       negative_prompt=FACE_NEG,
                       image=bpil, mask_image=mpil,
                       control_image=ctrl,
                       controlnet_conditioning_scale=CN_SCALE,
                       ip_adapter_image_embeds=[emb],
                       height=664, width=520,
                       num_inference_steps=8,
                       guidance_scale=GUIDANCE,
                       strength=STRENGTH,
                       generator=torch.Generator("cpu")
                       .manual_seed(seed)).images[0]
            g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
            g = cv2.resize(g, (W, H),
                           interpolation=cv2.INTER_CUBIC)
            name = (bname.replace(".png", "")
                    .replace("genr81", "genr82")
                    + f"_e{seed}.png")
            cv2.imwrite(os.path.join(OUT, name), g)
            cands.append((name[3:-4], g))
            print(f"[r80 {name}] {time.time() - t0:.0f}s",
                  flush=True)

    n = len(cands)
    crops = [im[240:330, 140:380] for _, im in cands]
    ch, cw = crops[0].shape
    zs = np.full((ch, cw * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), c in zip(cands, crops):
        p = c.copy()
        cv2.putText(p, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        zs[:, x:x + cw] = p
        x += cw + 10
    cv2.imwrite(os.path.join(OUT, "_genr80_eyes.png"), zs)
    print("[r80] sheet saved", flush=True)


if __name__ == "__main__":
    main()
