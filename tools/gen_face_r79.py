"""R79 (R78 review: mouth/brows fixed BUT hair repaint lost
the bangs identity + flyaway strands; eyes still big round
double-lidded = young):

R79:
1. HAIR: NO repaint. Small classical crown-level warp on
   genr77_s7: right crown pulled DOWN 9px, left crown raised
   6px (smooth gaussian column shifts, only y<210) - levels
   the silhouette without touching a single strand.
2. EYES (anti-young, stronger lock):
   - CLOSED almond outline in the guide (upper lash + lower
     lid joined at the corners) so the model treats it as
     the complete eye boundary
   - iris r5.5, ~60% covered
   - ControlNet 0.95 + guidance 3.0 so 'monolid' positive
     and 'double eyelid' negative actually bite
   - brows kept low/thick/flat (approved in R78)
3. Mouth: keep R78 short straight dark line.
Base: genr77_s7. 4 seeds.
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
                          TRIGGER, LORA_SCALE, CX, MASK_ELL,
                          _face_mask, _scale_lora)  # noqa: E402

GUIDANCE = 3.0
CN_SCALE = 0.95
IP_SCALE = 0.9
FACE_SEEDS = [7, 42, 998, 123]
FACE_STRENGTH = 0.85


def _level_crown(img):
    """pull the right crown down 9px, raise the left 6px"""
    H, W = img.shape
    xs = np.arange(W, dtype=np.float32)[None, :]
    ys = np.arange(H, dtype=np.float32)[:, None]
    dy = (9.0 * np.exp(-(((xs - 330) / 55.0) ** 2))
          - 6.0 * np.exp(-(((xs - 200) / 60.0) ** 2)))
    fade = np.clip((215.0 - ys) / 179.0, 0.0, 1.0)
    src_y = ys - dy * fade
    src_x = np.repeat(xs, H, axis=0)
    return cv2.remap(img, src_x.astype(np.float32),
                     src_y.astype(np.float32), cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=255)


def _face_guide(base):
    g = _lineart(base)
    y0, y1 = 245, 366
    right = g[y0:y1, CX:CX + 108]
    g[y0:y1, CX - 108:CX] = right[:, ::-1]
    g[246:298, 148:368] = 255     # brows + eyes
    g[290:332, 228:292] = 255     # nose
    g[330:366, 222:298] = 255     # mouth + chin interior
    # low flat thick brows
    for sgn in (-1, 1):
        cv2.line(g, (CX + sgn * 28, 257), (CX + sgn * 60, 255),
                 75, 3, cv2.LINE_AA)
        cv2.line(g, (CX + sgn * 60, 255), (CX + sgn * 94, 256),
                 75, 2, cv2.LINE_AA)
    # CLOSED almond monolid eyes
    for cx in (196, 320):
        cv2.circle(g, (cx, 287), 5, 100, -1, cv2.LINE_AA)
        cv2.rectangle(g, (cx - 32, 266), (cx + 32, 282), 255,
                      -1)
        up = [(cx - 31, 284), (cx - 12, 281), (cx + 12, 281),
              (cx + 31, 284)]
        for i in range(3):
            cv2.line(g, up[i], up[i + 1], 45, 3, cv2.LINE_AA)
        lo = [(cx - 31, 284), (cx - 12, 291), (cx + 12, 291),
              (cx + 31, 284)]
        for i in range(3):
            cv2.line(g, lo[i], lo[i + 1], 90, 1, cv2.LINE_AA)
    # nose (approved)
    cv2.line(g, (CX, 296), (CX, 311), 135, 1, cv2.LINE_AA)
    cv2.ellipse(g, (CX, 314), (5, 4), 0, 10, 170, 110, 1,
                cv2.LINE_AA)
    for sgn in (-1, 1):
        nx = CX + sgn * 8
        cv2.ellipse(g, (nx, 320), (3, 2), 0,
                    210 if sgn < 0 else 150,
                    340 if sgn < 0 else 380, 85, 1, cv2.LINE_AA)
    # short straight dark mouth
    cv2.line(g, (CX - 14, 339), (CX + 14, 339), 70, 2,
             cv2.LINE_AA)
    cv2.ellipse(g, (CX, 346), (8, 3), 0, 25, 155, 155, 1,
                cv2.LINE_AA)
    # slimmer longer chin
    cv2.ellipse(g, (CX, 328), (38, 42), 0, 22, 158, 95, 1,
                cv2.LINE_AA)
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr77_s7.png"), 0)
    assert base is not None
    H, W = base.shape
    level = _level_crown(base)
    cv2.imwrite(os.path.join(OUT, "_r79_level.png"), level)
    guide = _face_guide(level)
    mask = _face_mask((H, W))
    init = level.copy()
    cx, cy, ax, ay = MASK_ELL
    cv2.ellipse(init, (cx, cy), (ax - 6, ay - 6), 0, 0, 360,
                255, -1)
    cv2.imwrite(os.path.join(OUT, "_r79_guide.png"), guide)
    bpil = Image.fromarray(init).convert("RGB")
    mpil = Image.fromarray(mask).convert("RGB")
    ctrl = Image.fromarray(guide).convert("RGB")

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

    cands = [("level", level)]
    for seed in FACE_SEEDS:
        t0 = time.time()
        res = pipe(prompt=FACE_PROMPT, negative_prompt=FACE_NEG,
                   image=bpil, mask_image=mpil,
                   control_image=ctrl,
                   controlnet_conditioning_scale=CN_SCALE,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8,
                   guidance_scale=GUIDANCE,
                   strength=FACE_STRENGTH,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr79_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        cands.append((f"s{seed}", g))
        print(f"[r79 {name}] {time.time() - t0:.0f}s", flush=True)

    n = len(cands)
    crops = [im[230:520, 100:420] for _, im in cands]
    ch, cw = crops[0].shape
    zs = np.full((ch, cw * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), c in zip(cands, crops):
        p = c.copy()
        cv2.putText(p, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, 0, 2, cv2.LINE_AA)
        zs[:, x:x + cw] = p
        x += cw + 10
    cv2.imwrite(os.path.join(OUT, "_genr79_faces.png"), zs)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in cands:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr79_sheet.png"), sheet)
    print("[r79] sheets saved", flush=True)


if __name__ == "__main__":
    main()
