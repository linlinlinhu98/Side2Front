"""R86 (user: R85 不如base; 在 base 基础上再优化嘴唇和衣服):

Base: genr84_m42_cloth. MASKED zone passes only (the R85
full-frame pass drifted eyes/composition - rejected).

Lips v2: softer than R84 - thinner lighter lip line, wider
softer lower-lip shadow, plus a whisper of cupid's-bow hint
above the line center (classmate's lips are tone-built, not
line-built).
Clothes v2: MORE hatching - added lower-hem band + belly fold
valleys + sleeve outer folds to the guide zones, strength
0.55 (was 0.5).
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
from gen_r83 import (CLOTH_PROMPT, CLOTH_NEG, LORA_DIR,
                     TRIGGER, CX, _scale_lora, _mouth_mask,
                     _torso_mask)  # noqa: E402
from gen_r84 import (MOUTH_PROMPT, MOUTH_NEG, _hatch)  # noqa: E402


def _mouth_guide(base):
    g = _lineart(base)
    g[326:356, 222:296] = 255
    # softer thinner level lip line
    cv2.line(g, (CX - 14, 340), (CX + 14, 340), 65, 1,
             cv2.LINE_AA)
    # whisper of cupid's bow above center (tone hint)
    cv2.ellipse(g, (CX, 336), (6, 2), 0, 200, 340, 185, 1,
                cv2.LINE_AA)
    # wider softer lower-lip volume
    cv2.ellipse(g, (CX, 349), (13, 5), 0, 20, 160, 180, 1,
                cv2.LINE_AA)
    cv2.ellipse(g, (CX, 354), (9, 3), 0, 25, 155, 195, 1,
                cv2.LINE_AA)
    return g


def _cloth_guide(base):
    g = _lineart(base)
    zones = [
        # collar-roll shadows
        [(165, 425), (250, 430), (246, 458), (168, 452)],
        [(266, 430), (350, 425), (348, 452), (270, 458)],
        # zipper placket sides
        [(243, 445), (252, 445), (252, 645), (243, 645)],
        [(264, 445), (273, 445), (273, 645), (264, 645)],
        # side seams
        [(96, 480), (112, 480), (104, 650), (88, 650)],
        [(404, 480), (420, 480), (428, 650), (412, 650)],
        # chest fold valleys
        [(175, 520), (240, 528), (236, 548), (172, 540)],
        [(280, 545), (345, 536), (348, 556), (284, 565)],
        # belly folds
        [(200, 585), (320, 585), (316, 605), (204, 605)],
        # lower hem band
        [(120, 622), (396, 622), (396, 646), (120, 646)],
        # sleeve outer folds
        [(84, 505), (104, 500), (96, 585), (76, 590)],
        [(412, 500), (432, 505), (440, 590), (420, 585)],
    ]
    for poly in zones:
        m = np.zeros_like(g)
        cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
        tmp = np.full_like(g, 255)
        _hatch(tmp, poly, spacing=8, val=130)
        g[m > 0] = tmp[m > 0]
    return g


def main():
    base = cv2.imread(os.path.join(OUT, "genr84_m42_cloth.png"),
                      0)
    assert base is not None
    H, W = base.shape

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :]),
            Image.fromarray(rgb[280:, :])]

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
    pipe.set_progress_bar_config(disable=True)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(1.0)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    # ---- stage A: lips v2 (LoRA 0.3, CN 1.0) ----
    _scale_lora(pipe.unet, 0.3)
    mmask = _mouth_mask((H, W))
    mguide = _mouth_guide(base)
    minit = base.copy()
    minit[324:358, 218:300] = 255
    mouths = []
    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=MOUTH_PROMPT, negative_prompt=MOUTH_NEG,
                   image=Image.fromarray(minit).convert("RGB"),
                   mask_image=Image.fromarray(mmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(mguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8, guidance_scale=3.5,
                   strength=0.75,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr86_m{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        mouths.append((seed, g))
        print(f"[r86 {name}] {time.time() - t0:.0f}s", flush=True)

    # ---- stage B: clothes v2 (LoRA 1.0, strength 0.55) ----
    _scale_lora(pipe.unet, 1.0 / 0.3)
    tmask = _torso_mask((H, W))
    finals = []
    for seed, mimg in mouths:
        tguide = _cloth_guide(mimg)
        t0 = time.time()
        res = pipe(prompt=CLOTH_PROMPT, negative_prompt=CLOTH_NEG,
                   image=Image.fromarray(mimg).convert("RGB"),
                   mask_image=Image.fromarray(tmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(tguide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.9,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=8, guidance_scale=3.0,
                   strength=0.55,
                   generator=torch.Generator("cpu")
                   .manual_seed(998)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        name = f"genr86_m{seed}_cloth.png"
        cv2.imwrite(os.path.join(OUT, name), g)
        finals.append((f"m{seed}+cloth", g))
        print(f"[r86 {name}] {time.time() - t0:.0f}s", flush=True)

    panels = [("base", base)] + [(f"m{s}", m) for s, m in mouths] \
             + finals
    n = len(panels)
    sheet = np.full((660, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_genr86_sheet.png"), sheet)
    mz = [cv2.resize(im[322:360, 214:304], None, fx=5, fy=5,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    mh, mw = mz[0].shape
    msheet = np.full((mh, mw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, mz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        msheet[:, x:x + mw] = t
        x += mw + 8
    cv2.imwrite(os.path.join(OUT, "_genr86_mouths.png"), msheet)
    cz = [cv2.resize(im[420:650, 60:460], None, fx=1.5, fy=1.5,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    ch, cw = cz[0].shape
    csheet = np.full((ch, cw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, cz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        csheet[:, x:x + cw] = t
        x += cw + 8
    cv2.imwrite(os.path.join(OUT, "_genr86_cloths.png"), csheet)
    print("[r86] sheets saved", flush=True)


if __name__ == "__main__":
    main()
