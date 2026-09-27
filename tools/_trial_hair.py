"""R40 quick trial: regrow HAIR + JAW/CHIN + NECK with Counterfeit
(anime ckpt) on top of the APPROVED r39 output (eyes untouched —
deterministic seed, so the r39 file IS the pipeline state).

Why: the user rejected 3 rounds of classical patching (下巴模糊 /
脸无边界 / 脖子圆柱无纹理 / 发际模糊 / 头发拼接). The warp base is
inherently blurry+mirror-symmetric; threshold/手绘 patches can't
manufacture the original artist's crisp pencil texture. The anime
model draws hair/neck natively; ControlNet lineart holds the
composition, IP-Adapter locks the original's strand style.

Mask = everything above y400 EXCEPT the approved feature islands
(eyes+brows band, nose, mouth, blush) and the approved torso.
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

PROMPT = ("(monochrome:1.4), (pencil sketch:1.4), (traditional "
          "media:1.2), graphite, fine pencil strokes, 1boy, short "
          "black hair, flowing hair strands, crisp thin lineart, "
          "sharp jawline, (slim bare neck:1.3), visible neck skin, "
          "(white background:1.3)")
NEG = ("color, colored, photo, realistic, 3d, blurry, messy lines, "
       "long hair, hat, watermark, text, deformed, (turtleneck:1.4), "
       "(collar:1.3), clothes, speckles, noise, gray background")

final = cv2.imread(os.path.join(OUT, "eyes_r39_sd_s998.png"), 0)
assert final is not None
# the model keeps continuing the dark knot cluster UPWARD into a
# turtleneck — whiten the neck zone in the INPUT so there is no
# dark mass to continue (the knot below the mask stays visible as
# context, the pencil neck lines stay in the ControlNet guide)
final_in = final.copy()
final_in[315:402, 195:317] = 255
orig = cv2.imread(os.path.join(ROOT, "4", "face.png"))
# IP style reference: the ORIGINAL's own hair strands (pure hair,
# no face — R31 lesson: face in the IP crop plants phantom features)
ip_hair = Image.fromarray(cv2.cvtColor(orig[0:145, 100:420],
                                       cv2.COLOR_BGR2RGB))

# ControlNet lineart from the current image; WHITE the crown
# interior so the mirrored chevron lines do NOT bias the model
# toward the same symmetric pattern (silhouette + hairline stay)
g = final.astype(np.float32)
loc = cv2.GaussianBlur(g, (0, 0), 6)
hp = np.clip(loc - g, 0, None)
hp[hp < 8] = 0
lines = 255 - np.clip(hp * 2.0, 0, 255).astype(np.uint8)
lines[10:170, 140:372] = 255
cv2.imwrite(os.path.join(OUT, "_ctrl_hair.png"), lines)
ctrl_rgb = Image.fromarray(lines).convert("RGB")

hm = np.zeros(final.shape, np.uint8)
hm[0:400, 40:472] = 255
keep = np.zeros(final.shape, np.uint8)
for (y0, y1, x0, x1) in ((205, 268, 155, 357),      # eyes+brows
                         (245, 295, 225, 290),      # nose
                         (283, 312, 232, 282),      # mouth
                         (245, 270, 168, 215),      # blush L
                         (245, 270, 297, 345)):     # blush R
    keep[y0:y1, x0:x1] = 255
keep = cv2.GaussianBlur(keep, (0, 0), 4)
hmf = hm.astype(np.float32) / 255.0 * (1 - keep.astype(np.float32)
                                       / 255.0)
hmf = cv2.GaussianBlur(hmf, (0, 0), 3)
hm_u8 = np.clip(hmf * 255, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(OUT, "_mask_hair.png"), hm_u8)

from diffusers import (ControlNetModel, LCMScheduler,  # noqa: E402
                       StableDiffusionControlNetInpaintPipeline)

cn = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float32)
lcm = _find_file(os.path.join(r"D:\huggingface_cache",
                              "models--latent-consistency--lcm-lora-sdv1-5"),
                 "pytorch_lora_weights.safetensors")
ckpt = _find_file(os.path.join(r"D:\huggingface_cache",
                               "models--gsdf--Counterfeit-V3.0"),
                  "_fp16.safetensors")
pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
    ckpt, controlnet=cn, torch_dtype=torch.float32,
    safety_checker=None)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(lcm)
try:
    pipe.fuse_lora()
except Exception as e:
    print(f"[warn] fuse_lora: {e}", flush=True)
pipe.set_progress_bar_config(disable=True)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                     weight_name="ip-adapter_sd15.safetensors")
pipe.set_ip_adapter_scale(0.5)

t0 = time.time()
gen_path = os.path.join(OUT, "_hair_gen.png")
if os.path.exists(gen_path):  # iteration 5+: reuse pass 1, retune pass 2
    g2 = cv2.imread(gen_path, 0)
else:
    res = pipe(prompt=PROMPT, negative_prompt=NEG,
               image=Image.fromarray(final_in).convert("RGB"),
               mask_image=Image.fromarray(hm_u8),
               control_image=ctrl_rgb,
               controlnet_conditioning_scale=0.75,
               ip_adapter_image=ip_hair,
               height=664, width=520,
               num_inference_steps=8, guidance_scale=1.0,
               strength=0.9,
               generator=torch.Generator("cpu").manual_seed(998)
               ).images[0]
    g2 = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
    g2 = cv2.resize(g2, (final.shape[1], final.shape[0]),
                    interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(gen_path, g2)
out = (final.astype(np.float32) * (1 - hmf)
       + g2.astype(np.float32) * hmf).astype(np.uint8)

# SECOND pass: the model insists on painting a dark turtleneck over
# the neck (strong Counterfeit '1boy portrait' prior — prompt,
# negatives and input-whitening all failed). Iteration 4's repaint
# STILL drew a collar: the ControlNet lineart itself contains the
# collar fold lines, so the guide forced the collar back. Iteration
# 5: white the neck zone in the GUIDE as well, so pass 2 sees blank
# paper there and can only draw bare skin with light shading
lines2 = lines.copy()
lines2[312:404, 180:330] = 255
cv2.imwrite(os.path.join(OUT, "_ctrl_hair2.png"), lines2)
ctrl2_rgb = Image.fromarray(lines2).convert("RGB")
band = np.zeros(final.shape, np.uint8)
band[315:404, 185:325] = 255
bandf = cv2.GaussianBlur(band.astype(np.float32) / 255.0, (0, 0), 4)
band_u8 = np.clip(bandf * 255, 0, 255).astype(np.uint8)
in2 = out.copy()
in2[315:404, 185:325] = 255
res3 = pipe(
    prompt=("(monochrome:1.4), (pencil sketch:1.3), (slim bare "
            "neck:1.4), neck skin, throat, subtle pencil shading, "
            "white background"),
    negative_prompt=("(collar:1.5), (turtleneck:1.5), choker, "
                     "clothes, dark, black, color"),
    image=Image.fromarray(in2).convert("RGB"),
    mask_image=Image.fromarray(band_u8),
    control_image=ctrl2_rgb,
    controlnet_conditioning_scale=0.5,
    ip_adapter_image=ip_hair,
    height=664, width=520,
    num_inference_steps=8, guidance_scale=1.0,
    strength=0.95,
    generator=torch.Generator("cpu").manual_seed(998)).images[0]
g3 = cv2.cvtColor(np.asarray(res3), cv2.COLOR_RGB2GRAY)
g3 = cv2.resize(g3, (final.shape[1], final.shape[0]),
                interpolation=cv2.INTER_CUBIC)
out = (out.astype(np.float32) * (1 - bandf)
       + g3.astype(np.float32) * bandf).astype(np.uint8)
# clean the model's speckled paper-noise background wherever it
# touched (both masks, incl. feathers): lift 175+ to paper white —
# dark strands/lines/shading stay, the paste rectangle vanishes
cm = (np.clip(hmf + bandf, 0, 1) > 0.05).astype(np.float32)
cm = cv2.GaussianBlur(cm, (0, 0), 2)
of = out.astype(np.float32)
bg = np.clip((of - 175.0) / 35.0, 0, 1) * cm
out = np.clip(of * (1 - bg) + 252.0 * bg, 0, 255).astype(np.uint8)
# the model's bare neck comes out near-black (median 70 — Counterfeit
# under-chin shadow bias). The STROKES are right, only the tone is
# wrong: gamma-0.35 lift inside the band keeps stroke cores dark
# (5->138) while shading lifts to paper (70->207, 150->235)
of = out.astype(np.float32)
lift = 255.0 * np.power(np.clip(of / 255.0, 0, 1), 0.35)
bf3 = cv2.GaussianBlur(band.astype(np.float32) / 255.0, (0, 0), 3)
out = np.clip(of * (1 - bf3) + lift * bf3, 0, 255).astype(np.uint8)
# despeckle the model's paper noise: (1) erosion test lifts pixels
# with no stroke within 3px, but ONLY in background far (12px) from
# any dark mass so blush/neck shading survive; (2) tiny isolated
# gray components (<30px) near the hair ring -> paper white
of = out.astype(np.float32)
er = cv2.erode(of, np.ones((7, 7), np.uint8))
clean = np.where(er > 165.0, 252.0, of)
mass = (of < 180.0).astype(np.uint8)
mass = cv2.dilate(mass, np.ones((25, 25), np.uint8)).astype(np.float32)
mass = cv2.GaussianBlur(mass, (0, 0), 4)
zone = np.zeros(of.shape, np.float32)
zone[0:402, 40:472] = 1.0
zone = np.clip(cv2.GaussianBlur(zone, (0, 0), 2) * (1 - mass), 0, 1)
out = of * (1 - zone) + clean * zone
cand = ((out > 130.0) & (out < 251.0)).astype(np.uint8)
cand[402:, :] = 0
n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
kill = np.zeros(out.shape, np.uint8)
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] < 30:
        kill[lab == i] = 1
kf = cv2.GaussianBlur(kill.astype(np.float32), (0, 0), 1)
out = np.clip(out * (1 - kf) + 252.0 * kf, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(OUT, "eyes_r40_hair.png"), out)
print(f"[hair r40] {time.time() - t0:.0f}s", flush=True)
