# people.jpg 终装脚本: 全部已验证工序按正确顺序一键执行
# 源: result/rnr/rnr_frontal.png (A版管线输出, 0.450)
# 链: 补牙(s7) -> 唇0.78 -> E重塑(0.3,IP0.35) -> 发际线二次修复(d21)
#     -> 鼻孔结构(填大孔+渐变软缝x2+鼻尖压影) -> 鼻梁修容 -> 输出 512 终版
# 之后: python tools/make_full.py result/_fin_final.png 完成全帧
import importlib.util
import os
_HFC = os.environ.get("HF_HOME", r"D:\huggingface_cache")  # 模型缓存根目录, 可用环境变量 HF_HOME 覆盖
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ga = _load("gen_arc2face", "tools/gen_arc2face.py")
ga.torch = torch
gr = _load("gen_rnr", "tools/gen_rnr.py")

fin = cv2.imread("result/rnr/rnr_frontal.png")
assert fin is not None
H, W = fin.shape[:2]
lm, _ = gr._tddfa_68(fin)

# ---------- 1) 补牙 (SD inpaint, 内唇掩码, 种子7) ----------
inner = lm[60:68].astype(np.int32)
mask = np.zeros((H, W), np.uint8)
cv2.fillPoly(mask, [inner], 255)
mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
pipe = ga._load_pipeline(ip_scale=0)
from diffusers import (StableDiffusionInpaintPipeline,
                       StableDiffusionImg2ImgPipeline)
inp = StableDiffusionInpaintPipeline(**pipe.components)
inp.enable_attention_slicing()
pos = ga._encode_text(inp, "slightly open mouth with natural white teeth, "
                           "realistic teeth, photographic portrait")
neg = ga._encode_text(inp, ga.NEG_TEXT + ", black void, empty mouth, "
                      "no teeth, deformed")
gen = torch.Generator("cpu").manual_seed(7)
res = inp(prompt_embeds=pos, negative_prompt_embeds=neg, image=Image.fromarray(
    cv2.cvtColor(fin, cv2.COLOR_BGR2RGB)), mask_image=Image.fromarray(mask),
    guidance_scale=4.5, num_inference_steps=28, generator=gen).images[0]
res_np = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2BGR).astype(np.float32)
a = cv2.GaussianBlur(mask, (7, 7), 0).astype(np.float32) / 255.0
fin = np.clip(fin.astype(np.float32) * (1 - a[..., None])
              + res_np * a[..., None], 0, 255).astype(np.uint8)
print("[1] 补牙完成", flush=True)

# ---------- 2) 唇垂直压缩 k=0.78 ----------
outer = lm[48:60]
x0, x1 = outer[:, 0].min() - 14, outer[:, 0].max() + 14
y0, y1 = outer[:, 1].min() - 12, outer[:, 1].max() + 12
cx, cy = outer.mean(axis=0)
rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
K = 0.78
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
d = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) ** 0.5
t = np.clip((1.25 - d) / 0.25, 0, 1)
map_y = cy + (yy - cy) * (1 + (1 / K - 1) * t)
fin = cv2.remap(fin, xx, map_y, cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REPLICATE)
print("[2] 唇压缩完成", flush=True)
lm, _ = gr._tddfa_68(fin)          # 形变后重检

# ---------- 3) E 全脸低强度重塑 (0.3, IP0.35 真鼻参照) ----------
orig = cv2.imread("4/people.jpg")
lm_o, _ = gr._tddfa_68(orig)
nx, ny = int(lm_o[30][0]), int(lm_o[30][1])
nose_crop = orig[max(ny - 85, 0):ny + 85, max(nx - 85, 0):nx + 85]
ip_img = Image.fromarray(
    cv2.cvtColor(cv2.resize(nose_crop, (224, 224)), cv2.COLOR_BGR2RGB))
pipe.disable_attention_slicing()   # 切片与 IP 挂载冲突 (先解后挂)
pipe = ga._attach_ip_adapter(pipe, 0.35)
i2i = StableDiffusionImg2ImgPipeline(**pipe.components)
ip_emb = i2i.prepare_ip_adapter_image_embeds(ip_img, None, "cpu", 1, True)
pos = ga._encode_text(
    i2i, "portrait with elegant refined adult nose, high straight nose "
         "bridge, small refined tip, narrow symmetrical nostrils, "
         "natural face, photographic")
neg = ga._encode_text(
    i2i, ga.NEG_TEXT + ", deformed, crooked, drooping, extra tissue, "
         "childish button nose, hooked nose")
i2i.text_encoder = None
i2i.image_encoder = None
gen = torch.Generator("cpu").manual_seed(7)
res = i2i(prompt_embeds=pos, negative_prompt_embeds=neg,
          image=Image.fromarray(cv2.cvtColor(fin, cv2.COLOR_BGR2RGB)),
          ip_adapter_image_embeds=ip_emb, strength=0.3, guidance_scale=4.0,
          num_inference_steps=28, generator=gen).images[0]
res_np = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2BGR).astype(np.float32)
hull = cv2.convexHull(lm[27:36].astype(np.int32))
nmask = np.zeros((H, W), np.uint8)
cv2.fillConvexPoly(nmask, hull, 255)
nmask = cv2.dilate(nmask, np.ones((29, 29), np.uint8))
a = cv2.GaussianBlur(nmask, (9, 9), 0).astype(np.float32) / 255.0
fin = np.clip(fin.astype(np.float32) * (1 - a[..., None])
              + res_np * a[..., None], 0, 255).astype(np.uint8)
print("[3] E重塑完成 (只贴鼻区)", flush=True)

del pipe, i2i, inp
import gc
gc.collect()

# ---------- 4) 发际线二次修复 (E 重塑会重画花发际线, A版修复重跑) ----------
comp = cv2.imread("result/rnr/_outpaint_comp.png")
assert comp is not None, "缺 _outpaint_comp.png (管线调试件)"


def _skin_strict(im, dil):
    ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2]
    sk = ((cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
          & (y > 110)).astype(np.uint8) * 255
    return cv2.dilate(sk, np.ones((dil, dil), np.uint8)) if dil else sk


skin = _skin_strict(comp, 21)
hair = np.zeros((H, W), np.uint8)
hair[0:180, 0:340] = 255
glare = ((fin.sum(axis=2).astype(int)
          - comp.sum(axis=2).astype(int) > 60)).astype(np.uint8) * 255
neck_zone = np.zeros((H, W), np.uint8)
neck_zone[385:495, 120:330] = 255
neck = cv2.dilate(cv2.bitwise_and(glare, neck_zone),
                  np.ones((15, 15), np.uint8))
mask = cv2.bitwise_or(hair, neck)
mask[skin > 0] = 0
a = cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0
fin = np.clip(fin.astype(np.float32) * (1 - a[..., None])
              + comp.astype(np.float32) * a[..., None], 0, 255
              ).astype(np.uint8)
print(f"[4] 发际线二次修复 ({int((mask > 0).sum())}px)", flush=True)

# ---------- 5) 鼻孔结构: 大孔填平 + 渐变软缝x2 + 鼻尖压影 ----------
lm, _ = gr._tddfa_68(fin)
dark = np.zeros((H, W), np.uint8)
z = fin[302:324, 258:276]
dark[302:324, 258:276] = (z.sum(axis=2) < 200).astype(np.uint8) * 255
dark = cv2.dilate(dark, np.ones((3, 3), np.uint8))
fin = cv2.inpaint(fin, dark, 5, cv2.INPAINT_TELEA)
for cx in (266, 250):
    slit = np.zeros((H, W), np.float32)
    cv2.ellipse(slit, (cx, 314), (4, 3), 0, 0, 360, 1.0, -1)
    slit = cv2.GaussianBlur(slit, (5, 5), 0)
    fin = fin.astype(np.float32) * (1 - 0.62 * slit[..., None])
burn = np.zeros((H, W), np.float32)
cv2.ellipse(burn, (260, 296), (30, 18), 0, 15, 165, 0.06, -1)
burn = cv2.GaussianBlur(burn, (7, 7), 0)
fin = np.clip(fin.astype(np.float32) * (1 - burn[..., None]),
              0, 255).astype(np.uint8)
print("[5] 鼻孔结构完成", flush=True)

# ---------- 6) 鼻梁修容 ----------
lm, _ = gr._tddfa_68(fin)
top = lm[27].astype(float)
tip = lm[30].astype(float)
spine = tip - top
L = float(np.hypot(*spine))
u = spine / L
n = np.array([-u[1], u[0]])
field = np.zeros((H, W), np.float32)


def seg(p0, p1, peak, thick):
    steps = int(np.hypot(*(p1 - p0))) * 2
    for i in range(steps + 1):
        p = p0 + (p1 - p0) * (i / max(steps, 1))
        cv2.circle(field, tuple(np.round(p).astype(int)), thick,
                   float(peak), -1)


for side in (-1, 1):
    off = n * 7 * side
    seg(top + u * 6 + off, tip - u * 4 + off, 0.07, 5)
for idx in (31, 35):
    c = lm[idx].astype(float)
    seg(c + np.array([-4.0, 2.0]), c + np.array([4.0, 2.0]), 0.05, 4)
seg(tip - u * 3, tip - u * 1, -0.05, 5)
field = cv2.GaussianBlur(field, (9, 9), 0)
zone = np.zeros((H, W), np.uint8)
cv2.fillConvexPoly(zone, cv2.convexHull(lm[27:36].astype(np.int32)), 255)
zone = cv2.dilate(zone, np.ones((35, 35), np.uint8))
field[zone == 0] = 0
fin = np.clip(fin.astype(np.float32) * (1 - field[..., None]),
              0, 255).astype(np.uint8)
print("[6] 鼻梁修容完成", flush=True)

cv2.imwrite("result/_fin_final.png", fin)

onnx = os.path.join(
    ga._snapshot(_HFC, "models--FoivosPar--Arc2Face"),
    "arcface.onnx")
cos = gr._embed_cosine(orig, fin, onnx)
print(f"[sim] 512 终版 = {cos:.3f}", flush=True)
