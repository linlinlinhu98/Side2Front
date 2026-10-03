# region_fix.py: 部位定向修复 (GUI "部位重修"按钮的后端)
# 用法: python tools/region_fix.py <fix> <img> <out> [原图]
#   fix: nose(鼻=E重塑+修容) / teeth(牙=补牙) / lips(唇=压薄0.82)
#        jaw(下颌=颌下阴影) / mouth(嘴=补牙+压唇)
# 每个部位 = 本会话已验证的配方, 输出为新文件 (不覆盖输入)
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


FIX = sys.argv[1]
IMG = sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(IMG)[0] + "_fixed.png"
ORIG = sys.argv[4] if len(sys.argv) > 4 else "4/people.jpg"

ga = _load("gen_arc2face", "tools/gen_arc2face.py")
ga.torch = torch
gr = _load("gen_rnr", "tools/gen_rnr.py")

fin = cv2.imread(IMG)
assert fin is not None, f"读不到 {IMG}"
orig = cv2.imread(ORIG)
H, W = fin.shape[:2]
lm, _ = gr._tddfa_68(fin)
applied = []

# ---------- teeth: 内唇多边形 inpaint (补牙, 种子7) ----------
if FIX in ("teeth", "mouth"):
    inner = lm[60:68].astype(np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [inner], 255)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    dark_mouth = int(((fin.sum(axis=2) < 200) & (mask > 0)).sum())
    pipe = ga._load_pipeline(ip_scale=0)
    from diffusers import StableDiffusionInpaintPipeline
    inp = StableDiffusionInpaintPipeline(**pipe.components)
    inp.enable_attention_slicing()
    pos = ga._encode_text(inp, "slightly open mouth with natural white "
                          "teeth, realistic teeth, photographic portrait")
    neg = ga._encode_text(inp, ga.NEG_TEXT + ", black void, empty mouth, "
                          "no teeth, deformed")
    inp.text_encoder = None
    gen = torch.Generator("cpu").manual_seed(7)
    res = inp(prompt_embeds=pos, negative_prompt_embeds=neg,
              image=Image.fromarray(
                  cv2.cvtColor(fin, cv2.COLOR_BGR2RGB)),
              mask_image=Image.fromarray(mask), guidance_scale=4.5,
              num_inference_steps=28, generator=gen).images[0]
    res_np = cv2.cvtColor(np.asarray(res),
                          cv2.COLOR_RGB2BGR).astype(np.float32)
    a = cv2.GaussianBlur(mask, (7, 7), 0).astype(np.float32) / 255.0
    fin = np.clip(fin.astype(np.float32) * (1 - a[..., None])
                  + res_np * a[..., None], 0, 255).astype(np.uint8)
    applied.append(f"补牙(残留暗像素{dark_mouth}px)")
    print("[teeth] 补牙完成", flush=True)

# ---------- mouth: 唇压缩 k=0.82 (嘴 = 补牙后接压唇) ----------
if FIX == "mouth":
    outer = lm[48:60]
    x0, x1 = outer[:, 0].min() - 14, outer[:, 0].max() + 14
    y0, y1 = outer[:, 1].min() - 12, outer[:, 1].max() + 12
    cx, cy = outer.mean(axis=0)
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    K = 0.82
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) ** 0.5
    t = np.clip((1.25 - d) / 0.25, 0, 1)
    map_y = cy + (yy - cy) * (1 + (1 / K - 1) * t)
    fin = cv2.remap(fin, xx, map_y, cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REPLICATE)
    applied.append("唇压缩0.82")

# ---------- lips: 唇压缩 k=0.82 (独立按钮同款) ----------
if FIX == "lips":
    outer = lm[48:60]
    x0, x1 = outer[:, 0].min() - 14, outer[:, 0].max() + 14
    y0, y1 = outer[:, 1].min() - 12, outer[:, 1].max() + 12
    cx, cy = outer.mean(axis=0)
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    K = 0.82
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) ** 0.5
    t = np.clip((1.25 - d) / 0.25, 0, 1)
    map_y = cy + (yy - cy) * (1 + (1 / K - 1) * t)
    fin = cv2.remap(fin, xx, map_y, cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REPLICATE)
    applied.append("唇压缩0.82")

# ---------- nose: E重塑(0.3, IP真鼻参照, 只贴鼻区) + 鼻梁修容 ----------
if FIX == "nose":
    lm_o, _ = gr._tddfa_68(orig)
    nx, ny = int(lm_o[30][0]), int(lm_o[30][1])
    nose_crop = orig[max(ny - 85, 0):ny + 85, max(nx - 85, 0):nx + 85]
    ip_img = Image.fromarray(
        cv2.cvtColor(cv2.resize(nose_crop, (224, 224)), cv2.COLOR_BGR2RGB))
    pipe = ga._load_pipeline(ip_scale=0.35)
    from diffusers import StableDiffusionImg2ImgPipeline
    i2i = StableDiffusionImg2ImgPipeline(**pipe.components)
    ip_emb = i2i.prepare_ip_adapter_image_embeds(ip_img, None, "cpu", 1,
                                                 True)
    pos = ga._encode_text(
        i2i, "portrait with elegant refined adult nose, high straight "
             "nose bridge, small refined tip, narrow symmetrical nostrils, "
             "natural face, photographic")
    neg = ga._encode_text(
        i2i, ga.NEG_TEXT + ", deformed, crooked, drooping, extra tissue, "
             "childish button nose, hooked nose")
    i2i.text_encoder = None
    i2i.image_encoder = None
    gen = torch.Generator("cpu").manual_seed(7)
    res = i2i(prompt_embeds=pos, negative_prompt_embeds=neg,
              image=Image.fromarray(
                  cv2.cvtColor(fin, cv2.COLOR_BGR2RGB)),
              ip_adapter_image_embeds=ip_emb, strength=0.3,
              guidance_scale=4.0, num_inference_steps=28,
              generator=gen).images[0]
    res_np = cv2.cvtColor(np.asarray(res),
                          cv2.COLOR_RGB2BGR).astype(np.float32)
    hull = cv2.convexHull(lm[27:36].astype(np.int32))
    nmask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(nmask, hull, 255)
    nmask = cv2.dilate(nmask, np.ones((29, 29), np.uint8))
    a = cv2.GaussianBlur(nmask, (9, 9), 0).astype(np.float32) / 255.0
    fin = np.clip(fin.astype(np.float32) * (1 - a[..., None])
                  + res_np * a[..., None], 0, 255).astype(np.uint8)
    applied.append("E重塑0.3")

    # 鼻梁修容 (重检 landmarks)
    lm, _ = gr._tddfa_68(fin)
    top = lm[27].astype(float)
    tip = lm[30].astype(float)
    spine = tip - top
    L = float(np.hypot(*spine))
    u = spine / L
    nv = np.array([-u[1], u[0]])
    field = np.zeros((H, W), np.float32)

    def seg(p0, p1, peak, thick):
        steps = int(np.hypot(*(p1 - p0))) * 2
        for i in range(steps + 1):
            p = p0 + (p1 - p0) * (i / max(steps, 1))
            cv2.circle(field, tuple(np.round(p).astype(int)), thick,
                       float(peak), -1)

    for side in (-1, 1):
        off = nv * 7 * side
        seg(top + u * 6 + off, tip - u * 4 + off, 0.07, 5)
    for idx in (31, 35):
        c = lm[idx].astype(float)
        seg(c + np.array([-4.0, 2.0]), c + np.array([4.0, 2.0]), 0.05, 4)
    seg(tip - u * 3, tip - u * 1, -0.05, 5)
    field = cv2.GaussianBlur(field, (9, 9), 0)
    zn = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(zn, cv2.convexHull(lm[27:36].astype(np.int32)), 255)
    zn = cv2.dilate(zn, np.ones((35, 35), np.uint8))
    field[zn == 0] = 0
    fin = np.clip(fin.astype(np.float32) * (1 - field[..., None]),
                  0, 255).astype(np.uint8)
    applied.append("鼻梁修容")

# ---------- jaw: 颌下阴影 (网格下颌曲线, 需 _rnr_mesh.png) ----------
if FIX == "jaw":
    mesh_path = os.path.join("result", "rnr", "_rnr_mesh.png")
    mesh = cv2.imread(mesh_path)
    if mesh is None:
        print("[jaw] 缺 _rnr_mesh.png, 跳过", flush=True)
    else:
        mesh = cv2.resize(mesh, (W, H), interpolation=cv2.INTER_LANCZOS4)
        sil = (mesh.max(axis=2) > 25).astype(np.uint8) * 255
        sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE,
                               np.ones((15, 15), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(sil)
        sil = (lab == 1 + int(np.argmax(st[1:, 4]))).astype(np.uint8) * 255
        contour = np.full(W, -1.0)
        for x in range(W):
            ys = np.where(sil[:, x] > 0)[0]
            if len(ys):
                contour[x] = ys.max()
        valid = contour > 340
        xs = np.where(valid)[0]
        if len(xs):
            cs = contour[valid]
            cs_s = np.array([np.median(cs[max(0, i - 7):i + 8])
                             for i in range(len(cs))], np.float32)
            cs_s = cv2.GaussianBlur(cs_s.reshape(1, -1), (1, 31),
                                    0).ravel()
            contour[valid] = cs_s
            D = 14
            shadow = np.zeros((H, W), np.float32)
            for x in range(xs.min(), xs.max() + 1):
                y0 = int(contour[x])
                span = (xs.max() - xs.min()) / 2.0
                lat = 1.0 - max(0.0, (abs(x - (xs.min() + xs.max()) / 2)
                                      / span - 0.75) / 0.25)
                for d in range(D):
                    if y0 + d < H:
                        shadow[y0 + d, x] = 0.10 * (1 - d / D) * lat
            shadow = cv2.GaussianBlur(shadow, (7, 7), 0)
            fin = np.clip(fin.astype(np.float32)
                          * (1 - shadow[..., None]), 0, 255).astype(np.uint8)
            applied.append("颌下阴影")

cv2.imwrite(OUT, fin)
onnx = os.path.join(
    ga._snapshot(_HFC, "models--FoivosPar--Arc2Face"),
    "arcface.onnx")
cos = gr._embed_cosine(orig, fin, onnx)
print(f"[fix] {FIX} 完成: {', '.join(applied) if applied else '无变化'}",
      flush=True)
print(f"[sim] vs 原图 arcface = {cos:.3f}", flush=True)
print(f"[done] {OUT}", flush=True)
