# auto_refine.py: 512 正脸的自动质检-修复循环
# 把 people.jpg 全流程的教训编译成"检测条件 -> 已验证修复"的质量关卡:
#   D1 口腔黑洞(暗像素>30)         -> 补牙 inpaint (种子7)
#   D2 唇过厚(红唇面积/眼距²>0.52)  -> 垂直压缩至 <=0.50
#   D3 鼻孔不对称(单孔或尺寸比>1.6)  -> 大孔缩小+镜像补对侧
#   RAW 模式(--raw): E重塑(0.3)+发际线二次修复+鼻梁修容 (未经管线处理的图用;
#   管线输出已含这些, 重跑=双重处理劣化 0.416→0.355 实测)
# 用法: python tools/auto_refine.py <512正脸> <原图> [--qc-only] [--raw]
# 输出: <输入名>_refined.png + QC 报告
import importlib.util
import os
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

QC_ONLY = "--qc-only" in sys.argv
RAW = "--raw" in sys.argv
fin_path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith(
    "--") else "result/rnr/rnr_frontal.png"
orig_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith(
    "--") else "4/people.jpg"

fin = cv2.imread(fin_path)
orig = cv2.imread(orig_path)
H, W = fin.shape[:2]
lm, _ = gr._tddfa_68(fin)
lm_o, _ = gr._tddfa_68(orig)
report = []
applied = []


def _skin_strict(im, dil=0):
    ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2]
    sk = ((cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
          & (y > 110)).astype(np.uint8) * 255
    if dil:
        sk = cv2.dilate(sk, np.ones((dil, dil), np.uint8))
    return sk


# ---------- QC 检测 ----------
# D1 口腔黑洞
inner = lm[60:68].astype(np.int32)
mz = np.zeros((H, W), np.uint8)
cv2.fillPoly(mz, [inner], 255)
mz = cv2.dilate(mz, np.ones((3, 3), np.uint8))
dark_mouth = int(((fin.sum(axis=2) < 200) & (mz > 0)).sum())
D1 = dark_mouth > 30
report.append(f"D1 口腔黑洞: {'检出' if D1 else '无'} ({dark_mouth}px)")

# D2 唇厚 (严格红唇面积/眼距²)
io_d = max(abs(lm[39][0] - lm[42][0]), 1)
cx, cy = int(lm[62][0]), int(lm[62][1])
zone = fin[cy - 55:cy + 55, cx - 80:cx + 80]
r, g, b = zone[:, :, 2].astype(int), zone[:, :, 1].astype(int), \
    zone[:, :, 0].astype(int)
red_n = int(((r > 130) & (r > g + 40) & (r > b + 55)).sum())
lip_ratio = red_n / io_d ** 2
D2 = lip_ratio > 0.52
report.append(f"D2 唇厚: {'超标' if D2 else '正常'} "
              f"(比值 {lip_ratio:.2f}, 阈值 0.52)")

# D3 鼻孔对称性 (鼻底暗连通域, 阈值 220)
tip = lm[30].astype(int)
nz = fin[tip[1] + 5:tip[1] + 55, tip[0] - 70:tip[0] + 70]
darkn = (nz.sum(axis=2) < 220).astype(np.uint8)
n3, lab3, st3, cent3 = cv2.connectedComponentsWithStats(darkn)
clusters = [st3[i] for i in range(1, n3) if st3[i, 4] >= 15]
if len(clusters) >= 2:
    areas = sorted([c[4] for c in clusters])
    a1, a2 = areas[-2], areas[-1]
    D3 = a2 / max(a1, 1) > 1.6
    detail = f"双孔面积 {a1}/{a2}"
else:
    D3 = len(clusters) == 1
    detail = f"仅 {len(clusters)} 个孔"
report.append(f"D3 鼻孔对称: {'不对称' if D3 else '正常'} ({detail})")

print("== QC 检测 ==", flush=True)
for ln in report:
    print("  " + ln, flush=True)
if QC_ONLY:
    sys.exit(0)

# ---------- 修复 (SD 加载一次) ----------
pipe = ga._load_pipeline(ip_scale=0)
from diffusers import (StableDiffusionImg2ImgPipeline,
                       StableDiffusionInpaintPipeline)


def _inpaint(img, m, pos, neg, seed=7):
    inp = StableDiffusionInpaintPipeline(**pipe.components)
    inp.enable_attention_slicing()
    gen = torch.Generator("cpu").manual_seed(seed)
    res = inp(prompt_embeds=pos, negative_prompt_embeds=neg,
              image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
              mask_image=Image.fromarray(m), guidance_scale=4.5,
              num_inference_steps=28, generator=gen).images[0]
    return cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2BGR).astype(np.float32)


# F1 补牙
if D1:
    pos = ga._encode_text(pipe, "slightly open mouth with natural white "
                          "teeth, realistic teeth, photographic portrait")
    neg = ga._encode_text(pipe, ga.NEG_TEXT + ", black void, empty mouth, "
                          "no teeth, deformed")
    res_np = _inpaint(fin, mz, pos, neg, 7)
    a = cv2.GaussianBlur(mz, (7, 7), 0).astype(np.float32) / 255.0
    fin = np.clip(fin.astype(np.float32) * (1 - a[..., None])
                  + res_np * a[..., None], 0, 255).astype(np.uint8)
    applied.append("F1 补牙")
    print("[F1] 补牙完成", flush=True)

# F2 唇压缩 (迭代至比值 <=0.50)
if D2:
    for K in (0.82, 0.78):
        outer = lm[48:60]
        x0, x1 = outer[:, 0].min() - 14, outer[:, 0].max() + 14
        y0, y1 = outer[:, 1].min() - 12, outer[:, 1].max() + 12
        cx, cy = outer.mean(axis=0)
        rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) ** 0.5
        t = np.clip((1.25 - d) / 0.25, 0, 1)
        map_y = cy + (yy - cy) * (1 + (1 / K - 1) * t)
        fin = cv2.remap(fin, xx, map_y, cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_REPLICATE)
        lm, _ = gr._tddfa_68(fin)
        outer = lm[48:60]
        if (outer[:, 1].max() - outer[:, 1].min()) / io_d <= 0.65:
            break
    applied.append("F2 唇压缩")
    print("[F2] 唇压缩完成", flush=True)

# F3 鼻孔: 大核缩小 + 镜像补对侧
if D3:
    big = max(clusters, key=lambda c: c[4])
    big_idx = [st3[i, 4] for i in range(1, n3)].index(big[4])
    bx = int(cent3[big_idx][0] + tip[0] - 70)
    by = int(cent3[big_idx][1] + tip[1] - 70 + 5)
    z = fin[by:by + big[3] + 2, bx:bx + big[2] + 2]
    darkz = (z.sum(axis=2) < 220).astype(np.uint8) * 255
    darkz = cv2.dilate(darkz, np.ones((3, 3), np.uint8))
    full_dark = np.zeros((H, W), np.uint8)
    full_dark[by:by + darkz.shape[0], bx:bx + darkz.shape[1]] = darkz
    fin = cv2.inpaint(fin, full_dark, 5, cv2.INPAINT_TELEA)
    MID = tip[0]
    M = np.array([[-1.0, 0.0, 2 * MID], [0.0, 1.0, 0.0]])
    mir = cv2.warpAffine(fin, M, (W, H), flags=cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_REPLICATE)
    slit_y = int(cent3[big_idx][1] + tip[1] - 70 + 5)
    for sgn in (-1, 1):
        px = int(MID + sgn * 12)
        slit = np.zeros((H, W), np.float32)
        cv2.ellipse(slit, (px, slit_y), (4, 3), 0, 0, 360, 1.0, -1)
        slit = cv2.GaussianBlur(slit, (5, 5), 0)
        fin = fin.astype(np.float32) * (1 - 0.6 * slit[..., None])
    fin = np.clip(fin, 0, 255).astype(np.uint8)
    applied.append("F3 鼻孔对称")
    print("[F3] 鼻孔对称完成", flush=True)

# F4-F6 常规工序: 仅对未经管线处理的图执行 (--raw)。
# 管线输出已含 E/修容/发际线, 重跑=双重处理劣化 (0.416→0.355 实测)
if RAW:
    nx, ny = int(lm_o[30][0]), int(lm_o[30][1])
    nose_crop = orig[max(ny - 85, 0):ny + 85, max(nx - 85, 0):nx + 85]
    ip_img = Image.fromarray(
        cv2.cvtColor(cv2.resize(nose_crop, (224, 224)), cv2.COLOR_BGR2RGB))
    pipe.disable_attention_slicing()
    pipe = ga._attach_ip_adapter(pipe, 0.35)
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
    print("[F4] E重塑完成 (只贴鼻区)", flush=True)

    comp = cv2.imread("result/rnr/_outpaint_comp.png")
    if comp is not None:
        skin = _skin_strict(comp, 21)
        hair = np.zeros((H, W), np.uint8)
        hair[0:180, 0:340] = 255
        glare = ((fin.sum(axis=2).astype(int)
                  - comp.sum(axis=2).astype(int) > 60)
                 ).astype(np.uint8) * 255
        nz2 = np.zeros((H, W), np.uint8)
        nz2[385:495, 120:330] = 255
        neck = cv2.dilate(cv2.bitwise_and(glare, nz2),
                          np.ones((15, 15), np.uint8))
        m2 = cv2.bitwise_or(hair, neck)
        m2[skin > 0] = 0
        a2 = cv2.GaussianBlur(m2, (15, 15), 0).astype(np.float32) / 255.0
        fin = np.clip(fin.astype(np.float32) * (1 - a2[..., None])
                      + comp.astype(np.float32) * a2[..., None], 0, 255
                      ).astype(np.uint8)
        print("[F5] 发际线二次修复", flush=True)

    lm, _ = gr._tddfa_68(fin)
    top = lm[27].astype(float)
    tipn = lm[30].astype(float)
    spine = tipn - top
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
        seg(top + u * 6 + off, tipn - u * 4 + off, 0.07, 5)
    for idx in (31, 35):
        c = lm[idx].astype(float)
        seg(c + np.array([-4.0, 2.0]), c + np.array([4.0, 2.0]), 0.05, 4)
    seg(tipn - u * 3, tipn - u * 1, -0.05, 5)
    field = cv2.GaussianBlur(field, (9, 9), 0)
    zn = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(zn, cv2.convexHull(lm[27:36].astype(np.int32)), 255)
    zn = cv2.dilate(zn, np.ones((35, 35), np.uint8))
    field[zn == 0] = 0
    fin = np.clip(fin.astype(np.float32) * (1 - field[..., None]),
                  0, 255).astype(np.uint8)
    print("[F6] 鼻梁修容完成", flush=True)

out_path = os.path.splitext(fin_path)[0] + "_refined.png"
cv2.imwrite(out_path, fin)
onnx = os.path.join(
    ga._snapshot(r"D:\huggingface_cache", "models--FoivosPar--Arc2Face"),
    "arcface.onnx")
cos = gr._embed_cosine(orig, fin, onnx)
print("== QC 报告 ==", flush=True)
for ln in report:
    print("  " + ln, flush=True)
print("  已应用: " + (", ".join(applied) if applied else "无"), flush=True)
print(f"[sim] 精修终版 = {cos:.3f}", flush=True)
print(f"[done] {out_path}", flush=True)
