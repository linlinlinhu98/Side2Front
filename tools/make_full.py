# 全帧合成 v2 (方案C, 原图逐像素保真路线):
# 1) 512 正脸逆变换投回原图坐标系, 只贴"头"(脸+发紧掩码)
# 2) 侧脸头残影区(原图头部皮肤+暗发, 下颌线以上, 扣除正脸头区) telea 从
#    周围散景填补——原图背景一点不动
import importlib.util
import os
_HFC = os.environ.get("HF_HOME", r"D:\huggingface_cache")  # 模型缓存根目录, 可用环境变量 HF_HOME 覆盖

import cv2
import numpy as np
from skimage import transform as trans


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _skin_strict(im, dilate):
    ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2]
    sk = ((cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
          & (y > 110)).astype(np.uint8) * 255
    if dilate:
        sk = cv2.dilate(sk, np.ones((dilate, dilate), np.uint8))
    return sk


gr = _load("gen_rnr", "tools/gen_rnr.py")
ga = _load("gen_arc2face", "tools/gen_arc2face.py")

orig = cv2.imread("4/people.jpg")
H0, W0 = orig.shape[:2]
lm68, _ = gr._tddfa_68(orig)
S = 256
t5 = np.stack([lm68[36:42].mean(axis=0), lm68[42:48].mean(axis=0),
               lm68[31], lm68[48], lm68[54]]).astype(np.float32)
src = gr.ARCFACE5 * 290 / 112
src[:, 0] += 50
src[:, 1] += 60
src = src / 400 * S
tform = trans.SimilarityTransform()
tform.estimate(t5, src)
M512 = tform.params[0:2, :] * 2
Minv = cv2.invertAffineTransform(M512)

import sys as _sys
fin_path = _sys.argv[1] if len(_sys.argv) > 1 else "result/rnr/rnr_frontal.png"
fin = cv2.imread(fin_path)
# (v10 局部修斑已证伪: 压暗+模糊把斑块打成棕泥且掉分, 已撤;
#  正确解法是管线 A 版发际线重跑, 见 gen_rnr._restore_real dilate=21)

# ---- 正脸头掩码(512系): 皮肤凸包 ∪ 上半暗发, 膨胀20 ----
sk_fin = _skin_strict(fin, 0)
ys, xs = np.where(sk_fin > 0)
hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))
head512 = np.zeros(fin.shape[:2], np.uint8)
cv2.fillConvexPoly(head512, hull, 255)
dark = (fin.sum(axis=2) < 380).astype(np.uint8) * 255
dark[260:] = 0                       # 暗色只取上半(头发区)
head512 = cv2.bitwise_or(head512, dark)
head512 = cv2.morphologyEx(head512, cv2.MORPH_CLOSE,
                           np.ones((9, 9), np.uint8))
head512 = cv2.dilate(head512, np.ones((9, 9), np.uint8))
# 下颌裙边: 掩码底边以下 70px 线性渐隐, 下巴到真脖子长交叠
# (用户反馈拼接生硬; 侧缘/发缘仍由后续羽化保持自然轮廓)
a = head512.astype(np.float32) / 255.0
D = 70
for x in range(head512.shape[1]):
    ys = np.where(head512[:, x] > 0)[0]
    if not len(ys):
        continue
    yb = ys.max()
    for d in range(1, D):
        if yb + d < a.shape[0]:
            a[yb + d, x] = max(a[yb + d, x], 1.0 - d / D)
# 下颌曲线钳制 v2: 真实下颌折线先平滑(折线弦也会显形),
# 边界贴曲线+6, 下方 28px 短渐隐——把真脖子尽早还给原图
lm_fin, _ = gr._tddfa_68(fin)
jaw = lm_fin[0:17]
jx, jy = jaw[:, 0], jaw[:, 1]
xs_all = np.arange(a.shape[1], dtype=np.float32)
yb_raw = np.interp(xs_all, jx, jy)
yb_curve = cv2.GaussianBlur(yb_raw.reshape(1, -1), (1, 31), 0).ravel() + 6
D = 45                                     # 渐隐拉长+羽化加宽, 消灭线条感
for x in range(int(jx.min()), int(jx.max()) + 1):
    yb = int(yb_curve[x])
    for d in range(yb + 1, min(yb + D, a.shape[0])):
        a[d, x] = min(a[d, x], max(0.0, 1.0 - (d - yb) / D))
a512 = np.clip(cv2.GaussianBlur(a, (25, 25), 0), 0, 1)
# 画布边框渐隐: 512 画布逆投后是旋转矩形, 其底边斜切她的脖子形成
# 硬线(实测); 边框 25px 内渐隐到 0——该处原图本就有等价真实内容,
# 露出原图即无缝
B = 25
fade = np.ones_like(a512)
fade[:B] *= np.linspace(0, 1, B)[:, None]
fade[-B:] *= np.linspace(1, 0, B)[:, None]
fade[:, :B] *= np.linspace(0, 1, B)[None, :]
fade[:, -B:] *= np.linspace(1, 0, B)[None, :]
a512 = a512 * fade
cv2.imwrite("result/_alpha_dbg.png", (a512 * 255).astype(np.uint8))

head = cv2.warpAffine(fin, Minv, (W0, H0), borderValue=(0, 0, 0))
alpha = cv2.warpAffine(a512, Minv, (W0, H0), borderValue=0.0)
# 全分辨率空间再平滑: 512 系掩码的整像素台阶逆投放大后成块状半透明
# 边(用户反馈"重重叠叠"), 在输出尺度上抹平
alpha = cv2.GaussianBlur(alpha, (41, 41), 0)

# ---- 侧脸残影掩码(原图系): 只取含鼻根种子的皮肤连通域(脸+脖),
# 下颌尖以上, 扣正脸头区。手是独立皮肤连通域自动排除(v2 误吞事故);
# 不碰任何头发——fin 的发丝本来就是原发回贴, 位置自洽 ----
sk_orig = _skin_strict(orig, 0)
n, lab, st, _ = cv2.connectedComponentsWithStats(sk_orig)
seed_x, seed_y = int(lm68[27][0]), int(lm68[27][1])
face_comp = lab[seed_y, seed_x]
ghost = ((lab == face_comp) & (face_comp > 0)).astype(np.uint8) * 255
chin_y = int(lm68[8][1]) - 20
ghost[chin_y:] = 0                   # 下颌以下真脖子必须保留
ghost[alpha > 0.5] = 0               # 正脸头区不补
ghost = cv2.dilate(ghost, np.ones((9, 9), np.uint8))
ghost[alpha > 0.5] = 0
print(f"[ghost] 残影填补区 {int((ghost > 0).sum())}px", flush=True)

filled = cv2.inpaint(orig, ghost, 8, cv2.INPAINT_TELEA)

out = filled.astype(np.float32) * (1 - alpha[..., None]) \
    + head.astype(np.float32) * alpha[..., None]

# 边界焙烧: 沿下颌界上方 30px 渐强压暗 16% (对接界下真脖的自然颌下
# 阴影, 消灭"苍白板撞暖脖"的断层)
burn512 = np.zeros(fin.shape[:2], np.float32)
for x in range(int(jx.min()), int(jx.max()) + 1):
    yb = int(yb_curve[x])
    for d in range(30):
        if 0 <= yb - d:
            burn512[yb - d, x] = 0.16 * (1 - d / 30)
    for d in range(1, 15):                 # 界下真脖侧也轻压, 两侧同步过渡
        if yb + d < burn512.shape[0]:
            burn512[yb + d, x] = max(burn512[yb + d, x],
                                     0.08 * (1 - d / 15))
burn512 = cv2.GaussianBlur(burn512, (7, 7), 0)
burn = cv2.warpAffine(burn512, Minv, (W0, H0), borderValue=0.0)
out = out.astype(np.float32) * (1 - burn[..., None])
out = np.clip(out, 0, 255).astype(np.uint8)

# ---- 融合一体处理(用户核心诉求): 两项 ----
# 实测差距: 生成脸 166亮度/3.1颗粒 vs 原图皮肤 193/12.6 (灰暗+塑胶)
# T1 低频色调贴合: 向原图同位置皮肤的低频亮度场靠拢(力度0.6, 防过冲)
# T2 颗粒移植: 原图高频残差按 0.8 注入, 消灭塑胶感(零均值, 安全)
orig_f = orig.astype(np.float32)
out_f = out.astype(np.float32)
skin_o = _skin_strict(orig, 9) > 0
# v13: 融合权重连续化——w=alpha*(1-alpha) 边界中央最强, 两侧自然归零,
# 无内环(v12 硬内环读作一圈痕迹); 脸芯(w→0)身份像素不动
w = np.clip(alpha * (1 - alpha) * 4.0, 0, 1) ** 0.75
band = (w > 0.05) & skin_o
if band.any():
    lo_o = cv2.GaussianBlur(orig_f, (65, 65), 0)
    lo_g = cv2.GaussianBlur(out_f, (65, 65), 0)
    tone_gain = (lo_o - lo_g) * 0.45
    resid = orig_f - cv2.GaussianBlur(orig_f, (5, 5), 0)
    r_ch, g_ch = out_f[:, :, 2], out_f[:, :, 1]
    lips = (r_ch > 150) & (r_ch > g_ch + 25)
    grain_map = 0.5 * (~lips).astype(np.float32)
    adj = tone_gain + resid * grain_map[..., None]
    out_f = out_f + adj * (w * band)[..., None].astype(np.float32)
    print(f"[fuse] 连续权重边界融合 {int(band.sum())}px", flush=True)
out = np.clip(out_f, 0, 255).astype(np.uint8)

# ---- 肤色连续性: 贴区脸(SD+CF 工序后偏苍白哑光)向真实脖领色调对齐,
# Reinhard 各通道, mix=0.35 (贴区内 skin & alpha>0.8 的像素) ----
sk_out = _skin_strict(out, 0)
neck_band = np.zeros((H0, W0), bool)
neck_band[chin_y + 10:chin_y + 160] = True
ref_px = out[neck_band & (sk_out > 0) & (alpha < 0.2)]
face_sel = (sk_out > 0) & (alpha > 0.8)
face_px = out[face_sel]
TONE_MIX = 0.0   # 肤色对齐力度: 0.35 已证伪(下颌泥斑+掉分0.039), 0=关闭
if TONE_MIX > 0 and len(ref_px) > 500 and len(face_px) > 500:
    ref_lab = cv2.cvtColor(ref_px.reshape(-1, 1, 3),
                           cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
    face_lab = cv2.cvtColor(face_px.reshape(-1, 1, 3),
                            cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
    rm, rs = ref_lab.mean(0), ref_lab.std(0)
    fm, fs = face_lab.mean(0), face_lab.std(0)
    new = (face_lab - fm) * ((rs + 1e-6) / (fs + 1e-6)) + rm
    mix = TONE_MIX
    new = mix * new + (1 - mix) * face_lab
    new_bgr = cv2.cvtColor(np.clip(new, 0, 255).astype(np.uint8)
                           .reshape(-1, 1, 3), cv2.COLOR_Lab2BGR).reshape(-1, 3)
    out[face_sel] = new_bgr
    print(f"[tone] 脸->脖领肤色对齐 mix={mix} "
          f"({len(face_px)}px, ref {len(ref_px)}px)", flush=True)

cv2.imwrite("assets/final/people_full.png", out)
cv2.imwrite("result/_full_view.png", cv2.resize(out, (720, 960)))
cv2.imwrite("result/_full_ghost.png", ghost)

onnx = os.path.join(
    ga._snapshot(_HFC, "models--FoivosPar--Arc2Face"),
    "arcface.onnx")
cos = gr._embed_cosine(orig, out, onnx)
print(f"[sim] 全帧v2 vs 原图 arcface 余弦 = {cos:.3f} (512 版 0.476)",
      flush=True)
