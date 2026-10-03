r"""Rotate-and-Render (CVPR2020) CPU 移植推理。

原版依赖: neural_renderer(CUDA) + 多进程 + 多GPU。本脚本:
  - 渲染器用 OpenCV 画家算法软件光栅化替代（原渲染是每面片
    平均顶点色的无光照平色渲染, 语义可精确复刻）
  - 3DDFA 参数复用仓库已装的 3ddfa_v2 (同 62 维参数格式)
  - GAN (rotatespade) 权重原版加载, CPU float32

前置: D:\huggingface_cache\rnr_code 已克隆;
      rnr_code/checkpoints/ 下有 *_net_G.pth (唯一还需下载的权重)。
      BFM 网格/三角面直接复用本地 3ddfa_v2 的 bfm_noneck_v3.pkl
      与 tri.pkl, 不需要官方的 ckpt_and_bfm.zip。

用法 (repo root):
    python tools/gen_rnr.py 4/people.jpg [--out result/rnr.png]
"""
import argparse
import importlib.util
import os
_HFC = os.environ.get("HF_HOME", r"D:\huggingface_cache")  # 模型缓存根目录, 可用环境变量 HF_HOME 覆盖
import sys
import time
import types

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNR = os.path.join(_HFC, "rnr_code")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "3ddfa_v2"))
sys.path.insert(0, RNR)

# neural_renderer 只有 CUDA 内核; 我们不实例化它, 先占位让
# models.networks.render 能 import(只用它的纯 numpy 数学函数)
_nr = types.ModuleType("neural_renderer")


class _StubRenderer:
    def __init__(self, *a, **k):
        raise RuntimeError("CPU 移植版不使用 neural_renderer")


_nr.Renderer = _StubRenderer
sys.modules.setdefault("neural_renderer", _nr)

import skimage.transform as trans  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "rnr_curve", os.path.join(RNR, "data", "curve.py"))
curve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_spec and curve)

_spec2 = importlib.util.spec_from_file_location(
    "rnr_render", os.path.join(RNR, "models", "networks", "render.py"))
rrender = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(rrender)  # P2sRt / matrix2angle / angle2matrix

ARCFACE5 = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


class CPURender:
    """render.Render 的 CPU 等价实现（frontal 分支, batch=1）。

    注意: 不读 rnr_code/3ddfa/train.configs(那是 v1 的 53215 顶点网格,
    与 v2 的 62 维参数不兼容), 直接复用本地 3ddfa_v2 的
    bfm_noneck_v3.pkl(38365 顶点)——它与生成参数的 TDDFA 天然配套。
    """

    def __init__(self, render_size=256):
        import pickle
        self.render_size = render_size
        self.std_size = 120
        d3 = os.path.join(ROOT, "3ddfa_v2", "configs")
        with open(os.path.join(d3, "bfm_noneck_v3.pkl"), "rb") as f:
            bfm = pickle.load(f)
        # v2 布局: (3N,) 分块 [xs|ys|zs], reshape 用 order='F'
        self.u = np.asarray(bfm["u"], np.float32)                # (3N,1)
        self.w_shp = np.asarray(bfm["w_shp"], np.float32)[:, :40]
        self.w_exp = np.asarray(bfm["w_exp"], np.float32)[:, :10]
        kps = np.asarray(bfm["keypoints"]).reshape(-1).astype(np.int64)
        # keypoints 是交错三联索引 [3i, 3i+1, 3i+2, ...] (同 v1 的
        # keypoints_sim 格式, 顶点向量为交错布局), 还原顶点索引:
        self.lm68_idx = kps.reshape(-1, 3)[:, 0] // 3  # (68,)
        # v2 官方对 bfm_noneck_v3 用重建过的 tri.pkl
        with open(os.path.join(d3, "tri.pkl"), "rb") as f:
            tri = np.asarray(pickle.load(f)).astype(np.int64)
        if tri.shape[0] != 3:
            tri = tri.T
        n_vert = self.u.size // 3
        # 基准判定不能看 min(顶点 0 可能未被引用): 1-based 时
        # max==N, 0-based 时 max==N-1
        if tri.max() == n_vert:
            tri = tri - 1
        assert tri.max() == n_vert - 1 and tri.min() >= 0, \
            f"tri 索引异常: min={tri.min()} max={tri.max()} N={n_vert}"
        print(f"[render] 顶点 {n_vert}, 面片 {tri.shape[1]}", flush=True)
        self.faces = tri.T  # (F,3)

    # ---- 参数 -> 顶点 (图像坐标, std 流程与原版一致) ----
    def _vertices(self, param, frontal):
        p_ = param[:12].reshape(3, -1)
        p = p_[:, :3]
        s, R, t3d = rrender.P2sRt(p_)
        angle = rrender.matrix2angle(R)
        original_angle = angle[0]
        if frontal:
            angle[0] = 0.0
            if angle[1] < 0:
                angle[1] = 0.0
            p = rrender.angle2matrix(angle) * s
        offset = p_[:, -1].reshape(3, 1)
        alpha_shp = param[12:52].reshape(-1, 1)
        alpha_exp = param[52:-4].reshape(-1, 1)
        box = param[-4:]
        v = p @ (self.u + self.w_shp @ alpha_shp
                 + self.w_exp @ alpha_exp).reshape(3, -1, order="F") + offset
        v[1, :] = self.std_size + 1 - v[1, :]
        sx, sy, ex, ey = box
        v[0, :] = v[0, :] * (ex - sx) / 120 + sx
        v[1, :] = v[1, :] * (ey - sy) / 120 + sy
        v[2, :] = v[2, :] * ((ex - sx) + (ey - sy)) / 2 / 120
        return v, original_angle  # (3,N) 图像坐标(120 基)

    def _align_M(self, five_pts):
        src = ARCFACE5 * 290 / 112
        src[:, 0] += 50
        src[:, 1] += 60
        src = src / 400 * self.render_size
        tform = trans.SimilarityTransform()
        tform.estimate(five_pts.astype(np.float32), src)
        return tform.params[0:2, :]

    @staticmethod
    def _apply_M(M, v):
        out = v.copy()
        out[:2] = M[:2, :2] @ v[:2] + M[:2, 2:3]
        return out

    def _five_points(self, v):
        """v1 用固定顶点索引(仅适配 53215 网格); 改为由 68 点合成,
        与 test_frontal.landmark_68_to_5 一致(鼻尖取 31)。"""
        lm = v[:2, self.lm68_idx]  # (2,68)
        return np.stack([lm[:, 36:42].mean(axis=1),
                         lm[:, 42:48].mean(axis=1),
                         lm[:, 31], lm[:, 48], lm[:, 54]])

    def _sample_colors(self, img, v):
        """img: (3,H,W) torch float; v: (3,N) numpy 像素坐标
        -> (N,3) torch 顶点色"""
        _, h, w = img.shape
        xs = np.clip(np.round(v[0]).astype(np.int64), 0, w - 1)
        ys = np.clip(np.round(v[1]).astype(np.int64), 0, h - 1)
        cols = img[:, torch.from_numpy(ys), torch.from_numpy(xs)]
        return cols.t().contiguous()  # (N,3)

    def _raster(self, v_px, colors):
        """画家算法软光栅。v_px: (3,N) x 右 y 下; colors: (N,3) 顶点色
        (torch)。每面片取 3 顶点色均值(同原版
        texture_vertices_to_faces), 按面片平均深度远先近后绘制。
        返回 render (3,H,W) float, mask (H,W) uint8。"""
        S = self.render_size
        canvas = np.zeros((S, S, 3), np.float32)
        mask = np.zeros((S, S), np.uint8)
        pts = v_px[:2].T                       # (N,2)
        f = self.faces                         # (F,3)
        zf = v_px[2][f].mean(axis=1)           # 每面片平均深度
        cf = colors[torch.from_numpy(f)].mean(dim=1)  # (F,3) 面片色
        order = np.argsort(-zf)                # 远(z大)先画
        f = f[order]
        cf = cf[order].numpy()
        for i in range(f.shape[0]):
            tri3 = pts[f[i]].astype(np.int32)
            c = tuple(int(x * 255) for x in cf[i])
            cv2.fillPoly(canvas, [tri3], c)
            cv2.fillPoly(mask, [tri3], 255)
        render = torch.from_numpy(canvas.transpose(2, 0, 1)) / 255.0
        return render, mask

    @staticmethod
    def _erode(mask, k):
        return cv2.erode(mask, np.ones((k, k), np.uint8))

    debug_dir = None

    def _save_dbg(self, img3, name):
        arr = (img3.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.debug_dir, name),
                    cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    def rotate_render(self, param, image, M):
        """image: (3,S,S) 0-1 对齐图; M: 2x3 数据对齐阵。
        返回 rotated_mesh (3,S,S) 0-1, landmarks68 (1,68,3) 0-1坐标张量,
        original_angle 弧度。"""
        S = self.render_size
        v_t, ang = self._vertices(param, frontal=True)   # 目标正脸
        v_o, _ = self._vertices(param, frontal=False)    # 原始姿态
        v_o = self._apply_M(M, v_o)                      # -> 对齐图像素域

        # 顶点色: 从对齐图采样(原姿态位置)
        v_o_px = v_o.copy()
        v_o_px[:2] = v_o_px[:2] / S * S
        colors = self._sample_colors(image, v_o_px)

        # 1) 原姿态渲染 -> 腐蚀 -> 均值背景
        r0, m0 = self._raster(v_o_px, colors)
        if self.debug_dir:
            self._save_dbg(r0, "_rnr_r0.png")
        m0e = self._erode(m0, 15)
        me = (m0e > 0)
        if me.sum() > 0:
            bg = r0[:, me].mean(dim=1, keepdim=True)
        else:
            bg = torch.full((3, 1), 0.5)
        r0e = r0.clone()
        r0e[:, ~me] = bg.expand(3, (~me).sum())
        if self.debug_dir:
            self._save_dbg(r0e, "_rnr_r0e.png")
            print(f"[render] 原姿态掩码覆盖 {me.mean():.1%}",
                  flush=True)
        # 2) 腐蚀后的渲染重新采样为纹理(去边界渗透)
        colors2 = self._sample_colors(r0e, v_o_px)
        # 3) 目标姿态渲染(测试路径: 原始渲染, 无 erode/背景后处理)
        v_t2 = self._apply_M(self._align_M(self._five_points(v_t)), v_t)
        mesh, m1 = self._raster(v_t2, colors2)
        # 口腔/鼻孔等不可见区域在网格上是纯黑洞, img2img 会把纯黑
        # 发挥成"大张嘴"。原图嘴唇是微张露上齿 —— 把洞填成"比唇色
        # 暗的口腔色"(环带采样唇色 × 0.4), 底图即"微张的嘴"
        mesh8 = (mesh.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        hole = ((m1 > 0) & (mesh8.max(axis=2) < 10)).astype(np.uint8) * 255
        if hole.sum() > 0:
            dil = cv2.dilate(hole, np.ones((9, 9), np.uint8))
            ring = (dil > 0) & (hole == 0) & (m1 > 0)
            if ring.sum() > 0:
                base = (mesh8[ring].mean(axis=0) * 0.4).astype(np.uint8)
            else:
                base = np.array([40, 20, 22], np.uint8)
            mesh8[hole > 0] = base
            mesh = torch.from_numpy(
                mesh8.transpose(2, 0, 1)).float() / 255.0
            print(f"[render] 口腔洞填暗色 {hole.sum() // 255}px "
                  f"(base={base.tolist()})", flush=True)
        # 4) 目标姿态 68 关键点(像素坐标)
        lm68 = v_t2.T[self.lm68_idx]  # (68,3)
        lm = torch.from_numpy(lm68.astype(np.float32))[None]
        return mesh, lm, float(ang)


def _opt_namespace(crop_size=256):
    import argparse
    opt = argparse.Namespace(
        name="rs_model", gpu_ids=[], checkpoints_dir=os.path.join(
            RNR, "checkpoints"), isTrain=False, phase="test",
        model="rotatespade", netG="rotatespade",
        norm_G="spectralsyncbatch", crop_size=crop_size, aspect_ratio=1.0,
        label_nc=5, contain_dontcare_label=False, output_nc=3,
        no_instance=True, nef=16, ngf=64, init_type="xavier",
        init_variance=0.02, resnet_n_downsample=4, resnet_n_blocks=9,
        resnet_kernel_size=3, resnet_initial_kernel_size=7,
        semantic_nc=5, no_gaussian_landmark=True, label_mask=True,
        heatmap_size=2.5, erode_kernel=21, verbose=False,
        which_epoch="latest")
    return opt


def _find_netG():
    ckpt_dir = os.path.join(RNR, "checkpoints")
    cands = []
    if os.path.isdir(ckpt_dir):
        for dp, _dn, fn in os.walk(ckpt_dir):
            cands += [os.path.join(dp, f) for f in fn
                      if f.endswith("_net_G.pth")]
    return cands


def _load_generator(opt):
    import models.networks as networks
    cls = networks.find_network_using_name(opt.netG, "generator")
    netG = cls(opt)
    cands = _find_netG()
    assert cands, f"在 {os.path.join(RNR, 'checkpoints')} 下找不到 *_net_G.pth"
    sd = torch.load(cands[0], map_location="cpu", weights_only=False)
    sd = { (k[7:] if k.startswith("module.") else k): v
           for k, v in sd.items() }
    try:
        netG.load_state_dict(sd)
        print(f"[netG] strict load: {os.path.basename(cands[0])}",
              flush=True)
    except RuntimeError:
        md = netG.state_dict()
        sd2 = {k: v for k, v in sd.items()
               if k in md and md[k].size() == v.size()}
        netG.load_state_dict(sd2, strict=False)
        miss = {k.split(".")[0] for k in md if k not in sd2}
        print(f"[netG] partial load, 未覆盖: {sorted(miss)}", flush=True)
    netG.eval()
    return netG


def _seg_map(lm68, no_gaussian, size, original_angle):
    """复刻 RotateSPADEModel.get_seg_map (batch=1, CPU)。"""
    pts = lm68[:, :, :2].cpu().numpy().astype(np.float32)
    heatmap = curve.points_to_heatmap_68points(pts[0], 13, size,
                                               2.5)
    heatmap2 = curve.combine_map(heatmap, no_gaussian=no_gaussian)
    if abs(original_angle) < 0.255:
        heatmap = np.zeros_like(heatmap)
    seg = torch.from_numpy(heatmap2.astype(np.float32))[None]
    seg = seg.permute(0, 3, 1, 2)
    seg_all = torch.from_numpy(heatmap.astype(np.float32))[None]
    seg_all = seg_all.permute(0, 3, 1, 2)
    seg_all[seg_all > 0] = 2.0
    return seg, seg_all


def _arc2face_module():
    """复用 tools/gen_arc2face.py 里已验证的管线组件。"""
    spec = importlib.util.spec_from_file_location(
        "gen_arc2face", os.path.join(ROOT, "tools", "gen_arc2face.py"))
    ga = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ga)
    # gen_arc2face 以脚本方式运行时在 main() 里 global import torch,
    # 作为模块加载时补注入, 否则 _load_pipeline 等函数 NameError
    ga.torch = torch
    return ga


def _embed_cosine(img_a_bgr, img_b_bgr, onnx):
    """两图 arcface 余弦(同 _arc2face_sim.py 逻辑)。"""
    import onnxruntime as ort
    ga = _arc2face_module()
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])

    def embed(image):
        lm68 = ga._landmarks_68(image)
        M, _ = cv2.estimateAffinePartial2D(
            ga._five_points(lm68), ga.ARCFACE_DST, method=cv2.LMEDS)
        aligned = cv2.warpAffine(image, M, (112, 112),
                                 borderValue=(114, 114, 114))
        x = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = ((x - 127.5) / 127.5).transpose(2, 0, 1)[None]
        e = sess.run(None, {sess.get_inputs()[0].name: x})[0].reshape(-1)
        return e / np.linalg.norm(e)

    return float(embed(img_a_bgr) @ embed(img_b_bgr))


def _project_with_features(pipe, face_embs, features):
    """ga._project_face_embs 的扩展版: 模板从固定的
    'photo of a id person' 变为 'photo of a id person, <特征>',
    id token 位置照常注入身份向量, 特征文字参与条件编码。"""
    from transformers.masking_utils import create_causal_mask
    tok = pipe.tokenizer
    token_id = tok.encode("id", add_special_tokens=False)[0]
    text = f"photo of a id person, {features}"
    input_ids = tok(text, truncation=True, padding="max_length",
                    max_length=tok.model_max_length,
                    return_tensors="pt").input_ids
    enc = pipe.text_encoder
    token_embs = enc.embeddings.token_embedding(input_ids)
    padded = torch.nn.functional.pad(
        face_embs, (0, enc.config.hidden_size - 512))
    token_embs[input_ids == token_id] = padded
    hidden = enc.embeddings(input_ids=input_ids, inputs_embeds=token_embs)
    mask = create_causal_mask(config=enc.config, inputs_embeds=hidden,
                              attention_mask=None, past_key_values=None)
    out = enc.encoder(inputs_embeds=hidden, attention_mask=mask,
                      is_causal=True)
    return enc.final_layer_norm(out.last_hidden_state)


def _skin_mask(rgb):
    """经典 YCrCb 皮肤阈值 (输入 RGB, 返回 0/255 uint8)"""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)  # 通道序: Y, Cr, Cb
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    return ((cb >= 77) & (cb <= 127) &
            (cr >= 133) & (cr <= 173)).astype(np.uint8) * 255


def _match_skin_tone(face_np, base_rgb, core, mix=1.0):
    """Reinhard 颜色迁移: 把第一遍冷白的脸芯向原图真实肤色对齐。
    Lab 空间按均值/方差匹配, 两侧都只统计皮肤色像素(眼睛/嘴唇/
    头发/绿背景不进统计量); 目标 = warp 底图里她的真脸/脖子/手臂
    (同一人同一光线)。纯确定性, 不引入随机性。
    mix: 迁移力度 (1.0=全迁移; 全力度 sim 0.391 vs 不迁移 0.410,
    用户选了折中 0.5)"""
    src_skin = _skin_mask(face_np) & core
    tgt_skin = _skin_mask(base_rgb)
    if (src_skin > 0).sum() < 500 or (tgt_skin > 0).sum() < 500:
        print("[肤色] 皮肤样本不足, 跳过颜色迁移", flush=True)
        return face_np
    src_lab = cv2.cvtColor(face_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = src_lab.copy()
    for c in range(3):
        s = src_lab[:, :, c][src_skin > 0]
        t = tgt_lab[:, :, c][tgt_skin > 0]
        out[:, :, c] = ((src_lab[:, :, c] - s.mean())
                        * (t.std() / (s.std() + 1e-6)) + t.mean())
    if mix < 1.0:
        out = mix * out + (1.0 - mix) * src_lab
    out = np.clip(out, 0, 255).astype(np.uint8)
    print(f"[肤色] Reinhard 迁移(mix={mix}): 源皮肤 "
          f"{(src_skin > 0).sum()}px, 目标皮肤 {(tgt_skin > 0).sum()}px",
          flush=True)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def _outpaint_surroundings(inpaint, face_img, orig_bgr, align_M,
                           pos, neg, seed, guidance, dbg_dir):
    """第二遍: 以原图为底（真实旗袍/手臂/绿色背景全保留），正脸脸芯
    冻结，inpaint 重画头发/脖子/接缝——把生成的脸"缝"到她真的身体
    上（用户的点子，替代黑底 outpaint：身体不用 SD 瞎编）。
    face_img: 第一遍输出(黑底正脸);
    align_M: 主流程的 5 点相似变换(原图->256 正脸坐标系, ×2 到 512)
    ——网格纹理本来就是用它从原图采样的, 几何上天然自洽。
    (曾用鼻尖+下巴两点自算对齐: 侧脸投影压缩鼻颏距+鼻尖位置语义
    不同导致缩放和定位全错, 已证伪)"""
    from PIL import Image
    S2 = 512
    face_np = np.asarray(face_img.convert("RGB"))
    M512 = np.asarray(align_M, np.float64) * 2  # 原图 -> 512 正脸坐标系
    base = cv2.warpAffine(orig_bgr, M512, (S2, S2), borderValue=(0, 0, 0))
    cv2.imwrite(os.path.join(dbg_dir, "_outpaint_base.png"), base)
    base_rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)

    # 冻结脸芯: 第一遍输出的高亮区域取最大连通域
    gray = face_np.max(axis=2)
    face = (gray > 25).astype(np.uint8) * 255
    face = cv2.morphologyEx(face, cv2.MORPH_CLOSE,
                            np.ones((25, 25), np.uint8))
    # 只保留最大连通域(脸), 去掉零星噪点
    n, lab, stats, _ = cv2.connectedComponentsWithStats(face)
    if n > 1:
        face = (lab == 1 + np.argmax(stats[1:, 4])).astype(np.uint8) * 255
    # 腐蚀 51 冻得更小(只保眼/鼻/嘴/脸颊内圈): 太阳穴/发际线/下颌
    # 轮廓交给环带重画, SD 才能长出自然发际线盖住边缘——上轮冻核
    # 太大(31), 正脸椭圆轮廓整个贴上去, 太阳穴皮肤凸在她的真发上,
    # 用户指出"脸没有完全融合, 有突出的部分"
    core = cv2.erode(face, np.ones((51, 51), np.uint8))
    # 贴片前先把脸芯肤色向原图对齐(力度 50%: 全力度 0.391 vs 不迁移
    # 0.410, 用户选了折中)
    face_np = _match_skin_tone(face_np, base_rgb, core, mix=0.5)
    # 羽化贴片边界(~10px 过渡带), 消灭硬接缝
    alpha = cv2.GaussianBlur(core, (21, 21), 0).astype(np.float32) / 255.0
    base_rgb = (base_rgb * (1.0 - alpha[..., None])
                + face_np * alpha[..., None]).astype(np.uint8)
    Image.fromarray(base_rgb).save(
        os.path.join(dbg_dir, "_outpaint_comp.png"))
    # 只重画脸周一圈环带(头发/脖子/接缝), 环带之外冻结为原图真实像素。
    # (上一版掩码=脸芯以外全部: 真实 bokeh 背景/旗袍/白花全被 SD 重画
    #  成糊+方块伪影, 用户指出"背景模糊"后改为环带)
    ring = cv2.dilate(face, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (161, 161)))  # 脸外扩 ~80px
    bg_mask = ring.copy()
    bg_mask[core > 0] = 0
    Image.fromarray(bg_mask).save(os.path.join(dbg_dir,
                                               "_inpaint_mask.png"))
    gen = torch.Generator("cpu").manual_seed(seed + 1)
    t0 = time.time()
    out = inpaint(prompt_embeds=pos, negative_prompt_embeds=neg,
                  image=Image.fromarray(base_rgb),
                  mask_image=Image.fromarray(bg_mask),
                  strength=0.95, guidance_scale=guidance,
                  num_inference_steps=30, generator=gen).images[0]
    print(f"[outpaint] 原图底+环带重画头发/脖子 "
          f"(背景/身体冻结, {time.time() - t0:.0f}s)", flush=True)
    return out


# 发际线向原图看齐(用户指定): 斜刘海侧梳+高发际线+额头露出;
# framing the face 催两侧都长头发(右侧光晕无发问题)
_OUTPROMPT = ("with thick long straight black hair framing the face, "
              "side-swept bangs, high natural hairline, forehead visible, "
              "pale green qipao collar, green blurred background")


_CFDIR = os.path.join(_HFC, "codeformer_code")
_CFWEIGHTS = os.path.join(_HFC, "codeformer", "codeformer.pth")


def _codeformer_restore(img_bgr, w, ga):
    """第三遍: CodeFormer 面部修复(官方架构已 vendored, 权重严格加载)。
    w=保真权重(1 最贴近输入, 0 最强修复)。五点对齐 512 -> 修复 ->
    逆变换回原图, 凸包羽化蒙版只换脸+发际线区, 背景保持不动。"""
    if _CFDIR not in sys.path:
        sys.path.insert(0, _CFDIR)
    from basicsr.archs.codeformer_arch import CodeFormer
    t0 = time.time()
    net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8,
                     n_layers=9, connect_list=['32', '64', '128', '256'])
    sd = torch.load(_CFWEIGHTS, map_location="cpu",
                    weights_only=True)["params_ema"]
    net.load_state_dict(sd, strict=True)
    net.eval()
    del sd
    lm68, _ = _tddfa_68(img_bgr)
    t5 = ga._five_points(lm68)
    dst = ga.ARCFACE_DST * (512.0 / 112.0)
    M, _ = cv2.estimateAffinePartial2D(t5, dst)
    face = cv2.warpAffine(img_bgr, M, (512, 512))
    x = torch.from_numpy(
        face[:, :, ::-1].transpose(2, 0, 1).copy()).float()[None] / 255 * 2 - 1
    with torch.no_grad():
        y, _, _ = net(x, w=w, adain=False)
    del net
    out512 = ((y[0].permute(1, 2, 0).numpy() + 1) / 2 * 255
              ).clip(0, 255).astype(np.uint8)[:, :, ::-1]
    Minv, _ = cv2.estimateAffinePartial2D(dst, t5)
    back = cv2.warpAffine(out512, Minv,
                          (img_bgr.shape[1], img_bgr.shape[0]))
    # 512 对齐框是紧裁剪: 框外区域(下巴以下/两侧)逆变换得到的是黑像素,
    # 必须先算出有效覆盖区并缩进, 否则黑三角和覆盖边界硬线会贴回图上
    # (实测事故)
    valid = cv2.warpAffine(np.full((512, 512), 255, np.uint8), Minv,
                           (img_bgr.shape[1], img_bgr.shape[0]))
    valid = cv2.erode(valid, np.ones((31, 31), np.uint8))
    hull = cv2.convexHull(lm68.astype(np.int32))
    hullmask = np.zeros(img_bgr.shape[:2], np.uint8)
    cv2.fillConvexPoly(hullmask, hull, 255)
    mask = cv2.dilate(hullmask, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (101, 101)))  # 外扩 ~50px 罩住发际线
    # 下颌线以下+15px 一刀切: CF 羽化横跨下颌线会把 pass-2 的下巴像素
    # 二次磨平(下巴糊主因之一, 用户指定方案3); 嘴唇在下颌线以上不受影响
    for x in range(hullmask.shape[1]):
        ys = np.where(hullmask[:, x] > 0)[0]
        if len(ys):
            mask[ys.max() + 15:, x] = 0
    mask[valid == 0] = 0
    alpha = cv2.GaussianBlur(mask, (41, 41), 0).astype(np.float32) / 255.0
    out = (img_bgr * (1.0 - alpha[..., None])
           + back * alpha[..., None]).astype(np.uint8)
    print(f"[codeformer] w={w} 修复完成 ({time.time() - t0:.0f}s)",
          flush=True)
    return out


def _dim_glow(img_bgr):
    """右侧青辉(过曝散景斑)点压: 强度正比于亮度, 无几何边界。
    只动 Lab 的 L 通道, 色相/纹理保真; 作用域外逐像素不动。
    实测配方 (2026-09-29, 用户从 SD长发/全带Reinhard/点压 三案选定;
    全带 Reinhard 无论全量半量都贴出灰板色块, 已证伪)。"""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
    H, W = skin.shape
    region = np.zeros((H, W), np.float32)
    for y in range(80, 400):                      # 面右缘 +4~+145 带
        xs = np.where(skin[y])[0]
        if len(xs):
            region[y, xs.max() + 4:min(xs.max() + 145, W)] = 1.0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[:, :, 0]
    s = np.clip((L - 180.0) / 35.0, 0, 1) * region
    s = cv2.GaussianBlur(s, (9, 9), 0)
    painted = lab.copy()
    painted[:, :, 0] = L - s * (L - 158.0)        # 亮斑压向中绿
    painted = cv2.cvtColor(np.clip(painted, 0, 255).astype(np.uint8),
                           cv2.COLOR_Lab2BGR)
    out = (img_bgr.astype(np.float32) * (1 - s[..., None])
           + painted.astype(np.float32) * s[..., None])
    print(f"[glow] 右侧过曝斑已点压 (峰值强度 {s.max():.2f})", flush=True)
    return np.clip(out, 0, 255).astype(np.uint8)


def _fix_streak(img_bgr):
    """脖区黄色细条修复(实测定稿): 下巴白斑回贴会把原图旗袍领边缘
    带进脖区。在实测范围内找黄色连通域(r≈g>>b), telea 周围皮肤填。
    找不到就跳过, 对其他照片无害。"""
    b, g, r = (img_bgr[:, :, i].astype(int) for i in range(3))
    s = r + g + b
    st = ((r - b > 20) & (g - b > 15) & (np.abs(r - g) < 30) & (s > 380))
    zone = np.zeros(st.shape, bool)
    zone[448:492, 138:218] = True
    st &= zone
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        st.astype(np.uint8))
    keep = np.zeros(st.shape, np.uint8)
    for i in range(1, n):
        if stats[i, 4] > 8:
            keep[lab == i] = 255
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8))
    if keep.any():
        print(f"[streak] 脖区黄条修复 {int((keep > 0).sum())}px",
              flush=True)
        return cv2.inpaint(img_bgr, keep, 4, cv2.INPAINT_TELEA)
    return img_bgr


def _jaw_shadow(img_bgr, dbg_dir):
    """颌下阴影渐变合成(实测定稿 v6, 用户选定): 沿 3D 网格下颌曲线
    下方 14px 内 10% 渐弱暗化, 侧缘 25% 衰减——模拟原图下颌显锐的
    真正机制(阴影渐变而非硬边)。不移动任何边界, 零贴纸风险, 实测
    零分数成本。六次下巴边界重建全证伪后的唯一正解。"""
    mesh_path = os.path.join(dbg_dir, "_rnr_mesh.png")
    if not os.path.exists(mesh_path):
        return img_bgr
    mesh = cv2.imread(mesh_path)
    if mesh is None:
        return img_bgr
    H, W = img_bgr.shape[:2]
    mesh = cv2.resize(mesh, (W, H), interpolation=cv2.INTER_LANCZOS4)
    sil = (mesh.max(axis=2) > 25).astype(np.uint8) * 255
    sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(sil)
    if n <= 1:
        return img_bgr
    sil = (lab == 1 + int(np.argmax(st[1:, 4]))).astype(np.uint8) * 255
    contour = np.full(W, -1.0)
    for x in range(W):
        ys = np.where(sil[:, x] > 0)[0]
        if len(ys):
            contour[x] = ys.max()
    valid = contour > 340                      # 只取下巴段
    if not valid.any():
        return img_bgr
    xs = np.where(valid)[0]
    cs = contour[valid]
    cs_s = np.array([np.median(cs[max(0, i - 7):i + 8])
                     for i in range(len(cs))], np.float32)
    cs_s = cv2.GaussianBlur(cs_s.reshape(1, -1), (1, 31), 0).ravel()
    contour[valid] = cs_s
    D = 14
    shadow = np.zeros((H, W), np.float32)
    x_lo, x_hi = xs.min(), xs.max()
    for x in range(x_lo, x_hi + 1):
        y0 = int(contour[x])
        span = (x_hi - x_lo) / 2.0
        lat = 1.0 - max(0.0, (abs(x - (x_lo + x_hi) / 2)
                              / span - 0.75) / 0.25)   # 侧缘 25% 衰减
        for d in range(D):
            if y0 + d < H:
                shadow[y0 + d, x] = 0.10 * (1 - d / D) * lat
    shadow = cv2.GaussianBlur(shadow, (7, 7), 0)
    out = img_bgr.astype(np.float32) * (1 - shadow[..., None])
    print(f"[jaw] 颌下阴影渐变 (峰值 {shadow.max():.2f})", flush=True)
    return np.clip(out, 0, 255).astype(np.uint8)


def _restore_real(img_bgr, dbg_dir):
    """真实像素回贴, 两个阶段。实测定稿 2026-09-29:
    阶段1 从 _outpaint_comp.png (inpaint 前合成图) 恢复被环带重画毁掉的
      顶部真发整区 + 下巴白斑(过亮区)。
      - 皮肤保护掩码必须用 comp 的 (fin 洗白区是肤色调, 会误判成皮肤
        而被排除在回贴外——实测事故)
      - 暗像素选择会留斑块边界, 整区回贴才干净
      - 皮肤膨胀 9 (用户从 21/9 两版选定 9, 纹理优先)
    阶段2 比照原图, 从 _outpaint_base.png (原照片裁剪, 与输出同坐标系)
      回贴右侧浮发/左侧垂发/头花/左下区, 右背景散景半量恢复(用户选定)。
      - 暗选择会把侧脸眉毛/鼻子暗像素当发丝(实测分析), 必须 base 皮肤
        膨胀15 保护
      - 侧脸轮廓位置在正脸更内侧, 该处"发丝"贴回落在正脸皮肤上变绿条
        (实测事故), B 区必须再避让 fin 的正面脸皮肤"""
    comp_path = os.path.join(dbg_dir, "_outpaint_comp.png")
    if not os.path.exists(comp_path):
        return img_bgr
    comp = cv2.imread(comp_path)
    if comp is None or comp.shape != img_bgr.shape:
        return img_bgr
    H, W = img_bgr.shape[:2]

    def _skin_strict(img, dilate):
        """YCrCb 皮肤阈值 + Y>110: 深棕发(Y<110)不再被误判成皮肤而
        排除在回贴外(实测: 顶发区 58%/花瓣区 17% 被误杀出洞)"""
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = ycc[:, :, 0], ycc[:, :, 1], ycc[:, :, 2]
        sk = ((cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
              & (y > 110)).astype(np.uint8) * 255
        return cv2.dilate(sk, np.ones((dilate, dilate), np.uint8))

    skin = _skin_strict(comp, 21)  # 9->21: 全帧放大后 B 版头皮斑块读作秃斑+黑边(用户反馈), 回到干净的 A 版发际线(0.482); 局部修斑已证伪(棕泥)
    hair = np.zeros((H, W), np.uint8)
    hair[0:180, 0:340] = 255                     # 顶部真发整区
    glare = ((img_bgr.sum(axis=2).astype(int)
              - comp.sum(axis=2).astype(int) > 60)).astype(np.uint8) * 255
    neck_zone = np.zeros((H, W), np.uint8)
    neck_zone[385:495, 120:330] = 255            # 下巴白斑区
    neck = cv2.dilate(cv2.bitwise_and(glare, neck_zone),
                      np.ones((15, 15), np.uint8))
    mask = cv2.bitwise_or(hair, neck)
    mask[skin > 0] = 0
    a = cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0
    out = (img_bgr.astype(np.float32) * (1 - a[..., None])
           + comp.astype(np.float32) * a[..., None])
    out = np.clip(out, 0, 255).astype(np.uint8)
    print(f"[restore] 真发整区+下巴白斑回贴 ({int((mask > 0).sum())}px)",
          flush=True)

    base_path = os.path.join(dbg_dir, "_outpaint_base.png")
    if not os.path.exists(base_path):
        return out
    base = cv2.imread(base_path)
    if base is None or base.shape != img_bgr.shape:
        return out
    skinB = _skin_strict(base, 15)
    skinF = _skin_strict(out, 15)

    dark = (base.sum(axis=2) < 400).astype(np.uint8) * 255

    def _zone(x0, x1, y0, y1, sel):
        z = np.zeros((H, W), np.uint8)
        z[y0:y1, x0:x1] = 255
        return cv2.bitwise_and(dark, z) if sel else z

    maskB = _zone(300, 360, 80, 270, True)       # 右侧浮发(背景上的)
    maskB[skinB > 0] = 0                         # 避让侧脸五官(暗选择会
    maskB[skinF > 0] = 0                         # 误抓眉毛鼻子); 且须飘在
    # 两张脸之外, 否则落在正脸皮肤上(绿条事故)
    maskC = _zone(0, 140, 220, 440, True)        # 左侧垂发
    maskC[skinB > 0] = 0
    maskE = _zone(0, 140, 440, 512, False)       # 左下区整区
    maskE[skinB > 0] = 0
    maskD = _zone(0, 90, 135, 275, False)        # 头花整区(该侧无脸,
    # 不做皮肤保护——保护会把白花瓣误判成皮肤留出糊洞, 实测)
    mask2 = cv2.dilate(maskB | maskC | maskD | maskE,
                       np.ones((5, 5), np.uint8))
    a2 = cv2.GaussianBlur(mask2, (15, 15), 0).astype(np.float32) / 255.0
    out = (out.astype(np.float32) * (1 - a2[..., None])
           + base.astype(np.float32) * a2[..., None])
    # 右背景散景半量恢复, 两遍叠加是负载语义(用户批准的 R3 就是这么做
    # 出来的; 合成单遍会在重叠区少 25% 散景, 重放即变图):
    # 第一遍 x365+/y60-400 (R2 遗留), 第二遍 x325+全高统一(消接缝,
    # 避让两张脸皮肤 + fin 暗像素=正脸右上新生发块, 半透明化是大忌)
    reg = _zone(365, 512, 60, 400, False)
    reg[skinB > 0] = 0
    a3 = cv2.GaussianBlur(reg, (21, 21), 0).astype(np.float32) / 255.0 * 0.5
    out = (out * (1 - a3[..., None])
           + base.astype(np.float32) * a3[..., None])
    darkF = cv2.dilate((out.sum(axis=2) < 400).astype(np.uint8) * 255,
                       np.ones((15, 15), np.uint8))
    reg2 = _zone(325, 512, 0, 512, False)
    reg2[skinB > 0] = 0
    reg2[skinF > 0] = 0
    reg2[darkF > 0] = 0
    a4 = cv2.GaussianBlur(reg2, (41, 41), 0).astype(np.float32) / 255.0 * 0.5
    out = (out * (1 - a4[..., None])
           + base.astype(np.float32) * a4[..., None])
    print(f"[restore] 比照原图回贴 ({int((mask2 > 0).sum())}px) + "
          f"右背景统一半量 ({int((reg2 > 0).sum())}px)", flush=True)
    return np.clip(out, 0, 255).astype(np.uint8)


def _refine_with_arc2face(mesh, image_bgr, out_path, strength=0.8,
                          guidance=3.0, seed=42, steps=30,
                          id_src="photo", prompt=None, full=True,
                          outpaint_prompt=_OUTPROMPT,
                          align_M=None, ip_scale=0.5, cf_w=0.8,
                          seeds=None):
    """无 RnR GAN 权重时的本地替代(权重全部已在本地):
    正脸网格 -> SD1.5 img2img 细化 + Arc2Face 身份注入。"""
    from PIL import Image
    from diffusers import StableDiffusionImg2ImgPipeline
    ga = _arc2face_module()

    pipe = ga._load_pipeline(ip_scale=ip_scale)
    i2i = StableDiffusionImg2ImgPipeline(**pipe.components)

    cache = _HFC
    onnx = os.path.join(
        ga._snapshot(cache, "models--FoivosPar--Arc2Face"), "arcface.onnx")
    mesh_np = (mesh.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    mesh_bgr = cv2.cvtColor(mesh_np, cv2.COLOR_RGB2BGR)
    dbg = os.path.dirname(out_path)
    # 身份源: 90° 侧脸 TTA 余弦仅 0.141(基本是噪声); 正脸网格是
    # 正面且带真实纹理, ArcFace 信号强得多。网格检测失败才回退原图。
    # 身份向量(photo 源; mesh 源已证伪): 脸/头发两遍共用
    src_img = image_bgr
    if id_src == "mesh":
        try:
            id_emb, _al, _lm = ga._id_embedding(mesh_bgr, onnx, dbg)
            src_img = None
            print("[id emb] 身份源: 正脸网格", flush=True)
        except Exception as e:
            print(f"[id emb] 网格提身份失败({e}), 回退原图", flush=True)
    if src_img is not None:
        id_emb, _al, _lm = ga._id_embedding(image_bgr, onnx, dbg)
        print("[id emb] 身份源: 原图", flush=True)
    if prompt:
        # 长后缀会冲散 Arc2Face 的条件分布(0.035 已证伪),
        # 只建议极短后缀
        pos = _project_with_features(pipe, id_emb, prompt)
        print(f"[prompt] id 注入 + 后缀: {prompt[:50]}...", flush=True)
    else:
        pos = ga._project_face_embs(pipe, id_emb)
    neg = ga._encode_text(pipe, ga.NEG_TEXT)
    inpaint = None
    pos2 = None
    if full:
        # 第二遍用独立的短后缀条件(贴近 7-token 训练模板)
        pos2 = _project_with_features(pipe, id_emb, outpaint_prompt)
        from diffusers import StableDiffusionInpaintPipeline
        inpaint = StableDiffusionInpaintPipeline(**pipe.components)
    # IP-Adapter 图像提示嵌入: 源用 ArcFace 对齐的 112 脸切块(只含脸,
    # 不带旗袍/绿背景——防止场景泄漏进黑底第一遍, 破坏灰度掩码提取)
    ip_embeds = None
    if getattr(i2i, "image_encoder", None) is not None:
        ip_img = Image.fromarray(cv2.cvtColor(_al, cv2.COLOR_BGR2RGB))
        ip_embeds = i2i.prepare_ip_adapter_image_embeds(
            ip_img, None, "cpu", 1, True)
        print("[ip-adapter] 图像提示嵌入完成 (源: 112 对齐脸)",
              flush=True)
    # 提示词/图像嵌入全部编码完毕，文本编码器 (~0.5GB) 和图像编码器
    # (~2.5GB) 去噪全程不再使用，立刻释放——提交额度紧张的机器上这
    # 是生与死的差别
    for p in (pipe, i2i, inpaint):
        if p is not None:
            p.text_encoder = None
            p.image_encoder = None
    import gc
    gc.collect()

    init = Image.fromarray(cv2.resize(
        mesh_np, (512, 512), interpolation=cv2.INTER_LANCZOS4))
    # 多种子择优(用户指定: 未见图一次给多种子, arcface 自动选最优,
    # 只重跑第一遍 ~2min/种子; 方差实测 ±0.06, 一抽即中的概率低)
    seed_list = seeds if seeds else [seed]
    best = None
    for s in seed_list:
        gen = torch.Generator("cpu").manual_seed(s)
        t0 = time.time()
        cand = i2i(prompt_embeds=pos, negative_prompt_embeds=neg,
                   image=init, ip_adapter_image_embeds=ip_embeds,
                   strength=strength, guidance_scale=guidance,
                   num_inference_steps=steps, generator=gen).images[0]
        if len(seed_list) == 1:
            out = cand
            print(f"[refine] img2img strength={strength} "
                  f"guidance={guidance} seed={s} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            break
        cos = _embed_cosine(image_bgr, cv2.cvtColor(
            np.asarray(cand), cv2.COLOR_RGB2BGR), onnx)
        print(f"[seed {s}] pass-1 余弦 = {cos:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if best is None or cos > best[0]:
            best = (cos, s, cand)
    if best is not None:
        _, seed, out = best
        print(f"[seed] 采用 {seed} (余弦 {best[0]:.3f})", flush=True)
    if ip_embeds is not None:
        # 卸载 IP-Adapter(共享 UNet 的处理器和 encoder_hid_proj 复位);
        # 再给第二遍重开注意力切片恢复速度(此时已无 IP 处理器, 不再
        # 打架; SDPA 在 CPU 上比切片手动路径慢 ~3 倍, 686s->231s 实测)
        i2i.unload_ip_adapter()
        i2i.enable_attention_slicing()
        print("[ip-adapter] 已卸载, 第二遍回原管线(重开切片)", flush=True)
    if full:
        out = _outpaint_surroundings(inpaint, out, image_bgr, align_M,
                                     pos2, neg, seed,
                                     guidance, os.path.dirname(out_path))
    if cf_w and cf_w > 0:
        # 第三遍前 SD 管线已全部用完, 先释放 UNet/VAE 给 CodeFormer 腾地方
        del pipe, i2i, inpaint
        gc.collect()
        out_bgr = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
        out = Image.fromarray(cv2.cvtColor(
            _codeformer_restore(out_bgr, cf_w, ga), cv2.COLOR_BGR2RGB))
    out_bgr = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
    out_bgr = _jaw_shadow(
        _fix_streak(_restore_real(_dim_glow(out_bgr), dbg)), dbg)
    out = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
    out.save(out_path)
    print(f"[done] {out_path}", flush=True)

    out_bgr = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
    cos = _embed_cosine(image_bgr, out_bgr, onnx)
    print(f"[sim] 输出 vs 原图 arcface 余弦 = {cos:.3f}", flush=True)


def _tddfa_68(image):
    """仓库自带 3ddfa_v2: 人脸框 + 68 点 + 62 维参数。
    注意 RnR 也有 models/ 包(GAN), 会遮蔽 3ddfa_v2 的 models/
    (mobilenet) ——TDDFA 调用期间临时切换 sys.modules。"""
    import importlib.util
    import yaml
    d3 = os.path.join(ROOT, "3ddfa_v2")
    spec = importlib.util.spec_from_file_location(
        "models", os.path.join(d3, "models", "__init__.py"),
        submodule_search_locations=[os.path.join(d3, "models")])
    d3_models = importlib.util.module_from_spec(spec)
    saved = sys.modules.get("models")
    sys.modules["models"] = d3_models  # 必须先注册, __init__.py 的相对导入要用
    try:
        spec.loader.exec_module(d3_models)
        from FaceBoxes.FaceBoxes import FaceBoxes
        from TDDFA import TDDFA
        with open(os.path.join(d3, "configs", "mb1_120x120.yml")) as f:
            cfg = yaml.safe_load(f)
        cfg["gpu_mode"] = False
        for key, rel in (("bfm_fp", "configs/"),
                         ("checkpoint_fp", "weights/")):
            if str(cfg.get(key, "")).startswith(rel):
                cfg[key] = os.path.normpath(os.path.join(d3, cfg[key]))
        tddfa = TDDFA(**cfg)
        box = max(FaceBoxes()(image),
                  key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        with torch.no_grad():
            param_lst, roi_box_lst = tddfa(image, [box])
        ver = tddfa.recon_vers(param_lst, roi_box_lst,
                               dense_flag=False)[0]
    finally:
        if saved is None:
            sys.modules.pop("models", None)
        else:
            sys.modules["models"] = saved
    lm68 = ver.T[:, :2].astype(np.float32)   # (68,2) 图像 xy
    param = np.asarray(param_lst[0]).reshape(-1)
    roi = np.asarray(roi_box_lst[0]).reshape(-1)
    assert param.size == 62, param.shape
    return lm68, np.concatenate([param, roi]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "result", "rnr", "rnr_frontal.png"))
    ap.add_argument("--yaw", type=float, default=None,
                    help="额外指定目标 yaw(弧度), 默认转正")
    ap.add_argument("--strength", type=float, default=0.8,
                    help="img2img 去噪强度(fallback 路径), 越大越自由")
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default=None,
                    help="多种子择优, 逗号分隔 (如 7,42,123): "
                    "只重跑 pass-1, arcface 自动选最优")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--id-src", choices=["mesh", "photo"], default="photo",
                    help="身份向量来源(仅无 --prompt 时生效)")
    ap.add_argument("--prompt", default=None,
                    help="特征文字后缀(追加在 'photo of a id person, ' 后)")
    ap.add_argument("--no-full", action="store_true",
                    help="只做脸部细化, 不生长头发/背景")
    ap.add_argument("--outpaint-prompt", default=None,
                    help="第二遍(头发/背景)的短后缀提示词")
    ap.add_argument("--cf-w", type=float, default=0.8,
                    help="CodeFormer 第三遍保真权重(扫参定稿0.8), 0=关闭")
    ap.add_argument("--ip-scale", type=float, default=0.5,
                    help="IP-Adapter 图像提示强度(0=关闭; 0.3 已验证 "
                         "+0.075, 逐步加压试上限)")
    args = ap.parse_args()

    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    assert image is not None, f"读不到输入图: {args.input}"

    t0 = time.time()
    lm68, param = _tddfa_68(image)
    print(f"[3ddfa] 68点+参数 {time.time() - t0:.1f}s", flush=True)

    # 数据对齐(与 allface_dataset.affine_align 一致): 5 点 -> 256
    S = 256
    t5 = np.stack([lm68[36:42].mean(axis=0), lm68[42:48].mean(axis=0),
                   lm68[31], lm68[48], lm68[54]]).astype(np.float32)
    src = ARCFACE5 * 290 / 112
    src[:, 0] += 50
    src[:, 1] += 60
    src = src / 400 * S
    tform = trans.SimilarityTransform()
    tform.estimate(t5, src)
    M = tform.params[0:2, :]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    warped = cv2.warpAffine(rgb, M, (S, S), borderValue=0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(os.path.join(os.path.dirname(args.out),
                             "_rnr_aligned.png"),
                cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    img_t = torch.from_numpy(warped.transpose(2, 0, 1)) / 255.0

    t0 = time.time()
    renderer = CPURender(render_size=S)
    renderer.debug_dir = os.path.dirname(args.out)
    mesh, lm, ang = renderer.rotate_render(param, img_t, M)
    print(f"[render] 软光栅完成, 原yaw={ang:.2f}rad "
          f"({time.time() - t0:.1f}s)", flush=True)
    cv2.imwrite(os.path.join(os.path.dirname(args.out),
                             "_rnr_mesh.png"),
                cv2.cvtColor((mesh.numpy().transpose(1, 2, 0)
                              * 255).astype(np.uint8),
                             cv2.COLOR_RGB2BGR))

    t0 = time.time()
    opt = _opt_namespace(S)
    if not _find_netG():
        print("[fallback] 无 GAN 权重, 改用本地 Arc2Face img2img 细化",
              flush=True)
        _refine_with_arc2face(mesh, image, args.out,
                              strength=args.strength,
                              guidance=args.guidance, seed=args.seed,
                              steps=args.steps, id_src=args.id_src,
                              prompt=args.prompt, full=not args.no_full,
                              outpaint_prompt=(
                                  args.outpaint_prompt or _OUTPROMPT),
                              align_M=M, ip_scale=args.ip_scale,
                              cf_w=args.cf_w,
                              seeds=([int(x) for x in
                                      args.seeds.split(",")]
                                     if args.seeds else None))
        return
    netG = _load_generator(opt)
    print(f"[netG] {time.time() - t0:.0f}s", flush=True)

    seg, seg_all = _seg_map(lm, opt.no_gaussian_landmark, S, ang)
    rotated_mesh = mesh[None] * 2 - 1
    if opt.label_mask:
        rotated_mesh = (rotated_mesh
                        + seg_all[:, 4].unsqueeze(1)
                        + seg_all[:, 0].unsqueeze(1))
        rotated_mesh[rotated_mesh >= 1] = 0
    with torch.no_grad():
        fake = netG(rotated_mesh, seg)
    out = ((fake[0].clamp(-1, 1) + 1) / 2 * 255).byte().numpy()
    out = out.transpose(1, 2, 0)
    cv2.imwrite(args.out, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()
