"""Automatic keypoint detection for subjects with no trained detector
(animals, sculptures, ...) via DINOv2 dense semantic correspondence.

Idea: annotate ONE reference image once (assets/reference_animal.jpg +
reference_animal_kps.json). For a new image, each reference keypoint's
patch feature is matched against all patch features of the new image by
cosine similarity — "where in this image looks most like the reference
inner eye corner" — giving semantic keypoints with zero training.

Model weights (facebookresearch/dinov2, ViT-S/14, ~84MB) download once
on first use and are cached locally; runtime stays offline.
"""

import os

if "TORCH_HOME" not in os.environ and os.path.isdir(r"D:\huggingface_cache"):
    os.environ["TORCH_HOME"] = r"D:\huggingface_cache\torch"

import json

import numpy as np

PATCH = 14
_MAX_SIDE = 560  # feature grid stays <= 40x40: fast on CPU

_MODEL = None
_FAILED = False

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_REF_IMAGE = os.path.join(_PROJECT_ROOT, "assets", "reference_animal.jpg")
_REF_KPS = os.path.join(_PROJECT_ROOT, "assets", "reference_animal_kps.json")


_WEIGHTS_URL = ("https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/"
                "dinov2_vits14_pretrain.pth")


class _DinoV2ViTSmall:
    """Minimal DINOv2 ViT-S/14 (embed 384, 12 blocks, 6 heads, LayerScale).

    Reimplemented so no GitHub access is needed; weights come from Meta's
    public CDN (dl.fbaipublicfiles.com), the same file torch.hub fetches.
    """

    def __init__(self, state_dict):
        import torch
        import torch.nn as nn

        self.torch = torch
        dim, depth, heads = 384, 12, 6
        dev = "cpu"

        class Block(nn.Module):
            def __init__(self_, sd, pfx):
                super().__init__()
                self_.norm1 = nn.LayerNorm(dim)
                self_.norm1.weight.data = sd[f"{pfx}norm1.weight"]
                self_.norm1.bias.data = sd[f"{pfx}norm1.bias"]
                self_.qkv_w = sd[f"{pfx}attn.qkv.weight"]
                self_.qkv_b = sd[f"{pfx}attn.qkv.bias"]
                self_.proj_w = sd[f"{pfx}attn.proj.weight"]
                self_.proj_b = sd[f"{pfx}attn.proj.bias"]
                self_.ls1 = sd[f"{pfx}ls1.gamma"]
                self_.norm2 = nn.LayerNorm(dim)
                self_.norm2.weight.data = sd[f"{pfx}norm2.weight"]
                self_.norm2.bias.data = sd[f"{pfx}norm2.bias"]
                self_.fc1_w = sd[f"{pfx}mlp.fc1.weight"]
                self_.fc1_b = sd[f"{pfx}mlp.fc1.bias"]
                self_.fc2_w = sd[f"{pfx}mlp.fc2.weight"]
                self_.fc2_b = sd[f"{pfx}mlp.fc2.bias"]
                self_.ls2 = sd[f"{pfx}ls2.gamma"]

            def forward(self_, x):
                import torch.nn.functional as F
                B, N, C = x.shape
                h = self_.norm1(x)
                qkv = F.linear(h, self_.qkv_w, self_.qkv_b)
                qkv = qkv.reshape(B, N, 3, heads, C // heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
                o = o.transpose(1, 2).reshape(B, N, C)
                x = x + self_.ls1 * F.linear(o, self_.proj_w, self_.proj_b)
                h = self_.norm2(x)
                h = F.linear(F.gelu(F.linear(h, self_.fc1_w, self_.fc1_b)),
                             self_.fc2_w, self_.fc2_b)
                return x + self_.ls2 * h

        self.patch_w = state_dict["patch_embed.proj.weight"].to(dev)
        self.patch_b = state_dict["patch_embed.proj.bias"].to(dev)
        self.cls_token = state_dict["cls_token"].to(dev)
        self.pos_embed = state_dict["pos_embed"].to(dev)
        self.blocks = [Block(state_dict, f"blocks.{i}.") for i in range(depth)]
        self.norm_w = state_dict["norm.weight"].to(dev)
        self.norm_b = state_dict["norm.bias"].to(dev)

    def _pos_embed(self, gh, gw):
        import torch.nn.functional as F
        pos = self.pos_embed
        n = pos.shape[1] - 1
        gs = int(n ** 0.5)
        if gs * gs == n and (gs != gh or gs != gw):
            cls_pos, grid = pos[:, :1], pos[:, 1:]
            grid = grid.reshape(1, gs, gs, -1).permute(0, 3, 1, 2)
            grid = F.interpolate(grid, size=(gh, gw), mode="bicubic",
                                 align_corners=False)
            grid = grid.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)
            return self.torch.cat([cls_pos, grid], dim=1)
        return pos

    def patch_tokens(self, x):
        """x: (1, 3, H, W) normalized; returns (gh*gw, 384) after final norm."""
        import torch.nn.functional as F
        torch = self.torch
        f = F.conv2d(x, self.patch_w, self.patch_b, stride=PATCH)
        gh, gw = f.shape[2], f.shape[3]
        t = f.flatten(2).transpose(1, 2)  # (1, gh*gw, 384)
        t = torch.cat([self.cls_token, t], dim=1) + self._pos_embed(gh, gw)
        for blk in self.blocks:
            t = blk(t)
        t = F.layer_norm(t, (t.shape[-1],), self.norm_w, self.norm_b)
        return t[0, 1:]


def _download_weights(dest: str) -> bool:
    import urllib.request
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print("KeypointTransfer: downloading DINOv2 weights (~84MB, once)...")
        urllib.request.urlretrieve(_WEIGHTS_URL, dest)
        return True
    except Exception as e:
        print(f"KeypointTransfer: weight download failed ({e})")
        return False


def _get_model():
    """Lazily load DINOv2 ViT-S/14 (first call downloads the weights)."""
    global _MODEL, _FAILED
    if _MODEL is not None:
        return _MODEL
    if _FAILED:
        return None
    try:
        import torch
        cache = os.path.join(os.environ.get("TORCH_HOME", ""),
                             "hub", "checkpoints")
        dest = os.path.join(cache, "dinov2_vits14_pretrain.pth")
        if not os.path.isfile(dest) and not _download_weights(dest):
            raise RuntimeError("weights unavailable")
        sd = torch.load(dest, map_location="cpu", weights_only=True)
        _MODEL = _DinoV2ViTSmall(sd)
    except Exception as e:
        print(f"KeypointTransfer: unavailable ({e})")
        _FAILED = True
        _MODEL = None
    return _MODEL


def _dense_features(model, image: np.ndarray):
    """L2-normalized per-patch features and the resize scale.

    Returns (feats, scale) with feats of shape (H/PATCH, W/PATCH, C);
    scale maps original image pixels to resized pixels.
    """
    import cv2
    import torch

    h, w = image.shape[:2]
    scale = min(1.0, _MAX_SIDE / max(h, w))
    W = max(PATCH, round(w * scale / PATCH) * PATCH)
    H = max(PATCH, round(h * scale / PATCH) * PATCH)

    img = cv2.resize(image, (W, H))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1)[None])

    with torch.no_grad():
        tokens = model.patch_tokens(x)  # (gh*gw, C)
    gh, gw = H // PATCH, W // PATCH
    feats = tokens.reshape(gh, gw, -1).numpy().astype(np.float32)
    feats /= np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8
    return feats, W / w


def transfer_keypoints(image: np.ndarray, ref_image: np.ndarray,
                       ref_kps: dict):
    """Transfer reference keypoints onto a new image.

    Returns (keypoints, info) or (None, reason).
    """
    result = _transfer(image, ref_image, ref_kps)
    if result is None:
        return None, "DINOv2 模型不可用（首次使用需联网下载权重）"
    out, mean_sim = result
    if mean_sim < 0.25:
        return None, f"与参考图相似度过低 ({mean_sim:.2f})，请手动标注"
    info = f"DINOv2语义迁移: 平均相似度 {mean_sim:.2f}, {len(out)}个关键点"
    return out, info


def _transfer(image: np.ndarray, ref_image: np.ndarray, ref_kps: dict):
    """Core transfer; returns (keypoints, mean_similarity) or None."""
    model = _get_model()
    if model is None:
        return None

    f_ref, s_ref = _dense_features(model, ref_image)
    f_tgt, s_tgt = _dense_features(model, image)
    gh_r, gw_r = f_ref.shape[:2]
    gh_t, gw_t = f_tgt.shape[:2]
    flat = f_tgt.reshape(-1, f_tgt.shape[-1])

    names = list(ref_kps.keys())
    ref_pts = np.array([ref_kps[n] for n in names], dtype=np.float64)
    sims_map = {}
    for name, (x, y) in ref_kps.items():
        # keypoints are in ORIGINAL pixels; the grid lives on the resized
        # image — scale first (missing this put queries in the wrong cell)
        px = min(int(x * s_ref / PATCH), gw_r - 1)
        py = min(int(y * s_ref / PATCH), gh_r - 1)
        sims_map[name] = (flat @ f_ref[py, px]).reshape(gh_t, gw_t)

    def match_at(name, cy=None, cx=None, radius=None):
        """Best match (sub-patch refined), optionally inside a window."""
        sim = sims_map[name]
        if cy is None:
            y0, x0, y1, x1 = 0, 0, gh_t, gw_t
        else:
            y0 = max(0, cy - radius)
            y1 = min(gh_t, cy + radius + 1)
            x0 = max(0, cx - radius)
            x1 = min(gw_t, cx + radius + 1)
        win = sim[y0:y1, x0:x1]
        idx = int(np.argmax(win))
        gy, gx = divmod(idx, win.shape[1])
        best = float(win.flat[idx])
        # softmax centroid of the 3x3 neighborhood around the argmax
        z0, z1 = max(0, gy - 1), min(win.shape[0], gy + 2)
        u0, u1 = max(0, gx - 1), min(win.shape[1], gx + 2)
        patch = win[z0:z1, u0:u1]
        wgt = np.exp((patch - patch.max()) * 10.0)
        wgt /= wgt.sum()
        ys, xs = np.mgrid[z0:z1, u0:u1]
        fy = y0 + float((ys * wgt).sum())
        fx = x0 + float((xs * wgt).sum())
        # back to ORIGINAL target pixels (the grid lives on the resize)
        return ((fx + 0.5) * PATCH / s_tgt,
                (fy + 0.5) * PATCH / s_tgt, best)

    # First pass: global best match per point, plus a distinctiveness
    # margin (gap to the best match away from the winner). Fur stripes
    # and background tie everywhere -> low margin; eyes/nose stand out.
    matched = {}
    best_sim = {}
    margin = {}
    for name in names:
        mx, my, best = match_at(name)
        matched[name] = (mx, my)
        best_sim[name] = best
        sim = sims_map[name]
        gy = min(int(my * s_tgt / PATCH), gh_t - 1)
        gx = min(int(mx * s_tgt / PATCH), gw_t - 1)
        masked = sim.copy()
        y0, y1 = max(0, gy - 2), min(gh_t, gy + 3)
        x0, x1 = max(0, gx - 2), min(gw_t, gx + 3)
        masked[y0:y1, x0:x1] = -1.0
        margin[name] = best - float(masked.max())

    # Geometric consistency: symmetric body parts (two eyes, two ears,
    # both whisker pads) and periodic fur stripes make global argmax swap
    # or shift. Fit an affine on the DISTINCTIVE central anchors only
    # (fur/background points would out-vote them), then re-match every
    # inconsistent point inside a window around its predicted location.
    import cv2
    m_pts = np.array([matched[n] for n in names], dtype=np.float64)
    # Only distinctive matches vote on the global transform — stripe and
    # background points would otherwise form a wrong but consistent
    # majority (periodic texture consensus).
    margins = np.array([margin[n] for n in names])
    cut = max(0.015, float(np.median(margins)))
    anchors = [i for i, n in enumerate(names) if margin[n] >= cut]
    img_h, img_w = image.shape[:2]
    thresh = max(20.0, 0.04 * max(img_h, img_w))
    if len(anchors) >= 4:
        M, _inl = cv2.estimateAffine2D(
            ref_pts[None, anchors], m_pts[None, anchors],
            method=cv2.RANSAC, ransacReprojThreshold=thresh)
        if M is not None:
            pred = cv2.transform(ref_pts[None], M)[0]
            scale = max(1.0, float(np.sqrt(abs(np.linalg.det(M[:, :2])))))
            radius = max(2, int(0.06 * gw_t / scale))
            for i, name in enumerate(names):
                if np.hypot(*(pred[i] - m_pts[i])) <= thresh:
                    continue  # already consistent — trust the first pass
                cy = int(pred[i, 1] * s_tgt / PATCH)
                cx = int(pred[i, 0] * s_tgt / PATCH)
                mx, my, best = match_at(
                    name, min(max(cy, 0), gh_t - 1),
                    min(max(cx, 0), gw_t - 1), radius)
                matched[name] = (mx, my)
                best_sim[name] = best

    sims = list(best_sim.values())
    mean_sim = float(np.mean(sims)) if sims else 0.0
    return matched, mean_sim


def transfer_with_reference(image: np.ndarray):
    """Transfer using the bundled reference annotation, if present.

    The reference animal faces one way; a new subject may face the other.
    Both orientations of the reference are tried and the better-scoring
    one wins.
    """
    if not (os.path.isfile(_REF_IMAGE) and os.path.isfile(_REF_KPS)):
        return None, "未找到参考标注（assets/reference_animal.*）"

    import cv2
    ref_image = cv2.imread(_REF_IMAGE)
    if ref_image is None:
        return None, "参考图读取失败"
    with open(_REF_KPS, encoding="utf-8") as f:
        ref_kps = {k: tuple(v) for k, v in json.load(f).items()}

    candidates = [(ref_image, ref_kps)]
    rw = ref_image.shape[1]
    flipped = cv2.flip(ref_image, 1)
    flip_kps = {k: (rw - x, y) for k, (x, y) in ref_kps.items()}
    candidates.append((flipped, flip_kps))

    best = None
    for ref_img, kps in candidates:
        result = _transfer(image, ref_img, kps)
        if result is not None and (best is None or result[1] > best[1]):
            best = result
    if best is None:
        return None, "DINOv2 模型不可用（首次使用需联网下载权重）"

    out, mean_sim = best
    if mean_sim < 0.25:
        return None, f"与参考图相似度过低 ({mean_sim:.2f})，请手动标注"
    info = f"DINOv2语义迁移: 平均相似度 {mean_sim:.2f}, {len(out)}个关键点"
    return out, info
