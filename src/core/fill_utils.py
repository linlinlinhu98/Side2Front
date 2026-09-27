"""Shared image-fill helpers for the frontalization pipelines.

Both the real (3DMM) and anime (TPS) pipelines repair small regions with
the same multi-scale diffusion fill; it used to be copy-pasted half a
dozen times (with drifted sigma ladders and blur kernels), so it lives
here exactly once.
"""
import cv2
import numpy as np


def fast_blur(img, sigma: float):
    """Gaussian-like blur that stays fast for large sigmas: a true
    GaussianBlur with sigma≈40 needs a 300-tap kernel and dominates the
    render time; a box filter is O(1) per pixel regardless of kernel size
    and is plenty for diffusion fills and low-frequency light estimates."""
    if sigma <= 10:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    k = max(3, int(sigma * 1.4) | 1)
    return cv2.blur(img, (k, k))


def diffuse_fill(f, unknown, src_w, sigmas, dynamic=True):
    """Multi-scale diffusion fill: replace the `unknown` pixels of the
    float32 image `f` (H,W,3) from `src_w`-weighted surroundings, at
    increasing blur scales (small scales fill the boundary, large scales
    reach the interior).

    `src_w` is a float32 source-weight mask (1 = usable fill source).
    dynamic=True lets already-filled pixels act as sources for the next
    scale; dynamic=False keeps the source set fixed (used when the source
    mask was carefully built to exclude the whole repair neighborhood).

    Runs on a cropped ROI around the unknown region — the fill is local
    and full-canvas blur passes at large sigma dominated the render time
    (the ROI margin exceeds the largest blur support, so results are
    identical to a full-canvas fill). Mutates `f` and returns the mask of
    pixels that were filled (unknown pixels with too little support stay
    unfilled)."""
    filled = np.zeros_like(unknown)
    if not unknown.any():
        return filled
    ys, xs = np.where(unknown)
    h, w = unknown.shape
    mg = 96  # > largest blur support (sigma<=64 → box k=91, half=45)
    sl = np.s_[max(int(ys.min()) - mg, 0):min(int(ys.max()) + mg, h),
               max(int(xs.min()) - mg, 0):min(int(xs.max()) + mg, w)]
    fc = f[sl]
    u = unknown[sl].copy()
    sw = src_w[sl]
    for sigma in sigmas:
        if not u.any():
            break
        kn = (~u).astype(np.float32) * sw if dynamic else sw
        num = fast_blur(fc * kn[..., None], sigma)
        den = fast_blur(kn, sigma)
        upd = u & (den > 0.3)
        fc[upd] = (num / np.maximum(den[..., None], 1e-3))[upd]
        u[upd] = False
    f[sl] = fc
    filled[sl] = unknown[sl] & ~u
    return filled
