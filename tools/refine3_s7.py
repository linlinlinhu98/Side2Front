"""R56 (user on refine2: 头和衣服没有拼接起来; 要和原图一模一样
的衣服/腮红/笔触; 现在的笔触比较模糊):

- Torso from base.png (PRE-fusion composite = the ORIGINAL's own
  sharp clothes pixels), widened x1.15, shifted by an INTEGER
  offset (zero resampling -> stays sharp).
- The collar now OVERLAPS the neck (full-replace paste below an
  under-chin contour) instead of abutting a whitened band — head
  and clothes actually join.
- Blush = the ORIGINAL cheek-hatch patch, mirrored onto both
  cheeks (darkness-gated) — literally the original's strokes.
- s7 head sharpened (unsharp) above the paste line.

Output refine3_s7.png + sheet.
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from trial_inpaint_eyes import OUT  # noqa: E402
import gen_frontal_sdxl as gx  # noqa: E402

gx.BODY_K = 1.15
DY = 40            # integer shift: collar comes up under the chin

# original blush patch (measured on 4/face.png): diagonal cheek
# hatching, x145-218 y248-292
BL_SRC = (248, 292, 145, 218)          # y0, y1, x0, x1
BL_DST = [(176, 284, False),           # left cheek  (x, y, mirror)
          (266, 284, True)]            # right cheek


def _paste_blush(img, orig):
    out = img.astype(np.float32)
    y0, y1, x0, x1 = BL_SRC
    patch = orig[y0:y1, x0:x1]
    ph, pw = patch.shape
    ey, ex = np.ogrid[:ph, :pw]
    ell = (((ex - pw / 2) / (pw * 0.46)) ** 2
           + ((ey - ph / 2) / (ph * 0.42)) ** 2)
    alpha_patch = np.clip(1.35 - ell, 0, 1).astype(np.float32)
    alpha_patch = cv2.GaussianBlur(alpha_patch, (0, 0), 3)
    for dx, dy, mir in BL_DST:
        p = patch[:, ::-1] if mir else patch
        a = alpha_patch * np.clip((254.0 - p.astype(np.float32))
                                  / 35.0, 0, 1)
        roi = out[dy:dy + ph, dx:dx + pw]
        out[dy:dy + ph, dx:dx + pw] = (
            roi * (1 - a) + np.minimum(roi, p.astype(np.float32)) * a)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    s7 = cv2.imread(os.path.join(OUT, "genxlp3_s7.png"), 0)
    base = cv2.imread(os.path.join(OUT, "base.png"), 0)
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"), 0)
    assert s7 is not None and base is not None and orig is not None
    H, W = s7.shape

    torso = gx._proportion_fix(base)                  # widen x1.15
    shifted = np.full_like(torso, 255)                # integer shift
    shifted[DY:, :] = torso[:-DY, :]

    # sharpen s7's head (above the paste contour)
    blur = cv2.GaussianBlur(s7, (0, 0), 1.5)
    sharp = cv2.addWeighted(s7, 1.8, blur, -0.8, 0)

    # paste contour hugs under the chin; full replace below it so
    # the torso's collar roll overlaps the neck base
    m = np.zeros((H, W), np.float32)
    poly = np.array([(90, 392), (150, 398), (210, 408), (256, 420),
                     (302, 408), (360, 398), (430, 392), (430, H),
                     (90, H)], np.int32)
    cv2.fillPoly(m, [poly], 1.0)
    m = cv2.GaussianBlur(m, (0, 0), 6)

    out = sharp.astype(np.float32) * (1 - m) \
        + shifted.astype(np.float32) * m
    out = np.clip(out, 0, 255).astype(np.uint8)

    out = _paste_blush(out, orig)

    cv2.imwrite(os.path.join(OUT, "refine3_s7.png"), out)

    panels = [("original", orig), ("s7 head", s7),
              ("base torso", base), ("refine3", out)]
    sheet = np.full((H, W * 4 + 30), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_refine3_sheet.png"), sheet)
    print("refine3 saved", flush=True)


if __name__ == "__main__":
    main()
