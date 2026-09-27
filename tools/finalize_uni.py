"""R57 post: the unify pass fused the junction but (a) washed out
the cheek blush again, (b) left faint gray mottle in the
background. Fix both classically:
  - re-paste the ORIGINAL cheek-hatch patch (refine3_s7._paste_blush)
  - flood-whiten the connected near-white background
Outputs gen_uni_s30_f.png / gen_uni_s40_f.png + compares.
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from trial_inpaint_eyes import OUT  # noqa: E402
from refine3_s7 import _paste_blush  # noqa: E402


def _clean_bg(img):
    """whiten only the CONNECTED near-white background (flood fill
    from the borders), keeping light face/cloth shading intact"""
    near = (img > 230).astype(np.uint8)
    H, W = img.shape
    ff = np.zeros((H + 2, W + 2), np.uint8)
    flood = near.copy()
    for sx, sy in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1),
                   (W // 2, 0), (W // 2, H - 1)):
        if flood[sy, sx]:
            cv2.floodFill(flood, ff, (sx, sy), 2)
    bg = flood == 2
    out = img.copy()
    out[bg] = 255
    # border strips the flood can't reach (darker mottle hugging
    # the frame edges, above the shoulders = surely background)
    out[:460, :14][out[:460, :14] > 190] = 255
    out[:460, 505:][out[:460, 505:] > 190] = 255
    return out


def main():
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"), 0)
    for tag in ("s30", "s40"):
        g = cv2.imread(os.path.join(OUT, f"gen_uni_{tag}.png"), 0)
        assert g is not None
        g = _clean_bg(g)
        g = _paste_blush(g, orig)
        cv2.imwrite(os.path.join(OUT, f"gen_uni_{tag}_f.png"), g)
        print(f"[final] gen_uni_{tag}_f.png", flush=True)


if __name__ == "__main__":
    main()
