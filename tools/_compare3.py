"""3-way every-part comparison (user mandate: 每次生图后都
要和原图以及同学正面图仔细对比, 每个部分的特征都要对比).

Layout: rows = 11 parts (eyes/brows/nose/mouth/chin/ears/
hair/collar/zipper/straps/folds); columns = ORIGINAL |
CLASSMATE | candidate1 | candidate2 | ...

Usage: python tools/_compare3.py name1.png [name2.png ...]
(output: assets/geom/inpaint_eyes/_cmp3_<first>.png)
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from trial_inpaint_eyes import OUT  # noqa: E402

# (label, orig crop y0y1x0x1, ours crop y0y1x0x1)
PARTS = [
    ("eyes",   (175, 250, 105, 230), (255, 296, 150, 366)),
    ("brows",  (178, 210, 110, 220), (244, 268, 160, 356)),
    ("nose",   (245, 330, 30, 150),  (285, 325, 225, 295)),
    ("mouth",  (280, 330, 40, 135),  (305, 340, 220, 300)),
    ("chin",   (330, 410, 45, 180),  (350, 400, 190, 330)),
    ("ears",   (240, 330, 225, 320), (260, 325, 130, 390)),
    ("hair",   (55, 260, 30, 360),   (30, 230, 90, 430)),
    ("collar", (390, 500, 0, 520),   (395, 520, 110, 410)),
    ("zipper", (430, 660, 80, 220),  (440, 580, 220, 300)),
    ("straps", (420, 560, 300, 470), (430, 600, 60, 460)),
    ("folds",  (450, 660, 100, 400), (500, 660, 60, 460)),
]
# classmate panel crops (190x394 panel coords)
CLAS = {
    "eyes": (95, 135, 25, 165), "brows": (85, 108, 25, 165),
    "nose": (120, 160, 55, 140), "mouth": (150, 185, 55, 140),
    "chin": (170, 230, 45, 150), "ears": (100, 160, 5, 185),
    "hair": (45, 120, 10, 180), "collar": (195, 260, 0, 190),
    "zipper": (215, 320, 70, 130), "straps": (210, 330, 0, 190),
    "folds": (230, 394, 20, 170),
}
# the new AI reference (result/8aa...jpg) crops, as FRACTIONS
# (y0, y1, x0, x1) of its own size
REF_F = {
    "eyes": (0.30, 0.37, 0.28, 0.68),
    "brows": (0.26, 0.32, 0.28, 0.68),
    "nose": (0.35, 0.45, 0.40, 0.60),
    "mouth": (0.45, 0.51, 0.37, 0.63),
    "chin": (0.47, 0.57, 0.34, 0.66),
    "ears": (0.31, 0.45, 0.17, 0.83),
    "hair": (0.02, 0.34, 0.13, 0.87),
    "collar": (0.50, 0.65, 0.18, 0.86),
    "zipper": (0.58, 0.88, 0.42, 0.58),
    "straps": (0.54, 0.90, 0.04, 0.96),
    "folds": (0.65, 1.00, 0.15, 0.88),
}


def main():
    names = sys.argv[1:]
    if not names:
        print("usage: _compare3.py name1.png [name2.png ...]")
        return
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"), 0)
    clas = cv2.imread(os.path.join(
        ROOT, "result", "_classmate_panel.png"), 0)
    ref = cv2.imread(os.path.join(
        ROOT, "result",
        "8aa591a6c9e652b79697d55eeee769ec.jpg"), 0)
    RH, RW = ref.shape
    cands = []
    for n in names:
        im = cv2.imread(os.path.join(OUT, n), 0)
        if im is not None:
            cands.append((n.replace(".png", ""), im))

    TW, TH = 200, 90
    cols = ["ORIGINAL", "CLASSMATE", "REF(AI)"] \
        + [c[0] for c in cands]
    rows = []
    for label, oc, uc in PARTS:
        tiles = [cv2.resize(orig[oc[0]:oc[1], oc[2]:oc[3]],
                            (TW, TH),
                            interpolation=cv2.INTER_CUBIC)]
        cc = CLAS[label]
        tiles.append(cv2.resize(clas[cc[0]:cc[1], cc[2]:cc[3]],
                                (TW, TH),
                                interpolation=cv2.INTER_CUBIC))
        fy0, fy1, fx0, fx1 = REF_F[label]
        tiles.append(cv2.resize(
            ref[int(fy0 * RH):int(fy1 * RH),
                int(fx0 * RW):int(fx1 * RW)], (TW, TH),
            interpolation=cv2.INTER_CUBIC))
        for _, im in cands:
            tiles.append(cv2.resize(
                im[uc[0]:uc[1], uc[2]:uc[3]], (TW, TH),
                interpolation=cv2.INTER_CUBIC))
        r = np.full((TH, TW * len(tiles)
                     + 6 * (len(tiles) - 1)), 255, np.uint8)
        x = 0
        for t in tiles:
            r[:, x:x + TW] = t
            x += TW + 6
        rows.append((label, r))
    W = max(r.shape[1] for _, r in rows)
    sheet = np.full((26 + len(rows) * (TH + 28), W), 255,
                    np.uint8)
    x = 0
    for c in cols:
        cv2.putText(sheet, c[:22], (x + 4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2,
                    cv2.LINE_AA)
        x += TW + 6
    y = 26
    for label, r in rows:
        cv2.putText(sheet, label, (4, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2,
                    cv2.LINE_AA)
        sheet[y + 20:y + 20 + TH, :r.shape[1]] = r
        y += TH + 28
    out = os.path.join(OUT, f"_cmp3_{names[0].replace('.png', '')}.png")
    cv2.imwrite(out, sheet)
    print(f"[cmp3] saved {out}")


if __name__ == "__main__":
    main()
