"""Torso fix prototype for the face.png classical composite.

The mirrored jacket reads as TWO BODIES (user feedback on
composite_nose.png): (1) heavy doubled seam stripes on the upper
arms, (2) a dark diagonal fold cluster on each side forming its own
torso outline, (3) two long vertical fold lines acting as 'outer
edges' of two torsos, (4) a flat white center placket splitting the
garment. Fix: feathered lightening of (1)(2)(3) + faint horizontal
drape lines crossing the center strip so both halves read as ONE
jacket; thin zipper redrawn on top.
"""
import cv2
import numpy as np
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "assets", "geom", "donor_trial",
                   "composite_nose.png")
DST = os.path.join(ROOT, "assets", "geom", "donor_trial",
                   "composite_nose_torso.png")


def soft_keep(img, mask, keep, feather):
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather)
    factor = 1.0 - (1.0 - keep) * mask
    return np.clip(255 - (255 - img.astype(np.float32)) * factor,
                   0, 255).astype(np.uint8)


def main():
    g = cv2.imread(SRC, cv2.IMREAD_GRAYSCALE)
    assert g is not None
    h, w = g.shape

    # (1) arm seam stripes
    for x0, x1 in ((90, 210), (315, 435)):
        m = np.zeros(g.shape, np.float32)
        m[400:435, x0:x1] = 1.0
        g = soft_keep(g, m, 0.25, 6)
    # (2) diagonal fold clusters
    for cx, ang in ((185, -20), (335, 20)):
        m = np.zeros(g.shape, np.float32)
        cv2.ellipse(m, (cx, 455), (65, 95), ang, 0, 360, 1.0, -1)
        g = soft_keep(g, m, 0.35, 10)
    # (3) long vertical 'torso edge' fold lines
    for x0, x1 in ((155, 190), (335, 370)):
        m = np.zeros(g.shape, np.float32)
        m[440:640, x0:x1] = 1.0
        g = soft_keep(g, m, 0.30, 8)
    # (4) v2: NO zipper redraw, NO drape lines — v1's redrawn
    # zipper was a heavy black bar and the drape lines read as
    # ladder rungs. The original thin zipper survives untouched
    # (all lightening zones are lateral); the 'two bodies' read
    # came from the side clusters/stripes, not the placket.

    cv2.imwrite(DST, g)
    print("saved", DST)


if __name__ == "__main__":
    main()
