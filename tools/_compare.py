"""Build the 3-way comparison sheet (original | classmate | ours)
plus a neck zoom row — required by the user: every result must be
compared against the original before presenting.
Usage: python tools/_compare.py eyes_r42_sd_s998.png
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "geom", "inpaint_eyes")

name = sys.argv[1] if len(sys.argv) > 1 else "eyes_r42_sd_s998.png"
tag = os.path.splitext(name)[0]

orig = cv2.imread(os.path.join(ROOT, "4", "face.png"), 0)
cm = cv2.imread(os.path.join(OUT, "_classmate_torso.png"), 0)
ours = cv2.imread(os.path.join(OUT, name), 0)
assert ours is not None, name

H = 660


def sc(im, h):
    return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h),
                      interpolation=cv2.INTER_AREA)


panels = [sc(orig, H), sc(cm, H), sc(ours, H)]
W = sum(p.shape[1] for p in panels) + 20
sheet = np.full((H, W), 255, np.uint8)
x = 0
for p in panels:
    sheet[:, x:x + p.shape[1]] = p
    x += p.shape[1] + 10
cv2.imwrite(os.path.join(OUT, f"_compare_{tag}.png"), sheet)

# neck row: original neck | classmate neck | our neck
on = orig[280:520, 100:420]
cn = cm[550:950, 150:600]
rn = ours[300:480, 140:380]
row = [sc(on, 360), sc(cn, 360), sc(rn, 360)]
W2 = sum(p.shape[1] for p in row) + 20
sheet2 = np.full((360, W2), 255, np.uint8)
x = 0
for p in row:
    sheet2[:, x:x + p.shape[1]] = p
    x += p.shape[1] + 10
cv2.imwrite(os.path.join(OUT, f"_compare_neck_{tag}.png"), sheet2)

# hair row: original hair | classmate hair | our hair
oh = orig[0:260, 30:450]
ch = cm[40:400, 120:580]
rh = ours[0:260, 40:470]
row = [sc(oh, 430), sc(ch, 430), sc(rh, 430)]
W3 = sum(p.shape[1] for p in row) + 20
sheet3 = np.full((430, W3), 255, np.uint8)
x = 0
for p in row:
    sheet3[:, x:x + p.shape[1]] = p
    x += p.shape[1] + 10
cv2.imwrite(os.path.join(OUT, f"_compare_hair_{tag}.png"), sheet3)
print("compare sheets saved", flush=True)
