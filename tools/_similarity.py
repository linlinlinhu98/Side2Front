"""Same-person similarity vs 4/face.png (user criterion: 每张
生图后和原图比对, 同一人特征相似度 >= 60% 才算可以).

Metric: CLIP ViT-H/14 (cached, the IP-Adapter encoder) cosine
similarity — whole image + top/middle/bottom bands (hair / face
/ clothes), composite = mean of the four.

The classmate result (result/ce89...) and the user reference
(result/f156...) are scored too, as the calibration for what
'passing' looks like.

Usage: python tools/_similarity.py [name1.png name2.png ...]
(default: the current candidate set)
"""
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from trial_inpaint_eyes import OUT  # noqa: E402

DEFAULT = [
    "genr69_s7_eye.png", "genr69_s42_eye.png",
    "genr69_s998_eye.png", "genr69_s2024_eye.png",
    "genlora_ht_s7_eye.png", "genlora_full_s42_eye.png",
    "genplus4_ht_s7_eye.png", "genslim_torso_s7.png",
]


def _emb(enc, proc, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pv = proc(images=rgb, return_tensors="pt").pixel_values
    with torch.no_grad():
        e = enc(pv).pooler_output
    return (e / e.norm(dim=-1, keepdim=True)).numpy()[0]


def main():
    names = sys.argv[1:] or DEFAULT
    from transformers import (CLIPImageProcessor,
                              CLIPVisionModel)
    enc = CLIPVisionModel.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder",
        torch_dtype=torch.float32).eval()
    proc = CLIPImageProcessor()

    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    H, W = orig.shape[:2]
    bands = [(0, H // 3), (H // 3, 2 * H // 3), (2 * H // 3, H)]
    o_whole = _emb(enc, proc, orig)
    o_bands = [_emb(enc, proc, orig[y0:y1]) for y0, y1 in bands]

    rows = []
    cands = [(n, os.path.join(OUT, n)) for n in names]
    cands += [
        # classmate result cropped out of the photo-of-screen
        # (result/_classmate_panel.png); the raw photo scores
        # ~0.38 because 70% of its pixels are laptop/desk/wall
        ("CLASSMATE", os.path.join(
            ROOT, "result", "_classmate_panel.png")),
        ("REF_AI", os.path.join(
            ROOT, "result",
            "8aa591a6c9e652b79697d55eeee769ec.jpg")),
        # unrelated floors for scale calibration
        ("floor_people", os.path.join(ROOT, "4", "people.jpg")),
        ("floor_cat", os.path.join(ROOT, "4", "cat.jpg")),
    ]
    for name, path in cands:
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            continue
        im = cv2.resize(im, (W, H))
        s_whole = float(o_whole @ _emb(enc, proc, im))
        s_bands = [float(ob @ _emb(enc, proc, im[y0:y1]))
                   for ob, (y0, y1) in zip(o_bands, bands)]
        comp = (s_whole + sum(s_bands)) / 4
        rows.append((comp, name, s_whole, *s_bands))

    rows.sort(reverse=True)
    print(f"{'composite':>9} {'whole':>7} {'hair':>7} {'face':>7} "
          f"{'cloth':>7}  name")
    for comp, name, sw, s1, s2, s3 in rows:
        flag = " PASS" if comp >= 0.60 else ""
        print(f"{comp:9.3f} {sw:7.3f} {s1:7.3f} {s2:7.3f} "
              f"{s3:7.3f}  {name}{flag}")


if __name__ == "__main__":
    main()
