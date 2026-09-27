"""Similarity vs the AI reference (user mandate: 与
result/8aa...jpg 的相似度达到75%).

Same metric as _similarity.py (CLIP ViT-H cosine, whole +
top/mid/bottom bands, composite mean) but the anchor is the
AI reference instead of 4/face.png.

Usage: python tools/_similarity_ref.py name1.png [...]
"""
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from trial_inpaint_eyes import OUT  # noqa: E402

REF = os.path.join(ROOT, "result",
                   "8aa591a6c9e652b79697d55eeee769ec.jpg")


def _emb(enc, proc, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pv = proc(images=rgb, return_tensors="pt").pixel_values
    with torch.no_grad():
        e = enc(pv).pooler_output
    return (e / e.norm(dim=-1, keepdim=True)).numpy()[0]


def main():
    names = sys.argv[1:]
    if not names:
        print("usage: _similarity_ref.py name1.png [...]")
        return
    from transformers import (CLIPImageProcessor,
                              CLIPVisionModel)
    enc = CLIPVisionModel.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder",
        torch_dtype=torch.float32).eval()
    proc = CLIPImageProcessor()

    ref = cv2.imread(REF, cv2.IMREAD_COLOR)
    H, W = ref.shape[:2]
    bands = [(0, H // 3), (H // 3, 2 * H // 3), (2 * H // 3, H)]
    r_whole = _emb(enc, proc, ref)
    r_bands = [_emb(enc, proc, ref[y0:y1]) for y0, y1 in bands]

    rows = []
    for n in names:
        p = os.path.join(OUT, n)
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            p2 = os.path.join(ROOT, "result", n)
            im = cv2.imread(p2, cv2.IMREAD_COLOR)
        if im is None:
            continue
        im = cv2.resize(im, (W, H))
        sw = float(r_whole @ _emb(enc, proc, im))
        sb = [float(rb @ _emb(enc, proc, im[y0:y1]))
              for rb, (y0, y1) in zip(r_bands, bands)]
        rows.append(((sw + sum(sb)) / 4, n, sw, *sb))

    rows.sort(reverse=True)
    print(f"{'composite':>9} {'whole':>7} {'hair':>7} {'face':>7} "
          f"{'cloth':>7}  name (vs 8aa REF)")
    for comp, n, sw, s1, s2, s3 in rows:
        flag = " PASS75" if comp >= 0.75 else ""
        print(f"{comp:9.3f} {sw:7.3f} {s1:7.3f} {s2:7.3f} "
              f"{s3:7.3f}  {n}{flag}")


if __name__ == "__main__":
    main()
