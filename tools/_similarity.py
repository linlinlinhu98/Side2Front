"""Similarity scorer, CLIP ViT-H/14 (cached, the IP-Adapter
encoder): cosine similarity of the whole image + top/middle/bottom
bands (hair / face / clothes), composite = mean of the four.

Two anchors (merged from _similarity.py + _similarity_ref.py):
  default   vs the original 4/face.png (user criterion: 每张生图
            后和原图比对, 同一人特征相似度 >= 60% 才算可以);
            with no names it scores the historical candidate set
            plus calibration floors (classmate screen-photo crop
            scores ~0.38 raw because 70% of its pixels are
            laptop/desk/wall; unrelated people/cat).
  --ref     vs the AI reference result/8aa...jpg (user mandate:
            与 AI 参考相似度 >= 75%); names resolved in OUT
            then result/.

Usage: python tools/_similarity.py [--ref] [name1.png ...]
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
    args = sys.argv[1:]
    vs_ref = "--ref" in args
    names = [a for a in args if a != "--ref"]
    if vs_ref and not names:
        print("usage: _similarity.py --ref name1.png [...]")
        return
    if not names:
        names = DEFAULT

    from transformers import (CLIPImageProcessor,
                              CLIPVisionModel)
    enc = CLIPVisionModel.from_pretrained(
        "h94/IP-Adapter", subfolder="models/image_encoder",
        torch_dtype=torch.float32).eval()
    proc = CLIPImageProcessor()

    anchor = cv2.imread(REF if vs_ref
                        else os.path.join(ROOT, "4", "face.png"),
                        cv2.IMREAD_COLOR)
    H, W = anchor.shape[:2]
    bands = [(0, H // 3), (H // 3, 2 * H // 3), (2 * H // 3, H)]
    a_whole = _emb(enc, proc, anchor)
    a_bands = [_emb(enc, proc, anchor[y0:y1]) for y0, y1 in bands]

    rows = []
    cands = [(n, os.path.join(OUT, n)) for n in names]
    cands += [(n, os.path.join(ROOT, "result", n)) for n in names]
    if not vs_ref:
        cands += [
            # classmate result cropped out of the photo-of-screen
            # (result/_classmate_panel.png); the raw photo scores
            # ~0.38 because 70% of its pixels are laptop/desk/wall
            ("CLASSMATE", os.path.join(
                ROOT, "result", "_classmate_panel.png")),
            ("REF_AI", REF),
            # unrelated floors for scale calibration
            ("floor_people", os.path.join(ROOT, "4", "people.jpg")),
            ("floor_cat", os.path.join(ROOT, "4", "cat.jpg")),
        ]
    for name, path in cands:
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            continue
        im = cv2.resize(im, (W, H))
        s_whole = float(a_whole @ _emb(enc, proc, im))
        s_bands = [float(ab @ _emb(enc, proc, im[y0:y1]))
                   for ab, (y0, y1) in zip(a_bands, bands)]
        comp = (s_whole + sum(s_bands)) / 4
        rows.append((comp, name, s_whole, *s_bands))

    rows.sort(reverse=True)
    tag = "vs 8aa REF" if vs_ref else "vs 4/face.png"
    print(f"{'composite':>9} {'whole':>7} {'hair':>7} {'face':>7} "
          f"{'cloth':>7}  name ({tag})")
    for comp, name, sw, s1, s2, s3 in rows:
        flag = (" PASS75" if comp >= 0.75 else "") if vs_ref \
            else (" PASS" if comp >= 0.60 else "")
        print(f"{comp:9.3f} {sw:7.3f} {s1:7.3f} {s2:7.3f} "
              f"{s3:7.3f}  {name}{flag}")


if __name__ == "__main__":
    main()
