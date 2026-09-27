"""Final-face trial v2. v1 proved the direction (donor frontal
eyes) but had: rectangular transplant-zone edges (paper-blur
flattened skin), old wedge-eye ghosts, brow-move ghosts, mouth
restore mud. v2 rebuilds with clean compositing on top of the
APPROVED nose version:
1) erase the mirrored wedge eyes, fill with a SKIN BASE (blur-25
   of the pre-erased image — no flat boxes, keeps shading gradient)
2) extract ONLY the donor eye dark strokes from the interior-
   transplant result (high-pass > 5 inside per-eye ellipses) and
   lay them on the erased base
3) brows outward 8/up 3 with the same skin-base erase (no ghost)
4) nose + mouth: untouched from the nose version (user approved)
"""
import cv2
import json
import numpy as np
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from core.anime_face import AnimeFaceFrontalizer  # noqa: E402
from trial_final_face import DONOR_KPS, _gray  # noqa: E402


def skin_erase(img, rect, feather=6):
    """Erase rect, fill with blur-25 skin base of the pre-erased
    image (preserves the global shading gradient)."""
    x0, y0, x1, y1 = rect
    m = np.zeros(img.shape, np.float32)
    m[y0:y1, x0:x1] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), feather)
    prelim = img.astype(np.float32) * (1 - m) + 255.0 * m
    skin = cv2.GaussianBlur(prelim, (0, 0), 25)
    return (img.astype(np.float32) * (1 - m)
            + skin * m).astype(np.uint8)


def main():
    img = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    mk = json.load(open(os.path.join(ROOT, "manual_keypoints.json"),
                        encoding="utf-8"))
    entry = mk["13ef60de932bb979aaa107ad210a327a"]
    kps = {k: tuple(v) for k, v in entry["keypoints"].items()}
    donor = cv2.imread(os.path.join(
        ROOT, "assets", "ref_gen", "face27", "cand_cf_s85.png"))

    f1 = AnimeFaceFrontalizer()
    f1.set_nose_reference(donor, DONOR_KPS)
    nose_ver = _gray(f1.convert(img, keypoints=kps).image)

    f2 = AnimeFaceFrontalizer()
    f2.set_face_reference(donor, DONOR_KPS)
    interior = _gray(f2.convert(img, keypoints=kps).image)

    out = nose_ver.copy()

    # 1) erase wedge eyes (measured on the 520x660 composite)
    for rect in ((192, 200, 243, 234), (270, 200, 321, 234)):
        out = skin_erase(out, rect)

    # 2) lay donor eye strokes extracted from the interior result
    blur8 = cv2.GaussianBlur(interior, (0, 0), 8)
    hp = np.clip(blur8.astype(np.float32) - interior.astype(np.float32),
                 0, None)
    hp[hp < 5] = 0
    for ecx in (215, 297):
        zone = np.zeros(out.shape, np.float32)
        cv2.ellipse(zone, (ecx, 217), (30, 15), 0, 0, 360, 1.0, -1)
        zone = cv2.GaussianBlur(zone, (0, 0), 3)
        out = np.clip(out.astype(np.float32) - hp * zone * 1.25,
                      0, 255).astype(np.uint8)

    # 3) brows outward/up with skin-base erase
    for (x0, y0, x1, y1, dx, dy) in (
            (185, 188, 250, 213, -8, -3),
            (268, 188, 333, 213, 8, -3)):
        patch = out[y0:y1, x0:x1].copy()
        out = skin_erase(out, (x0, y0, x1, y1))
        pm = np.zeros(out.shape, np.float32)
        pm[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = 1.0
        pm = cv2.GaussianBlur(pm, (0, 0), 5)
        layer = np.full_like(out, 255)
        layer[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = patch
        out = (out.astype(np.float32) * (1 - pm)
               + layer.astype(np.float32) * pm).astype(np.uint8)

    outdir = os.path.join(ROOT, "assets", "geom", "final_v1")
    cv2.imwrite(os.path.join(outdir, "final_v2.png"), out)
    tiles = []
    for name, im in (("nose_version", nose_ver), ("final_v2", out)):
        tile = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, 34),
                      (255, 255, 255), -1)
        cv2.putText(tile, name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
        tiles.append(tile)
    cv2.imwrite(os.path.join(outdir, "sheet_v2.png"),
                np.hstack(tiles))
    crop = out[150:340, 140:390]
    cv2.imwrite(os.path.join(outdir, "_face_zoom2.png"),
                cv2.resize(crop, (500, 380),
                           interpolation=cv2.INTER_CUBIC))
    print("saved final_v2.png + sheet_v2.png", flush=True)


if __name__ == "__main__":
    main()
