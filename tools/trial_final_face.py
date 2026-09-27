"""Final-face trial v1, per user review of composite_nose.png:
- eyes: mirrored profile wedges are NOT frontal eyes -> transplant
  the cf_s85 donor's real frontal almond eyes (interior route)
- nose: donor transplant (user approved)
- mouth: KEEP the composite's own (user said it's fine) -> restore
  the mouth zone from the nose-only version over the interior one
- brows: original pixels but 'too tight' -> shift each brow
  outward 8px and up 3px (feathered patch move)
- clothes: ORIGINAL pixels, no torso lightening (user: the clothes
  were correct before)
"""
import cv2
import json
import numpy as np
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from core.anime_face import AnimeFaceFrontalizer  # noqa: E402

DONOR_KPS = {
    "hairline_left": (330, 300), "hairline_center": (253, 290),
    "hairline_right": (180, 300),
    "brow_left_inner": (288, 350), "brow_left_outer": (360, 330),
    "eye_left_inner": (295, 380), "eye_left_outer": (355, 372),
    "eye_right_inner": (215, 380), "eye_right_outer": (158, 372),
    "nose_tip": (253, 458), "nose_left": (242, 465),
    "nose_right": (264, 465),
    "mouth_left": (235, 518), "mouth_center": (253, 521),
    "mouth_right": (272, 518),
    "chin_tip": (253, 598), "chin_left": (208, 572),
    "chin_right": (300, 572),
    "ear_top": (370, 355), "ear_bottom": (360, 445),
}

# face28 cand_sd_s85 (512x640), measured on _grid_sd_s85.png.
# Semi-realistic SD1.5 donor — style-matched to the original.
DONOR_KPS_SD85 = {
    "hairline_left": (330, 320), "hairline_center": (256, 300),
    "hairline_right": (180, 320),
    "brow_left_inner": (297, 348), "brow_left_outer": (355, 335),
    "eye_left_inner": (300, 388), "eye_left_outer": (350, 380),
    "eye_right_inner": (212, 388), "eye_right_outer": (162, 380),
    "nose_tip": (256, 468), "nose_left": (242, 472),
    "nose_right": (270, 472),
    "mouth_left": (228, 525), "mouth_center": (256, 528),
    "mouth_right": (284, 525),
    "chin_tip": (256, 592), "chin_left": (212, 565),
    "chin_right": (300, 565),
    "ear_top": (375, 360), "ear_bottom": (368, 445),
}

DONORS = {
    "cf": (os.path.join("assets", "ref_gen", "face27",
                        "cand_cf_s85.png"), DONOR_KPS, "v5"),
    "sd85": (os.path.join("assets", "ref_gen", "face28",
                          "cand_sd_s85.png"), DONOR_KPS_SD85, "v6"),
}


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 \
        else img


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "cf"
    donor_rel, donor_kps, tag = DONORS[which]
    img = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    mk = json.load(open(os.path.join(ROOT, "manual_keypoints.json"),
                        encoding="utf-8"))
    entry = mk["13ef60de932bb979aaa107ad210a327a"]
    kps = {k: tuple(v) for k, v in entry["keypoints"].items()}
    donor = cv2.imread(os.path.join(ROOT, donor_rel))

    f1 = AnimeFaceFrontalizer()
    f1.set_nose_reference(donor, donor_kps)
    nose_ver = _gray(f1.convert(img, keypoints=kps).image)

    f2 = AnimeFaceFrontalizer()
    f2.set_face_reference(donor, donor_kps)
    interior = _gray(f2.convert(img, keypoints=kps).image)

    # v3: mouth restore with WIDE feather (v1's 6px feather left a
    # muddy zone edge above the mouth)
    m = np.zeros(interior.shape, np.float32)
    m[272:306, 222:298] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), 10)
    out = (interior.astype(np.float32) * (1 - m)
           + nose_ver.astype(np.float32) * m).astype(np.uint8)

    # v5: back to v3's patch move (v4's redraw erased the eye tops
    # and its strokes were heavy black bars) but with WIDE paste
    # feather (10) and a post-soften of the band to hide the patch
    # rectangle edges v3 had.
    from trial_final_face2 import skin_erase
    for (x0, y0, x1, y1, dx, dy) in (
            (185, 183, 250, 203, -8, -3),
            (268, 183, 333, 203, 8, -3)):
        patch = out[y0:y1, x0:x1].copy()
        out = skin_erase(out, (x0, y0, x1, y1))
        pm = np.zeros(out.shape, np.float32)
        pm[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = 1.0
        pm = cv2.GaussianBlur(pm, (0, 0), 10)
        layer = np.full_like(out, 255)
        layer[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = patch
        out = (out.astype(np.float32) * (1 - pm)
               + layer.astype(np.float32) * pm).astype(np.uint8)
    # reinforce the moved brows with a light pencil stroke (val
    # 100 / 2px AA — v4's 45-55/4-5px were black bars). NO band
    # blurs: blurring the whole strip smeared brow ink sideways
    # into a continuous gray goggle-band; the feather-10 paste is
    # enough to hide patch edges.
    for (xi, yi, xo, yo) in ((242, 198, 179, 188),
                             (276, 198, 339, 188)):
        cv2.line(out, (xi, yi), (xo, yo), 100, 2, cv2.LINE_AA)

    outdir = os.path.join(ROOT, "assets", "geom", "final_v1")
    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(os.path.join(outdir, "final_" + tag + ".png"), out)

    tiles = []
    for name, im in (("nose_version", nose_ver), ("final_" + tag, out)):
        tile = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, 34),
                      (255, 255, 255), -1)
        cv2.putText(tile, name, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
        tiles.append(tile)
    cv2.imwrite(os.path.join(outdir, "sheet_" + tag + ".png"),
                np.hstack(tiles))
    crop = out[150:340, 140:390]
    cv2.imwrite(os.path.join(outdir, "_face_zoom_" + tag + ".png"),
                cv2.resize(crop, (500, 380),
                           interpolation=cv2.INTER_CUBIC))
    print("saved final_" + tag, flush=True)


if __name__ == "__main__":
    main()
