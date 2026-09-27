"""Trial: build the REAL final composite for face.png with the
round-25 s80 donor — (0) pure classical (100% original pixels),
(a) classical + donor nose-stroke transplant. Everything except the
nose linework is original-pixel: hair, garment, background, ears,
eyes, brows. This is the actual deliverable-quality image, not a
donor sheet."""
import cv2
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from core.anime_face import AnimeFaceFrontalizer  # noqa: E402

DONOR_KPS = {
    # face27 cand_cf_s85 (512x640), measured on a grid overlay.
    # Convention follows the pipeline: eye_left/brow_left = the
    # image-RIGHT feature; mouth_left/chin_left = image-LEFT.
    # Hairline is hidden under the 'brim' artifact -> approximate;
    # the interior-only fit barely uses it.
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


def main():
    img = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    mk = json.load(open(os.path.join(ROOT, "manual_keypoints.json"),
                        encoding="utf-8"))
    entry = mk["13ef60de932bb979aaa107ad210a327a"]
    kps = {k: tuple(v) for k, v in entry["keypoints"].items()}

    donor = cv2.imread(os.path.join(
        ROOT, "assets", "ref_gen", "face27", "cand_cf_s85.png"))

    outdir = os.path.join(ROOT, "assets", "geom", "donor_trial")
    os.makedirs(outdir, exist_ok=True)

    # (0) pure classical
    f0 = AnimeFaceFrontalizer()
    r0 = f0.convert(img, keypoints=kps)
    cv2.imwrite(os.path.join(outdir, "composite_classical.png"),
                r0.image)
    print("classical:", r0.info, flush=True)

    # (a) + donor nose strokes
    f1 = AnimeFaceFrontalizer()
    f1.set_nose_reference(donor, DONOR_KPS)
    r1 = f1.convert(img, keypoints=kps)
    cv2.imwrite(os.path.join(outdir, "composite_nose.png"), r1.image)
    print("nose-ref:", r1.info, flush=True)

    # (b) + donor full face-interior strokes
    f2 = AnimeFaceFrontalizer()
    f2.set_face_reference(donor, DONOR_KPS)
    r2 = f2.convert(img, keypoints=kps)
    cv2.imwrite(os.path.join(outdir, "composite_interior.png"),
                r2.image)
    print("interior-ref:", r2.info, flush=True)


if __name__ == "__main__":
    main()
