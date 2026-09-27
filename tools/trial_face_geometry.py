"""Geometry-corrected classical composite for face.png.

The pure-classical composite is 100% original pixels (identity,
clothes, background can never drift) but its mirrored face interior
has profile geometry: brow-to-hairline only ~16px (should be a full
third of the face), eyes too high, midface too long. This script
corrects the interior geometry with a keypoint TPS warp driven by
MEASURED frontal proportions (three equal thirds:
hairline->brow->nose bottom->chin), then blends the warp only
inside a feathered face mask. No generative model anywhere.
"""
import cv2
import json
import numpy as np
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from core.anime_face import AnimeFaceFrontalizer  # noqa: E402


def build_composite():
    img = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    mk = json.load(open(os.path.join(ROOT, "manual_keypoints.json"),
                        encoding="utf-8"))
    entry = mk["13ef60de932bb979aaa107ad210a327a"]
    kps = {k: tuple(v) for k, v in entry["keypoints"].items()}
    r = AnimeFaceFrontalizer().convert(img, keypoints=kps)
    return (cv2.cvtColor(r.image, cv2.COLOR_BGR2GRAY)
            if r.image.ndim == 3 else r.image)


def _move_patch(orig, canvas, rect, dy, feather=5):
    """Cut `rect` from `orig` and alpha-paste it `dy` px lower onto
    `canvas` (feathered edges). Patch translation never smears,
    unlike sparse-point TPS with >10 px moves."""
    x0, y0, x1, y1 = rect
    patch = orig[y0:y1, x0:x1].astype(np.float32)
    ny0, ny1 = y0 + dy, y1 + dy
    pm = np.zeros(orig.shape, np.float32)
    pm[ny0:ny1, x0:x1] = 1.0
    pm = cv2.GaussianBlur(pm, (0, 0), feather)
    layer = np.full_like(orig, 255, np.float32)
    layer[ny0:ny1, x0:x1] = patch
    canvas[:] = canvas * (1 - pm) + layer * pm


def main():
    comp = build_composite()
    h, w = comp.shape
    print("composite:", comp.shape, flush=True)
    outdir = os.path.join(ROOT, "assets", "geom")
    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(os.path.join(outdir, "comp_raw.png"), comp)

    # --- measured current geometry on the 520x660 composite:
    # fringe bottom ~184, brows 195-212, eyes 205-230, nose tip
    # ~250, mouth ~294, chin ~308. Frontal thirds need brow at
    # hairline + face/3 = 184+41 = 225; original profile's own
    # forehead fraction is 0.26 -> brow ~216. Move brow+eye blocks
    # down 18, nose +8, mouth UP 14 (mouth-chin gap is only 14px,
    # should be ~30).
    # v3: MINIMAL moves only. v2's brow+eye block move degraded the
    # eyes (mirrored profile eyes are dark wedges — fine at their
    # native spot, 'rat eyes' when pasted onto brighter skin) and
    # left a bright band. The glaring defects of the raw composite
    # are: mouth almost touching the chin line (gap ~14px, should
    # be ~30) and the nose sitting slightly high vs the corrected
    # mouth. So: nose +8, mouth -12, eyes/brows UNTOUCHED.
    RECTS = [  # (x0, y0, x1, y1, dy)
        (238, 238, 278, 264, 8),    # nose
        (230, 284, 284, 304, -12),  # mouth
    ]
    # 1) erase all feature rects, filling with a SKIN BASE (heavy
    # blur of the feature-erased image keeps the global shading
    # gradient — pure-white fill was visible as bright boxes)
    erased = comp.astype(np.float32)
    union = np.zeros(comp.shape, np.float32)
    for (x0, y0, x1, y1, dy) in RECTS:
        union[y0:y1, x0:x1] = 1.0
    union = cv2.GaussianBlur(union, (0, 0), 8)
    prelim = comp.astype(np.float32) * (1 - union) + 255.0 * union
    skin = cv2.GaussianBlur(prelim, (0, 0), 25)
    erased = comp.astype(np.float32) * (1 - union) \
        + skin * union
    # 2) paste patches at corrected positions (wide feather)
    canvas = erased.copy()
    for (x0, y0, x1, y1, dy) in RECTS:
        _move_patch(comp, canvas, (x0, y0, x1, y1), dy, feather=9)
    out = np.clip(canvas, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(outdir, "comp_geom.png"), out)

    dbg = cv2.cvtColor(comp, cv2.COLOR_GRAY2BGR)
    for (x0, y0, x1, y1, dy) in RECTS:
        cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.rectangle(dbg, (x0, y0 + dy), (x1, y1 + dy),
                      (0, 255, 0), 1)
    cv2.imwrite(os.path.join(outdir, "comp_debug.png"), dbg)
    print("saved comp_geom.png + comp_debug.png", flush=True)


if __name__ == "__main__":
    main()
