"""R55 (user on refine_s7: 衣服还是不太像, 笔触不太像, 没有腮红):

The SDXL torso comes out as a BLANK hoodie — none of the original's
hatched fold texture. But the R47 torso was already approved
(衣服算对的): it IS the original's pixels (classical remap), so its
clothes and strokes are the original's own. TRANSPLANT it:

  s7 head + arms-at-sides pose (approved)  +
  R47 torso (proportion-widened x1.25) pasted into the chest with
  the darkness-gated min-blend (white stays white, strokes carry) +
  clearly visible diagonal-hatch blush.

Pure classical, seconds. Output refine2_s7.png + compare.
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from trial_inpaint_eyes import OUT  # noqa: E402
import gen_frontal_sdxl as gx  # noqa: E402

gx.BODY_K = 1.15   # widen, but leave s7's arms visible at the sides
DY = 10            # shift the R47 torso down to meet s7's neck
MASK_X0, MASK_X1 = 95, 425


def _blush(img):
    out = img.copy().astype(np.float32)
    H, W = img.shape
    for cx, ang in ((212, 20), (302, -20)):
        hatch = np.full((H, W), 255, np.uint8)
        for i in range(-H, W + H, 5):
            x0 = i if ang > 0 else i + H
            cv2.line(hatch, (x0, 0),
                     (x0 + (H if ang > 0 else -H), H), 140, 1,
                     cv2.LINE_AA)
        m = np.zeros((H, W), np.float32)
        cv2.ellipse(m, (cx, 306), (28, 15), ang, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), 6) * 0.45
        dark = np.minimum(out, hatch.astype(np.float32))
        out = out * (1 - m) + dark * m
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    s7 = cv2.imread(os.path.join(OUT, "genxlp3_s7.png"), 0)
    r47 = cv2.imread(os.path.join(OUT, "eyes_r47_sd_s998.png"), 0)
    assert s7 is not None and r47 is not None
    H, W = s7.shape

    torso = gx._proportion_fix(r47)                   # widen body
    M = np.float32([[1, 0, 0], [0, 1, DY]])           # align collar
    torso = cv2.warpAffine(torso, M, (W, H),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=255)

    # chest mask hugging the neck: clears s7's own collar remnants
    # at the jaw SIDES while protecting the chin (x 218..302 below
    # y425); s7's arms hang OUTSIDE x 95..425 and survive
    m = np.zeros((H, W), np.float32)
    poly = np.array([(MASK_X0, 375), (218, 380), (232, 425),
                     (288, 425), (302, 380), (MASK_X1, 375),
                     (MASK_X1, H), (MASK_X0, H)], np.int32)
    cv2.fillPoly(m, [poly], 1.0)
    m = cv2.GaussianBlur(m, (0, 0), 10)

    out = s7.astype(np.float32)
    out = out * (1 - m) + 255.0 * m                   # clear chest
    a = np.clip((254.0 - torso.astype(np.float32)) / 60.0, 0, 1) * m
    out = out * (1 - a) + np.minimum(out,
                                     torso.astype(np.float32)) * a
    out = np.clip(out, 0, 255).astype(np.uint8)

    out = _blush(out)

    cv2.imwrite(os.path.join(OUT, "refine2_s7.png"), out)

    # side-by-side: original | R47 | s7 | result
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"), 0)
    panels = [("original", orig), ("R47 torso src", r47),
              ("s7 base", s7), ("refine2", out)]
    sheet = np.full((H, W * 4 + 30), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(OUT, "_refine2_sheet.png"), sheet)
    print("refine2 saved", flush=True)


if __name__ == "__main__":
    main()
