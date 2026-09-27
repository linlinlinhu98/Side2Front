"""R156 (user: strap widths still differ and the left
strap's texture looks odd - model redraws drift every time;
switch to a DETERMINISTIC fix: mirror-transplant the right
strap's pixels onto the left):

The right strap (genr154_s42 look, approved) is bit-identical
in genr155_s7 (R155's mask only covered the left). Copy its
patch x382-438/y440-660, flip horizontally about the zipper
center (x -> 516-x), blend onto the left at x79-135 with
feathered edges. Width, tone and texture become EXACTLY the
right strap's by construction - no model, zero drift.
Output: genr156_s42.png. Pure cv2/numpy, no inference.
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from trial_inpaint_eyes import OUT  # noqa: E402

SX0, SX1 = 382, 438          # source patch (strap + margin)
Y0, Y1 = 440, 660
DX0 = 516 - SX1 + 1          # 79: mirror about x=258
FW = 7                       # x feather px
FH = 9                       # y feather px


def main():
    base = cv2.imread(os.path.join(OUT, "genr155_s7.png"), 0)
    assert base is not None
    H, W = base.shape

    patch = base[Y0:Y1, SX0:SX1][:, ::-1].astype(np.float32)
    ph, pw = patch.shape
    DX1 = DX0 + pw

    a = np.ones((ph, pw), np.float32)
    rx = np.linspace(0.0, 1.0, FW, dtype=np.float32)
    a[:, :FW] *= rx[None, :]
    a[:, -FW:] *= rx[::-1][None, :]
    ry = np.linspace(0.0, 1.0, FH, dtype=np.float32)
    a[:FH, :] *= ry[:, None]
    a[-FH:, :] *= ry[::-1][:, None]

    region = base[Y0:Y1, DX0:DX1].astype(np.float32)
    blended = region * (1 - a) + patch * a
    final = base.copy()
    final[Y0:Y1, DX0:DX1] = np.clip(blended, 0, 255) \
        .astype(np.uint8)

    name = "genr156_s42.png"
    cv2.imwrite(os.path.join(OUT, name), final)

    # verify: outside the dst rect everything bit-identical
    m = np.zeros((H, W), np.uint8)
    m[Y0:Y1, DX0:DX1] = 255
    unchanged = (final[m == 0] == base[m == 0]).mean()

    def _run(im, y, lo, hi, anchor):
        row = im[y]
        xs = [x for x in range(lo, hi) if row[x] < 165]
        if not xs:
            return None
        runs, cur = [], [xs[0]]
        for x in xs[1:]:
            if x - cur[-1] <= 2:
                cur.append(x)
            else:
                runs.append(cur)
                cur = [x]
        runs.append(cur)
        r = [r for r in runs if r[0] <= anchor <= r[-1]]
        return (r[0][0], r[0][-1]) if r else None

    print(f"[r156] unchanged_outside={unchanged:.3f}",
          flush=True)
    for y in (455, 480, 520, 560, 600, 640):
        rr = _run(final, y, 370, 455, 410)
        lr = _run(final, y, 55, 165, 106)
        print(f"[r156] y={y}: right={rr} left={lr} "
              f"(mirror expect {516 - rr[1]}..{516 - rr[0]})"
              if rr else f"[r156] y={y}: right={rr} left={lr}",
              flush=True)

    lt = final[480:640, 88:124].mean()
    rt = final[480:640, 392:428].mean()
    print(f"[r156] tone left={lt:.0f} right={rt:.0f} "
          f"diff={abs(lt - rt):.0f}", flush=True)

    # sheet 1: before | after left-strap zoom
    panels = [("before", base), ("after", final)]
    cz = [cv2.resize(im[420:660, 40:170], None, fx=2.4,
                     fy=2.4, interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    ch, cw = cz[0].shape
    sheet = np.full((ch, cw * 2 + 8), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, cz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + cw] = t
        x += cw + 8
    cv2.imwrite(os.path.join(OUT, "_genr156_lstrap.png"),
                sheet)
    # sheet 2: full torso symmetry view
    crop = final[420:660, 20:500]
    z = cv2.resize(crop, None, fx=1.6, fy=1.6,
                   interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(OUT, "_genr156_sym.png"), z)
    print("[r156] sheets saved", flush=True)


if __name__ == "__main__":
    main()
