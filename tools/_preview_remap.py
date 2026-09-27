import os
import cv2
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from trial_inpaint_eyes import build_base, OUT  # noqa: E402

base, erased, *_ = build_base()
cv2.imwrite(os.path.join(OUT, "_remap_preview.png"), base)
cv2.imwrite(os.path.join(OUT, "_remap_preview_erased.png"), erased)
crop = base[300:660, 30:490]
cv2.imwrite(os.path.join(OUT, "_remap_zoom.png"),
            cv2.resize(crop, (690, 540),
                       interpolation=cv2.INTER_CUBIC))
print("saved preview", flush=True)
