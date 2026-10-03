# 一次性整合: 4 个共享库 -> gen_anime_common; trial_inpaint_eyes 内联 3 个符号;
# 恢复 gen_plus4; 删除 archive 与旧库. 全部逐字搬运, 重放链 BIT-IDENTICAL 不受影响.
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
ARCH = os.path.join(TOOLS, "archive")

COMMON = ("from gen_anime_common import CX, LORA_DIR, STEPS, TRIGGER, "
          "_lineart, _load_dpm, _scale_lora  # noqa: E402\n")
COMMON_REF = ("from gen_anime_common import STEPS, TRIGGER, _lineart, "
              "_load_dpm, _scale_lora  # noqa: E402\n")

OLD = ["from gen_cn_unify import _lineart",
       "from gen_r101 import _load_dpm",
       "from gen_r102 import STEPS",
       "from gen_r83 import LORA_DIR, TRIGGER, CX, _scale_lora",
       "from gen_r83 import TRIGGER, _scale_lora"]

CHAIN = ["gen_r119", "gen_r145", "gen_r146", "gen_r147", "gen_r148",
         "gen_r149", "gen_r150", "gen_r151", "gen_r152b", "gen_r153",
         "gen_r154", "gen_r155", "gen_r156"]

n_files, n_skip = 0, 0
for name in CHAIN:
    p = os.path.join(TOOLS, name + ".py")
    with open(p, encoding="utf-8") as f:
        lines = f.readlines()
    if not any(any(ln.startswith(o) for o in OLD) for ln in lines):
        if any("gen_anime_common" in ln for ln in lines):
            n_skip += 1          # 已处理过
        else:
            n_skip += 1          # 本来就不依赖旧库 (如 gen_r156 纯像素操作)
        continue
    out, done = [], False
    for ln in lines:
        if any(ln.startswith(o) for o in OLD):
            if not done:
                out.append(COMMON)
                done = True
            continue
        out.append(ln)
    assert done
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write("".join(out))
    n_files += 1
print(f"[1] {n_files} 个重放链脚本 import 已改写, {n_skip} 个跳过")

# gen_frontal_ref: 4 行 gen_* import -> 1 行
p = os.path.join(TOOLS, "gen_frontal_ref.py")
with open(p, encoding="utf-8") as f:
    lines = f.readlines()
if any("gen_anime_common" in ln for ln in lines):
    print("[2] gen_frontal_ref 已处理过, 跳过")
else:
    out, done = [], False
    for ln in lines:
        if any(ln.startswith(o) for o in OLD):
            if not done:
                out.append(COMMON_REF)
                done = True
            continue
        out.append(ln)
    assert done, "gen_frontal_ref 未匹配"
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write("".join(out))
    print("[2] gen_frontal_ref import 已改写")

# trial_inpaint_eyes: 内联 DONOR_KPS/_gray/skin_erase (逐字)
p = os.path.join(TOOLS, "trial_inpaint_eyes.py")
with open(p, encoding="utf-8") as f:
    t = f.read()
if "DONOR_KPS = {" in t and "trial_final_face" not in t:
    print("[3] trial_inpaint_eyes 已处理过, 跳过")
else:
    t = t.replace(
        "    from trial_final_face2 import skin_erase\n", "")
    INLINE = '''DONOR_KPS = {
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


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 \\
        else img


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
'''
    t = t.replace(
        "from trial_final_face import DONOR_KPS, _gray  # noqa: E402\n"
        "from trial_final_face2 import skin_erase  # noqa: E402\n",
        INLINE)
    assert "trial_final_face" not in t, "trial_inpaint_eyes 仍有外部引用"
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(t)
    print("[3] trial_inpaint_eyes 3 个符号已内联")

# 恢复 gen_plus4 (gen_lora_style 依赖)
shutil.move(os.path.join(ARCH, "gen_plus4.py"),
            os.path.join(TOOLS, "gen_plus4.py"))
print("[4] gen_plus4 已恢复到 tools/")

# 删除旧库与 archive
for lib in ("gen_r83.py", "gen_r101.py", "gen_r102.py",
            "gen_cn_unify.py"):
    os.remove(os.path.join(TOOLS, lib))
shutil.rmtree(ARCH)
print("[5] 4 个旧库 + tools/archive 已删除")
