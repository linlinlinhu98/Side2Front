"""Automatic anime-face keypoint detection.

Wraps hysts/anime-face-detector (YOLOv3 face detector + HRNetV2 28-landmark
model, pure PyTorch, CPU-friendly) and maps its 28 landmarks onto the
17-keypoint template used by the anime TPS frontalization pipeline.

Weights are downloaded once from HuggingFace Hub and cached locally
(same cache as the GAN frontalization model), so runtime stays offline.
"""

import os

# Resolve the shared HF cache the same way the rest of the app does; fall
# back to the known local cache so runtime stays offline.
if "HF_HOME" not in os.environ and "HUGGINGFACE_HUB_CACHE" not in os.environ:
    if os.path.isdir(r"D:\huggingface_cache"):
        os.environ["HF_HOME"] = r"D:\huggingface_cache"

import numpy as np

_detector = None
_available = None


def _get_detector():
    """Lazily create the detector (first call loads torch + weights)."""
    global _detector, _available
    if _detector is not None:
        return _detector
    if _available is False:
        return None
    try:
        from anime_face_detector import create_detector
        _detector = create_detector("yolov3", device="cpu")
        _available = True
    except Exception as e:
        print(f"AnimeDetector: unavailable ({e})")
        _available = False
        _detector = None
    return _detector


def is_available() -> bool:
    return _get_detector() is not None


def _pick_group(kp: np.ndarray, a: tuple, b: tuple) -> np.ndarray:
    """Pick the landmark group with higher mean confidence.

    On a profile the occluded-side landmarks collapse onto the silhouette
    with low confidence; the drawn (visible) side scores high. This is
    robust to the left/right indexing convention of the model.
    """
    if kp[list(a), 2].mean() >= kp[list(b), 2].mean():
        return kp[list(a)]
    return kp[list(b)]


def detect_keypoints(image: np.ndarray):
    """Detect an anime face and return keypoints in the GUI template.

    Args:
        image: BGR image (H, W, 3).

    Returns:
        (keypoints, info) where keypoints maps template names to (x, y)
        pixel coordinates, or (None, reason) if detection failed.
    """
    det = _get_detector()
    if det is None:
        return None, "检测器不可用（anime-face-detector 未安装或模型缺失）"

    preds = det(image)
    if not preds:
        return None, "未检测到动漫人脸"

    pred = max(preds, key=lambda p: p["bbox"][4])
    bbox = np.asarray(pred["bbox"], dtype=np.float64)
    kp = np.asarray(pred["keypoints"], dtype=np.float64)  # (28, 3)

    if bbox[4] < 0.5:
        return None, f"人脸置信度过低 ({bbox[4]:.2f})"

    # 28-point layout: 0-4 face contour (2=chin, 3/4=jaw/ear on the face's
    # left side, 0/1 on the right side), 5-10 brows (3+3), 11-22 eyes
    # (6+6), 23 nose, 24-27 mouth (corner, upper, corner, lower).
    eye = _pick_group(kp, tuple(range(11, 17)), tuple(range(17, 23)))
    brow = _pick_group(kp, (5, 6, 7), (8, 9, 10))

    if eye[:, 2].mean() < 0.35:
        return None, f"眼部关键点置信度过低 ({eye[:, 2].mean():.2f})"

    nose = kp[23, :2]
    chin = kp[2, :2]
    ear = kp[4, :2] if kp[4, 2] >= kp[0, 2] else kp[0, :2]
    jaw = kp[3, :2] if kp[3, 2] >= kp[1, 2] else kp[1, :2]
    mouth = kp[24:28, :2]

    # Eye/brow corners: index 0 and 2 of each group are the two corners;
    # the inner one sits closer to the nose silhouette.
    if abs(eye[0, 0] - nose[0]) <= abs(eye[2, 0] - nose[0]):
        eye_inner, eye_outer = eye[0, :2], eye[2, :2]
    else:
        eye_inner, eye_outer = eye[2, :2], eye[0, :2]
    if abs(brow[0, 0] - nose[0]) <= abs(brow[2, 0] - nose[0]):
        brow_inner, brow_outer = brow[0, :2], brow[2, :2]
    else:
        brow_inner, brow_outer = brow[2, :2], brow[0, :2]

    # Mouth: corners are entries 0 and 2, lip midpoints 1 and 3. The
    # template's center point must sit between the corners (the detected
    # lip midpoints lie on the silhouette and would shear the mouth).
    corners = sorted((mouth[0], mouth[2]), key=lambda p: p[0])
    mouth_center = np.array([(corners[0][0] + corners[1][0]) / 2,
                             (mouth[1, 1] + mouth[3, 1]) / 2])

    kps = {
        "eye_left_inner": tuple(eye_inner),
        "eye_left_outer": tuple(eye_outer),
        "brow_left_inner": tuple(brow_inner),
        "brow_left_outer": tuple(brow_outer),
        "nose_tip": tuple(nose),
        "mouth_left": tuple(corners[0]),
        "mouth_center": tuple(mouth_center),
        "mouth_right": tuple(corners[1]),
        "chin_tip": tuple(chin),
        "ear_top": tuple(ear),
        "ear_bottom": (float(ear[0] + 0.2 * (jaw[0] - ear[0])),
                       float(ear[1] + 0.5 * (jaw[1] - ear[1]))),
    }

    # The detector provides no forehead/hairline landmarks — synthesize
    # them from the face box and brow position (TPS anchors for the hair).
    bx1, by1, bx2, by2 = bbox[:4]
    bh = by2 - by1
    kps["hairline_left"] = (float(brow_inner[0]), float(by1 + 0.10 * bh))
    kps["hairline_center"] = (float((brow_inner[0] + ear[0]) / 2),
                              float(by1 - 0.04 * bh))
    kps["hairline_right"] = (float(ear[0] - 0.25 * (ear[0] - brow_outer[0])),
                             float(by1 + 0.02 * bh))

    # Chin sides: the silhouette-side point stays close to the chin curve
    # (placing it far forward stretches the chin outline into a bulge);
    # the face-side point carries the jaw width.
    cx = float(np.mean([nose[0], mouth_center[0], chin[0]]))
    half_w = max(abs(ear[0] - cx), 20.0)
    if corners[0][0] < cx:  # face looks left: silhouette side is image-left
        kps["chin_left"] = (float((corners[0][0] + chin[0]) / 2),
                            float((corners[0][1] + chin[1]) / 2))
        kps["chin_right"] = (float(chin[0] + 0.22 * half_w),
                             float(chin[1] - 0.30 * (chin[1] - mouth_center[1])))
    else:
        kps["chin_right"] = (float((corners[1][0] + chin[0]) / 2),
                             float((corners[1][1] + chin[1]) / 2))
        kps["chin_left"] = (float(chin[0] - 0.22 * half_w),
                            float(chin[1] - 0.30 * (chin[1] - mouth_center[1])))

    info = (f"自动检测: 置信度 {bbox[4]:.2f}, "
            f"眼部置信度 {eye[:, 2].mean():.2f}, {len(kps)}个关键点")
    return kps, info
