import os
import sys
import time
import numpy as np
import cv2

# 3DDFA_V2 paths
_3DDFA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "3ddfa_v2")
TDDFA_AVAILABLE = False
_FACEBOXES_AVAILABLE = False

# Top-level imports — resolved once at module load
try:
    sys.path.insert(0, _3DDFA_ROOT)
    import yaml
    from FaceBoxes.FaceBoxes import FaceBoxes
    from TDDFA import TDDFA
    from utils.tddfa_util import _parse_param
    TDDFA_AVAILABLE = True
    _FACEBOXES_AVAILABLE = True
except ImportError:
    TDDFA_AVAILABLE = False
    _FACEBOXES_AVAILABLE = False

# GAN-based frontalizer (synthesizes new pixels for occluded regions)
GAN_AVAILABLE = False
try:
    from .gan_frontalizer import GANFrontalizer
    GAN_AVAILABLE = True
except ImportError:
    GAN_AVAILABLE = False

from .base import FaceFrontalizer, FrontalizationResult
from .fill_utils import fast_blur as _fast_blur, diffuse_fill as _diffuse_fill

# Debug stage dumps (S2F_DEBUG=1) land in the repo's 4/ folder.
_DBG_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "4"))


def _dbg_dump(name, img):
    """Write a render-stage image for debugging (S2F_DEBUG=1 only)."""
    if os.environ.get("S2F_DEBUG"):
        cv2.imwrite(os.path.join(_DBG_DIR, name), img)


def _no_grad():
    """torch.no_grad() without importing torch eagerly (null context when
    torch is absent). The GUI also imports the anime pipeline, which can
    leave grad enabled globally — without this guard the 3DDFA regression
    fails with "Can't call numpy() on Tensor that requires grad"."""
    try:
        import torch
        return torch.no_grad()
    except ImportError:
        import contextlib
        return contextlib.nullcontext()


def _with_forehead(lm: np.ndarray) -> np.ndarray:
    """Extend the 68-landmark set upward with a synthetic forehead arc so
    a hull built from it covers the whole face (the 68 landmarks stop at
    the eyebrows). Used by both the GAN paste-back and the ghost-chin
    erasure."""
    brow = lm[17:27].copy()
    brow_y = float(brow[:, 1].mean())
    chin_y = float(lm[8, 1])
    ext = brow.copy()
    ext[:, 1] -= 0.45 * (chin_y - brow_y)
    return np.vstack([lm, ext])


class RealFaceFrontalizer(FaceFrontalizer):
    name = "real"
    display_name = "真实人脸"

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = _3DDFA_ROOT

        cfg_path = os.path.join(model_dir, "configs", "mb1_120x120.yml")
        self._model_dir = model_dir
        self._initialized = False
        self._tddfa = None
        self._face_boxes = None
        self._gan_frontalizer = None

        # Default pose params (user-adjustable yaw/pitch/roll)
        self._yaw = 0
        self._pitch = 0
        self._roll = 0
        self._texture_completion = True
        self._edge_smooth = True
        # Strength of the full-res photo-detail boost (0 disables): the
        # vertex-color render is mesh-resolution limited, so the original
        # photo's own high frequencies are re-added where the render holds
        # real, visible, unedited texture.
        self._photo_detail = 0.7

        # Detected pose (set after first successful detection)
        self._detected_yaw = 0
        self._detected_pitch = 0
        self._detected_roll = 0
        self._pose_initialized = False
        self._sym_map = None  # lazily-built BFM bilateral-symmetry vertex map

        # Per-image caches: the 3DDFA fit + texture sampling and the GAN
        # warp are slider-independent — with them cached, dragging a
        # rotation slider only re-renders (sub-second) instead of
        # re-running the whole multi-second pipeline.
        self._boxes_key = None
        self._boxes_cache = None
        self._fit_key = None
        self._fit_cache = None
        self._gan_key = None
        self._gan_cache = None

        # Try to initialize GAN frontalizer (synthesizes new pixels for occluded regions)
        if GAN_AVAILABLE:
            try:
                from .gan_frontalizer import GANFrontalizer
                self._gan_frontalizer = GANFrontalizer(device=None)
                if self._gan_frontalizer.is_initialized:
                    print("RealFaceFrontalizer: GAN frontalizer ready (synthesizes occluded regions)")
                else:
                    self._gan_frontalizer = None
            except Exception as e:
                print(f"GAN frontalizer init failed: {e}")
                self._gan_frontalizer = None

        # Fall back to 3DDFA_V2
        if not TDDFA_AVAILABLE or not os.path.exists(cfg_path):
            if self._gan_frontalizer is None:
                self._initialized = False
            return

        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)

            cfg["gpu_mode"] = False  # Force CPU

            # Fix relative paths in YAML -> absolute paths relative to model_dir (3ddfa_v2/)
            if cfg.get("bfm_fp", "").startswith("configs/"):
                cfg["bfm_fp"] = os.path.normpath(os.path.join(model_dir, cfg["bfm_fp"]))
            if cfg.get("checkpoint_fp", "").startswith("weights/"):
                cfg["checkpoint_fp"] = os.path.normpath(os.path.join(model_dir, cfg["checkpoint_fp"]))

            self._tddfa = TDDFA(**cfg)
            self._face_boxes = FaceBoxes()
            self._initialized = True
        except Exception as e:
            print(f"3DDFA_V2 init failed: {e}")
            if self._gan_frontalizer is None:
                self._initialized = False

    def get_default_params(self) -> dict:
        return {
            "yaw": self._yaw,
            "pitch": self._pitch,
            "roll": self._roll,
            "texture_completion": self._texture_completion,
            "edge_smooth": self._edge_smooth,
        }

    def get_detected_pose(self) -> tuple:
        """Return the detected yaw, pitch, roll after first conversion."""
        return self._detected_yaw, self._detected_pitch, self._detected_roll

    def update_params(self, **kwargs) -> None:
        if "yaw" in kwargs:
            self._yaw = float(kwargs["yaw"])
        if "pitch" in kwargs:
            self._pitch = float(kwargs["pitch"])
        if "roll" in kwargs:
            self._roll = float(kwargs["roll"])
        if "texture_completion" in kwargs:
            self._texture_completion = bool(kwargs["texture_completion"])
        if "edge_smooth" in kwargs:
            self._edge_smooth = bool(kwargs["edge_smooth"])
        if "photo_detail" in kwargs:
            self._photo_detail = float(kwargs["photo_detail"])

    def _detect_face(self, image):
        key = self._image_key(image)
        if self._boxes_key == key:
            return self._boxes_cache
        boxes = self._detect_face_uncached(image)
        self._boxes_key = key
        self._boxes_cache = boxes
        return boxes

    def _image_key(self, image):
        return (image.shape, float(image[::32, ::32].sum()))

    def _detect_face_uncached(self, image):
        if not self._initialized or self._face_boxes is None:
            try:
                cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                boxes_raw = cascade.detectMultiScale(gray, 1.1, 5)
                return [[x, y, x + w, y + h] for x, y, w, h in boxes_raw]
            except Exception:
                return []
        try:
            boxes = self._face_boxes(image)
            return [b[:4] for b in boxes]
        except Exception:
            return []

    def convert(self, image: np.ndarray, **kwargs) -> FrontalizationResult:
        t0 = time.time()

        if image is None:
            return FrontalizationResult(
                np.zeros((100, 100, 3), dtype=np.uint8), 0, "输入为空"
            )

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        elapsed = (time.time() - t0) * 1000
        h, w = image.shape[:2]

        # Detect faces
        boxes = self._detect_face(image)

        # Determine if likely a human face (large, reasonably centered)
        # NB: detector boxes are [x1, y1, x2, y2], NOT [x, y, w, h].
        likely_human = False
        if boxes:
            bx1, by1, bx2, by2 = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            bw, bh = bx2 - bx1, by2 - by1
            face_cx, face_cy = bx1 + bw//2, by1 + bh//2
            face_size = max(bw, bh)
            likely_human = (face_size > min(h, w) * 0.05 and
                            abs(face_cx - w//2) < w * 0.35 and
                            abs(face_cy - h//2) < h * 0.35)

        # --- Path 1: frontal target on a human face → identity-preserving hybrid ---
        # The 3DMM re-render keeps the subject's REAL identity/texture; the GAN
        # only synthesizes pixels where the 3DMM has no real texture (occluded
        # side, clutter, holes). A pure-GAN paste hallucinates a different
        # person (small 128x128 model), so it is never used alone when the
        # 3DMM is available.
        # NB: the GAN synthesizes a strictly FRONTAL face — it cannot honor a
        # target rotation. Non-frontal slider values go to the 3DMM path below.
        target_frontal = max(abs(self._yaw), abs(self._pitch), abs(self._roll)) < 2.0
        if (target_frontal and likely_human
                and self._initialized and self._tddfa is not None):
            try:
                hybrid = self._hybrid_frontalize(image, boxes)
                if hybrid is not None:
                    elapsed = (time.time() - t0) * 1000
                    used_gan = (self._gan_frontalizer is not None
                                and self._gan_frontalizer.is_initialized)
                    tag = ("3DMM身份保持+GAN遮挡区合成" if used_gan
                           else "3DMM纹理重渲染+软对称")
                    return FrontalizationResult(hybrid, elapsed,
                                                f"{tag} | {elapsed:.0f}ms")
            except Exception as e:
                print(f"Hybrid frontalization failed: {e}")
                import traceback; traceback.print_exc()

        # --- Path 1b: no 3DDFA — GAN-only landmark-aligned paste-back fallback ---
        if (target_frontal and likely_human and self._gan_frontalizer is not None
                and self._gan_frontalizer.is_initialized):
            try:
                aligned = self._gan_aligned_frontalize(image, boxes)
                if aligned is not None:
                    elapsed = (time.time() - t0) * 1000
                    return FrontalizationResult(aligned, elapsed,
                                                f"GAN正脸合成(关键点对齐) | {elapsed:.0f}ms")
            except Exception as e:
                print(f"GAN aligned failed: {e}")
                import traceback; traceback.print_exc()

        # --- Path 2: Human face + 3DDFA → 3DMM + Sim3DR (rotation params work here) ---
        if likely_human and self._initialized and self._tddfa is not None:
            return self._hassner_frontalize(image, boxes, elapsed)

        # --- Path 3: Animal / no face detected → simple mirror with center crop ---
        elapsed = (time.time() - t0) * 1000
        if boxes:
            bx1, by1, bx2, by2 = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            face_bbox = (int(bx1), int(by1), int(bx2 - bx1), int(by2 - by1))
        else:
            # No face detected — use center region as face region
            h, w = image.shape[:2]
            cs = int(min(h, w) * 0.2)  # 20% of smaller dimension
            cx, cy = w // 2, h // 2
            x, y = cx - cs // 2, cy - cs // 2
            bw, bh = cs, cs
            face_bbox = (x, y, bw, bh)

        # Near-frontal face (symmetric texture) → mirroring would only damage
        # it. Honest no-op instead. Only meaningful when a REAL face box was
        # detected — with no detection the "ROI" is a synthetic center patch
        # whose left/right texture energy says nothing about pose (this is
        # what silently swallowed the test cat: fur texture reads as
        # symmetric at any angle).
        x, y, bw, bh = face_bbox
        h, w = image.shape[:2]
        x1, x2 = max(0, x), min(w, x + bw)
        y1, y2 = max(0, y), min(h, y + bh)
        roi = image[y1:y2, x1:x2]
        if boxes and roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mid = gray.shape[1] // 2
            lv = cv2.Laplacian(gray[:, :mid], cv2.CV_64F).var()
            rv = cv2.Laplacian(gray[:, mid:], cv2.CV_64F).var()
            asym = abs(lv - rv) / (lv + rv + 1e-6)
            if asym < 0.10:
                return FrontalizationResult(
                    image, elapsed,
                    f"接近正脸(对称度{1-asym:.2f})，无需转换 | {elapsed:.0f}ms")

        return self._simple_mirror(image, elapsed, face_bbox)

    def _kp_indices(self):
        """68-landmark vertex indices (keypoints index the flattened (3N,) array)."""
        return self._tddfa.bfm.keypoints.reshape(-1, 3)[:, 0] // 3

    def _frontal_mesh(self, image, box):
        """3DDFA fit → (ver_detected (3,N), ver_frontal (3,N), tri).
        ver_frontal = mesh with detected pose undone, kept at original location."""
        with _no_grad():
            param_lst, roi_box_lst = self._tddfa(image, [box])
        R, offset, a_shp, a_exp = _parse_param(param_lst[0])
        _s = np.linalg.norm(R[:, 0])  # param[:12] is s*R — strip the scale
        if _s > 1e-8:
            R = R / _s
        ver = self._tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)[0]
        tri = self._tddfa.tri
        cx_v = (ver[0].min() + ver[0].max()) / 2
        cy_v = (ver[1].min() + ver[1].max()) / 2
        vc = ver.copy()
        vc[0] -= cx_v; vc[1] -= cy_v; vc[2] -= ver[2].mean()
        vf = R.T @ vc
        vf[0] += cx_v; vf[1] += cy_v
        return ver, vf, tri

    def _gan_warp_to_canvas(self, image: np.ndarray, box, lm_tgt: np.ndarray):
        """
        GAN frontal synthesis warped into the original canvas at the target
        landmarks (Cole et al. CVPR'17 style paste-back):
        1. GAN synthesizes the frontal texture on a face crop
        2. Detect landmarks on the GAN output, estimate a similarity warp
           into the original image canvas
        3. Paste mask = warped SOURCE hull ∩ target hull, eroded inward
        4. Reinhard LAB color transfer onto the original face tone
        Returns (warped BGR uint8, mask_f (h, w, 1) float32) or None.
        """
        h, w = image.shape[:2]
        x, y, bw, bh = [int(v) for v in box]
        kp = self._kp_indices()

        # 1. GAN frontal synthesis on the face crop
        pad = int(0.25 * bw)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
        gan_face = self._gan_frontalizer.frontalize(image, [x1, y1, x2, y2])

        # 2. Landmarks on the GAN output (upscale 3x so FaceBoxes can detect)
        up = cv2.resize(gan_face, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gboxes = self._face_boxes(up)
        if not len(gboxes):
            return None
        gbox = max(gboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))[:4]
        with _no_grad():
            gparam, groi = self._tddfa(up, [gbox])
        gver = self._tddfa.recon_vers(gparam, groi, dense_flag=True)[0]
        gver = self._tddfa.recon_vers(gparam, groi, dense_flag=True)[0]
        lm_src = (gver[:2, kp] / 3.0).T.astype(np.float32)  # back to GAN canvas

        # 3. Similarity warp GAN canvas → original image canvas
        M = cv2.estimateAffinePartial2D(lm_src, lm_tgt)[0]
        if M is None:
            return None
        # INTER_CUBIC: the GAN canvas is far smaller than the target face
        # region — linear interpolation on this upscale reads as blur.
        warped = cv2.warpAffine(gan_face, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)

        # 4. Paste mask = warped SOURCE hull ∩ target hull, eroded inward.
        # The GAN canvas is dark background beyond the face, and the landmark
        # fit on the synthetic face is imperfect — the target hull alone can
        # extend past the actual GAN face (e.g. below the chin), sampling
        # black junk and ringing the paste. The warped source hull bounds
        # where real GAN face pixels actually are.
        # The 68 landmarks stop at the eyebrows — their hull never covers the
        # forehead. Extend both hulls upward (brow line + ~45% of brow→chin)
        # so the GAN can also synthesize forehead skin where the source
        # photo's bangs/hair occlude it.
        hull_src = cv2.convexHull(_with_forehead(lm_src).astype(np.int32))
        alpha_src = np.zeros(gan_face.shape[:2], np.uint8)
        cv2.fillConvexPoly(alpha_src, hull_src, 255)
        er_src = max(3, int(0.05 * np.ptp(lm_src[:, 0])))
        alpha_src = cv2.erode(alpha_src, np.ones((er_src, er_src), np.uint8))
        alpha_w = cv2.warpAffine(
            alpha_src, M, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        ).astype(np.float32) / 255.0

        hull = cv2.convexHull(_with_forehead(lm_tgt).astype(np.int32))
        mask_tgt = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask_tgt, hull, 255)
        erode_px = max(15, int(0.10 * np.ptp(lm_tgt[:, 0])))
        mask_tgt = cv2.erode(
            mask_tgt, np.ones((erode_px, erode_px), np.uint8)
        ).astype(np.float32) / 255.0

        mask = np.minimum(alpha_w, mask_tgt)
        feather = max(4.0, 0.03 * float(np.ptp(lm_tgt[:, 0])))
        mask_f = cv2.GaussianBlur(mask, (0, 0), feather)[..., None]

        # 5. Reinhard color transfer (LAB mean/std match) from the original
        # face region to the warped GAN face — removes the tone mismatch.
        # Stats on the core region only (>0.9), so the dark GAN rim in the
        # feather band does not skew the transfer.
        m_bin = (mask_f[..., 0] > 0.9)
        if m_bin.sum() > 100:
            src_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
            tgt_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
            for c in range(3):
                s_vals, t_vals = src_lab[..., c][m_bin], tgt_lab[..., c][m_bin]
                s_mean, s_std = s_vals.mean(), s_vals.std() + 1e-6
                t_mean, t_std = t_vals.mean(), t_vals.std() + 1e-6
                src_lab[..., c] = (src_lab[..., c] - s_mean) * (t_std / s_std) + t_mean
            warped = cv2.cvtColor(
                np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

        # The generator's native output is small and lands soft after the
        # upscale paste — but unsharpening it also amplifies its rim junk
        # (dark canvas smears inside the mask feather), so the GAN layer is
        # left as-is; sharpness is recovered on the composite instead.

        return warped, mask_f

    def _gan_aligned_frontalize(self, image: np.ndarray, boxes: list):
        """GAN-only paste-back (fallback when the 3DMM render is unavailable)."""
        if self._tddfa is None:
            return None
        largest = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        _, ver_frontal, _ = self._frontal_mesh(image, largest)
        kp = self._kp_indices()
        lm_tgt = ver_frontal[:2, kp].T.astype(np.float32)
        out = self._gan_warp_to_canvas(image, largest, lm_tgt)
        if out is None:
            return None
        warped, mask_f = out
        result = (warped.astype(np.float32) * mask_f +
                  image.astype(np.float32) * (1 - mask_f)).astype(np.uint8)
        return result

    def _hybrid_frontalize(self, image: np.ndarray, boxes: list):
        """
        Identity-preserving frontalization:
        - the 3DMM re-render supplies the subject's REAL texture/identity
          everywhere it has confident real samples (tiers 1-3)
        - the GAN synthesizes new pixels only where the 3DMM had nothing real
          (occluded side fallback, clutter, rasterization holes), plus a small
          base fraction so the synthesized detail blends in smoothly
        Both are anchored to the SAME frontal landmark geometry, so the
        per-pixel blend stays registered.
        """
        d = self._render_frontal_3dmm(image, boxes)
        if d is None:
            return None
        base = d["result"].astype(np.float32)
        if (self._gan_frontalizer is None
                or not self._gan_frontalizer.is_initialized):
            return d["result"]
        largest = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        # The GAN synthesis + warp is slider-independent (the hybrid path is
        # only taken for a frontal target) — cache it per image alongside the
        # fit, or every slider tick would re-run the multi-second GAN.
        key = self._fit_key
        if self._gan_key == key and self._gan_cache is not None:
            warped, mask_f = self._gan_cache
        else:
            out = self._gan_warp_to_canvas(image, largest, d["lm_tgt"])
            if out is None:
                return d["result"]
            warped, mask_f = out
            self._gan_key = key
            self._gan_cache = (warped, mask_f)
        # conf_map: 1 where the 3DMM rendered real sampled texture.
        # The GAN layer replaces identity (small model → a different face),
        # so the 3DMM render keeps the majority wherever it has real
        # texture; the GAN dominates only where the render has nothing
        # real (occluded side, holes).
        conf = cv2.GaussianBlur(d["conf_map"], (0, 0), 6.0)
        # Luminance-gate the GAN layer: its canvas holds dark smear junk
        # beyond the actual face, and parts of that land inside the paste
        # mask feather — without the gate it darkens the cheeks as ghost
        # patches. The gate is feathered so no hard edge appears.
        gan_lum = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gate = np.clip((gan_lum - 0.20) / 0.15, 0.0, 1.0)
        mask_use = (mask_f[..., 0] * gate).astype(np.float32)
        g = (mask_use * (0.20 + 0.80 * (0.8 - conf).clip(0, 1)))[..., None]
        hybrid = base * (1.0 - g) + warped.astype(np.float32) * g

        # --- Detail recovery ---
        # The 3DMM render is inherently soft: vertex-sampled texture,
        # diffusion-filled cleanup regions, and the 1024px work canvas
        # upsampled back to full resolution. Transfer the GAN's
        # high-frequency band on top, but ONLY where the render has real
        # texture (conf) — the GAN's own low-conf areas are hallucination,
        # and sharpening those amplifies the junk instead.
        sig_d = max(3.0, 0.012 * float(np.ptp(d["lm_tgt"][:, 0])))
        gan_f = warped.astype(np.float32)
        hf = gan_f - cv2.GaussianBlur(gan_f, (0, 0), sig_d)
        w_hf = np.clip(mask_f[..., 0], 0, 1) * conf * 0.7
        # The repaired mouth's conf was BOOSTED (mouth_trust) to keep the
        # GAN's fabricated teeth out of the blend above — do not let that
        # same boost re-admit the GAN's teeth high-frequencies here.
        mt = d.get("mouth_trust")
        if mt is not None:
            w_hf = w_hf * (1.0 - np.clip(mt, 0.0, 1.0))
        hybrid = hybrid + hf * w_hf[..., None]

        # Mid-frequency recovery on the composite: the base is soft
        # (vertex-sampled texture + canvas upsample). Sharpen ONLY where
        # the render has real sampled texture (conf high) — unsharpening
        # the GAN/diffusion content just amplifies its ghosting and rim
        # junk. Weights are feathered so no mask edge shows.
        sig_m = max(2.0, 0.006 * float(np.ptp(d["lm_tgt"][:, 0])))
        w_sharp = (conf * mask_f[..., 0]).astype(np.float32)
        w_sharp = cv2.GaussianBlur(w_sharp, (0, 0), 6.0)[..., None]
        try:
            sharp_on = float(os.environ.get("S2F_SHARP", "1")) > 0
        except ValueError:  # e.g. S2F_SHARP=off — keep the default, not a crash
            sharp_on = True
        if sharp_on:
            sharp = cv2.addWeighted(hybrid, 1.4,
                                    cv2.GaussianBlur(hybrid, (0, 0), sig_m),
                                    -0.4, 0)
            hybrid = hybrid * (1.0 - w_sharp) + sharp * w_sharp

        _dbg_dump("dbg_gan_warped.png", warped)
        _dbg_dump("dbg_hybrid.png", np.clip(hybrid, 0, 255).astype(np.uint8))
        return np.clip(hybrid, 0, 255).astype(np.uint8)

    def _hassner_frontalize(self, image: np.ndarray, boxes: list,
                             elapsed_ms: float) -> FrontalizationResult:
        """Slider path: 3DMM textured re-render at the requested pose."""
        if not self._initialized or self._tddfa is None:
            return self._simple_mirror(image, elapsed_ms)
        d = self._render_frontal_3dmm(image, boxes)
        if d is None:
            return self._simple_mirror(image, elapsed_ms, None)
        return FrontalizationResult(d["result"], elapsed_ms, d["info"])

    def _fit_face(self, image: np.ndarray, boxes: list):
        """Slider-independent stage (cached per image): 3DDFA regression,
        dense mesh, and multi-tier texture sampling from the original photo
        (Hassner et al. CVPR 2015, stages 1-2). With this cached, dragging
        a rotation slider only re-runs the render stage (sub-second) instead
        of the full multi-second pipeline. Returns a fit dict or None."""
        key = self._image_key(image)
        if self._fit_key == key:
            return self._fit_cache
        try:
            largest = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            with _no_grad():
                param_lst, roi_box_lst = self._tddfa(image, [largest])
            param = param_lst[0]

            R, offset, alpha_shp, alpha_exp = _parse_param(param)
            # NB: param[:12] is s*R (scaled rotation) — extract the pure rotation
            _s = np.linalg.norm(R[:, 0])
            if _s > 1e-8:
                R = R / _s
            self._detected_yaw = float(np.degrees(np.arctan2(R[0, 1], R[0, 0])))
            self._detected_pitch = float(np.degrees(np.arctan2(-R[1, 2], R[2, 2])))
            self._detected_roll = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
            self._pose_initialized = True

            # Dense mesh (all ~38k vertices, not just 68 landmarks)
            with _no_grad():
                ver_lst = self._tddfa.recon_vers(param_lst, roi_box_lst,
                                                 dense_flag=True)
            ver = ver_lst[0]  # (3, N), already projected into image coords
            tri = self._tddfa.tri

            h, w = image.shape[:2]

            # --- Step 2: sample texture from original image at profile positions ---
            # NB: Sim3DR_Cython kernel writes alpha*255*p_color into a uint8 bg,
            # so colors/texture MUST be in [0,1] float32 and bg MUST be uint8.
            xs = np.clip(ver[0].astype(np.int32), 0, w - 1)
            ys = np.clip(ver[1].astype(np.int32), 0, h - 1)
            tex_org = image[ys, xs].astype(np.float32) / 255.0  # (N, 3) BGR in [0,1]

            # Occluded-side texture (Hassner-style symmetry in texture space):
            # vertices facing AWAY from the camera in the detected pose sampled
            # background/hair. Resample them from the position reflected across
            # the face's symmetry axis (fitted through midline landmarks:
            # nose bridge -> nose tip -> chin), where real skin texture exists.
            from Sim3DR import get_normal
            ver0_c = np.ascontiguousarray(ver.T).astype(np.float32)
            normals = get_normal(ver0_c, tri)  # (N, 3), +z ~ toward camera
            w_vis = np.clip(normals[:, 2] * 2.0, 0, 1)  # 1=visible, 0=occluded

            # 3D-intrinsic symmetric vertex map: BFM is bilaterally symmetric,
            # so each occluded vertex has a visible 3D counterpart whose 2D
            # position (on the visible side) holds the correct texture.
            if self._sym_map is None:
                from scipy.spatial import cKDTree
                U = self._tddfa.bfm.u.reshape(-1, 3)
                U_flip = U.copy()
                U_flip[:, 0] = -U_flip[:, 0]
                _, sym = cKDTree(U).query(U_flip)
                self._sym_map = sym.astype(np.int64)
            sym_pos = ver[:2, self._sym_map]  # counterpart 2D positions
            xs_m = np.clip(sym_pos[0].astype(np.int32), 0, w - 1)
            ys_m = np.clip(sym_pos[1].astype(np.int32), 0, h - 1)
            tex_mir = image[ys_m, xs_m].astype(np.float32) / 255.0

            # Three-tier texture (fixes the "stitch" artifacts on the rim):
            #  tier 1: vertex faces the camera        → its own pixel
            #  tier 2: 3D-symmetric counterpart visible → counterpart's pixel
            #  tier 3: rim vertices (fold line, nz≈0) whose own pixel is
            #          hair/background clutter → reflect their 2D position
            #          across the landmark midline, which lands INSIDE the
            #          visible face (guaranteed skin/hair, not clutter)
            kp = self._kp_indices()
            lm = ver[:2, kp]
            mid = lm[:, [27, 28, 29, 30, 8]].T
            a0 = mid.mean(axis=0)
            _, _, vt = np.linalg.svd(mid - a0, full_matrices=False)
            d = vt[0] / (np.linalg.norm(vt[0]) + 1e-8)
            P = ver[:2, :]
            tt = ((P - a0[:, None]) * d[:, None]).sum(axis=0)
            P_r = 2.0 * (a0[:, None] + tt[None, :] * d[:, None]) - P
            xs_r = np.clip(P_r[0].astype(np.int32), 0, w - 1)
            ys_r = np.clip(P_r[1].astype(np.int32), 0, h - 1)
            tex_refl = image[ys_r, xs_r].astype(np.float32) / 255.0

            # Clutter gating: samples that are green background or bright
            # near-white objects (flowers) are NEVER valid face texture —
            # redirect their weight to the symmetry tiers. Dark samples
            # (hair/brows) are kept: bangs on the forehead are legitimate.
            def _clutter_mask(tex):
                bb, gg, rr = tex[:, 0], tex[:, 1], tex[:, 2]
                Y = 0.299 * rr + 0.587 * gg + 0.114 * bb
                mx = np.maximum(np.maximum(rr, gg), bb)
                mn = np.minimum(np.minimum(rr, gg), bb)
                is_green = (gg > rr + 0.05) & (gg > bb + 0.05)
                is_white = (Y > 0.85) & ((mx - mn) < 0.06)
                return is_green | is_white

            bad_org = _clutter_mask(tex_org)
            w_mir_ok = w_vis[self._sym_map].copy()
            w_mir_ok[_clutter_mask(tex_mir)] = 0.0  # counterpart also clutter
            bad_refl = _clutter_mask(tex_refl)

            w_t1 = w_vis * (~bad_org)
            w_bad = w_vis * bad_org                   # redirect clutter weight
            w_t2 = (1.0 - w_vis) * w_mir_ok + w_bad * w_mir_ok
            w_t3 = ((1.0 - w_vis) * (1.0 - w_mir_ok) +
                    w_bad * (1.0 - w_mir_ok)) * (~bad_refl)
            w_t4 = ((1.0 - w_vis) * (1.0 - w_mir_ok) +
                    w_bad * (1.0 - w_mir_ok)) * bad_refl  # all tiers clutter

            # Tier 4: median skin color of the good visible samples
            good_skin = (w_vis > 0.5) & (~bad_org)
            med_skin = (np.median(tex_org[good_skin], axis=0)
                        if good_skin.sum() > 50
                        else np.array([0.75, 0.62, 0.57], np.float32))
            texture = (tex_org * w_t1[:, None] +
                       tex_mir * w_t2[:, None] +
                       tex_refl * w_t3[:, None] +
                       med_skin[None, :] * w_t4[:, None])

            # Mark forehead vertices whose samples are hair/clutter debris
            # (the source photo's bangs physically cover the forehead) — the
            # hybrid path drops their confidence so the GAN adds detail there.
            # The actual cleanup happens in image space after rendering.
            tex_lum = (0.299 * texture[:, 2] + 0.587 * texture[:, 1] +
                       0.114 * texture[:, 0])
            brow_y_det = float(ver[1, kp[17:27]].mean())
            span_y_det = float(np.ptp(ver[1, kp])) + 1e-6
            is_forehead = ver[1] < (brow_y_det - 0.05 * span_y_det)
            bad_fore = is_forehead & ((tex_lum < 0.32) | _clutter_mask(texture))

            conf_v = np.clip(w_t1 + w_t2 + w_t3, 0.0, 1.0).astype(np.float32)
            # Forehead vertices cleaned above hold synthetic skin tone, not
            # real texture — low confidence so the GAN adds detail there.
            conf_v[bad_fore] *= 0.15

            cx_v = (ver[0].min() + ver[0].max()) / 2
            cy_v = (ver[1].min() + ver[1].max()) / 2
            ver_centered = ver.copy()
            ver_centered[0] -= cx_v
            ver_centered[1] -= cy_v
            ver_centered[2] -= ver[2].mean()

            fit = {"largest": largest, "R": R, "ver": ver, "tri": tri,
                   "h": image.shape[0], "w": image.shape[1],
                   "texture": texture, "conf_v": conf_v,
                   # 0.55 (normals_z > ~0.28): grazing far-half vertices are
                   # only borderline visible, and their synthetic texture
                   # lands at extrapolated positions — the doubled nose/
                   # mouth. Treat them as occluded so the mirror fill (whose
                   # source is position-exact) takes over instead.
                   "vis_v": (w_vis > 0.55).astype(np.float32),
                   "cx_v": cx_v, "cy_v": cy_v, "ver_centered": ver_centered,
                   "lm_det": ver[:2, self._kp_indices()].copy()}
            self._fit_key = key
            self._fit_cache = fit
            return fit
        except Exception as e:
            print(f"3DDFA fit failed: {e}")
            import traceback; traceback.print_exc()
            return None

    def _render_frontal_3dmm(self, image: np.ndarray, boxes: list):
        """
        Render stage: rotate the fitted mesh to the target pose, render with
        the sampled texture, Hassner soft-symmetry fill, hole/debris cleanup,
        and composite back onto the original photo (Hassner stages 3-6).

        Returns a dict with the composited result and intermediate maps
        (conf_map = where the render used REAL sampled texture, lm_tgt =
        frontal landmarks) for the hybrid frontal path, or None on failure.
        """
        try:
            fit = self._fit_face(image, boxes)
            if fit is None:
                return None
            R = fit["R"]
            ver = fit["ver"]
            tri = fit["tri"]
            texture = fit["texture"]
            conf_v = fit["conf_v"]
            vis_v = fit["vis_v"]
            cx_v = fit["cx_v"]
            cy_v = fit["cy_v"]
            ver_centered = fit["ver_centered"]

            # Render + post-process on a capped working canvas: Sim3DR
            # rasterization is cheap, but the image-space post (hole fill,
            # debris cleanup, equalization) is not — and at ~38k mesh
            # vertices the render is vertex-limited, not pixel-limited, so
            # downscaling costs almost nothing visually. The final composite
            # goes back onto the full-resolution original.
            h0, w0 = image.shape[:2]
            scale = min(1.0, 1024.0 / max(h0, w0))
            if scale < 1.0:
                work = cv2.resize(image, (int(round(w0 * scale)),
                                          int(round(h0 * scale))))
            else:
                work = image
            h, w = work.shape[:2]

            # --- Step 3: rotate mesh to target pose ---
            def rot_x(rad):
                c, s = np.cos(rad), np.sin(rad)
                return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)

            def rot_y(rad):
                c, s = np.cos(rad), np.sin(rad)
                return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

            def rot_z(rad):
                c, s = np.cos(rad), np.sin(rad)
                return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            # Re-center mesh around origin before rotation, then re-center at image center
            R_target = (rot_y(np.radians(self._yaw)) @
                        rot_x(np.radians(self._pitch)) @
                        rot_z(np.radians(self._roll)))
            # Undo detected pose, apply target pose
            R_total = R_target @ R.T

            ver_frontal = R_total @ ver_centered
            # Keep the frontalized face at its ORIGINAL location, not image center
            ver_frontal[0] += cx_v
            ver_frontal[1] += cy_v

            # --- Deroll in-plane ---
            # On a 3/4 view the far half's fitted 3DMM geometry is
            # extrapolated, so even after R.T the render keeps residual
            # roll (the far eye/chin landmarks sit too low or too high).
            # Upright the mesh with the RELIABLE midline landmarks — nose
            # bridge, nose base, lip centers — which are visible and on
            # the symmetry axis at any yaw. A vertical face axis is also
            # what the mirror fill below assumes.
            kp = self._tddfa.bfm.keypoints.reshape(-1, 3)[:, 0] // 3
            mid_idx = kp[[27, 28, 29, 30, 33, 51, 57]]
            mx_m = float(ver_frontal[0, mid_idx].mean())
            my_m = float(ver_frontal[1, mid_idx].mean())
            pts = np.stack([ver_frontal[0, mid_idx] - mx_m,
                            ver_frontal[1, mid_idx] - my_m])
            _cov = np.cov(pts)
            _evals, _evecs = np.linalg.eigh(_cov)
            axis = _evecs[:, -1]
            if axis[1] < 0:
                axis = -axis
            roll_res = float(np.arctan2(axis[0], axis[1]))
            c_r, s_r = np.cos(roll_res), np.sin(roll_res)
            vx = ver_frontal[0] - mx_m
            vy = ver_frontal[1] - my_m
            ver_frontal[0] = c_r * vx - s_r * vy + mx_m
            ver_frontal[1] = s_r * vx + c_r * vy + my_m

            # --- Step 4: render with sampled texture ---
            from Sim3DR import RenderPipeline
            # Ambient=1, no directional light → preserves original texture brightness
            render_app = RenderPipeline(
                intensity_ambient=1.0, color_ambient=(1, 1, 1),
                intensity_directional=0.0, intensity_specular=0.0,
                light_pos=(0, 0, 5), view_pos=(0, 0, 5)
            )
            # Scale the mesh into the working canvas (the fit's mesh lives in
            # full-resolution image coordinates).
            ver_work = ver_frontal.copy()
            ver_work[0] *= scale
            ver_work[1] *= scale
            ver_c = np.ascontiguousarray(ver_work.T).astype(np.float32)
            bg = np.zeros_like(work)  # uint8 required by Sim3DR_Cython
            rendered = render_app(ver_c, tri, bg, texture=texture).astype(np.float32) / 255.0

            # --- Step 5: accumulation map (coverage per pixel) ---
            # texture=ones → rendered value 255 where covered, 0 elsewhere
            ones_tex = np.ones_like(texture)
            acc = render_app(ver_c, tri, np.zeros_like(work), texture=ones_tex)
            acc_gray = acc[..., 0].astype(np.float32) / 255.0

            # Texture-confidence map: 1 where the render used REAL sampled
            # texture (tiers 1-3), 0 where it fell back to the skin-color
            # fill (tier 4). The hybrid path blends the GAN in where this
            # confidence is low. (conf_v is computed in the cached fit.)
            conf_tex = np.repeat(conf_v[:, None], 3, axis=1)
            conf_img = render_app(ver_c, tri, np.zeros_like(work),
                                  texture=conf_tex)
            conf_map = conf_img[..., 0].astype(np.float32) / 255.0

            # Visibility map: 1 where the vertex was visible in the ORIGINAL
            # photo (real texture), 0 where it was occluded and the render
            # only holds a tier-2/3/4 synthetic guess. The far half's
            # extrapolated geometry puts those guesses at slightly wrong
            # positions — the classic doubled nose/mouth. Such pixels are
            # handed to the image-space mirror fill below instead, which is
            # position-exact by construction.
            vis_tex = np.repeat(vis_v[:, None], 3, axis=1)
            vis_img = render_app(ver_c, tri, np.zeros_like(work),
                                 texture=vis_tex)
            vis_map = cv2.GaussianBlur(
                vis_img[..., 0].astype(np.float32) / 255.0, (0, 0), 5.0)

            # Soft visibility ramp: clip(v*3) compressed the transition to
            # ~3px, which showed as a hard vertical seam between the
            # real-texture half and the mirror/tier-filled half at rotated
            # (slider) poses. Ramp over the middle band of a wider blur
            # instead — the blend then spans ~12px and disappears.
            def _soft_vis(v):
                return np.clip((v - 0.35) / 0.30, 0.0, 1.0)

            # --- Step 6: soft symmetry fill (Hassner) ---
            # Mirror the render around the FRONTAL MESH's midline (not the
            # image center) so the fill aligns with the face, not the canvas.
            kp = self._tddfa.bfm.keypoints.reshape(-1, 3)[:, 0] // 3
            # Mirror about the derolled midline landmarks (the chin
            # landmark is excluded: on a 3/4 view its fitted position is
            # pulled toward the visible side).
            x_ax = float(ver_work[0, mid_idx].mean())
            xs_idx = np.arange(w)
            xs_src = np.clip((2 * int(round(x_ax))) - xs_idx, 0, w - 1)
            # The mirrored IMAGE is taken later, from the cleaned base —
            # taking it from the raw render would reflect debris/holes into
            # the fill. (mirrored_acc is a weight map, pixel-free of debris.)
            mirrored_acc = acc_gray[:, xs_src]

            # Weight: covered + originally-visible pixels → real texture;
            # occluded pixels → mirrored render; anything else → the tier
            # render as a last resort.
            w_org = (np.clip(acc_gray * 3.0, 0, 1) *
                     _soft_vis(vis_map))
            # Mirror fill only makes sense near the frontal pose — when the
            # user rotates away (yaw/roll), it would paste a duplicated face.
            frontalness = (max(0.0, 1.0 - abs(self._yaw) / 45.0) *
                           max(0.0, 1.0 - abs(self._roll) / 45.0))
            # Mirror fill may only plug holes INSIDE the rendered silhouette.
            # Outside it, the mirrored content is a displaced ghost face
            # (and its texture may come from background clutter). The mirror
            # SOURCE must be originally-visible content too, or junk gets
            # reflected into the fill.
            sil = cv2.morphologyEx((acc_gray * 255).astype(np.uint8),
                                   cv2.MORPH_CLOSE,
                                   np.ones((15, 15), np.uint8))
            sil = cv2.dilate(sil, np.ones((9, 9), np.uint8)).astype(np.float32) / 255.0
            w_mir = (np.clip(mirrored_acc * 3.0, 0, 1) *
                     _soft_vis(vis_map[:, xs_src]) *
                     (1.0 - w_org) * frontalness * sil)
            # Last-resort weight: keep the tier render where neither real
            # texture nor a mirror source exists.
            w_res = np.clip(1.0 - (w_org + w_mir) * 2.0, 0, 1)

            denom = w_org + w_mir + w_res + 1e-6
            w_org_3 = np.stack([w_org / denom] * 3, axis=-1)
            w_mir_3 = np.stack([w_mir / denom] * 3, axis=-1)
            w_res_3 = np.stack([w_res / denom] * 3, axis=-1)

            # Coverage after morphological closing — needed by the hole fill
            # below AND by the debris cleanup (the mesh's top rim sampled
            # dark hair and fell outside plain acc_gray, which is what left
            # the visible black edge on the head).
            acc_closed = cv2.morphologyEx(
                (acc_gray * 255).astype(np.uint8), cv2.MORPH_CLOSE,
                np.ones((9, 9), np.uint8)).astype(np.float32) / 255.0

            # Forehead debris cleanup (image space): the source photo's bangs
            # physically cover the forehead (a center hair strand even hangs
            # between the brows), so the rendered forehead texture is
            # hair/flower debris. Diffusion-fill non-skin pixels above the
            # brow polyline — plus a narrow glabella column just below it —
            # from the surrounding rendered skin. Eyebrows are lateral to the
            # column and the eyes are below the band, so both survive.
            lm_f = ver_work[:2, kp]
            xs_b = lm_f[0, 17:27]
            ys_b = lm_f[1, 17:27]
            order = np.argsort(xs_b)
            xs_b, ys_b = xs_b[order], ys_b[order]
            xs_img = np.arange(w, dtype=np.float32)
            brow_line = np.interp(xs_img, xs_b, ys_b,
                                  left=ys_b[0], right=ys_b[-1])
            yy = np.arange(h, dtype=np.float32)[:, None]
            xx = xs_img[None, :]
            covered = acc_closed > 0.3
            # Above the brows there is no real feature to preserve in the
            # render, so thresholds are loose: anything not plainly skin is
            # refilled. Below the brow line (glabella column) thresholds stay
            # strict so the inner eyebrow ends survive.
            region_up = (yy < (brow_line[None, :] - 2.0)) & covered
            gx = 0.5 * (lm_f[0, 21] + lm_f[0, 22])
            gw = abs(lm_f[0, 22] - lm_f[0, 21]) + 1e-6
            eye_y = float(lm_f[1, 36:48].mean())
            # The band reaches just past the eye line: the glabella/nasion
            # is hairless at any depth, only the eyes must stay out of it.
            # Extend it down to the nose tip: dark hair-strand debris also
            # streaks the nose bridge, which is hairless too. The nostrils
            # sit lateral to this narrow column, so they survive.
            nose_y = float(lm_f[1, 30])
            band_h = max(1.25 * max(eye_y - float(np.interp(gx, xs_b, ys_b)), 1.0),
                         nose_y - float(np.interp(gx, xs_b, ys_b)))
            # The glabella between the inner brow points (landmarks 21/22) is
            # hairless, so a narrow column there may use loose thresholds
            # without touching eyebrow hair (which starts at +-0.5*gw).
            col = (np.abs(xs_img[None, :] - gx) < 0.45 * gw)
            glab = (col & (yy >= brow_line[None, :] - 2.0) &
                    (yy < brow_line[None, :] + band_h) & covered)
            region = region_up | glab
            # Supra-brow band lateral to the column: dark debris (hair
            # strands) sits here too, but so do the eyebrows — cleaned by
            # connected-component size below, brows are large and survive.
            band = ((yy >= brow_line[None, :] - 2.0) &
                    (yy < brow_line[None, :] +
                     0.75 * (eye_y - brow_line[None, :])) &
                    covered & ~col)
            # Per-eye boxes: dark specks cluster on and around the eye rims
            # (the lower-lid vertices sampled the lash line in the original
            # pose). Everything dark OUTSIDE the eye's own landmark hull is
            # debris — the specks fuse with the eye into one big component,
            # so area/contact filters cannot separate them; geometry can.
            eye_masks = []
            for e0, e1 in ((36, 41), (42, 47)):
                ex = lm_f[0, e0:e1 + 1]
                ey = lm_f[1, e0:e1 + 1]
                eye_h = max(float(ey.max() - ey.min()), 1.0)
                pad = 0.8 * eye_h
                box = ((yy >= np.maximum(ey.min() - pad,
                                         brow_line[None, :] - 2.0)) &
                       (yy <= ey.max() + pad) &
                       (xx >= ex.min() - pad) & (xx <= ex.max() + pad) &
                       covered)
                eh = cv2.convexHull(lm_f.T[e0:e1 + 1].astype(np.float32))
                em = cv2.fillPoly(np.zeros((h, w), np.uint8),
                                  [eh.astype(np.int32)], 1)
                # 5px margin keeps the lash line / rim.
                em = cv2.dilate(em, np.ones((11, 11), np.uint8)).astype(bool)
                eye_masks.append((box, em))
            work_region = region | band
            for box, _ in eye_masks:
                work_region |= box

            # All cleanup work happens near brows/eyes — crop once to that
            # ROI (+ margin so the diffusion blurs see enough context);
            # this keeps the fill ~10x cheaper than full-canvas blurs.
            ys_r, xs_r = np.where(work_region)
            if len(ys_r):
                mg = 96
                sl = np.s_[max(int(ys_r.min()) - mg, 0):
                           min(int(ys_r.max()) + mg, h),
                           max(int(xs_r.min()) - mg, 0):
                           min(int(xs_r.max()) + mg, w)]
            else:
                sl = np.s_[0:h, 0:w]
            region_up_s = region_up[sl]
            glab_s = glab[sl]
            band_s = band[sl]
            eye_masks_s = [(box[sl], em[sl]) for box, em in eye_masks]

            def _clean_debris(img_u8, record=None):
                """Refill hair/clutter debris above the brows, in the
                glabella→nose-tip column, and dark specks near the eyes.
                Runs on the PRE-MIRROR render (a dirty mirror source would
                copy the junk onto the occluded half) and again after the
                blend. When `record` (a list) is given, the full-canvas
                mask of erased pixels is appended — the photo-detail boost
                must not resurrect the removed specks/strands."""
                if not region.any():
                    return img_u8
                f = img_u8[sl].astype(np.float32)
                lum = f.mean(axis=2) / 255.0
                good = region_up_s & (lum > 0.40)
                if good.sum() <= 200:
                    return img_u8
                skin_ref = np.median(f[good], axis=0)
                dist = np.linalg.norm((f - skin_ref[None, None, :]) / 255.0,
                                      axis=2)
                unknown = ((region_up_s & ((lum < 0.50) | (dist > 0.10))) |
                           (glab_s & ((lum < 0.55) | (dist > 0.12))))
                dark_band = (band_s & (lum < 0.55) &
                             (dist > 0.08)).astype(np.uint8)
                n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(
                    dark_band, connectivity=8)
                for lbl in range(1, n_lbl):
                    if stats[lbl, cv2.CC_STAT_AREA] < 500:
                        unknown[labels == lbl] = True
                for box_s, em_s in eye_masks_s:
                    dk = box_s & (lum < 0.65) & (dist > 0.12)
                    unknown[dk & ~em_s] = True
                if record is not None and unknown.any():
                    full = np.zeros((h, w), bool)
                    full[sl] = unknown
                    record.append(full)
                # Fill ONLY from skin-colored pixels: an unrestricted
                # source lets dark hair beyond the hairline bleed into
                # the refilled forehead.
                skin_like = ((lum > 0.45) & (dist < 0.18)).astype(np.float32)
                _diffuse_fill(f, unknown, skin_like, (4, 8, 16, 32))
                # Even out residual mottle: blend the above-brow area
                # toward a blur of the SKIN-LIKE pixels only — an
                # unrestricted blur would smear the dark hair beyond the
                # hairline back into the refilled forehead.
                sig_s = max(6.0, float(np.ptp(lm_f[0])) * 0.04)
                smooth = _fast_blur(f * skin_like[..., None], sig_s) / \
                    np.maximum(_fast_blur(skin_like, sig_s)[..., None], 1e-3)
                m_s = region_up_s.astype(np.float32)
                m_s = _fast_blur(m_s, sig_s * 0.5)
                m3 = (m_s * 0.5)[..., None]
                f = f * (1.0 - m3) + smooth * m3
                out = img_u8.copy()
                out[sl] = np.clip(f, 0, 255).astype(np.uint8)
                return out

            # Fill rasterization holes and clean debris BEFORE mirroring:
            # both the mirror source and the w_org/w_res terms must come
            # from a hole-free, debris-free base.
            clean_edits = []
            base = np.clip(rendered * 255.0, 0, 255).astype(np.uint8)
            unknown = ((acc_closed > 0.5) & (acc_gray < 0.5))
            if unknown.any():
                base_f = base.astype(np.float32)
                _diffuse_fill(base_f, unknown, np.ones((h, w), np.float32),
                              (4, 8, 16, 32))
                base = np.clip(base_f, 0, 255).astype(np.uint8)
            base = _clean_debris(base, clean_edits)

            # Soft symmetry fill (Hassner), now from the cleaned base.
            mirrored = base[:, xs_src]
            frontal = (base * w_res_3 + mirrored * w_mir_3 + base * w_org_3)
            # base is already 0-255 — no rescale (that saturated the face white).
            frontal = np.clip(frontal, 0, 255).astype(np.uint8)
            # Post-blend pass: the blend may still expose specks where the
            # mirror contributed, and the base pass cannot know those.
            frontal = _clean_debris(frontal, clean_edits)
            _dbg_dump("dbg_s1_base.png", base)
            _dbg_dump("dbg_s2_blend.png", frontal)

            # --- Mouth repair ---
            # The canonical mesh keeps a gap between the inner-lip vertex
            # rings, and on a 3/4 fit the rendered "opening" often sits
            # ABOVE the landmark hull (geometry drift) — a fabricated dark
            # mouth with teeth. Skin and lipstick are bright (lum > 0.6);
            # scan the whole nostril-to-lips neighborhood for very dark
            # pixels and refill them from their bright surroundings.
            mouth_trust = None
            mo = cv2.convexHull(lm_f.T[48:60].astype(np.float32))
            mo_mask = cv2.fillPoly(np.zeros((h, w), np.uint8),
                                   [mo.astype(np.int32)], 1)
            f32m = frontal.astype(np.float32)
            if mo_mask.sum() > 30:
                xb, yb, wb, hb = cv2.boundingRect(mo_mask)
                y_top = int(lm_f[1, 31:36].max()) + 2
                y_bot = int(lm_f[1, 57] + 0.45 * hb)
                x0 = max(0, xb - wb // 6)
                x1 = min(w, xb + wb + wb // 6 + 1)
                neigh = np.zeros((h, w), dtype=bool)
                neigh[y_top:min(y_bot, h), x0:x1] = True
                lum_m = f32m.mean(axis=2)
                # Lip pixels are red-dominant (R well above G and B) even
                # though their luminance is low — they must count as valid
                # content, not as "dark opening", or the fill eats the lips
                # and leaves a gray blob.
                red_m = ((f32m[..., 2] > f32m[..., 1] + 25.0) &
                         (f32m[..., 2] > f32m[..., 0] + 10.0))
                opening = neigh & (lum_m < 0.52 * 255.0) & ~red_m
                if opening.sum() > 10:
                    # Fill sources: bright skin AND lip pixels — the gap
                    # between the inner-lip rings must refill as lip, not
                    # as skin, or the mouth reads as a pale smear.
                    like = ((lum_m >= 0.55 * 255.0) | red_m).astype(np.float32)
                    _diffuse_fill(f32m, opening, like, (3, 6, 12, 24))
                    frontal = np.clip(f32m, 0, 255).astype(np.uint8)
                    # Trust the repair: keep the GAN's own mouth synthesis
                    # (which fabricated teeth here) out of the lips.
                    mouth_trust = cv2.dilate(
                        neigh.astype(np.uint8),
                        np.ones((9, 9), np.uint8)).astype(np.float32)
                if os.environ.get("S2F_DEBUG"):
                    ov = frontal.copy()
                    ov[neigh] = (0, 255, 0)
                    ov[opening] = (0, 0, 255)
                    _dbg_dump("dbg_s3_mouth.png", ov)
            _dbg_dump("dbg_s3_mouth_done.png", frontal)

            # --- Midline seam equalization ---
            # One half of the render keeps real texture while the other is
            # mirror-filled / color-filled, so a brightness step shows across
            # the midline ("the middle did not blend well"). Replace the
            # render's low-frequency component by the average of itself and
            # its mirror about the frontal midline — symmetric illumination
            # without touching texture detail. Applied only inside the face.
            f32 = frontal.astype(np.float32)
            sig_l = max(10.0, float(np.ptp(lm_f[0])) * 0.10)
            wgt = np.clip(acc_closed, 0, 1).astype(np.float32)
            B = _fast_blur(f32 * wgt[..., None], sig_l) / \
                np.maximum(_fast_blur(wgt, sig_l)[..., None], 1e-3)
            target = 0.5 * (B + B[:, xs_src])
            gain = np.clip((target + 8.0) / (B + 8.0), 0.80, 1.25)
            fmask_soft = cv2.GaussianBlur(wgt, (0, 0), 3.0)[..., None]
            f32 = f32 * (1.0 - fmask_soft) + f32 * gain * fmask_soft

            # Hide the residual seam: narrow bilateral-smoothed band on the
            # midline, weighted by face coverage.
            hb = max(6, int(0.02 * float(np.ptp(lm_f[0]))))
            band = np.zeros((h, w), np.float32)
            band[:, max(0, int(x_ax) - hb):min(w, int(x_ax) + hb)] = 1.0
            band = cv2.GaussianBlur(band, (0, 0), max(2.0, hb / 2.5))[..., None]
            sm = cv2.bilateralFilter(np.clip(f32, 0, 255).astype(np.uint8),
                                     7, 40, 40).astype(np.float32)
            f32 = f32 * (1.0 - band * fmask_soft) + sm * (band * fmask_soft)
            frontal = np.clip(f32, 0, 255).astype(np.uint8)

            # Original-pose landmarks on the work canvas — the protrusion
            # side below is pure geometry (nose tip vs render midline);
            # pixel voting is unreliable (hair darkness, pink flowers and
            # red objects flip it — tried twice, wrong both times).
            lm_det = (fit["lm_det"] * scale).T.astype(np.float32)
            span_det = float(np.ptp(lm_det[:, 0]))
            nose_dx = float(lm_det[30, 0]) - x_ax

            # --- Stray-lip erasure (render space) ---
            # Mesh-boundary texels on the protrusion side sample the ORIGINAL
            # protruding lips, baking a lip-colored strip into the rendered
            # cheek (shows as ghost lips at rotated slider poses — and the
            # composite-side ghost erasure cannot touch it, it is inside the
            # mesh). Legit lips live inside the mouth landmark hull; recolor
            # any lip-red render pixel far outside that hull from skin.
            mf = frontal.astype(np.int16)
            # Strict saturated red: the baked lip strip keeps the original
            # lip saturation; ruddy skin and brownish rim/ear texels drop out.
            lip_r = (mf[..., 2] > mf[..., 1] + 30) & \
                    (mf[..., 2] > mf[..., 0] + 25)
            mouth_hull = cv2.convexHull(lm_f[:2, 48:68].T.astype(np.int32))
            mh = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(mh, mouth_hull, 255)
            # 12px margin per side (25x25 kernel): legit lip texels extend
            # ~8-12px past the landmark hull; the baked strip lies further
            # out and must NOT be shielded.
            mh = cv2.dilate(mh, np.ones((25, 25), np.uint8))
            # Mouth-height band only (brows/chin excluded: eye makeup, rim
            # and chin texels are not the target). Side handling: when the
            # red mass is concentrated on one side, erase only that side —
            # but when it is spread over BOTH sides, erase both: the baked
            # strip STRADDLES the render midline, because the mirror-filled
            # half re-copies lip texels across it (restricting the erasure
            # to the geometric protrusion side left half the strip alive —
            # doubled mouth regressed).
            brow_yf = float(lm_f[1, 17:27].min())
            mouth_yf = float(lm_f[1, 60:68].mean())
            chin_yf = float(lm_f[1, 8])
            yband = (yy > mouth_yf - 0.5 * (mouth_yf - brow_yf)) & \
                    (yy < mouth_yf + 0.6 * (chin_yf - mouth_yf))
            stray = lip_r & (mh == 0) & (acc_closed > 0.05) & yband
            x_ax_i = int(np.clip(round(x_ax), 1, w - 2))
            s_l = float(stray[:, :x_ax_i].sum())
            s_r = float(stray[:, x_ax_i:].sum())
            if max(s_l, s_r) > 80 and min(s_l, s_r) < 0.4 * max(s_l, s_r):
                if s_r >= s_l:
                    stray[:, :x_ax_i] = False
                else:
                    stray[:, x_ax_i:] = False
            if os.environ.get("S2F_DEBUG"):
                ov = frontal.copy()
                ov[stray] = (0, 0, 255)
                _dbg_dump("dbg_s35_stray.png", ov)
            if stray.sum() > 30:
                s32 = frontal.astype(np.float32)
                # Sources exclude lip red (the legit lips sit next to the
                # strip — with them the fill repaints the strip pink).
                skn = ((~stray) & (acc_closed > 0.1) & (~lip_r)).astype(np.float32)
                _diffuse_fill(s32, stray, skn, (4, 8, 16, 32), dynamic=False)
                frontal = np.clip(s32, 0, 255).astype(np.uint8)
                if os.environ.get("S2F_DEBUG"):
                    print(f"[stray-lip] erased {int(stray.sum())} px")

            # Feather the seam between face region and background. Erode the
            # coverage first: the outermost mesh ring sampled off-face clutter
            # (flowers, background), so the blend must start INSIDE the mesh.
            acc_core = cv2.erode((acc_closed * 255).astype(np.uint8),
                                 np.ones((11, 11), np.uint8)).astype(np.float32) / 255.0
            face_mask = np.clip(np.maximum(w_org + w_mir, acc_core), 0, 1)
            # Pull the whole mask ~7px INSIDE the mesh rim: the outermost
            # vertex ring samples off-face pixels and renders as a flat
            # pale "helmet" edge at rotated (slider) poses — the blend
            # must start inside the mesh, with a wide feather below.
            face_mask = cv2.erode(face_mask, np.ones((15, 15), np.uint8))
            # Pre-blur solidity map for the ghost erasure below: includes the
            # mirror-filled interior (w_mir) but, being unblurred, does NOT
            # spill past the mesh rim onto the protruding old silhouette.
            mask_solid = cv2.dilate(
                (face_mask > 0.3).astype(np.uint8),
                np.ones((5, 5), np.uint8)) > 0
            # Whatever dark hair debris survives the forehead cleanup must
            # NOT be composited — carve it out of the face mask (feathered,
            # so the removal fades into the hair above).
            lum_r = frontal.astype(np.float32).mean(axis=2) / 255.0
            # Dark debris carve: (a) forehead/hair remnants above the brows,
            # (b) the dark silhouette rim (hair sampled at the mesh boundary,
            # which shows as a dark halo around the composite edge).
            rim_outer = (acc_closed > 0.05).astype(np.uint8)
            rim_inner = cv2.erode(rim_outer, np.ones((13, 13), np.uint8))
            rim = (rim_outer > 0) & (rim_inner == 0)
            debris = ((region_up & (lum_r < 0.45)) |
                      (rim & (lum_r < 0.40))).astype(np.float32)
            if debris.sum() > 50:
                dm = cv2.GaussianBlur(debris, (0, 0), 5.0)
                face_mask = face_mask * (1.0 - np.clip(dm * 2.0, 0.0, 1.0))
            face_mask = cv2.GaussianBlur(face_mask, (41, 41), 0)
            _dbg_dump("dbg_s4_mask.png", (face_mask * 255).astype(np.uint8))
            _dbg_dump("dbg_s4_frontal.png", frontal)
            face_mask_3 = np.stack([face_mask] * 3, axis=-1)

            result = (frontal.astype(np.float32) * face_mask_3 +
                      work.astype(np.float32) * (1 - face_mask_3)).astype(np.uint8)

            # --- Ghost-chin erasure ---
            # The detected profile's chin/jaw sticks out beyond the frontal
            # face mask and shows as a second chin. Build the original-pose
            # face hull (68 landmarks + forehead extension), subtract the
            # frontal mask, and diffusion-fill the ghost region from the
            # surrounding background/neck. Restricted to below the mouth:
            # that is where the double chin appears, and it keeps the hand
            # near the cheek out of the fill region.
            hull_gh = cv2.convexHull(_with_forehead(lm_det).astype(np.int32))
            gh = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(gh, hull_gh, 255)
            gh = cv2.erode(gh, np.ones(
                (max(3, int(0.02 * span_det)),) * 2, np.uint8))
            protected = cv2.dilate((face_mask > 0.35).astype(np.uint8),
                                   np.ones((9, 9), np.uint8))
            mouth_y = float(lm_det[60:68, 1].mean())
            chin_y_det = float(lm_det[8, 1])
            ghost = ((gh > 0) & (protected == 0) &
                     (yy > mouth_y - 0.10 * (chin_y_det - mouth_y)))
            # The profile's nose tip and lips also protrude beyond the
            # rotated render at mouth/nose height (ghost lips beside the
            # rendered mouth). Extend the erasure upward — but ONLY on the
            # protrusion side: the other side can hold a hand/props near
            # the cheek which must stay untouched. The protrusion side is
            # where the uncovered hull area at brow..mouth height is
            # larger.
            brow_y = float(lm_det[17:27, 1].min())
            # The nose-tip / lip-corner skin protrudes BEYOND the (eroded)
            # landmark hull — use a dilated raw hull for the band so the
            # protrusion rim is inside the erase region.
            gh_band = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(gh_band, hull_gh, 255)
            gh_band = cv2.dilate(gh_band, np.ones((41, 41), np.uint8))
            band_rows = (gh_band > 0) & (yy > brow_y) & \
                (yy <= mouth_y + 0.35 * (chin_y_det - mouth_y))
            # Feature-driven: only erase uncovered-hull pixels where the
            # ORIGINAL image shows facial features — red lips or notably
            # darker-than-local content (nose shadow/nostrils). Skin-toned
            # regions (and a hand/prop near the cheek) stay untouched, so no
            # side selection is needed even though the ghost straddles the
            # render midline.
            wf = work.astype(np.int16)
            lip_red = (wf[..., 2] > wf[..., 1] + 20) & \
                      (wf[..., 2] > wf[..., 0] + 10)
            lum_w = work.astype(np.float32).mean(axis=2) / 255.0
            # Nose/lip shadows are MID-dark; very dark pixels are hair and
            # must not count as features (they would flip the side vote).
            dark_feat = (lum_w < (_fast_blur(lum_w, 12.0) - 0.10)) & \
                        (lum_w > 0.30)
            feat = cv2.dilate(
                ((lip_red | dark_feat) & band_rows).astype(np.uint8),
                np.ones((21, 21), np.uint8)) > 0
            # The ghost lips/nose sit in the face_mask FEATHER zone (the wide
            # mask blur spills past the mesh rim), so `protected` shields
            # them. Test the PRE-BLUR mask instead (mask_solid, captured
            # above): it covers the mirror-filled interior — which acc-based
            # tests miss on back-facing halves — without spilling past the
            # mesh rim.
            render_solid = mask_solid
            band = feat & band_rows & (~render_solid)
            # The pale RIM of the protrusion (lip corner / nose wing, too
            # bright for the feature test) survives the feature band. Sweep
            # the whole uncovered hull on the PROTRUSION side (nose_dx
            # geometry, computed before the stray erasure) — the old
            # silhouette there should become background/neck anyway. The
            # ORIGINAL EYES are carved out: at larger target rotations an
            # eye can protrude past the render, and sweeping it would smear
            # background over the visible eye (a row-based cut would also
            # spare the temple rim and leave a halo — protect the eye hulls
            # specifically instead).
            x_ax_i = int(np.clip(round(x_ax), 1, w - 2))
            eye_h = np.zeros((h, w), np.uint8)
            for e0, e1 in ((36, 42), (42, 48)):
                eh = cv2.convexHull(lm_det[e0:e1].astype(np.int32))
                cv2.fillConvexPoly(eye_h, eh, 255)
            eye_h = cv2.dilate(eye_h, np.ones((31, 31), np.uint8))
            sweep_zone = band_rows & (eye_h == 0)
            side = None
            if abs(nose_dx) > 0.03 * span_det:
                side = np.zeros((h, w), bool)
                if nose_dx > 0:
                    side[:, x_ax_i:] = True
                else:
                    side[:, :x_ax_i] = True
                band = band | (sweep_zone & (~render_solid) & side)
            # Doubled-lip corridor: the render rim baked lip-colored texels
            # (fixed above in render space) AND the original lips show
            # through the mask feather at the same spot — erase the union,
            # dilated into a corridor, on the protrusion side at mouth
            # height, protecting the legit lips (mouth hull + margin).
            corridor = np.zeros((h, w), bool)
            if side is not None:
                corridor = cv2.dilate(
                    (lip_red | (stray > 0)).astype(np.uint8),
                    np.ones((21, 21), np.uint8)) > 0
                corridor = corridor & side & band_rows & (gh_band > 0) & (mh == 0)
                # Anchor to the protrusion: the corridor may only grow
                # ~32px inward from the outside-mesh sweep zone. Without
                # this, reddish eye-makeup / mirrored-rim texels deep inside
                # the face spawn erasure islands (yaw=-20 face-eating bug).
                corridor &= cv2.dilate(
                    (sweep_zone & (~render_solid) & side).astype(np.uint8),
                    np.ones((65, 65), np.uint8)) > 0
                if corridor.sum() > 50:
                    ghost = ghost | corridor
                else:
                    # Below the trust threshold: drop it, or the fill below
                    # would erase an unvetted region AND treat its own
                    # dilated ring as a valid fill source.
                    corridor = np.zeros((h, w), bool)
            if band.sum() > 50:
                ghost = ghost | band
            if os.environ.get("S2F_DEBUG"):
                print(f"[ghost] band={band.sum():.0f} ghost={ghost.sum():.0f} "
                      f"nose_dx={nose_dx:.0f} "
                      f"rows={band_rows.sum():.0f} "
                      f"unsolid={(band_rows & ~render_solid).sum():.0f} "
                      f"brow_y={brow_y:.0f} mouth_y={mouth_y:.0f}")
            erased_small = None
            ghost_mask = None
            if ghost.sum() > 100:
                ghost_mask = ghost.copy()
                if os.environ.get("S2F_DEBUG"):
                    ov = result.copy()
                    ov[ghost_mask] = (0, 0, 255)
                    _dbg_dump("dbg_s5_ghost_ov.png", ov)
                r32 = result.astype(np.float32)
                # Fill ONLY from outside the DILATED original face hull:
                # sources inside the hull would re-paint the ghost chin with
                # chin skin, and sources in the ring between the eroded and
                # dilated hull still hold the protruding lip/nose color —
                # both keep the ghost alive. Neck and background give a
                # natural continuation.
                # The lip corridor sits between cheek skin (above/below,
                # inside the render) and background (right): fill it from
                # BOTH, so it becomes cheek-colored, not a green bite. The
                # skin source uses mask_solid, NOT acc_closed — Sim3DR culls
                # back-facing triangles, so the mirror-filled occluded half
                # has acc=0 and would be rejected as a source, leaving only
                # background pixels (the green bite returns).
                # Union: mask_solid adds the mirror-filled interior
                # (acc=0 on the culled half), acc_closed>0.3 keeps the
                # front-facing render rim — mask_solid alone is eroded
                # ~10px inside it and starves the fill of skin sources.
                skin_src = (mask_solid | (acc_closed > 0.3)) & (~lip_r)
                if corridor.any():
                    kn_c = ((~ghost) &
                            ((gh_band == 0) | skin_src)).astype(np.float32)
                    filled_c = _diffuse_fill(r32, corridor, kn_c,
                                             (8, 16, 32, 64), dynamic=False)
                    ghost[filled_c] = False
                else:
                    filled_c = np.zeros((h, w), bool)
                kn = ((~ghost) & (gh_band == 0)).astype(np.float32)
                filled_g = _diffuse_fill(r32, ghost, kn,
                                         (8, 16, 32, 64), dynamic=False)
                # Second pass for the ghost pixels outside-hull sources
                # cannot reach (den never exceeds the threshold deep inside
                # the hull). Unfilled ghost pixels keep the ORIGINAL photo
                # content — that IS the doubled-lip/chin ghost — so fill
                # them from the render's own skin (they all hug the render
                # rim). Never use the hull ring between the eroded/dilated
                # hulls: it still holds the protruding lip/nose color.
                rem = ghost & ~filled_g
                if rem.any():
                    kn2 = ((~rem) &
                           ((gh_band == 0) | skin_src)).astype(np.float32)
                    filled_g |= _diffuse_fill(r32, rem, kn2,
                                              (8, 16, 32, 64), dynamic=False)
                if os.environ.get("S2F_DEBUG"):
                    print(f"[ghost-fill] corridor={int(corridor.sum())} "
                          f"filled_c={int(filled_c.sum())} "
                          f"ghost_pre={int(ghost.sum())} "
                          f"filled_g={int(filled_g.sum())} "
                          f"kn={int((kn > 0).sum())}")
                result = np.clip(r32, 0, 255).astype(np.uint8)
                erased_small = result
                _dbg_dump("dbg_s55_erased_small.png", erased_small)
                if os.environ.get("S2F_DEBUG"):
                    ov = result.copy()
                    ov[ghost & ~filled_g] = (0, 0, 255)
                    _dbg_dump("dbg_s57_unfilled.png", ov)

            # --- Full-resolution photo-detail pass ---
            # The vertex-color render is mesh-resolution limited (~4px per
            # vertex at full res), which is the main source of the "blurry"
            # look. Rasterize triangle ids + barycentric weights (the kernel
            # returns them in float — no uint8 color quantization), then
            # sample the ORIGINAL photo per-pixel and add its high-frequency
            # detail back — but only where the render holds REAL, visible,
            # unedited texture: the mirror-filled half, synthetic fills and
            # cleanup repairs must not get their artifacts (specks, strands,
            # stray lips, the repaired mouth opening) resurrected.
            tex_hf = None
            detail_gate = None
            detail_crop = None
            if self._photo_detail > 0:
                import Sim3DR_Cython
                tri_buf = np.full((h, w), -1, np.int32)
                # The pyx declares the weight buffer 2D; the kernel writes
                # it flat as [y*w*3 + x*3 + k] → shape (h, w*3).
                bary = np.zeros((h, w * 3), np.float32)
                depth = np.full((h, w), -1e8, np.float32)
                Sim3DR_Cython.rasterize_triangles(
                    ver_c, tri, depth, tri_buf, bary, tri.shape[0], h, w)
                valid_d = tri_buf >= 0
                if valid_d.any():
                    T = tri[tri_buf[valid_d]]          # (P,3) vertex ids
                    wts = bary.reshape(h, w, 3)[valid_d]  # (P,3)
                    map_x = np.zeros((h, w), np.float32)
                    map_y = np.zeros((h, w), np.float32)
                    # Barycentric UV = source position in the ORIGINAL
                    # (full-res) photo — ver[:2] holds full-res coords.
                    map_x[valid_d] = (wts * ver[0, T]).sum(axis=1)
                    map_y[valid_d] = (wts * ver[1, T]).sum(axis=1)
                    # Work on the full-res bbox of the mesh only — remap +
                    # blur at full canvas res dominated the boost cost.
                    dys, dxs = np.where(valid_d)
                    mg_w = int(16 * scale) + 2  # blur support, work px
                    dx0 = max(int(dxs.min()) - mg_w, 0)
                    dx1 = min(int(dxs.max()) + mg_w, w - 1)
                    dy0 = max(int(dys.min()) - mg_w, 0)
                    dy1 = min(int(dys.max()) + mg_w, h - 1)
                    fx0 = int(dx0 / scale)
                    fy0 = int(dy0 / scale)
                    fx1 = min(int(np.ceil((dx1 + 1) / scale)), w0)
                    fy1 = min(int(np.ceil((dy1 + 1) / scale)), h0)
                    cw, ch = fx1 - fx0, fy1 - fy0
                    # Upscale the coordinate maps, not the texture: UV is
                    # piecewise-linear across a triangle, so bilinear
                    # upsampling is exact there — full-res sampling for the
                    # price of a work-res rasterization.
                    map_x = cv2.resize(map_x[dy0:dy1 + 1, dx0:dx1 + 1],
                                       (cw, ch), interpolation=cv2.INTER_LINEAR)
                    map_y = cv2.resize(map_y[dy0:dy1 + 1, dx0:dx1 + 1],
                                       (cw, ch), interpolation=cv2.INTER_LINEAR)
                    tex_full = cv2.remap(
                        image, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
                    sig_d = 2.0
                    tex_hf = tex_full - cv2.GaussianBlur(
                        tex_full, (0, 0), sig_d)
                    # Taper at the silhouette: the blur bleeds the
                    # zero-filled exterior into the rim (ringing).
                    val_f = cv2.resize(
                        valid_d[dy0:dy1 + 1, dx0:dx1 + 1].astype(np.float32),
                        (cw, ch), interpolation=cv2.INTER_LINEAR)
                    val_f = cv2.GaussianBlur(val_f, (0, 0), sig_d)
                    tex_hf *= np.clip((val_f - 0.5) / 0.3, 0.0, 1.0)[..., None]
                    detail_crop = (fy0, fx0, ch, cw)

                    # Gate (work res): conf_map = real sampled texture,
                    # hard_vis = originally-visible (not mirror-filled),
                    # minus every repair that removed real content.
                    detail_gate = conf_map * np.clip(
                        (vis_map - 0.55) / 0.15, 0.0, 1.0)
                    if mouth_trust is not None:
                        detail_gate = detail_gate * (
                            1.0 - 0.8 * np.clip(mouth_trust, 0.0, 1.0))
                    detail_gate[stray] = 0.0
                    for em in clean_edits:
                        detail_gate[em] = 0.0
                    detail_gate = cv2.GaussianBlur(detail_gate, (0, 0), 2.0)
                    if os.environ.get("S2F_DEBUG"):
                        print(f"[detail] valid={int(valid_d.sum())} "
                              f"gate_px={int((detail_gate > 0.3).sum())} "
                              f"strength={self._photo_detail:.2f}")
                        _dbg_dump("dbg_detail_gate.png",
                                  (detail_gate * 255).astype(np.uint8))

            # Upsample the small-canvas render + mask and composite onto the
            # FULL-resolution original: background and neck keep their native
            # sharpness, the face is vertex-limited anyway.
            if scale < 1.0:
                frontal_full = cv2.resize(frontal, (w0, h0),
                                          interpolation=cv2.INTER_CUBIC)
                fm_full = cv2.resize(face_mask, (w0, h0),
                                     interpolation=cv2.INTER_LINEAR)
                # Blur AFTER upsampling: keep the work-canvas-equivalent
                # feather width (sigma/scale), or large photos get a
                # harder composite edge than intended.
                fm_full = cv2.GaussianBlur(fm_full, (0, 0), 3.0 / scale)
            else:
                frontal_full, fm_full = frontal, face_mask
            if tex_hf is not None:
                fy0, fx0, ch, cw = detail_crop
                # detail_gate lives at work res — resize the matching crop.
                gy0 = int(fy0 * scale)
                gx0 = int(fx0 * scale)
                gate_crop = detail_gate[gy0:gy0 + int(ch * scale) + 2,
                                        gx0:gx0 + int(cw * scale) + 2]
                gate_full = cv2.resize(gate_crop, (cw, ch),
                                       interpolation=cv2.INTER_LINEAR)
                region = frontal_full[fy0:fy0 + ch, fx0:fx0 + cw]
                frontal_full[fy0:fy0 + ch, fx0:fx0 + cw] = np.clip(
                    region.astype(np.float32) +
                    tex_hf * (gate_full * self._photo_detail)[..., None],
                    0, 255).astype(np.uint8)
            fm_full_3 = np.stack([fm_full] * 3, axis=-1)
            result = (frontal_full.astype(np.float32) * fm_full_3 +
                      image.astype(np.float32) * (1 - fm_full_3)).astype(np.uint8)
            # The re-composite above overwrote the ghost-filled result —
            # paste the erased small-canvas content back inside the ghost
            # region (feathered), or the ghost chin/lips reappear.
            if ghost_mask is not None:
                gm = cv2.resize(ghost_mask.astype(np.float32), (w0, h0),
                                interpolation=cv2.INTER_LINEAR)
                # Wide feather: the erasure can shift the face silhouette
                # (lip corridor), and a soft edge reads as the photo's own
                # shallow depth of field instead of a cut-out step. The
                # blur runs at full resolution, so the 1024-canvas-tuned
                # sigma must be scaled up (sigma/scale) or large photos
                # get a harder seam than intended.
                gm = cv2.GaussianBlur(gm, (0, 0), 4.0 / scale)[..., None]
                erased_full = erased_small if scale >= 1.0 else cv2.resize(
                    erased_small, (w0, h0), interpolation=cv2.INTER_CUBIC)
                result = (result.astype(np.float32) * (1 - gm) +
                          erased_full.astype(np.float32) * gm).astype(np.uint8)
                _dbg_dump("dbg_s6_erased.png", result)

            if self._edge_smooth:
                # Only smooth the face region, not the whole image
                smoothed = cv2.bilateralFilter(result, 5, 30, 30)
                inner = cv2.erode((fm_full * 255).astype(np.uint8),
                                  np.ones((15, 15), np.uint8)) / 255.0
                inner_3 = np.stack([inner] * 3, axis=-1)
                result = (smoothed.astype(np.float32) * inner_3 +
                          result.astype(np.float32) * (1 - inner_3)).astype(np.uint8)

            info = (f"3DMM纹理重渲染+软对称 | 检测偏转: yaw={self._detected_yaw:.0f}deg "
                    f"pitch={self._detected_pitch:.0f}deg roll={self._detected_roll:.0f}deg | "
                    f"目标: yaw={self._yaw:.0f} pitch={self._pitch:.0f} roll={self._roll:.0f}")
            # ver_frontal keeps full-resolution coordinates — lm_tgt must be
            # in original-image coords for the GAN warp paste-back.
            lm_tgt = ver_frontal[:2, kp].T.astype(np.float32)
            if scale < 1.0:
                conf_map = cv2.resize(conf_map, (w0, h0),
                                      interpolation=cv2.INTER_LINEAR)
                acc_gray = cv2.resize(acc_gray, (w0, h0),
                                      interpolation=cv2.INTER_LINEAR)
            if mouth_trust is not None:
                if conf_map.shape != mouth_trust.shape:
                    mouth_trust = cv2.resize(
                        mouth_trust, (conf_map.shape[1], conf_map.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
                conf_map = np.maximum(conf_map, mouth_trust)
            return {"result": result, "rendered": rendered,
                    "acc_gray": acc_gray, "face_mask": fm_full,
                    "conf_map": conf_map, "ver_frontal": ver_frontal,
                    "lm_tgt": lm_tgt, "frontal_small": frontal,
                    "mouth_trust": mouth_trust,
                    "info": info}

        except Exception as e:
            print(f"3DMM render failed: {e}")
            import traceback; traceback.print_exc()
            return None

    def _simple_mirror(self, image: np.ndarray, elapsed_ms: float,
                       face_bbox: tuple = None) -> FrontalizationResult:
        """
        Simple horizontal mirror for animals or when no 3D model is available.

        Direction: mirror the side of the image where the face is located
        (visible cheek) to fill the occluded side.
        Fixes the "two noses, two mouths" bug by NOT concatenating halves.
        Instead, we do a complete horizontal flip of the face region and
        blend with feathered edge at the center.
        """
        h, w = image.shape[:2]

        if face_bbox is None:
            try:
                cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                boxes = cascade.detectMultiScale(gray, 1.1, 5)
            except (AttributeError, cv2.error):
                boxes = []
            if len(boxes) == 0:
                # No detector available — fall back to the image center.
                cs = int(min(h, w) * 0.4)
                x, y, bw, bh = (w - cs) // 2, (h - cs) // 2, cs, cs
        else:
            x, y, bw, bh = face_bbox

        face_cx = x + bw // 2
        img_cx = w // 2

        # Face on left side of image → visible is left → mirror left to fill right
        # Face on right side of image → visible is right → mirror right to fill left
        mirror_left_to_right = (face_cx < img_cx)

        pad = int(bw * 0.1)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
        face_roi = image[y1:y2, x1:x2].copy()
        fw, fh = x2 - x1, y2 - y1

        # Complete horizontal flip — NOT half-and-half concatenation
        mirrored = cv2.flip(face_roi, 1)

        # Feathered blend at center
        blend_w = max(3, fw // 8)
        mask = np.zeros((fh, fw), dtype=np.float32)
        if mirror_left_to_right:
            # Blend zone at center-right
            mask[:, fw//2 - blend_w:fw//2] = np.linspace(0, 1, blend_w)[np.newaxis, :]
            mask[:, fw//2:] = 1.0
        else:
            # Blend zone at center-left
            mask[:, fw//2:fw//2 + blend_w] = np.linspace(1, 0, blend_w)[np.newaxis, :]
            mask[:, :fw//2] = 1.0
        mask_3ch = np.stack([mask] * 3, axis=-1)

        blended = (mirrored.astype(np.float32) * mask_3ch +
                   face_roi.astype(np.float32) * (1 - mask_3ch)).astype(np.uint8)

        result = image.copy()
        result[y1:y2, x1:x2] = blended
        result = cv2.bilateralFilter(result, 5, 30, 30)

        direction = "左→右" if mirror_left_to_right else "右→左"
        return FrontalizationResult(result, elapsed_ms, f"镜像({direction}) | {elapsed_ms:.0f}ms")
