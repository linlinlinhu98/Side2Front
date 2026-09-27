import numpy as np
import cv2
import time
from scipy.interpolate import RBFInterpolator
from .base import FaceFrontalizer, FrontalizationResult
from .fill_utils import diffuse_fill


DEFAULT_KEYPOINTS = [
    "hairline_left", "hairline_center", "hairline_right",
    "brow_left_inner", "brow_left_outer",
    "eye_left_inner", "eye_left_outer",
    "nose_tip", "nose_left", "nose_right",
    "mouth_left", "mouth_center", "mouth_right",
    "chin_tip", "chin_left", "chin_right",
    "ear_top", "ear_bottom",
]

# Points that must end up ON the symmetry axis after warping.
CENTER_POINTS = ["hairline_center", "nose_tip", "mouth_center", "chin_tip"]


class AnimeFaceFrontalizer(FaceFrontalizer):
    name = "anime"
    display_name = "动漫人脸"

    def __init__(self):
        self._mirror_strength = 1.0
        # Warp strength toward the frontal template: 1.0 = full
        # frontalization, 0.0 = no deformation (mirror only).
        self._tps_flexibility = 1.0
        self._poisson_blend = True
        self._color_match = True
        self._erase_center = True
        # Optional generated frontal reference used ONLY as a nose
        # stroke donor (see _transplant_nose_strokes). Set via
        # set_nose_reference(); None = classical nose synthesis only.
        self._nose_ref_img = None
        self._nose_ref_kps = None
        # Optional generated frontal reference used as a FULL
        # face-interior stroke donor (eyes+nose+mouth). Set via
        # set_face_reference(); takes precedence over the nose donor.
        self._face_ref_img = None
        self._face_ref_kps = None

    def set_nose_reference(self, ref_img: np.ndarray,
                           ref_kps: dict) -> None:
        """Register a generated frontal reference as a nose-stroke
        donor. Only its nose linework is transplanted; everything else
        in the result stays original-pixel."""
        self._nose_ref_img = ref_img
        self._nose_ref_kps = ref_kps

    def set_face_reference(self, ref_img: np.ndarray,
                           ref_kps: dict) -> None:
        """Register a generated frontal reference as a face-interior
        stroke donor (eyes + nose + mouth linework; the composite's
        own mirrored interior is erased to paper tone first). Brows
        and everything outside the face stay original-pixel."""
        self._face_ref_img = ref_img
        self._face_ref_kps = ref_kps

    def get_default_params(self) -> dict:
        return {
            "mirror_strength": 1.0,
            "tps_flexibility": 1.0,
            "poisson_blend": True,
            "color_match": True,
        }

    def update_params(self, **kwargs) -> None:
        if "mirror_strength" in kwargs:
            self._mirror_strength = float(kwargs["mirror_strength"])
        if "tps_flexibility" in kwargs:
            self._tps_flexibility = float(kwargs["tps_flexibility"])
        if "poisson_blend" in kwargs:
            self._poisson_blend = bool(kwargs["poisson_blend"])
        if "color_match" in kwargs:
            self._color_match = bool(kwargs["color_match"])

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _axis_x(self, keypoints: dict) -> float:
        """Vertical symmetry axis of the frontal face.

        For a profile, the mid-sagittal plane projects onto the silhouette
        line (forehead -> nose tip -> lips -> chin), so the axis is the
        average x of the annotated center points.
        """
        xs = [keypoints[n][0] for n in CENTER_POINTS if n in keypoints]
        if xs:
            return float(np.mean(xs))
        return float(np.mean([p[0] for p in keypoints.values()]))

    def _visible_side(self, keypoints: dict, cx: float) -> str:
        """Which side of the axis the drawn features are on.

        A face looking left has its nose tip at the far left and all drawn
        features (eye, brow, ear) to the RIGHT of the axis -> "right".
        Mirroring must copy from this side, not the other way around.
        """
        left = right = 0
        for name, (x, _y) in keypoints.items():
            if name in CENTER_POINTS:
                continue
            if any(t in name for t in ("eye", "brow", "ear", "hairline")):
                if x < cx:
                    left += 1
                else:
                    right += 1
        if left == 0 and right == 0:
            for x, _y in keypoints.values():
                if x < cx:
                    left += 1
                else:
                    right += 1
        return "right" if right >= left else "left"

    def _build_frontal_template(self, keypoints: dict, cx: float,
                                pin_center: bool = False) -> dict:
        """Symmetric target positions for every annotated keypoint.

        Center points go onto the axis; lateral points keep their distance
        to the axis (averaged for annotated pairs) with a mild widening —
        a profile compresses the face depth, a frontal view is wider.
        With `pin_center` (3/4 views) center points keep their source
        position: the muzzle is already near-frontal, and pulling the
        nose tip sideways onto the axis would stretch it 2x.
        """
        widen = 1.15
        frontal = {}
        # Average the distance of annotated left/right pairs. In 3/4 mode
        # the hairline "pair" is not a real pair: the silhouette-side
        # point sits almost on the muzzle axis, so averaging would yank
        # it across the axis and fold the forehead in the TPS.
        pair_dist = {}
        for name, (x, _y) in keypoints.items():
            if "left" in name:
                if pin_center and "hairline" in name:
                    continue
                other = name.replace("left", "right")
                if other in keypoints:
                    d = (abs(x - cx) + abs(keypoints[other][0] - cx)) / 2
                    pair_dist[name] = pair_dist[other] = d
        for name, (x, y) in keypoints.items():
            if name in CENTER_POINTS:
                frontal[name] = (x, y) if pin_center else (cx, y)
            else:
                d = pair_dist.get(name, abs(x - cx)) * widen
                side = -1.0 if x < cx else 1.0
                frontal[name] = (cx + side * d, y)

        # Re-space the eyes and brows: in a profile the inner corner sits
        # almost on the axis, so plain mirroring leaves the face cross-eyed.
        # A frontal face has roughly one eye-width between the eyes — keep
        # each feature's drawn width and push it outward as a unit.
        eye_d = None
        for inner, outer in (("eye_left_inner", "eye_left_outer"),
                             ("brow_left_inner", "brow_left_outer")):
            if inner in keypoints and outer in keypoints:
                fw = abs(keypoints[outer][0] - keypoints[inner][0])
                if fw < 3:
                    continue
                side = 1.0 if keypoints[outer][0] >= cx else -1.0
                # In a 3/4 view the visible eye is already near its
                # frontal position — re-space it less, or the inward pull
                # shears the forehead against the pinned hairline.
                d_in = (0.75 if pin_center else 0.55) * fw
                frontal[inner] = (cx + side * d_in, keypoints[inner][1])
                frontal[outer] = (cx + side * (d_in + fw),
                                  keypoints[outer][1])
                if "eye" in inner:
                    eye_d = d_in + fw

        # Ears: no widening (a profile ear is already far from the axis —
        # widening it like the facial features makes the frontal ears
        # huge), and cap the spread relative to the eye span. Real frontal
        # heads: ear-tip span ≈ 2-2.3x the outer-eye span. In 3/4 mode the
        # whole side-of-head outline (hairline + ear) is scaled UNIFORMLY
        # instead — scaling one point alone reorders it against its
        # neighbours and folds the TPS.
        if pin_center:
            for name in ("hairline_right", "ear_top", "ear_bottom"):
                if name in keypoints:
                    x, y = keypoints[name]
                    side = -1.0 if x < cx else 1.0
                    d = abs(x - cx) * widen * 0.88
                    frontal[name] = (cx + side * d, y)
        elif eye_d is not None:
            for name in ("ear_top", "ear_bottom"):
                if name in frontal:
                    x, y = frontal[name]
                    side = 1.0 if x >= cx else -1.0
                    d = min(abs(x - cx) / widen, 2.3 * eye_d)
                    frontal[name] = (cx + side * d, y)
        return frontal

    def _tps_warp(self, image: np.ndarray, src_points: np.ndarray,
                  dst_points: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]

        # Anchor the image borders so the background does not rubber-sheet:
        # only the face interior is free to deform.
        m = 8
        border_src = np.array([
            [m, m], [w - 1 - m, m], [m, h - 1 - m], [w - 1 - m, h - 1 - m],
            [w / 2, m], [w / 2, h - 1 - m], [m, h / 2], [w - 1 - m, h / 2],
        ], dtype=np.float64)
        src = np.vstack([src_points, border_src])
        dst = np.vstack([dst_points, border_src])

        # cv2.remap needs the BACKWARD map (for each destination pixel,
        # which source pixel to sample), so fit the RBF dst -> src.
        # Small smoothing only regularizes near-duplicate points; the
        # user-facing warp strength is applied by the caller as a blend
        # between src and dst (RBF smoothing mapped 0..3px was invisible).
        try:
            rbf = RBFInterpolator(dst, src, kernel="thin_plate_spline",
                                  smoothing=1.0)
        except Exception:
            rbf = RBFInterpolator(dst, src, smoothing=1.0)

        step = 2
        ys, xs = np.mgrid[0:h:step, 0:w:step]
        grid = np.column_stack([xs.ravel(), ys.ravel()])
        mapped = rbf(grid).astype(np.float32)

        map_x = mapped[:, 0].reshape(ys.shape)
        map_y = mapped[:, 1].reshape(ys.shape)
        map_x = cv2.resize(map_x, (w, h), interpolation=cv2.INTER_CUBIC)
        map_y = cv2.resize(map_y, (w, h), interpolation=cv2.INTER_CUBIC)

        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    def _face_hull_mask(self, frontal_kps: dict, shape: tuple,
                        cx: float) -> np.ndarray:
        """Feathered convex hull of the frontal keypoints (float 0..1).

        Each point's reflection about the axis is included: the mirrored
        copy of an eye/ear lands there, and it must stay inside the hull
        or the composite would cut it off.
        """
        h, w = shape[:2]
        pts = list(frontal_kps.values())
        # The detector has no keypoint on the cheek/jaw arc, so the
        # straight ear->chin hull edge cuts through the cheek bulge — the
        # mirrored jawline lands OUTSIDE the mask and fades to background
        # ("washed-out mirrored half"). Add synthetic cheek points pushed
        # outward along the ear->chin segment to keep the jawline inside.
        ear = frontal_kps.get("ear_bottom")
        if ear is not None:
            for chin_n in ("chin_left", "chin_right"):
                if chin_n in frontal_kps:
                    gx, gy = frontal_kps[chin_n]
                    if (gx - cx) * (ear[0] - cx) >= 0:  # same side only (>=: a
                        # point exactly ON the axis must not drop the bulge)
                        mx, my = (ear[0] + gx) / 2.0, (ear[1] + gy) / 2.0
                        d = float(np.hypot(gx - ear[0], gy - ear[1]))
                        out_dir = 1.0 if mx >= cx else -1.0
                        pts.append((mx + out_dir * 0.30 * d, my))
        pts += [(2 * cx - x, y) for x, y in pts]
        pts = np.array(pts, dtype=np.float32)
        span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1.0)
        hull = cv2.convexHull(pts.astype(np.int32))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        grow = int(span * 0.12) | 1
        mask = cv2.dilate(mask, np.ones((grow, grow), np.uint8))
        mask = cv2.GaussianBlur(mask, (0, 0), span * 0.05)
        mask = mask.astype(np.float32) / 255.0

        # Cut the hull along the jawline: the dilation above extends the
        # mask below the chin, where the composite would show the mirrored
        # neck next to the original one ("two necks"). Below the chin only
        # a centered NECK BAND of the composite shows through — the mirror
        # step already filled it with a symmetric copy of the visible
        # neck, so the frontal face gets a neck centered on its axis
        # instead of the original one dangling off to the profile side.
        chin_pts = [frontal_kps[n] for n in
                    ("chin_left", "chin_tip", "chin_right")
                    if n in frontal_kps]
        if chin_pts:
            chin_pts = np.array(chin_pts, dtype=np.float32)
            # mirror side points about the axis and sort along x so the
            # polyline spans the whole jaw
            jaw = np.concatenate([chin_pts,
                                  [(2 * cx - x, y) for x, y in chin_pts]])
            jaw = jaw[np.argsort(jaw[:, 0])]
            xs = np.arange(w, dtype=np.float32)
            jaw_y = np.interp(xs, jaw[:, 0], jaw[:, 1])
            feather = max(10.0, span * 0.05)
            yy = np.arange(h, dtype=np.float32)[:, None]
            below = np.clip((yy - jaw_y[None, :]) / feather, 0.0, 1.0)

            chin_xs = [frontal_kps[n][0] for n in ("chin_left", "chin_right")
                       if n in frontal_kps]
            neck_half = None
            if chin_xs:
                # 1.6x, not 0.8x: the mirrored chin contour bulges wider
                # than the chin keypoints, and a narrow band clipped the
                # chin corners to background (faint chin, center notch).
                neck_half = 1.6 * max(abs(x - cx) for x in chin_xs)
            elif "chin_tip" in frontal_kps:
                # No chin side points: without a neck band the cut below
                # zeroes the mask across the FULL width and the original
                # profile neck dangles under the frontalized face. Fall
                # back to a chin_tip-centered band scaled from the eyes.
                eye_xs = [frontal_kps[n][0] for n in
                          ("eye_left_inner", "eye_left_outer")
                          if n in frontal_kps]
                if eye_xs:
                    neck_half = max(abs(x - cx) for x in eye_xs)
            if neck_half is not None:
                neck_half = max(neck_half, 12.0)
                xs1 = xs[None, :]
                neck = np.clip((neck_half - np.abs(xs1 - cx)) /
                               max(8.0, 0.35 * neck_half), 0.0, 1.0)
                below = below * (1.0 - neck)
            mask *= (1.0 - below)
        return mask

    def _erase_center_lines(self, warped: np.ndarray, cx: float,
                            frontal_kps: dict) -> np.ndarray:
        """Erase the profile nose/lips silhouette around the axis.

        After the TPS warp the nose tip sits on the axis, but the dark
        profile outline (nose bridge -> tip -> philtrum) still cuts
        through the middle of the face. A frontal anime face needs none
        of it, so diffusion-fill those dark line pixels from the
        surrounding skin. Eyes, brows and the mouth stay outside the band.
        """
        h, w = warped.shape[:2]
        eye_ys = [frontal_kps[n][1] for n in
                  ("eye_left_inner", "eye_left_outer") if n in frontal_kps]
        brow_ys = [frontal_kps[n][1] for n in
                   ("brow_left_inner", "brow_left_outer") if n in frontal_kps]
        mouth_ys = [frontal_kps[n][1] for n in
                    ("mouth_left", "mouth_center", "mouth_right")
                    if n in frontal_kps]
        eye_xs = [frontal_kps[n][0] for n in
                  ("eye_left_inner", "eye_left_outer") if n in frontal_kps]
        if not eye_ys or not mouth_ys or not eye_xs:
            return warped

        eye_d = max(abs(x - cx) for x in eye_xs)
        y0 = max(eye_ys) + 8               # stay below the eyes
        y1 = min(mouth_ys) - 6             # stop above the mouth
        xs = np.arange(w, dtype=np.float32)[None, :]
        yy = np.arange(h, dtype=np.float32)[:, None]

        # Lower band: the whole nose silhouette (bridge curves well off
        # the axis before the tip lands back on it) down to the philtrum.
        bands = [(np.abs(xs - cx) < max(14.0, 0.9 * eye_d)) &
                 (yy >= y0) & (yy < y1)]
        # Upper band: the bridge segment between the brows. Narrow, so the
        # (mirrored) brows at +-0.37*eye_d and wider stay untouched.
        if brow_ys:
            bands.append(
                (np.abs(xs - cx) < max(10.0, 0.32 * eye_d)) &
                (yy >= min(brow_ys) - 12) & (yy < y0))
        # Sub-mouth band: the profile lip/chin silhouette remnant that
        # lands on the axis between mouth and chin.
        chin_ys = [frontal_kps[n][1] for n in
                   ("chin_tip", "chin_left", "chin_right") if n in frontal_kps]
        if chin_ys:
            bands.append(
                (np.abs(xs - cx) < max(8.0, 0.40 * eye_d)) &
                (yy >= max(mouth_ys) + 4) & (yy < max(chin_ys) - 6))

        f = warped.astype(np.float32)
        lum = f.mean(axis=2) / 255.0
        ones = np.ones((h, w), np.float32)
        for band in bands:
            unknown = band & (lum < 0.65)
            if unknown.sum() < 20:
                continue
            diffuse_fill(f, unknown, ones, (3, 6, 12, 24))
            # Blend the band toward its own blur: kills leftover gray
            # edges/rings from partially-erased anti-aliased lines.
            bm = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 4.0)
            smooth = cv2.GaussianBlur(f, (0, 0), 6.0)
            b3 = (bm * 0.6)[..., None]
            f = f * (1.0 - b3) + smooth * b3
        return np.clip(f, 0, 255).astype(np.uint8)

    def _synthesize_nose(self, image: np.ndarray, cx: float,
                         frontal_kps: dict, mask: np.ndarray) -> np.ndarray:
        """Draw a soft nose-bridge shadow on the midline (pure profiles).

        Mirroring a pure profile leaves a blank midline between the eyes
        and the mouth: the profile nose was erased and there is no
        far-side nose to copy. A frontal anime face only needs a hint —
        a soft vertical bridge shadow plus a faint blob under the tip, a
        gentle multiplicative darkening of the local skin tone.
        """
        h, w = image.shape[:2]
        eye_ys = [frontal_kps[n][1] for n in
                  ("eye_left_inner", "eye_left_outer") if n in frontal_kps]
        eye_xs = [frontal_kps[n][0] for n in
                  ("eye_left_inner", "eye_left_outer") if n in frontal_kps]
        mouth_ys = [frontal_kps[n][1] for n in
                    ("mouth_left", "mouth_center", "mouth_right")
                    if n in frontal_kps]
        if not eye_ys or not mouth_ys or not eye_xs:
            return image
        eye_d = max(abs(x - cx) for x in eye_xs)
        y_eye = max(eye_ys)
        y_mouth = min(mouth_ys)
        nose = frontal_kps.get("nose_tip")
        y_tip = (nose[1] if nose is not None
                 else y_mouth - 0.28 * (y_mouth - y_eye))
        y_top = y_eye + 0.12 * (y_mouth - y_eye)
        length = y_tip - y_top
        if length < 8:
            return image

        xs = np.arange(w, dtype=np.float32)[None, :]
        yy = np.arange(h, dtype=np.float32)[:, None]

        # Bridge: narrow vertical soft band, fading in below the eyes and
        # out at the tip.
        sig_x = max(2.5, 0.075 * eye_d)
        prof_y = (np.clip((yy - y_top) / (0.35 * length), 0.0, 1.0) *
                  np.clip((y_tip + 0.10 * length - yy) /
                          (0.25 * length), 0.0, 1.0))
        bridge = np.exp(-((xs - cx) ** 2) / (2 * sig_x ** 2)) * prof_y

        # Tip: small soft blob just under the nose tip (base shadow).
        sig_tx = max(3.0, 0.16 * eye_d)
        sig_ty = max(2.0, 0.045 * length)
        tip = np.exp(-((xs - cx) ** 2) / (2 * sig_tx ** 2) -
                     ((yy - (y_tip + 0.06 * length)) ** 2) /
                     (2 * sig_ty ** 2))

        a = np.clip(bridge + 1.1 * tip, 0.0, 1.0) * mask
        a = cv2.GaussianBlur(a, (0, 0), 1.2)[..., None]
        f = image.astype(np.float32)
        out = f * (1.0 - 0.20 * a)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _mirror_completion(self, warped: np.ndarray, cx: float,
                           side: str, strength: float,
                           keep_region: tuple = None) -> np.ndarray:
        """Copy the visible half onto the occluded half (flipped).

        `side` is where the visible features are ("right" for a face
        looking left). `keep_region` = (y0, y1, d0, d1) — or a list of
        such tuples — marks bands of a 3/4 view: rows y0..y1 within d0 px
        of the axis keep the ORIGINAL content (the real, already
        near-frontal nose/mouth — mirroring them would double the nose —
        and the near-axis forehead, where mirroring the far ear/stray fur
        that straddles the axis paints a symmetric "diamond" on the
        forehead), ramping to full mirror strength at d1. Outside the
        bands (eye level) the mirror applies at full strength so the
        foreshortened far eye is replaced.
        """
        h, w = warped.shape[:2]
        cx_i = int(np.clip(round(cx), 1, w - 2))
        result = warped.copy()

        if side == "right":
            # The near-axis band of the visible half, so that
            # output[cx-d] = warped[cx+d].
            src = warped[:, cx_i:min(2 * cx_i, w)]
            flipped = cv2.flip(src, 1)
            if flipped.shape[1] < cx_i:
                flipped = cv2.copyMakeBorder(
                    flipped, 0, 0, 0, cx_i - flipped.shape[1],
                    cv2.BORDER_REFLECT_101)
            patch = flipped[:, :cx_i]
            target = slice(0, cx_i)
        else:
            need = w - cx_i
            src = warped[:, max(0, cx_i - need):cx_i]
            flipped = cv2.flip(src, 1)
            if flipped.shape[1] < need:
                flipped = cv2.copyMakeBorder(
                    flipped, 0, 0, 0, need - flipped.shape[1],
                    cv2.BORDER_REFLECT_101)
            patch = flipped[:, :need]
            target = slice(cx_i, w)

        pw = patch.shape[1]
        # Distance of each target column from the axis: for side="right"
        # the patch fills columns 0..cx_i-1 (column cx_i-1 hugs the axis),
        # for side="left" it fills cx_i..w-1 (column 0 of the patch hugs
        # the axis).
        if side == "right":
            d = cx_i - 1 - np.arange(pw, dtype=np.float32)
        else:
            d = np.arange(pw, dtype=np.float32)
        # Nearly flat: the profile nose/lip lines near the axis were
        # already erased, so there is nothing left worth preserving at
        # the seam — a short ramp only hides the flip discontinuity.
        feather = max(4.0, pw / 30.0)
        base_alpha = strength * np.where(d < feather,
                                         0.4 + 0.6 * d / feather, 1.0)
        if keep_region:
            bands_in = (keep_region if isinstance(keep_region, list)
                        else [keep_region])
            yy = np.arange(h, dtype=np.float32)[:, None]
            alpha = None
            for spec in bands_in:
                y0, y1, d0, d1 = spec[:4]
                # Short feather above (the far eye sits just over the band
                # and must stay fully mirrored), longer one below. A band
                # may override both via a 6-tuple — the forehead band needs
                # a SHORT lower feather, or it reaches the brow/eye rows
                # and half-keeps the far brow/eye the mirror must replace.
                fu = spec[4] if len(spec) > 4 else max(8.0, 0.08 * (y1 - y0))
                fd = spec[5] if len(spec) > 5 else max(10.0, 0.3 * (y1 - y0))
                band = (np.clip((yy - (y0 - fu)) / fu, 0.0, 1.0) *
                        np.clip(((y1 + fd) - yy) / fd, 0.0, 1.0))
                ramp = np.clip((d - d0) / max(d1 - d0, 1.0),
                               0.0, 1.0)[None, :]
                a_band = strength * (1.0 - band * (1.0 - ramp))
                alpha = a_band if alpha is None else np.minimum(alpha,
                                                                a_band)
            # Keep the near-axis taper in the gaps between bands — without
            # it the mirror splices at full strength right at the axis
            # (visible vertical seam at eye level).
            alpha = np.minimum(alpha, base_alpha[None, :])
            alpha3 = alpha.astype(np.float32)[:, :, None]
        else:
            alpha3 = base_alpha.astype(np.float32)[None, :, None]

        region = result[:, target].astype(np.float32)
        blended = patch.astype(np.float32) * alpha3 + region * (1.0 - alpha3)
        result[:, target] = np.clip(blended, 0, 255).astype(np.uint8)
        return result

    def _ear_mirror_pass(self, img: np.ndarray, axis: float, side: str,
                         y1: float, eye_d: float,
                         strength: float) -> np.ndarray:
        """Mirror the head-top (ears + forehead) about the CRANIUM axis.

        In a 3/4 animal view the muzzle midline (the main mirror axis)
        sits far from the cranium midline: the visible ear's mirror
        about the muzzle axis lands off-canvas, and the original
        side-view near ear survives the forehead keep-band as a ghost.
        Cat/dog ears stick up above the head, so BOTH are visible in
        the source — mirroring the visible ear about the cranium axis
        (estimated from the two hairline points) lands it right where
        the near ear is, replacing it with a proper symmetric twin.
        A thin original strip around the axis is kept (mirroring the
        axis-straddling pale fur would paint the forehead "diamond").
        """
        h, w = img.shape[:2]
        ax = int(np.clip(round(axis), 1, w - 2))
        xs = np.arange(w, dtype=np.float32)
        src_x = 2 * ax - xs
        fill = (xs < ax) if side == "right" else (xs > ax)
        # Soft cut where the mirrored source runs off the canvas edge —
        # a boolean mask leaves a vertical seam in the background.
        src_ok = (np.clip((w - 1 - src_x) / 50.0, 0.0, 1.0) *
                  np.clip(src_x / 50.0, 0.0, 1.0))
        d = np.abs(xs - ax)
        a = np.clip((d - 0.25 * eye_d) / (0.35 * eye_d), 0.0, 1.0)
        a = a * fill * src_ok * strength
        yy = np.arange(h, dtype=np.float32)
        fd = max(10.0, 0.15 * eye_d)
        row = np.clip((y1 + fd - yy) / fd, 0.0, 1.0)
        alpha = (a[None, :] * row[:, None])[..., None]
        if not alpha.any():
            return img
        sx = np.clip(src_x, 0, w - 1).astype(np.int64)
        mirrored = img[:, sx]
        out = (img.astype(np.float32) * (1.0 - alpha) +
               mirrored.astype(np.float32) * alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _fit_reference(self, ref_img: np.ndarray, ref_kps: dict,
                       target_kps: dict, shape: tuple,
                       offset: tuple = (0.0, 0.0)):
        """Fit a pre-generated frontal REFERENCE onto a frontal template.

        The fit target must be a FRONTAL keypoint set (the classical
        mirrored/anatomical template), never the profile keypoints — a
        similarity fit ref->profile is geometrically inconsistent and
        RANSAC latches onto a degenerate subset. The fit is a
        LEAST-SQUARES SIMILARITY over the face-INTERIOR points only
        (brow/eye/nose/mouth/chin), not RANSAC over all points and not
        an exact TPS: a generated face has different proportions than
        the anatomical template (a cat's eyes sit nearly twice as far
        out as the template's 0.42*ear-span fraction, and its muzzle is
        half as tall). With all points, RANSAC discards the conflicting
        majority and vortexes the downstream warp; an exact TPS honors
        every keypoint but shears the interior texture into smear
        BETWEEN them. The interior-only similarity keeps the reference
        pixels crisp and undistorted where the transplant zones
        actually sample; skull points (ears/hairline) are pinned in the
        original and never transplanted, so their mismatch is harmless.
        Returns (refit_keypoints, ref_aligned_image) or (None, None).
        """
        ordered = [k for k in DEFAULT_KEYPOINTS
                   if k in ref_kps and k in target_kps]
        if len(ordered) < 6:
            return None, None
        interior = [k for k in ordered
                    if k not in ("ear_top", "ear_bottom",
                                 "hairline_left", "hairline_center",
                                 "hairline_right")]
        fit_names = interior if len(interior) >= 4 else ordered
        fit_src = np.array([ref_kps[k] for k in fit_names], np.float64)
        fit_dst = np.array([target_kps[k] for k in fit_names], np.float64)
        fit_dst = fit_dst + np.array(offset, dtype=np.float64)

        # Umeyama least-squares similarity (all points, no subsetting).
        mu_s, mu_d = fit_src.mean(0), fit_dst.mean(0)
        cs, cd = fit_src - mu_s, fit_dst - mu_d
        var_s = (cs ** 2).sum() / len(cs)
        if var_s < 1e-6:
            return None, None
        U, S, Vt = np.linalg.svd(cd.T @ cs / len(cs))
        d = np.sign(np.linalg.det(U @ Vt))
        Dm = np.diag([1.0, d])
        R = U @ Dm @ Vt
        scale = float(np.trace(np.diag(S) @ Dm) / var_s)
        t = mu_d - scale * R @ mu_s
        M = np.hstack([scale * R, t[:, None]])

        h, w = shape[:2]
        # Explicit remap with the backward map (canvas -> ref coords).
        # warpAffine's convention proved unreliable across OpenCV
        # versions (4.x samples dst->src, this 5.0 build maps src->dst),
        # so build the sampling grid by hand.
        Minv = cv2.invertAffineTransform(M)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = (Minv[0, 0] * xs + Minv[0, 1] * ys + Minv[0, 2]
                 ).astype(np.float32)
        map_y = (Minv[1, 0] * xs + Minv[1, 1] * ys + Minv[1, 2]
                 ).astype(np.float32)
        ref_aligned = cv2.remap(ref_img, map_x, map_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        ref_pts = np.array([ref_kps[k] for k in ordered], dtype=np.float64)
        refit = {k: (float(p[0]), float(p[1]))
                 for k, p in zip(ordered,
                                 cv2.transform(ref_pts[None], M)[0])}
        return refit, ref_aligned

    def _composite_reference(self, warped: np.ndarray,
                             ref_aligned: np.ndarray, zone: np.ndarray,
                             span: float, soft: bool = False,
                             tone_src: np.ndarray = None) -> np.ndarray:
        """Transplant reference pixels inside `zone` onto the warped
        original: Lab-match the reference to the original on the seam
        band (so the join is tone-invisible), then composite. Both
        layers are warped onto the SAME frontal geometry, so blending is
        safe anywhere — with soft=True the zone is used as the alpha
        directly (a wide sigmoid seam has no hard edge at all).
        `tone_src` overrides the image the Lab seam statistics are taken
        from (used when `warped` had its features erased and the seam
        ring should still be measured on real content)."""
        z = zone.astype(np.float32)
        if soft:
            band = ((z > 0.15) & (z < 0.85)).astype(np.uint8)
        else:
            z8 = (z > 0).astype(np.uint8)
            band = cv2.dilate(z8, np.ones((25, 25), np.uint8)) - z8
        ref = ref_aligned
        if band.sum() > 50:
            src_img = warped if tone_src is None else tone_src
            wl = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB).astype(np.float32)
            rl = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
            rm = band > 0
            for c in range(3):
                ws, rs = wl[..., c][rm], rl[..., c][rm]
                rl[..., c] = (rl[..., c] - rs.mean()) * \
                    (ws.std() / (rs.std() + 1e-6)) + ws.mean()
            ref = cv2.cvtColor(np.clip(rl, 0, 255).astype(np.uint8),
                               cv2.COLOR_LAB2BGR)
        if soft:
            alpha = z
        else:
            alpha = cv2.GaussianBlur(z, (0, 0), span * 0.04 + 2.0)
        alpha = (alpha * float(np.clip(self._mirror_strength, 0, 1))
                 )[..., None]
        out = (warped.astype(np.float32) * (1 - alpha) +
               ref.astype(np.float32) * alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _fill_neck_wedge(self, result: np.ndarray, cx: float,
                         frontal_kps: dict) -> np.ndarray:
        """Fill the blank background wedge between chin and collar.

        Mirroring the whole figure also mirrors the BACKGROUND that sat
        between the profile's neck and chest: a pure-white trapezoid
        under the chin, bounded by the chin line (top) and the two
        collar edges (sides). It reads as a hole. The wedge is found as
        the bright CONNECTED COMPONENT just below the chin (the dark
        chin/collar lines enclose it, so background and chest fabric
        outside are not touched) and diffuse-filled from the surrounding
        tones — chin shading gradients downward, collar lines shade the
        edges — turning it into a soft neck/upper-chest. Only bright
        pixels are filled, so sketch lines survive.
        """
        ct = frontal_kps.get("chin_tip")
        cl = frontal_kps.get("chin_left")
        cr = frontal_kps.get("chin_right")
        if ct is None or cl is None or cr is None:
            return result
        h, w = result.shape[:2]
        half_jaw = max(abs(cr[0] - cx), abs(cx - cl[0]), 8.0)
        lum = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bright = (lum > 214).astype(np.uint8)
        # Seed: first bright pixel scanning down the axis below the chin
        # (the chin underside shading band may sit right under the tip).
        seed_x = int(np.clip(round(cx), 0, w - 1))
        seed_y = None
        for y in range(int(round(ct[1])) + 4,
                       min(h, int(round(ct[1])) + 80)):
            if bright[y, seed_x]:
                seed_y = y
                break
        if seed_y is None:
            return result
        n, labels = cv2.connectedComponents(bright)
        wedge = labels == labels[seed_y, seed_x]
        # Safety bounds: stay near the axis and not too far below the
        # chin — a leak through a gap in the collar lines must not
        # repaint outside background or the chest.
        yy, xx = np.mgrid[0:h, 0:w]
        bounds = ((yy >= ct[1] - 14) &
                  (yy <= ct[1] + 3.0 * half_jaw) &
                  (np.abs(xx - cx) <= 2.6 * half_jaw))
        unknown = wedge & bounds
        if int(unknown.sum()) < 200:
            return result
        f = result.astype(np.float32)
        # Dark lines are weak sources so they shade the wedge edges
        # softly instead of bleeding black into it.
        src_w = np.clip((lum - 100.0) / 120.0, 0.15, 1.0)
        src_w = src_w.astype(np.float32)
        diffuse_fill(f, unknown, src_w, (2, 6, 15), dynamic=True)
        return np.clip(f, 0, 255).astype(np.uint8)

    def _narrow_figure(self, out: np.ndarray, frontal_kps: dict,
                       bg_color: tuple) -> np.ndarray:
        """Horizontally compress the symmetrized figure about the axis.

        A head's front-back DEPTH is much larger than its WIDTH, so a
        mirrored profile produces a figure ~1.4x too wide — the hair
        spans the whole canvas and the doubled shoulders read as two
        bodies. Compress x about the axis: head rows more (they carry
        the full depth excess), body rows less, with a linear ramp
        across the neck so no seam appears. The canvas is first padded
        with the BACKGROUND color — the figure touches the canvas
        edges, and replicating those edge pixels would smear dark
        streaks across the margins. The figure ends up proportioned
        like a real frontal portrait, centered with margins.
        """
        ct = frontal_kps.get("chin_tip")
        if ct is None:
            return out
        h, w = out.shape[:2]
        k_head, k_body = 0.60, 0.70
        y1 = float(ct[1])
        y2 = y1 + max(40.0, h * 0.10)
        pad = w // 2
        wide = cv2.copyMakeBorder(out, 0, 0, pad, pad,
                                  cv2.BORDER_CONSTANT, value=bg_color)
        cx = pad + w / 2.0
        yy = np.arange(h, dtype=np.float32)
        t = np.clip((yy - y1) / max(1.0, y2 - y1), 0.0, 1.0)
        s = k_head + (k_body - k_head) * t
        xs = np.arange(w + 2 * pad, dtype=np.float32)
        map_x = (cx + (xs[None, :] - cx) / s[:, None]).astype(np.float32)
        map_y = np.repeat(yy[:, None], w + 2 * pad, axis=1)
        wide = cv2.remap(wide, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=bg_color)
        return wide[:, pad:pad + w]

    def _cut_shoulder_silhouette(self, out: np.ndarray, frontal_kps: dict,
                                 bg_color: tuple) -> np.ndarray:
        """Cut the doubled shoulders back to ONE clean silhouette.

        The mirrored body keeps the profile's full front-to-back depth:
        below the collar a SECOND pair of shoulder humps (the hoodie's
        back shoulder + dark strap) sits outside the collar slope and
        reads as a second body behind the first. Everything outside a
        straight shoulder line — from just outside the collar at the
        chin row, sloping gently outward to the bottom — is background,
        not body: erase it to the background colour. The collar slope,
        zipper and drawstring rings all lie inside the line and stay.
        """
        ct = frontal_kps.get("chin_tip")
        if ct is None:
            return out
        h, w = out.shape[:2]
        cx = w / 2.0
        y0 = float(ct[1])
        half0, half1 = 0.27 * w, 0.35 * w
        yy, xx = np.mgrid[0:h, 0:w]
        yy = yy.astype(np.float32)
        xx = xx.astype(np.float32)
        half = half0 + np.clip((yy - y0) / max(1.0, h - y0), 0, 1) \
            * (half1 - half0)
        half = np.where(yy >= y0, half, np.float32(w))
        over = np.abs(xx - cx) - half  # >0 outside the silhouette
        alpha = np.clip(over / 4.0, 0.0, 1.0)[..., None]  # 4px feather
        bg = np.array(bg_color, dtype=np.float32)[None, None, :]
        res = out.astype(np.float32) * (1 - alpha) + bg * alpha
        return np.clip(res, 0, 255).astype(np.uint8)

    def _trim_side_hair(self, out: np.ndarray, final_kps: dict,
                        bg_color: tuple) -> np.ndarray:
        """Trim the mirrored side-hair curtains to a short frontal cut.

        Mirroring a profile hangs the BACK hair on both sides of the
        face as long curtains reaching the shoulders — it reads as
        twin-tails, not a real frontal short haircut. Frontal short
        hair ends around ear level; below that the jaw and neck show.
        Erase dark pixels outside the face/jaw ellipse in the band
        from just below the earlobe to the chin (background shows
        through). The ears themselves sit above the band and are safe.
        `final_kps` must be in the FINAL (cropped+narrowed) coords.
        """
        eb = final_kps.get("ear_bottom")
        ct = final_kps.get("chin_tip")
        if ct is None:
            return out
        h, w = out.shape[:2]
        cx = w / 2.0
        chin_y = float(ct[1])
        ear_y = float(eb[1]) if eb is not None else chin_y - 90.0
        ear_x = abs(eb[0] - cx) if eb is not None else 0.28 * w
        yy, xx = np.mgrid[0:h, 0:w]
        yy = yy.astype(np.float32)
        xx = xx.astype(np.float32)
        cy = (ear_y + chin_y) / 2.0
        ay = max(20.0, (chin_y - ear_y) * 0.62)
        ax = max(20.0, ear_x * 1.02)
        inside = (((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2) < 1.0
        band = (yy >= ear_y + 15.0) & (yy <= chin_y + 5.0)
        lum = out.astype(np.float32).mean(axis=2) / 255.0
        trim = band & (~inside) & (lum < 0.78)
        if trim.sum() < 50:
            return out
        alpha = cv2.GaussianBlur(
            trim.astype(np.float32), (0, 0), 3.0)[..., None]
        bg = np.array(bg_color, dtype=np.float32)[None, None, :]
        res = out.astype(np.float32) * (1 - alpha) + bg * alpha
        return np.clip(res, 0, 255).astype(np.uint8)

    def _close_chest_gap(self, out: np.ndarray, final_kps: dict,
                         bg_color: tuple) -> np.ndarray:
        """Make the mirrored chest read as ONE closed zip-up hoodie.

        Mirroring a profile leaves (a) a blank background wedge between
        the two hoodie fronts and (b) TWO front-opening plackets, each
        with its own zipper pull, running diagonally down the chest —
        the "two necks / two bodies" reading. A real frontal zip
        hoodie is CONTINUOUS: find where the collar closes at the
        center, fill the wedge with per-row jacket tone, erase the
        duplicate placket curves/pulls below the collar, then draw the
        single center zipper (starting at the collar close — a zipper
        drawn higher up cuts a line through the NECK, which is what
        made the result read as two necks).
        `final_kps` must be in the FINAL (cropped+narrowed) coords.
        """
        ct = final_kps.get("chin_tip")
        if ct is None:
            return out
        h, w = out.shape[:2]
        cx = w / 2.0
        cxi = int(cx)
        y_top = int(ct[1]) + 8
        if y_top >= h - 20:
            return out
        f = out.astype(np.float32)
        lum = f.mean(axis=2) / 255.0
        yy = np.arange(h)

        # 1) Zipper start: just below the collar dip. (The mirrored
        #    open front never actually closes at the centre, so it
        #    cannot be detected from centre-strip darkness — anchor a
        #    fixed fraction below the chin instead.)
        y_meet = y_top + int(0.055 * h)

        # 2) Per-row cloth tone: median of the row's BRIGHT jacket
        #    pixels sampled across the chest (zipper/print/shading
        #    would grey it), floored near the background tone. Used
        #    both to fill the wedge and to paint over erased plackets.
        tone = np.zeros((h, 3), np.float32)
        for y in range(y_top, h):
            l0 = int(max(0, cx - 0.30 * w))
            r1 = int(min(w, cx + 0.30 * w))
            samp = f[y, l0:r1].reshape(-1, 3)
            sl = samp.mean(axis=1) / 255.0
            bright_s = samp[sl > 0.80]
            if len(bright_s) >= 8:
                tone[y] = np.median(bright_s, axis=0)
            elif len(samp[sl > 0.55]):
                tone[y] = np.percentile(samp[sl > 0.55], 80, axis=0)
        good = tone.sum(axis=1) > 0
        if good.sum() < 10:
            return out
        tone = np.maximum(
            tone, (np.array(bg_color, np.float32) * 0.93)[None, :])
        for c in range(3):  # fill gaps + smooth vertically
            idx = np.where(good)[0]
            tone[:, c] = np.interp(yy, idx, tone[idx, c])
        tone = cv2.GaussianBlur(tone[:, None, :], (1, 31), 0)[:, 0, :]
        tone3 = np.repeat(tone[:, None, :], w, axis=1)

        # 3) Fill the bright background wedge between the fronts.
        top_half = 0.072 * w
        bot_half = 0.042 * w
        poly = np.array([
            [cx - top_half, y_top], [cx + top_half, y_top],
            [cx + bot_half, h], [cx - bot_half, h]], np.int32)
        zone = np.zeros((h, w), np.uint8)
        cv2.fillPoly(zone, [poly], 255)
        bright = ((zone > 0) & (lum > 0.70)).astype(np.float32)
        alpha = cv2.GaussianBlur(bright, (0, 0), 3.0)[..., None]
        res = f * (1 - alpha) + tone3 * alpha

        # 4) Tame the open-front V below the collar WITHOUT stripping
        #    the chest texture: trace each placket as the DARKEST
        #    SMOOTH vertical path through its half of the chest band
        #    (dynamic programming — short fold strokes and the pull
        #    rings cannot out-score a full-length path), FADE it to a
        #    faint seam (a real zip hoodie's front seams flank the
        #    zipper), and fully erase only the small dark blobs (the
        #    duplicated drawstring rings) hanging inside the V.
        band_half = int(0.22 * w)

        def _trace(x0, x1):
            sub = lum[y_meet:, x0:x1].astype(np.float32)
            cost = sub ** 2  # dark = cheap
            H, W = cost.shape
            if H < 10 or W < 5:
                return None
            dp = cost.copy()
            par = np.zeros((H, W), np.int8)
            BIG = np.float32(1e9)
            for y in range(1, H):
                prev = dp[y - 1]
                cand = np.full((5, W), BIG, np.float32)
                for k, dx in enumerate(range(-2, 3)):
                    s0, s1 = max(0, -dx), min(W, W - dx)
                    cand[k, s0:s1] = prev[s0 + dx:s1 + dx]
                best = np.argmin(cand, axis=0)
                dp[y] = cost[y] + cand[best, np.arange(W)]
                par[y] = best.astype(np.int8) - 2
            path = np.zeros(H, np.int32)
            path[-1] = int(np.argmin(dp[-1]))
            for y in range(H - 1, 0, -1):
                path[y - 1] = path[y] + int(par[y, path[y]])
            return path + x0

        path_l = _trace(cxi - band_half, cxi - 9)
        path_r = _trace(cxi + 9, cxi + band_half)
        fade = np.zeros((h, w), np.float32)
        if path_l is not None and path_r is not None:
            for i, y in enumerate(range(y_meet, h)):
                for p in (path_l[i], path_r[i]):
                    fade[y, max(0, p - 4):min(w, p + 5)] = 1.0
        # small dark blobs (drawstring rings) inside the V: full erase
        blobs = np.zeros((h, w), np.uint8)
        blobs[y_meet:, cxi - band_half:cxi + band_half] = (
            lum[y_meet:, cxi - band_half:cxi + band_half] < 0.45)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(blobs, 8)
        for i in range(1, n):
            if 12 <= stats[i, cv2.CC_STAT_AREA] <= 150:
                blobs[lab == i] = 2
        # inner V strip: anything dark still hanging between the faded
        # plackets (ring stubs, cord ends) goes completely — a closed
        # zip chest is plain between its front seams
        inner = np.zeros((h, w), np.float32)
        x_in = int(0.10 * w)
        inner[y_meet + 20:, cxi - x_in:cxi + x_in + 1] = (
            lum[y_meet + 20:, cxi - x_in:cxi + x_in + 1] < 0.58)
        inner[:, cxi - 8:cxi + 9] = 0.0  # keep the zipper column
        strong = np.clip((blobs == 2).astype(np.float32) + inner, 0, 1)
        strong = cv2.dilate(strong, np.ones((3, 3), np.float32))
        fade = np.maximum(fade, strong)
        ramp = np.clip((yy - (y_meet - 10)) / 18.0, 0, 1).astype(np.float32)
        fade *= ramp[:, None]
        fade[:y_top] = 0.0
        fade = cv2.GaussianBlur(fade, (0, 0), 2.0)
        strength = 0.70 + 0.30 * cv2.GaussianBlur(strong, (0, 0), 2.0)
        alpha = np.maximum(fade * strength,
                           cv2.GaussianBlur(strong, (0, 0), 2.0)
                           * 0.98)[..., None]
        res = res * (1 - alpha) + tone3 * alpha
        res = np.clip(res, 0, 255)

        # 5) ONE centre zipper from the collar close down, with ONE
        #    small ring pull like the original garment's, blended
        #    gently to keep the pencil feel.
        zip_layer = res.copy()
        z0 = y_meet + 2
        cv2.line(zip_layer, (cxi, z0), (cxi, h - 1),
                 (70, 70, 70), 2, cv2.LINE_AA)
        cv2.line(zip_layer, (cxi - 3, z0), (cxi - 3, h - 1),
                 (150, 150, 150), 1, cv2.LINE_AA)
        cv2.line(zip_layer, (cxi + 3, z0), (cxi + 3, h - 1),
                 (150, 150, 150), 1, cv2.LINE_AA)
        y_pull = z0 + int(0.30 * (h - z0))
        cv2.circle(zip_layer, (cxi + 5, y_pull), 4,
                   (90, 90, 90), 2, cv2.LINE_AA)
        zm = np.zeros((h, w), np.float32)
        zm[z0:, cxi - 4:cxi + 5] = 1.0
        zm[y_pull - 7:y_pull + 7, cxi:cxi + 11] = 1.0
        zm = cv2.GaussianBlur(zm, (0, 0), 1.2)[..., None] * 0.55
        res = res * (1 - zm) + zip_layer * zm
        return np.clip(res, 0, 255).astype(np.uint8)

    def _transplant_nose_strokes(self, out: np.ndarray, final_kps: dict,
                                 ref_img: np.ndarray,
                                 ref_kps: dict) -> np.ndarray:
        """Transplant ONLY the nose strokes from a generated frontal
        reference onto the classical result.

        A pure profile genuinely lacks a frontal nose — that is the one
        feature the reference must supply. Everything identity-critical
        (eyes, brows, hair, ears, clothes, background) stays
        original-pixel, so the result cannot drift from the source
        drawing the way a full generated face does.

        Stroke-only transfer: pencil style is dark strokes on light
        paper, so we take only the amount the reference is DARKER than
        its local surroundings (high-pass) inside a tight nose zone and
        subtract it from the base. No tone patches, no seams, no color
        — the transferred content is pure linework.

        The zone is drawn in REFERENCE space around the annotated nose
        and warped along with the strokes: a least-squares fit cannot
        make the reference's proportions match the base exactly, so a
        target-space zone would clip the reference's eyes/fringe
        hanging into the nose area. A ref-space zone only ever
        contains the reference's own nose.
        `final_kps` must be in the FINAL (cropped+narrowed) coords.
        """
        refit, _ = self._fit_reference(
            ref_img, ref_kps, final_kps, out.shape, (0.0, 0.0))
        if refit is None:
            return out
        h, w = out.shape[:2]
        rh, rw = ref_img.shape[:2]

        def _mean_y(kps, names):
            v = [kps[n][1] for n in names if n in kps]
            return float(np.mean(v)) if v else None

        eye_y_r = _mean_y(ref_kps, ("eye_left_inner", "eye_right_inner",
                                    "eye_left_outer", "eye_right_outer"))
        mouth_y_r = _mean_y(ref_kps, ("mouth_left", "mouth_right",
                                      "mouth_center"))
        nose_r = ref_kps.get("nose_tip")
        eye_y = _mean_y(final_kps, ("eye_left_inner", "eye_right_inner",
                                    "eye_left_outer", "eye_right_outer"))
        mouth_y = _mean_y(final_kps, ("mouth_left", "mouth_right",
                                      "mouth_center"))
        if (eye_y_r is None or mouth_y_r is None or nose_r is None
                or eye_y is None or mouth_y is None):
            return out
        fh_r = mouth_y_r - eye_y_r
        if fh_r < 15:
            return out

        # Nose zone in REFERENCE space. The top edge must start BELOW
        # the fringe tips hanging between the eyes (~0.30 face-height
        # under the eye line) or hair strokes get transplanted onto the
        # nose bridge; the bottom stops just above the mouth.
        y_t = eye_y_r + fh_r * 0.30
        y_b = mouth_y_r - fh_r * 0.08
        cy_r = (y_t + y_b) / 2.0
        ry = (y_b - y_t) / 2.0
        rx = max(15.0, fh_r * 0.28)
        yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
        zone = (((xx - nose_r[0]) / rx) ** 2
                + ((yy - cy_r) / ry) ** 2 < 1.0).astype(np.float32)
        zone = cv2.GaussianBlur(zone, (0, 0), 2.5)

        # High-pass stroke strength: how much darker each reference
        # pixel is than its local neighbourhood (tone-independent).
        ref_g = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        loc = cv2.GaussianBlur(ref_g, (0, 0), 6.0)
        hp = np.clip(loc - ref_g, 0, None)
        hp[hp < 6] = 0  # ignore paper grain
        hp *= zone

        # Warp the masked stroke map with the same similarity the fit
        # used (ref -> final canvas), via an explicit backward remap.
        ordered = [k for k in DEFAULT_KEYPOINTS
                   if k in ref_kps and k in refit
                   and k not in ("ear_top", "ear_bottom",
                                 "hairline_left", "hairline_center",
                                 "hairline_right")]
        if len(ordered) < 4:
            return out
        src = np.array([ref_kps[k] for k in ordered], np.float32)
        dst = np.array([refit[k] for k in ordered], np.float32)
        est = getattr(cv2, "estimateAffinePartial2D", None) or \
            getattr(cv2, "estimateAffinePartial", None)
        if est is None:
            return out
        A, _ = est(src, dst)
        if A is None:
            return out
        Ainv = cv2.invertAffineTransform(A)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = (Ainv[0, 0] * xs + Ainv[0, 1] * ys
                 + Ainv[0, 2]).astype(np.float32)
        map_y = (Ainv[1, 0] * xs + Ainv[1, 1] * ys
                 + Ainv[1, 2]).astype(np.float32)
        hp_t = cv2.remap(hp, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Auto-center: the drawn nose strokes may sit slightly off the
        # face axis (annotation/placement offset) — nudge the stroke
        # map so its centroid in the nose band lands on the base axis.
        cx = w / 2.0
        face_h = mouth_y - eye_y
        y_a = int(eye_y + face_h * 0.10)
        y_b = int(mouth_y - 4)
        x_a = int(cx - 0.15 * w)
        x_b = int(cx + 0.15 * w)
        band = hp_t[y_a:y_b, x_a:x_b]
        colsum = band.sum(0)
        tot = float(colsum.sum())
        if tot > 0:
            stroke_cx = float((colsum * np.arange(
                x_a, x_b, dtype=np.float32)).sum() / tot)
            dx = float(np.clip(cx - stroke_cx, -15, 15))
            if abs(dx) > 2:
                Sh = np.float32([[1, 0, dx], [0, 1, 0]])
                hp_t = cv2.warpAffine(hp_t, Sh, (w, h))

        out_f = out.astype(np.float32) - (hp_t * 1.4)[..., None]
        return np.clip(out_f, 0, 255).astype(np.uint8)

    def _transplant_face_interior(self, out: np.ndarray,
                                  final_kps: dict,
                                  ref_img: np.ndarray,
                                  ref_kps: dict) -> np.ndarray:
        """Transplant the reference's EYES + NOSE + MOUTH linework
        onto the classical result, replacing the washed-out mirrored
        interior.

        A mirrored profile half-face leaves the interior nearly blank
        (tiny side-view eye, ghost nose). The generated reference owns
        a full frontal interior in a matching pencil style, so:
          1) the composite's interior zone is erased to paper tone
             (heavy local blur keeps the soft cheek shading, drops the
             faint mirrored strokes);
          2) the reference's high-pass strokes inside the SAME zone
             (drawn in reference space, so only its own features are
             captured — the fringe hanging between the brows is cut by
             the zone's top edge) are warped over with the similarity
             fit and subtracted.
        Brows, hair, ears, jaw contour, clothes and background stay
        original-pixel. `final_kps` in FINAL canvas coords.
        """
        refit, _ = self._fit_reference(
            ref_img, ref_kps, final_kps, out.shape, (0.0, 0.0))
        if refit is None:
            return out
        h, w = out.shape[:2]
        rh, rw = ref_img.shape[:2]

        def _mean_y(kps, names):
            v = [kps[n][1] for n in names if n in kps]
            return float(np.mean(v)) if v else None

        eye_names = ("eye_left_inner", "eye_right_inner",
                     "eye_left_outer", "eye_right_outer")
        mouth_names = ("mouth_left", "mouth_right", "mouth_center")
        eye_y_r = _mean_y(ref_kps, eye_names)
        mouth_y_r = _mean_y(ref_kps, mouth_names)
        nose_r = ref_kps.get("nose_tip")
        if eye_y_r is None or mouth_y_r is None or nose_r is None:
            return out
        fh_r = mouth_y_r - eye_y_r
        if fh_r < 15:
            return out

        # Zone in REFERENCE space as THREE tight sub-zones — one
        # ellipse per eye plus a nose/mouth column. A single big
        # ellipse also captures the fringe tips hanging between and
        # beside the eyes (long hair strokes got transplanted onto the
        # face in the first attempt).
        yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
        zone = np.zeros((rh, rw), np.float32)
        for side in ("left", "right"):
            ei = ref_kps.get(f"eye_{side}_inner")
            eo = ref_kps.get(f"eye_{side}_outer")
            if ei is None or eo is None:
                continue
            ecx, ecy = (ei[0] + eo[0]) / 2.0, (ei[1] + eo[1]) / 2.0
            rx_e = max(10.0, abs(eo[0] - ei[0]) * 0.90)
            ry_e = fh_r * 0.26
            z = (((xx - ecx) / rx_e) ** 2
                 + ((yy - ecy) / ry_e) ** 2 < 1.0)
            zone = np.maximum(zone, z.astype(np.float32))
        y_t = eye_y_r + fh_r * 0.26
        y_b = mouth_y_r + fh_r * 0.22
        cy_r = (y_t + y_b) / 2.0
        ry = (y_b - y_t) / 2.0
        rx = fh_r * 0.38
        z = (((xx - nose_r[0]) / rx) ** 2
             + ((yy - cy_r) / ry) ** 2 < 1.0)
        zone = np.maximum(zone, z.astype(np.float32))
        zone = cv2.GaussianBlur(zone, (0, 0), 6.0)

        # Reference strokes inside the zone (tone-independent
        # high-pass). Finer local window + lower threshold + higher
        # gain than the nose transplant: the eye linework is thin,
        # and the first version's faint strokes read as 'blurry,
        # not transplanted' (user feedback).
        ref_g = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(
            np.float32)
        loc = cv2.GaussianBlur(ref_g, (0, 0), 4.0)
        hp = np.clip(loc - ref_g, 0, None)
        hp[hp < 3] = 0
        hp *= zone

        # Warp zone + strokes with the fit similarity (ref -> final).
        ordered = [k for k in DEFAULT_KEYPOINTS
                   if k in ref_kps and k in refit
                   and k not in ("ear_top", "ear_bottom",
                                 "hairline_left", "hairline_center",
                                 "hairline_right")]
        if len(ordered) < 4:
            return out
        src = np.array([ref_kps[k] for k in ordered], np.float32)
        dst = np.array([refit[k] for k in ordered], np.float32)
        est = getattr(cv2, "estimateAffinePartial2D", None) or \
            getattr(cv2, "estimateAffinePartial", None)
        if est is None:
            return out
        A, _ = est(src, dst)
        if A is None:
            return out
        Ainv = cv2.invertAffineTransform(A)
        ys, xs2 = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = (Ainv[0, 0] * xs2 + Ainv[0, 1] * ys
                 + Ainv[0, 2]).astype(np.float32)
        map_y = (Ainv[1, 0] * xs2 + Ainv[1, 1] * ys
                 + Ainv[1, 2]).astype(np.float32)
        hp_t = cv2.remap(hp, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        zone_t = cv2.remap(zone, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
        zone_t = np.clip(zone_t, 0, 1)[..., None]

        # 1) erase the composite's own interior strokes by INPAINTING
        # only the dark stroke pixels inside the zone — keeps the
        # paper grain. The earlier blur-to-'paper' erase flattened
        # the zone texture and made the transplant look blurry.
        zone_u = (zone_t[..., 0] if zone_t.ndim == 3
                  else zone_t)
        core = (zone_u > 0.35)
        dark = ((out.mean(axis=2) if out.ndim == 3 else out) < 185)
        mask = (dark & core).astype(np.uint8) * 255
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        src_u = out if out.ndim == 3 else out
        inpainted = cv2.inpaint(src_u, mask, 4, cv2.INPAINT_TELEA)
        erased = out.astype(np.float32) * (1 - zone_t) \
            + inpainted.astype(np.float32) * zone_t
        # 2) lay the reference's strokes down
        out_f = erased - (hp_t * 1.9)[..., None]
        return np.clip(out_f, 0, 255).astype(np.uint8)

    def _smooth_seam(self, image: np.ndarray, cx: float,
                     mask: np.ndarray) -> np.ndarray:
        """Hide the mirror seam: blend a narrow band around the axis with
        its bilateral/Gaussian smoothed version, inside the hull only."""
        h, w = image.shape[:2]
        cx_i = int(np.clip(round(cx), 0, w - 1))
        half = max(8, w // 60)
        x0, x1 = max(0, cx_i - half), min(w, cx_i + half)
        band = np.zeros((h, w), dtype=np.float32)
        band[:, x0:x1] = 1.0
        band = cv2.GaussianBlur(band, (0, 0), half * 0.6) * mask
        smooth = cv2.bilateralFilter(image, 7, 40, 40)
        b3 = (band * 0.8)[..., None]
        out = (image.astype(np.float32) * (1 - b3) +
               smooth.astype(np.float32) * b3)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _match_color_lab(self, result: np.ndarray, cx: float, side: str,
                         mask: np.ndarray) -> np.ndarray:
        """Reinhard transfer in Lab: match the mirrored half's statistics
        to the visible half, computed on hull pixels only (never the
        background, which used to poison global histogram matching)."""
        h, w = result.shape[:2]
        cx_i = int(np.clip(round(cx), 1, w - 2))
        xs = np.arange(w)[None, :]
        if side == "right":
            ref_m = (mask > 0.5) & (xs >= cx_i)
            tgt_m = (mask > 0.5) & (xs < cx_i)
        else:
            ref_m = (mask > 0.5) & (xs < cx_i)
            tgt_m = (mask > 0.5) & (xs >= cx_i)
        if ref_m.sum() < 200 or tgt_m.sum() < 200:
            return result

        lab = cv2.cvtColor(result, cv2.COLOR_BGR2Lab).astype(np.float32)
        out = lab.copy()
        for ch in range(3):
            ref = lab[:, :, ch][ref_m]
            tgt = lab[:, :, ch][tgt_m]
            mr, sr = ref.mean(), ref.std() + 1e-3
            mt, st = tgt.mean(), tgt.std() + 1e-3
            out[:, :, ch][tgt_m] = (tgt - mt) * (sr / st) + mr
        out = np.clip(out, 0, 255).astype(np.uint8)
        matched = cv2.cvtColor(out, cv2.COLOR_Lab2BGR)

        a3 = (mask * 0.7)[..., None]
        blend = (result.astype(np.float32) * (1 - a3) +
                 matched.astype(np.float32) * a3)
        return np.clip(blend, 0, 255).astype(np.uint8)

    # Anatomical frontal template for the 3/4 rotation path, as signed
    # fractions of the ear half-span about the skull midline (positive =
    # visible side). Zero = midline features swing to the center.
    _ROT_FRAC = {
        "nose_tip": 0.0, "mouth_center": 0.0, "chin_tip": 0.0,
        "eye_left_inner": 0.15, "eye_left_outer": 0.42,
        "brow_left_inner": 0.12, "brow_left_outer": 0.40,
        "mouth_left": -0.18, "mouth_right": 0.18,
        "chin_left": -0.25, "chin_right": 0.25,
    }
    # Skull points stay exactly where they are — the ears and the top of
    # the head must not move.
    _ROT_PIN = ("ear_top", "ear_bottom",
                "hairline_left", "hairline_center", "hairline_right")

    def _rotation_frontal(self, image, keypoints, side, cx, t0,
                          reference=None, reference_kps=None,
                          reference_offset=(0.0, 0.0)):
        """Turn a 3/4-view head to frontal with a rotation-style warp.

        The TPS template is ANATOMICAL (frontal angles about the skull
        midline), not mirrored: midline features (nose/mouth/chin) swing
        to the skull center, the visible eye/brow un-foreshorten to
        their frontal offsets, ears and hairline are pinned. The warp is
        a smooth bijection anchored at the image borders, so the
        foreshortened cheek/whiskers simply STRETCH into place — every
        output pixel still comes from the original photo. The only
        content that cannot be recovered by stretching is the occluded
        far eye, synthesized by copying the visible eye (scaled by
        mirror_strength). tps_flexibility blends the rotation amount.

        With a pre-generated frontal REFERENCE registered, its keypoints
        (fitted onto the anatomical template) replace the template, and
        its pixels replace the two zones stretching cannot fix: the
        dragged-muzzle smear and the occluded far eye.
        """
        h, w = image.shape[:2]
        hl = keypoints.get("hairline_left")
        hr = keypoints.get("hairline_right")
        c_f = ((hl[0] + hr[0]) / 2.0
               if hl is not None and hr is not None else cx)
        r_e = 0.0
        for n in ("ear_top", "ear_bottom"):
            p = keypoints.get(n)
            if p is not None:
                r_e = max(r_e, abs(p[0] - c_f))
        if r_e < 1.0:
            r_e = (abs(hr[0] - hl[0]) / 2.0
                   if hl is not None and hr is not None else 100.0)
        sign = 1.0 if side == "right" else -1.0

        ordered = [k for k in DEFAULT_KEYPOINTS if k in keypoints]
        src_pts = np.array([keypoints[k] for k in ordered],
                           dtype=np.float64)
        dst_pts = src_pts.copy()
        for i, k in enumerate(ordered):
            if k in self._ROT_PIN or k not in self._ROT_FRAC:
                continue  # pinned: ears / skull top never move
            x, y = keypoints[k]
            dst_pts[i] = (c_f + sign * self._ROT_FRAC[k] * r_e, y)

        # Reference path: fit the pre-generated frontal reference onto
        # the anatomical template and hand off — the original is NOT
        # TPS-warped at all (its skull is already near-frontal; warping
        # only vortexes the fur).
        if reference is not None and reference_kps:
            tmpl = {k: (float(dst_pts[i][0]), float(dst_pts[i][1]))
                    for i, k in enumerate(ordered)}
            refit, ref_aligned = self._fit_reference(
                reference, reference_kps, tmpl, image.shape,
                reference_offset)
            if refit is not None:
                return self._composite_rotation_reference(
                    image, keypoints, refit, ref_aligned, c_f, sign,
                    side, t0)

        f = float(np.clip(self._tps_flexibility, 0.0, 1.0))
        dst_eff = src_pts + f * (dst_pts - src_pts)
        warped = self._tps_warp(image, src_pts, dst_eff)
        t1 = time.time()

        # The one thing stretching cannot recover: the occluded far eye.
        # Copy the visible eye (already at its frontal position in the
        # warped image) to the mirrored spot about the skull midline.
        if (self._mirror_strength > 0
                and "eye_left_inner" in ordered
                and "eye_left_outer" in ordered):
            ei_t = dst_eff[ordered.index("eye_left_inner")]
            eo_t = dst_eff[ordered.index("eye_left_outer")]
            ecx = (ei_t[0] + eo_t[0]) / 2.0
            ecy = (ei_t[1] + eo_t[1]) / 2.0
            span = max(abs(eo_t[0] - ei_t[0]), 8.0)
            tcx = 2.0 * c_f - ecx
            ax, ay = span * 0.88, span * 0.72
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            e = ((xx - tcx) / ax) ** 2 + ((yy - ecy) / ay) ** 2
            em = np.clip((1.0 - e) / 0.22, 0.0, 1.0)
            em = cv2.GaussianBlur(em, (0, 0), span * 0.10)
            alpha = (em * self._mirror_strength)[..., None]
            mxx = np.clip(np.rint(2.0 * c_f - np.arange(w)),
                          0, w - 1).astype(np.int64)
            flipped = warped[:, mxx]
            out = (warped.astype(np.float32) * (1 - alpha) +
                   flipped.astype(np.float32) * alpha)
            warped = np.clip(out, 0, 255).astype(np.uint8)
        t2 = time.time()

        elapsed = (time.time() - t0) * 1000
        info = (f"旋转正面化: {(t1-t0)*1000:.0f}ms | "
                f"远眼补全: {(t2-t1)*1000:.0f}ms | 镜像侧: {side}")
        return FrontalizationResult(warped, elapsed, info)

    def _composite_rotation_reference(self, image, keypoints, refit,
                                      ref_aligned, c_f, sign, side, t0):
        """Reference-guided frontalization for a 3/4-view head whose
        SKULL is already near-frontal (both ears, forehead, cheeks,
        background are real and correctly placed). The original stays
        pixel-exact — warping it only smears fur into vortices. Three
        steps: erase the leftward-pointing features (muzzle, near
        eye/brow, and the half-hidden far eye peeking behind the nose
        bridge) by diffusion; then transplant the reference's frontal
        features (fitted to the anatomical template) over the erased
        canvas; ears, skull outline and background never change.
        tps_flexibility blends the result with the untouched original
        (0 = identity)."""
        h, w = image.shape[:2]
        interior = ("brow_left_inner", "brow_left_outer",
                    "eye_left_inner", "eye_left_outer", "nose_tip",
                    "mouth_left", "mouth_center", "mouth_right",
                    "chin_left", "chin_right", "chin_tip")
        hp = np.array(list(refit.values()), np.float32)
        span = float(np.ptp(hp[:, 0]) + 1e-6)

        # 1. Erase the original features that would double the donated
        #    frontal ones: hull of the annotated interior, plus an
        #    ellipse over the half-visible far eye (it sits just behind
        #    the nose bridge, outside the annotated hull).
        erase = np.zeros((h, w), np.uint8)
        src_in = np.array([keypoints[n] for n in interior
                           if n in keypoints], np.int32)
        if len(src_in) >= 3:
            cv2.fillConvexPoly(erase, cv2.convexHull(src_in), 1)
        if all(n in keypoints for n in
               ("eye_left_inner", "eye_left_outer", "nose_tip")):
            ei, eo = keypoints["eye_left_inner"], keypoints["eye_left_outer"]
            nt = keypoints["nose_tip"]
            ecx, ecy = (ei[0] + eo[0]) / 2.0, (ei[1] + eo[1]) / 2.0
            d = float(np.hypot(ecx - nt[0], ecy - nt[1])) + 1e-6
            fx = nt[0] + sign * 0.10 * d
            fy = ecy + 0.15 * d
            cv2.ellipse(erase, (int(round(fx)), int(round(fy))),
                        (max(int(0.28 * d), 8), max(int(0.22 * d), 8)),
                        0, 0, 360, 1, -1)
        k = max(9, int(span * 0.10) | 1)
        erase = cv2.dilate(erase, np.ones((k, k), np.uint8))

        t1 = time.time()
        canvas = image.copy()
        if erase.any():
            # The erased feature region is huge (~500px across) and
            # fast_blur's per-pass support tops out ~90px — even at 1/4
            # scale the hole center sees no sources (den>0.3 never met)
            # and stays original. Scale down so the hole fits inside one
            # blur box (~60px), fill, paste back. The fill is smooth by
            # design, nothing is lost.
            eys, exs = np.where(erase > 0)
            hole = max(int(np.ptp(eys)), int(np.ptp(exs))) + 1
            ds = max(1, int(np.ceil(hole / 60.0)))
            swd, shd = max(w // ds, 1), max(h // ds, 1)
            small = cv2.resize(canvas, (swd, shd),
                               interpolation=cv2.INTER_AREA
                               ).astype(np.float32)
            um = cv2.resize(erase, (swd, shd),
                            interpolation=cv2.INTER_NEAREST) > 0
            lum = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            # Floor 0.5, not 0.15: the muzzle is ringed by DARK tabby
            # fur, and at 0.15 the support-weighted blur denominator
            # never crosses diffuse_fill's den>0.3 acceptance threshold
            # — the whole hole stayed unfilled (original nose/eye kept
            # showing through). 0.5 still downweights dark line pixels.
            src_w = np.clip((lum.astype(np.float32) - 100.0) / 120.0,
                            0.5, 1.0)
            sw_s = cv2.resize(src_w, (swd, shd),
                              interpolation=cv2.INTER_AREA)
            diffuse_fill(small, um, sw_s, (2, 8, 30, 60), dynamic=True)
            up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            up = np.clip(up, 0, 255).astype(np.uint8)
            canvas[erase > 0] = up[erase > 0]
        t2 = time.time()

        # 2. Donate the reference's frontal features. The zone must be
        #    SYMMETRIC: the annotated points cover only the visible-side
        #    eye/brow, so the raw hull is a sliver that leaves the
        #    reference's far eye (which lands far left of it) outside.
        #    Mirror the fitted interior points about the fitted midline
        #    and hull the union. Also union the dilated erase mask — its
        #    diffusion fill is smooth gray and must not stay visible.
        face = np.array([refit[n] for n in interior if n in refit],
                        np.float64)
        zone = np.zeros((h, w), np.float32)
        if len(face) >= 3:
            mid_xs = [refit[n][0] for n in
                      ("nose_tip", "mouth_center", "chin_tip")
                      if n in refit]
            mid_x = float(np.mean(mid_xs)) if mid_xs else c_f
            mirrored = face.copy()
            mirrored[:, 0] = 2.0 * mid_x - mirrored[:, 0]
            both = np.vstack([face, mirrored]).astype(np.int32)
            m = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(m, cv2.convexHull(both), 1)
            k1 = max(9, int(span * 0.18) | 1)
            m = cv2.dilate(m, np.ones((k1, k1), np.uint8))
            k3 = max(9, int(span * 0.06) | 1)
            m = np.maximum(m, cv2.dilate(erase, np.ones((k3, k3),
                                                        np.uint8)))
            zone = m.astype(np.float32)
        result = self._composite_reference(canvas, ref_aligned, zone,
                                           span, tone_src=image)
        f = float(np.clip(self._tps_flexibility, 0.0, 1.0))
        if f < 1.0:
            result = np.clip(result.astype(np.float32) * f +
                             image.astype(np.float32) * (1.0 - f),
                             0, 255).astype(np.uint8)
        t3 = time.time()
        elapsed = (time.time() - t0) * 1000
        info = (f"参考移植: 抹除 {(t2-t1)*1000:.0f}ms | "
                f"融合 {(t3-t2)*1000:.0f}ms | 镜像侧: {side}")
        return FrontalizationResult(result, elapsed, info)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def convert(self, image: np.ndarray, keypoints: dict = None,
                **kwargs) -> FrontalizationResult:
        t0 = time.time()

        if image is None:
            return FrontalizationResult(
                np.zeros((100, 100, 3), dtype=np.uint8), 0, "无图像")
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if keypoints is None or len(keypoints) < 6:
            return FrontalizationResult(image, 0, "关键点不足，至少需要6个")

        cx = self._axis_x(keypoints)

        # In a pure profile the inner eye corner sits almost on the axis,
        # so the warped silhouette cuts through the face center and must be
        # erased. In a 3/4 view (e.g. animals with both eyes visible) the
        # center holds real features (nose, stripes) — erasing would wash
        # them out. Gate on the source geometry.
        self._erase_center = True
        ei, eo = keypoints.get("eye_left_inner"), keypoints.get("eye_left_outer")
        if ei is not None and eo is not None:
            fw = abs(eo[0] - ei[0])
            if fw > 3 and abs(ei[0] - cx) / fw > 0.45:
                self._erase_center = False
                # 3/4 view: mirror about the MUZZLE midline (nose/mouth/
                # chin), not the mean of all center points — on a rotated
                # head the forehead/cranium center projects far toward the
                # visible side, and using it as the axis doubles the nose.
                cxs = [keypoints[n][0] for n in
                       ("nose_tip", "mouth_center", "chin_tip")
                       if n in keypoints]
                if cxs:
                    cx = float(np.mean(cxs))

        side = self._visible_side(keypoints, cx)

        # Pre-generated frontal reference (registered by image md5):
        # supplies natural asymmetric GEOMETRY (replacing the mirrored
        # template) and real PIXELS for the occluded regions (replacing
        # mirroring/synthesis). Prepared offline, runs fully offline.
        ref_img = kwargs.get("reference")
        ref_kps = kwargs.get("reference_kps")
        ref_off = kwargs.get("reference_offset", (0.0, 0.0))
        has_ref = ref_img is not None and bool(ref_kps)

        # 3/4 view (both eyes visible, e.g. the cat): TURN the head to
        # frontal by warping to an anatomical frontal template — no
        # mirroring across the face, ears/skull pinned in place, only
        # the occluded far eye is synthesized. Mirroring the whole half
        # face produced a narrow symmetric mask disconnected from the
        # skull and replaced the ears; a rotation-style warp keeps all
        # original content coherent (the foreshortened side is simply
        # stretched, like re-photographing the turned head).
        if not self._erase_center:
            return self._rotation_frontal(
                image, keypoints, side, cx, t0,
                reference=ref_img if has_ref else None,
                reference_kps=ref_kps if has_ref else None,
                reference_offset=ref_off)

        frontal_kps = self._build_frontal_template(
            keypoints, cx, pin_center=not self._erase_center)

        # A mirrored frontal face is wider than the profile head: pad the
        # canvas (edge-replicated) so the mirrored half is not clipped.
        # Pure profiles pad enough to also RE-CENTER the figure: the
        # mirror axis lands on the canvas middle and the output is
        # cropped around it (below), so the symmetric frontal figure
        # sits in the center of the frame like a portrait photo.
        orig_w = image.shape[1]
        xs_f = [p[0] for p in frontal_kps.values()]
        span = max(xs_f) - min(xs_f) + 1e-6
        margin = span * 0.10
        right_ext = max(xs_f) + margin
        pad_left = int(max(0, right_ext - 2 * cx))
        pad_left = max(pad_left, orig_w // 2 - int(round(cx)))
        if pad_left > 0:
            image = cv2.copyMakeBorder(
                image, 0, 0, pad_left, 0, cv2.BORDER_REPLICATE)
            cx += pad_left
            frontal_kps = {k: (x + pad_left, y)
                           for k, (x, y) in frontal_kps.items()}
            keypoints = {k: (x + pad_left, y)
                         for k, (x, y) in keypoints.items()}

        # Reference geometry: fit the pre-generated frontal reference
        # onto the classical template — its natural asymmetric keypoints
        # REPLACE the mirrored template, its pixels replace the mirror.
        refit_aligned = None
        if has_ref:
            refit, refit_aligned = self._fit_reference(
                ref_img, ref_kps, frontal_kps, image.shape, ref_off)
            if refit is not None:
                frontal_kps = refit
            else:
                refit_aligned = None

        ordered = [k for k in DEFAULT_KEYPOINTS if k in keypoints]
        src_pts = np.array([keypoints[k] for k in ordered], dtype=np.float64)
        dst_pts = np.array([frontal_kps[k] for k in ordered],
                           dtype=np.float64)

        t1 = time.time()
        # 1) TPS warp annotated points onto the frontal template
        #    (borders anchored, so only the face deforms). The warp
        #    strength blends src->dst: 1.0 = full frontalization,
        #    0.0 = no deformation (the mirror step still frontalizes).
        f = float(np.clip(self._tps_flexibility, 0.0, 1.0))
        dst_eff = src_pts + f * (dst_pts - src_pts)
        warped = self._tps_warp(image, src_pts, dst_eff)

        # 2) Erase the profile nose/lips silhouette around the axis, then
        #    fill the occluded half — from the REFERENCE when available
        #    (natural asymmetric content), else mirror the VISIBLE side
        #    every row (face, hair AND shoulders become symmetric).
        warped = self._erase_center_lines(warped, cx, frontal_kps)
        if refit_aligned is not None:
            hp = np.array(list(frontal_kps.values()), np.float32)
            span_r = float(np.ptp(hp[:, 0]) + 1e-6)
            yy, xx = np.mgrid[0:warped.shape[0], 0:warped.shape[1]]
            # Sigmoid seam across the axis: reference on the far side,
            # warped original on the visible side. Both layers share the
            # SAME frontal geometry, so a wide soft seam cannot ghost —
            # it just hides the texture transition completely.
            sig = ((xx - cx) if side == "right" else (cx - xx)) \
                / (0.06 * span_r)
            zone = (1.0 / (1.0 + np.exp(np.clip(sig, -30, 30)))
                    ).astype(np.float32)
            result = self._composite_reference(warped, refit_aligned,
                                               zone, span_r, soft=True)
            t2 = time.time()
            if self._poisson_blend:
                try:
                    result = self._smooth_seam(
                        result, cx,
                        np.ones(result.shape[:2], np.float32))
                except Exception:
                    pass
            # Reference geometry is already naturally proportioned: skip
            # nose synthesis, wedge fill, narrowing and shoulder cut.
            x0 = int(round(cx)) - orig_w // 2
            if x0 < 0:
                result = cv2.copyMakeBorder(
                    result, 0, 0, -x0, 0, cv2.BORDER_REPLICATE)
                x0 = 0
            if x0 + orig_w > result.shape[1]:
                result = cv2.copyMakeBorder(
                    result, 0, 0, 0, x0 + orig_w - result.shape[1],
                    cv2.BORDER_REPLICATE)
            out = result[:, x0:x0 + orig_w]
            elapsed = (time.time() - t0) * 1000
            info = (f"参考TPS: {(t2-t1)*1000:.0f}ms | 区域移植 | "
                    f"镜像侧: {side}")
            return FrontalizationResult(out, elapsed, info)

        result = self._mirror_completion(warped, cx, side,
                                         self._mirror_strength,
                                         keep_region=None)
        t2 = time.time()

        mask = self._face_hull_mask(frontal_kps, image.shape, cx)

        # Pure profile: the midline between eyes and mouth is blank after
        # the mirror — synthesize a soft nose-bridge shadow.
        result = self._synthesize_nose(result, cx, frontal_kps, mask)

        # The mirrored canvas still shows the original BACKGROUND between
        # neck and collar — a blank white wedge under the chin. Fill it.
        result = self._fill_neck_wedge(result, cx, frontal_kps)

        # 3) Optional: match the mirrored half's tone to the visible half.
        if self._color_match:
            result = self._match_color_lab(result, cx, side, mask)
        t3 = time.time()

        # 4) Optional: soften the mirror seam (all rows — the seam now
        #    runs through hair and shoulders too, not just the face).
        if self._poisson_blend:
            try:
                result = self._smooth_seam(
                    result, cx, np.ones(result.shape[:2], np.float32))
            except Exception:
                pass
        t4 = time.time()

        # 5) Full-canvas composite: the whole symmetrized figure (face +
        #    hair + body) is the result. Then crop a window CENTERED on
        #    the mirror axis back to the input width — the frontal
        #    figure sits in the middle of the frame like a portrait
        #    (the pad guaranteed the axis is at/right-of the canvas center,
        #    so the left margin is the edge-replicated background).
        x0 = int(round(cx)) - orig_w // 2
        if x0 < 0:
            result = cv2.copyMakeBorder(
                result, 0, 0, -x0, 0, cv2.BORDER_REPLICATE)
            x0 = 0
        if x0 + orig_w > result.shape[1]:
            result = cv2.copyMakeBorder(
                result, 0, 0, 0, x0 + orig_w - result.shape[1],
                cv2.BORDER_REPLICATE)
        out = result[:, x0:x0 + orig_w]

        # 6) A mirrored profile is ~1.4x too WIDE (head depth > head
        #    width) — compress horizontally so the figure has natural
        #    frontal proportions instead of looking like two bodies.
        corners = np.concatenate([
            image[:12, :12].reshape(-1, 3), image[:12, -12:].reshape(-1, 3),
            image[-12:, :12].reshape(-1, 3), image[-12:, -12:].reshape(-1, 3)])
        bg_color = tuple(float(v) for v in np.median(corners, axis=0))
        out = self._narrow_figure(out, frontal_kps, bg_color)

        # 7) The mirrored body has TWO shoulder humps per side (front +
        #    back of the profile hoodie) — cut to one clean silhouette.
        out = self._cut_shoulder_silhouette(out, frontal_kps, bg_color)

        # 8) Final-coords keypoints (crop offset + 0.6 head compression)
        #    for the last two touch-ups: trim the mirrored side-hair
        #    curtains (twin-tail look) to a short frontal cut, and close
        #    the white V wedge between the jacket fronts (two-necks
        #    reading) with jacket tone + the hoodie's own center zipper.
        cxo = orig_w / 2.0
        final_kps = {k: (cxo + 0.60 * ((x - x0) - cxo), y)
                     for k, (x, y) in frontal_kps.items()}
        out = self._trim_side_hair(out, final_kps, bg_color)
        out = self._close_chest_gap(out, final_kps, bg_color)

        # 9) Optional: the one feature a pure profile truly lacks is the
        #    frontal nose — transplant ONLY its linework from the
        #    registered generated reference (identity-critical regions
        #    never come from generation).
        if self._face_ref_img is not None and self._face_ref_kps:
            out = self._transplant_face_interior(
                out, final_kps, self._face_ref_img, self._face_ref_kps)
        elif self._nose_ref_img is not None and self._nose_ref_kps:
            out = self._transplant_nose_strokes(
                out, final_kps, self._nose_ref_img, self._nose_ref_kps)

        elapsed = (time.time() - t0) * 1000
        info = (f"TPS: {(t2-t1)*1000:.0f}ms | 色调: {(t3-t2)*1000:.0f}ms | "
                f"融合: {(t4-t3)*1000:.0f}ms | 镜像侧: {side}")
        return FrontalizationResult(out, elapsed, info)
