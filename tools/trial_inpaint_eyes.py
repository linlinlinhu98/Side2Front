"""Brow+eye band INPAINT trial — round 4.

Round-3 verdict (user): eyes finally NORMAL but still not close
enough to the original, and there are THREE pairs of brows (moved
patch + reinforce line offset from each other + dark smudge above).

Original's eye studied from _orig_eye_zoom.png: DELICATE — thin
upper lash (val ~70, 2px), SMALL medium-dark iris (not black),
bright eye white, a clearly visible thin lid crease, faint lower
lid, tiny outer lash flick. Brow: THICK SOFT-EDGED shaded shape,
tapered at the outer end, not a hard line.

Round 4: erase the WHOLE brow+eye band to clean skin (no ghosts
possible), hand-draw ONE soft thick brow + one delicate eye per
side, then inpaint at strength 0.55 (model only adds pencil
texture). Single band mask. SD1.5 base (best style match),
2 seeds. Output: assets/geom/inpaint_eyes/
"""
import cv2
import json
import numpy as np
import os
import sys
import time
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from core.anime_face import AnimeFaceFrontalizer  # noqa: E402
from trial_final_face import DONOR_KPS, _gray  # noqa: E402
from trial_final_face2 import skin_erase  # noqa: E402

OUT = os.path.join(ROOT, "assets", "geom", "inpaint_eyes")
EYES = ((215, 219), (297, 219))       # composite eye centers
# band rect covering brow+eye (both sides, incl. glabella skin);
# top extended to 156 to erase the dark forehead smudges that read
# as a third pair of brows
BAND = (169, 178, 343, 240)

PROMPTS = {
    "sd": ("pencil sketch, monochrome, sharp narrow eyes, small "
           "dark pupils, semi-realistic, fine pencil strokes, "
           "thick eyebrows, front view",
           "color, photo, anime, cartoon, big eyes, cute, child, "
           "deformed, ugly, watermark, text, blurry"),
    "cf": ("(monochrome:1.3), (sketch:1.2), fine pencil strokes, "
           "sharp narrow eyes, tsurime, small pupils, thick "
           "eyebrows, looking at viewer, 1boy, mature face, "
           "semi-realistic",
           "color, colored, (big eyes:1.3), (glowing eyes:1.2), "
           "child, shota, photo, realistic, 3d, deformed, "
           "watermark, text"),
}


def _find_file(folder, suffix):
    for dp, _, fns in os.walk(folder):
        for fn in fns:
            if fn.endswith(suffix):
                return os.path.join(dp, fn)
    raise FileNotFoundError(suffix)


def _brow_poly(cx, mirror, shrink=0):
    """SWORD brow (user R8 verdict: still too thick, not curved).
    Re-measured the original: her brow is a 剑眉 — low at the inner
    end, RISING in a clear arc toward the temple, thin (~5-6px
    rendered, so drawn thinner — the model thickens ~1.5x).
    mirror=1 -> image-left brow (inner = larger x)."""
    s = shrink
    # R15/R16: thickest at the INNER end (~10px) with a ROUNDED
    # inner head (user: "左边圆润一点" — no flat cut), tapering
    # steadily to a point at the outer tip, keeping the rising
    # sword-brow arc. mirror=1 -> image-left brow (inner = +x).
    return np.array([(cx + 28 * mirror, 187 + s),   # inner top
                     (cx + 10 * mirror, 182 + s),
                     (cx - 12 * mirror, 183 + s),
                     (cx - 32 * mirror, 187),       # outer tip
                     (cx - 10 * mirror, 188 - s),
                     (cx + 12 * mirror, 190 - s),
                     (cx + 28 * mirror, 197 - s),   # inner bottom
                     (cx + 31 * mirror, 195),       # rounded cap
                     (cx + 33 * mirror, 192),       # cap nose
                     (cx + 31 * mirror, 189)],      # rounded cap
                    np.int32)


def _stroked_brow(img, cx, mirror):
    """R13: SMOOTH SHADED brow — the original's own brow is a
    soft-edged shaded shape, NOT hair strokes (user: "为什么眉毛是
    由一条条短线构成的"). Solid fill + darker core + feathered
    edges, same sword-brow shape and R12 thickness (both accepted)."""
    # soft edge mask (R17: blur 2.0 — user: brows not smooth enough)
    m = np.zeros(img.shape, np.float32)
    cv2.fillPoly(m, [_brow_poly(cx, mirror)], 1.0, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 2.0)
    # body: mid-gray fill, darker core (R17: darker — user: 淡了)
    body = np.full(img.shape, 255, np.float32)
    cv2.fillPoly(body, [_brow_poly(cx, mirror)], 135.0, cv2.LINE_AA)
    cv2.fillPoly(body, [_brow_poly(cx, mirror, shrink=2)], 85.0,
                 cv2.LINE_AA)
    body = cv2.GaussianBlur(body, (0, 0), 2.0)
    img[:] = np.clip(img.astype(np.float32) * (1 - m)
                     + body * m, 0, 255).astype(np.uint8)


def _draw_features(img, for_cond):
    """Draw one delicate eye per side (R5: NO brows here — the brows
    are original pixels pasted back in build_base; the cond version
    only needs eye lines since the pasted brows already show up in
    the extracted lineart). for_cond=True -> darker/thicker strokes
    for the ControlNet lineart image."""
    for (cx, cy), mirror in zip(EYES, (1, -1)):
        t = 8 * mirror
        if for_cond:
            cv2.fillPoly(img, [_brow_poly(cx, mirror)], 55,
                         cv2.LINE_AA)
            cv2.ellipse(img, (cx, cy), (28, 8), t, 192, 348, 35, 3,
                        cv2.LINE_AA)
            cv2.ellipse(img, (cx, cy), (28, 8), t, 25, 165, 100, 1,
                        cv2.LINE_AA)
            cv2.circle(img, (cx, cy - 1), 5, 80, -1, cv2.LINE_AA)
            cv2.ellipse(img, (cx, cy - 4), (28, 11), t, 212, 328,
                        140, 1, cv2.LINE_AA)
        else:
            _stroked_brow(img, cx, mirror)
            # delicate eye: narrow almond, iris tucked under the
            # upper lash (original: iris spans the whole opening)
            cv2.ellipse(img, (cx, cy), (28, 8), t, 0, 360, 245, -1,
                        cv2.LINE_AA)                    # eye white
            cv2.circle(img, (cx, cy - 1), 5, 150, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy - 1), 2, 85, -1, cv2.LINE_AA)
            cv2.ellipse(img, (cx, cy), (28, 8), t, 192, 348, 65, 2,
                        cv2.LINE_AA)                    # upper lash
            cv2.ellipse(img, (cx, cy), (28, 8), t, 25, 165, 145, 1,
                        cv2.LINE_AA)                    # lower lid
            cv2.ellipse(img, (cx, cy - 4), (28, 11), t, 212, 328,
                        160, 1, cv2.LINE_AA)            # crease
            # tiny outer lash flick (signature of the original eye)
            cv2.line(img, (cx - 26 * mirror, cy - 5),
                     (cx - 31 * mirror, cy - 8), 65, 1, cv2.LINE_AA)
    return img


def _fix_body(img):
    """R19: repair the mirrored torso (user: 两个身体、两个脖子、
    明显接缝线). Mirroring the profile hoodie gives each half BOTH
    the chest-front edge (= the doubled center seam) AND the
    back-hood outline (= a full-length inner contour reading as a
    SECOND body), plus a wide square collar hole (= second neck).
    Fixes: targeted Telea-erase of the inner contours / wide collar
    / seam strip, then redraw ONE narrow V collar + ONE zipper."""
    from trial_final_face2 import skin_erase
    def erase_lines(pts, w=6, thresh=220):
        m = np.zeros(img.shape, np.uint8)
        cv2.polylines(m, [np.array(pts, np.int32)], False, 255, w)
        dark = ((img < thresh).astype(np.uint8)) * 255
        m = cv2.bitwise_and(m, dark)
        return cv2.dilate(m, np.ones((5, 5), np.uint8))
    out = img.copy()
    # 1) white-erase BOTH shoulder-top bands (y338-402): the mirror
    # duplicated the hood-behind-neck + shoulder outlines, so each
    # side had TWO stacked shoulder humps = the 'second body'.
    # (Compared against the original: folds/stripe/pocket below
    # y405 are all legit single-body lines — keep them.)
    out = skin_erase(out, (90, 338, 226, 402), feather=6)
    out = skin_erase(out, (286, 338, 422, 402), feather=6)
    # leftover jaw-side shading smudges clipped by the band top
    out = skin_erase(out, (100, 322, 142, 346), feather=4)
    out = skin_erase(out, (370, 322, 412, 346), feather=4)
    # 2) erase: the wide flat collar line ('second neck'), the
    # collar-adjacent inner curve stub below the band, the seam
    masks = [erase_lines([(198, 353), (256, 362), (316, 353)]),
             erase_lines([(196, 392), (183, 410), (172, 428),
                          (168, 445)]),
             erase_lines([(316, 392), (329, 410), (340, 428),
                          (344, 445)])]
    strip = np.zeros(img.shape, np.uint8)
    strip[388:660, 244:268] = 255
    strip = cv2.bitwise_and(strip, ((out < 210).astype(np.uint8)) * 255)
    masks.append(cv2.dilate(strip, np.ones((5, 5), np.uint8)))
    m = masks[0]
    for mm in masks[1:]:
        m = cv2.bitwise_or(m, mm)
    out = cv2.inpaint(out, m, 4, cv2.INPAINT_TELEA)
    # 3) redraw ONE smooth shoulder line per side (curved, soft,
    # ending ON the fabric mass so nothing floats) + narrow V
    # collar + single zipper + pull ring
    for pts in ([(222, 358), (190, 368), (155, 384), (120, 402),
                 (100, 418), (95, 428)],
                [(290, 358), (322, 368), (357, 384), (392, 402),
                 (412, 418), (417, 428)]):
        cv2.polylines(out, [np.array(pts, np.int32)], False, 110, 2,
                      cv2.LINE_AA)
    cv2.line(out, (228, 358), (256, 398), 110, 2, cv2.LINE_AA)
    cv2.line(out, (285, 358), (256, 398), 110, 2, cv2.LINE_AA)
    cv2.line(out, (256, 398), (256, 655), 120, 2, cv2.LINE_AA)
    cv2.circle(out, (256, 463), 4, 110, 1, cv2.LINE_AA)
    return out, m


def _redraw_torso(img):
    """R20: REDRAW the torso as a real frontal hoodie (user: 你没有
    把衣服从侧面转正面，只是两个侧面加框). Recipe from the
    classmate's result (_classmate_torso.png): bunched hood collar
    AROUND the neck (soft shaded mass), wide V opening, single
    zipper, smooth shoulders, DARK SIDE PANELS (= the original
    hoodie's dark back/side fabric), a few folds, everything soft
    (blurred fills) — NOT a line tracing of the profile folds."""
    out = img.copy().astype(np.float32)
    H, W = out.shape

    def xm(pts):                    # mirror about the center x=256
        return [(512 - x, y) for x, y in pts]

    # ---- 1) erase the whole torso below the jaw, keep the neck
    fill = cv2.GaussianBlur(img, (0, 0), 25).astype(np.float32)
    m = np.zeros(out.shape, np.float32)
    m[338:H, 50:470] = 1.0
    cv2.fillPoly(m, [np.array([(212, 328), (300, 328), (290, 375),
                               (222, 375)], np.int32)], 0.0)
    m = cv2.GaussianBlur(m, (0, 0), 4)
    out = out * (1 - m) + fill * m

    # ---- 2) MINIMAL tone scaffold only (user R22: 我是要你生成 —
    # the MODEL must generate the clothes, not procedural drawing).
    # Just enough signal: light gray where dark fabric belongs, so
    # the model generates ITS OWN pencil texture at high strength.
    out = np.clip(out, 0, 255).astype(np.uint8)
    # R23: bunch extended outward-up beside the jaw — R22's model
    # invented two white lumps at the jaw sides where the scaffold
    # gave no fabric signal
    bunch = [(216, 346), (186, 346), (158, 354), (130, 362),
             (116, 382), (118, 406), (136, 424), (146, 428),
             (170, 432), (190, 420), (204, 396), (222, 354)]
    panel = [(150, 418), (126, 444), (103, 484), (88, 544),
             (80, 620), (78, 660), (126, 660), (130, 545),
             (140, 478), (154, 428)]
    soft = np.full(out.shape, 255, np.float32)
    cov = np.zeros(out.shape, np.float32)
    for pts, v in ((bunch, 225.0), (panel, 150.0)):
        for p2 in (pts, xm(pts)):
            cv2.fillPoly(soft, [np.array(p2, np.int32)], v,
                         cv2.LINE_AA)
            cv2.fillPoly(cov, [np.array(p2, np.int32)], 1.0,
                         cv2.LINE_AA)
    soft = cv2.GaussianBlur(soft, (0, 0), 5)
    cov = cv2.GaussianBlur(cov, (0, 0), 5)
    out = np.clip(out.astype(np.float32) * (1 - cov) + soft * cov,
                  0, 255).astype(np.uint8)

    # ---- 3) line work — kept LIGHT (scaffold only; the model
    # renders the final fabric). Zipper is just a short stub so the
    # model extends/fades it naturally (R20's full-length perfectly
    # straight zipper read as a SEAM — user: 依旧有缝合线).
    def pl(pts, val=135, w=1):
        cv2.polylines(out, [np.array(pts, np.int32)], False, val, w,
                      cv2.LINE_AA)
    silhouette = [(150, 418), (128, 445), (105, 485), (90, 545),
                  (82, 620), (80, 659)]
    bunch_edge = [(216, 347), (186, 347), (158, 355), (130, 363),
                  (117, 382), (119, 406), (136, 424), (146, 428)]
    bunch_open = [(222, 353), (202, 366), (188, 388), (184, 410)]
    panel_edge = [(154, 428), (140, 478), (130, 545), (126, 659)]
    for pts in (silhouette, bunch_edge, bunch_open, panel_edge):
        pl(pts)
        pl(xm(pts))
    # collar V down to the zipper top
    pl([(224, 353), (256, 415)], 120)
    pl([(288, 353), (256, 415)], 120)
    # short zipper stub + small ring pull
    pl([(256, 415), (256, 480)], 140)
    cv2.circle(out, (256, 452), 3, 110, 1, cv2.LINE_AA)
    # the original's signature dark shoulder stripe (tapered band,
    # kept INSIDE the silhouette — R20 preview: it poked out like
    # little wings)
    for pts in ([(158, 430), (136, 446), (120, 462)],
                [(354, 430), (376, 446), (392, 462)]):
        cv2.polylines(out, [np.array(pts, np.int32)], False, 70, 6,
                      cv2.LINE_AA)
    # light fold strokes: collar bunch + chest + side panels
    for pts, v, w in ([(200, 360), (185, 385)], 165, 1), \
                     ([(180, 362), (168, 388)], 165, 1), \
                     ([(212, 375), (200, 400)], 165, 1), \
                     ([(230, 480), (196, 495)], 175, 1), \
                     ([(232, 540), (194, 552)], 175, 1), \
                     ([(240, 600), (206, 610)], 175, 1):
        cv2.polylines(out, [np.array(pts, np.int32)], False, v, w,
                      cv2.LINE_AA)
        cv2.polylines(out, [np.array(xm(pts), np.int32)], False, v,
                      w, cv2.LINE_AA)
    return out, m


def _fix_hair(img, orig):
    """R44 (user: 头发像两个人 + 白色斑点 + 完全不像原图侧面转正面):
    patchwork can never become ONE head of hair — do what
    'side -> front' actually means geometrically: a CYLINDRICAL
    DEPTH REMAP of the original's own hair. The profile's
    front->back axis maps to the front view's center->sides:
    bangs (src x~95-160) -> the center columns, mid crown
    (x160-280) -> mid, back/nape (x280-412) -> the sides. Both
    halves sample the SAME source column, so the whole head is one
    continuous original texture — no patches, no seams, no foreign
    strokes; the silhouette, the bang hairline and the side locks
    all fall out of the source content itself, including the
    original's own thin white lock-gaps (streaks, not blob holes)."""
    g = img.astype(np.float32)
    H, W = g.shape
    og = _gray(orig).astype(np.float32)
    # the original's bang-bottom white gap (y~170-190, x95-200 and
    # x260-330) is a wide bright band; depth-mapped to the center
    # columns it would read as a white slash above the brows. Fill
    # it in the source copy (Telea pulls the surrounding dark
    # strands through)
    gapm = np.zeros(og.shape, np.uint8)
    gapm[166:194, 90:205] = (og[166:194, 90:205] > 170
                             ).astype(np.uint8) * 255
    gapm[180:194, 260:330] = (og[180:194, 260:330] > 170
                              ).astype(np.uint8) * 255
    og = cv2.inpaint(og.astype(np.uint8), gapm, 4,
                     cv2.INPAINT_TELEA).astype(np.float32)
    # the ear (og x215-300, y190-300) sits IN FRONT of the side
    # hair — depth-mapped to the front-view sides it would float
    # inside the side locks. Fill the box with the dense nape hair
    # directly behind it
    og[185:306, 215:306] = og[185:306, 305:396]
    # depth axis -> front half-width: piecewise-linear column map.
    # TWO maps blended by height: the hairline map (center <- the
    # bangs front) and the crown-top map (center <- the crown
    # front at src x~210, where hair starts at y~15) — a single
    # map left a bald V notch at the top center (the bang column
    # starts at y~60, the crown at y~15)
    xs = np.arange(W, dtype=np.float32)
    d = np.abs(xs - 256.0)
    knots = [0.0, 45.0, 80.0, 115.0, 160.0]
    sx_main = np.interp(d, knots, [125.0, 180.0, 250.0, 330.0, 405.0])
    sx_top = np.interp(d, knots, [210.0, 245.0, 275.0, 305.0, 340.0])
    yy1 = np.arange(H, dtype=np.float32)[:, None]
    wt = np.clip((90.0 - yy1) / 70.0, 0, 1)        # 1 at the top
    src_x = sx_main[None, :] * (1 - wt) + sx_top[None, :] * wt
    map_x = src_x.astype(np.float32)
    map_y = np.tile(np.arange(H, dtype=np.float32)[:, None],
                    (1, W)).astype(np.float32)
    hair = cv2.remap(og, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    # silhouette from the remapped mass itself (the interp clamps
    # outside |dx|>160 and would smear one column — cut it there)
    dark = (hair < 150).astype(np.uint8)
    dark[:8, :] = 0
    dark[:, :92] = 0
    dark[:, 420:] = 0
    sil = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                           np.ones((15, 15), np.uint8))
    sil = cv2.dilate(sil, np.ones((5, 5), np.uint8))
    # keep OFF the face; the ellipse top (y178 at the center) lets
    # the bangs hang inside its upper part like the original's
    face = np.zeros(g.shape, np.uint8)
    cv2.ellipse(face, (256, 282), (92, 104), 0, 0, 360, 1, -1)
    sil = sil & (1 - face)
    silf = cv2.GaussianBlur(sil.astype(np.float32), (0, 0), 3)
    # erase ALL the old warp hair in the head region (but the face
    # and ears), then paste the remapped hair — 100% original
    # pixels inside the silhouette, clean paper outside
    zone = np.zeros(g.shape, np.float32)
    zone[0:340, 56:456] = 1.0
    protect = face.astype(np.float32)
    ears = np.zeros(g.shape, np.uint8)
    ears[220:322, 132:202] = 1
    ears[220:322, 310:380] = 1
    protect = np.clip(protect + ears.astype(np.float32), 0, 1)
    protect = cv2.GaussianBlur(protect, (0, 0), 2)
    zone = cv2.GaussianBlur(zone * (1 - protect), (0, 0), 3)
    out = img.astype(np.float32)
    out = out * (1 - zone) + 252.0 * zone
    out = out * (1 - silf) + hair * silf
    img[:] = np.clip(out, 0, 255).astype(np.uint8)


_CHIN_LINE = True


def _fix_neck(img, orig):
    """User: 还是两个脖子 + 两条竖线, then 没有脖子一片空白.
    Comparing with the classmate reference: her neck HAS soft pencil
    side lines (tapered, fading into the collar) and a soft shadow
    under the jaw — just not hard straight marker lines. Erase the
    whole zone clean, then redraw the neck in that soft style."""
    m = np.zeros(img.shape, np.float32)
    # R34: the big center erase poly is GONE — the frontalized
    # ORIGINAL's chin + neck trapezoid (blank front + contour
    # lines at x~194/310) is the original artist's own drawing;
    # erasing it forced SD to redraw the chin (user: 下巴和脖子
    # 太诡异). Only the OUTER doubled smudges are erased now.
    # clear the smudges BELOW THE EARS (old mirrored collar/
    # hair mix at x140-190 / 322-372, y340-400) — narrowed so they
    # don't clip the trapezoid contour lines at x~194/310
    cv2.fillPoly(m, [np.array([(140, 340), (190, 340), (188, 400),
                               (146, 400)], np.int32)], 1.0)
    cv2.fillPoly(m, [np.array([(322, 340), (372, 340), (366, 400),
                               (324, 400)], np.int32)], 1.0)
    # trim the scraggly doubled hair TAILS beside the jaw (user:
    # 白色发尾) — L-shaped: outer zone from y282, plus the x148-170
    # strip from y293 (R36 measured leftover strands at x149-166
    # y290-322 that the old x100-148/y305 box missed); top edge
    # stays ~4px below the earlobe so the ear is not clipped
    cv2.fillPoly(m, [np.array([(100, 308), (148, 308), (148, 318),
                               (170, 318), (170, 405), (100, 405)],
                              np.int32)], 1.0)
    cv2.fillPoly(m, [np.array([(412, 308), (364, 308), (364, 318),
                               (342, 318), (342, 405), (412, 405)],
                              np.int32)], 1.0)
    # R36 (user: R35 还是两个脖子 + 发尾还是白色): the SECOND 'neck'
    # is the soft jaw-shading flare ABOVE the trapezoid (y300-340,
    # just outside the hard contour lines) — erase it; the white
    # tails are the wispy mirrored strands hanging below the hair
    # mass (x146-196 / 316-366, y322+) — erase. (NO seam-stub erase
    # at the trapezoid bottom: white-ing x250-266 y394-422 cut a
    # hard rectangular NOTCH through the collar fold cluster; the
    # short center stub reads as the zipper top once the knot
    # cluster is pasted around it)
    cv2.fillPoly(m, [np.array([(168, 300), (207, 300), (205, 340),
                               (170, 340)], np.int32)], 1.0)
    cv2.fillPoly(m, [np.array([(305, 300), (344, 300), (342, 340),
                               (307, 340)], np.int32)], 1.0)
    cv2.fillPoly(m, [np.array([(146, 334), (196, 334), (194, 354),
                               (148, 354)], np.int32)], 1.0)
    cv2.fillPoly(m, [np.array([(316, 334), (366, 334), (364, 354),
                               (318, 354)], np.int32)], 1.0)
    m = cv2.GaussianBlur(m, (0, 0), 3)
    img[:] = np.clip(img.astype(np.float32) * (1 - m) + 252 * m,
                     0, 255).astype(np.uint8)
    # R41 (user: 脖子和衣服接不上像两个躯体 + 纹理太多 + 一定要
    # 参考原图): the ORIGINAL's neck is nearly EMPTY — clean white
    # skin, ONE soft shadow under the jaw, two clean contour lines.
    # The 'two bodies' disconnect came from the neck being a short
    # white stub ENDING at y405 above the collar: extend the column
    # down to y428 with a flaring base so the shading + contour
    # lines run INTO the collar knot cluster (the hood wraps the
    # neck base). Almost no interior texture.
    # R42 (user: 脖子太细像圆柱 + 和衣服接不上): compare with the
    # classmate ref — the neck is WIDE (~2/3 of jaw width) and
    # SHORT, and the collar wraps UP around its base. Widen the
    # column to x~222/290, end it at y~414 where the (raised)
    # collar knot takes over
    zone = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(zone, [np.array([(178, 352), (208, 343), (256, 339),
                                  (304, 343), (334, 352), (356, 418),
                                  (156, 418)], np.int32)], 255)
    zone_f = cv2.GaussianBlur(zone.astype(np.float32) / 255.0,
                              (0, 0), 2)
    img[:] = np.clip(img.astype(np.float32) * (1 - zone_f)
                     + 252.0 * zone_f, 0, 255).astype(np.uint8)
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = np.abs(xx - 256.0)
    # WIDE column (~68px), gentle taper, flaring base (trapezius)
    hw = (34.0 - 2.0 * np.clip((yy - 345.0) / 55.0, 0, 1)
          + 20.0 * np.clip((yy - 392.0) / 26.0, 0, 1))
    side = np.clip(dx / np.maximum(hw, 1.0), 0, 1.5)
    col = np.clip((hw + 2.0 - dx) / 5.0, 0, 1)
    col *= np.clip((yy - 342.0) / 5.0, 0, 1)
    col *= np.clip((416.0 - yy) / 6.0, 0, 1)
    chin = (np.exp(-((yy - 354.0) / 9.0) ** 2)
            * np.clip(1.0 - (dx / 52.0) ** 2, 0, 1))     # rounded
    # R44: nearly flat base — the original's own neck line + throat
    # shadow are STAMPED below and carry all the shading now
    shade = 3.0
    img[:] = np.clip(img.astype(np.float32) * (1 - col)
                     + (252.0 - shade) * col, 0, 255).astype(np.uint8)
    # R45 (user: 下巴是一条弯曲的线 + 脖子上有一条黑线): the single
    # tilted chin arc AND the tilted neck-line stamps are GONE (the
    # 'neck line' stamps pasted a loose chin fragment as a diagonal
    # slash across the neck). Measured original anatomy: face line
    # (98,325)->(116,358), chin bottom nearly FLAT (116,358)->
    # (190,360), throat line descends from (190,360). So the
    # profile's lower-face outline IS the front view's lower-face
    # outline: face line + chin bottom = left jaw + chin base, the
    # MIRRORED face line closes the right jaw. One canvas, one
    # affine, min-blend — 100% the artist's own chin line
    og = _gray(orig).astype(np.float32)
    if _CHIN_LINE:
        C = np.full((64, 190), 255.0, np.float32)
        base = og[326:364, 94:198].copy()    # face line + chin bottom
        C[0:38, 0:104] = base
        # mirrored WHOLE base: the flipped face line closes the
        # right jaw and the flipped chin bottom lands exactly on
        # itself, so BOTH corners are the artist's own curve join
        C[0:38, 11:115] = np.minimum(C[0:38, 11:115], base[:, ::-1])
        C[34:, 97:] = 255.0                  # throat stub (orig)
        C[34:, 0:20] = 255.0                 # throat stub (mirror)
        pts_src = np.float32([[4, 1], [21, 32], [110, 1]])
        pts_dst = np.float32([[207, 316], [226, 360], [305, 316]])
        Ma = cv2.getAffineTransform(pts_src, pts_dst)
        st = cv2.warpAffine(C, Ma, (img.shape[1], img.shape[0]),
                            borderValue=255)
        aj = np.clip((254.0 - st) / 60.0, 0, 1)
        out = img.astype(np.float32)
        img[:] = np.clip(out * (1 - aj) + np.minimum(out, st) * aj,
                         0, 255).astype(np.uint8)
    # under-chin shadow: CLEAN source window og[362:378,214:246] —
    # pure shading (the old window caught the throat line + the
    # collar shading edge, which pasted as a dark diagonal smudge)
    sh = cv2.resize(og[362:378, 214:246], (100, 40),
                    interpolation=cv2.INTER_LINEAR)
    shC = np.full(img.shape, 255.0, np.float32)
    shC[356:396, 206:306] = sh
    zm = np.zeros(img.shape, np.float32)
    cv2.ellipse(zm, (256, 374), (58, 24), 0, 0, 360, 1, -1)
    zm = cv2.GaussianBlur(zm, (0, 0), 6)
    ash = np.clip((254.0 - shC) / 50.0, 0, 1) * zm
    out = img.astype(np.float32)
    img[:] = np.clip(out * (1 - ash) + np.minimum(out, shC) * ash,
                     0, 255).astype(np.uint8)


def _remap_torso(img, orig):
    """R24: REMAP the ORIGINAL's own torso pixels into the frontal
    layout (user: 我要和原图一模一样的 — generated/redrawn fabric
    never matches the original's pencil-hatching). The profile
    torso from the ZIPPER (front edge, measured per-row) to the
    INNER BACK (dark fabric) maps to each frontal half: zipper ->
    center (both halves share it -> single zipper, NO seam), inner
    back -> the sides (the original's dark back fabric becomes the
    frontal dark side panels). 100% original strokes/texture."""
    g = _gray(orig).astype(np.float32)
    H, W = img.shape
    out = img.copy().astype(np.float32)
    # measured source boundaries (y, x) in ORIGINAL coords
    sf = np.array([(390, 190), (410, 178), (420, 172), (450, 171),
                   (470, 165), (490, 166), (510, 156), (530, 166),
                   (550, 158), (570, 147), (590, 137), (610, 124),
                   (630, 122), (660, 120)], float)
    # top rows stop BEFORE the darkest hood edge (x~380-400) so the
    # dark mass hugs the neck instead of arcing over the shoulders;
    # lower rows keep it -> vertical dark side panels (classmate ref)
    sb = np.array([(390, 340), (430, 365), (470, 385), (510, 398),
                   (550, 398), (590, 398), (630, 390), (660, 388)],
                  float)
    # dest side silhouette (y, x), composite coords. RELAXED
    # shoulders (user: 耸肩): the top starts LOW at the neck
    # (y~400) and slopes gently outward-down, not up toward the neck
    db = np.array([(400, 170), (435, 132), (475, 104), (540, 84),
                   (620, 74), (660, 72)], float)
    Y = np.arange(H, dtype=float)
    sfx = np.interp(Y, sf[:, 0], sf[:, 1])
    sbx = np.interp(Y, sb[:, 0], sb[:, 1])
    dbx = np.interp(Y, db[:, 0], db[:, 1])
    xs = np.arange(W, dtype=float)
    map_x = np.zeros((H, W), np.float32)
    map_y = np.zeros((H, W), np.float32)
    alpha = np.zeros((H, W), np.float32)
    # fabric top edge: shallow V dropped with the relaxed shoulders
    # (sides y400 -> center y418)
    ytop = 400 + 18 * np.clip(1 - np.abs(xs - 256) / 110, 0, 1)
    for y in range(396, H):
        ys = 390 + (y - 396) * (270 / 264)
        yi = min(int(ys), H - 1)
        xbd = dbx[y]
        xfs, xbs = sfx[yi], sbx[yi]
        t = (256.0 - xs) / (256.0 - xbd)      # 0 center -> 1 side
        srcx = xfs + np.clip(t, 0, 1) * (xbs - xfs)
        row_ok = (xs >= xbd) & (xs <= 512 - xbd) & (y >= ytop)
        # left half direct, right half mirrored
        lx = np.clip(xs, xbd, 256)
        map_x[y] = np.where(xs <= 256, srcx, 0)
        map_y[y] = ys
        # right half: mirror of left, but sampled from slightly
        # DIFFERENT src (row +7, inner edge -12) so the folds are
        # not a giveaway-perfect mirror
        ys2 = min(ys + 7, H - 1)
        y2 = int(ys2)
        xfs2, xbs2 = sfx[y2], sbx[y2] - 12
        t2 = (xs - 256.0) / ((512 - xbd) - 256.0)
        srcx2 = xfs2 + np.clip(t2, 0, 1) * (xbs2 - xfs2)
        map_x[y] = np.where(xs > 256, srcx2, map_x[y])
        map_y[y] = np.where(xs > 256, ys2, map_y[y])
        alpha[y] = row_ok.astype(np.float32)
    # R34: collar wrap loop REMOVED — the frontalized ORIGINAL's
    # own neck trapezoid (blank front + two contour lines) is now
    # PRESERVED (widened trapezoid below), and the collar alone is
    # GENERATED by the model pass in main(). Every classical warp
    # of the collar produced the rejected 'towers'.
    warp = cv2.remap(g, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=255)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 2.5)
    # white-erase the old torso first (CONSTANT paper fill — the old
    # blur fill smeared leftover dark pixels into the '模糊黑影'
    # around the neck); keep the redrawn slim neck trapezoid
    em = np.zeros(out.shape, np.float32)
    em[335:H, 40:480] = 1.0
    # R34: protect the ORIGINAL's whole neck trapezoid (blank front
    # + its two contour lines at x~194/310, down to y~400) — the
    # old narrow trapezoid (x226-286) let the erase cut the contour
    # lines, forcing the model to redraw the neck (weird SD chin).
    # R36: TIGHTENED to hug the drawn lines (user: 还是两个脖子) —
    # the old wide poly also protected the soft shading OUTSIDE the
    # lines, whose silhouette read as a second, wider neck
    cv2.fillPoly(em, [np.array([(200, 336), (312, 336), (330, 418),
                                (182, 418)], np.int32)], 0.0)
    em = cv2.GaussianBlur(em, (0, 0), 4)
    out = out * (1 - em) + 252.0 * em
    out = out * (1 - alpha) + warp * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    # one clean silhouette line per side (the src cut has no drawn
    # edge), starting at the COLLAR and tracing the fabric top edge
    # over the shoulder, then down the side — otherwise the line
    # reads as a floating stub and the shoulder top is undefined
    pts = _shoulder_pts()
    cv2.polylines(out, [np.array(pts, np.int32)], False, 100, 2,
                  cv2.LINE_AA)
    cv2.polylines(out, [np.array([(512 - x, y) for x, y in pts],
                                 np.int32)], False, 100, 2,
                  cv2.LINE_AA)
    # R35: collar = the ORIGINAL's own roll band (orig crop, scaled,
    # one per side of the preserved neck, mirrored) — the band
    # already slopes down-out with the layered rolls + dark
    # underside, exactly the frontal collar shape. 100% original
    # pencil strokes, deterministic. MIN-blend strokes only, and
    # never over the preserved neck trapezoid.
    crop = g[345:465, 195:350]                     # 120 x 155
    sc = 0.58
    cw, ch = int(crop.shape[1] * sc), int(crop.shape[0] * sc)
    small = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA)
    # fade the top-left corner (= BOTH bands' outer shoulder
    # corner after mirroring) so stray background strokes there
    # taper out instead of poking past the shoulder as 'antennas'
    fy, fx = np.mgrid[0:ch, 0:cw]
    corner = np.clip((fx + fy) / 40.0, 0, 1).astype(np.float32)
    small = 255.0 - (255.0 - small) * corner
    ax, ay = 208, 360               # R42: raised + moved inward so
                                    # the collar WRAPS the wider
                                    # neck base (user: 接不上)
    canvas = np.full(out.shape, 255.0, np.float32)
    canvas[ay:ay + ch, ax - cw:ax] = small
    canvas[ay:ay + ch, 512 - ax:512 - ax + cw] = small[:, ::-1]
    # the original's knot+fold cluster fills the blank center
    # funnel between the trapezoid bottom and the remapped zipper
    knot = g[418:452, 195:270]                     # 38 x 75
    kw, kh = int(knot.shape[1] * 0.8), int(knot.shape[0] * 0.8)
    ksmall = cv2.resize(knot, (kw, kh), interpolation=cv2.INTER_AREA)
    canvas[392:392 + kh, 256 - kw // 2:256 - kw // 2 + kw] = ksmall
    trap = np.zeros(out.shape, np.uint8)
    cv2.fillPoly(trap, [np.array([(184, 322), (328, 322),
                                  (326, 390), (186, 390)],
                                 np.int32)], 255)
    wm = ((canvas < 195) & (trap == 0)).astype(np.float32)
    wm = cv2.dilate(wm, np.ones((3, 3), np.uint8))
    wm = cv2.GaussianBlur(wm, (0, 0), 1.5)
    outf = out.astype(np.float32)
    dark = np.minimum(outf, canvas)
    out = np.clip(outf * (1 - wm) + dark * wm, 0, 255).astype(np.uint8)
    return out, alpha


def _shoulder_pts():
    """Left shoulder silhouette polyline (collar -> over the fabric
    top edge -> down the side); mirrored by the caller. Shared by
    _remap_torso and the neck-pass seam repair."""
    pts = [(215, 414)]
    for x in range(205, 143, -8):
        pts.append((x, int(400 + 18 *
                           max(1 - abs(x - 256) / 110, 0))))
    pts += [(int(x), int(y)) for y, x in
            [(435, 132), (475, 104), (540, 84), (620, 74), (660, 72)]]
    return pts


def _blush(img):
    """R18: blush distribution measured numerically on the original
    (band mask 140<val<222): a nearly HORIZONTAL band of ~12 thin
    steep strokes at y~218-240, from under the eye (x125) extending
    OUTWARD toward the ear (x185), vals only 15-45 below skin.
    Profile->frontal mapping: 'backward toward ear' becomes
    'outward toward the temple', staying horizontal (R17's band
    descended diagonally and its strokes were 2x too dark/long —
    user: 太明显, 分布不对)."""
    layer = np.zeros(img.shape, np.float32)
    for m in (-1, 1):            # -1 image-left cheek, +1 right
        for i in range(11):
            x = 256 + m * (46 + 5 * i)   # under-eye -> outward
            y = 248 + i                  # ~horizontal, slight arc
            cv2.line(layer, (x, y), (x + m * 3, y + 15),
                     35.0, 1, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (0, 0), 0.8)
    img[:] = np.clip(img.astype(np.float32) - layer, 0, 255
                     ).astype(np.uint8)


def _move_strokes(img, rect, dx, dy, thresh=185.0):
    """Move only the DARK strokes inside rect by (dx,dy): erase
    them at the source with Telea inpaint (keeps paper grain, no
    ghost), stamp them at the destination as a dark-only overlay
    (no skin rectangle — R5's feathered patch paste left visible
    boxes)."""
    x0, y0, x1, y1 = rect
    patch = img[y0:y1, x0:x1].astype(np.float32)
    dark = np.clip(thresh - patch, 0, None)
    # erase at source
    sm = np.zeros(img.shape, np.uint8)
    sm[y0:y1, x0:x1] = (dark > 12).astype(np.uint8) * 255
    sm = cv2.dilate(sm, np.ones((3, 3), np.uint8))
    out = cv2.inpaint(img, sm, 4, cv2.INPAINT_TELEA)
    # stamp at destination
    alpha = np.clip(dark / 12.0, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.5)
    dm = np.zeros(img.shape, np.float32)
    dm[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = alpha
    dk = np.zeros(img.shape, np.float32)
    dk[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = dark
    out = np.clip(out.astype(np.float32) - dk * dm, 0, 255
                  ).astype(np.uint8)
    return out


def build_base():
    """Round 7 — back to the R4 band framework (clean erase +
    model-rendered brow+eye) but with STROKED brows in the fill.
    Classmate reference shows thick tapered brows with hair
    texture; R4's flat fill became a slab, R6's stroke-move of the
    composite's fragmented mirrored brows became broken arcs +
    bracket artifacts. Band erase also removes the mirrored
    fringe-shadow tufts (they read as extra brows)."""
    img = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    mk = json.load(open(os.path.join(ROOT, "manual_keypoints.json"),
                        encoding="utf-8"))
    entry = mk["13ef60de932bb979aaa107ad210a327a"]
    kps = {k: tuple(v) for k, v in entry["keypoints"].items()}
    donor = cv2.imread(os.path.join(ROOT, "assets", "ref_gen",
                                    "face27", "cand_cf_s85.png"))
    f1 = AnimeFaceFrontalizer()
    f1.set_nose_reference(donor, DONOR_KPS)
    out = _gray(f1.convert(img, keypoints=kps).image)
    # R25: pure-black hair (kill the doubled swirls + white tails)
    # and ONE slim neck (kill the mirrored double neck + smudges)
    _fix_hair(out, img)
    _fix_neck(out, img)
    # R24: remap the ORIGINAL's own torso pixels into the frontal
    # layout (replaces R20-23's redraw/generate attempts — only the
    # original pixels match the original's pencil style)
    out, _ = _remap_torso(out, img)
    # erase the whole band (fringe tufts + brows + eyes) to skin
    out = skin_erase(out, BAND, feather=4)
    # restore the original's cheek blush (lost in the warp)
    _blush(out)
    # hand-draw stroked brows + delicate eyes
    erased = _draw_features(out.copy(), for_cond=False)
    x0, y0, x1, y1 = BAND
    # paste-back mask: the WHOLE band (model output there is the
    # VAE-reconstructed erased image incl. the hand-drawn brows)
    mask_paste = np.zeros(out.shape, np.uint8)
    mask_paste[y0:y1, x0:x1] = 255
    mask_paste = cv2.GaussianBlur(mask_paste, (0, 0), 4)
    # R10: model mask covers ONLY the eye strip — R9 proved the
    # model always re-renders brows as thick slabs no matter how
    # thin the drawn fill is, so the drawn sword brows are kept
    # pixel-exact by simply not letting the model touch them
    mask_model = np.zeros(out.shape, np.uint8)
    mask_model[203:242, x0:x1] = 255
    mask_model = cv2.GaussianBlur(mask_model, (0, 0), 3)
    # R21 torso mask for the SECOND model pass (renders real fabric
    # over the hand-drawn scaffold); neck trapezoid excluded
    mask_torso = np.zeros(out.shape, np.uint8)
    mask_torso[338:660, 50:470] = 255
    cv2.fillPoly(mask_torso, [np.array([(212, 328), (300, 328),
                                        (290, 375), (222, 375)],
                                       np.int32)], 0)
    mask_torso = cv2.GaussianBlur(mask_torso, (0, 0), 4)
    return out, erased, mask_paste, mask_model, mask_torso


PROMPT_TORSO = ("pencil sketch, monochrome, hand-drawn, visible "
                "pencil strokes, hatching, sketch of a hoodie "
                "jacket, bunched hood collar around neck, dark "
                "side panels, zipper, fabric folds, front view, "
                "semi-realistic",
                "color, photo, painting, airbrush, smooth shading, "
                "anime, cartoon, deformed, ugly, watermark, text, "
                "blurry, face, eyes")

PROMPT_NECK = ("pencil sketch, monochrome, hand-drawn, visible "
               "pencil strokes, sketch of a bare human neck, "
               "visible neck skin, and a low bunched hood collar "
               "sitting on the shoulders, layered fabric rolls at "
               "the base of the neck, soft pencil shading, fabric "
               "folds, zipper, front view, semi-realistic",
               "color, photo, painting, anime, cartoon, deformed, "
               "ugly, watermark, text, blurry, face, eyes, "
               "lips, mouth, second face, "
               "turtleneck, high collar, scarf, covered neck, "
               "shirt collar, bare chest")

NECK_ZONE = (150, 366, 362, 436)      # x0, y0, x1, y1 — R34: the
# band sits BELOW the original's preserved chin/neck trapezoid;
# the model generates ONLY the collar (a complete face above =
# no more phantom lips / weird SD chin)

MOUTH_EL = ((256, 297), (44, 16))     # center, axes — protect me


def _neck_scaffold():
    """ControlNet scaffold for the R34 COLLAR-ONLY generation band
    (the neck itself is the original's own drawing now): stubs
    continuing the trapezoid contour lines into the band, then
    short banana-curved rolls with rounded tip caps and dark
    undersides (R31 shape — its collar was good), center knot +
    zipper stub. Tuple: (points, gray value, stroke width)."""
    lines = [([(194, 366), (193, 378), (192, 390)], 105, 2),
             ([(318, 366), (319, 378), (320, 390)], 105, 2),
             ([(224, 384), (205, 394), (190, 406), (180, 417)], 120, 2),
             ([(288, 384), (307, 394), (322, 406), (332, 417)], 120, 2),
             ([(230, 395), (212, 404), (198, 414), (189, 423)], 125, 2),
             ([(282, 395), (300, 404), (314, 414), (323, 423)], 125, 2),
             ([(236, 405), (221, 413), (208, 421), (200, 429)], 130, 2),
             ([(276, 405), (291, 413), (304, 421), (312, 429)], 130, 2),
             ([(180, 417), (174, 423), (176, 429)], 125, 2),
             ([(332, 417), (338, 423), (336, 429)], 125, 2),
             ([(189, 430), (203, 428), (217, 422)], 90, 3),
             ([(323, 430), (309, 428), (295, 422)], 90, 3),
             ([(248, 406), (256, 417), (264, 406)], 120, 2),
             ([(256, 417), (256, 428)], 130, 2)]
    return lines


def main():
    os.makedirs(OUT, exist_ok=True)
    base, erased, mask, mask_model, mask_torso = build_base()
    cv2.imwrite(os.path.join(OUT, "base.png"), base)
    cv2.imwrite(os.path.join(OUT, "erased.png"), erased)
    cv2.imwrite(os.path.join(OUT, "mask.png"), mask)
    cv2.imwrite(os.path.join(OUT, "mask_model.png"), mask_model)
    cv2.imwrite(os.path.join(OUT, "mask_torso.png"), mask_torso)

    # IP reference: the ORIGINAL's own brow+eye crop
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"))
    eye_crop = orig[140:260, 100:280]
    ip_rgb = Image.fromarray(cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB))
    # torso pass IP: the ORIGINAL's own clothes (same artist's
    # pencil-fabric style)
    ip_torso = Image.fromarray(cv2.cvtColor(orig[380:660, 60:460],
                                            cv2.COLOR_BGR2RGB))
    # neck pass IP: the ORIGINAL's own collar rolls (same pencil
    # stroke style the model should imitate). R31 lesson: the crop
    # MUST NOT include the jaw curve (y<375) — the IP-Adapter
    # encoded it as a face and the model painted a SECOND MOUTH
    # on the neck (R28/R31 phantom lips)
    ip_neck = Image.fromarray(cv2.cvtColor(orig[375:475, 150:440],
                                           cv2.COLOR_BGR2RGB))

    erased_rgb = Image.fromarray(erased).convert("RGB")
    mask_pil = Image.fromarray(mask_model)

    g_er = erased.astype(np.float32)
    loc = cv2.GaussianBlur(g_er, (0, 0), 6)
    hp = np.clip(loc - g_er, 0, None)
    hp[hp < 8] = 0
    lines = 255 - np.clip(hp * 2.0, 0, 255).astype(np.uint8)
    lines = _draw_features(lines, for_cond=True)
    cv2.imwrite(os.path.join(OUT, "control_lines.png"), lines)
    ctrl_rgb = Image.fromarray(lines).convert("RGB")

    from diffusers import (ControlNetModel, LCMScheduler,
                           StableDiffusionControlNetInpaintPipeline)
    cn = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype=torch.float32)
    lcm = _find_file(os.path.join(
        r"D:\huggingface_cache",
        "models--latent-consistency--lcm-lora-sdv1-5"),
        "pytorch_lora_weights.safetensors")
    ckpts = {
        "sd": (_find_file(os.path.join(
            r"D:\huggingface_cache",
            "models--runwayml--stable-diffusion-v1-5"),
            "emaonly.safetensors"), True),
        "cf": (_find_file(os.path.join(
            r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
            "_fp16.safetensors"), True),
    }
    which = sys.argv[1:] or ["sd"]
    mask_f = mask.astype(np.float32) / 255.0
    torso_f = mask_torso.astype(np.float32) / 255.0
    mask_torso_pil = Image.fromarray(mask_torso)
    for key in which:
        ckpt, single = ckpts[key]
        if single:
            pipe = StableDiffusionControlNetInpaintPipeline \
                .from_single_file(
                    ckpt, controlnet=cn, torch_dtype=torch.float32,
                    safety_checker=None)
        else:
            pipe = StableDiffusionControlNetInpaintPipeline \
                .from_pretrained(
                    ckpt, controlnet=cn, torch_dtype=torch.float32,
                    safety_checker=None)
        pipe.scheduler = LCMScheduler.from_config(
            pipe.scheduler.config)
        pipe.load_lora_weights(lcm)
        try:
            pipe.fuse_lora()
        except Exception as e:
            print(f"[warn] fuse_lora: {e}", flush=True)
        pipe.set_progress_bar_config(disable=True)
        pipe.load_ip_adapter(
            "h94/IP-Adapter", subfolder="models",
            weight_name="ip-adapter_sd15.safetensors")
        pipe.set_ip_adapter_scale(0.5)
        prompt, neg = PROMPTS[key]
        for seed in (998,):
            t0 = time.time()
            res = pipe(prompt=prompt, negative_prompt=neg,
                       image=erased_rgb, mask_image=mask_pil,
                       control_image=ctrl_rgb,
                       controlnet_conditioning_scale=0.9,
                       ip_adapter_image=ip_rgb,
                       height=664, width=520,
                       num_inference_steps=8, guidance_scale=1.0,
                       strength=0.55,
                       generator=torch.Generator("cpu").manual_seed(
                           seed)).images[0]
            # R24: torso model pass REMOVED — the torso is the
            # original's own remapped pixels now; the model could
            # never match its exact pencil style (R21-23 rejected)
            g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
            g = cv2.resize(g, (base.shape[1], base.shape[0]),
                           interpolation=cv2.INTER_CUBIC)
            # paste back ONLY the eye band: everything else stays
            # bit-identical to the composite
            final = (base.astype(np.float32) * (1 - mask_f)
                     + g.astype(np.float32) * mask_f).astype(np.uint8)

            # R35: neck model pass REMOVED — 8 generations (R27-34)
            # all failed (turtleneck / lapels / snood / phantom
            # lips / weird SD chin / cat-ear humps). The final
            # recipe is 100% classical and deterministic: the
            # original's own neck trapezoid PRESERVED + the
            # original's own collar roll band pasted per side
            # (_remap_torso). The model only ever touches the eye
            # strip.

            tag = f"r47_{key}_s{seed}"
            print(f"[inpaint {tag}] {time.time()-t0:.0f}s",
                  flush=True)

            # R46 (user: 和同学的一点也不像 + 脸和头发没有融合有明显
            # 黑边 + 下巴脖子怪): the classmate's front view is ONE
            # coherent soft pencil drawing — soft shaded skin, NO
            # hard chin line, hair strands overlapping the forehead,
            # zero composite edges. Our collage can never get there
            # by patching. UNIFYING REPAINT: low-strength (0.45)
            # img2img over the whole head (approved feature islands
            # protected) with the anime checkpoint, ControlNet
            # lineart holding the composition, IP-Adapter holding
            # the original's strand style. Two variants:
            #   A = base keeps the original's chin U line
            #   B = no chin line (classmate-style soft chin)
            del pipe
            ckpt_f, _ = ckpts["cf"]
            pipe_f = StableDiffusionControlNetInpaintPipeline \
                .from_single_file(ckpt_f, controlnet=cn,
                                  torch_dtype=torch.float32,
                                  safety_checker=None)
            pipe_f.scheduler = LCMScheduler.from_config(
                pipe_f.scheduler.config)
            pipe_f.load_lora_weights(lcm)
            try:
                pipe_f.fuse_lora()
            except Exception as e:
                print(f"[warn] fuse_lora cf: {e}", flush=True)
            pipe_f.set_progress_bar_config(disable=True)
            pipe_f.load_ip_adapter(
                "h94/IP-Adapter", subfolder="models",
                weight_name="ip-adapter_sd15.safetensors")
            pipe_f.set_ip_adapter_scale(0.5)
            ip_hair = Image.fromarray(cv2.cvtColor(
                orig[0:145, 100:420], cv2.COLOR_BGR2RGB))
            # R47: A's collar top was smeared into a white band
            # (region reached y400 + knot protect box) and the
            # eye-band protect box left STRAIGHT shading edges on
            # the cheeks. Fix: region ends at y375 (the collar roll
            # band stays 100% original pixels); protect ONLY the eye
            # ellipses + brows + nose + mouth so the model itself
            # blends the temples/cheeks/chin/neck with no straight
            # boundary anywhere
            reg = np.zeros(base.shape, np.uint8)
            reg[0:375, 48:464] = 255
            reg = cv2.GaussianBlur(reg, (0, 0), 6)
            keep = np.zeros(base.shape, np.uint8)
            cv2.ellipse(keep, (213, 215), (45, 18), 0, 0, 360,
                        255, -1)
            cv2.ellipse(keep, (299, 215), (45, 18), 0, 0, 360,
                        255, -1)
            for (y0, y1, x0, x1) in ((192, 210, 176, 252),
                                     (192, 210, 260, 336),
                                     (246, 292, 232, 282),
                                     (288, 314, 226, 286)):
                keep[y0:y1, x0:x1] = 255
            keep = cv2.GaussianBlur(keep, (0, 0), 8)
            fm = np.clip(reg.astype(np.float32) / 255.0
                         * (1 - keep.astype(np.float32) / 255.0),
                         0, 1)
            fm_u8 = np.clip(fm * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(OUT, "_mask_fusion.png"), fm_u8)
            PROMPT_F = ("(monochrome:1.4), (pencil sketch:1.4), "
                        "(traditional media:1.2), graphite, fine "
                        "pencil strokes, soft delicate shading, 1boy, "
                        "short black hair, bangs, pale skin, "
                        "(white background:1.3)")
            NEG_F = ("color, colored, photo, realistic, 3d, blurry, "
                     "messy lines, long hair, hat, watermark, text, "
                     "deformed, (turtleneck:1.4), (collar:1.3), "
                     "speckles, gray background, dark neck")
            for vtag, fv in ((f"r47_{key}_s{seed}", final),):
                t1 = time.time()
                gf = fv.astype(np.float32)
                locf = cv2.GaussianBlur(gf, (0, 0), 6)
                hpf = np.clip(locf - gf, 0, None)
                hpf[hpf < 8] = 0
                lnf = 255 - np.clip(hpf * 2.0, 0, 255) \
                    .astype(np.uint8)
                ctrl_f = Image.fromarray(lnf).convert("RGB")
                res = pipe_f(
                    prompt=PROMPT_F, negative_prompt=NEG_F,
                    image=Image.fromarray(fv).convert("RGB"),
                    mask_image=Image.fromarray(fm_u8),
                    control_image=ctrl_f,
                    controlnet_conditioning_scale=0.75,
                    ip_adapter_image=ip_hair,
                    height=664, width=520,
                    num_inference_steps=8, guidance_scale=1.0,
                    strength=0.45,
                    generator=torch.Generator("cpu").manual_seed(seed)
                    ).images[0]
                g2 = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
                g2 = cv2.resize(g2, (fv.shape[1], fv.shape[0]),
                                interpolation=cv2.INTER_CUBIC)
                out = (fv.astype(np.float32) * (1 - fm)
                       + g2.astype(np.float32) * fm).astype(np.uint8)
                # despeckle tiny gray specks the model left on the
                # paper — ONLY inside the fused zone (the protected
                # blush/eye strokes must survive)
                cand = ((out > 130.0) & (out < 251.0)
                        & (fm > 0.3)).astype(np.uint8)
                cand[400:, :] = 0
                n, lab, stats, _ = cv2.connectedComponentsWithStats(
                    cand, 8)
                kill = np.zeros(out.shape, np.uint8)
                for i in range(1, n):
                    if stats[i, cv2.CC_STAT_AREA] < 30:
                        kill[lab == i] = 1
                kf = cv2.GaussianBlur(kill.astype(np.float32),
                                      (0, 0), 1)
                out = np.clip(out * (1 - kf) + 252.0 * kf, 0, 255) \
                    .astype(np.uint8)
                cv2.imwrite(os.path.join(OUT, f"eyes_{vtag}.png"),
                            out)
                crop = out[150:340, 140:390]
                cv2.imwrite(os.path.join(OUT, f"_zoom_{vtag}.png"),
                            cv2.resize(crop, (500, 380),
                                       interpolation=cv2.INTER_CUBIC))
                crop2 = out[280:460, 120:400]
                cv2.imwrite(os.path.join(OUT, f"_neckzoom_{vtag}.png"),
                            cv2.resize(crop2, (560, 360),
                                       interpolation=cv2.INTER_CUBIC))
                print(f"[fusion {vtag}] {time.time()-t1:.0f}s",
                      flush=True)
            del pipe_f
        if 'pipe' in dir():
            del pipe


if __name__ == "__main__":
    main()
