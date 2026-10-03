"""R149 (user review fix #1: the dangling line LEFT of the
zipper - a half-drawn drawstring inherited from the REF's
hoodie cords; the original is a pure zip-up, NO cords):

Thin-line removal: mask = the dark pixels of the dangling
line only (dilated+feathered), whiten them in init AND
guide so neither img2img nor ControlNet can redraw it,
inpaint with a clean-fabric prompt, composite back only
the thin mask (everything else bit-identical).
Zone measured: x=197..215 line body, y=477..613; removal
zone x=190..245, y=474..625 (collar edge above y~470 is
protected). Base: genr148_s42. DPM-26, LoRA 0.8, CN 1.0,
strength 0.70. 2 seeds.
"""
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from trial_inpaint_eyes import _find_file, OUT  # noqa: E402
from gen_anime_common import CX, LORA_DIR, STEPS, TRIGGER, _lineart, _load_dpm, _scale_lora  # noqa: E402


def _seeds():
    return tuple(int(x) for x in
                 os.environ.get("S2F_SEEDS", "7,42").split(",")
                 if x)

PROMPT = (f"{TRIGGER}, clean fabric, smooth cloth, subtle "
          "fabric folds, zip-up hoodie, pencil sketch, "
          "monochrome, white background")
NEG = ("drawstring, cord, hanging string, rope, strap, "
       "dangling line, scribble, color, colored, lowres, "
       "blurry, watermark")

# removal zone (x0, x1, y0, y1) - tight around the cord,
# starts just below the collar edge
ZONE = (190, 245, 474, 625)


def _line_mask(base):
    """Dark pixels of the cord only, dilated + feathered."""
    x0, x1, y0, y1 = ZONE
    m = np.zeros(base.shape, np.uint8)
    z = base[y0:y1, x0:x1]
    core = (z < 160).astype(np.uint8) * 255
    core = cv2.dilate(core, np.ones((7, 7), np.uint8))
    m[y0:y1, x0:x1] = core
    return cv2.GaussianBlur(m, (0, 0), 5)


def _composite(base, result, mask):
    a = mask.astype(np.float32) / 255.0
    out = base.astype(np.float32) * (1 - a) \
        + result.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    base = cv2.imread(os.path.join(OUT, "genr148_s42.png"),
                      0)
    assert base is not None
    H, W = base.shape
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[280:, :])]
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    pipe = _load_dpm(None, ckpt,
                     "lllyasviel/control_v11p_sd15_lineart")
    _scale_lora(pipe.unet, 0.8)
    pipe.set_ip_adapter_scale(0.8)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    mask = _line_mask(base)
    # erase the cord from the guide too (else CN redraws it)
    guide = _lineart(base)
    guide[mask > 60] = 255
    # whiten the cord in init under the mask core
    init = base.copy()
    init[mask > 120] = 255
    cv2.imwrite(os.path.join(OUT, "_r149_guide.png"), guide)

    n_dark_before = int((base[474:625, 190:245] < 160).sum())
    print(f"[r149] cord dark px before={n_dark_before}",
          flush=True)
    for seed in _seeds():
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=Image.fromarray(init).convert("RGB"),
                   mask_image=Image.fromarray(mask)
                   .convert("RGB"),
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=1.0,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.70,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        final = _composite(base, g, mask)
        name = f"genr149_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), final)
        unchanged = (final[mask == 0] == base[mask == 0]).mean()
        n_dark = int((final[474:625, 190:245] < 160).sum())
        print(f"[r149 {name}] {time.time() - t0:.0f}s "
              f"unchanged_outside={unchanged:.3f} "
              f"dark_after={n_dark}", flush=True)

    panels = [("base", base)]
    for seed in _seeds():
        im = cv2.imread(os.path.join(OUT, f"genr149_s{seed}.png"), 0)
        panels.append((f"s{seed}", im))
    n = len(panels)
    cz = [cv2.resize(im[440:640, 150:300], None, fx=3.0, fy=3.0,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    ch, cw = cz[0].shape
    sheet = np.full((ch, cw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, cz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + cw] = t
        x += cw + 8
    cv2.imwrite(os.path.join(OUT, "_genr149_cord.png"), sheet)
    print("[r149] sheets saved", flush=True)


if __name__ == "__main__":
    main()
