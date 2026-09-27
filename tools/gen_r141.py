"""R141 (user: 发色又变了 + 你就不能每次只改一处然后固定
其他特征不变):

Hair drift measured: mean 92 (genr119_s7) -> 136
(genr140_s7) - every pass bleaches the hair.

NEW PROTOCOL (user's demand: change ONE thing, keep
everything else EXACTLY):
1. masked inpaint ONLY the hair region (init darkened
   x0.62, 'black hair' prompt, strength 0.50)
2. composite ONLY the feathered hair mask back onto the
   base - every other pixel bit-identical (verified)
DPM-26, LoRA 0.6. Base: genr140_s7. 2 seeds.
"""
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from trial_inpaint_eyes import _find_file, OUT  # noqa: E402
from gen_cn_unify import _lineart  # noqa: E402
from gen_r83 import LORA_DIR, TRIGGER, CX, _scale_lora  # noqa: E402
from gen_r101 import _load_dpm  # noqa: E402
from gen_r102 import STEPS  # noqa: E402
from gen_r97 import _hair_mask  # noqa: E402

HAIR_PROMPT = (f"{TRIGGER}, black hair, dark hair, "
               "short dark hair, fine hair strands, "
               "pencil sketch, monochrome, white "
               "background")
HAIR_NEG = ("light hair, gray hair, white hair, "
            "blonde, color, colored, lowres, blurry, "
            "watermark")


def _composite(base, result, mask):
    """paste ONLY the feathered masked region of result
    onto base; everything else bit-identical"""
    a = mask.astype(np.float32) / 255.0
    out = base.astype(np.float32) * (1 - a) \
        + result.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    base = cv2.imread(os.path.join(OUT, "genr140_s7.png"),
                      0)
    assert base is not None
    H, W = base.shape
    orig = cv2.imread(os.path.join(ROOT, "4", "face.png"),
                      cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    refs = [Image.fromarray(rgb), Image.fromarray(rgb[:400, :])]
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")

    pipe = _load_dpm(None, ckpt,
                     "lllyasviel/control_v11p_sd15_lineart")
    _scale_lora(pipe.unet, 0.6)
    pipe.set_ip_adapter_scale(0.7)
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)],
                        dim=0)

    hmask = _hair_mask((H, W))
    init = base.astype(np.float32)
    a = (hmask.astype(np.float32) / 255.0)
    dark = init * 0.62
    hinit = np.clip(init * (1 - a) + dark * a, 0, 255) \
        .astype(np.uint8)
    guide = _lineart(base)

    for seed in (7, 42):
        t0 = time.time()
        res = pipe(prompt=HAIR_PROMPT,
                   negative_prompt=HAIR_NEG,
                   image=Image.fromarray(hinit).convert("RGB"),
                   mask_image=Image.fromarray(hmask)
                   .convert("RGB"),
                   control_image=Image.fromarray(guide)
                   .convert("RGB"),
                   controlnet_conditioning_scale=0.85,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=520,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=0.50,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        # composite ONLY the hair region back
        final = _composite(base, g, hmask)
        name = f"genr141_s{seed}.png"
        cv2.imwrite(os.path.join(OUT, name), final)
        # verify non-masked area is unchanged
        nomask = hmask == 0
        unchanged = (final[nomask] == base[nomask]).mean()
        hair_mean = final[60:200, 150:370].mean()
        print(f"[r141 {name}] {time.time() - t0:.0f}s "
              f"unchanged_outside={unchanged:.3f} "
              f"hair_mean={hair_mean:.0f}", flush=True)

    # comparison sheet
    panels = [("base(136)", base)]
    for seed in (7, 42):
        im = cv2.imread(os.path.join(OUT, f"genr141_s{seed}.png"), 0)
        panels.append((f"s{seed}", im))
    n = len(panels)
    hz = [cv2.resize(im[30:280, 80:440], None, fx=1.1, fy=1.1,
                     interpolation=cv2.INTER_CUBIC)
          for _, im in panels]
    hh, hw = hz[0].shape
    sheet = np.full((hh, hw * n + 8 * (n - 1)), 255, np.uint8)
    x = 0
    for (label, _), t in zip(panels, hz):
        t = t.copy()
        cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + hw] = t
        x += hw + 8
    cv2.imwrite(os.path.join(OUT, "_genr141_hair.png"), sheet)
    print("[r141] sheets saved", flush=True)


if __name__ == "__main__":
    main()
