r"""通用离线正脸参考生成器（任意输入图 -> 正面候选参考）。

对任意动漫侧脸图，复用 R119 配方（Counterfeit-V3.0 + DPM++ 2M
Karras 26 步 + s2fstyle LoRA + IP-Adapter Plus 身份保持 +
ControlNet lineart）做一次 img2img：以输入图自身为底、提示词
翻转到 front view，IP-Adapter 从输入图全图/头肩/躯干三路取身份。

这是"把数据传给生图模型"的通用通道——GUI 运行时本身不调用
生图模型；本脚本离线跑出的候选图可注册进 manual_keypoints.json
的 references 表，供 GUI 参考移植通道使用。

用法（从 repo root；需 SD 模型缓存约 16G，见 PIPELINE.md）:
    set HF_HOME=D:\huggingface_cache
    set HF_HUB_OFFLINE=1
    python tools/gen_frontal_ref.py <输入图片> [选项]

选项:
    --seeds 7,42      候选种子（默认 7,42，与主链一致）
    --strength 0.7    img2img 强度（越高越脱离原图构图）
    --lora 0.8        s2fstyle LoRA 强度（0 关闭）
    --ip 0.85         IP-Adapter 身份保持强度
    --out-dir DIR     输出目录（默认 result/ref_gen_custom）

产出: <out-dir>/frontal_s<seed>.png + _frontal_sheet.png 对比图。
后续步骤（人工挑图 -> 精标 16 关键点 -> 注册 references 表）
见 README.md "为一张新图启用生图参考"。
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from trial_inpaint_eyes import _find_file  # noqa: E402
from gen_cn_unify import _lineart  # noqa: E402
from gen_r83 import TRIGGER, _scale_lora  # noqa: E402
from gen_r101 import _load_dpm  # noqa: E402
from gen_r102 import STEPS  # noqa: E402

PROMPT = (f"{TRIGGER}, 1person, front view, facing the viewer, "
          "symmetrical face, hand drawn, pencil sketch, "
          "line weight variation, subtle grey shading, "
          "monochrome, white background")
NEG = ("profile, side view, three-quarter view, looking away, "
       "colored, 3d render, deformed, extra limbs, watermark")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--seeds", default="7,42")
    ap.add_argument("--strength", type=float, default=0.7)
    ap.add_argument("--lora", type=float, default=0.8)
    ap.add_argument("--ip", type=float, default=0.85)
    ap.add_argument("--out-dir", default=os.path.join(
        ROOT, "result", "ref_gen_custom"))
    args = ap.parse_args()

    src = cv2.imread(args.input, cv2.IMREAD_COLOR)
    assert src is not None, f"读不到输入图: {args.input}"
    h, w = src.shape[:2]
    # 生成画布: 高 664，宽按原比例取 8 的倍数（SD1.5 友好）
    W = int(np.clip(round(w * 664 / h / 8) * 8, 320, 640))
    base = cv2.resize(src, (W, 664), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)

    seeds = tuple(int(x) for x in args.seeds.split(",") if x)
    ckpt = _find_file(os.path.join(
        r"D:\huggingface_cache", "models--gsdf--Counterfeit-V3.0"),
        "_fp16.safetensors")
    pipe = _load_dpm(None, ckpt,
                     "lllyasviel/control_v11p_sd15_lineart")
    if args.lora > 0:
        _scale_lora(pipe.unet, args.lora)
    pipe.set_ip_adapter_scale(args.ip)
    # IP-Adapter 三路身份参考: 全图 / 头肩 / 躯干（与主链一致）
    refs = [Image.fromarray(rgb),
            Image.fromarray(rgb[:rgb.shape[0] // 2 + 100, :]),
            Image.fromarray(rgb[rgb.shape[0] // 2:, :])]
    with torch.no_grad():
        emb, unc = pipe.encode_image(refs, "cpu", 1,
                                     output_hidden_states=True)
        emb = torch.cat([unc.unsqueeze(0), emb.unsqueeze(0)], dim=0)

    ctrl = Image.fromarray(
        _lineart(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))).convert("RGB")
    full_mask = Image.fromarray(
        np.full((664, W), 255, np.uint8)).convert("RGB")

    os.makedirs(args.out_dir, exist_ok=True)
    panels = [("input", cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))]
    for seed in seeds:
        t0 = time.time()
        res = pipe(prompt=PROMPT, negative_prompt=NEG,
                   image=Image.fromarray(rgb).convert("RGB"),
                   mask_image=full_mask,
                   control_image=ctrl,
                   controlnet_conditioning_scale=0.7,
                   ip_adapter_image_embeds=[emb],
                   height=664, width=W,
                   num_inference_steps=STEPS,
                   guidance_scale=3.5, strength=args.strength,
                   generator=torch.Generator("cpu")
                   .manual_seed(seed)).images[0]
        g = cv2.cvtColor(np.asarray(res), cv2.COLOR_RGB2GRAY)
        name = f"frontal_s{seed}.png"
        out = os.path.join(args.out_dir, name)
        cv2.imwrite(out, g)
        panels.append((f"s{seed}", g))
        print(f"[frontal {name}] {time.time() - t0:.0f}s",
              flush=True)

    n = len(panels)
    sheet = np.full((664, W * n + 10 * (n - 1)), 255, np.uint8)
    x = 0
    for label, im in panels:
        p = im.copy()
        cv2.putText(p, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 0, 2, cv2.LINE_AA)
        sheet[:, x:x + W] = p
        x += W + 10
    cv2.imwrite(os.path.join(args.out_dir, "_frontal_sheet.png"),
                sheet)
    print("对比图:", os.path.join(args.out_dir,
                                  "_frontal_sheet.png"), flush=True)
    print("下一步: 挑一张 -> 精标关键点 -> 注册 manual_keypoints.json"
          " references 表（见 README.md）", flush=True)


if __name__ == "__main__":
    main()
