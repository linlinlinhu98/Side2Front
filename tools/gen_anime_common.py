"""动漫生图公共库: 13 段重放链 + gen_frontal_ref 共用的配方常量与装载函数.

由原 gen_r83 / gen_r101 / gen_r102 / gen_cn_unify 四个实验脚本中的
共享部分合并而来(代码逐字保留, 保证 pipeline_final 重放 BIT-IDENTICAL):
  - TRIGGER / LORA_DIR / CX / _scale_lora   (原 gen_r83)
  - STEPS = 26                              (原 gen_r102)
  - _lineart                                (原 gen_cn_unify)
  - _load_dpm                               (原 gen_r101)
"""
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "tools"))

LORA_DIR = os.path.join(ROOT, "assets", "lora", "s2fstyle")
TRIGGER = "s2fstyle"
CX = 258

STEPS = 26


def _scale_lora(unet, factor):
    for m in unet.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict) and "default" in sc:
            sc["default"] = sc["default"] * factor


def _lineart(img):
    g = cv2.GaussianBlur(img, (0, 0), 3)
    hp = np.clip(g.astype(np.float32) - img.astype(np.float32),
                 0, None)
    return 255 - np.clip(hp * 2.0, 0, 255).astype(np.uint8)


def _load_dpm(lcm_unused, ckpt, cn_path):
    from diffusers import (ControlNetModel,
                           DPMSolverMultistepScheduler,
                           StableDiffusionControlNetInpaintPipeline)
    from peft import PeftModel
    cn = ControlNetModel.from_pretrained(cn_path,
                                         torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetInpaintPipeline \
        .from_single_file(ckpt, controlnet=cn,
                          torch_dtype=torch.float32,
                          safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
    pipe.set_progress_bar_config(disable=True)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models",
        weight_name="ip-adapter-plus_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.9)
    return pipe
