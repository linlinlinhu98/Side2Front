r"""Arc2Face 真实人脸正脸生成（用户 2026-09-27 批准下载并集成）。

流程（对应官方 gradio_demo/app.py，无 insightface 依赖）:
  1. 3DDFA_V2 检测人脸框 + 回归 68 关键点（仓库已装好）
  2. 由 68 点导出 ArcFace 5 点（左右眼中心/鼻尖/嘴角两端）
  3. 5 点相似变换对齐到 112x112 模板 -> arcface.onnx (onnxruntime)
     取 512 维身份向量，L2 归一化
  4. Arc2Face 管线（官方权重 + SD1.5 底模）:
     project_face_embs 把身份向量编码进 "photo of a id person"
     -> DPMSolver 25 步, guidance 3.0, 512x512 正脸

用法（repo root）:
    set HF_HOME=D:\huggingface_cache
    set HF_HUB_OFFLINE=1
    python tools/gen_arc2face.py <输入照片> [--seeds 0,42] [--out-dir DIR]

产出: <out-dir>/arc2face_s<seed>.png
"""
import argparse
import os
_HFC = os.environ.get("HF_HOME", r"D:\huggingface_cache")  # 模型缓存根目录, 可用环境变量 HF_HOME 覆盖
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_3DDFA_ROOT = os.path.join(ROOT, "3ddfa_v2")
sys.path.insert(0, _3DDFA_ROOT)

ARCFACE_DST = np.array(  # insightface antelopev2 112x112 标准 5 点模板
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def _landmarks_68(image):
    """3DDFA_V2: 人脸框 + 68 关键点（图像坐标）。"""
    import yaml
    from FaceBoxes.FaceBoxes import FaceBoxes
    from TDDFA import TDDFA

    cfg_path = os.path.join(_3DDFA_ROOT, "configs", "mb1_120x120.yml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["gpu_mode"] = False
    if cfg.get("bfm_fp", "").startswith("configs/"):
        cfg["bfm_fp"] = os.path.normpath(
            os.path.join(_3DDFA_ROOT, cfg["bfm_fp"]))
    if cfg.get("checkpoint_fp", "").startswith("weights/"):
        cfg["checkpoint_fp"] = os.path.normpath(
            os.path.join(_3DDFA_ROOT, cfg["checkpoint_fp"]))
    tddfa = TDDFA(**cfg)
    boxes = FaceBoxes()(image)
    assert len(boxes), "3DDFA 未检测到人脸"
    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    import torch
    with torch.no_grad():
        param_lst, roi_box_lst = tddfa(image, [box])
    ver = tddfa.recon_vers(param_lst, roi_box_lst,
                           dense_flag=False)[0]  # (3,68)
    return ver.T[:, :2].astype(np.float32)  # (68,2) 图像 xy


def _five_points(lm68):
    """68 点 -> ArcFace 5 点（与 insightface 模板点序一致）。"""
    return np.stack([
        lm68[36:42].mean(axis=0),   # 左眼中心（图像左侧）
        lm68[42:48].mean(axis=0),   # 右眼中心
        lm68[30],                   # 鼻尖
        lm68[48],                   # 嘴角左
        lm68[54],                   # 嘴角右
    ]).astype(np.float32)


NEG_TEXT = ("side view, profile view, looking away, head turned away, "
            "distorted face, asymmetrical eyes, deformed features")


def _id_embedding(image, onnx_path, dbg_dir):
    """对齐 112x112 -> arcface.onnx -> 512 维归一化身份向量。
    镜像 TTA: 原图 + 水平翻转各提一次向量取平均（识别任务标准
    增强），压掉单次对齐的随机偏差。"""
    import onnxruntime as ort

    def embed(aligned):
        x = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = ((x - 127.5) / 127.5).transpose(2, 0, 1)[None]
        sess = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"])
        e = sess.run(None, {sess.get_inputs()[0].name: x})[0].reshape(-1)
        return e / np.linalg.norm(e)

    lm68 = _landmarks_68(image)
    src = _five_points(lm68)
    M, _ = cv2.estimateAffinePartial2D(src, ARCFACE_DST, method=cv2.LMEDS)
    aligned = cv2.warpAffine(image, M, (112, 112),
                             flags=cv2.INTER_LINEAR,
                             borderValue=(114, 114, 114))
    cv2.imwrite(os.path.join(dbg_dir, "_aligned_112.png"), aligned)

    # 翻转视角: 关键点 x 镜像后, 与镜像模板按左右互换对应
    dst_m = ARCFACE_DST.copy()
    dst_m[:, 0] = 112 - dst_m[:, 0]
    dst_m = dst_m[[1, 0, 2, 4, 3]]
    src_f = src.copy()
    src_f[:, 0] = image.shape[1] - src_f[:, 0]
    M_f, _ = cv2.estimateAffinePartial2D(src_f, dst_m, method=cv2.LMEDS)
    aligned_f = cv2.warpAffine(image, M_f, (112, 112),
                               flags=cv2.INTER_LINEAR,
                               borderValue=(114, 114, 114))

    e1, e2 = embed(aligned), embed(aligned_f)
    emb = (e1 + e2)
    emb = emb / np.linalg.norm(emb)
    print(f"[id emb] TTA(原图+镜像) 两向量余弦 {float(e1 @ e2):.3f}",
          flush=True)
    return torch.tensor(emb, dtype=torch.float32)[None], aligned, lm68


def _encode_text(pipe, text):
    """用 Arc2Face 的 CLIP 编码器编码普通文本（负向提示词用）。"""
    input_ids = pipe.tokenizer(text, truncation=True, padding="max_length",
                               max_length=pipe.tokenizer.model_max_length,
                               return_tensors="pt").input_ids
    enc = pipe.text_encoder
    hidden = enc.embeddings(input_ids=input_ids)
    from transformers.masking_utils import create_causal_mask
    mask = create_causal_mask(config=enc.config, inputs_embeds=hidden,
                              attention_mask=None, past_key_values=None)
    out = enc.encoder(inputs_embeds=hidden, attention_mask=mask,
                      is_causal=True)
    return enc.final_layer_norm(out.last_hidden_state)


def _run_with_ref(pipe, id_emb, image, lm68, out_dir, ref_scale=1.0,
                  guidance=3.0):
    """Reference Adapter 模式（官方 app_exp_adapter.py 的精简版，
    去掉表情适配器）：参考图像素过参考 UNet 写入特征库，去噪时读取，
    从而把外观/颜色/背景直接带进生成过程。参考图默认是原图，
    也可用 --ref-image 指定正脸参考（避免侧脸姿态泄漏）。
    返回一个生成闭包（含 update/clear 生命周期）。"""
    from arc2face import ReferenceAdapter
    from arc2face.utils import image_align
    from diffusers import UNet2DConditionModel

    a2f = _snapshot(_HFC,
                    "models--FoivosPar--Arc2Face")
    lora_path = os.path.join(
        a2f, "ref_adapter", "pytorch_lora_weights.safetensors")
    pipe.load_lora_weights(lora_path, adapter_name="ref")
    pipe.set_adapters("ref", ref_scale)

    ref_unet = UNet2DConditionModel.from_pretrained(
        a2f, subfolder="arc2face", torch_dtype=torch.float32).to("cpu")
    ref_w = ReferenceAdapter(ref_unet, mode="write")
    ref_r = ReferenceAdapter(pipe.unet, mode="read", cfg=True)

    neg = _encode_text(pipe, NEG_TEXT)  # 负向: 把侧脸姿态往外推

    def generate(seed):
        # FFHQ 对齐到 512（官方 reference 用同款对齐）
        img512 = image_align(Image.fromarray(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            lm68, output_size=512)
        img512.save(os.path.join(out_dir, "_ref_512.png"))
        ref_img = (torch.tensor(np.array(img512), dtype=torch.float32)
                   .permute(2, 0, 1) / 255) * 2 - 1
        ref_img = torch.stack([ref_img, ref_img])  # batch 2 (uc+cond)
        ref_img = pipe.vae.encode(ref_img).latent_dist.sample()
        ref_img = ref_img * pipe.vae.config.scaling_factor
        enc = torch.cat([id_emb, id_emb], dim=0)
        ref_unet(ref_img, torch.zeros(2, dtype=torch.long), enc,
                 return_dict=False)
        ref_r.update(ref_w)
        try:
            img = pipe(prompt_embeds=id_emb,
                       negative_prompt_embeds=neg,
                       num_inference_steps=25, guidance_scale=guidance,
                       generator=torch.Generator("cpu")
                       .manual_seed(seed)).images[0]
        finally:
            ref_r.clear()
            ref_w.clear()
        return img

    return generate


def _project_face_embs(pipe, face_embs):
    """transformers 5.x 原生版 project_face_embs（官方实现依赖 4.x 的
    text_model 包装层，5.17 已移除——语义完全一致）。

    把归一化身份向量写入 "photo of a id person" 的 id token 位置，
    走完整个 CLIP text encoder 得到 prompt_embeds。
    """
    from transformers.masking_utils import create_causal_mask

    tok = pipe.tokenizer
    token_id = tok.encode("id", add_special_tokens=False)[0]
    input_ids = tok("photo of a id person", truncation=True,
                    padding="max_length", max_length=tok.model_max_length,
                    return_tensors="pt").input_ids
    enc = pipe.text_encoder
    token_embs = enc.embeddings.token_embedding(input_ids)
    padded = torch.nn.functional.pad(
        face_embs, (0, enc.config.hidden_size - 512))
    token_embs[input_ids == token_id] = padded
    hidden = enc.embeddings(input_ids=input_ids, inputs_embeds=token_embs)
    mask = create_causal_mask(config=enc.config, inputs_embeds=hidden,
                              attention_mask=None, past_key_values=None)
    out = enc.encoder(inputs_embeds=hidden, attention_mask=mask,
                      is_causal=True)
    return enc.final_layer_norm(out.last_hidden_state)


def _attach_ip_adapter(pipe, scale=0.3):
    """IP-Adapter: 把原图人脸作为图像提示注入去噪（身份/肤色/质感与
    Arc2Face id 双路并行——90° 侧脸提的 ArcFace 向量先天弱, TTA 余弦
    仅 0.141）。权重全部本地 (h94/IP-Adapter), 零下载零新包。
    图像编码器 (ViT-H ~2.5GB) 沿用逐张量 mmap 灌法, 并在调用方算完
    图像嵌入后随 text_encoder 一起释放, 不背着进去噪循环。"""
    import gc  # noqa: E402
    import json  # noqa: E402

    from safetensors import safe_open  # noqa: E402
    from transformers import (CLIPVisionConfig,  # noqa: E402
                              CLIPVisionModelWithProjection)

    ip = _snapshot(_HFC, "models--h94--IP-Adapter")
    enc_dir = os.path.join(ip, "models", "image_encoder")
    with open(os.path.join(enc_dir, "config.json")) as f:
        cfg = json.load(f)
    # config 里写着 torch_dtype=float16: transformers 5.x 会按它构建,
    # 且只染 vision_model 不染 visual_projection, 混合型 dtype 在
    # 投影层直接 RuntimeError (实测)。CPU 上强制全 fp32。
    cfg["torch_dtype"] = "float32"
    image_encoder = CLIPVisionModelWithProjection(CLIPVisionConfig(**cfg))
    # 缓冲 IO 读整个编码器 (mmap 页入在内存压力下段错误, 与 VAE 同因)
    sd = _read_st_slice(os.path.join(enc_dir, "model.safetensors"), "")
    live = image_encoder.state_dict()
    for k in live:
        if k in sd:
            live[k].copy_(sd[k])
    missing = [k for k in live if k not in sd]
    if missing:
        # position_ids 类 buffer 允许不在权重文件里(初始化值即正确)
        print(f"[ip-adapter] 编码器 {len(missing)} 个键用初始化值: "
              f"{missing[:3]}", flush=True)
    del sd
    image_encoder.eval()
    del live
    gc.collect()
    # 先注册编码器再 load: load_ip_adapter 见 image_encoder 非 None
    # 会跳过它自己的 from_pretrained (2.5GB 物化+拷贝双份峰值)
    pipe.register_modules(image_encoder=image_encoder)
    pipe.load_ip_adapter(ip, subfolder="models",
                         weight_name="ip-adapter_sd15.safetensors",
                         local_files_only=True)
    pipe.set_ip_adapter_scale(scale)
    print(f"[ip-adapter] 已挂载 sd15 权重 (scale={scale}, "
          f"图像编码器 mmap 逐张量)", flush=True)
    return pipe


def _read_st_slice(path, prefix):
    """顺序缓冲 IO 读 safetensors 切片 (不走 mmap)。
    内存吃紧时 Windows 对 4.3GB ckpt 的 mmap 页入会 access violation
    (2026-10-01 实锤: faulthandler 指向 get_tensor 读 VAE 切片段);
    普通 seek+read 无页入压力。"""
    import json as _json
    import struct
    _DT = {"F64": np.float64, "F32": np.float32, "F16": np.float16,
           "I64": np.int64, "I32": np.int32, "I16": np.int16,
           "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}
    out = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = _json.loads(f.read(n))
        base = 8 + n
        for k, meta in hdr.items():
            if "data_offsets" not in meta:      # __metadata__ 跳过
                continue
            if not k.startswith(prefix):
                continue
            s, e = meta["data_offsets"]
            f.seek(base + s)
            buf = f.read(e - s)
            arr = np.frombuffer(buf, dtype=_DT[meta["dtype"]])
            out[k] = torch.from_numpy(arr.reshape(meta["shape"]).copy())
    return out


def _load_pipeline(ip_scale=0.5):
    """官方 app.py 的管线构造，SD1.5 底模用本地组件拼装。

    内存安全版：不走 from_single_file——diffusers 的 load_state_dict
    会把 4.3GB 单文件 ckpt 整体物化进内存，物理内存吃紧时直接
    MemoryError（2026-09-28 实测）。改为 safe_open 内存映射，只抽取
    first_stage_model.* 的 VAE 切片（~335MB）；unet/text_encoder 本来
    就来自 Arc2Face 自带文件，与 SD1.5 单文件 ckpt 无关。"""
    sys.path.insert(0, os.path.join(_HFC, "arc2face_code"))
    import gc  # noqa: E402
    import json  # noqa: E402

    from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler,  # noqa: E402
                           StableDiffusionPipeline, UNet2DConditionModel)
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (  # noqa: E402
        convert_ldm_vae_checkpoint)
    from safetensors import safe_open  # noqa: E402
    # 不用官方 CLIPTextModelWrapper：其 forward 依赖 transformers 4.x
    # 的 text_model 属性（5.17 已删）。权重去前缀后与 5.x 原生
    # CLIPTextModel 结构完全一致，直接用原生类。
    from transformers import CLIPTextConfig, CLIPTextModel  # noqa: E402

    cache = _HFC
    a2f = _snapshot(cache, "models--FoivosPar--Arc2Face")
    sd15 = _snapshot(cache,
                     "models--stable-diffusion-v1-5--stable-diffusion-v1-5")
    ckpt = os.path.join(
        _snapshot(cache, "models--runwayml--stable-diffusion-v1-5"),
        "v1-5-pruned-emaonly.safetensors")

    # 逐张量 mmap 加载：from_pretrained 默认把 3.4GB 权重字典整体物化
    # 后再 copy_ 进模型，瞬时占两份（6.8GB）；物理内存吃紧时 torch 原生
    # 层分配失败直接 0xC0000005 段错误（2026-09-29 事件日志实锤，5 次
    # 崩溃同一偏移）。这里从 config 建模型后逐张量 copy_，峰值只有模型
    # 本身。注意：_run_with_ref 里的第二个 UNet 仍是 from_pretrained，
    # ref 模式需要更大内存余量。
    with open(os.path.join(a2f, "arc2face", "config.json")) as f:
        unet = UNet2DConditionModel.from_config(json.load(f))
    live = unet.state_dict()
    with safe_open(os.path.join(a2f, "arc2face",
                                "diffusion_pytorch_model.safetensors"),
                   framework="pt") as f:
        disk = set(f.keys())
        assert set(live) == disk, (
            f"unet 键不匹配: missing={len(set(live) - disk)} "
            f"unexpected={len(disk - set(live))}")
        for k in live:
            live[k].copy_(f.get_tensor(k))
    unet.eval()
    del live
    gc.collect()
    print("[unet] 逐张量 mmap 加载 (~3.4GB, 无双倍峰值)", flush=True)

    with open(os.path.join(sd15, "vae", "config.json")) as f:
        vae = AutoencoderKL.from_config(json.load(f))
    vae_ckpt = _read_st_slice(ckpt, "first_stage_model.")
    vae.load_state_dict(convert_ldm_vae_checkpoint(vae_ckpt, vae.config),
                        strict=True)
    vae.eval()
    del vae_ckpt
    gc.collect()
    print("[vae] 从单文件 ckpt 抽取 VAE 切片 (mmap, ~335MB)", flush=True)

    # encoder 放最后加载（UNet/VAE 写入阶段不背这 0.5GB），同样
    # mmap + 逐张量灌入（瞬时只有模型本体 + 页缓存，不再整体物化）。
    # transformers 5.x 重构后 checkpoint 的 text_model.* 键对不上顶层
    # embeddings.*/encoder.*，逐键加前缀取（与旧 remap 严格互逆）。
    with open(os.path.join(a2f, "encoder", "config.json")) as f:
        encoder = CLIPTextModel(CLIPTextConfig(**json.load(f)))
    sd = torch.load(os.path.join(a2f, "encoder", "pytorch_model.bin"),
                    map_location="cpu", weights_only=True, mmap=True)
    live = encoder.state_dict()
    for k in live:
        live[k].copy_(sd["text_model." + k])
    encoder.eval()
    del sd, live
    gc.collect()
    print("[encoder] weights loaded from ckpt (mmap, 逐张量)", flush=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        sd15, vae=vae, unet=unet, text_encoder=encoder,
        torch_dtype=torch.float32, safety_checker=None,
        requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config)
    pipe = pipe.to("cpu")
    # 不再开注意力切片: SD1.5@512 最大注意力分数矩阵仅 1024×1024
    # (~67MB), 切片本就省不到什么; 且 SlicedAttnProcessor 需要构造
    # 参数, load/unload_ip_adapter 的无参 __class__() 会被它噎死
    # (实测 TypeError), 与 IP-Adapter 根本不兼容。
    if ip_scale and ip_scale > 0:
        pipe = _attach_ip_adapter(pipe, ip_scale)
    return pipe


def _snapshot(cache, name):
    d = os.path.join(cache, name, "snapshots")
    subs = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
    assert subs, f"缺快照: {name}"
    return os.path.join(d, subs[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--seeds", default="0,42")
    ap.add_argument("--ref", action="store_true",
                    help="Reference Adapter: 原图像素直通生成过程")
    ap.add_argument("--ref-image", default=None,
                    help="参考图（默认用原图；可传正脸参考避免姿态泄漏）")
    ap.add_argument("--ref-scale", type=float, default=1.0)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--out-dir", default=os.path.join(
        ROOT, "result", "arc2face"))
    args = ap.parse_args()

    global torch
    import torch

    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    assert image is not None, f"读不到输入图: {args.input}"

    a2f = _snapshot(_HFC,
                    "models--FoivosPar--Arc2Face")
    onnx_path = os.path.join(a2f, "arcface.onnx")
    os.makedirs(args.out_dir, exist_ok=True)
    id_emb, aligned, lm68 = _id_embedding(image, onnx_path,
                                          args.out_dir)
    t0 = time.time()
    print(f"[id emb] shape={tuple(id_emb.shape)} "
          f"norm={id_emb.norm().item():.4f} "
          f"({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    pipe = _load_pipeline()
    print(f"[pipeline] {time.time() - t0:.0f}s", flush=True)

    id_emb = _project_face_embs(pipe, id_emb)  # 身份向量 -> prompt_embeds

    os.makedirs(args.out_dir, exist_ok=True)
    if args.ref:
        ref_img = image
        ref_lm = lm68
        if args.ref_image:
            ref_img = cv2.imread(args.ref_image, cv2.IMREAD_COLOR)
            assert ref_img is not None, f"读不到参考图: {args.ref_image}"
            ref_lm = _landmarks_68(ref_img)
            print(f"[ref-image] {args.ref_image}", flush=True)
        gen = _run_with_ref(pipe, id_emb, ref_img, ref_lm, args.out_dir,
                            args.ref_scale, args.guidance)
        for seed in (int(s) for s in args.seeds.split(",") if s):
            t0 = time.time()
            img = gen(seed)
            out = os.path.join(args.out_dir, f"arc2face_ref_s{seed}.png")
            img.save(out)
            print(f"[arc2face+ref s{seed}] {out} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        return

    for seed in (int(s) for s in args.seeds.split(",") if s):
        t0 = time.time()
        img = pipe(prompt_embeds=id_emb,
                   num_inference_steps=25, guidance_scale=3.0,
                   generator=torch.Generator("cpu").manual_seed(seed)
                   ).images[0]
        out = os.path.join(args.out_dir, f"arc2face_s{seed}.png")
        img.save(out)
        print(f"[arc2face s{seed}] {out} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
