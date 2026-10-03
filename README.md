# Side2Front 侧脸转正脸系统

交互式侧脸→正脸图像编辑器（PySide6 GUI，纯 CPU、运行时完全离线）。

采用**两层路线**：

1. **离线生图层**（时间不计入运行）：用 Stable Diffusion（Counterfeit-V3.0 + IP-Adapter Plus + ControlNet lineart + 风格 LoRA，DPM++ 2M Karras 26 步）离线预生成高质量正面参考图；
2. **运行时经典层**：GUI 不调用生图模型，只做轻量几何处理——TPS 薄板样条移植 / 3DMM / GAN 正面化，秒级响应。

对竞赛测试图 `4/face.png`（动漫少年侧脸），13 段 SD 流水线的定稿结果
（CLIP 复合相似度 **0.762**，whole 0.853 / hair 0.774 / face 0.704 / cloth 0.717）
已直接集成进 GUI：加载该图即刻显示定稿正面图。

## 运行时各模式走什么管线

| 输入类型 | 运行时管线 | 是否用生图模型 |
|---|---|---|
| `4/face.png`（已注册） | 定稿直出（离线 SD 流水线成品） | 成品来自离线 SD |
| 其他动漫图 | 关键点 + TPS 变形 + 镜像补全 + 参考合成 | 否（可配离线生成的参考图） |
| 真人人脸 | GAN 正面化（主）/ 3DDFA_V2 3DMM（备） | 否 |
| 动物脸 | DINOv2 特征迁移 | 否 |

## 安装

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 真人人脸模式（可选）：克隆 3DDFA_V2
git clone https://github.com/cleardusk/3DDFA_V2.git 3ddfa_v2
cd 3ddfa_v2 && sh ./build.sh && cd ..
```

**GUI 运行时模型**（首次运行自动下载，共约 380MB，缓存到 `HF_HOME` 指向的目录）：

| 模型 | 用途 | 大小 | 下载 |
|---|---|---|---|
| anime-face-detector yolov3 + hrnetv2 | 动漫人脸/关键点检测 | 235M + 38M | 自动（[hysts/anime-face-detector-yolov3](https://huggingface.co/hysts/anime-face-detector-yolov3) / [hrnetv2](https://huggingface.co/hysts/anime-face-detector-hrnetv2)） |
| face-frontalization | 真人脸 GAN 正面化 | 22M | 自动（[opetrova/face-frontalization](https://huggingface.co/spaces/opetrova/face-frontalization)） |
| DINOv2 ViT-S/14 | 动物特征迁移 | 84M | 自动（[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)，权重 [直链](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth)） |

## 离线生图层模型清单（全部模型与下载链接）

GUI 本身只依赖上表；要用离线生图脚本（`tools/gen_rnr.py` 真人 /
`tools/gen_frontal_ref.py` 动漫 / `tools/gen_frontal_refs.py` 动物、
`tools/pipeline_final.py` 一键重放）还需以下模型。

**一键下载**（HuggingFace 部分 + CodeFormer 权重，共约 12GB）：

```bash
HF_ENDPOINT=https://hf-mirror.com python tools/download_models.py
```

**需要单独 git clone 的代码仓库**（含推理代码，放进 `HF_HOME` 目录）：

```bash
git clone https://github.com/cleardusk/3DDFA_V2.git 3ddfa_v2    # 仓库根目录
cd 3ddfa_v2 && sh ./build.sh && cd ..                            # 含 68 点权重下载
git clone https://github.com/foivospar/Arc2Face        "$HF_HOME/arc2face_code"
git clone https://github.com/Hangz-nju-cuhk/Rotate-and-Render "$HF_HOME/rnr_code"
```

CodeFormer 推理代码为内置的最小 vendored 版（`HF_HOME/codeformer_code/`
下只需 `basicsr/archs/codeformer_arch.py` 与 `vqgan_arch.py`，
取自 [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer)）；权重
`codeformer.pth` 由上面的下载脚本自动从
[GitHub Release](https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth)
获取。RnR 的 GAN 权重**不需要下载**（真人管线自动回退到 Arc2Face 路径）。

**HuggingFace 模型明细**：

| # | 模型 | 用途 | 大小 | 链接 | 管线 |
|---|---|---|---|---|---|
| 1 | SD1.5（v1-5-pruned-emaonly.safetensors） | 真人/动物生图底模 | 4.3G | https://huggingface.co/runwayml/stable-diffusion-v1-5 | 真人、动物 |
| 2 | Counterfeit-V3.0（fp16） | 动漫生图底模 | 2.1G | https://huggingface.co/gsdf/Counterfeit-V3.0 | 动漫 |
| 3 | Arc2Face（unet + arcface.onnx + ref_adapter） | 真人身份注入 + 相似度评分 | ~4G | https://huggingface.co/FoivosPar/Arc2Face | 真人 |
| 4 | IP-Adapter（image_encoder + ip-adapter_sd15） | 身份/参考保持 | ~1.8G | https://huggingface.co/h94/IP-Adapter | 真人、动漫、动物 |
| 5 | ControlNet lineart | 线稿结构控制 | 1.4G | https://huggingface.co/lllyasviel/control_v11p_sd15_lineart | 动漫 |
| 6 | Anything-V5 | 动漫生图（脚本其他实验模式） | 2.1G | https://huggingface.co/stablediffusionapi/anything-v5 | 可选 |
| 7 | CodeFormer codeformer.pth | 人脸修复（w=0.8） | 376M | https://github.com/sczhou/CodeFormer/releases | 真人 |

另外：
- **s2fstyle LoRA**（动漫风格）已随仓库分发：`assets/lora/s2fstyle/`，无需下载
- 首次运行请**不要**设置 `HF_HUB_OFFLINE=1`（要联网下载）；下完后设上即可完全离线

```bash
# Windows 示例
set HF_HOME=D:\huggingface_cache
python main.py
```

## 运行

```bash
python main.py
```

加载 `4/face.png` → 自动命中注册的关键点与定稿参考 → 直接输出正面图。
其他图片按上表走对应管线，也可在 GUI 里手动标注关键点。

## 为一张新图启用生图参考（GUI → 生图模型的数据通道）

GUI 运行时不调用生图模型，但提供**离线生成脚本**把任意输入图的数据
（全图 + 头肩 + 躯干三路身份参考）传给 SD：

```bash
set HF_HOME=D:\huggingface_cache & set HF_HUB_OFFLINE=1
python tools/gen_frontal_ref.py <你的侧脸图> --seeds 7,42
# 产出 result/ref_gen_custom/frontal_s7.png、frontal_s42.png + 对比图
```

随后（见 `manual_keypoints.json` 的 `references` 表结构）：
挑一张候选 → 放大精标 16 关键点 → 按输入图内容 md5 注册 → GUI 里
该图即可走"参考移植"通道。

## 复现 13 段 SD 流水线（离线，约 1.5-2 小时）

完整生成配方、逐段说明与一键重放见 [PIPELINE.md](PIPELINE.md)：

```bash
python tools/pipeline_final.py            # 断点续跑
python tools/pipeline_final.py --force    # 从零重放
```

需要上表"离线生图层模型清单"的全部模型（约 12GB 下载 + 代码仓库
clone），重放结果与归档定稿**逐像素一致**。

## 项目结构

```
Side2Front/
├── main.py                    # 入口
├── manual_keypoints.json      # 人工精标关键点 + 正面参考注册表（按 md5）
├── PIPELINE.md                # 13 段 SD 流水线复现文档
├── src/
│   ├── core/                  # 各 frontalizer（anime / real / gan / animal）
│   └── gui/                   # PySide6 主窗口、图像查看、控制面板
├── tools/                     # 离线工具：13 段生成链、一键重放、通用生成器、相似度
├── result/final_frontal.png   # face.png 的定稿正面图（GUI 直接引用）
├── assets/                    # LoRA、参考锚图、参考注册辅助数据
├── 4/                         # 测试图（face.png / cat.jpg / people.jpg）
└── 3ddfa_v2/                  # 第三方，需单独 clone（真实人脸备选路径，可选）
```

## 环境说明

- Python 3.8+，CPU 即可（运行时不需要 GPU）
- 运行时离线：GUI 与推理不访问网络（`HF_HUB_OFFLINE=1` 推荐）
- Windows / macOS / Linux
