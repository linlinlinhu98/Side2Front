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

**模型缓存**（首次运行自动下载，共约 380MB，放到 `HF_HOME` 指向的目录）：

| 模型 | 用途 | 大小 |
|---|---|---|
| `hysts/anime-face-detector`（yolov3 + hrnetv2） | 动漫人脸/关键点检测 | 273M |
| `opetrova/face-frontalization` | 真人脸 GAN 正面化 | 22M |
| `dinov2_vits14_pretrain` | 动物特征迁移 | 84M |

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

需要约 16GB 的 SD 模型缓存（Counterfeit-V3.0、IP-Adapter Plus、
ControlNet lineart、SD1.5 等），重放结果与归档定稿**逐像素一致**。

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
