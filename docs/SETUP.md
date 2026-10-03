# Side2Front 复现指南（从整合包到跑通）

本整合包是**纯代码包**，不含任何模型权重（体量太大不适合邮件传输）。
按本文档顺序操作，即可完整复现：GUI 界面、三条生图管线（真人/动漫/动物）、
竞赛三张样例图的定稿直出。

---

## 0. 包内容与系统要求

**包内**：全部源码（GUI + 经典几何层 + 离线生图脚本）、3 张样例图、
3 张已注册定稿图、动漫风格 LoRA 权重（6.4M）、CodeFormer 最小 vendored
代码（`third_party/codeformer_min/`）、全部文档。

**包外（需自行下载）**：
- Python 依赖（pip 安装）
- 模型权重（见第 3 节，共约 12GB 下载量）
- 三个第三方代码仓库（git clone，见第 2 节）

**系统要求**：

| 项 | 要求 |
|---|---|
| Python | 3.10+（实测 3.12） |
| CPU | 任意多核 x86，**不需要 GPU** |
| 内存 | 8GB 可跑 GUI 经典层；真人 SD 管线建议 ≥16GB |
| 硬盘 | 代码 ~10MB + 模型缓存 ~14GB |
| 网络 | 仅安装/下载阶段需要；运行时完全离线 |
| 系统 | Windows（实测）/ macOS / Linux |

---

## 1. 安装依赖

```bash
# 在解压后的 Side2Front 目录里
python -m venv .venv
.venv\Scripts\activate            # Windows；macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install cython                # 3ddfa_v2 编译需要
```

---

## 2. 克隆三个第三方代码仓库

真人管线依赖三个开源仓库的**推理代码**（不是权重）。前两个直接放
`HF_HOME` 指向的目录（脚本会自动找到）：

```bash
# 设定模型缓存目录（所有模型都会放这里）
# Windows (cmd):
set HF_HOME=D:\huggingface_cache
# macOS/Linux:
export HF_HOME=~/huggingface_cache

mkdir -p "$HF_HOME"

# (a) 3DDFA_V2 —— 人脸 68 点/3D 参数（放仓库根目录）
git clone https://github.com/cleardusk/3DDFA_V2.git 3ddfa_v2
cd 3ddfa_v2
# 权重 mb1_120x120.pth / bfm_noneck_v3.pkl 均随 clone 自带, 无需下载

# (a-1) NMS: 二选一
#   方式1(免编译, 推荐): 把包内兜底实现复制进去
cp ../third_party/3ddfa_patches/cpu_nms.py FaceBoxes/utils/nms/cpu_nms.py
#   方式2: 编译 Cython 版
#   cd FaceBoxes/utils && python build.py build_ext --inplace && cd ../..

# (a-2) Sim3DR: 必须编译 (真人网格旋转要用)
cd Sim3DR && python setup.py build_ext --inplace && cd ..
cd ..
# Linux/macOS 直接 sh ./build.sh 亦可

# (b) Arc2Face 推理代码 —— ReferenceAdapter 身份注入
git clone https://github.com/foivospar/Arc2Face "$HF_HOME/arc2face_code"

# (c) Rotate-and-Render 推理代码 —— 网格旋转骨架
#     GAN 权重不需要下载, 管线自动回退 Arc2Face 路径
git clone https://github.com/Hangz-nju-cuhk/Rotate-and-Render "$HF_HOME/rnr_code"
```

---

## 3. 下载模型

### 3.1 一键下载（推荐）

```bash
HF_ENDPOINT=https://hf-mirror.com python tools/download_models.py
```

自动下载下表全部 HuggingFace 模型 + CodeFormer 权重，断点可重跑。
（国内网络用 hf-mirror.com 镜像；能直连 huggingface.co 可去掉前缀）

### 3.2 模型明细（说明 + 手动下载链接）

#### A. GUI 运行时经典层（必需，~380MB，**首次运行 GUI 自动下载**，无需手动）

| 模型 | 说明 | 大小 | 链接 |
|---|---|---|---|
| anime-face-detector yolov3 | 动漫/动物侧脸人脸框检测 | 235M | https://huggingface.co/hysts/anime-face-detector-yolov3 |
| anime-face-detector hrnetv2 | 动漫 68 关键点 | 38M | https://huggingface.co/hysts/anime-face-detector-hrnetv2 |
| face-frontalization | 真人脸 GAN 正面化（经典层主路径） | 22M | https://huggingface.co/spaces/opetrova/face-frontalization |
| DINOv2 ViT-S/14 | 动物特征迁移（关键点无关语义对齐） | 84M | https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth |

#### B. 离线生图层（用生图脚本/④按钮时必需，~12GB）

| 模型 | 说明 | 大小 | 链接 |
|---|---|---|---|
| SD1.5 (`v1-5-pruned-emaonly.safetensors`) | 真人/动物管线的生图底模 | 4.3G | https://huggingface.co/runwayml/stable-diffusion-v1-5 |
| Counterfeit-V3.0 fp16 | 动漫管线底模（二次元风格） | 2.1G | https://huggingface.co/gsdf/Counterfeit-V3.0 |
| Arc2Face（unet + arcface.onnx + ref_adapter） | 真人身份注入 + 相似度评分（arcface.onnx） | ~4G | https://huggingface.co/FoivosPar/Arc2Face |
| IP-Adapter（image_encoder + ip-adapter_sd15.safetensors） | 三条管线通用的身份/参考保持 | ~1.8G | https://huggingface.co/h94/IP-Adapter |
| ControlNet lineart | 动漫管线线稿结构控制 | 1.4G | https://huggingface.co/lllyasviel/control_v11p_sd15_lineart |
| CodeFormer `codeformer.pth` | 真人管线人脸修复（w=0.8） | 376M | https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth |
| Anything-V5（可选） | gen_frontal_refs.py 其他实验模式用 | 2.1G | https://huggingface.co/stablediffusionapi/anything-v5 |

不需要下载的：
- **s2fstyle LoRA**（动漫风格）已随包分发：`assets/lora/s2fstyle/`
- **RnR GAN 权重**（rotatespade）：真人管线自动回退 Arc2Face，无需下载
- **CodeFormer 推理代码**：已内置 `third_party/codeformer_min/`，脚本自动找到；
  权重 `codeformer.pth` 放到 `$HF_HOME/codeformer/codeformer.pth`
  （一键下载脚本会自动放好）

### 3.3 下载后的目录结构

```
$HF_HOME/                        （例如 D:\huggingface_cache）
├── arc2face_code/               ← git clone
├── rnr_code/                    ← git clone
├── codeformer/
│   └── codeformer.pth
├── hub/
│   ├── models--runwayml--stable-diffusion-v1-5/
│   ├── models--gsdf--Counterfeit-V3.0/
│   ├── models--FoivosPar--Arc2Face/
│   ├── models--h94--IP-Adapter/
│   ├── models--lllyasviel--control_v11p_sd15_lineart/
│   ├── models--hysts--anime-face-detector-yolov3/      (自动)
│   ├── models--hysts--anime-face-detector-hrnetv2/     (自动)
│   ├── models--opetrova--face-frontalization/          (自动)
│   └── ...
（Side2Front 仓库根目录）
├── 3ddfa_v2/                    ← git clone + 编译（见第 2 节）
└── third_party/
    ├── codeformer_min/basicsr/  （已在包内, gen_rnr 自动回退使用）
    └── 3ddfa_patches/cpu_nms.py （已在包内, 复制到 3ddfa_v2 对应位置）
```

---

## 4. 验证

```bash
set HF_HUB_OFFLINE=1            # 下载完成后设上, 运行时完全离线 (可选)
python main.py
```

1. **加载 `4/face.png`** → 点"转换"：应即刻（0ms）显示已注册定稿正脸
   （CLIP 复合相似度 0.762）。`4/cat.jpg`、`4/people.jpg` 同理。
2. **加载任意新动漫/真人/动物侧脸图** → 点"转换"：走对应经典管线，秒级出图。
3. **（可选）点"SD 生成正脸"④**：对未注册新图跑完整 SD 管线
   （真人 ~7min / 动漫 ~5min / 动物 ~4min，CPU），完成后自动质检精修，
   可点"注册为定稿"⑤ 固化。
4. **（可选）一键重放动漫 13 段流水线**：
   `python tools/pipeline_final.py`（约 1.5-2h，结果与包内定稿逐像素一致）。

---

## 5. 常见问题

| 问题 | 处理 |
|---|---|
| 下载模型 401/超时 | 用 `HF_ENDPOINT=https://hf-mirror.com` 镜像；脚本已自动禁用 Xet 后端 |
| `3ddfa_v2` 编译失败 | 确认 `pip install cython`；Windows 无 bash 时用 `python build.py` |
| 真人管线段错误 (0xC0000005) | 内存不足：关掉其他大内存进程；脚本已内置逐张量加载兜底 |
| 加载新图后按钮灰色 | 正常——等待自动分类完成（状态栏提示）即解锁 |
| 生成结果不满意 | GUI ⑥"换一批"换种子组 / 部位重修四钮定向修复 / ◀▶ 翻版本历史 |
| 想看完整技术细节 | `PIPELINE.md`（动漫 13 段配方）、`docs/RNR_LESSONS.md`（真人管线教训+一键指南） |
