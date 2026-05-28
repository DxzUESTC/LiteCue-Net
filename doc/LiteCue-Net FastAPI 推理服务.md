# LiteCue-Net FastAPI 推理服务

> 将训练好的 LiteCue‑Net 权重封装为 RESTful API，支持视频上传、鉴伪推理、Grad‑CAM 热力图可视化。

---

## 目录

- [架构总览](#架构总览)
- [文件说明](#文件说明)
- [启动方式](#启动方式)
- [API 端点](#api-端点)
- [请求与响应](#请求与响应)
- [处理管线详解](#处理管线详解)
- [Grad-CAM 热力图](#grad-cam-热力图)
- [环境变量配置](#环境变量配置)
- [常见问题](#常见问题)

---

## 架构总览

```
                    ┌──────────────────┐
                    │  上传视频 (POST)  │
                    └────────┬─────────┘
                             │
              ╔═════════════╧══════════════════╗
              ║  Core Layer — 人脸检测与对齐    ║
              ║                                 ║
              ║  RetinaFaceDetector              ║  ONNX Runtime (det_10g.onnx)
              ║    → 5-pt landmark 检测          ║  api/core/retinaface_detector.py
              ║    → 仿射变换 norm_crop          ║  api/core/face_align.py
              ║    → ImageNet 标准化              ║  api/core/normalize.py
              ║                                 ║
              ║  FaceProcessor                    ║  全局稀疏+局部密集采样
              ╚═════════════╤════════════════════╝
                             │  tensor (1, M, K, 3, H, W)
              ╔═════════════╧════════════════════╗
              ║  Model Layer — 模型定义          ║
              ║  LiteCueNet                       ║  MobileNetV4 + Intra/Inter/HRM
              ║    → backbones/mobilenet_v4.py   ║
              ║    → components/                  ║  时序组件（intra_clip 等）
              ╚═════════════╤════════════════════╝
                             │
              ╔═════════════╧════════════════════╗
              ║  Backend Layer — 推理引擎         ║
              ║  LiteCueNetBackend                ║  架构参数自动从 checkpoint 推断
              ║    → GradCAM 前向+反向            ║  api/backends/gradcam.py
              ║    → base.py (抽象接口)           ║  可扩展其他模型
              ╚═════════════╤════════════════════╝
                             │  JSON 响应 + base64 热力图
                             ▼
```

**关键设计决策：**

- **剥离 InsightFace**：人脸检测改用 ONNX Runtime 直接运行 `det_10g.onnx`（原来 buffalo_l 中的检测模型），不再依赖 insightface Python 包。
- **3 层架构**：`core/`（检测+对齐） / `model/`（网络定义） / `backends/`（推理引擎+可解释性）。
- **架构参数自动发现**：`LiteCueNetBackend` 在加载 checkpoint 时自动推断 `feature_dim`、`num_classes`、`temporal_module` 等参数，不必在配置中硬编码。
- **cuDNN 自动发现**：启动时自动将 PyTorch 自带的 cuDNN 9 DLL 目录加入 `PATH`，使 ONNX Runtime 也能使用 GPU。

---

## 文件说明

```
api/                          # FastAPI 推理服务
├── __init__.py
├── config.py                 # 配置：路径、端口、检测参数
├── main.py                   # FastAPI 应用：路由、生命周期、CORS
├── processor.py              # FaceProcessor：视频预处理入口
│
├── core/                     # Core Layer：人脸检测与对齐
│   ├── __init__.py
│   ├── retinaface_detector.py  # RetinaFace ONNX 检测器
│   ├── face_detector.py        # 检测结果处理（选最大人脸、提取 landmarks）
│   ├── face_align.py           # 5-pt landmark 仿射对齐（替代 insightface）
│   └── normalize.py            # ImageNet 标准化 + 反标准化
│
├── model/                    # Model Layer：网络定义
│   ├── __init__.py
│   ├── detector.py             # LiteCueNet 主模型
│   ├── backbones/
│   │   └── mobilenet_v4.py     # MobileNetV4 封装
│   └── components/
│       ├── intra_clip.py       # Stage 1: Intra-Clip
│       ├── inter_clip.py       # Stage 2: Inter-Clip Transformer
│       └── reviewer.py         # Stage 3: 历史回顾 HRM
│
└── backends/                 # Backend Layer：推理引擎
    ├── __init__.py             # 注册中心 + create_backend() 工厂
    ├── base.py                 # ModelBackend 抽象基类
    ├── litecuenet.py           # LiteCueNetBackend 实现
    └── gradcam.py              # Grad-CAM 热力图（通用，不依赖具体模型）

models/
└── buffalo_l/
    └── det_10g.onnx          # RetinaFace 检测模型（~17 MB，随仓库提交）

checkpoints/
└── exp_20260511/
    ├── best_model.pth        # LiteCue-Net 最佳权重（AUC 99.34%）
    ├── best_model_epoch_045_auc_99.34_*.pth  # 同文件（可追溯原始 epoch）
    └── checkpoint_epoch_*.pth.tar            # 完整训练检查点

requirements-api.txt          # 推理服务依赖-cpu
requirements-api-gpu.txt      # 推理服务依赖-gpu
```

### 各模块职责

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `config.py` | `Settings` | 统一管理路径、检测参数、服务端口；启动时自动暴露 cuDNN 9 DLL |
| `processor.py` | `FaceProcessor` | 加载视频、采样帧、调用 core 层检测+对齐、构建输入张量 |
| `core/retinaface_detector.py` | `RetinaFaceDetector` | ONNX Runtime 封装，运行 det_10g.onnx，输出 NMS 后的人脸框+landmarks |
| `core/face_detector.py` | `pick_largest_face` / `get_landmarks` | 从检测结果中选取面积最大人脸、提取 5 点 landmarks |
| `core/face_align.py` | `norm_crop` | 5 点仿射变换对齐人脸（替代 insightface.utils.face_align） |
| `core/normalize.py` | `normalize_frames` / `denormalize_frame` | ImageNet 标准化 / 反标准化（复用训练变换参数） |
| `model/detector.py` | `LiteCueNet` | MobileNetV4 骨干 + Intra/Inter/HRM 三段时序建模 |
| `backends/__init__.py` | `create_backend` | 后端注册工厂，按名称实例化推理引擎 |
| `backends/base.py` | `ModelBackend` | 抽象基类：`predict` / `predict_with_explain` / `device` / `model_config` |
| `backends/litecuenet.py` | `LiteCueNetBackend` | 自动推断架构参数、加载 checkpoint、组装模型、调用 Grad-CAM |
| `backends/gradcam.py` | `GradCAM` | backbone 钩子注册，前向+反向计算类激活热力图 |
| `main.py` | `lifespan` | FastAPI 生命周期，启动时加载 FaceProcessor + ModelBackend |
| `main.py` | `detect` | `POST /api/v1/detect` 端点处理函数 |

---

## 启动方式

### 前提

- Python >= 3.10
- 完整的仓库克隆（包含 Git LFS 大文件）

### 一键启动

```bash
# 0. 确保 Git LFS 资源已拉取（首次）
git lfs install
git lfs pull

# 1. 安装推理服务依赖（PyTorch CUDA 版按需选择版本）
pip install -r requirements-api.txt

# 若需 GPU 加速 ONNX Runtime（RetinaFace 检测）：
pip install onnxruntime-gpu

# 2. 从项目根目录启动服务
python api/main.py

# 或使用 uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```

> 所有推理资源均在仓库内：
> - `checkpoints/exp_20260511/best_model.pth` — LiteCue-Net 权重
> - `models/buffalo_l/det_10g.onnx` — RetinaFace 检测模型
>
> 默认监听 **8001** 端口（避免与常见服务冲突），可通过环境变量覆盖。

### 验证服务

启动后访问 http://localhost:8001/docs 可看到 Swagger 交互式文档。

健康检查：

```bash
curl http://localhost:8001/api/v1/health
# {"status":"ok","device":"cuda"}
```

---

## API 端点

### `GET /api/v1/health`

健康检查。

**响应 200：**
```json
{
  "status": "ok",
  "device": "cuda"
}
```

### `POST /api/v1/detect`

上传视频进行鉴伪。

**请求：** `multipart/form-data`

| 字段 | 类型 | 说明 |
|------|------|------|
| `file` | `video/*` | 人脸视频文件（支持 mp4/avi/mov 等常见格式） |

**响应 200：**

```json
{
  "is_fake": true,
  "fake_probability": 0.9734,
  "real_probability": 0.0266,
  "processing_time_ms": 3120.5,
  "video_info": {
    "total_frames": 150,
    "fps": 30.0,
    "duration_sec": 5.0,
    "faces_detected": 64,
    "total_sampled": 64
  },
  "heatmap_frames": [
    {
      "frame_index": 42,
      "clip_index": 5,
      "clip_fake_probability": 0.99,
      "heatmap_base64": "/9j/4AAQ..."
    }
  ]
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `is_fake` | bool | 是否判定为伪造 |
| `fake_probability` | float | 伪造概率 [0, 1] |
| `real_probability` | float | 真实概率 [0, 1] |
| `processing_time_ms` | float | 端到端处理耗时（含人脸检测+推理） |
| `video_info.total_frames` | int | 视频总帧数 |
| `video_info.fps` | float | 视频帧率 |
| `video_info.duration_sec` | float | 视频时长（秒） |
| `video_info.faces_detected` | int | 成功检测到人脸的帧数 |
| `video_info.total_sampled` | int | 实际采样的帧数（始终=64） |
| `heatmap_frames` | array | Top-6 关键帧热力图（始终返回） |
| `heatmap_frames[].frame_index` | int | 帧在展平序列中的索引 (0~63) |
| `heatmap_frames[].clip_index` | int | 所属 clip 索引 (0~15) |
| `heatmap_frames[].clip_fake_probability` | float | 该 clip 的伪造概率 |
| `heatmap_frames[].heatmap_base64` | str | Grad-CAM 叠在对齐人脸上的 JPEG（base64） |

**错误响应：**

| 状态码 | 说明 |
|--------|------|
| 400 | 非视频文件 |
| 413 | 文件超过 100MB |
| 422 | 处理失败（如无法读取视频、无人脸） |

**请求示例：**

```bash
curl -X POST http://localhost:8001/api/v1/detect \
  -F "file=@test_video.mp4" \
  | python -m json.tool
```

---

## 处理管线详解

### 1. 帧采样 (Frame Sampling)

采用与训练一致的 **全局稀疏 + 局部密集** 策略。

```
视频总帧数: N
clip 数 M: 16
clip 内帧数 K: 4
所需帧数: M × K = 64

采样方式:
  将视频均匀分为 16 段
  每段内取中心位置开始的连续 4 帧
  共 16 × 4 = 64 帧，覆盖全视频时长

代码: processor.py::_sample_indices()
```

对于不足 64 帧的短视频，使用 `np.linspace` 均匀采样（允许重复），不足部分重复最后一帧补齐。

### 2. 人脸检测与对齐 (Face Detection & Alignment)

```
每帧:
  1. BGR → RGB 转换
  2. RetinaFace (det_10g.onnx) 通过 ONNX Runtime 检测人脸
     → 自动优先使用 CUDAExecutionProvider，回退至 CPUExecutionProvider
  3. 选取面积最大的 face
  4. 5 点 landmark → 仿射变换 norm_crop(image_size=224)
     （独立实现，不依赖 insightface）
  5. 输出 224×224 对齐人脸 (RGB, uint8)

无人脸时的降级策略:
  - 已有成功帧 → 复用最近一帧的人脸
  - 无一帧有人脸 → 中心裁剪整帧并缩放到 224×224
  - 全片均无人脸 → 返回 422 错误

代码:
  - processor.py::FaceProcessor.process_video()
  - core/retinaface_detector.py::RetinaFaceDetector.detect()
  - core/face_align.py::norm_crop()
```

使用 landmark 仿射对齐而非简单 bbox 裁剪，与训练预处理一致，保证最佳推理精度。

### 3. 归一化与张量构建

```python
# ImageNet 标准化（与训练 val 变换一致）
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# 处理流程 (core/normalize.py::normalize_frames)
arr = np.stack(faces) / 255.0          # (64, 224, 224, 3) float32 [0,1]
arr = (arr - mean) / std               # 标准化
arr = arr.transpose(0, 3, 1, 2)        # (64, 3, 224, 224) CHW
arr = arr.reshape(1, 16, 4, 3, 224, 224)  # (B, M, K, C, H, W)
```

### 4. 模型推理

```python
# 后端自动从 checkpoint 推断架构参数
outputs = model(input_tensor)
# video_logits: (1, 2)   → softmax → [real_prob, fake_prob]
# clip_logits:  (1, 16, 2) → 每个 clip 的独立预测

# 判定规则
is_fake = fake_prob > real_prob
```

模型架构参数由 `LiteCueNetBackend` 在加载 checkpoint 时自动推断（参考 `backends/litecuenet.py::_infer_model_config`），常见值：

| 参数 | 推断方式 | 典型值 |
|------|---------|--------|
| `feature_dim` | 从 `backbone.projector.0.weight` 形状 | 256 |
| `clip_num` (M) | 从 `inter_clip.pos_embed` / `reviewer._mask` 形状 | 16 |
| `clip_len` (K) | 输入协议参数（与 processor 保持一致） | 4 |
| `num_classes` | 从 `head.weight` 形状 | 2 |
| `backbone` | 固定为 mobilenetv4_conv_small.e2400_r224_in1k | — |
| `use_frequency_branch` | 检测 `frequency_branch.0.weight` 是否存在 | True |
| `temporal_module` | 检测 `inter_clip.pos_embed` → attention（Transformer） | attention |

---

## Grad-CAM 热力图

### 原理

无论鉴定结果为真或假，始终执行 Grad-CAM，为前端提供模型决策的空间关注点。

```
步骤:
  1. 第二次前向（启用梯度）
  2. video_logits[:, 1].sum().backward()  ← 对伪造类反向传播
  3. 从 backbone.blocks[-1] 的 forward hook 获取:
       - activations: (B*M*K, 960, 7, 7)
       - gradients:   (B*M*K, 960, 7, 7)  ← 伪造类的梯度
  4. 计算 CAM:
       weights = gradients.mean(dim=(2,3), keepdim=True)  → (N, 960, 1, 1)
       cam = ReLU(Σ(weights × activations))              → (N, 7, 7)
  5. 逐帧 min-max 归一化到 [0, 1]

代码: backends/gradcam.py::GradCAM.generate()
```

### 关键帧筛选

```
Top-6 clips 筛选:
  1. clip_logits → softmax → 每 clip 伪造概率 (1, 16, 2)
  2. 按伪造概率降序排列，取前 6 个 clip
  3. 每个选中 clip 内:
       - 4 帧分别计算 CAM 均值
       - 取均值最高的帧作为关键帧
  4. 反归一化对应 224×224 对齐人脸 → CAM 叠加热力图（75% 人脸 + 25% JET）→ JPEG → base64

输出: 最多 6 帧热力图（无论真假始终返回）
```

### 代码位置

- `backends/gradcam.py::GradCAM` — 钩子注册与管理（通用，不依赖具体模型）
- `backends/litecuenet.py::LiteCueNetBackend.predict_with_explain` — 推理+热力图组合
- `backends/litecuenet.py::_overlay_cam_to_base64` — CAM 叠在对齐人脸上 → base64 JPEG

---

## 环境变量配置

所有配置均支持通过环境变量覆盖，方便容器化部署。

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `CHECKPOINT_PATH` | `checkpoints/exp_20260511/best_model.pth` | 权重文件路径 |
| `DEVICE` | `cuda` | 推理设备（`cuda` / `cpu`） |
| `API_HOST` | `0.0.0.0` | 监听地址 |
| `API_PORT` | `8001` | 监听端口 |

> **注意**：人脸检测模型路径固定为 `models/buffalo_l/det_10g.onnx`（通过 `RETINA_MODEL_PATH` 在 `config.py` 中定义），一般无需修改。
> InsightFace 相关环境变量（`INSIGHTFACE_ROOT` 等）已废弃。

---

## 常见问题

### 服务启动报 "No module named 'src'"

确保在项目根目录执行启动命令，API 会自动将项目根目录加入 `sys.path`。

### 启动报 "Face detection model not found"

RetinaFace ONNX 模型 `det_10g.onnx` 未拉取。执行：

```bash
git lfs install
git lfs pull
```

或从 Git LFS 恢复：

```bash
git checkout -- models/buffalo_l/det_10g.onnx
```

### 启动报 "LiteCue-Net checkpoint not found"

`checkpoints/exp_20260511/best_model.pth` 缺失。同理执行 `git lfs pull`。

### ONNX Runtime 报 "No available CUDA device"

ONNX Runtime 默认从 `PATH` 中查找 cuDNN 9 DLL。如果 PyTorch 安装了 CUDA 但 ONNX Runtime 仍回退到 CPU（或报找不到 CUDA），检查：

1. 是否安装了 `onnxruntime-gpu`（仅 `onnxruntime` 不支持 CUDA）
2. PyTorch 的 cuDNN 版本是否匹配（`api/config.py` 会自动查找 PyTorch 安装目录下的 `cudnn64_9.dll`，适用 PyTorch 2.5+）

### 推理速度慢

瓶颈通常在 RetinaFace 人脸检测（~30-50ms/帧 × 64帧 ≈ 2-3秒）。确保安装了 `onnxruntime-gpu` 并正确使用 CUDA。

### 上传超时

默认上传限制 100MB，可通过 `MAX_UPLOAD_SIZE` 调整。对于长视频，建议在客户端先裁剪或降采样。

### 返回 "no face detected"

确保视频中包含清晰可见的人脸。侧脸、极端遮挡、低分辨率等情况可能导致检测失败。

### 仍想使用旧的 InsightFace 依赖？

当前 API 已剥离 InsightFace。如果你需要基于 InsightFace 的人脸检测，可以使用 `tools/preprocess/extract_faces.py`（仍保留 InsightFace 依赖）进行离线预处理，然后直接用预处理后的图片做推理。在线 API 已统一为 ONNX Runtime 方案。
