# LiteCue-Net FastAPI 推理服务

> 将训练好的 LiteCue-Net 权重封装为 RESTful API，支持视频上传、鉴伪推理、Grad-CAM 热力图可视化。

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
                    ┌────────▼─────────┐
                    │   FaceProcessor   │  InsightFace buffalo_l
                    │  人脸检测+对齐裁剪 │  人脸检测 + landmark 对齐
                    │  帧采样+归一化     │  全局稀疏+局部密集采样
                    └────────┬─────────┘
                             │  tensor (1, 16, 4, 3, 224, 224)
                    ┌────────▼─────────┐
                    │  InferenceEngine  │  LiteCueNet 加载权重
                    │  Grad-CAM 前向+反向│  一次前向同时得到分数和热力图
                    └────────┬─────────┘
                             │  JSON 响应 + base64 热力图
                             ▼
```

---

## 文件说明

```
api/
├── __init__.py          # 空文件，包标记
├── config.py            # 配置：模型架构参数、路径、服务端口
├── processor.py         # 视频预处理：人脸检测/对齐/采样/归一化
├── engine.py            # 推理引擎：模型加载/前向/Grad-CAM
└── main.py              # FastAPI 应用：路由、生命周期、CORS

doc/
└── LiteCue-Net FastAPI 推理服务.md   # 本文档
```

### 各模块职责

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `config.py` | `Settings` | 统一管理所有可配置项，支持环境变量覆盖 |
| `processor.py` | `FaceProcessor` | 初始化 InsightFace，加载视频，逐帧人脸检测+对齐，构建输入张量 |
| `processor.py` | `_sample_indices` | 全局稀疏+局部密集采样策略（与训练一致） |
| `processor.py` | `_build_tensor` | 归一化并 reshape 为 `(1, M, K, 3, H, W)` |
| `engine.py` | `InferenceEngine` | 构建 LiteCueNet、加载 checkpoint、执行推理 |
| `engine.py` | `GradCAM` | 注册 backbone 钩子，反向传播计算类激活热力图 |
| `main.py` | `lifespan` | FastAPI 生命周期，启动时加载模型 |
| `main.py` | `detect` | `POST /api/v1/detect` 端点处理函数 |

---

## 启动方式

### 前提

- LiteCue conda 环境已激活（或等效虚拟环境）
- 权重文件位于 `checkpoints/exp_20260511/best_model.pth`
- InsightFace 模型首次使用时自动下载到 `~/.insightface/models/buffalo_l/`

### 启动命令

```bash
# 从项目根目录启动
cd d:\01_Lab\Project\LiteCue-Net

# 方式一：直接运行
python api/main.py

# 方式二：uvicorn（推荐，支持热重载）
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 验证服务

启动后访问 http://localhost:8000/docs 可看到 Swagger 交互式文档。

健康检查：

```bash
curl http://localhost:8000/api/v1/health
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
| `heatmap_frames` | array | Top-6 关键帧热力图（始终返回，判真时也展示模型关注的区域） |
| `heatmap_frames[].frame_index` | int | 帧在采样序列中的索引 (0~63) |
| `heatmap_frames[].clip_index` | int | 所属 clip 索引 (0~15) |
| `heatmap_frames[].clip_fake_probability` | float | 该 clip 的伪造概率 |
| `heatmap_frames[].heatmap_base64` | str | JET 伪彩热力图 JPEG 的 base64 编码 |

**错误响应：**

| 状态码 | 说明 |
|--------|------|
| 400 | 非视频文件 |
| 413 | 文件超过 100MB |
| 422 | 处理失败（如无法读取视频、无人脸） |

**请求示例：**

```bash
curl -X POST http://localhost:8000/api/v1/detect \
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
  2. InsightFace buffalo_l 模型检测人脸
  3. 选取面积最大的 face
  4. 5 点 landmark → face_align.norm_crop(image_size=224)
  5. 输出 224×224 对齐人脸 (RGB, uint8)

无人脸时的降级策略:
  - 已有成功帧 → 复用最近一帧的人脸
  - 无一帧有人脸 → 中心裁剪整帧并缩放到 224×224
  - 全片均无人脸 → 返回 422 错误

代码: processor.py::FaceProcessor.process_video()
```

使用 landmark 仿射对齐而非简单 bbox 裁剪，与训练预处理一致，保证最佳推理精度。

### 3. 归一化与张量构建

```python
# ImageNet 标准化（与训练 val 变换一致）
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# 处理流程
arr = np.stack(faces) / 255.0          # (64, 224, 224, 3) float32 [0,1]
arr = (arr - mean) / std               # 标准化
arr = arr.transpose(0, 3, 1, 2)        # (64, 3, 224, 224) CHW
arr = arr.reshape(1, 16, 4, 3, 224, 224)  # (B, M, K, C, H, W)
```

### 4. 模型推理

```python
outputs = model(input_tensor)
# video_logits: (1, 2)   → softmax → [real_prob, fake_prob]
# clip_logits:  (1, 16, 2) → 每个 clip 的独立预测

# 判定规则
is_fake = fake_prob > real_prob
```

模型架构参数（必须与 checkpoint 匹配）：

| 参数 | 值 |
|------|-----|
| `feature_dim` | 256 |
| `clip_num` (M) | 16 |
| `clip_len` (K) | 4 |
| `num_classes` | 2 |
| `backbone` | mobilenetv4_conv_small.e2400_r224_in1k |
| `use_frequency_branch` | True |
| `frequency_fuse_block` | 2 |
| `temporal_module` | attention |

---

## Grad-CAM 热力图

### 原理

无论鉴定结果为真或假，始终执行 Grad-CAM，为前端提供模型决策的空间关注点。

```
步骤:
  1. 第二次前向（启用梯度，不包裹 torch.no_grad）
  2. video_logits[:, 1].sum().backward()  ← 对伪造类反向传播
  3. 从 backbone.blocks[-1] 的 forward hook 获取:
       - activations: (B*M*K, 960, 7, 7)  ← 最后一个 block 输出
       - gradients:   (B*M*K, 960, 7, 7)  ← 伪造类的梯度
  4. 计算 CAM:
       weights = gradients.mean(dim=(2,3))  → (N, 960, 1, 1)
       cam = ReLU(Σ(weights × activations)) → (N, 7, 7)
  5. 逐帧 min-max 归一化到 [0, 1]
```

### 关键帧筛选

```
Top-6 clips 筛选:
  1. clip_logits → softmax → 每 clip 伪造概率 (1, 16, 2)
  2. 按伪造概率降序排列，取前 6 个 clip
  3. 每个选中 clip 内:
       - 4 帧分别计算 CAM 均值
       - 取均值最高的帧作为关键帧
  4. CAM resize 到 224×224 → JET 伪彩 → JPEG → base64

输出: 最多 6 帧热力图（无论真假始终返回）
```

### 代码位置

- `engine.py::GradCAM` — 钩子注册与管理
- `engine.py::GradCAM.generate` — 前向+反向+CAM 计算
- `engine.py::_cam_to_base64` — CAM → JET 伪彩 → base64 JPEG

---

## 环境变量配置

所有配置均支持通过环境变量覆盖，方便容器化部署。

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `CHECKPOINT_PATH` | `checkpoints/exp_20260511/best_model.pth` | 权重文件路径 |
| `INSIGHTFACE_ROOT` | `~/.insightface/models` | InsightFace 模型目录 |
| `API_HOST` | `0.0.0.0` | 监听地址 |
| `API_PORT` | `8000` | 监听端口 |
| `DEVICE` | `cuda` | 推理设备 (`cuda` / `cpu`) |

---

## 常见问题

### 服务启动报 "No module named 'src'"

确保在项目根目录执行启动命令，API 会自动将项目根目录加入 `sys.path`。

### 人脸检测模型下载失败

InsightFace `buffalo_l` 模型在首次使用时自动下载。如网络受限，可手动下载模型文件放置到 `~/.insightface/models/buffalo_l/` 目录。

### 推理速度慢

瓶颈通常在 InsightFace 人脸检测（~30-50ms/帧 × 64帧 ≈ 2-3秒）。如 GPU 可用，确保 `DEVICE=cuda`。也可以通过设置 `TOP_K_FRAMES=3` 减少无意义的热力图输出。

### 上传超时

默认上传限制 100MB，可通过 `MAX_UPLOAD_SIZE` 调整。对于长视频，建议在客户端先裁剪或降采样。

### 返回 "no face detected"

确保视频中包含清晰可见的人脸。侧脸、极端遮挡、低分辨率等情况可能导致检测失败。
