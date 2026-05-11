# LiteCue-Net 数据集预处理指南

本文档详细说明如何使用 `tools/preprocess/` 目录下的脚本完成数据集的预处理工作。

## 目录

- [概述](#概述)
- [环境准备](#环境准备)
- [预处理流程](#预处理流程)
- [脚本详解](#脚本详解)
- [完整示例](#完整示例)
- [常见问题](#常见问题)

---

## 概述

LiteCue-Net 的数据预处理流程包含三个主要步骤：

1. **人脸提取** (`extract_faces.py` + `process_dataset.py`): 从原始视频中提取人脸并保存为图片序列
2. **索引构建** (`build_dataset_index.py`): 扫描处理后的图片目录，生成数据集索引文件
3. **数据验证** (`verify_data.py`): 验证数据的连续性和完整性

### 数据目录结构

预处理后的数据应遵循以下目录结构：

```
data/
├── raw_videos/                    # 原始视频文件（建议使用软链接）
│   ├── FaceForensics++/
│   │   ├── original_sequences/    # 真实视频
│   │   └── manipulated_sequences/ # 伪造视频
│   └── Celeb-DF-v2/
│       ├── Celeb-real/
│       ├── YouTube-real/
│       └── Celeb-synthesis/
│
└── clips/                         # 预处理后的图片序列
    ├── FaceForensics++/
    │   ├── original_sequences/
    │   │   └── [video_id]/       # 每个视频一个文件夹
    │   │       ├── frame_00000.jpg
    │   │       ├── frame_00001.jpg
    │   │       └── ...
    │   └── manipulated_sequences/
    │       └── ...
    └── Celeb-DF-v2/
        └── ...
```

---

## 环境准备

### 1. 安装依赖

确保已安装以下 Python 包：

```bash
pip install opencv-python insightface tqdm numpy
```

### 2. InsightFace 模型下载

`extract_faces.py` 使用 InsightFace 的 RetinaFace 模型进行人脸检测。首次运行时会自动下载模型到 `~/.insightface/models/`。

如果下载失败，可以手动下载 `buffalo_l` 模型并放置到对应目录。

### 3. GPU 支持（推荐）

人脸检测和提取过程计算密集，强烈建议使用 GPU 加速。确保已安装支持 CUDA 的 PyTorch 和 onnxruntime-gpu：

```bash
pip install onnxruntime-gpu
```

---

## 预处理流程

### 步骤 1: 人脸提取

从原始视频中提取人脸并保存为图片序列。

#### 1.1 单视频处理（测试用）

使用 `extract_faces.py` 处理单个视频：

```python
# 直接运行脚本（修改脚本中的测试路径）
python tools/preprocess/extract_faces.py
```

或在代码中使用：

```python
from extract_faces import FaceExtractor

extractor = FaceExtractor(image_size=(224, 224), device='cuda')
extractor.process_video("path/to/video.mp4", "path/to/output_dir")
```

#### 1.2 批量处理数据集

使用 `process_dataset.py` 批量处理整个数据集：

```bash
cd tools/preprocess
python process_dataset.py
```

**注意**: 需要修改脚本中的 `INPUT_ROOT` 和 `OUTPUT_ROOT` 变量：

```python
INPUT_ROOT = "data/raw_videos/FaceForensics++"  # 原始视频根目录
OUTPUT_ROOT = "data/clips/FaceForensics++"       # 输出图片根目录
```

**功能特性**:
- 自动扫描所有视频文件（支持 `.mp4`, `.avi`, `.mov`, `.mkv`）
- 自动跳过 `c40` 文件夹（低质量压缩版本）
- 断点续传：如果输出目录已存在且非空，自动跳过
- 错误日志：处理失败的视频会记录到 `error_log.txt`

### 步骤 2: 构建数据集索引

扫描处理后的图片目录，生成数据集索引文件（`dataset_index.pkl`）。

```bash
python tools/preprocess/build_dataset_index.py \
    --data_root data/clips \
    --save_path data/dataset_index.pkl
```

**参数说明**:
- `--data_root`: 预处理后图片的根目录（通常是 `data/clips`）
- `--save_path`: 索引文件保存路径（默认: `data/dataset_index.pkl`）

**索引文件格式**:

索引文件是一个 Python pickle 文件，包含一个列表，每个元素是一个字典：

```python
{
    'video_id': 'video_001',           # 视频ID（文件夹名称）
    'label': 0,                        # 标签：0=真实，1=伪造
    'num_frames': 300,                 # 帧数
    'path': 'FaceForensics++/original_sequences/video_001',  # 相对路径
    'abs_path': '/absolute/path/to/video_001'  # 绝对路径
}
```

**标签自动识别**:

脚本会根据路径中的目录名自动识别标签：

- **真实视频** (`label=0`):
  - `Celeb-real`
  - `YouTube-real`
  - `original_sequences`

- **伪造视频** (`label=1`):
  - `manipulated_sequences`
  - `Celeb-synthesis`

### 步骤 3: 验证数据连续性

验证采样出的帧是否在物理上连续（LiteCue-Net 要求每个 clip 内的帧必须连续）。

```bash
python tools/preprocess/verify_data.py \
    --index_path data/dataset_index.pkl \
    --data_root data/clips \
    --num_samples 100 \
    --mode train
```

**参数说明**:
- `--index_path`: 数据集索引文件路径
- `--data_root`: 图片数据根目录（必须提供）
- `--num_samples`: 随机检查的视频数量（0 表示检查全部）
- `--mode`: 采样模式，`train` 或 `val`（默认: `train`）

**验证内容**:
1. **索引连续性**: 检查采样器返回的索引是否连续
2. **物理连续性**: 检查对应文件名中的帧序号是否连续

**示例输出**:

```
Verifying 100 random videos from root: data/clips ...
100%|████████████| 100/100 [00:30<00:00,  3.33it/s]

==================================================
Verification Summary
==================================================
PASSED! All 100 checked videos have physically consecutive frames.
```

如果发现问题，会输出详细的错误信息，包括：
- 视频ID
- 错误类型（索引不连续或物理帧不连续）
- 具体的文件名和帧序号

---

## 脚本详解

### extract_faces.py

**功能**: 从单个视频中提取人脸并保存为对齐后的图片。

**核心类**: `FaceExtractor`

**初始化参数**:
- `det_size`: 检测器输入尺寸（默认: `(640, 640)`）
- `image_size`: 输出图片尺寸（默认: `(224, 224)`）
- `device`: 运行设备，`'cuda'` 或 `'cpu'`（默认: `'cuda'`）

**处理流程**:
1. 使用 RetinaFace 检测每帧中的人脸
2. 选择面积最大的人脸（如果有多个人脸）
3. 使用 5 个关键点进行人脸对齐
4. 调整尺寸到指定大小（默认 224×224）
5. 保存为 JPG 格式，文件名格式: `frame_XXXXX.jpg`

**注意事项**:
- 如果某帧未检测到人脸，会跳过该帧
- 输出目录不存在时会自动创建
- 每处理 100 帧会打印一次进度

### process_dataset.py

**功能**: 批量处理数据集中的所有视频。

**处理逻辑**:
1. 递归扫描 `input_root` 目录下的所有视频文件
2. 自动跳过 `c40` 文件夹（低质量压缩版本）
3. 为每个视频创建对应的输出目录（保持目录结构）
4. 检查输出目录是否已存在且非空（断点续传）
5. 调用 `FaceExtractor` 处理视频
6. 记录处理失败的视频到 `error_log.txt`

**目录结构保持**:
- 输入: `data/raw_videos/FaceForensics++/original_sequences/video_001.mp4`
- 输出: `data/clips/FaceForensics++/original_sequences/video_001/frame_XXXXX.jpg`

### build_dataset_index.py

**功能**: 扫描图片目录并生成数据集索引。

**处理流程**:
1. 递归查找所有包含图片的文件夹
2. 根据路径中的目录名自动识别标签
3. 统计每个视频的帧数
4. 生成索引项并保存为 pickle 文件

**支持的图片格式**: `.jpg`, `.jpeg`, `.png`, `.bmp`

**输出统计信息**:
- 总视频数
- 真实视频数
- 伪造视频数
- 前 5 条记录示例

### verify_data.py

**功能**: 验证数据的连续性和完整性。

**验证逻辑**:
1. 加载数据集索引
2. 随机选择指定数量的视频（或全部）
3. 对每个视频使用 `LiteCueSampler` 进行采样
4. 检查每个 clip 内的帧是否连续：
   - 索引连续性：采样索引是否连续
   - 物理连续性：文件名中的帧序号是否连续

**采样器配置**:
- `clip_num=16`: 每个视频采样 16 个 clip
- `clip_len=4`: 每个 clip 包含 4 帧

**错误类型**:
- `Path not found`: 数据路径不存在（检查 `--data_root` 参数）
- `Index Discontinuity`: 采样器返回的索引不连续
- `Physical Frame Discontinuity`: 文件名中的帧序号不连续

---

## 完整示例

### 示例 1: 处理 FaceForensics++ 数据集

```bash
# 步骤 1: 批量提取人脸
cd tools/preprocess
# 修改 process_dataset.py 中的路径
python process_dataset.py

# 步骤 2: 构建索引
python build_dataset_index.py \
    --data_root ../../data/clips \
    --save_path ../../data/dataset_index.pkl

# 步骤 3: 验证数据
python verify_data.py \
    --index_path ../../data/dataset_index.pkl \
    --data_root ../../data/clips \
    --num_samples 100 \
    --mode train
```

### 示例 2: 处理 Celeb-DF v2 数据集

```bash
# 步骤 1: 批量提取人脸
cd tools/preprocess
# 修改 process_dataset.py 中的路径
INPUT_ROOT = "data/raw_videos/Celeb-DF-v2"
OUTPUT_ROOT = "data/clips/Celeb-DF-v2"
python process_dataset.py

# 步骤 2: 构建索引（如果已有索引，会合并）
python build_dataset_index.py \
    --data_root ../../data/clips \
    --save_path ../../data/dataset_index.pkl

# 步骤 3: 验证数据
python verify_data.py \
    --index_path ../../data/dataset_index.pkl \
    --data_root ../../data/clips \
    --num_samples 200 \
    --mode train
```

### 示例 3: 仅处理部分视频（测试）

```python
# 在 process_dataset.py 中添加过滤逻辑
video_paths = [v for v in video_paths if 'test_video' in v][:10]  # 只处理前10个
```

---

## 常见问题

### Q1: 处理速度慢怎么办？

**A**: 
- 确保使用 GPU（`device='cuda'`）
- 检查 GPU 使用率，确保模型在 GPU 上运行
- 如果 CPU 处理，考虑降低 `det_size`（如 `(320, 320)`）

### Q2: 某些视频处理失败，提示 "Could not open video"

**A**: 
- 检查视频文件是否损坏
- 确认视频格式是否支持（`.mp4`, `.avi`, `.mov`, `.mkv`）
- 查看 `error_log.txt` 获取详细错误信息

### Q3: 构建索引时提示 "Could not determine label"

**A**: 
- 检查目录结构是否符合预期
- 确认路径中包含以下目录名之一：
  - 真实: `Celeb-real`, `YouTube-real`, `original_sequences`
  - 伪造: `manipulated_sequences`, `Celeb-synthesis`
- 如果使用自定义目录结构，需要修改 `build_dataset_index.py` 中的 `LABEL_MAP`

### Q4: 验证数据时提示 "Path not found"

**A**: 
- 检查 `--data_root` 参数是否正确
- 确认索引文件中的路径是相对路径，需要与 `--data_root` 拼接
- 示例：如果索引中路径是 `FaceForensics++/original_sequences/video_001`，`--data_root` 应该是 `data/clips`

### Q5: 验证时发现帧不连续

**A**: 
- 检查原始视频是否完整
- 确认人脸提取过程中是否有帧被跳过（未检测到人脸）
- 如果某些帧确实无法检测到人脸，这是正常的，但会影响采样连续性
- 考虑使用更宽松的人脸检测阈值或使用其他检测方法

### Q6: 如何合并多个数据集的索引？

**A**: 
- 多次运行 `build_dataset_index.py`，每次指定不同的 `--data_root`
- 脚本会自动合并到同一个索引文件中
- 或手动加载多个索引文件并合并：

```python
import pickle

# 加载多个索引
with open('index1.pkl', 'rb') as f:
    index1 = pickle.load(f)
with open('index2.pkl', 'rb') as f:
    index2 = pickle.load(f)

# 合并
merged_index = index1 + index2

# 保存
with open('merged_index.pkl', 'wb') as f:
    pickle.dump(merged_index, f)
```

### Q7: 内存不足怎么办？

**A**: 
- 批量处理时，`process_dataset.py` 是逐个处理视频的，内存占用应该不大
- 如果仍然不足，可以分批处理，每次处理部分视频
- 检查是否有其他进程占用内存

### Q8: 如何检查预处理进度？

**A**: 
- `process_dataset.py` 使用 `tqdm` 显示进度条
- 可以通过输出目录中的文件夹数量估算进度
- 检查 `error_log.txt` 查看失败的视频

---

## 性能优化建议

1. **并行处理**: 如果有多个 GPU，可以手动分割数据集并并行处理
2. **批量大小**: 当前是逐个处理，可以考虑批量处理以提高 GPU 利用率
3. **断点续传**: `process_dataset.py` 已支持断点续传，可以随时中断和恢复
4. **存储优化**: 考虑使用 SSD 存储预处理后的图片，提高训练时的读取速度

---

## 相关文档

- [LiteCue-Net 项目整体框架](./LiteCue-Net%20项目整体框架.md)
- [LiteCue-Net 跨数据集测试](./LiteCue-Net%20跨数据集测试.md)
- [LiteCue-Net 使用 Gated-MLP 时序建模的思路](./LiteCue-Net%20使用%20Gated-MLP%20时序建模的思路.md)

---

**最后更新**: 2024-12-02

