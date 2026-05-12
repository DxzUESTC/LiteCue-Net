# LiteCue-Net

> **LiteCue-Net: Lightweight Forgery Cue Unraveling Network**  
> 基于 Lightweight Attention 时序建模的轻量级 deepfake 检测框架。

---

## 目录

1. [项目概述](#1-项目概述)
2. [目录结构](#2-目录结构)
3. [环境配置](#3-环境配置)
4. [数据准备](#4-数据准备)
5. [配置说明](#5-配置说明)
6. [训练模型](#6-训练模型)
7. [测试与评估](#7-测试与评估)
8. [推理与部署](#8-推理与部署)
9. [常见问题](#9-常见问题)

---

## 1. 项目概述

LiteCue-Net 是一个面向 deepfake 检测的轻量化视频伪造特征拆解网络。其核心设计理念是 **"用轻量 Attention 替代重型 Transformer"**，在保持高检测精度的同时大幅降低参数量和计算开销。

### 模型流水线（当前版本 v2，参数量 ~3.54M）

```
输入视频 (B, M=16, K=4, C, H, W)
    │
    ▼
[Stage 0] 空间特征提取 (MobileNetV4 Backbone)
    │  ┌─ 可选：频域分支高频残差 → 中间层特征图融合 (block_2)
    │  (B*M*K, D=256)
    ▼
[Stage 1] 局部微动捕捉 — Intra-Clip (DW-Conv1D + 可学习注意力池化)
    │  捕捉 K 帧内的瞬时运动异常 → 注意力加权聚合为 Clip Token
    │  (B, M, D)
    ▼
[Stage 2] 全局不一致性分析 — Inter-Clip
    │  可选：Gated-MLP / Lightweight Attention（含 Causal Mask）
    │  M 个片段间全局交互，捕捉长程逻辑矛盾
    │  (B, M, D)
    ▼
[Stage 3] 历史回顾 — LearnableDecayHRM（推理阶段启用）
    │  可学习衰减系数，利用未来信息修正过去判断
    │  (B, M, D)
    ▼
分类头 → 可学习加权融合 → 视频级预测 + 片段级预测
          └─ 熵正则化防止权重退化
```

### 版本演进

| 阶段 | 参数量 | 说明 |
|------|--------|------|
| 原始版本 | ~3.02M | MobileNetV4 + Gated-MLP + Mean Pool + 固定 HRM |
| 当前版本 | ~3.54M | +频域分支 / Attention / 注意力池化 / 加权融合 / LearnableDecayHRM / 熵正则 |

### 关键特性

| 特性 | 说明 |
|------|------|
| 轻量骨干 | MobileNetV4-Small，~2.82M 参数 |
| 频域分支 | 高频残差 → 中间层特征图融合（block_2），~150K 参数 |
| 时序建模 | Gated-MLP 或 Lightweight Attention（含 Causal Mask）可选 |
| Intra-Clip 池化 | 可学习注意力池化（SiLU 激活，替代 Tanh 防饱和） |
| 视频级融合 | 可学习权重网络加权融合 + 熵正则化 |
| HRM 历史回顾 | LearnableDecayHRM，单参数可学习衰减系数 |
| 遮挡增强 | 训练期随机遮挡，提升对局部遮挡的鲁棒性 |
| 域泛化 | 支持 SupCon / CORAL / MMD / 域对抗 / GroupDRO 等多目标训练 |
| AMP | 自动混合精度训练 |
| Gradio Demo | 提供交互式 Web 演示界面 |

### 实验结果

| 实验 | AUC | ACC | AP | EER | TPR@1%FPR |
|------|-----|-----|----|-----|-----------|
| FF++ (c23) 同数据集测试 | 99.13% | 95.33% | 99.83% | 4.10% | 90.80% |
| FF++ → Celeb-DF-v2 跨数据集 | 84.42% | 88.08% | 96.49% | 23.53% | 9.03% |

---

## 2. 目录结构

```
LiteCue-Net/
│
├── main.py                         # 训练/测试入口
├── requirements.txt                # Python 依赖
├── LICENSE                         # Apache 2.0
│
├── configs/                        # 配置文件
│   ├── train.yaml                  # 训练配置（基础，最新改进版）
│   ├── train_generalization.yaml   # 训练配置（域泛化）
│   ├── dataset/                    # 数据集配置
│   │   ├── faceforensics.yaml      # FaceForensics++
│   │   ├── celebdfv2.yaml          # Celeb-DF-v2
│   │   ├── ffiw10k.yaml            # FFIW10K
│   │   └── multisource.yaml        # 多源数据集
│   ├── model/                      # 模型配置（预留）
│   └── crosstest/                  # 跨数据集测试配置
│       ├── FF++_crosstest.yaml
│       ├── celebdfv2_crosstest.yaml
│       └── ffiw10k_crosstest.yaml
│
├── src/                            # 核心源码
│   ├── data/                       # 数据模块
│   │   ├── dataset.py              # 数据集定义
│   │   ├── sampler.py              # 时序采样器
│   │   ├── balanced_sampler.py     # 类别均衡采样
│   │   ├── transforms.py           # 数据预处理与增强
│   │   └── augmentation.py         # 遮挡增强 / 域随机化
│   ├── models/                     # 模型定义
│   │   ├── detector.py             # LiteCueNet 主模型
│   │   ├── backbones/
│   │   │   └── mobilenet_v4.py     # MobileNetV4（支持中间层特征融合）
│   │   └── components/
│   │       ├── intra_clip.py       # Stage 1: Intra-Clip（DW-Conv1D + 注意力池化）
│   │       ├── inter_clip.py       # Stage 2: Inter-Clip（Gated-MLP / Attention + Causal Mask）
│   │       └── reviewer.py         # Stage 3: LearnableDecayHRM（可学习衰减历史回顾）
│   ├── losses/                     # 损失函数
│   │   ├── loss.py                 # 主损失（Focal Loss 视频/片段 加权 + 熵正则）
│   │   ├── focal_loss.py           # Focal Loss 实现
│   │   └── generalization.py      # 域泛化损失（SupCon / CORAL / MMD / 域对抗 / GroupDRO）
│   ├── training/                   # 训练逻辑
│   │   └── trainer.py              # 训练器（AMP / Cosine Annealing / 指标追踪）
│   └── utils/                      # 工具
│       ├── checkpoint.py           # 模型保存/加载
│       ├── logger.py               # 日志系统
│       └── metrics.py              # 评估指标 (AUC/ACC/AP/EER/TPR@FPR)
│
├── tools/                          # 辅助工具
│   ├── check_env.py                # 环境检测脚本
│   ├── data/
│   │   └── split_by_identity.py    # 基于身份的防泄露划分
│   ├── preprocess/                 # 数据预处理
│   │   ├── extract_faces.py        # 人脸提取
│   │   ├── build_dataset_index.py  # 构建数据集索引
│   │   ├── build_ffiw10k_index.py  # FFIW10K 专用索引构建
│   │   ├── process_dataset.py      # 数据集批量处理
│   │   ├── verify_data.py          # 数据完整性验证
│   │   └── FFIW/
│   │       └── process_ffiw_all_frames.py
│   └── analysis/                   # 分析与演示
│       ├── crosstest_evaluate.py   # 跨数据集测试
│       ├── ablation_report.py      # 消融实验报告
│       ├── pretrain_temporal.py    # 时序预训练脚本
│       ├── distill_train.py        # 知识蒸馏训练
│       └── video_demo_gradio.py    # Gradio 交互式 Web Demo
│
├── notebooks/                      # Jupyter Notebooks
│   ├── data_pipeline.ipynb         # 数据流水线演示
│   └── training_monitor.ipynb      # 训练监控
│
├── doc/                            # 技术文档
│   ├── 使用指南.md                  # 详细使用指南
│   ├── LiteCue-Net 数据集预处理指南.md
│   └── LiteCue-Net 跨数据集测试.md
│
├── data/                           # [不提交 Git] 数据集
├── checkpoints/                    # 模型权重（best_model.pth 已提交 Git）
├── runs/                           # [不提交 Git] 训练输出
└── logs/                           # [不提交 Git] 日志
```

---

## 3. 环境配置

### 3.1 系统要求

- Python >= 3.10
- PyTorch >= 2.1.0（推荐 2.x 以支持 `torch.compile`）
- CUDA >= 11.8（GPU 训练）
- 显存 >= 8GB（batch_size=8 可训练）

### 3.2 安装步骤

```bash
# 1. 创建虚拟环境（推荐）
conda create -n litecue python=3.11
conda activate litecue

# 2. 安装 PyTorch（根据你的 CUDA 版本选择命令）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装项目依赖
pip install -r requirements.txt

# 4. 验证环境
python tools/check_env.py
```

### 3.3 依赖清单

| 包 | 用途 |
|------|------|
| torch >= 2.1.0 | 深度学习框架 |
| torchvision >= 0.16.0 | 图像处理与模型 |
| timm >= 1.0.7 | 模型库（MobileNetV4） |
| einops >= 0.7.0 | 张量操作 |
| albumentations >= 1.4.0 | 数据增强 |
| opencv-python >= 4.8.0 | 图像/视频处理 |
| insightface | 人脸检测与对齐 |
| gradio >= 4.0.0 | Web 演示界面 |
| scikit-learn | 评估指标 |
| hydra-core >= 1.3.0 | 配置管理 |
| thop / fvcore | 模型 FLOPs 分析 |

---

## 4. 数据准备

### 4.1 数据集格式

项目使用统一的 **视频片段（clip）** 格式：每个视频分解为帧图片存放于独立文件夹中，文件夹内图片按帧序号命名。

```
data/
├── clips/                              # 视频片段数据
│   ├── original_sequences/
│   │   └── c23/videos/
│   │       ├── video_001/             # 一个视频的所有帧
│   │       │   ├── frame_0000.jpg
│   │       │   ├── frame_0001.jpg
│   │       │   └── ...
│   │       └── video_002/
│   └── manipulated_sequences/
│       └── c23/Deepfakes/videos/
│           └── ...
└── dataset_index.pkl                   # 数据集索引文件
```

> **注意**：`data/` 目录已被 `.gitignore` 排除，不会提交到 Git。

### 4.2 预处理流程

```bash
# 1. 人脸提取（从原始视频提取人脸帧）
python tools/preprocess/extract_faces.py

# 2. 构建数据集索引
python tools/preprocess/build_dataset_index.py

# 3. 验证数据完整性
python tools/preprocess/verify_data.py
```

详细说明请参考 [LiteCue-Net 数据集预处理指南.md](doc/LiteCue-Net%20数据集预处理指南.md)。

### 4.3 支持的数据集

| 数据集 | 配置 | 说明 |
|--------|------|------|
| FaceForensics++ | `configs/dataset/faceforensics.yaml` | FF++ c23，含 Deepfakes/F2F/FS/NT |
| Celeb-DF-v2 | `configs/dataset/celebdfv2.yaml` | 大规模 Deepfake 数据集 |
| FFIW10K | `configs/dataset/ffiw10k.yaml` | 高精度伪造检测 |
| 多源混合 | `configs/dataset/multisource.yaml` | 多个数据集联合训练 |

---

## 5. 配置说明

### 5.1 配置结构

训练配置通过 YAML 文件组织，主配置入口为 `configs/train.yaml`，通过 `dataset_config` 字段引用数据集配置。

```
train.yaml
  ├── dataset_config ───→ dataset/faceforensics.yaml
  │                        (数据路径、过滤规则、划分比例)
  ├── 训练超参数 (epochs, batch_size, lr, ...)
  ├── 模型参数 (feature_dim, clip_num, clip_len, backbone, ...)
  ├── 增强配置 (augmentation, occlusion, domain_randomization)
  ├── 采样配置 (balanced sampling)
  └── 损失与泛化 (loss, generalization)
```

### 5.2 核心参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model.clip_num` | 16 | 全局片段数 M |
| `model.clip_len` | 4 | 片段内帧数 K |
| `model.feature_dim` | 256 | 内部特征维度 D |
| `model.backbone` | mobilenetv4_conv_small.e2400_r224_in1k | 骨干网络 |
| `model.use_frequency_branch` | true | 频域分支 |
| `model.frequency_fuse_block` | 2 | 频域融合位置（backbone block 索引） |
| `model.temporal_module` | "attention" | Inter-Clip 时序建模方式（gated_mlp / attention） |
| `model.use_temporal_diff` | false | 显式时序差分 |
| `loss.video_weight` | 1.0 | 视频级损失权重 |
| `loss.clip_weight` | 1.0 | 片段级损失权重 |
| `loss.focal_alpha` | 0.25 | Focal Loss α |
| `loss.focal_gamma` | 2.0 | Focal Loss γ |
| `loss.entropy_weight` | 0.01 | Clip 权重熵正则强度 |
| `occlusion.enabled` | true | 遮挡增强 |
| `generalization.enabled` | false | 域泛化训练 |

### 5.3 域泛化配置

`configs/train_generalization.yaml` 提供了完整的域泛化训练预设，支持以下技术：

| 技术 | 参数 | 说明 |
|------|------|------|
| SupCon | `generalization.supcon_weight` | 监督对比学习 |
| CORAL | `generalization.coral_weight` | 相关性对齐 |
| MMD | `generalization.mmd_weight` | 最大均值差异 |
| 域对抗 | `generalization.domain_adv_weight` | 对抗性域分类器 |
| GroupDRO | `generalization.groupdro_weight` | 分组鲁棒优化 |

---

## 6. 训练模型

### 6.1 基本训练

```bash
# 使用默认配置训练
python main.py --config configs/train.yaml
```

### 6.2 域泛化训练

```bash
# 使用域泛化配置
python main.py --config configs/train_generalization.yaml
```

### 6.3 仅测试（使用已有权重）

```bash
# 需要已有 best_model.pth 在 save_dir 中
python main.py --config configs/train.yaml --test_only
```

### 6.4 训练流程说明

1. **数据加载**：根据索引文件加载数据
2. **身份隔离划分**：基于身份 ID 将数据划分为 train/val/test，防止身份泄露
3. **训练循环**：
   - 每个 epoch 遍历训练集，支持 AMP 混合精度
   - 可选：遮挡增强
   - 每个 epoch 后验证集评估（AUC / ACC / AP / EER / TPR@FPR）
   - 保存最佳模型（按 AUC）到 `save_dir/best_model.pth`
4. **测试**：训练完成后在测试集上最终评估

### 6.5 训练输出

```
checkpoints/exp_20260511/
├── best_model.pth          # 验证集最优权重
├── last_epoch.pth          # 最后一轮权重
└── checkpoint_epoch_XX.pth # 中间检查点

logs/
└── train_exp_20260511_*.log  # 训练日志
```

---

## 7. 测试与评估

### 7.1 跨数据集测试

跨数据集测试用于评估模型在未见过的数据集上的泛化能力。

```bash
python tools/analysis/crosstest_evaluate.py --config configs/crosstest/celebdfv2_crosstest.yaml
```

详见 [LiteCue-Net 跨数据集测试.md](doc/LiteCue-Net%20跨数据集测试.md)。

### 7.2 消融实验

```bash
python tools/analysis/ablation_report.py
```

### 7.3 时序预训练 / 知识蒸馏

```bash
# 时序模块预训练
python tools/analysis/pretrain_temporal.py

# 知识蒸馏训练
python tools/analysis/distill_train.py
```

### 7.4 评估指标

- **AUC**：ROC 曲线下面积（主要指标）
- **ACC**：准确率
- **AP**：平均精度
- **EER**：等错误率
- **TPR@1%FPR**：1% 假阳性率下的召回率
- 支持按数据集、按伪造方法分层统计

---

## 8. 推理与部署

### 8.1 Gradio Web Demo

项目提供了基于 Gradio 的交互式 Web 界面，支持上传视频/图片进行实时检测。

```bash
# 启动 Gradio 服务（需要先准备好模型权重）
python tools/analysis/video_demo_gradio.py
```

### 8.2 代码调用推理

```python
import torch
from src.models.detector import LiteCueNet

# 加载模型（当前改进版配置）
model = LiteCueNet(
    feature_dim=256, clip_num=16, clip_len=4, num_classes=2,
    use_frequency_branch=True,
    frequency_fuse_block=2,
    temporal_module="attention",
)
checkpoint = torch.load("checkpoints/exp_20260511/best_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# 推理（输入形状: B, M, K, C, H, W）
video_tensor = torch.randn(1, 16, 4, 3, 224, 224)
with torch.no_grad():
    outputs = model(video_tensor)
    if isinstance(outputs, dict):
        video_logits = outputs['video_logits']
    else:
        video_logits, _ = outputs
    pred = video_logits.argmax(dim=1)
```

### 8.3 注意事项

- **data/** 目录不会提交到 Git，需自行准备数据。
- 仓库已包含 **FF++ (c23) 训练的最佳权重** `checkpoints/exp_20260511/best_model.pth`（AUC 99.34%），可直接用于推理和跨数据集测试。
- 首次训练前务必先完成数据预处理（人脸提取 + 索引构建）。
- 如果使用 Windows，建议 `num_workers=0` 以避免多进程加载问题。
- Gradio Demo 需要 HuggingFace Hub 支持以下载预训练权重（可选）。

---

## 9. 常见问题

**Q: 训练时报显存不足？**  
A: 降低 `batch_size` 和/或 `model.clip_num` 的值；也可以关闭 `use_amp: false`。

**Q: 数据加载报错 `FileNotFoundError`？**  
A: 检查 `configs/dataset/*.yaml` 中的 `data_root` 和 `index_path` 是否与实际路径一致。

**Q: 跨平台路径问题？**  
A: 项目内部统一使用 `os.path.join` 处理路径；配置文件中建议使用 `/` 作为分隔符。

**Q: 如何在自己的数据集上训练？**  
A: 参考 [数据集预处理指南](doc/LiteCue-Net%20数据集预处理指南.md)，按格式准备 clip 数据并构建索引文件，然后在 `configs/dataset/` 下添加新的数据集配置。

**Q: 频域分支的作用是什么？**  
A: 频域分支对输入图像计算高频残差（原图 - 低通滤波），通过 4 层 Conv2D 输出特征图，在 backbone 的中间层（block_2）做残差融合，为模型提供跨数据集稳定的频域伪造信号。

**Q: Attention 模式和 Gated-MLP 模式如何选择？**  
A: Attention 模式（默认）参数量更大（~3.54M vs ~3.02M），Gated-MLP 通过 `temporal_module: "gated_mlp"` 切换。
