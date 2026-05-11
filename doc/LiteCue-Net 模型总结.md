# LiteCue-Net 模型总结

本文根据项目现有 `doc` 文档、`configs/train.yaml`、训练代码以及 `logs` 下的训练/测试日志整理。当前项目中的模型实现对应 `src/models/detector.py`，训练主线以 `checkpoints/exp_20260305` 和 `logs/train_exp_20260305_20260305_211815.log` 为准。

## 1. 模型思想

LiteCue-Net 的目标是在较低参数量和计算量下完成视频深伪检测，核心关注点不是单帧分类，而是视频中随时间累积、扩散或不一致的伪造线索。

项目早期文档中提到的思路可以概括为两点：

- 单帧模型速度快，但容易忽略深伪痕迹在时间维度上的累积，面对高质量伪造时泛化不足。
- 3D-CNN 或 Transformer 等视频模型表达力强，但通常计算量较高，不利于端侧或轻量部署。

因此，LiteCue-Net 采用轻量解耦设计：

- 使用 MobileNetV4 负责单帧空间纹理和伪造痕迹提取。
- 使用局部时序卷积捕捉短片段内的微动异常。
- 使用 Gated-MLP 在片段序列上做全局时序交互，避免 Self-Attention 的高复杂度。
- 推理阶段使用 HRM 历史回顾模块，让后续片段中的伪造证据反向补强前序片段表示。

整体思想可以理解为：先用轻量 CNN 找到每帧的空间线索，再用两级时序模块把局部微动和全局不一致累积成视频级判断。

## 2. 方法与网络结构

### 2.1 输入与采样

模型输入形状为 `(B, M, K, C, H, W)`：

- `B`：batch size。
- `M=16`：每个视频均匀采样 16 个全局片段。
- `K=4`：每个片段内连续采样 4 帧。
- `C,H,W=3,224,224`：人脸帧图像。

采样策略由 `src/data/sampler.py` 实现，属于 “Global Sparse + Local Dense”：

- 全局稀疏：把视频划分为 16 个时间段，每段取一个 clip，覆盖完整视频。
- 局部密集：每个 clip 内取 4 帧连续图像，用于捕捉短时间微动和局部抖动。
- 训练阶段在段内随机选择起点；验证/测试阶段使用中心起点，结果更稳定。
- 如果视频帧数不足，会通过循环扩展索引保证输出张量形状一致。

### 2.2 Stage 0: MobileNetV4 空间特征提取

实现位置：`src/models/backbones/mobilenet_v4.py`。

所有帧先被展平为 `(B*M*K, 3, 224, 224)`，送入 `timm` 的 `mobilenetv4_conv_small.e2400_r224_in1k`。模型移除原分类头，保留全局平均池化，再通过轻量投影头映射到 256 维：

```text
Frame image -> MobileNetV4 GAP feature -> Linear + BN + SiLU -> 256-d feature
```

这一步的作用是以较小成本提取单帧空间纹理、边缘、高频伪影和语义信息，并为后续时序模块降低通道维度。

### 2.3 Stage 1: Intra-Clip 局部微动建模

实现位置：`src/models/components/intra_clip.py`。

该阶段处理每个 4 帧 clip 内的短时变化：

- 将展平后的帧特征恢复为 `(B*M, K, D)`。
- 使用 `Depth-wise Conv1D` 在时间维度上卷积，卷积核大小为 3。
- 通过残差方式将微动线索加回原始空间特征。
- 对 4 帧做平均池化，得到每个 clip 的 token。
- 输出形状为 `(B, M, D)`，即 `(B, 16, 256)`。

Depth-wise Conv1D 的参数量很小，但可以捕捉相邻帧之间的局部抖动、不连续表情边界、局部伪影变化等短时线索。

### 2.4 Stage 2: Inter-Clip Gated-MLP 全局时序建模

实现位置：`src/models/components/inter_clip.py`。

该阶段处理 16 个 clip token 之间的长程关系。它不使用 Transformer Self-Attention，而是使用 Gated-MLP：

- 先通过 `Linear` 将通道扩展为 `2D`。
- 将特征切分为内容流 `u` 和门控流 `v`。
- 对门控流在时间维度 `M=16` 上做线性投影，使不同 clip 之间发生全局信息交换。
- 用 `u * v` 执行门控融合，放大关键伪造线索，抑制无效或噪声片段。
- 通过输出投影恢复到 256 维，并加残差。

这一阶段承担全局不一致性建模，例如不同时间片段之间的身份纹理不稳定、边缘融合质量变化、表情/光照逻辑冲突等。

### 2.5 Stage 3: HRM 历史回顾模块

实现位置：`src/models/components/reviewer.py`。

HRM 仅在 `model.eval()` 推理阶段启用，训练阶段不启用。它没有可学习参数，逻辑是 “Future Guide History”：

```text
Feature_m' = Feature_m + sum_{s>m} Feature_s / (s - m + 1)
```

也就是说，后续片段的特征会按距离衰减权重累积到前序片段上。其动机是深伪线索可能随时间逐渐显现，后续更明显的证据可以帮助修正前序不确定判断。

### 2.6 分类与损失

分类头是一个线性层，将每个 clip token 映射为 2 类 logits：

- `clip_logits`: `(B, 16, 2)`，每个片段单独预测。
- `video_logits`: `(B, 2)`，对 16 个 clip logits 求平均得到视频级预测。

损失函数实现于 `src/losses/loss.py`，采用视频级和片段级双 Focal Loss：

- 视频级损失权重：`1.0`。
- 片段级损失权重：`1.0`。
- `focal_alpha=0.25`。
- `focal_gamma=2.0`。

这样既约束最终视频判断，也迫使每个 clip 学到可判别的局部证据；Focal Loss 用于缓解真实/伪造类别比例和难易样本不平衡问题。

## 3. 数据与训练配置

当前训练配置来自 `configs/train.yaml`，并合并 `configs/dataset/faceforensics.yaml`。

核心配置如下：

```yaml
save_dir: checkpoints/exp_20260305
log_dir: logs
dataset_config: configs/dataset/faceforensics.yaml

epochs: 50
batch_size: 16
lr: 0.0001
weight_decay: 0.0001
num_workers: 0
seed: 42
use_amp: true

model:
  feature_dim: 256
  clip_num: 16
  clip_len: 4
  num_classes: 2
  backbone: mobilenetv4_conv_small.e2400_r224_in1k

loss:
  video_weight: 1.0
  clip_weight: 1.0
  focal_alpha: 0.25
  focal_gamma: 2.0
```

训练数据配置：

- 数据集：FaceForensics++。
- 数据根目录：`data/clips`。
- 索引文件：`data/dataset_index.pkl`。
- 路径过滤：`original_sequences` 和 `manipulated_sequences`。
- 划分比例：训练 80%，验证 10%，测试 10%。
- 归一化：ImageNet mean/std，即 `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`。

日志中实际按身份划分后的数据规模为：

- 训练集：400 个 identity groups，800 个 identities，4800 个 videos。
- 验证集：50 个 identity groups，100 个 identities，600 个 videos。
- 测试集：50 个 identity groups，100 个 identities，600 个 videos。

划分使用 `tools/data/split_by_identity.py`，目标是避免同一身份同时出现在训练集和验证/测试集造成泄漏。日志显示共发现 1000 个 unique identities，合并后为 500 个 identity groups，其中 5000 个多身份视频会被绑定到同一组。

## 4. 数据增强

训练阶段启用了空间遮挡增强，配置如下：

```yaml
occlusion:
  enabled: true
  prob: 0.5
  scale_range: [0.1, 0.35]
  fill: mean
  num_patches: 1
```

实现位置为 `src/data/augmentation.py`，调用点在 `src/data/dataset.py`。其行为是：只在 `mode='train'` 时，对整段 `(M,K)` clip 使用同一个随机空间遮挡块，填充值默认为 ImageNet 归一化均值。

这个增强的目的不是模拟普通图像噪声，而是迫使模型不能只依赖脸部中心区域或单一局部区域完成分类，从而更多关注边缘、背景、融合边界和跨帧一致性，有助于提升真实场景和跨数据集泛化。

## 5. 训练过程

训练入口是 `main.py`，主要流程如下：

1. 读取 `configs/train.yaml` 并合并数据集配置。
2. 固定随机种子为 42。
3. 构建 `LiteCueDataset`，训练集使用随机采样和遮挡增强，验证/测试集使用确定性中心采样。
4. 按身份分组划分训练/验证/测试集。
5. 构建 LiteCue-Net，日志记录参数量为 3.02 M。
6. 使用 `LiteCueLoss`，即视频级 Focal Loss + clip 级 Focal Loss。
7. 使用 AdamW 优化器，学习率 `1e-4`，权重衰减 `1e-4`。
8. 使用 CosineAnnealingLR，最小学习率 `1e-6`。
9. 开启 AMP 混合精度训练。
10. 每个 epoch 后在验证集评估 AUC/Acc，并按验证 AUC 保存最佳模型。

训练日志 `logs/train_exp_20260305_20260305_211815.log` 显示：

- 训练设备：`cuda`。
- 总训练轮数：50。
- 每轮训练 batch 数：300。
- 总训练耗时：10.70 小时。
- 训练开始时第 1 轮验证结果：Loss 0.2151，AUC 61.53%，Acc 68.33%。
- 第 13 轮验证 AUC 提升到 97.44%，Acc 94.17%。
- 第 26 轮验证 AUC 达到 99.13%。
- 第 35 轮取得最佳验证结果：Loss 0.0376，AUC 99.59%，Acc 96.50%。
- 第 50 轮验证结果：Loss 0.0513，AUC 99.49%，Acc 97.67%。

最佳 checkpoint 来自第 35 轮，跨数据集日志也记录该 checkpoint 的 `best_auc=99.59%`。

## 6. 实验结果

### 6.1 FaceForensics++ 身份隔离测试

`logs/test_only_exp_20260305_20260306_094017.log` 使用 `checkpoints/exp_20260305/best_model.pth` 在身份隔离的测试集上评估，结果为：

- Test Loss：0.0591。
- Test AUC：99.14%。
- Test Acc：95.67%。

这说明在同源 FaceForensics++ 数据集、且训练/测试身份不重叠的设定下，模型可以取得很高的视频级判别性能。

### 6.2 FaceForensics++ 验证集最佳表现

训练过程中最佳验证结果出现在第 35 轮：

- Val Loss：0.0376。
- Val AUC：99.59%。
- Val Acc：96.50%。

第 50 轮验证 AUC 仍为 99.49%，说明后半段训练整体较稳定，没有明显崩溃；但最佳模型仍按第 35 轮保存。

### 6.3 FFIW10K 跨数据集测试

最新跨数据集日志为 `logs/crosstest/crosstest_20260306_101450.log`。该实验使用 FF++ 训练得到的 `best_model.pth`，在 FFIW10K-v1-CrossTest 上测试：

- 目标数据集：FFIW10K-v1-CrossTest。
- 测试样本数：3472 videos。
- 真实/伪造样本：1736 / 1736。
- 每个视频 forward 次数：5。
- Checkpoint epoch：35。
- Checkpoint best AUC：99.59%。
- Cross AUC：62.15%。
- Cross Accuracy：51.15%。

这个结果明显低于 FF++ 同源测试，说明当前模型在同源数据上拟合良好，但迁移到 FFIW10K 时泛化仍然不足。可能原因包括数据域差异、伪造方法差异、压缩/分辨率/人脸检测分布差异，以及模型仍可能学习到部分 FF++ 数据集特有线索。

### 6.4 历史跨数据集日志

项目中还保留了若干历史跨数据集记录，可作为对比参考：

- `crosstest_20251201_222857.log`：Celeb-DF-v2-CrossTest，6529 videos，AUC 86.97%，Acc 88.37%。
- `crosstest_20251202_155126.log`：FaceForensics++ Cross-Dataset Test，2000 videos，AUC 89.72%，Acc 63.50%。
- `crosstest_20251211_232222.log`：FFIW10K-v1-CrossTest，3472 videos，1 次 forward，AUC 64.10%，Acc 51.41%。
- `crosstest_20251211_233026.log`：FFIW10K-v1-CrossTest，3472 videos，5 次 forward，AUC 63.98%，Acc 51.30%。

这些日志不一定都对应当前 `exp_20260305` 的同一模型版本，但整体趋势一致：FF++ 内部表现很高，部分跨数据集测试仍有明显域泛化压力，尤其是 FFIW10K 上准确率接近随机水平。

## 7. 结论

LiteCue-Net 当前实现是一个轻量视频深伪检测模型，参数量约 3.02 M。它通过 MobileNetV4 提取单帧空间线索，通过 Intra-Clip DW-Conv1D 捕捉 4 帧局部微动，通过 Inter-Clip Gated-MLP 建模 16 个片段之间的全局不一致，并在推理阶段使用无参数 HRM 将未来片段证据反向累积到历史片段。

从训练结果看，模型在 FaceForensics++ 身份隔离测试集上表现优秀，AUC 达到 99.14%，说明架构具备较强的同源视频深伪判别能力。训练过程也较稳定，最佳验证 AUC 为 99.59%。

从跨数据集结果看，当前模型的泛化能力仍是主要短板。最新 FFIW10K 跨数据集测试 AUC 为 62.15%，Acc 为 51.15%，说明模型虽然能学到 FF++ 的有效判别模式，但面对分布差异较大的真实场景/新数据集时仍存在显著退化。后续优化应优先围绕跨数据集鲁棒性展开，例如更强的数据增强、更多源域训练、伪造方法均衡、压缩扰动、颜色/频域扰动，以及对时序线索的可解释性分析。

## 8. 关键文件索引

- 模型主类：`src/models/detector.py`
- MobileNetV4 Backbone：`src/models/backbones/mobilenet_v4.py`
- Intra-Clip 局部微动模块：`src/models/components/intra_clip.py`
- Inter-Clip Gated-MLP 模块：`src/models/components/inter_clip.py`
- HRM 历史回顾模块：`src/models/components/reviewer.py`
- 数据采样器：`src/data/sampler.py`
- 数据集读取与遮挡增强调用：`src/data/dataset.py`
- 训练器：`src/training/trainer.py`
- 损失函数：`src/losses/loss.py`
- 主训练配置：`configs/train.yaml`
- FaceForensics++ 数据配置：`configs/dataset/faceforensics.yaml`
- 最新主训练日志：`logs/train_exp_20260305_20260305_211815.log`
- 最新独立测试日志：`logs/test_only_exp_20260305_20260306_094017.log`
- 最新 FFIW10K 跨数据集日志：`logs/crosstest/crosstest_20260306_101450.log`
