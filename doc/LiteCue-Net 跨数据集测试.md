目前完成了 FF++ 上的训练并得到了 `model_best.pth.tar`

**跨数据集实验 (Cross-Dataset Evaluation)** 是验证模型泛化能力的核心步骤。TFCU 论文中是通过在 **Celeb-DF**、**DFDC** 等从未见过的数据集上测试，来证明其“时序解耦”策略的有效性 。

实现这一步，需要做三件事：

1. **数据准备**：为目标数据集（如 Celeb-DF）生成索引。
2. **配置**：新建目标数据集的配置文件。
3. **脚本**：编写一个专门的评估脚本。

---

### 第一步：准备目标数据集 (Data Preparation) **必须步骤**

已经下载并用 `extract_faces.py` 处理好了 **Celeb-DF-v2** 数据集，存放于 `data/clips/Celeb-DF-v2`。

**在运行评估脚本之前，必须先为数据集生成索引文件 `.pkl`。**

**运行命令：**

**Linux/Mac (Bash):**
```Bash
python tools/preprocess/build_dataset_index.py \
  --data_root data/clips/Celeb-DF-v2 \
  --save_path tools/analysis/crosstestindex/celebdfv2_crosstest_index.pkl
```

**Windows (PowerShell):**
```PowerShell
python tools/preprocess/build_dataset_index.py `
  --data_root data/clips/Celeb-DF-v2 `
  --save_path tools/analysis/crosstestindex/celebdfv2_crosstest_index.pkl
```

**注意：** `--save_path` 参数必须是**完整的文件路径**（包含文件名 `.pkl`），不能只是目录路径。

**说明：**
- 这个命令会扫描 `data/clips/Celeb-DF-v2` 目录下的所有视频片段
- 自动识别真实/伪造标签（根据目录名：`Celeb-real`、`YouTube-real` 为真实，`Celeb-synthesis` 为伪造）
- 生成索引文件保存到 `tools/analysis/crosstestindex/celebdfv2_crosstest_index.pkl`
- 如果索引文件已存在，评估脚本会直接使用；如果不存在，脚本会提示错误并显示生成命令

---

### 第二步：创建数据集配置 (Config)

在跨数据集测试中，默认**不划分**训练/验证集，是将**整个**目标数据集作为 **测试集 (Test Set)**。

新建文件 `configs/crosstest/celebdfv2_crosstest.yaml`：

```YAML
# configs/crosstest/celebdfv2_crosstest.yaml

name: "Celeb-DF-v2-CrossTest"

# --- 路径设置 ---
data_root: "data/clips/Celeb-DF-v2"    # 修改为你的实际路径
index_path: "tools/analysis/crosstestindex/celebdfv2_crosstest_index.pkl" # 刚才生成的索引路径

# --- 数据集过滤 (Dataset Filtering) ---
# 如果索引文件包含多个数据集，通过路径模式过滤出当前数据集
path_patterns:
  - "Celeb-real"      # Celeb-DF 真实视频 (Celebrity)
  - "YouTube-real"    # Celeb-DF 真实视频 (YouTube)
  - "Celeb-synthesis" # Celeb-DF 伪造视频

# --- 划分策略 ---
# Cross-Dataset 实验通常使用全量数据进行测试
split_ratios:
  train: 0.0
  val: 0.0
  test: 1.0

# --- 预处理参数 ---
# 保持与训练时一致的归一化参数
normalization:
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
```

---

### 第三步：运行评估脚本 (Evaluation Script)

评估脚本 `tools/analysis/crosstest_evaluate.py` 已实现完整功能，使用单个 `--config` 参数指向跨数据集测试配置。

**当前脚本特性：**
- 支持新旧 checkpoint 格式自动检测 (`.pth` / `.pth.tar`)
- 使用 `weights_only=True` 安全加载
- 详细的 key mismatch 检测与警告
- 输出完整指标：AUC / ACC / AP / EER / TPR@1%FPR / TPR@0.1%FPR / 混淆矩阵
- 支持 batch 中可选的 video path 字段 (3 元素 tuple)

**模型参数传递：**
脚本自动从配置中读取以下新参数并传递给模型：

```python
model = LiteCueNet(
    feature_dim=config['model']['feature_dim'],
    clip_num=config['model']['clip_num'],
    clip_len=config['model']['clip_len'],
    num_classes=config['model']['num_classes'],
    backbone_name=config['model']['backbone'],
    token_dropout=config['model'].get('token_dropout', 0.0),
    use_temporal_diff=config['model'].get('use_temporal_diff', False),
    use_frequency_branch=config['model'].get('use_frequency_branch', False),
    frequency_fuse_block=config['model'].get('frequency_fuse_block', 2),
    temporal_module=config['model'].get('temporal_module', 'gated_mlp'),
    num_domains=config.get('generalization', {}).get('num_domains', 0),
    grl_lambda=config.get('generalization', {}).get('grl_lambda', 1.0),
).to(device)
```

---

### 第四步：运行实验

**重要提示：** 在运行评估脚本之前，请确保已经完成了以下步骤：

1. 数据已准备好：目标数据集（如 Celeb-DF-v2）已下载并提取到 `data/clips/Celeb-DF-v2`
2. **索引文件已生成**：已运行第一步的命令生成索引文件

如果索引文件不存在，评估脚本会提示错误并显示生成索引的命令。

在项目根目录下运行评估：

```Bash
python tools/analysis/crosstest_evaluate.py \
  --config configs/crosstest/celebdfv2_crosstest.yaml \
  --checkpoint checkpoints/exp_001/best_model.pth \
  --batch_size 32
```

> **注意**：当前脚本使用单个 `--config` 参数指向 crosstest 配置（而非旧版的 `--config` + `--dataset_config` 双参数）。旧版接口已废弃。

**日志记录：**

评估脚本会自动将日志保存到 `logs/crosstest/` 目录下，日志文件名格式为 `crosstest_{timestamp}.log`，包含：
- 配置信息（模型参数、数据集信息、时序模块类型、频域分支等）
- 数据加载统计（测试集大小、真实/伪造样本数量）
- Checkpoint 加载信息（含 key mismatch 检测）
- 评估进度
- **最终评估结果**（AUC / ACC / AP / EER / TPR@FPR / 混淆矩阵）

所有信息会同时输出到控制台和日志文件，方便后续查看和分析。
