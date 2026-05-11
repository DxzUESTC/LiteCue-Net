目前完成了 FF++ 上的训练并得到了 `model_best.pth.tar`

**跨数据集实验 (Cross-Dataset Evaluation)** 是验证模型泛化能力的核心步骤。TFCU 论文中是通过在 **Celeb-DF**、**DFDC** 等从未见过的数据集上测试，来证明其“时序解耦”策略的有效性 。

实现这一步，需要做三件事：

1. **数据准备**：为目标数据集（如 Celeb-DF）生成索引。
2. **配置**：新建目标数据集的配置文件。
3. **脚本**：编写一个专门的评估脚本。

---

### 第一步：准备目标数据集 (Data Preparation) ⚠️ **必须步骤**

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

在跨数据集测试中，通常**不划分**训练/验证集，而是将**整个**目标数据集作为 **测试集 (Test Set)**。

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

### 第三步：编写评估脚本 (Evaluation Script)

我们需要一个独立的脚本来加载 FF++ 训练好的权重，并在 Celeb-DF 上跑推理。这个脚本需要能够处理 `.pth.tar` 这种包含元数据的检查点格式。

新建文件 `tools/analysis/crosstest_evaluate.py`：

```Python
import os
import sys
import torch
import yaml
import argparse
import logging
from tqdm import tqdm

# 将项目根目录加入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataset import LiteCueDataset
from src.data.transforms import get_transforms
from src.models.detector import LiteCueNet
from src.utils.metrics import calculate_metrics
from torch.utils.data import DataLoader

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("CrossTestEval")

def load_config(config_path):
    """加载 YAML 配置文件，并处理数据集嵌套配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    if 'dataset_config' in config:
        dataset_yaml_path = config['dataset_config']
        if not os.path.exists(dataset_yaml_path):
            raise FileNotFoundError(f"Dataset config not found: {dataset_yaml_path}")
            
        with open(dataset_yaml_path, 'r', encoding='utf-8') as df:
            dataset_config = yaml.safe_load(df)
        config.update(dataset_config)
        
    return config

def main():
    parser = argparse.ArgumentParser(description="LiteCue-Net Cross-Dataset Evaluation")
    parser.add_argument('--config', type=str, required=True, help='Path to model training config (e.g. configs/train.yaml)')
    parser.add_argument('--dataset_config', type=str, required=True, help='Path to target dataset config (e.g. configs/crosstest/celebdfv2_crosstest.yaml)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pth.tar')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    args = parser.parse_args()

    logger = setup_logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 1. 加载配置 (Model Config + Dataset Config)
    config = load_config(args.config)
    with open(args.dataset_config, 'r', encoding='utf-8') as f:
        ds_config = yaml.safe_load(f)
    
    # 合并配置
    config.update(ds_config)
    logger.info(f"Evaluating on dataset: {config['name']}")

    # 2. 准备数据
    # Cross-Dataset 测试不需要数据增强，也不需要按身份划分(因为全是测试集)
    path_patterns = ds_config.get('path_patterns', None)
    test_ds = LiteCueDataset(
        index_path=config['index_path'],
        data_root=config['data_root'],
        transforms=get_transforms(mode='val'), # 仅 Resize + Normalize
        mode='test', # 采样模式为 test (中心采样)
        clip_num=config['model']['clip_num'],
        clip_len=config['model']['clip_len'],
        path_patterns=path_patterns
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True if torch.cuda.is_available() else False
    )
    logger.info(f"Test set size: {len(test_ds)} videos")

    # 3. 构建模型
    model = LiteCueNet(
        feature_dim=config['model']['feature_dim'],
        clip_num=config['model']['clip_num'],
        clip_len=config['model']['clip_len'],
        num_classes=config['model']['num_classes'],
        backbone_name=config['model']['backbone']
    ).to(device)

    # 4. 加载权重
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return

    logger.info(f"Loading weights from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 处理 checkpoint 字典结构 (如果是 .pth.tar 通常包含 'state_dict')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # 处理 DataParallel 前缀 'module.'
    if len(state_dict) > 0 and list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # 加载参数
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")
    
    # 5. 开始推理
    model.eval() # 开启 Evaluation 模式 (Stage 3 HRM 生效)
    
    all_targets = []
    all_logits = []
    
    logger.info("Starting inference...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            
            # Forward (Video Logits, Clip Logits)
            # Eval 模式下，model 内部会自动调用 HRM (Stage 3)
            video_logits, _ = model(images)
            
            all_targets.extend(labels.cpu().numpy().tolist())
            all_logits.extend(video_logits.cpu().numpy().tolist())

    # 6. 计算指标
    metrics = calculate_metrics(all_targets, all_logits)
    
    print("\n" + "="*50)
    print(f"Cross-Dataset Evaluation Result")
    print("="*50)
    print(f"Target Dataset: {config['name']}")
    print(f"Source Model: {os.path.basename(args.checkpoint)}")
    print("-" * 50)
    print(f"AUC: {metrics['auc']:.2f}%")
    print(f"Accuracy: {metrics['acc']:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
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
  --config configs/train.yaml \
  --dataset_config configs/crosstest/celebdfv2_crosstest.yaml \
  --checkpoint checkpoints/exp_001/model_best.pth.tar \
  --batch_size 32
```

**日志记录：**

评估脚本会自动将日志保存到 `logs/crosstest/` 目录下，日志文件名格式为 `crosstest_{timestamp}.log`，包含：
- 配置信息（模型参数、数据集信息）
- 数据加载统计（测试集大小、真实/伪造样本数量）
- Checkpoint 加载信息
- 评估进度
- **最终评估结果**（AUC、Accuracy）

所有信息会同时输出到控制台和日志文件，方便后续查看和分析。
