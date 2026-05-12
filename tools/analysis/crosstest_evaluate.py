import os
import sys
import platform
import csv
import json
import torch
import yaml
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 将项目根目录加入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataset import LiteCueDataset
from src.data.transforms import get_transforms
from src.models.detector import LiteCueNet
from src.utils.metrics import calculate_metrics, logits_to_fake_probs
from src.utils.logger import setup_logger
from torch.utils.data import DataLoader

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
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file (.pth or .pth.tar)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of data loading workers (default: 0 on Windows, 4 on Linux/Mac)')
    parser.add_argument('--log_dir', type=str, default='logs/crosstest', help='Directory to save evaluation logs')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save JSON/CSV artifacts')
    parser.add_argument('--num_forward', type=int, default=1, help='Number of forward passes per video (for averaging predictions, default: 1)')
    parser.add_argument('--load_workers', type=int, default=4, help='Multi-forward 时并行加载样本的线程数，0 表示串行 (default: 8)')
    parser.add_argument('--failure_topk', type=int, default=200, help='Number of most confident wrong predictions to save')
    args = parser.parse_args()
    
    # Windows 系统上多进程 DataLoader 容易出现共享内存问题，默认使用单进程
    if args.num_workers is None:
        if platform.system() == 'Windows':
            args.num_workers = 0
            print("Detected Windows OS, setting num_workers=0 to avoid shared memory issues")
        else:
            args.num_workers = 4

    # 设置日志
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"crosstest_{timestamp}.log"
    logger = setup_logger(args.log_dir, filename=log_filename)
    
    logger.info("="*60)
    logger.info("LiteCue-Net Cross-Dataset Evaluation Started!")
    logger.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 1. 加载配置 (Model Config + Dataset Config)
    logger.info(f"Loading training config from: {args.config}")
    config = load_config(args.config)
    logger.info(f"Loading dataset config from: {args.dataset_config}")
    with open(args.dataset_config, 'r', encoding='utf-8') as f:
        ds_config = yaml.safe_load(f)
    
    # 合并配置
    config.update(ds_config)
    logger.info(f"Target Dataset: {config['name']}")
    logger.info(f"Model Config: feature_dim={config['model']['feature_dim']}, "
                f"clip_num={config['model']['clip_num']}, clip_len={config['model']['clip_len']}, "
                f"backbone={config['model']['backbone']}, "
                f"temporal_module={config['model'].get('temporal_module', 'gated_mlp')}, "
                f"freq_branch={config['model'].get('use_frequency_branch', False)}, "
                f"token_dropout={config['model'].get('token_dropout', 0.0)}")

    # 检查索引文件是否存在
    index_path = config.get('index_path')
    if not index_path:
        logger.error("Index path not found in config!")
        return
    
    if not os.path.exists(index_path):
        logger.error("="*60)
        logger.error(f"Index file not found: {index_path}")
        logger.error("="*60)
        logger.error("Please generate the index file first by running:")
        logger.error("")
        logger.error(f"python tools/preprocess/build_dataset_index.py \\")
        logger.error(f"  --data_root {config.get('data_root', 'data/clips/Celeb-DF-v2')} \\")
        logger.error(f"  --save_path {index_path}")
        logger.error("")
        logger.error("="*60)
        return
    
    logger.info(f"Index file found: {index_path}")

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
    
    # Windows 上使用多进程时，pin_memory 可能导致共享内存问题
    # num_workers=0 时 pin_memory 没有实际收益，且可能 OOM
    use_pin_memory = torch.cuda.is_available() and args.num_workers > 0
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=use_pin_memory
    )
    logger.info(f"Test set size: {len(test_ds)} videos")
    logger.info(f"Number of forward passes per video: {args.num_forward}")

    # 3. 构建模型
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

    # 4. 加载权重
    checkpoint_path = args.checkpoint
    
    # 如果传入的是目录，自动查找权重文件
    if os.path.isdir(checkpoint_path):
        logger.info(f"Checkpoint path is a directory: {checkpoint_path}")
        # 按优先级查找常见的权重文件名（将 .pth.tar 文件优先，避免选择到 .pth 目录）
        possible_names = ['best_model.pth.tar', 'model_best.pth.tar', 'checkpoint.pth.tar', 'best_model.pth', 'model_best.pth']
        found = False
        for name in possible_names:
            candidate_path = os.path.join(checkpoint_path, name)
            # 检查是否是文件（不是目录）
            if os.path.isfile(candidate_path):
                checkpoint_path = candidate_path
                found = True
                logger.info(f"Found checkpoint file: {checkpoint_path}")
                break
        
        if not found:
            logger.error(f"No checkpoint file found in directory: {checkpoint_path}")
            logger.error(f"Looked for: {', '.join(possible_names)}")
            return
    elif not os.path.isfile(checkpoint_path):
        logger.error(f"Checkpoint not found or is not a file: {checkpoint_path}")
        return

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # 检测 checkpoint 格式并提取 state_dict
    # .pth.tar 格式通常包含 'state_dict' 键和其他元信息
    # .pth 格式通常是直接的 state_dict 字典
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # .pth.tar 格式：包含 state_dict 和其他元信息
        logger.info("Detected .pth.tar format (contains 'state_dict' key)")
        state_dict = checkpoint['state_dict']
        # 记录 checkpoint 中的其他信息（如果有）
        if 'epoch' in checkpoint:
            logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'best_auc' in checkpoint:
            logger.info(f"Checkpoint best AUC: {checkpoint['best_auc']:.2f}%")
    else:
        # .pth 格式：直接的 state_dict 字典，或未知格式
        if isinstance(checkpoint, dict):
            logger.info("Detected .pth format (direct state_dict)")
        else:
            logger.warning("Unknown checkpoint format, attempting to use directly as state_dict")
        state_dict = checkpoint
        
    # 处理 DataParallel 前缀 'module.'
    if len(state_dict) > 0 and list(state_dict.keys())[0].startswith('module.'):
        logger.info("Removing 'module.' prefix from state_dict keys (DataParallel model)")
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # 加载参数
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning("="*60)
        logger.warning("CHECKPOINT KEY MISMATCH DETECTED!")
        logger.warning("="*60)
    if missing:
        logger.warning(f"Missing keys in checkpoint: {len(missing)} keys (present in model but not in checkpoint)")
        if len(missing) <= 10:
            logger.warning(f"Missing keys: {missing}")
        logger.warning("These layers will use random initialization — results may be invalid!")
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {len(unexpected)} keys (in checkpoint but not in model)")
        if len(unexpected) <= 10:
            logger.warning(f"Unexpected keys: {unexpected}")
    if missing or unexpected:
        logger.warning("="*60)

    if not missing and not unexpected:
        logger.info("Checkpoint loaded successfully with all keys matched")
    
    # 5. 开始推理
    model.eval() # 开启 Evaluation 模式 (Stage 3 HRM 生效)
    
    all_targets = []
    all_logits = []
    
    # 如果 num_forward > 1，需要为每个视频进行多次采样和 Forward
    # 为了增加测试覆盖，每次采样使用随机模式（train mode）
    if args.num_forward > 1:
        logger.info(f"Using multi-forward evaluation: {args.num_forward} passes per video with random sampling")
        # 创建一个使用 train mode 的数据集用于随机采样
        train_mode_ds = LiteCueDataset(
            index_path=config['index_path'],
            data_root=config['data_root'],
            transforms=get_transforms(mode='val'),
            mode='train',  # 使用 train mode 进行随机采样
            clip_num=config['model']['clip_num'],
            clip_len=config['model']['clip_len'],
            path_patterns=path_patterns
        )
        
        def load_batch(indices):
            return torch.stack([train_mode_ds[i][0] for i in indices], dim=0)

        load_workers = max(0, args.load_workers)
        if load_workers > 0:
            logger.info("Starting multi-forward inference (batched, batch_size=%d, load_workers=%d)...", args.batch_size, load_workers)
        else:
            logger.info("Starting multi-forward inference (batched, batch_size=%d, sequential load)...", args.batch_size)
        batch_size = args.batch_size
        executor = ThreadPoolExecutor(max_workers=load_workers) if load_workers > 0 else None

        with torch.no_grad():
            prefetch_future = None
            for start in tqdm(range(0, len(test_ds), batch_size), desc="Evaluating batches"):
                end = min(start + batch_size, len(test_ds))
                chunk_indices = list(range(start, end))
                labels_chunk = [test_ds[i][1] for i in chunk_indices]

                video_logits_chunk = []
                for forward_idx in range(args.num_forward):
                    if prefetch_future is not None:
                        batch_tensors = prefetch_future.result().to(device, non_blocking=True)
                        prefetch_future = None
                    else:
                        if executor is not None:
                            tensors = list(executor.map(lambda i: train_mode_ds[i][0], chunk_indices))
                            batch_tensors = torch.stack(tensors, dim=0).to(device, non_blocking=True)
                        else:
                            batch_tensors = load_batch(chunk_indices).to(device, non_blocking=True)

                    if forward_idx < args.num_forward - 1:
                        if executor is not None:
                            prefetch_future = executor.submit(load_batch, chunk_indices)
                        else:
                            prefetch_future = None
                    elif start + batch_size < len(test_ds):
                        next_start = start + batch_size
                        next_end = min(next_start + batch_size, len(test_ds))
                        next_indices = list(range(next_start, next_end))
                        if executor is not None and next_indices:
                            prefetch_future = executor.submit(load_batch, next_indices)
                        else:
                            prefetch_future = None
                    else:
                        prefetch_future = None

                    video_logits, _ = model(batch_tensors)
                    video_logits_chunk.append(video_logits)

                avg_logits = torch.stack(video_logits_chunk, dim=0).mean(dim=0)
                all_targets.extend(labels_chunk)
                all_logits.extend(avg_logits.cpu().numpy().tolist())

        if executor is not None:
            executor.shutdown(wait=True)
    else:
        # 原始的单次 Forward 逻辑（保持向后兼容）
        logger.info("Starting single-forward inference...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                images = images.to(device)

                # Forward (Video Logits, Clip Logits)
                # Eval 模式下，model 内部会自动调用 HRM (Stage 3)
                video_logits, _ = model(images)

                all_targets.extend(labels.cpu().numpy().tolist())
                all_logits.extend(video_logits.cpu().numpy().tolist())

    # 6. 计算指标
    logger.info("Calculating evaluation metrics...")
    metrics = calculate_metrics(all_targets, all_logits)
    fake_probs, probs = logits_to_fake_probs(all_logits)
    
    # 输出结果到控制台和日志
    result_summary = "\n" + "="*60 + "\n"
    result_summary += f"Cross-Dataset Evaluation Result\n"
    result_summary += "="*60 + "\n"
    result_summary += f"Target Dataset: {config['name']}\n"
    result_summary += f"Source Model: {os.path.basename(checkpoint_path)}\n"
    result_summary += f"Checkpoint Path: {checkpoint_path}\n"
    result_summary += f"Forward Passes per Video: {args.num_forward}\n"
    result_summary += "-" * 60 + "\n"
    result_summary += f"AUC: {metrics['auc']:.2f}%\n"
    result_summary += f"Accuracy: {metrics['acc']:.2f}%\n"
    result_summary += f"Balanced Accuracy: {metrics['balanced_acc']:.2f}%\n"
    result_summary += f"AP: {metrics['ap']:.2f}%\n"
    result_summary += f"EER: {metrics['eer']:.2f}%\n"
    result_summary += f"TPR@FPR=1%: {metrics['tpr_at_fpr_1']:.2f}%\n"
    result_summary += f"TPR@FPR=0.1%: {metrics['tpr_at_fpr_0_1']:.2f}%\n"
    result_summary += "-" * 60 + "\n"
    result_summary += f"Confusion Matrix:\n"
    result_summary += f"  TN={metrics['tn']}  FP={metrics['fp']}\n"
    result_summary += f"  FN={metrics['fn']}  TP={metrics['tp']}\n"
    result_summary += "="*60 + "\n"
    
    print(result_summary)
    logger.info(result_summary)
    
    # 记录详细统计信息
    logger.info(f"Total samples evaluated: {len(all_targets)}")
    real_count = sum(1 for label in all_targets if label == 0)
    fake_count = sum(1 for label in all_targets if label == 1)
    logger.info(f"Real samples: {real_count}, Fake samples: {fake_count}")
    output_dir = args.output_dir or os.path.join(args.log_dir, f"artifacts_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    records = []
    for idx, (target, logit, prob_fake, prob_pair) in enumerate(zip(all_targets, all_logits, fake_probs, probs)):
        item = test_ds.data[idx] if idx < len(test_ds.data) else {}
        pred = int(prob_pair.argmax())
        confidence = float(prob_pair[pred])
        records.append({
            'index': idx,
            'video_id': item.get('video_id', ''),
            'path': item.get('path', ''),
            'dataset': item.get('dataset', config['name']),
            'method': item.get('method', ''),
            'label': int(target),
            'pred': pred,
            'prob_fake': float(prob_fake),
            'confidence': confidence,
            'correct': int(pred == int(target)),
            'logit_real': float(logit[0]),
            'logit_fake': float(logit[1]),
        })

    metrics_payload = {
        'target_dataset': config['name'],
        'checkpoint_path': checkpoint_path,
        'num_forward': args.num_forward,
        'total_samples': len(all_targets),
        'real_samples': real_count,
        'fake_samples': fake_count,
        'metrics': metrics,
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    pred_csv = os.path.join(output_dir, 'predictions.csv')
    with open(pred_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ['index'])
        writer.writeheader()
        writer.writerows(records)

    failures = sorted(
        [r for r in records if r['correct'] == 0],
        key=lambda r: r['confidence'],
        reverse=True,
    )[:args.failure_topk]
    failure_csv = os.path.join(output_dir, 'failures_topk.csv')
    with open(failure_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ['index'])
        writer.writeheader()
        writer.writerows(failures)

    logger.info(f"Saved metrics JSON: {os.path.join(output_dir, 'metrics.json')}")
    logger.info(f"Saved predictions CSV: {pred_csv}")
    logger.info(f"Saved top failure CSV: {failure_csv}")
    logger.info("="*60)
    logger.info("Cross-Dataset Evaluation Completed!")
    logger.info("="*60)

if __name__ == "__main__":
    main()

