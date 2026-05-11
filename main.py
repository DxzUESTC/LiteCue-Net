import os
import torch
import numpy as np
import random
import yaml
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, Subset

# 引入项目模块
from src.data.dataset import LiteCueDataset
from src.data.transforms import get_transforms
from src.data.balanced_sampler import build_weighted_sampler
from src.models.detector import LiteCueNet
from src.losses.loss import LiteCueLoss
from src.training.trainer import Trainer
from src.utils.logger import setup_logger
from tools.data.split_by_identity import split_dataset_by_identity

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
            
        print(f"Merging dataset config from: {dataset_yaml_path}")
        with open(dataset_yaml_path, 'r', encoding='utf-8') as df:
            dataset_config = yaml.safe_load(df)
        config.update(dataset_config)
        
    return config

def set_seed(seed):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # 1. 解析参数与加载配置
    parser = argparse.ArgumentParser(description="LiteCue-Net Training Launcher")
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to YAML config file')
    parser.add_argument('--test_only', action='store_true', help='仅加载 best_model.pth 在测试集上跑一次评估，不训练')
    args = parser.parse_args()
    CONFIG = load_config(args.config)

    # 初始化环境
    set_seed(CONFIG['seed'])
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)
    
    # 生成带时间戳的日志文件名，避免多次训练覆盖或混淆
    exp_name = os.path.basename(CONFIG['save_dir'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"train_{exp_name}_{timestamp}.log" if not args.test_only else f"test_only_{exp_name}_{timestamp}.log"
    logger = setup_logger(CONFIG['log_dir'], filename=log_filename)
    
    logger.info("LiteCue-Net Training Started!" if not args.test_only else "LiteCue-Net Test-Only Evaluation")
    logger.info(f"Loaded Configuration:\n{yaml.dump(CONFIG, default_flow_style=False)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # ==========================================
    # 2. 数据准备 (Data Preparation)
    # ==========================================
    logger.info(f"Loading Dataset: {CONFIG.get('name', 'Unknown')}")
    
    # 实例化数据集 (此时不包含具体的 indices，包含全量数据)
    # 获取路径过滤模式（如果配置中有的话）
    path_patterns = CONFIG.get('path_patterns', None)
    
    full_train_ds = LiteCueDataset(
        index_path=CONFIG['index_path'],
        data_root=CONFIG['data_root'],
        transforms=get_transforms(
            mode='train',
            use_aug=CONFIG.get('augmentation', {}).get('use_aug', False),
            aug_level=CONFIG.get('augmentation', {}).get('level', 'light'),
        ),
        mode='train',
        clip_num=CONFIG['model']['clip_num'],
        clip_len=CONFIG['model']['clip_len'],
        path_patterns=path_patterns,  # 传递过滤模式
        occlusion_cfg=CONFIG.get('occlusion'),  # 训练期遮挡增强，见 doc/LiteCue-Net 遮挡增强说明.md
        domain_aug_cfg=CONFIG.get('domain_randomization'),
        return_metadata=CONFIG.get('generalization', {}).get('enabled', False),
    )
    if CONFIG.get('generalization', {}).get('enabled', False) and CONFIG.get('generalization', {}).get('num_domains', 0) <= 0:
        CONFIG['generalization']['num_domains'] = len(full_train_ds.dataset_to_id)
        logger.info(f"Auto-detected num_domains={CONFIG['generalization']['num_domains']} for domain generalization.")
    
    full_val_ds = LiteCueDataset(
        index_path=CONFIG['index_path'],
        data_root=CONFIG['data_root'],
        transforms=get_transforms(mode='val'),
        mode='val',
        clip_num=CONFIG['model']['clip_num'],
        clip_len=CONFIG['model']['clip_len'],
        path_patterns=path_patterns,  # 传递过滤模式
        return_metadata=CONFIG.get('generalization', {}).get('enabled', False),
    )
    
    # [核心修改] 调用防泄露划分函数
    logger.info("Splitting dataset by Identity to prevent leakage...")
    result = split_dataset_by_identity(
        full_train_ds.data,  # 传入原始数据列表
        CONFIG['split_ratios'],
        seed=CONFIG.get('seed', None),
        verbose=True,
        logger=logger  # 传递 logger 以便记录详细信息
    )
    train_indices = result['train_indices']
    val_indices = result['val_indices']
    test_indices = result['test_indices']
    
    logger.info(f"Split Ratios: {CONFIG['split_ratios']}")
    logger.info(f"Indices Count: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # 构建 Subsets
    train_dataset = Subset(full_train_ds, train_indices)
    val_dataset = Subset(full_val_ds, val_indices)
    test_dataset = Subset(full_val_ds, test_indices)
    
    # 构建 DataLoaders
    sampler_cfg = CONFIG.get('sampling', {})
    train_sampler = None
    shuffle_train = True
    if sampler_cfg.get('balanced', False):
        train_sampler = build_weighted_sampler(
            train_dataset,
            balance_keys=sampler_cfg.get('balance_keys', ['label']),
            replacement=sampler_cfg.get('replacement', True),
        )
        shuffle_train = False
        logger.info(f"Using weighted balanced sampler with keys: {sampler_cfg.get('balance_keys', ['label'])}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    # ==========================================
    # 3. 模型构建 (Model Building)
    # ==========================================
    logger.info("Building Model...")
    model = LiteCueNet(
        feature_dim=CONFIG['model']['feature_dim'],
        clip_num=CONFIG['model']['clip_num'],
        clip_len=CONFIG['model']['clip_len'],
        num_classes=CONFIG['model']['num_classes'],
        backbone_name=CONFIG['model']['backbone'],
        token_dropout=CONFIG['model'].get('token_dropout', 0.0),
        use_temporal_diff=CONFIG['model'].get('use_temporal_diff', False),
        use_frequency_branch=CONFIG['model'].get('use_frequency_branch', False),
        num_domains=CONFIG.get('generalization', {}).get('num_domains', 0),
        grl_lambda=CONFIG.get('generalization', {}).get('grl_lambda', 1.0),
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {total_params / 1e6:.2f} M")

    # ==========================================
    # 4. 损失与优化 (Loss & Optimizer)
    # ==========================================
    criterion = LiteCueLoss(
        video_loss_weight=CONFIG['loss']['video_weight'],
        clip_loss_weight=CONFIG['loss']['clip_weight'],
        alpha=CONFIG['loss']['focal_alpha'],
        gamma=CONFIG['loss']['focal_gamma'],
        label_smoothing=CONFIG['loss'].get('label_smoothing', 0.0),
        supcon_weight=CONFIG.get('generalization', {}).get('supcon_weight', 0.0),
        supcon_temperature=CONFIG.get('generalization', {}).get('supcon_temperature', 0.2),
        coral_weight=CONFIG.get('generalization', {}).get('coral_weight', 0.0),
        mmd_weight=CONFIG.get('generalization', {}).get('mmd_weight', 0.0),
        domain_adv_weight=CONFIG.get('generalization', {}).get('domain_adv_weight', 0.0),
        groupdro_weight=CONFIG.get('generalization', {}).get('groupdro_weight', 0.0),
        groupdro_eta=CONFIG.get('generalization', {}).get('groupdro_eta', 0.1),
    )

    # ==========================================
    # 5. 开始训练 (Start Training)，或跳过
    # ==========================================
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=CONFIG,
        device=device
    )

    best_model_path = os.path.join(CONFIG['save_dir'], "best_model.pth")
    if not args.test_only:
        trainer.fit()
    else:
        if not os.path.exists(best_model_path):
            logger.error(f"--test_only 需要已有权重文件: {best_model_path} 不存在")
            return

    # ==========================================
    # 6. 最终测试 (Final Testing)
    # ==========================================
    logger.info("\n===============================")
    logger.info("Starting Final Test on Test Set...")
    logger.info("===============================")

    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        if isinstance(state_dict, dict) and state_dict and list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
    else:
        logger.warning("Best model not found, using last epoch weights!")

    # 使用与训练时相同的测试集划分进行测试
    trainer.val_loader = test_loader
    test_loss, test_auc, test_acc = trainer.validate(epoch="TEST")

    logger.info(f"Final Test Result (Identity Disjoint) -> AUC: {test_auc:.2f}% | Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    main()