import os
import torch
import shutil
from datetime import datetime

def save_checkpoint(state, is_best, save_dir, epoch=None, best_auc=None):
    """
    保存检查点，避免覆盖历史检查点
    Args:
        state (dict): 要保存的字典对象 (包含 model, optimizer, epoch 等)
        is_best (bool): 当前是否是历史最佳模型
        save_dir (str): 保存目录
        epoch (int): 当前 epoch 编号
        best_auc (float): 当前最佳 AUC 值
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 保存带 epoch 编号的检查点 (保留历史)
    if epoch is not None:
        epoch_checkpoint = os.path.join(save_dir, f'checkpoint_epoch_{epoch:03d}.pth.tar')
        torch.save(state, epoch_checkpoint)
    
    # 2. 保存最新的检查点 (用于断点续训，会被覆盖)
    latest_checkpoint = os.path.join(save_dir, 'latest_checkpoint.pth.tar')
    torch.save(state, latest_checkpoint)
    
    # 3. 如果是最佳模型，保存带时间戳和 AUC 的最佳模型
    if is_best and epoch is not None and best_auc is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_filename = f'best_model_epoch_{epoch:03d}_auc_{best_auc:.2f}_{timestamp}.pth'
        best_path = os.path.join(save_dir, best_filename)
        
        # 只保存模型权重用于推理（不包含 optimizer 等）
        model_state = {
            'epoch': epoch,
            'state_dict': state['state_dict'],
            'best_auc': best_auc
        }
        torch.save(model_state, best_path)
        
        # 同时保存一个符号链接式的 latest best（方便快速访问）
        latest_best = os.path.join(save_dir, 'best_model.pth')
        shutil.copyfile(best_path, latest_best)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    加载检查点
    Returns:
        start_epoch (int): 恢复后的起始轮数
        best_auc (float): 历史最佳 AUC
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")
        
    print(f"=> Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 1. 加载模型权重
    # 处理 DataParallel 带来的 'module.' 前缀问题 (如果之前用了多卡训练)
    state_dict = checkpoint['state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    # 2. 加载优化器和调度器 (如果是恢复训练)
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    # 兼容旧版本的 best_acc 字段
    best_auc = checkpoint.get('best_auc', checkpoint.get('best_acc', 0.0))
    
    print(f"=> Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint.get('epoch', 0)})")
    return start_epoch, best_auc