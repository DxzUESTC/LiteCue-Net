import os
import time
import torch
import torch.optim as optim
# [移除] import torch.cuda.amp as amp  <-- 旧版导入方式移除
from tqdm import tqdm
import logging

# 导入工具
from src.utils.metrics import AverageMeter, calculate_metrics
from src.utils.checkpoint import save_checkpoint

class Trainer:
    """
    LiteCue-Net 标准训练器
    集成：自动混合精度(AMP)、Cosine学习率调度、指标计算(AUC/ACC)、模型保存、日志记录
    """
    def __init__(self, model, train_loader, val_loader, criterion, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.device = device
        self.logger = logging.getLogger("LiteCue")

        # 1. 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-4)
        )

        # 2. 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=1e-6
        )

        # 3. [修改] 混合精度 Scaler (新版 API)
        # 显式指定 'cuda' 设备类型
        self.scaler = torch.amp.GradScaler(device.type, enabled=config.get('use_amp', True) and device.type == 'cuda')

        # 4. 状态追踪
        self.start_epoch = 1
        self.best_auc = 0.0
        self.save_dir = config['save_dir']
        
        # Resume 逻辑
        if config.get('resume_path'):
            self._resume_checkpoint(config['resume_path'])

    def _unpack_batch(self, batch):
        if len(batch) == 3:
            images, labels, metadata = batch
        else:
            images, labels = batch
            metadata = None
        return images, labels, metadata

    def _metadata_tensor(self, metadata, key):
        if metadata is None or key not in metadata:
            return None
        value = metadata[key]
        if torch.is_tensor(value):
            return value.to(self.device, non_blocking=True).long()
        return torch.as_tensor(value, device=self.device, dtype=torch.long)

    def train_epoch(self, epoch):
        """训练一个 Epoch"""
        self.model.train()
        
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4f')
        loss_v_meter = AverageMeter('LossVid', ':.4f')
        loss_c_meter = AverageMeter('LossClip', ':.4f')
        top1 = AverageMeter('Acc', ':6.2f')

        end = time.time()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch}/{self.config['epochs']}] Train", leave=False)

        for i, batch in enumerate(pbar):
            data_time.update(time.time() - end)
            images, labels, metadata = self._unpack_batch(batch)

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            domain_labels = self._metadata_tensor(metadata, 'dataset_id')

            # --- [修改] 混合精度前向传播 (新版 API) ---
            # 使用 torch.amp.autocast 并指定 device_type='cuda'
            with torch.amp.autocast(self.device.type, enabled=self.config.get('use_amp', True) and self.device.type == 'cuda'):
                need_features = self.config.get('generalization', {}).get('enabled', False)
                outputs = self.model(
                    images,
                    return_features=need_features,
                    return_domain=need_features,
                )
                if isinstance(outputs, dict):
                    video_logits = outputs['video_logits']
                    clip_logits = outputs['clip_logits']
                    features = outputs.get('features')
                    domain_logits = outputs.get('domain_logits')
                else:
                    video_logits, clip_logits = outputs
                    features = None
                    domain_logits = None
                loss, loss_dict = self.criterion(
                    video_logits,
                    clip_logits,
                    labels,
                    features=features,
                    domain_labels=domain_labels,
                    domain_logits=domain_logits,
                )

            # --- 反向传播与更新 ---
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # --- 记录 ---
            batch_size = labels.size(0)
            losses.update(loss.item(), batch_size)
            loss_v_meter.update(loss_dict['loss_video'], batch_size)
            loss_c_meter.update(loss_dict['loss_clip'], batch_size)

            _, preds = torch.max(video_logits, 1)
            acc = (preds == labels).float().mean().item() * 100.0
            top1.update(acc, batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_postfix({
                'loss': f"{losses.avg:.4f}",
                'acc': f"{top1.avg:.1f}%",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            if i % max(1, len(self.train_loader) // 5) == 0:
                self.logger.info(
                    f"Epoch: [{epoch}][{i}/{len(self.train_loader)}] "
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                    f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                    f"Acc {top1.val:.2f} ({top1.avg:.2f})"
                )

        return losses.avg, top1.avg

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        
        losses = AverageMeter('Loss', ':.4f')
        all_targets = []
        all_logits = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch [{epoch}/{self.config['epochs']}] Val  ", leave=False)

        for i, batch in enumerate(pbar):
            images, labels, metadata = self._unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # 验证时通常不需要 autocast，但为了稳健性加上也没问题
            outputs = self.model(images)
            if isinstance(outputs, dict):
                video_logits = outputs['video_logits']
                clip_logits = outputs['clip_logits']
            else:
                video_logits, clip_logits = outputs
            loss, _ = self.criterion(video_logits, clip_logits, labels)

            losses.update(loss.item(), labels.size(0))

            all_targets.extend(labels.cpu().numpy().tolist())
            all_logits.extend(video_logits.cpu().numpy().tolist())

        metrics = calculate_metrics(all_targets, all_logits)
        auc = metrics['auc']
        acc = metrics['acc']

        self.logger.info(
            f"Epoch [{epoch}] Validation Result: "
            f"Loss: {losses.avg:.4f} | AUC: {auc:.2f}% | Acc: {acc:.2f}%"
        )
        
        return losses.avg, auc, acc

    def fit(self):
        self.logger.info(f"Start training on {self.device} for {self.config['epochs']} epochs.")
        self.logger.info(f"AMP Enabled: {self.config.get('use_amp', True)}")
        
        start_time = time.time()

        for epoch in range(self.start_epoch, self.config['epochs'] + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_auc, val_acc = self.validate(epoch)
            
            self.scheduler.step()
            
            is_best = val_auc > self.best_auc
            if is_best:
                self.best_auc = val_auc
                self.logger.info(f"⭐ New Best AUC: {self.best_auc:.2f}% (Epoch {epoch})")
            
            save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'best_auc': self.best_auc,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }, is_best, self.save_dir, epoch=epoch, best_auc=self.best_auc if is_best else None)

        total_time = time.time() - start_time
        self.logger.info(f"Training Complete. Total Time: {total_time/3600:.2f} hours.")
        self.logger.info(f"Best AUC: {self.best_auc:.2f}%")

    def _resume_checkpoint(self, resume_path):
        from src.utils.checkpoint import load_checkpoint
        try:
            start_epoch, best_auc = load_checkpoint(
                resume_path, self.model, self.optimizer, self.scheduler, device=self.device
            )
            self.start_epoch = start_epoch
            self.best_auc = best_auc
            self.logger.info(f"Resumed from epoch {self.start_epoch} with best AUC {self.best_auc:.2f}")
        except Exception as e:
            self.logger.error(f"Failed to resume from {resume_path}: {e}")