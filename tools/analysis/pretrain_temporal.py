import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from main import load_config, set_seed
from src.data.dataset import LiteCueDataset
from src.data.transforms import get_transforms
from src.models.detector import LiteCueNet


def make_order_task(images):
    """
    Binary temporal order task. Half the batch keeps chronological clip order,
    half gets a random clip permutation. The model predicts ordered vs shuffled.
    """
    labels = torch.ones(images.size(0), dtype=torch.long, device=images.device)
    out = images.clone()
    for i in range(images.size(0)):
        if random.random() < 0.5:
            perm = torch.randperm(images.size(1), device=images.device)
            out[i] = out[i, perm]
            labels[i] = 0
    return out, labels


def main():
    parser = argparse.ArgumentParser(description='LiteCue-Net temporal self-supervised pretraining')
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    parser.add_argument('--save_dir', type=str, default='checkpoints/pretrain_temporal')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get('seed', 42))
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = LiteCueDataset(
        index_path=config['index_path'],
        data_root=config['data_root'],
        transforms=get_transforms(mode='train', use_aug=True, aug_level='medium'),
        mode='train',
        clip_num=config['model']['clip_num'],
        clip_len=config['model']['clip_len'],
        path_patterns=config.get('path_patterns'),
        domain_aug_cfg=config.get('domain_randomization'),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=torch.cuda.is_available(),
    )

    model = LiteCueNet(
        feature_dim=config['model']['feature_dim'],
        clip_num=config['model']['clip_num'],
        clip_len=config['model']['clip_len'],
        num_classes=2,
        backbone_name=config['model']['backbone'],
        token_dropout=config['model'].get('token_dropout', 0.0),
        use_temporal_diff=True,
        use_frequency_branch=config['model'].get('use_frequency_branch', False),
        frequency_fuse_block=config['model'].get('frequency_fuse_block', 2),
        temporal_module=config['model'].get('temporal_module', 'gated_mlp'),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.get('weight_decay', 1e-4))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, _ in tqdm(loader, desc=f'Pretrain epoch {epoch}/{args.epochs}'):
            images = images.to(device, non_blocking=True)
            task_images, task_labels = make_order_task(images)
            logits, _ = model(task_images)
            loss = F.cross_entropy(logits, task_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * task_labels.size(0)
            correct += (logits.argmax(dim=1) == task_labels).sum().item()
            total += task_labels.size(0)

        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total) * 100.0
        print(f'Epoch {epoch}: loss={avg_loss:.4f}, order_acc={acc:.2f}%')
        torch.save(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'pretrain_task': 'temporal_order_prediction',
                'config': config,
            },
            os.path.join(args.save_dir, f'pretrain_epoch_{epoch:03d}.pth.tar'),
        )


if __name__ == '__main__':
    main()
