import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from main import load_config, set_seed
from src.data.dataset import LiteCueDataset
from src.data.transforms import get_transforms
from src.models.detector import LiteCueNet
from tools.data.split_by_identity import split_dataset_by_identity


def load_state_dict(path, device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    if state_dict and list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def main():
    parser = argparse.ArgumentParser(description='Teacher-student distillation for LiteCue-Net')
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    parser.add_argument('--teacher_checkpoint', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='checkpoints/distill')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--alpha', type=float, default=0.5, help='CE weight; 1-alpha is KL distillation weight')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get('seed', 42))
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = LiteCueDataset(
        index_path=config['index_path'],
        data_root=config['data_root'],
        transforms=get_transforms(mode='train', use_aug=config.get('augmentation', {}).get('use_aug', False), aug_level=config.get('augmentation', {}).get('level', 'light')),
        mode='train',
        clip_num=config['model']['clip_num'],
        clip_len=config['model']['clip_len'],
        path_patterns=config.get('path_patterns'),
        domain_aug_cfg=config.get('domain_randomization'),
    )
    split = split_dataset_by_identity(dataset.data, config['split_ratios'], seed=config.get('seed', 42), verbose=False)
    train_dataset = Subset(dataset, split['train_indices'])
    loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=torch.cuda.is_available(),
    )

    common_kwargs = dict(
        feature_dim=config['model']['feature_dim'],
        clip_num=config['model']['clip_num'],
        clip_len=config['model']['clip_len'],
        num_classes=config['model']['num_classes'],
        backbone_name=config['model']['backbone'],
        temporal_module=config['model'].get('temporal_module', 'gated_mlp'),
    )
    teacher = LiteCueNet(**common_kwargs).to(device)
    teacher.load_state_dict(load_state_dict(args.teacher_checkpoint, device), strict=False)
    teacher.eval()

    student = LiteCueNet(
        **common_kwargs,
        token_dropout=config['model'].get('token_dropout', 0.0),
        use_temporal_diff=config['model'].get('use_temporal_diff', False),
        use_frequency_branch=config['model'].get('use_frequency_branch', False),
        frequency_fuse_block=config['model'].get('frequency_fuse_block', 2),
    ).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 1e-4))

    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        total = 0
        for images, labels in tqdm(loader, desc=f'Distill epoch {epoch}/{args.epochs}'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                teacher_logits, _ = teacher(images)
            student_logits, _ = student(images)

            ce_loss = F.cross_entropy(student_logits, labels)
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / args.temperature, dim=1),
                F.softmax(teacher_logits / args.temperature, dim=1),
                reduction='batchmean',
            ) * (args.temperature ** 2)
            loss = args.alpha * ce_loss + (1.0 - args.alpha) * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        avg_loss = total_loss / max(1, total)
        print(f'Epoch {epoch}: distill_loss={avg_loss:.4f}')
        torch.save(
            {
                'epoch': epoch,
                'state_dict': student.state_dict(),
                'teacher_checkpoint': args.teacher_checkpoint,
                'temperature': args.temperature,
                'alpha': args.alpha,
            },
            os.path.join(args.save_dir, f'distill_epoch_{epoch:03d}.pth.tar'),
        )


if __name__ == '__main__':
    main()
