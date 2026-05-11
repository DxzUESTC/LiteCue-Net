import argparse
import csv
import json
import os
from pathlib import Path


KEY_METRICS = [
    'auc',
    'acc',
    'balanced_acc',
    'ap',
    'eer',
    'tpr_at_fpr_1',
    'tpr_at_fpr_0_1',
]


def collect_metrics(root):
    rows = []
    for path in Path(root).rglob('metrics.json'):
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        metrics = payload.get('metrics', {})
        row = {
            'run': str(path.parent.relative_to(root)),
            'target_dataset': payload.get('target_dataset', ''),
            'checkpoint_path': payload.get('checkpoint_path', ''),
            'num_forward': payload.get('num_forward', ''),
            'total_samples': payload.get('total_samples', ''),
            'real_samples': payload.get('real_samples', ''),
            'fake_samples': payload.get('fake_samples', ''),
        }
        for key in KEY_METRICS:
            row[key] = metrics.get(key, '')
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description='Aggregate LiteCue-Net ablation metrics.')
    parser.add_argument('--root', type=str, default='logs/crosstest', help='Root directory containing metrics.json files')
    parser.add_argument('--save_path', type=str, default='logs/crosstest/ablation_summary.csv')
    args = parser.parse_args()

    rows = collect_metrics(args.root)
    if not rows:
        print(f'No metrics.json found under {args.root}')
        return

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(args.save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Saved ablation summary to {args.save_path}')
    print(f'Collected {len(rows)} runs.')


if __name__ == '__main__':
    main()
