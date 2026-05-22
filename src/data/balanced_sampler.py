from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler


def _metadata_key(item, key):
    if key == 'label':
        return str(item.get('label', 'unknown'))
    if key == 'dataset':
        return str(item.get('dataset') or _infer_dataset(item.get('path', '')))
    if key == 'method':
        return str(item.get('method') or _infer_method(item))
    if key == 'compression':
        return str(item.get('compression', 'unknown'))
    return str(item.get(key, 'unknown'))


def _infer_dataset(path):
    normalized = str(path).replace('\\', '/')
    for name in ['FaceForensics++', 'Celeb-DF-v2', 'FFIW10K', 'DFDC', 'DeeperForensics', 'WildDeepfake']:
        if name in normalized:
            return name
    parts = [p for p in normalized.split('/') if p]
    return parts[0] if parts else 'unknown'


def _infer_method(item):
    if item.get('label', 0) == 0:
        return 'real'
    path = str(item.get('path', '')).replace('\\', '/')
    for method in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'Celeb-synthesis', 'target']:
        if method in path:
            return method
    return 'fake_unknown'


def build_weighted_sampler(dataset, indices=None, balance_keys=None, replacement=True):
    """
    基于 label/dataset/method/compression 等元数据构建 WeightedRandomSampler。
    该函数适配 torch.utils.data.Subset 包装前后的 LiteCueDataset 数据。
    """
    if balance_keys is None:
        balance_keys = ['label']
    if isinstance(balance_keys, str):
        balance_keys = [balance_keys]

    base_dataset = getattr(dataset, 'dataset', dataset)
    if indices is None:
        indices = list(getattr(dataset, 'indices', range(len(base_dataset))))

    items = [base_dataset.data[i] for i in indices]
    counters = {
        key: Counter(_metadata_key(item, key) for item in items)
        for key in balance_keys
    }

    weights = []
    for item in items:
        weight = 1.0
        for key in balance_keys:
            value = _metadata_key(item, key)
            count = max(1, counters[key][value])
            weight *= 1.0 / count
        weights.append(weight)

    weights = torch.as_tensor(weights, dtype=torch.double)
    weights = weights / weights.mean().clamp_min(1e-12)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=replacement,
    )
