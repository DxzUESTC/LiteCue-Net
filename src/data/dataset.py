import os
import torch
import pickle
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
from .sampler import LiteCueSampler
from .augmentation import apply_clip_occlusion, apply_domain_randomization

class LiteCueDataset(Dataset):
    def __init__(
        self,
        index_path,
        data_root,
        transforms=None,
        mode='train',
        clip_num=16,
        clip_len=4,
        path_patterns=None,
        occlusion_cfg=None,
        domain_aug_cfg=None,
        return_metadata=False,
    ):
        """
        Args:
            index_path (str): dataset_index.pkl 的路径
            data_root (str): 视频片段(clips)的根目录
            transforms (callable): 预处理函数
            mode (str): 'train', 'val', or 'test'
            clip_num (int): 采样的片段数 M (默认 16)
            clip_len (int): 每个片段的帧数 K (默认 4)
            path_patterns (list, optional): 路径过滤模式列表。如果提供，只保留路径中包含任一模式的样本。
                                           例如: ["original_sequences", "manipulated_sequences"] 用于 FF++
            occlusion_cfg (dict, optional): 训练期遮挡增强配置，仅 mode='train' 时生效。见 doc/LiteCue-Net 遮挡增强说明.md
            domain_aug_cfg (dict, optional): 训练期域随机化增强，包含 occlusion/temporal 等组合增强。
            return_metadata (bool): 为 True 时额外返回 domain/method 等元信息。
        """
        self.data_root = data_root
        self.transforms = transforms
        self.mode = mode
        self.occlusion_cfg = occlusion_cfg
        self.domain_aug_cfg = domain_aug_cfg
        self.return_metadata = return_metadata
        
        # 1. 加载索引文件
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
            
        print(f"Loading dataset index from {index_path}...")
        with open(index_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        # 2. 根据路径模式过滤数据（如果提供了过滤模式）
        if path_patterns is not None and len(path_patterns) > 0:
            print(f"Filtering dataset by path patterns: {path_patterns}")
            original_count = len(raw_data)
            self.data = []
            for item in raw_data:
                item_path = item.get('path', '')
                # 统一路径分隔符为 '/' 以便跨平台兼容
                normalized_path = item_path.replace('\\', '/')
                
                # 支持精确路径匹配和简单包含匹配
                matched = False
                for pattern in path_patterns:
                    normalized_pattern = pattern.replace('\\', '/')
                    
                    # 如果模式包含路径分隔符，进行精确路径匹配（用于匹配特定子目录）
                    # 例如: "manipulated_sequences/Deepfakes" 只匹配 Deepfakes 子目录
                    if '/' in normalized_pattern:
                        if normalized_pattern in normalized_path:
                            matched = True
                            break
                    else:
                        # 简单包含匹配（用于匹配顶层目录，如 "original_sequences"）
                        if normalized_pattern in normalized_path:
                            matched = True
                            break
                
                if matched:
                    self.data.append(item)
            filtered_count = len(self.data)
            print(f"Filtered: {original_count} -> {filtered_count} samples (kept {filtered_count/original_count*100:.1f}%)")
        else:
            self.data = raw_data
            print(f"Loaded {len(self.data)} samples (no filtering applied)")
            
        # 3. 初始化采样器
        self.sampler = LiteCueSampler(clip_num=clip_num, clip_len=clip_len)
        self.clip_num = clip_num
        self.clip_len = clip_len

        self._build_metadata_maps()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            frames: Tensor shape (M, K, C, H, W) -> (16, 4, 3, 224, 224)
            label: int (0 or 1)
        """
        # 获取当前视频的元数据
        video_info = self.data[idx]
        video_id = video_info['video_id']
        label = video_info['label']
        num_frames = video_info['num_frames']
        rel_path = video_info['path']
        
        # 1. 动态拼接绝对路径
        video_dir = os.path.join(self.data_root, rel_path)
        
        # 2. 调用采样器获取索引
        # indices shape: (M * K,) -> (64,)
        indices = self.sampler(num_frames, mode=self.mode)
        
        # 3. 物理读取图片
        # 假设文件夹内的图片是按文件名排序的 (frame_0.jpg, frame_1.jpg...)
        # 为了效率，我们先获取该目录下所有图片名并排序
        # 优化：如果是生产环境，可以将 filenames 列表也缓存在 pkl 里，这里为了节省内存先实时listdir
        try:
            all_files = sorted([
                f for f in os.listdir(video_dir) 
                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))
            ])
        except FileNotFoundError:
            # 极少数情况路径不存在，返回全0的dummy数据防止训练中断
            print(f"[Warning] Path missing: {video_dir}")
            return self._get_dummy_data(), label

        # 根据采样索引读取帧
        frames = []
        for frame_idx in indices:
            # 防止索引越界 (虽然 sampler 做了保护，但为了健壮性)
            safe_idx = frame_idx if frame_idx < len(all_files) else 0
            img_name = all_files[safe_idx]
            img_path = os.path.join(video_dir, img_name)
            
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    if self.transforms:
                        img_tensor = self.transforms(img) # (C, H, W)
                    else:
                        img_tensor = T.ToTensor()(img)
                    frames.append(img_tensor)
            except Exception as e:
                print(f"[Error] Failed to load {img_path}: {e}")
                # 加载失败补一张黑图
                frames.append(torch.zeros(3, 224, 224))

        # 4. 堆叠与重塑 (Stack & Reshape)
        # 此时 frames 是 list of (C, H, W)
        video_tensor = torch.stack(frames) # (T, C, H, W), T = M * K
        
        # Reshape 为 (M, K, C, H, W) 以符合 TFCU/LiteCue-Net 的输入要求
        # M=16, K=4, C=3, H=224, W=224
        C, H, W = video_tensor.shape[1:]
        video_tensor = video_tensor.view(self.clip_num, self.clip_len, C, H, W)

        # 仅训练时对整段 clip 施加同一块空间遮挡，强制模型关注边缘/细节
        if self.mode == 'train' and self.occlusion_cfg and self.occlusion_cfg.get('enabled', False):
            video_tensor = apply_clip_occlusion(video_tensor, self.occlusion_cfg)

        if self.mode == 'train' and self.domain_aug_cfg and self.domain_aug_cfg.get('enabled', False):
            video_tensor = apply_domain_randomization(video_tensor, self.domain_aug_cfg)

        if self.return_metadata:
            return video_tensor, label, self.get_metadata(idx)

        return video_tensor, label

    def _get_dummy_data(self):
        """返回全0数据用于错误处理"""
        return torch.zeros(self.clip_num, self.clip_len, 3, 224, 224)

    def _build_metadata_maps(self):
        datasets = sorted({self._infer_dataset(item) for item in self.data})
        methods = sorted({self._infer_method(item) for item in self.data})
        self.dataset_to_id = {name: idx for idx, name in enumerate(datasets)}
        self.method_to_id = {name: idx for idx, name in enumerate(methods)}

    @staticmethod
    def _infer_dataset(item):
        if item.get('dataset'):
            return str(item['dataset'])
        path = item.get('path', '').replace('\\', '/')
        parts = [p for p in path.split('/') if p]
        known = [
            'FaceForensics++',
            'Celeb-DF-v2',
            'FFIW10K-v1-release-test',
            'DFDC',
            'DeeperForensics',
            'WildDeepfake',
        ]
        for name in known:
            if name in path:
                return name
        if parts:
            return parts[0]
        return 'unknown'

    @staticmethod
    def _infer_method(item):
        if item.get('method'):
            return str(item['method'])
        path = item.get('path', '').replace('\\', '/')
        for method in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'Celeb-synthesis', 'target']:
            if method in path:
                return method
        return 'real' if item.get('label', 0) == 0 else 'fake_unknown'

    def get_metadata(self, idx):
        item = self.data[idx]
        dataset = self._infer_dataset(item)
        method = self._infer_method(item)
        metadata = {
            'index': idx,
            'video_id': item.get('video_id', ''),
            'path': item.get('path', ''),
            'dataset': dataset,
            'dataset_id': self.dataset_to_id.get(dataset, 0),
            'method': method,
            'method_id': self.method_to_id.get(method, 0),
            'compression': item.get('compression', 'unknown'),
            'source_video_id': item.get('source_video_id', item.get('video_id', '')),
        }
        return metadata

# ==========================================
# 单元测试 (Unit Test)
# ==========================================
if __name__ == "__main__":
    import sys
    # 临时添加 transforms 路径以便测试
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from src.data.transforms import get_transforms

    # 配置测试参数 (修改为实际路径)
    INDEX_PATH = "data/dataset_index.pkl"
    DATA_ROOT = "data/clips"  

    if not os.path.exists(INDEX_PATH):
        print(f"Skipping test: {INDEX_PATH} not found.")
    else:
        # 1. 初始化 Dataset
        dataset = LiteCueDataset(
            index_path=INDEX_PATH,
            data_root=DATA_ROOT,
            transforms=get_transforms(mode='train'),
            mode='train'
        )
        
        print(f"Dataset loaded. Total videos: {len(dataset)}")
        
        # 2. 读取一个样本
        print("Fetching a sample...")
        frames, label = dataset[0]
        
        # 3. 验证形状
        # 预期: (16, 4, 3, 224, 224)
        print(f"Frames Shape: {frames.shape}")
        print(f"Label: {label}")
        
        assert frames.shape == (16, 4, 3, 224, 224), "Shape mismatch!"
        print("Test Passed: Data pipeline is ready.")