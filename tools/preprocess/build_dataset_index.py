import os
import argparse
import pickle
import glob
from tqdm import tqdm
from pathlib import Path

# 启发式映射，涵盖常见的目录名
LABEL_MAP = {
    # 真实视频目录名
    'Celeb-real': 0,
    'YouTube-real': 0,
    'original_sequences': 0,
    
    # 伪造视频目录名
    'manipulated_sequences': 1,
    'Celeb-synthesis': 1
}

# 支持的图片扩展名
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}

# 根据路径中的父目录名称推断标签
def get_label_from_path(path):
    parts = path.parts
    for part in parts:
        if part in LABEL_MAP:
            return LABEL_MAP[part]
    return None


def infer_dataset_name(path):
    normalized = str(path).replace('\\', '/')
    known = [
        'FaceForensics++',
        'Celeb-DF-v2',
        'FFIW10K',
        'DFDC',
        'DeeperForensics',
        'WildDeepfake',
    ]
    for name in known:
        if name in normalized:
            return name
    parts = [p for p in Path(normalized).parts if p]
    return parts[0] if parts else 'unknown'


def infer_forgery_method(path, label):
    normalized = str(path).replace('\\', '/')
    if label == 0:
        return 'real'
    for method in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'Celeb-synthesis', 'target']:
        if method in normalized:
            return method
    return 'fake_unknown'


def infer_compression(path):
    normalized = str(path).replace('\\', '/')
    for token in ['c23', 'c40', 'raw', 'low', 'hq', 'lq']:
        if token in normalized:
            return token
    return 'unknown'


def infer_source_video_id(video_id):
    # FF++ 常见伪造视频名包含源/目标身份，保留第一个 token 作为粗粒度源 ID。
    for sep in ['__', '_', '-']:
        if sep in video_id:
            return video_id.split(sep)[0]
    return video_id

# 扫描整个数据目录，寻找包含图片的子目录
def scan_dataset(root_dir):
    dataset_index = []
    root_path = Path(root_dir)

    # 1. 递归查找所有包含图片的文件夹
    # 使用 os.walk 遍历目录
    print(f"Scanning directory: {root_dir}...")

    video_folders = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 检查该目录下是否有图片文件
        has_images = any(os.path.splitext(f)[1].lower() in IMAGE_EXT for f in filenames)
        if has_images:
            video_folders.append(Path(dirpath))
    
    print(f"Found {len(video_folders)} potential video clips. Processing metadata...")

    # 2. 遍历每个视频文件夹，收集元数据
    # 使用 tqdm 创建进度条
    for folder_path in tqdm(video_folders):
        folder_p = Path(folder_path)

        # A. 获取 VideoID（文件夹名称）
        video_id = folder_p.name

        # B. 获取标签
        label = get_label_from_path(folder_p)
        if label is None:
            print(f"Warning: Could not determine label for {folder_p}, skipping.")
            continue

        # C. 计算帧数
        # 扫描文件夹内的图片
        frames = [
            f.name for f in os.scandir(folder_path)
            if f.is_file() and os.path.splitext(f.name)[1].lower() in IMAGE_EXT
        ]
        frame_count = len(frames)

        if frame_count < 1:
            continue

        # D. 构建索引项
        # 保存相对路径节省空间保持可移植性
        try:
            rel_path = folder_p.relative_to(root_path)
        except ValueError:
            rel_path = folder_path
        
        item = {
            'video_id': video_id,
            'label': label,
            'num_frames': frame_count,
            'path': str(rel_path),
            'abs_path': str(folder_p),
            'dataset': infer_dataset_name(folder_p),
            'method': infer_forgery_method(folder_p, label),
            'compression': infer_compression(folder_p),
            'source_video_id': infer_source_video_id(video_id),
        }

        dataset_index.append(item)
    
    return dataset_index

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_root", type=str, required=True, help="")
    parser.add_argument("--save_path", type=str, default="data/dataset_index.pkl", help="")

    args = parser.parse_args()

    # 1. 扫描
    data = scan_dataset(args.data_root)

    # 2. 统计信息
    real_count = sum(1 for item in data if item['label'] == 0)
    fake_count = sum(1 for item in data if item['label'] == 1) 

    print(f"\nScan complete. Found {len(data)} video clips.")
    print(f"  Real videos: {real_count}")
    print(f"  Fake videos: {fake_count}\n")

    if len(data) == 0:
        print("No data found, exiting without saving.")
        return 
    
    # 3. 保存索引
    # 检查 save_path 是否是目录（如果是目录，给出错误提示）
    if os.path.isdir(args.save_path):
        print(f"\n[Error] --save_path should be a file path, not a directory!")
        print(f"  You provided: {args.save_path}")
        print(f"  Expected format: path/to/directory/filename.pkl")
        print(f"  Example: tools/analysis/crosstestindex/celebdfv2_crosstest_index.pkl")
        return
    
    # 确保目录存在
    save_dir = os.path.dirname(args.save_path)
    if save_dir:  # 如果路径包含目录部分
        os.makedirs(save_dir, exist_ok=True)
    
    with open(args.save_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Dataset index saved to {args.save_path}")

    # 4. 示例输出前5条记录
    print("Sample records:")
    for item in data[:5]:
        print(item)

if __name__ == "__main__":
    main()