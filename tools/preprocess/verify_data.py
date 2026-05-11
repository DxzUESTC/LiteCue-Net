import argparse
import pickle
import os
import sys
import random
import re
from pathlib import Path
from tqdm import tqdm

# -----------------------------------------------------------
# Hack: 将项目根目录加入路径，确保能导入 src.data.sampler
# -----------------------------------------------------------
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.data.sampler import LiteCueSampler
except ImportError:
    print("Error: Could not import 'src.data.sampler'.")
    print(f"Please run this script from the project root directory: {project_root}")
    sys.exit(1)

# -----------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------
def extract_frame_num(filename):
    """
    从文件名中提取帧序号。
    例如: "0001.png" -> 1, "frame_015.jpg" -> 15
    """
    nums = re.findall(r'\d+', filename)
    if nums:
        return int(nums[-1])
    return -1

def verify_video_continuity(video_info, data_root, sampler, mode='train'):
    """
    对单个视频进行采样并验证物理连续性
    """
    # [核心修改] 动态拼接路径：data_root + relative_path_from_index
    # 这样无论数据搬到哪里，只要相对结构不变都能找到
    video_path = os.path.join(data_root, video_info['path'])
    
    if not os.path.exists(video_path):
        return False, f"Path not found: {video_path}"

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    try:
        # 获取目录下所有图片文件并排序
        all_files = sorted([
            f for f in os.listdir(video_path) 
            if os.path.splitext(f)[1].lower() in image_extensions
        ])
    except Exception as e:
        return False, f"IO Error: {str(e)}"

    total_frames = len(all_files)
    if total_frames == 0:
        return False, "No frames found"
    
    # 获取采样索引
    indices = sampler(total_frames, mode=mode)
    
    # 检查每一个 Clip
    clip_len = sampler.clip_len 
    num_clips = len(indices) // clip_len
    
    for i in range(num_clips):
        current_clip_indices = indices[i*clip_len : (i+1)*clip_len]
        
        # 1. 索引层面的连续性检查
        for k in range(clip_len - 1):
            if current_clip_indices[k+1] != current_clip_indices[k] + 1:
                return False, f"Sampler Index Discontinuity at clip {i}: {current_clip_indices}"

        # 2. 物理层面的连续性检查
        try:
            filenames = [all_files[idx] for idx in current_clip_indices]
        except IndexError:
             return False, f"Index out of bounds. Total files: {total_frames}, Index: {max(current_clip_indices)}"

        frame_nums = [extract_frame_num(f) for f in filenames]
        
        for k in range(clip_len - 1):
            diff = frame_nums[k+1] - frame_nums[k]
            if diff != 1:
                return False, f"Physical Frame Discontinuity in Clip {i}!\nFiles: {filenames}\nNums: {frame_nums}"

    return True, "OK"

# -----------------------------------------------------------
# 主程序
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Verify data continuity for LiteCue-Net")
    parser.add_argument("--index_path", type=str, default="data/dataset_index.pkl", help="Path to dataset index pickle")
    parser.add_argument("--data_root", type=str, required=True, help="[Must Provide] Root directory of the clips (e.g. data/clips)")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of videos to randomly check (0 for all)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val"], help="Sampling mode to verify")
    
    args = parser.parse_args()

    # 1. 加载索引
    if not os.path.exists(args.index_path):
        print(f"Error: Index file not found at {args.index_path}")
        return

    print(f"Loading index from {args.index_path}...")
    with open(args.index_path, 'rb') as f:
        dataset_index = pickle.load(f)
    
    print(f"Total videos in index: {len(dataset_index)}")

    # 2. 确定检查样本
    if args.num_samples == 0 or args.num_samples > len(dataset_index):
        samples = dataset_index
        print("Verifying ALL videos...")
    else:
        samples = random.sample(dataset_index, args.num_samples)
        print(f"Verifying {len(samples)} random videos from root: {args.data_root} ...")

    # 3. 初始化采样器
    sampler = LiteCueSampler(clip_num=16, clip_len=4)
    
    # 4. 开始验证
    error_list = []
    
    for video_info in tqdm(samples):
        # 传入 args.data_root
        is_valid, msg = verify_video_continuity(video_info, args.data_root, sampler, mode=args.mode)
        
        if not is_valid:
            error_list.append({
                "video_id": video_info['video_id'],
                "path": video_info['path'],
                "error": msg
            })

    # 5. 报告结果
    print("\n" + "="*50)
    print("Verification Summary")
    print("="*50)
    
    if len(error_list) == 0:
        print(f"PASSED! All {len(samples)} checked videos have physically consecutive frames.")
    else:
        print(f"FAILED! Found issues in {len(error_list)} videos.")
        print("Sample errors:")
        for i, err in enumerate(error_list[:5]): 
            print(f"  {i+1}. [{err['video_id']}] {err['error']}")
        
        print("\nFix Tip:")
        print("If 'Path not found', check if --data_root matches where your 'clips' folder actually is.")

if __name__ == "__main__":
    main()