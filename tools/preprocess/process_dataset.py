import os
import glob
from tqdm import tqdm
from extract_faces import FaceExtractor

# 定义批量处理函数
def process_dataset(input_root, output_root, gpu_id=0):
    """
    """
    # 1. 实例化 FaceExtractor 对象
    print(f"initializing FaceExtractor on GPU {gpu_id}...")
    extractor = FaceExtractor(det_size=(640, 640), image_size=(224, 224), device='cuda')

    # 2. 扫描所有视频文件
    video_paths = []
    print(f"scanning videos in {input_root} ...")

    extensions = ('.mp4', '.avi', '.mov', '.mkv')

    for root, dirs, files in os.walk(input_root):
        # 跳过 c40 文件夹(在遍历子目录前移除)
        dirs[:] = [d for d in dirs if d.lower() != 'c40']
        
        for file in files:
            if file.lower().endswith(extensions):
                full_path = os.path.join(root, file)
                video_paths.append(full_path)
    
    print(f"found {len(video_paths)} videos.")

    # 3. 使用 tqdm 创建进度条循环
    # enumerate 用于获取序号，path 是当前视频路径
    for i, video_path in enumerate(tqdm(video_paths, desc="Processing")):

        try:
            # A. 计算输出路径（镜像结构）
            # a. 计算相对路径
            relative_path = os.path.relpath(video_path, input_root)

            # b. 去掉文件后缀
            no_ext_path = os.path.splitext(relative_path)[0]

            # c. 拼接输出路径
            target_dir = os.path.join(output_root, no_ext_path)

            # B. 断点续传检查
            if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
                # print(f"skipping {video_path}, output already exists.")
                continue

            # C. 调用核心功能处理视频
            extractor.process_video(video_path, target_dir)
        
        except Exception as e:
            print(f"Error processing video {video_path}")
            print(f"Error details: {e}")
            with open("error_log.txt", "a") as f:
                f.write(f"{video_path} | {e}\n")

    print("All done.")

if __name__ == "__main__":
    INPUT_ROOT = "data/raw_videos/FFIW10K-v1"
    OUTPUT_ROOT = "data/clips/FFIW10K-v1"
    process_dataset(INPUT_ROOT, OUTPUT_ROOT, gpu_id=0)