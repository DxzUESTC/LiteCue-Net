import cv2
import os
import numpy as np
import glob
from pathlib import Path
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from tqdm import tqdm

# ================= 配置区域 =================
CROP_SIZE = 224       # 输出图像尺寸
JPEG_QUALITY = 100     # JPG 质量 (1-100)，100 质量最高
# ===========================================

class FFIWPairedProcessor:
    def __init__(self):
        # 初始化 InsightFace (CUDA)
        print("正在加载 InsightFace (CUDA)...")
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # det_size=(640, 640) 是性价比最高的选择
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("模型加载完成。")

    def get_target_face(self, faces, mask_frame):
        """利用Mask找出被篡改的那张脸"""
        if mask_frame is None: return None
        
        # 快速处理 mask：如果是3通道直接取单通道，不进行完整的 cvtColor 以节省微小开销
        if len(mask_frame.shape) == 3:
            gray_mask = mask_frame[:, :, 0] # 假设 mask 是黑白的，RGB 通道值相同
        else:
            gray_mask = mask_frame

        # 快速二值化
        _, binary_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)
        
        max_score = 0
        target_face = None

        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            h, w = binary_mask.shape
            
            # 快速边界截断
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # 统计重叠区域
            mask_roi = binary_mask[y1:y2, x1:x2]
            score = cv2.countNonZero(mask_roi) # 比 np.count_nonzero 稍微快一点点
            
            if score > max_score:
                max_score = score
                target_face = face
        
        if max_score > 50:
            return target_face
        return None

    def process_pair(self, fake_path, real_path, mask_path, output_dir_target, output_dir_source):
        """处理单组视频 - 提取全部帧 - 保存为 JPG"""
        
        # 预先创建目录
        if not os.path.exists(output_dir_target): os.makedirs(output_dir_target)
        if not os.path.exists(output_dir_source): os.makedirs(output_dir_source)

        cap_fake = cv2.VideoCapture(fake_path)
        cap_real = cv2.VideoCapture(real_path)
        cap_mask = cv2.VideoCapture(mask_path)

        if not cap_fake.isOpened() or not cap_real.isOpened() or not cap_mask.isOpened():
            failed = []
            if not cap_fake.isOpened(): failed.append("fake")
            if not cap_real.isOpened(): failed.append("real")
            if not cap_mask.isOpened(): failed.append("mask")
            print(f"  错误: 无法打开视频文件 ({', '.join(failed)})")
            return

        idx = 0
        saved_count = 0
        total_frames = 0
        no_face_count = 0
        no_target_count = 0
        
        # 预设 JPG 参数，避免循环内重复创建对象
        jpg_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

        while True:
            ret_f, frame_fake = cap_fake.read()
            ret_r, frame_real = cap_real.read()
            ret_m, frame_mask = cap_mask.read()

            if not (ret_f and ret_r and ret_m):
                break

            total_frames += 1

            # 1. 在 Fake 帧检测人脸
            faces = self.app.get(frame_fake)
            
            if len(faces) > 0:
                # 2. 结合 Mask 锁定目标
                target_face = self.get_target_face(faces, frame_mask)

                if target_face is not None:
                    try:
                        # 3. 对齐矩阵
                        M = face_align.estimate_norm(target_face.kps, image_size=CROP_SIZE)

                        # 4. 同步裁剪 (双线性插值速度快)
                        crop_fake = cv2.warpAffine(frame_fake, M, (CROP_SIZE, CROP_SIZE), borderValue=0.0)
                        crop_real = cv2.warpAffine(frame_real, M, (CROP_SIZE, CROP_SIZE), borderValue=0.0)

                        # 5. 保存为 JPG，格式 frame_00000.jpg
                        file_name = f"frame_{idx:05d}.jpg"
                        
                        cv2.imwrite(os.path.join(output_dir_target, file_name), crop_fake, jpg_params)
                        cv2.imwrite(os.path.join(output_dir_source, file_name), crop_real, jpg_params)
                        saved_count += 1
                        
                    except Exception as e:
                        print(f"  警告: 处理帧 {idx} 时出错: {e}")
                else:
                    no_target_count += 1
            else:
                no_face_count += 1
            
            idx += 1

        cap_fake.release()
        cap_real.release()
        cap_mask.release()
        
        # 输出处理统计信息
        video_name = os.path.basename(fake_path)
        if saved_count == 0:
            print(f"  [WARN] {video_name}: 总帧数={total_frames}, 保存=0, 无人脸={no_face_count}, 无匹配目标={no_target_count}")
        else:
            print(f"  [OK] {video_name}: 总帧数={total_frames}, 已保存={saved_count} 帧")

def main():
    # ================= 路径配置 =================
    # FFIW 原始数据根目录（注意：实际数据在 FFIW10K-v1-release 子目录下）
    DATASET_ROOT = "D:/01_Lab/Project/LiteCue-Net/data/raw_videos/FFIW10K-v1-release-test"  
    
    # 输出根目录
    OUTPUT_ROOT = "D:/01_Lab/Project/LiteCue-Net/data/clips/FFIW10K-v1-release-test"
    # ===========================================

    processor = FFIWPairedProcessor()
    
    # 遍历 train、val 和 test
    sub_sets = ['train', 'val', 'test']

    for subset in sub_sets:
        print(f"\n正在处理: {subset} ...")
        
        # 搜索伪造视频（使用 pathlib 确保路径正确）
        fake_search_path = Path(DATASET_ROOT) / "target" / subset / "*.mp4"
        print(f"搜索路径: {fake_search_path}")
        fake_files = list(Path(DATASET_ROOT).glob(f"target/{subset}/*.mp4"))
        print(f"找到 {len(fake_files)} 个视频文件")
        
        if not fake_files:
            print(f"警告: 在 {fake_search_path} 下未找到任何 .mp4 文件，跳过")
            continue

        processed_count = 0
        skipped_count = 0
        
        for fake_path in tqdm(fake_files, desc=f"处理 {subset}"):
            # 转换为 Path 对象以便处理
            fake_path = Path(fake_path)
            filename = fake_path.name
            video_id = fake_path.stem  # 例如 train_00000000
            name_without_ext = fake_path.stem
            ext = fake_path.suffix  # 分离文件名和扩展名

            # 构造原始路径
            real_path = Path(DATASET_ROOT) / "source" / subset / filename
            # mask文件名格式: val_00000000_mask.mp4
            mask_filename = f"{name_without_ext}_mask{ext}"
            mask_path = Path(DATASET_ROOT) / "target_mask" / subset / mask_filename

            if not real_path.exists() or not mask_path.exists():
                missing = []
                if not real_path.exists(): missing.append("source")
                if not mask_path.exists(): missing.append("mask")
                if skipped_count < 3:  # 只显示前3个警告，避免刷屏
                    print(f"  警告: {filename} 缺少文件 ({', '.join(missing)})")
                skipped_count += 1
                continue

            # === 关键修改：输出路径完全匹配原结构 ===
            # output/target/train/video_id/
            out_target_dir = Path(OUTPUT_ROOT) / "target" / subset / video_id
            # output/source/train/video_id/
            out_source_dir = Path(OUTPUT_ROOT) / "source" / subset / video_id
            
            # 如果想断点续传（如果文件夹里已经有图了就跳过），可以取消下面两行的注释
            if out_target_dir.exists() and len(list(out_target_dir.iterdir())) > 0:
                continue

            processor.process_pair(str(fake_path), str(real_path), str(mask_path), str(out_target_dir), str(out_source_dir))
            processed_count += 1
        
        print(f"{subset} 处理完成: 处理={processed_count}, 跳过={skipped_count}, 总计={len(fake_files)}")

if __name__ == "__main__":
    main()