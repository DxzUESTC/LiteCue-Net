import os
import cv2
import numpy as np
import insightface
from insightface.utils import face_align

class FaceExtractor:
    def __init__(self, det_size=(640, 640), image_size=(224, 224), device = 'cuda'):
        """
        Args:
            det-size: 检测器输入尺寸
            image-size: 人脸对齐后输出尺寸
            device: 设备
        """
        self.image_size = image_size
        
        print(f"正在加载 RetinaFace 模型到 {device} ...")

        # 1. 设置运行后端
        # 如果有显卡，使用 CUDAExecutionProvider，否则使用 CPU
        if device == 'cuda':
            providers = ['CUDAExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # 2. 初始化 FaceAnalysis 对象
        self.app = insightface.app.FaceAnalysis(
            name = 'buffalo_l',
            root = '~/.insightface/models',
            providers = providers
        )

        # 3. 准备模型
        self.app.prepare(ctx_id=0, det_size=det_size)
        print("Finished loading Face Analysis model.")

    def process_video(self, video_path, output_dir):
        """
        处理单个视频的核心逻辑
        流程：读取视频 -> 逐帧检测 -> 挑出最大人脸 -> 对齐裁剪 -> 保存图片

        Args:
            video_path: 原始视频文件的路径
            output_dir: 结果保存的文件夹路径
        """
        # 1. 检查输出文件夹是否存在，不存在就创建
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"created output directory: {output_dir}")

        # 2. 打开视频文件
        cap = cv2.VideoCapture(video_path)
        # 检查视频是否成功打开
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        # 3. 逐帧循环处理
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 视频读取完毕，退出循环
            # A.人脸检测
            faces = self.app.get(frame)
            if len(faces) == 0:
                frame_idx += 1
                continue

            # B.筛选人脸，选择面积最大的人脸
            target_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]

            # C.人脸对齐
            aligned_face = face_align.norm_crop(frame, landmark=target_face.kps, image_size=112)

            # D.调整尺寸，调整到224x224
            if self.image_size != (112, 112):
                aligned_face = cv2.resize(aligned_face, self.image_size)

            # E.保存图片
            save_name = f"frame_{frame_idx:05d}.jpg"
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, aligned_face)

            if frame_idx % 100 == 0:
                print(f"processed frame {frame_idx}")
            frame_idx += 1

        # 4. 循环结束，释放视频资源
        cap.release()
        print(f"Finished processing video: {video_path}")

if __name__ == "__main__":
    extractor = FaceExtractor(image_size = (224, 224), device='cuda')
    test_video_path = "./data/raw/myphone/1.mp4"
    test_output_dir = "./data/processed"
    extractor.process_video(test_video_path, test_output_dir)