import numpy as np

class LiteCueSampler:
    """
    LiteCue-Net 专用时序采样器
    实现 "Global Sparse (Inter-Clip) + Local Dense (Intra-Clip)" 策略
    对应 TFCU 论文中的采样逻辑：将视频划分为 M 段，每段采样 K 帧连续图像。
    """
    def __init__(self, clip_num=16, clip_len=4):
        """
        Args:
            clip_num (int): 全局采样的片段数量 (M)。TFCU 论文中通常为 16 或 32。
            clip_len (int): 每个片段内的连续帧数 (K)。TFCU 固定为 4。
        """
        self.clip_num = clip_num
        self.clip_len = clip_len

    def __call__(self, total_frames, mode='train'):
        """
        根据模式生成采样索引列表。

        Args:
            total_frames (int): 视频的总帧数
            mode (str): 'train' (随机采样) 或 'val'/'test' (中心/均匀采样)

        Returns:
            np.ndarray: 形状为 (clip_num * clip_len,) 的索引数组，已排序。
        """
        # 1. 异常处理：如果视频太短，不够采样的，则采用"循环填充"策略
        # 比如视频只有 10 帧，但我们需要 16*4=64 帧，逻辑上需要重复读取
        required_frames = self.clip_num * self.clip_len
        
        # 生成原始的候选索引池 (0, 1, ..., total-1)
        # 如果视频太短，通过 tile 进行复制扩展，确保 logic 不会 crash
        if total_frames < required_frames:
            # 简单的策略：将索引池复制多份直到足够长，虽然物理上是重复的，但能保证 tensor shape 对齐
            base_indices = np.arange(total_frames)
            tile_count = (required_frames // total_frames) + 1
            extended_indices = np.tile(base_indices, tile_count)[:required_frames * 2] # 多取一点buffer
            # 更新 total_frames 为虚拟长度
            total_frames = len(extended_indices)
            indices_pool = extended_indices
        else:
            indices_pool = np.arange(total_frames)

        # 2. 将视频划分为 M 个片段 (Segments)
        # interval 是每个片段的理论最大长度
        interval = total_frames // self.clip_num
        
        sampled_indices = []

        for i in range(self.clip_num):
            # 当前片段的起始和结束边界
            seg_start = i * interval
            seg_end = (i + 1) * interval

            # 确保片段内至少有 clip_len 帧的空间
            # 如果 interval < clip_len (极端情况)，则在该片段范围内尽量取
            if seg_end - seg_start < self.clip_len:
                # 这种情况下直接取片段起点后的 K 帧 (可能会超出 seg_end 但在 total 内)
                start_idx = seg_start
            else:
                # 正常情况：在片段允许的范围内采样
                # 留出 clip_len 的余量，防止溢出
                valid_start_range = seg_end - self.clip_len - seg_start
                
                if valid_start_range <= 0:
                    start_idx = seg_start
                else:
                    if mode == 'train':
                        # [训练]：段内随机起点 (Random Shift) -> 数据增强
                        offset = np.random.randint(0, valid_start_range)
                        start_idx = seg_start + offset
                    else:
                        # [推理]：段内中心起点 (Center Crop) -> 结果确定性
                        start_idx = seg_start + (valid_start_range // 2)

            # 3. 获取连续的 K 帧
            # 注意：这里的 start_idx 是相对于 indices_pool 的索引
            # 如果原始视频太短用了 padding，indices_pool[idx] 会映射回真实的 img_idx
            clip_indices = [indices_pool[start_idx + k] for k in range(self.clip_len)]
            sampled_indices.extend(clip_indices)

        # 转换为 numpy 数组并返回
        # 最终 shape: (M * K, ) -> 后续 Dataset 会 reshape 成 (M, K, C, H, W)
        return np.array(sampled_indices, dtype=np.int32)

# ==========================================
# 单元测试 (Unit Test)
# ==========================================
if __name__ == "__main__":
    # 模拟几种边缘情况进行测试
    sampler = LiteCueSampler(clip_num=16, clip_len=4)
    
    print("Test 1: Normal Video (300 frames)")
    indices = sampler(300, mode='train')
    print(f"Output shape: {indices.shape}") # 应该是 (64,)
    print(f"Sample (First 8): {indices[:8]}")
    # 验证连续性：每4个一组，组内应该是连续的 (例如 10, 11, 12, 13)
    assert indices[1] == indices[0] + 1
    
    print("\nTest 2: Short Video (10 frames) - Check Padding")
    indices = sampler(10, mode='val')
    print(f"Output shape: {indices.shape}")
    print(f"Sample (First 12): {indices[:12]}")
    # 验证是否包含重复帧 (因为只有10帧)
    
    print("\nTest 3: Inference Stability")
    indices_1 = sampler(100, mode='val')
    indices_2 = sampler(100, mode='val')
    # 验证两次推理采样是否完全一致
    assert np.array_equal(indices_1, indices_2)
    print("Inference is deterministic. Pass.")