import os
import pickle
from tqdm import tqdm
from pathlib import Path

# 支持的图片扩展名
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}

def scan_split(root_dir, split_name):
    """
    扫描指定划分（train或val）的数据
    
    Args:
        root_dir: FFIW10K-v1-release的根目录
        split_name: 'train' 或 'val'
    
    Returns:
        dataset_index: 索引列表
    """
    dataset_index = []
    root_path = Path(root_dir)
    
    # source目录（真实视频，label=0）
    source_dir = root_path / 'source' / split_name
    # target目录（伪造视频，label=1）
    target_dir = root_path / 'target' / split_name
    
    print(f"\n扫描 {split_name} 划分...")
    print(f"  真实视频目录: {source_dir}")
    print(f"  伪造视频目录: {target_dir}")
    
    # 处理source（真实视频）
    if source_dir.exists():
        source_folders = [d for d in source_dir.iterdir() if d.is_dir()]
        print(f"  找到 {len(source_folders)} 个真实视频文件夹")
        
        for folder_path in tqdm(source_folders, desc=f"处理真实视频 ({split_name})"):
            # 扫描文件夹内的图片
            frames = [
                f.name for f in os.scandir(folder_path)
                if f.is_file() and os.path.splitext(f.name)[1].lower() in IMAGE_EXT
            ]
            frame_count = len(frames)
            
            if frame_count < 1:
                continue
            
            # 构建相对路径
            try:
                rel_path = folder_path.relative_to(root_path)
            except ValueError:
                rel_path = folder_path
            
            item = {
                'video_id': folder_path.name,
                'label': 0,  # 真实视频
                'num_frames': frame_count,
                'path': str(rel_path),
                'abs_path': str(folder_path),
                'split': split_name  # 添加划分信息：'train' 或 'val'
            }
            dataset_index.append(item)
    else:
        print(f"  警告: {source_dir} 不存在")
    
    # 处理target（伪造视频）
    if target_dir.exists():
        target_folders = [d for d in target_dir.iterdir() if d.is_dir()]
        print(f"  找到 {len(target_folders)} 个伪造视频文件夹")
        
        for folder_path in tqdm(target_folders, desc=f"处理伪造视频 ({split_name})"):
            # 扫描文件夹内的图片
            frames = [
                f.name for f in os.scandir(folder_path)
                if f.is_file() and os.path.splitext(f.name)[1].lower() in IMAGE_EXT
            ]
            frame_count = len(frames)
            
            if frame_count < 1:
                continue
            
            # 构建相对路径
            try:
                rel_path = folder_path.relative_to(root_path)
            except ValueError:
                rel_path = folder_path
            
            item = {
                'video_id': folder_path.name,
                'label': 1,  # 伪造视频
                'num_frames': frame_count,
                'path': str(rel_path),
                'abs_path': str(folder_path),
                'split': split_name  # 添加划分信息：'train' 或 'val'
            }
            dataset_index.append(item)
    else:
        print(f"  警告: {target_dir} 不存在")
    
    return dataset_index

def main():
    # 配置路径
    data_root = "data/clips/FFIW10K-v1-release-test"
    index_dir = "data/index"
    
    # 确保索引目录存在
    os.makedirs(index_dir, exist_ok=True)
    
    print("=" * 60)
    print("FFIW10K-v1-release-test 数据集索引建立")
    print("=" * 60)
    print(f"数据根目录: {data_root}")
    print(f"索引保存目录: {index_dir}")
    
    # 处理train划分
    print("\n" + "=" * 60)
    print("处理 TRAIN 划分")
    print("=" * 60)
    train_index = scan_split(data_root, 'train')
    
    # 统计train信息
    train_real = sum(1 for item in train_index if item['label'] == 0)
    train_fake = sum(1 for item in train_index if item['label'] == 1)
    
    print(f"\nTrain 划分统计:")
    print(f"  总视频数: {len(train_index)}")
    print(f"  真实视频: {train_real}")
    print(f"  伪造视频: {train_fake}")
    
    # 处理val划分
    print("\n" + "=" * 60)
    print("处理 VAL 划分")
    print("=" * 60)
    val_index = scan_split(data_root, 'val')
    
    # 统计val信息
    val_real = sum(1 for item in val_index if item['label'] == 0)
    val_fake = sum(1 for item in val_index if item['label'] == 1)
    
    print(f"\nVal 划分统计:")
    print(f"  总视频数: {len(val_index)}")
    print(f"  真实视频: {val_real}")
    print(f"  伪造视频: {val_fake}")

     # 处理test划分
    print("\n" + "=" * 60)
    print("处理 TEST 划分")
    print("=" * 60)
    test_index = scan_split(data_root, 'test')
    
    # 统计test信息
    test_real = sum(1 for item in test_index if item['label'] == 0)
    test_fake = sum(1 for item in test_index if item['label'] == 1)
    
    print(f"\nTest 划分统计:")
    print(f"  总视频数: {len(test_index)}")
    print(f"  真实视频: {test_real}")
    print(f"  伪造视频: {test_fake}")
    
    # 合并train和val索引，创建统一索引文件
    print("\n" + "=" * 60)
    print("创建统一索引文件")
    print("=" * 60)
    all_index = train_index + val_index + test_index
    
    # 总体统计
    total_count = len(all_index)
    total_real = train_real + val_real + test_real
    total_fake = train_fake + val_fake + test_fake
    
    print(f"\n总体统计:")
    print(f"  总视频数: {total_count}")
    print(f"  真实视频: {total_real}")
    print(f"  伪造视频: {total_fake}")
    print(f"  Train: {len(train_index)} ({len(train_index)/total_count*100:.1f}%)")
    print(f"  Val: {len(val_index)} ({len(val_index)/total_count*100:.1f}%)")
    print(f"  Test: {len(test_index)} ({len(test_index)/total_count*100:.1f}%)")
    
    # 保存三个索引文件
    print("\n" + "=" * 60)
    print("保存索引文件")
    print("=" * 60) 
    
    # 1. 统一索引文件（包含train、val和test）
    unified_save_path = os.path.join(index_dir, "FFIW10K-v1-release-test.pkl")
    with open(unified_save_path, 'wb') as f:
        pickle.dump(all_index, f)
    print(f"\n[OK] 统一索引已保存到: {unified_save_path}")
    print(f"  包含 {len(train_index)} 个train样本、 {len(val_index)} 个val样本和 {len(test_index)} 个test样本")

    # 2. 训练集索引文件
    train_save_path = os.path.join(index_dir, "FFIW10K-v1-release-test_train.pkl")
    with open(train_save_path, 'wb') as f:
        pickle.dump(train_index, f)
    print(f"\n[OK] Train 索引已保存到: {train_save_path}")
    print(f"  包含 {len(train_index)} 个样本")

    # 3. 验证集索引文件
    val_save_path = os.path.join(index_dir, "FFIW10K-v1-release-test_val.pkl")
    with open(val_save_path, 'wb') as f:
        pickle.dump(val_index, f)
    print(f"\n[OK] Val 索引已保存到: {val_save_path}")
    print(f"  包含 {len(val_index)} 个样本")

    # 4. 测试集索引文件
    test_save_path = os.path.join(index_dir, "FFIW10K-v1-release-test_test.pkl")
    with open(test_save_path, 'wb') as f:
        pickle.dump(test_index, f)
    print(f"\n[OK] Test 索引已保存到: {test_save_path}")
    print(f"  包含 {len(test_index)} 个样本")
    
    # 显示示例记录
    print("\n" + "=" * 60)
    print("示例记录 (Train)")
    print("=" * 60)
    for item in train_index[:3]:
        print(f"  {item}")
    
    print("\n示例记录 (Val)")
    print("=" * 60)
    for item in val_index[:3]:
        print(f"  {item}")
    
    print("\n" + "=" * 60)
    print("索引建立完成！")
    print("=" * 60)
    print(f"\n生成的索引文件:")
    print(f"  1. {unified_save_path} - 全部数据（train + val）")
    print(f"  2. {train_save_path} - 训练集")
    print(f"  3. {val_save_path} - 验证集")
    print(f"  4. {test_save_path} - 测试集")
if __name__ == "__main__":
    main()

