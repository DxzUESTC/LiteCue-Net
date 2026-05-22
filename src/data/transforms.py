import io
import random
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T


class RandomJPEGCompression:
    """PIL 图像级 JPEG 重压缩增强，用来模拟平台转码和低质量压缩。"""
    def __init__(self, quality_range=(35, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        low, high = self.quality_range
        quality = random.randint(int(low), int(high))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomResolutionDegradation:
    """随机降采样再升采样，模拟分辨率、码率和平台缩放差异。"""
    def __init__(self, scale_range=(0.35, 0.9), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        width, height = img.size
        scale = random.uniform(*self.scale_range)
        small_size = (max(8, int(width * scale)), max(8, int(height * scale)))
        resample_down = random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
        resample_up = random.choice([Image.BILINEAR, Image.BICUBIC])
        return img.resize(small_size, resample_down).resize((width, height), resample_up)


class RandomBlurSharpen:
    """随机模糊或锐化，覆盖运动模糊、对焦差异和后处理锐化。"""
    def __init__(self, blur_radius=(0.2, 1.5), sharpen_factor=(1.2, 2.0), p=0.4):
        self.blur_radius = blur_radius
        self.sharpen_factor = sharpen_factor
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        if random.random() < 0.5:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.blur_radius)))
        return ImageEnhance.Sharpness(img).enhance(random.uniform(*self.sharpen_factor))

def get_transforms(mode='train', img_size=224, use_aug=False, aug_level='light'):
    """
    预处理与数据增强工厂函数
    
    Args:
        mode (str): 'train' 或 'val'
        img_size (int): 目标尺寸 (224)
        use_aug (bool): 是否开启强数据增强 (LiteCue-Net/TFCU 默认 False)
                        保留此参数是为了兼容 main.py 的调用接口
    """
    # ImageNet 标准归一化 (MobileNetV4 需要)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == 'train':
        transforms_list = [
            T.Resize((img_size, img_size)),
            # 即使不开启强增强，随机水平翻转通常也是安全的且推荐的
            # 如果想严格对应"完全无增强"，可以注释掉下面这行
            T.RandomHorizontalFlip(p=0.5), 
        ]
        
        if use_aug:
            if aug_level in ('medium', 'strong_domain'):
                transforms_list.extend([
                    RandomJPEGCompression(quality_range=(45, 95), p=0.5),
                    RandomResolutionDegradation(scale_range=(0.45, 0.95), p=0.4),
                    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.03),
                ])
            if aug_level == 'strong_domain':
                transforms_list.extend([
                    RandomJPEGCompression(quality_range=(25, 85), p=0.7),
                    RandomResolutionDegradation(scale_range=(0.30, 0.90), p=0.6),
                    RandomBlurSharpen(p=0.5),
                    T.RandomAffine(
                        degrees=6,
                        translate=(0.04, 0.04),
                        scale=(0.92, 1.08),
                        shear=3,
                    ),
                ])
            
        transforms_list.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        
        return T.Compose(transforms_list)
    
    else:
        # 验证集/测试集：只做确定性变换
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])