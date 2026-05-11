import torch
import timm
import einops

def check_environment():
    print(f"PyTorch version:{torch.__version__}")
    print(f"    CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    Device Name: {torch.cuda.get_device_name(0)}")
    
    print(f"Timm version: {timm.__version__}")

    # 尝试加载 MobileNetV4
    try:
        model_name = 'mobilenetv4_conv_small.e2400_r224_in1k'
        model = timm.create_model(model_name, pretrained=False)
        print(f"    Successfully loaded model: {model_name}")

        # 将模型设置为评估模式
        model.eval()

        # 测试 einops 是否正常，创建假数据
        x = torch.randn(1, 3, 224, 224)

        # 前向传播测试
        with torch.no_grad():
            output = model(x)
        print(f"    Forward pass OK. Output shape: {output.shape}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"加载 MobileNetV4 模型失败，检查 timm 版本\n错误信息: {e}")

if __name__ == "__main__":
    check_environment()