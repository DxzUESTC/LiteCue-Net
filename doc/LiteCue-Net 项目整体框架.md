## LiteCue-Net 项目整体框架

LiteCue-Net: Lightweight Forgery Cue Unraveling Network for Video Deepfake Detection

```bash
LiteCue-Net/
├── configs/                                # [配置中心] 使用 Hydra/YAML 管理所有参数
│   ├── dataset/
│   │   └── faceforensics.yaml              # 数据集配置：定义 root 路径、压缩率 c23
│   ├── model/
│   │   └── litecue_net.yaml                # [核心] 模型超参：clip_num=16, clip_len=4, dims=256
│   ├── train.yaml                          # 训练参数：LR, BatchSize, Epochs, Optimizer
│   └── inference.yaml                      # 推理参数：定义测试时的采样策略
│
├── data/                                   # [数据仓库] (结构参考 data目录说明.md)
│   ├── raw_videos/                         # 原始视频 (建议软链接)
│   ├── clips/                              # 预处理后的帧序列 (按 video_id/clip_id 存储)
│   ├── meta/                               # [关键] 元数据 JSON，记录每段视频的总帧数、是否伪造
│   ├── dataset_index.pkl                   # 全局索引文件，加速 DataLoader 读取
│   └── splits/                             # 训练/验证/测试集的划分列表 (train.json, val.json)
│
├── deploy/                                 # [落地部署]
│   ├── export_onnx.py                      # 将模型导出为 ONNX (需处理 Conv1D 的输入维度)
│   └── run_inference.py                    # 加载 ONNX 模型对本地视频文件进行检测
│
├── logs/                                   # [实验记录] Tensorboard 日志, Checkpoints 权重文件
│
├── src/                                    # [源代码核心]
│   ├── __init__.py
│   ├── data/                               # [数据管道]
│   │   ├── __init__.py
│   │   ├── dataset.py                      # 数据集类：读取图片，应用增强，返回 (M, K, 3, H, W)
│   │   ├── sampler.py                      # [核心算法] 实现“16段均匀分布+4帧连续”的采样逻辑
│   │   └── transforms.py                   # 基础预处理，包含缩放和归一化
│   │
│   ├── models/                             # [模型架构]
│   │   ├── __init__.py
│   │   ├── detector.py                     # [主类] 组装 Stage 1 -> Stage 2 -> Stage 3 的流水线
│   │   ├── backbones/
│   │   │   └── mobilenet_v4.py             # 空间特征提取 (移除 Classifier Head)
│   │   └── components/                     # [核心组件库] TFCU 三阶段对应实现
│   │       ├── __init__.py
│   │       ├── intra_clip.py               # [Stage 1] 局部微动模块: DW-Conv1D (Lite-CCM)
│   │       ├── inter_clip.py               # [Stage 2] 全局一致性模块: Gated-MLP (Lite-FGM)
│   │       └── reviewer.py                 # [Stage 3] 历史回顾模块: Momentum Accumulation (HRM)
│   │
│   ├── training/                           # [训练器]
│   │   ├── __init__.py
│   │   └── trainer.py                      # 核心逻辑 训练器 (Trainer)
│   │
│   ├── losses/                             # [损失函数]
│   │   └── focal_loss.py                   # 解决正负样本(真/假视频)不平衡问题
│   │
│   └── utils/                              # [工具箱]
│       ├── checkpoint.py                   # 权重保存与加载
│       ├── logger.py                       # 格式化日志打印
│       └── metrics.py                      # 计算 Video-level AUC, Frame-level Accuracy
│
├── notebooks/                              # [新增] 实验监控与可视化中心 
│   ├── 01_Data_Pipeline_Check.ipynb        # 监控：数据采样策略验证 (16x4 逻辑可视化) 
│   ├── 02_Training_Monitor.ipynb           # 监控：调用训练脚本并实时画 Loss 曲线 
│   ├── 03_Feature_Analysis.ipynb           # 分析：可视化 Stage 1 (微动) vs Stage 2 (一致性) 特征 
│   └── 04_Inference_Demo.ipynb             # 演示：单视频推理与归因热力图
│
├── tools/                                  # [工程脚本] 
│   ├── preprocess/
│   │   ├── extract_faces.py                # 视频 -> 人脸检测(RetinaFace/MTCNN) -> 对齐 -> 存图
│   │   ├── build_dataset_index.py          # 扫描 clips 目录，生成 dataset_index.pkl 
│   │   └── verify_data.py                  # 检查采样出的 4 帧是否真的连续
│   └── analysis/
│       ├── count_flops.py                  # [论文] 计算 GFLOPs 和 FPS
│       └── visualize_cues.py               # [可视化] 绘制 Stage 1 和 Stage 2 的特征热力图
│
├── main.py                                 # [训练入口] 解析 Config -> 初始化 Trainer -> 跑训练
└── README.md                               # 项目说明文档
```


## Roadmap

- **基建阶段 (Infrastructure)**
    
    - 安装依赖：`pip install torch timm einops hydra-core opencv-python`
    - 编写 `tools/preprocess/extract_faces.py`：这是最耗时的步骤，先让机器跑起来处理数据，与此同时可以去写模型代码

- **模型核心 (Model Core)**

- **数据管道 (Data Pipeline)**

- **训练循环 (Training Loop)**

- **论文验证 (Evaluation & Demo)**