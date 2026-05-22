# 推理模型资源（随仓库同步）

本目录包含 FastAPI 推理服务所需的 **InsightFace `buffalo_l`** ONNX 权重，克隆仓库后即可离线使用，无需再运行下载脚本或配置 `~/.insightface`。

## 目录结构

```
models/
└── buffalo_l/
    ├── det_10g.onnx      # 人脸检测（RetinaFace）
    ├── 2d106det.onnx     # 106 点关键点（对齐用）
    ├── 1k3d68.onnx       # 3D 关键点（buffalo_l 包内文件）
    ├── w600k_r50.onnx    # 人脸识别（buffalo_l 包内文件）
    └── genderage.onnx    # 性别年龄（buffalo_l 包内文件）
```

LiteCue-Net 检测权重位于：`checkpoints/exp_20260511/best_model.pth`（已纳入版本库）。

## Git LFS

`1k3d68.onnx`、`w600k_r50.onnx` 超过 GitHub 单文件 100MB 限制，通过 **Git LFS** 跟踪。克隆后若 ONNX 只有几 KB 的指针文件，请执行：

```bash
git lfs install
git lfs pull
```

## 环境变量（一般无需修改）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `INSIGHTFACE_ROOT` | 项目根目录 | InsightFace 会在 `{root}/models/buffalo_l` 下加载模型 |
| `CHECKPOINT_PATH` | `checkpoints/exp_20260511/best_model.pth` | LiteCue-Net 权重路径 |

## 备用下载

仅在 LFS 未拉取或文件损坏时，可运行：

```bash
python scripts/download_models.py
```
