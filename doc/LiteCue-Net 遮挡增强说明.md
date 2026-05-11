# LiteCue-Net 遮挡增强说明

## 目的

在训练阶段对输入 clip 施加**空间遮挡**，使模型无法依赖画面中心或局部区域完成分类，从而**强制关注边缘、背景等未遮挡区域**，提升跨数据集与真实场景下的泛化能力。

## 实现位置

- **逻辑实现**：`src/data/augmentation.py` 中的 `apply_clip_occlusion()`
- **调用位置**：`src/data/dataset.py` 的 `LiteCueDataset.__getitem__()`，在得到 `(M, K, C, H, W)` 的 `video_tensor` 之后、`return` 之前
- **生效条件**：仅当 `mode='train'` 且配置中 `occlusion.enabled=true` 时对训练集施加遮挡；验证/测试集不施加

## 行为说明

- **整段 clip 共用同一块遮挡**：同一空间矩形在所有 M×K 帧上被遮住，模型必须从剩余区域（含边缘）找线索。
- **填充方式**：默认使用 ImageNet 归一化均值填充，与现有 `transforms` 一致，避免引入异常值。
- **随机性**：每个样本独立以 `prob` 概率是否遮挡，以及遮挡块的位置与面积，均在 DataLoader 的 worker 内随机。

## 配置项

在 `configs/train.yaml`（或合并后的配置）中增加 `occlusion` 段，例如：

```yaml
occlusion:
  enabled: true       # 是否启用遮挡增强
  prob: 0.5           # 每个样本被遮挡的概率，范围 [0, 1]
  scale_range: [0.1, 0.35]   # 遮挡面积占整图比例范围，如 0.1 表示约 10% 面积
  fill: "mean"        # 填充值："mean" 表示 ImageNet 均值；或写数值如 0
  num_patches: 1      # 每段 clip 内遮挡块数量，建议 1~4
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `enabled` | bool | 为 `true` 时训练集启用遮挡，为 `false` 或不写 `occlusion` 则不启用 |
| `prob` | float | 对当前样本应用遮挡的概率，默认 0.5 |
| `scale_range` | [float, float] | 单块遮挡面积相对整图比例的范围，如 `[0.1, 0.35]` 表示约 10%~35% |
| `fill` | str / float | `"mean"` 使用 ImageNet 均值；或填常数如 `0` |
| `num_patches` | int | 每段 clip 内随机矩形块数，1~4，默认 1 |

## 关闭遮挡

- **方式一**：配置中设 `occlusion.enabled: false`
- **方式二**：删除或注释掉配置中的 `occlusion` 段；`main.py` 中会传 `occlusion_cfg=None`，Dataset 不会施加遮挡

## 调参建议

- **泛化优先**：可适当增大 `scale_range` 上限（如 0.35~0.5）或提高 `prob`，让模型更依赖边缘与多区域。
- **稳定训练**：若训练不稳定，可先减小 `prob` 或 `scale_range`，再逐步加强。
- **多块遮挡**：`num_patches=2` 或 3 可进一步迫使模型综合多处局部与边缘信息。

## 与其它增强的关系

- 遮挡在 **transforms 之后** 施加（即对已 resize、flip、ToTensor、Normalize 的 tensor 操作），与现有 `get_transforms()` 兼容。
- 验证/测试阶段不启用遮挡，评估结果与“无遮挡输入”一致。
