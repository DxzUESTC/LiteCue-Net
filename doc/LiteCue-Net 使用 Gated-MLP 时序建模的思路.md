## 核心设计

LiteCue-Net 目标是在极低的计算资源下，解决视频深伪检测中时序不一致（Temporal Inconsistency）的捕捉难题

现有方法的不足：
- 在单帧（Image-based）检测的场景下，检测速度快但是忽略了深伪视频中深伪痕迹在时序上的累积。仅靠单帧检测已经难以解决日益成熟的伪造技术
- 在视频的时序检测场景中，多使用重型的时序网络（3D-CNN/Transformer），检测的精度高，但是难以在端侧进行部署

MoGa-DF 采用解耦设计：
- 空间特征（Spatial）：使用经过预训练的 MobileNetV4，负责提取深伪视频单帧中的高频纹理和伪造痕迹
- 时序特征（Temporal）：实现与MobileNetV4 联合的 Gated-MLP，利用门控机制（Gating Mechanism）在时间维度上累积和放大伪造特征，避开 Attention 的高计算量和传统 RNN/GRU 的顺序计算导致推理慢的缺点

## 整体流水线设计

### 阶段一：高效空间特征提取（Efficient Spatial Extraction）

输入：一段视频被采样为 T 帧的图像序列 $X\in\mathbb{R}^T\times3\times H\times W$

操作：
- 利用 MobileNetV4 - Small 作为BackBone
- 一组经过采样的固定帧数 $T$ 的单帧被视为 $T$ 个独立的 Batch 进行并行推理
> 这里的并行是不是必要的呢，如果处理实时的视频流应该怎么先处理这些单帧呢，先缓存到足够的帧数吗？
- 移除原始的分类头，提取 Global Average Pooling 之前的特征图或 Pooling 后的特征向量 $F\in\mathbb{R}^{T\times C_{mnv4}}$

目的：以最小 FLOPs 获取每一帧的丰富语义和纹理信息

### 阶段二：特征投影与位置感知（Projection & Positional Encoding）

- 维度瓶颈（Bottleneck）：MNv4 输出的通道数（如1280）对于时序建模来说过于冗余。设计一个轻量级的Liner 层将维度降至 $D$ （如256）：$F'=\mathrm{Linear}(F),\quad F'\in\mathbb{R}^{T\times D}$
- 位置编码：MLP 本身对顺序不敏感，而深伪视频中的伪影或者不连续的痕迹与时间顺序强相关，所以需要注入可学习的位置编码$P$：$Z_0=F^{\prime}+P,\quad P\in\mathbb{R}^{T\times D}$

### 阶段三：时序门控建模（Temporal Gated Modeling）

操作：输入特征序列 $Z$ 进入堆叠的 Gated-MLP Blocks

机制：
- 不同于 Transformer 在 T 维度做 O(T2) 的 Self-Attention
- Gated-MLP 通过 Spatial Gating Unit (SGU) 在 T 维度进行全局线性混合 (Global Linear Mixing)
- 它学习一个“门控权重”，自动识别哪些帧包含了关键的伪造信息，并让这些信息“流”向后续层，同时抑制无效帧（如模糊或无脸帧）的噪声

公式直觉：$Output=Input\times Gate(Input)$。这里的 Gate 是跨越整个时间轴 $T$ 计算出来的

目的：捕捉视频中累积的（Accumulative）深伪痕迹，且推理速度极快


### 阶段四：聚合与判别 (Aggregation & Decision)

操作：对 Gated-MLP 输出的序列特征进行 Temporal Average Pooling，得到整个视频的特征表示

输出：通过一个简单的全连接层 (FC) + Softmax 输出真/假概率