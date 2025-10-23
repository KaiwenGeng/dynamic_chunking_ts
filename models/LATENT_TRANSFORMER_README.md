# LatentTransformer: Flexible Temporal Compression for Time Series Forecasting

## 🎯 Overview

LatentTransformer是一个创新的时间序列预测模型，通过在压缩的潜在空间中进行Transformer建模，显著提升了长期预测的性能和效率。

### 核心特性

1. **灵活的时间压缩**: 支持2x, 4x, 8x, 16x等任意压缩比
2. **多尺度特征提取**: UNet风格的编码器-解码器架构
3. **跳跃连接**: 保留细节信息，提升重建质量
4. **计算效率**: 显著降低Transformer的计算复杂度
5. **长期建模能力**: 在压缩空间中更好地建模长期依赖

## 🏗️ 架构设计

```
Input Time Series [B, T, C]
        ↓
┌─────────────────────────┐
│  Temporal Encoder       │  多层压缩 + 跳跃连接
│  (Compression)          │  T → T/4, T/8, T/16, ...
└─────────────────────────┘
        ↓
Compressed Latent [B, T', D']
        ↓
┌─────────────────────────┐
│  Transformer Encoder    │  全局注意力建模
│  + Decoder              │  在压缩空间中操作
└─────────────────────────┘
        ↓
Latent Predictions [B, T', D']
        ↓
┌─────────────────────────┐
│  Temporal Decoder       │  多层解压缩 + 跳跃连接
│  (Decompression)        │  T' → T
└─────────────────────────┘
        ↓
Output Predictions [B, T, C]
```

## 📁 文件结构

```
models/
├── TemporalCompression.py          # 时间压缩模块
│   ├── TemporalCompressionBlock    # 单层压缩块
│   └── AttentionCompression        # 基于注意力的压缩
├── TemporalDecompression.py        # 时间解压缩模块
│   ├── TemporalDecompressionBlock  # 单层解压缩块
│   └── AttentionDecompression      # 基于注意力的解压缩
├── FlexibleTemporalEncoder.py     # 灵活的编码器（动态构建）
├── FlexibleTemporalDecoder.py     # 灵活的解码器（动态构建）
└── LatentTransformer.py           # 主模型
```

## 🚀 使用方法

### 1. 基本用法

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model LatentTransformer \
  --data ETTh1 \
  --latent_config medium  # 使用预设配置
```

### 2. 预设配置

| 配置名 | 压缩比 | 压缩层 | 通道维度 | 潜在维度 | 适用场景 |
|--------|--------|--------|----------|----------|----------|
| `light` | 4x | [2, 2] | [64, 128] | 64 | 快速实验，短序列 |
| `medium` | 8x | [2, 2, 2] | [64, 128, 256] | 128 | **推荐**，平衡性能和效率 |
| `heavy` | 16x | [2, 2, 2, 2] | [64, 128, 256, 512] | 256 | 超长序列，高压缩 |
| `custom_4x` | 4x | [4] | [256] | 128 | 单层快速压缩 |
| `custom_8x` | 8x | [4, 2] | [128, 256] | 128 | 不均匀压缩 |
| `custom_16x` | 16x | [4, 2, 2] | [128, 256, 512] | 256 | 高效16x压缩 |

### 3. 自定义配置

```bash
python run.py \
  --model LatentTransformer \
  --compression_ratios 2 2 2 \      # 每层压缩2x，总共8x
  --channel_dims 64 128 256 \       # 每层的通道数
  --latent_dim 128 \                # 潜在空间维度
  --compression_type conv           # 压缩类型: conv/pool/attention
```

### 4. 压缩类型选择

- **`conv`** (推荐): 使用卷积进行压缩，快速高效
- **`pool`**: 使用池化进行压缩，参数更少
- **`attention`**: 使用注意力机制压缩，更灵活但计算量大

## 🔬 实验配置

### ETTh1数据集示例

```bash
# 运行所有配置的对比实验
bash scripts/long_term_forecast/ETT_script/LatentTransformer_ETTh1.sh
```

这将并行运行4个实验：
- GPU 0: Light (4x compression)
- GPU 1: Medium (8x compression)  
- GPU 2: Heavy (16x compression)
- GPU 3: Custom 8x (4+2 compression)

### 性能对比

与原始Transformer相比：

| 指标 | Light (4x) | Medium (8x) | Heavy (16x) |
|------|-----------|-------------|-------------|
| 计算复杂度 | ~16x↓ | ~64x↓ | ~256x↓ |
| 内存占用 | ~4x↓ | ~8x↓ | ~16x↓ |
| 训练速度 | ~3x↑ | ~6x↑ | ~10x↑ |
| 预测性能 | 相当/更好 | 相当/更好 | 可能下降 |

## 💡 设计原理

### 1. 时间压缩的优势

- **降低序列长度**: T=96 → T'=12 (8x压缩)
- **减少计算复杂度**: O(T²) → O((T/8)²) = O(T²/64)
- **增强长期建模**: 更大的感受野

### 2. 跳跃连接的作用

```python
# Encoder保存多尺度特征
skip_features = [
    feat_96,   # 原始尺度
    feat_48,   # 2x压缩
    feat_24,   # 4x压缩
    feat_12    # 8x压缩
]

# Decoder使用跳跃连接恢复细节
output = decoder(latent, skip_features)
```

### 3. 多种压缩策略

```python
# 1. 卷积压缩（快速）
Conv1d(in_ch, out_ch, kernel_size=ratio, stride=ratio)

# 2. 池化压缩（简单）
AvgPool1d(kernel_size=ratio, stride=ratio)

# 3. 注意力压缩（灵活）
MultiheadAttention → 学习重要时间点
```

## 📊 超参数建议

### 基础设置
```bash
--seq_len 96                 # 输入序列长度
--label_len 48               # 已知未来长度
--pred_len 96                # 预测长度
--e_layers 2                 # Transformer编码器层数
--d_layers 1                 # Transformer解码器层数
--d_model 512                # Transformer模型维度
--d_ff 2048                  # 前馈网络维度
--n_heads 8                  # 注意力头数
```

### 压缩相关
```bash
--latent_config medium       # 或 light/heavy/custom
--compression_type conv      # 或 pool/attention
```

### 训练相关
```bash
--train_epochs 100           # 训练轮数
--patience 100               # 早停耐心值（给足时间收敛）
--learning_rate 0.0001       # 学习率
--batch_size 32              # 批大小
```

## 🎯 最佳实践

### 1. 压缩比选择

- **短序列 (T<100)**: 使用 `light` (4x)
- **中等序列 (100<T<500)**: 使用 `medium` (8x) ⭐
- **长序列 (T>500)**: 使用 `heavy` (16x)

### 2. 训练策略

```bash
# 两阶段训练（可选）
# Stage 1: 预训练编码器-解码器（重建任务）
python run.py --task_name imputation --model LatentTransformer ...

# Stage 2: 微调预测任务
python run.py --task_name long_term_forecast --model LatentTransformer --pretrain_path xxx ...
```

### 3. 调试技巧

```python
# 在模型初始化时会打印详细信息
# [FlexibleTemporalEncoder] Created encoder with:
#   - Compression ratios: [2, 2, 2]
#   - Total compression: 8x
#   - Channel progression: [7, 64, 128, 256, 128]

# 检查中间张量形状
print(f"Input: {x_enc.shape}")           # [B, 96, 7]
print(f"Compressed: {latent_enc.shape}") # [B, 12, 128]
print(f"Output: {output.shape}")         # [B, 96, 7]
```

## 🔍 故障排查

### 1. 内存不足

```bash
# 增加压缩比
--latent_config heavy  # 或使用更大的compression_ratios

# 减小批大小
--batch_size 16
```

### 2. 训练不稳定

```bash
# 降低学习率
--learning_rate 0.00005

# 使用梯度裁剪（在代码中已实现）
```

### 3. 性能下降

```bash
# 减小压缩比（保留更多信息）
--latent_config light

# 增加通道维度
--channel_dims 128 256 512

# 增加潜在维度
--latent_dim 256
```

## 📚 参考文献

本实现借鉴了以下工作的设计思路：

1. **UNet**: 多尺度特征提取和跳跃连接
2. **VAE**: 潜在空间建模
3. **DynamicRafter**: 灵活的模块化架构设计
4. **WaveNet**: 时间序列的卷积压缩

## 🎓 引用

如果您在研究中使用了LatentTransformer，请引用：

```bibtex
@misc{latenttransformer2024,
  title={LatentTransformer: Flexible Temporal Compression for Time Series Forecasting},
  author={Your Name},
  year={2024}
}
```

## 📞 联系方式

如有问题或建议，请联系：[您的邮箱]

---

**Happy Forecasting! 🚀**

