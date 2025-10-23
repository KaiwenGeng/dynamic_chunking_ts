# LatentTransformer重建模式完整指南

## 📋 概览

现在我们的**LatentTransformer**支持**三种重建模式**，每种模式都有不同的目标和优势：

### 当前实现的Decoder类型

| 模式 | 类型 | 重建损失 | KL散度 | Masking | 用途 |
|------|------|----------|--------|---------|------|
| **原始LatentTransformer** | 无重建 | ❌ | ❌ | ❌ | 纯预测任务 |
| **AE** | Auto-Encoder | ✅ | ❌ | ❌ | 学习更好的表示 |
| **VAE** | Variational AE | ✅ | ✅ | ❌ | 概率建模+泛化 |
| **MAE** | Masked AE | ✅ | ❌ | ✅ | 鲁棒表示学习 |

## 🏗️ 架构对比

### 1. 原始LatentTransformer (无重建)
```
Input → Encoder → Latent → Transformer → Latent' → Decoder → Prediction
                                                              ↓
                                                         只有预测损失
```

### 2. AE模式 (Auto-Encoder)
```
Input → Encoder → Latent → Transformer → Latent' → Decoder → Prediction
   ↓              ↓                                              ↓
   └──────→ Reconstruction Decoder ────────────────→ Reconstructed Input
                                                              ↓
                                               预测损失 + 重建损失
```

### 3. VAE模式 (Variational Auto-Encoder)
```
Input → Encoder → [μ, σ²] → Reparameterization → Latent → Transformer → Prediction
   ↓                ↓                                ↓
   │         KL Divergence Loss                     │
   │                                                 │
   └──────────→ Reconstruction Decoder ←────────────┘
                        ↓
           预测损失 + 重建损失 + KL散度损失
```

### 4. MAE模式 (Masked Auto-Encoder)
```
Input → Random Masking → Masked Input → Encoder → Latent → Transformer → Prediction
   ↓                                       ↓
   └────────────→ Reconstruction Decoder ←┘
                        ↓
                Only Masked部分重建
                        ↓
           预测损失 + Masked重建损失
```

## 🎯 损失函数详解

### 1. 原始LatentTransformer
```python
total_loss = prediction_loss
```

### 2. AE模式
```python
total_loss = prediction_loss + λ_recon * reconstruction_loss

where:
- reconstruction_loss = MSE(reconstructed_input, original_input)
- λ_recon = reconstruction_loss_weight (default: 0.5)
```

### 3. VAE模式
```python
total_loss = prediction_loss + λ_recon * reconstruction_loss + λ_kl * kl_loss

where:
- reconstruction_loss = MSE(reconstructed_input, original_input)
- kl_loss = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
- λ_recon = reconstruction_loss_weight (default: 0.5)
- λ_kl = kl_loss_weight (default: 0.01)
```

### 4. MAE模式
```python
total_loss = prediction_loss + λ_recon * masked_reconstruction_loss

where:
- masked_reconstruction_loss = MSE(reconstructed[masked], original[masked])
- 只计算被mask部分的重建损失
- λ_recon = reconstruction_loss_weight (default: 0.5)
```

## 💻 使用方法

### 1. 无重建模式（原始LatentTransformer）
```bash
python run.py \
  --model LatentTransformer \
  --task_name long_term_forecast \
  --data ETTh1 \
  --latent_config medium \
  # 其他参数...
```

### 2. AE模式
```bash
python run.py \
  --model LatentTransformerWithReconstruction \
  --reconstruction_mode AE \
  --reconstruction_loss_weight 0.5 \
  --task_name long_term_forecast \
  --data ETTh1 \
  --latent_config medium \
  # 其他参数...
```

### 3. VAE模式
```bash
python run.py \
  --model LatentTransformerWithReconstruction \
  --reconstruction_mode VAE \
  --reconstruction_loss_weight 0.5 \
  --kl_loss_weight 0.01 \
  --task_name long_term_forecast \
  --data ETTh1 \
  --latent_config medium \
  # 其他参数...
```

### 4. MAE模式
```bash
python run.py \
  --model LatentTransformerWithReconstruction \
  --reconstruction_mode MAE \
  --reconstruction_loss_weight 0.5 \
  --mask_ratio 0.25 \
  --task_name long_term_forecast \
  --data ETTh1 \
  --latent_config medium \
  # 其他参数...
```

### 5. 运行对比实验
```bash
bash scripts/long_term_forecast/ETT_script/LatentTransformer_Reconstruction_Comparison.sh
```

## 📊 超参数建议

### 重建损失权重 (`reconstruction_loss_weight`)
- **推荐范围**: 0.1 ~ 1.0
- **默认值**: 0.5
- **说明**: 
  - 太小：重建损失作用不明显
  - 太大：可能影响预测性能
  - 建议从0.5开始调整

### KL散度权重 (`kl_loss_weight`, VAE专用)
- **推荐范围**: 0.001 ~ 0.1
- **默认值**: 0.01
- **说明**:
  - 太小：潜在空间不够正则化
  - 太大：重建质量下降（posterior collapse）
  - 通常设置为reconstruction_loss_weight的1/10~1/100

### Masking比例 (`mask_ratio`, MAE专用)
- **推荐范围**: 0.15 ~ 0.75
- **默认值**: 0.25
- **说明**:
  - 0.25: 适合时间序列（类比BERT的15%）
  - 0.50: 更强的自监督信号
  - 0.75: MAE论文中图像的设置（可能对时间序列太aggressive）

## 🔍 各模式的优缺点

### 原始LatentTransformer (无重建)
✅ **优点:**
- 最简单，参数最少
- 训练速度最快
- 专注于预测任务

⚠️ **缺点:**
- 可能学不到最优的潜在表示
- 编码器没有显式的表示学习目标

### AE模式
✅ **优点:**
- 显式的表示学习目标
- 重建损失帮助学习更好的特征
- 确定性编码，容易理解

⚠️ **缺点:**
- 可能过拟合到训练数据
- 潜在空间可能不够smooth

### VAE模式
✅ **优点:**
- 概率建模，提供不确定性估计
- KL散度正则化潜在空间
- 更好的泛化能力
- 潜在空间更连续smooth

⚠️ **缺点:**
- 需要调整KL权重（balance problem）
- 可能出现posterior collapse
- 训练稍微复杂

### MAE模式
✅ **优点:**
- 类似BERT的预训练范式
- 学习更鲁棒的表示
- 对噪声和缺失值更鲁棒
- 可以作为预训练方法

⚠️ **缺点:**
- 需要调整masking比例
- 训练时间稍长（masking操作）
- 对时间序列的最优masking策略需要探索

## 📈 实验建议

### 1. 基础对比实验
```bash
# 运行4个模式的对比
bash scripts/long_term_forecast/ETT_script/LatentTransformer_Reconstruction_Comparison.sh
```

这将运行：
- GPU 0: 无重建（baseline）
- GPU 1: AE模式
- GPU 2: VAE模式
- GPU 3: MAE模式

### 2. 超参数搜索

#### AE超参数搜索
```bash
for weight in 0.1 0.3 0.5 0.7 1.0; do
    python run.py \
      --model LatentTransformerWithReconstruction \
      --reconstruction_mode AE \
      --reconstruction_loss_weight $weight \
      --model_id ETTh1_AE_w${weight} \
      # 其他参数...
done
```

#### VAE超参数搜索
```bash
for kl_weight in 0.001 0.01 0.1; do
    python run.py \
      --model LatentTransformerWithReconstruction \
      --reconstruction_mode VAE \
      --reconstruction_loss_weight 0.5 \
      --kl_loss_weight $kl_weight \
      --model_id ETTh1_VAE_kl${kl_weight} \
      # 其他参数...
done
```

#### MAE超参数搜索
```bash
for mask_ratio in 0.15 0.25 0.5 0.75; do
    python run.py \
      --model LatentTransformerWithReconstruction \
      --reconstruction_mode MAE \
      --mask_ratio $mask_ratio \
      --model_id ETTh1_MAE_mask${mask_ratio} \
      # 其他参数...
done
```

### 3. 评估指标

需要关注的指标：
1. **预测性能**: MSE, MAE on test set
2. **重建质量**: Reconstruction MSE (训练时)
3. **KL散度**: KL loss value (VAE)
4. **训练效率**: Training time per epoch
5. **收敛速度**: Epochs to converge

## 🔧 与exp_long_term_forecasting.py的集成

### 训练Loss计算

当前的`exp_long_term_forecasting.py`已经支持reconstruction loss：

```python
# 在train()方法中
if self.args.reconstruction_mode != 'None':
    outputs, reconstructed_input, reconstruction_loss, kl_loss = self.model(...)
    
    # 计算总损失
    loss = pred_loss + \
           self.args.reconstruction_loss_weight * reconstruction_loss + \
           self.args.kl_loss_weight * kl_loss
else:
    outputs = self.model(...)
    loss = pred_loss
```

### 验证和测试

验证和测试时**不使用**重建损失，只评估预测性能：

```python
# 在vali()方法中
if self.args.reconstruction_mode != 'None':
    outputs, _, _, _ = self.model(...)  # 忽略重建输出
else:
    outputs = self.model(...)

# 只计算预测损失
loss = criterion(pred_outputs, pred_targets)
```

## 💡 最佳实践

### 1. 训练策略

#### 方法1: 端到端训练
```bash
python run.py \
  --model LatentTransformerWithReconstruction \
  --reconstruction_mode VAE \
  --train_epochs 100
```

#### 方法2: 两阶段训练（推荐）
```bash
# Stage 1: 预训练encoder-decoder (重建任务)
python run.py \
  --model LatentTransformerWithReconstruction \
  --reconstruction_mode MAE \
  --reconstruction_loss_weight 1.0 \  # 更高的重建权重
  --train_epochs 50 \
  --model_id ETTh1_pretrain

# Stage 2: 微调预测任务
python run.py \
  --model LatentTransformerWithReconstruction \
  --reconstruction_mode MAE \
  --reconstruction_loss_weight 0.1 \  # 降低重建权重
  --pretrain_path checkpoints/ETTh1_pretrain/checkpoint.pth \
  --train_epochs 50 \
  --model_id ETTh1_finetune
```

### 2. 模式选择指南

| 场景 | 推荐模式 | 原因 |
|------|----------|------|
| 数据充足，追求最优性能 | AE | 简单有效 |
| 数据有限，需要泛化 | VAE | 正则化效果好 |
| 数据有噪声/缺失 | MAE | 鲁棒性强 |
| 需要不确定性估计 | VAE | 概率建模 |
| 追求速度 | 无重建 | 最快 |
| 预训练+微调范式 | MAE | 类似BERT |

### 3. 调试技巧

#### 检查重建质量
```python
# 在训练脚本中添加
if reconstruction_mode != 'None':
    print(f"Reconstruction Loss: {reconstruction_loss.item():.4f}")
    print(f"Prediction Loss: {pred_loss.item():.4f}")
    
    # 可视化重建
    import matplotlib.pyplot as plt
    plt.plot(original_input[0, :, 0].cpu(), label='Original')
    plt.plot(reconstructed_input[0, :, 0].cpu(), label='Reconstructed')
    plt.legend()
    plt.savefig('reconstruction.png')
```

#### 检查VAE的KL散度
```python
# 监控KL loss，避免posterior collapse
if reconstruction_mode == 'VAE':
    print(f"KL Loss: {kl_loss.item():.6f}")
    # 健康的KL loss应该在0.001~0.1之间
    # 如果接近0，可能是posterior collapse
```

#### 检查MAE的masking效果
```python
# 验证masking是否正确
if reconstruction_mode == 'MAE':
    print(f"Mask ratio: {mask.mean().item():.2f}")
    # 应该接近设定的mask_ratio
```

## 🎓 理论背景

### AE (Auto-Encoder)
- **目标**: 学习数据的低维表示
- **损失**: L = L_pred + λ * L_recon
- **应用**: 特征学习，降维

### VAE (Variational Auto-Encoder)
- **目标**: 学习数据的概率分布
- **损失**: L = L_pred + λ_recon * L_recon + λ_kl * KL(q(z|x) || p(z))
- **应用**: 生成建模，不确定性估计

### MAE (Masked Auto-Encoder)
- **目标**: 从部分观测重建完整数据
- **损失**: L = L_pred + λ * L_masked_recon
- **应用**: 自监督预训练，鲁棒表示学习

## 📚 参考文献

1. **Auto-Encoder**: Hinton & Salakhutdinov, "Reducing the Dimensionality of Data with Neural Networks", Science 2006
2. **VAE**: Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014
3. **MAE**: He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
4. **BERT**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", NAACL 2019

## ✅ 总结

现在你的LatentTransformer支持：

| 特性 | 状态 |
|------|------|
| ✅ 无重建模式 | 已实现 |
| ✅ AE模式 | 已实现 |
| ✅ VAE模式 | 已实现（带KL散度） |
| ✅ MAE模式 | 已实现（带masking） |
| ✅ 训练loss支持 | 已集成到exp_long_term_forecasting.py |
| ✅ 灵活的超参数 | reconstruction_loss_weight, kl_loss_weight, mask_ratio |
| ✅ 对比实验脚本 | LatentTransformer_Reconstruction_Comparison.sh |

祝实验顺利！🚀

