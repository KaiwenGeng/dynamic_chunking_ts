import torch
import torch.nn as nn
from .TemporalDecompression import TemporalDecompressionBlock
import numpy as np


class FlexibleTemporalDecoder(nn.Module):
    """
    灵活的时序解码器，支持跳跃连接
    
    Args:
        latent_dim: 潜在空间维度
        output_dim: 输出特征维度
        compression_ratios: 每层的压缩比列表（将被反向用于解压缩）
        channel_dims: 每层的通道数列表（将被反向）
        compression_type: 压缩类型 ('conv', 'pool', 'attention')
    
    Example:
        >>> decoder = FlexibleTemporalDecoder(
        ...     latent_dim=128,
        ...     output_dim=7,
        ...     compression_ratios=[2, 2, 2],  # 8x decompression
        ...     channel_dims=[256, 128, 64]
        ... )
        >>> latent = torch.randn(32, 128, 12)  # [B, C, T]
        >>> skip_features = [...]  # from encoder
        >>> output = decoder(latent, skip_features)
        >>> print(output.shape)  # [32, 7, 96]
    """
    def __init__(self, latent_dim, output_dim, compression_ratios=[2, 2, 2, 2], 
                 channel_dims=[512, 256, 128, 64], compression_type='conv'):
        super().__init__()
        # 反向压缩比，用于解压缩
        self.compression_ratios = compression_ratios[::-1]
        self.total_decompression_ratio = int(np.prod(compression_ratios))
        
        # 构建通道维度列表: [latent_dim] + channel_dims + [output_dim]
        self.channel_dims = [latent_dim] + list(channel_dims) + [output_dim]
        
        # 动态构建解压缩层
        self.decompression_layers = nn.ModuleList()
        
        # 构建 len(compression_ratios) + 1 层，与 encoder 对应
        for i in range(len(self.compression_ratios) + 1):
            in_ch = self.channel_dims[i]
            out_ch = self.channel_dims[i + 1]
            
            # 对于前 len(compression_ratios) 层，使用解压缩比
            if i < len(self.compression_ratios):
                ratio = self.compression_ratios[i]
            else:
                # 最后一层不进行时间解压缩，只进行通道变换
                ratio = 1
            
            # 创建解压缩块
            block = TemporalDecompressionBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                decompression_ratio=ratio,
                compression_type=compression_type,
                use_attention=(i % 2 == 0),  # 每隔一层使用注意力
                use_residual=True
            )
            self.decompression_layers.append(block)
        
        print(f"[FlexibleTemporalDecoder] Created decoder with:")
        print(f"  - Decompression ratios: {list(self.compression_ratios)}")
        print(f"  - Total decompression: {self.total_decompression_ratio}x")
        print(f"  - Channel progression: {self.channel_dims}")
    
    def forward(self, x, skip_features):
        """
        前向传播
        
        Args:
            x: [B, C, T] - 来自latent space的输入
            skip_features: List[Tensor] - 来自encoder的跳跃连接特征
            
        Returns:
            output: [B, output_dim, T'] - 解压缩后的输出
        """
        current_x = x
        
        # 反向遍历解压缩层
        for i, layer in enumerate(self.decompression_layers):
            # 获取对应的跳跃连接特征
            # skip_features是按encoder顺序存储的，需要正确对应
            # 前 len(compression_ratios) 层需要跳跃连接，最后一层不需要
            if i < len(self.compression_ratios) and i < len(skip_features):
                # 反向访问：i=0 对应 skip_features[2], i=1 对应 skip_features[1], i=2 对应 skip_features[0]
                skip_idx = len(self.compression_ratios) - 1 - i
                skip_feat = skip_features[skip_idx]
                current_x = layer(current_x, skip_feat)
            else:
                current_x = layer(current_x, None)
        
        return current_x

