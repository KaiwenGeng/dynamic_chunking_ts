import torch
import torch.nn as nn
import torch.nn.functional as F
from .TemporalCompression import TemporalCompressionBlock
import numpy as np


class FlexibleTemporalEncoder(nn.Module):
    """
    灵活的时序编码器，支持动态构建压缩层
    
    Args:
        input_dim: 输入特征维度
        latent_dim: 潜在空间维度
        compression_ratios: 每层的压缩比列表 (e.g., [2, 2, 2] for 8x compression)
        channel_dims: 每层的通道数列表 (e.g., [64, 128, 256])
        compression_type: 压缩类型 ('conv', 'pool', 'attention')
    
    Example:
        >>> encoder = FlexibleTemporalEncoder(
        ...     input_dim=7, 
        ...     latent_dim=128,
        ...     compression_ratios=[2, 2, 2],  # 8x compression
        ...     channel_dims=[64, 128, 256]
        ... )
        >>> x = torch.randn(32, 7, 96)  # [B, C, T]
        >>> latent, skip_features = encoder(x)
        >>> print(latent.shape)  # [32, 128, 12]
    """
    def __init__(self, input_dim, latent_dim, compression_ratios=[2, 2, 2, 2], 
                 channel_dims=[64, 128, 256, 512], compression_type='conv'):
        super().__init__()
        self.compression_ratios = compression_ratios
        self.total_compression_ratio = int(np.prod(compression_ratios))
        
        # 构建通道维度列表: [input_dim] + channel_dims + [latent_dim]
        self.channel_dims = [input_dim] + list(channel_dims) + [latent_dim]
        
        # 记录跳跃连接信息
        self.skip_connections = []
        
        # 动态构建压缩层
        self.compression_layers = nn.ModuleList()
        current_ratio = 1
        
        # 构建 len(compression_ratios) + 1 层，确保最后一层输出 latent_dim
        for i in range(len(compression_ratios) + 1):
            in_ch = self.channel_dims[i]
            out_ch = self.channel_dims[i + 1]
            
            # 对于前 len(compression_ratios) 层，使用压缩比
            if i < len(compression_ratios):
                ratio = compression_ratios[i]
                current_ratio *= ratio
            else:
                # 最后一层不进行时间压缩，只进行通道变换
                ratio = 1
            
            # 创建压缩块
            block = TemporalCompressionBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                compression_ratio=ratio,
                compression_type=compression_type,
                use_attention=(i % 2 == 0),  # 每隔一层使用注意力
                use_residual=True
            )
            self.compression_layers.append(block)
            
            # 记录跳跃连接信息
            self.skip_connections.append({
                'layer_idx': i,
                'ratio': current_ratio,
                'channels': out_ch
            })
        
        print(f"[FlexibleTemporalEncoder] Created encoder with:")
        print(f"  - Compression ratios: {compression_ratios}")
        print(f"  - Total compression: {self.total_compression_ratio}x")
        print(f"  - Channel progression: {self.channel_dims}")
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, C, T] - 输入时间序列
            
        Returns:
            latent: [B, latent_dim, T'] - 压缩后的潜在表示
            skip_features: List[Tensor] - 跳跃连接特征列表
        """
        skip_features = []
        current_x = x
        
        for i, layer in enumerate(self.compression_layers):
            current_x = layer(current_x)
            # 保存跳跃连接特征
            skip_features.append(current_x)
        
        return current_x, skip_features

