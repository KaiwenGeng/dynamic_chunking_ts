import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalDecompressionBlock(nn.Module):
    """
    时间解压缩块
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        decompression_ratio: 时间解压缩比 (e.g., 2, 4, 8)
        compression_type: 压缩类型 ('conv', 'pool', 'attention')
        use_attention: 是否使用注意力机制
        use_residual: 是否使用跳跃连接
    """
    def __init__(self, in_channels, out_channels, decompression_ratio, 
                 compression_type='conv', use_attention=True, use_residual=True):
        super().__init__()
        self.decompression_ratio = decompression_ratio
        self.use_residual = use_residual
        
        # 解压缩层选择
        if compression_type == 'conv':
            # 使用转置卷积进行解压缩
            self.decompressor = nn.ConvTranspose1d(
                in_channels, out_channels, 
                kernel_size=decompression_ratio, 
                stride=decompression_ratio,
                padding=0
            )
        elif compression_type == 'pool':
            # 使用上采样进行解压缩
            self.decompressor = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.Upsample(scale_factor=decompression_ratio, mode='linear', align_corners=False)
            )
        elif compression_type == 'attention':
            # 使用注意力机制进行解压缩
            self.decompressor = AttentionDecompression(in_channels, out_channels, decompression_ratio)
        else:
            raise ValueError(f"Unknown compression_type: {compression_type}")
        
        # 跳跃连接融合
        if use_residual:
            self.skip_fusion = nn.Conv1d(out_channels * 2, out_channels, kernel_size=1)
        else:
            self.skip_fusion = None
        
        # 可选注意力机制
        if use_attention:
            self.attention = nn.MultiheadAttention(out_channels, num_heads=8, batch_first=False)
        else:
            self.attention = None
        
        # 归一化和激活
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
    
    def forward(self, x, skip_feature=None):
        """
        前向传播
        
        Args:
            x: [B, C, T] - 输入张量
            skip_feature: [B, C', T'] - 跳跃连接特征（可选）
            
        Returns:
            decompressed: [B, C'', T''] - 解压缩后的张量
        """
        # 解压缩
        decompressed = self.decompressor(x)
        
        # 跳跃连接融合
        if skip_feature is not None and self.skip_fusion is not None:
            # 调整skip_feature的尺寸以匹配decompressed
            if skip_feature.shape[-1] != decompressed.shape[-1]:
                skip_feature = F.interpolate(
                    skip_feature, 
                    size=decompressed.shape[-1], 
                    mode='linear', 
                    align_corners=False
                )
            # 拼接并融合
            fused = torch.cat([decompressed, skip_feature], dim=1)
            decompressed = self.skip_fusion(fused)
        
        # 注意力机制
        if self.attention is not None:
            # 转换维度用于注意力: [B, C, T] -> [T, B, C]
            B, C, T = decompressed.shape
            decompressed_attn = decompressed.permute(2, 0, 1)  # [T, B, C]
            attn_out, _ = self.attention(decompressed_attn, decompressed_attn, decompressed_attn)
            decompressed = attn_out.permute(1, 2, 0)  # [B, C, T]
        
        # 归一化和激活
        decompressed = decompressed.transpose(1, 2)  # [B, T, C]
        decompressed = self.norm(decompressed)
        decompressed = self.activation(decompressed)
        decompressed = decompressed.transpose(1, 2)  # [B, C, T]
        
        return decompressed


class AttentionDecompression(nn.Module):
    """
    基于注意力的解压缩层
    将压缩的时间序列通过注意力机制还原到更长的表示
    """
    def __init__(self, in_channels, out_channels, decompression_ratio):
        super().__init__()
        self.decompression_ratio = decompression_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 投影层
        self.projection = nn.Linear(in_channels, out_channels)
        
        # 注意力层
        self.attention = nn.MultiheadAttention(out_channels, num_heads=8, batch_first=True)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, decompression_ratio, out_channels))
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, C, T] - 输入张量
            
        Returns:
            decompressed: [B, C', T'] - 解压缩后的张量
        """
        B, C, T = x.shape
        
        # 计算解压缩后的时间步数
        decompressed_T = T * self.decompression_ratio
        
        # 转换维度: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        
        # 投影到输出维度
        x_proj = self.projection(x)  # [B, T, out_channels]
        
        # 扩展每个时间步到decompression_ratio个时间步
        x_expanded = x_proj.unsqueeze(2).repeat(1, 1, self.decompression_ratio, 1)
        x_expanded = x_expanded.reshape(B, decompressed_T, self.out_channels)
        
        # 添加位置编码
        pos_enc = self.pos_encoding.repeat(B, T, 1)
        x_expanded = x_expanded + pos_enc
        
        # 注意力解压缩
        decompressed, _ = self.attention(x_expanded, x_expanded, x_expanded)
        
        # 转换回 [B, out_channels, decompressed_T]
        decompressed = decompressed.transpose(1, 2)
        
        return decompressed

