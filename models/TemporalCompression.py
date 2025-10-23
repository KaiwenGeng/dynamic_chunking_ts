import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalCompressionBlock(nn.Module):
    """
    单个时间压缩块，支持多种压缩策略
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        compression_ratio: 时间压缩比 (e.g., 2, 4, 8)
        compression_type: 压缩类型 ('conv', 'pool', 'attention')
        use_attention: 是否使用注意力机制
        use_residual: 是否使用残差连接
    """
    def __init__(self, in_channels, out_channels, compression_ratio, 
                 compression_type='conv', use_attention=True, use_residual=True):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.use_residual = use_residual
        
        # 压缩层选择
        if compression_ratio == 1:
            # No compression: just project channels
            self.compressor = nn.Conv1d(
                in_channels, out_channels, 
                kernel_size=1, 
                stride=1,
                padding=0
            )
        elif compression_type == 'conv':
            # 使用卷积进行压缩
            self.compressor = nn.Conv1d(
                in_channels, out_channels, 
                kernel_size=compression_ratio, 
                stride=compression_ratio,
                padding=0
            )
        elif compression_type == 'causal_conv':
            # 使用因果卷积进行压缩，保持时间因果性
            self.compressor = nn.Conv1d(
                in_channels, out_channels, 
                kernel_size=compression_ratio, 
                stride=compression_ratio,
                padding=0,  # 无padding确保因果性
                dilation=1
            )
        elif compression_type == 'pool':
            # 使用池化进行压缩
            self.compressor = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.AvgPool1d(kernel_size=compression_ratio, stride=compression_ratio)
            )
        elif compression_type == 'causal_pool':
            # 使用因果池化进行压缩，保持时间因果性
            self.compressor = CausalPoolingCompression(
                in_channels, out_channels, compression_ratio
            )
        elif compression_type == 'attention':
            # 使用注意力机制进行压缩
            self.compressor = AttentionCompression(in_channels, out_channels, compression_ratio)
        else:
            raise ValueError(f"Unknown compression_type: {compression_type}")
        
        # 可选注意力机制
        if use_attention:
            self.attention = nn.MultiheadAttention(out_channels, num_heads=8, batch_first=False)
        else:
            self.attention = None
        
        # 归一化和激活
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        
        # 残差连接
        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, C, T] - 输入张量
            
        Returns:
            compressed: [B, C', T'] - 压缩后的张量
        """
        # 压缩
        compressed = self.compressor(x)
        
        # 注意力机制
        if self.attention is not None:
            # 转换维度用于注意力: [B, C, T] -> [T, B, C]
            B, C, T = compressed.shape
            compressed_attn = compressed.permute(2, 0, 1)  # [T, B, C]
            attn_out, _ = self.attention(compressed_attn, compressed_attn, compressed_attn)
            compressed = attn_out.permute(1, 2, 0)  # [B, C, T]
        
        # 归一化和激活
        compressed = compressed.transpose(1, 2)  # [B, T, C]
        compressed = self.norm(compressed)
        compressed = self.activation(compressed)
        compressed = compressed.transpose(1, 2)  # [B, C, T]
        
        # 残差连接
        if self.use_residual and self.residual_proj is not None:
            residual = self.residual_proj(x)
            # 下采样残差以匹配压缩后的时间维度
            if residual.shape[-1] != compressed.shape[-1]:
                residual = F.adaptive_avg_pool1d(residual, compressed.shape[-1])
            compressed = compressed + residual
        
        return compressed


class AttentionCompression(nn.Module):
    """
    基于注意力的压缩层
    将时间序列通过注意力机制压缩到更短的表示
    """
    def __init__(self, in_channels, out_channels, compression_ratio):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 投影层
        self.projection = nn.Linear(in_channels, out_channels)
        
        # 注意力层
        self.attention = nn.MultiheadAttention(out_channels, num_heads=8, batch_first=True)
        
        # 可学习的查询向量，用于压缩
        self.query = nn.Parameter(torch.randn(1, 1, out_channels))
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, C, T] - 输入张量
            
        Returns:
            compressed: [B, C', T'] - 压缩后的张量
        """
        B, C, T = x.shape
        
        # 计算压缩后的时间步数
        compressed_T = T // self.compression_ratio
        
        # 转换维度: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        
        # 投影到输出维度
        x_proj = self.projection(x)  # [B, T, out_channels]
        
        # 分组进行注意力压缩
        x_grouped = x_proj.reshape(B, compressed_T, self.compression_ratio, self.out_channels)
        x_grouped = x_grouped.reshape(B * compressed_T, self.compression_ratio, self.out_channels)
        
        # 使用可学习的查询向量进行压缩
        query = self.query.expand(B * compressed_T, -1, -1)  # [B*compressed_T, 1, out_channels]
        compressed, _ = self.attention(query, x_grouped, x_grouped)
        
        # 重塑: [B*compressed_T, 1, out_channels] -> [B, compressed_T, out_channels]
        compressed = compressed.squeeze(1)  # [B*compressed_T, out_channels]
        compressed = compressed.view(B, compressed_T, self.out_channels)
        
        # 转换回 [B, out_channels, compressed_T]
        compressed = compressed.transpose(1, 2)
        
        return compressed


class CausalPoolingCompression(nn.Module):
    """
    因果池化压缩，确保时间因果性
    只使用当前及之前的时间步进行压缩
    """
    def __init__(self, in_channels, out_channels, compression_ratio):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 投影层
        self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, C, T] - 输入张量
            
        Returns:
            compressed: [B, C', T'] - 压缩后的张量
        """
        B, C, T = x.shape
        
        # 投影到输出通道数
        x_proj = self.projection(x)  # [B, out_channels, T]
        
        # 计算压缩后的时间步数
        compressed_T = T // self.compression_ratio
        
        # 因果池化：只使用当前及之前的时间步
        compressed = []
        for i in range(compressed_T):
            start_idx = i * self.compression_ratio
            end_idx = start_idx + self.compression_ratio
            # 只取当前及之前的时间步
            window = x_proj[:, :, start_idx:end_idx]  # [B, out_channels, compression_ratio]
            # 对窗口进行平均池化
            pooled = torch.mean(window, dim=-1, keepdim=True)  # [B, out_channels, 1]
            compressed.append(pooled)
        
        # 拼接结果
        compressed = torch.cat(compressed, dim=-1)  # [B, out_channels, compressed_T]
        
        return compressed

