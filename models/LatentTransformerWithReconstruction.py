import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .FlexibleTemporalEncoder import FlexibleTemporalEncoder
from .FlexibleTemporalDecoder import FlexibleTemporalDecoder
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class LatentTransformerEncoder(nn.Module):
    """
    Encoder for LatentTransformer with VAE/MAE support
    
    支持三种模式：
    1. AE (Auto-Encoder): 确定性编码
    2. VAE (Variational Auto-Encoder): 概率编码，带KL散度
    3. MAE (Masked Auto-Encoder): 带masking的编码
    """
    def __init__(self, input_dim, latent_dim, compression_ratios, channel_dims, 
                 compression_type='conv', mode='AE', mask_ratio=0.0):
        super().__init__()
        self.mode = mode
        self.mask_ratio = mask_ratio
        self.latent_dim = latent_dim
        
        # 时间压缩编码器
        self.temporal_encoder = FlexibleTemporalEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            compression_ratios=compression_ratios,
            channel_dims=channel_dims,
            compression_type=compression_type
        )
        
        # VAE的均值和方差投影
        if mode == 'VAE':
            self.mu_proj = nn.Linear(latent_dim, latent_dim)
            self.logvar_proj = nn.Linear(latent_dim, latent_dim)
    
    def reparameterize(self, mu, logvar):
        """VAE重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def random_masking(self, x, mask_ratio):
        """
        MAE风格的随机masking
        
        Args:
            x: [B, C, T]
            mask_ratio: 要mask的比例
            
        Returns:
            x_masked: [B, C, T] - masked后的输入
            mask: [B, T] - mask标记 (0表示被mask)
        """
        B, C, T = x.shape
        len_keep = int(T * (1 - mask_ratio))
        
        # 生成随机噪声
        noise = torch.rand(B, T, device=x.device)
        
        # 排序得到mask indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 生成mask: 0是被mask的，1是保留的
        mask = torch.ones([B, T], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # 对输入进行masking
        mask_expanded = mask.unsqueeze(1).expand_as(x)  # [B, C, T]
        x_masked = x * (1 - mask_expanded)  # mask的位置置为0
        
        return x_masked, mask, ids_restore
    
    def forward(self, x, training=True):
        """
        前向传播
        
        Args:
            x: [B, C, T] - 输入时间序列
            training: 是否在训练模式
            
        Returns:
            根据mode不同返回不同内容：
            - AE: latent, skip_features, None, None, None
            - VAE: latent, skip_features, mu, logvar, None
            - MAE: latent, skip_features, None, None, mask
        """
        # MAE: 在训练时进行masking
        mask = None
        ids_restore = None
        if self.mode == 'MAE' and training and self.mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        # 编码
        latent, skip_features = self.temporal_encoder(x)
        
        # VAE: 计算均值和方差
        mu, logvar = None, None
        if self.mode == 'VAE':
            # latent: [B, latent_dim, T']
            latent_flat = latent.transpose(1, 2)  # [B, T', latent_dim]
            mu = self.mu_proj(latent_flat)
            logvar = self.logvar_proj(latent_flat)
            
            # 重参数化
            if training:
                latent = self.reparameterize(mu, logvar)
            else:
                latent = mu
            
            latent = latent.transpose(1, 2)  # [B, latent_dim, T']
        
        return latent, skip_features, mu, logvar, mask


class Model(nn.Module):
    """
    Latent Space Transformer with Reconstruction Support
    
    支持三种重建模式：
    1. AE (Auto-Encoder): 基本重建损失
    2. VAE (Variational Auto-Encoder): 重建损失 + KL散度
    3. MAE (Masked Auto-Encoder): Masked重建损失
    
    Args:
        configs: 配置对象，包含：
            - reconstruction_mode: 'AE', 'VAE', 'MAE', 'None'
            - mask_ratio: MAE的masking比例 (default: 0.25)
            - 其他LatentTransformer的标准参数
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        
        # 获取重建模式
        self.reconstruction_mode = getattr(configs, 'reconstruction_mode', 'None')
        self.mask_ratio = getattr(configs, 'mask_ratio', 0.25)
        
        # 获取压缩配置
        compression_ratios = getattr(configs, 'compression_ratios', [2, 2, 2])
        channel_dims = getattr(configs, 'channel_dims', [64, 128, 256])
        compression_type = getattr(configs, 'compression_type', 'conv')
        latent_dim = getattr(configs, 'latent_dim', 128)
        
        self.total_compression_ratio = int(np.prod(compression_ratios))
        
        print(f"\n{'='*60}")
        print(f"[LatentTransformer with Reconstruction] Initializing model:")
        print(f"  - Input dim: {configs.enc_in}")
        print(f"  - Latent dim: {latent_dim}")
        print(f"  - Output dim: {configs.c_out}")
        print(f"  - Compression ratios: {compression_ratios}")
        print(f"  - Total compression: {self.total_compression_ratio}x")
        print(f"  - Channel dims: {channel_dims}")
        print(f"  - Compression type: {compression_type}")
        print(f"  - Reconstruction mode: {self.reconstruction_mode}")
        if self.reconstruction_mode == 'MAE':
            print(f"  - Mask ratio: {self.mask_ratio}")
        print(f"{'='*60}\n")
        
        # ========== 1. 带重建支持的TEMPORAL ENCODER ==========
        self.temporal_encoder = LatentTransformerEncoder(
            input_dim=configs.enc_in,
            latent_dim=latent_dim,
            compression_ratios=compression_ratios,
            channel_dims=channel_dims,
            compression_type=compression_type,
            mode=self.reconstruction_mode if self.reconstruction_mode != 'None' else 'AE',
            mask_ratio=self.mask_ratio
        )
        
        # ========== 2. 重建DECODER（用于重建encoder输入） ==========
        if self.reconstruction_mode != 'None':
            self.reconstruction_decoder = FlexibleTemporalDecoder(
                latent_dim=latent_dim,
                output_dim=configs.enc_in,
                compression_ratios=compression_ratios,
                channel_dims=channel_dims[::-1],
                compression_type=compression_type
            )
        
        # ========== 3. LATENT SPACE EMBEDDINGS ==========
        self.enc_embedding = DataEmbedding(
            latent_dim, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        
        # ========== 4. TRANSFORMER ENCODER ==========
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=False), 
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # ========== 5. TRANSFORMER DECODER ==========
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding(
                latent_dim, configs.d_model, configs.embed, configs.freq, configs.dropout
            )
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads
                        ),
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads
                        ),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    ) for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, latent_dim, bias=True)
            )
        
        # ========== 6. 预测DECODER（用于最终预测输出） ==========
        self.prediction_decoder = FlexibleTemporalDecoder(
            latent_dim=latent_dim,
            output_dim=configs.c_out,
            compression_ratios=compression_ratios,
            channel_dims=channel_dims[::-1],
            compression_type=compression_type
        )
    
    def compute_kl_loss(self, mu, logvar):
        """计算KL散度损失"""
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # 归一化
        kl_loss = kl_loss / (mu.size(0) * mu.size(1) * mu.size(2))
        return kl_loss
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, training=True):
        """
        长期预测 + 重建
        
        Returns:
            prediction: [B, T, C] - 预测输出
            reconstructed_input: [B, T, C] - 重建的输入（如果有重建模式）
            reconstruction_loss: Tensor - 重建损失
            kl_loss: Tensor - KL散度损失（VAE模式）
        """
        # ========== Step 1: 编码并可能进行重建 ==========
        latent_enc, skip_features_enc, mu, logvar, mask = self.temporal_encoder(
            x_enc.transpose(1, 2), training=training
        )
        
        # 计算重建损失
        reconstruction_loss = torch.tensor(0.0).to(x_enc.device)
        reconstructed_input = None
        kl_loss = torch.tensor(0.0).to(x_enc.device)
        
        if self.reconstruction_mode != 'None' and training:
            # 重建encoder输入
            reconstructed_input = self.reconstruction_decoder(latent_enc, skip_features_enc)
            reconstructed_input = reconstructed_input.transpose(1, 2)  # [B, T, C]
            
            # 计算重建损失
            if self.reconstruction_mode == 'MAE' and mask is not None:
                # MAE: 只计算masked部分的损失
                mask_expanded = mask.unsqueeze(-1).expand_as(x_enc)  # [B, T, C]
                reconstruction_loss = F.mse_loss(
                    reconstructed_input * mask_expanded,
                    x_enc * mask_expanded
                )
            else:
                # AE/VAE: 计算全部的重建损失
                reconstruction_loss = F.mse_loss(reconstructed_input, x_enc)
            
            # VAE: 计算KL散度
            if self.reconstruction_mode == 'VAE' and mu is not None:
                kl_loss = self.compute_kl_loss(mu, logvar)
        
        # ========== Step 2: Transformer处理 ==========
        # 压缩时间特征
        compressed_T = latent_enc.shape[-1]
        x_mark_enc_compressed = F.interpolate(
            x_mark_enc.transpose(1, 2), 
            size=compressed_T, 
            mode='linear', 
            align_corners=False
        ).transpose(1, 2)
        
        # Embed和encode
        latent_enc_seq = latent_enc.transpose(1, 2)  # [B, T', latent_dim]
        enc_out = self.enc_embedding(latent_enc_seq, x_mark_enc_compressed)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # ========== Step 3: Decoder输入编码 ==========
        x_dec_compressed, skip_features_dec, _, _, _ = self.temporal_encoder(
            x_dec.transpose(1, 2), training=False  # decoder输入不需要masking
        )
        latent_dec = x_dec_compressed.transpose(1, 2)
        
        # 压缩decoder时间特征
        compressed_T_dec = latent_dec.shape[1]
        x_mark_dec_compressed = F.interpolate(
            x_mark_dec.transpose(1, 2), 
            size=compressed_T_dec, 
            mode='linear', 
            align_corners=False
        ).transpose(1, 2)
        
        # ========== Step 4: Transformer Decoder ==========
        dec_out = self.dec_embedding(latent_dec, x_mark_dec_compressed)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        
        # ========== Step 5: 解压缩到原始时间空间 ==========
        dec_out = dec_out.transpose(1, 2)  # [B, latent_dim, T']
        output = self.prediction_decoder(dec_out, skip_features_dec)
        output = output.transpose(1, 2)  # [B, T, C]
        
        return output, reconstructed_input, reconstruction_loss, kl_loss
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        前向传播
        
        Returns:
            如果reconstruction_mode != 'None':
                prediction, reconstructed_input, reconstruction_loss, kl_loss
            否则:
                prediction
        """
        training = self.training
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output, reconstructed_input, reconstruction_loss, kl_loss = self.forecast(
                x_enc, x_mark_enc, x_dec, x_mark_dec, training=training
            )
            
            # 只返回预测部分
            prediction = output[:, -self.pred_len:, :]
            
            # 根据reconstruction_mode返回不同的内容
            if self.reconstruction_mode != 'None':
                return prediction, reconstructed_input, reconstruction_loss, kl_loss
            else:
                return prediction
        
        return None

