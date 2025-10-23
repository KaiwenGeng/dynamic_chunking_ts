import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .FlexibleTemporalEncoder import FlexibleTemporalEncoder
from .FlexibleTemporalDecoder import FlexibleTemporalDecoder
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Latent Space Transformer for Time Series Forecasting
    
    Architecture:
        Encoder Path: Input → (Early Merge: Embed → Compress) → Latent Space → Transformer Encoder
        Decoder Path: Decoder Input → Causal Compress → Latent Space → Transformer Decoder → Decompress → Output
    
    Key Features:
        - Dual temporal encoders: regular conv for encoder, causal conv for decoder
        - Flexible temporal compression (2x, 4x, 8x, 16x, ...)
        - Multi-scale feature extraction with skip connections
        - Optional early merge of time features (default: True)
        - Maintains causality for prediction tasks
        - Reduced computational complexity for long sequences
        - Better long-horizon forecasting capability
    
    Args:
        configs: Configuration object with attributes:
            - enc_in: Input feature dimension
            - dec_in: Decoder input feature dimension
            - c_out: Output feature dimension
            - d_model: Transformer model dimension
            - latent_dim: Latent space dimension
            - compression_ratios: List of compression ratios per layer
            - channel_dims: List of channel dimensions per layer
            - compression_type: 'conv', 'pool', or 'attention'
            - early_merge: Whether to merge time features before compression (default: True)
            - n_heads: Number of attention heads
            - e_layers: Number of encoder layers
            - d_layers: Number of decoder layers
            - d_ff: Feed-forward dimension
            - dropout: Dropout rate
            - activation: Activation function
            - factor: Attention factor
            - freq: Time feature frequency
            - embed: Embedding type
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        
        # Get compression configuration
        compression_ratios = getattr(configs, 'compression_ratios', [2, 2, 2])
        channel_dims = getattr(configs, 'channel_dims', [64, 128, 256])
        compression_type = getattr(configs, 'compression_type', 'conv')
        latent_dim = getattr(configs, 'latent_dim', 128)
        
        # Feature fusion configuration
        self.early_merge = getattr(configs, 'early_merge', True)
        
        self.total_compression_ratio = int(np.prod(compression_ratios))
        
        print(f"\n{'='*60}")
        print(f"[LatentTransformer] Initializing model:")
        print(f"  - Input dim: {configs.enc_in}")
        print(f"  - Latent dim: {latent_dim}")
        print(f"  - Output dim: {configs.c_out}")
        print(f"  - Compression ratios: {compression_ratios}")
        print(f"  - Total compression: {self.total_compression_ratio}x")
        print(f"  - Channel dims: {channel_dims}")
        print(f"  - Compression type: {compression_type}")
        print(f"{'='*60}\n")
        
        # ========== 1. TEMPORAL ENCODERS (Compression) ==========
        # Encoder for Transformer Encoder input (can use future info)
        self.enc_temporal_encoder = FlexibleTemporalEncoder(
            input_dim=configs.enc_in,
            latent_dim=latent_dim,
            compression_ratios=compression_ratios,
            channel_dims=channel_dims,
            compression_type=compression_type  # Regular conv for encoder
        )
        
        # Encoder for early merge mode (input is already embedded)
        if self.early_merge:
            # Create a temporal encoder that works with embedded input
            # Use channel_dims directly, FlexibleTemporalEncoder will add input_dim and latent_dim
            self.embedded_temporal_encoder = FlexibleTemporalEncoder(
                input_dim=configs.d_model,
                latent_dim=latent_dim,
                compression_ratios=compression_ratios,
                channel_dims=channel_dims,
                compression_type=compression_type
            )
        
        # Encoder for Transformer Decoder input (must be causal)
        self.dec_temporal_encoder = FlexibleTemporalEncoder(
            input_dim=configs.dec_in,
            latent_dim=latent_dim,
            compression_ratios=compression_ratios,
            channel_dims=channel_dims,
            compression_type='causal_conv'  # Causal conv for decoder
        )
        
        # ========== 2. LATENT SPACE EMBEDDINGS ==========
        # Embedding for encoder input in latent space
        self.enc_embedding = DataEmbedding(
            latent_dim, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        
        # Embedding for raw input (used in early merge mode)
        self.input_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        
        # ========== 3. TRANSFORMER ENCODER ==========
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
        
        # ========== 4. TRANSFORMER DECODER (if needed) ==========
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Embedding for decoder input in latent space
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
        
        # ========== 5. TEMPORAL DECODER (Decompression) ==========
        self.temporal_decoder = FlexibleTemporalDecoder(
            latent_dim=latent_dim,
            output_dim=configs.c_out,
            compression_ratios=compression_ratios,
            channel_dims=channel_dims[::-1],  # Reverse for decoder
            compression_type=compression_type  # Regular conv for decoder output
        )
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Long-term forecasting
        
        Data Flow:
        1. x_enc [B, seq_len, enc_in] → Temporal Encoder → latent_enc [B, latent_dim, compressed_T]
        2. latent_enc [B, compressed_T, latent_dim] → Transformer Encoder → enc_out
        3. x_dec [B, label_len+pred_len, dec_in] → Temporal Encoder → latent_dec
        4. latent_dec + enc_out → Transformer Decoder → latent_pred
        5. latent_pred [B, latent_dim, compressed_T] → Temporal Decoder → output [B, T, c_out]
        """
        # ========== Step 1: Compress encoder input to latent space ==========
        # Input: [B, seq_len, enc_in]
        if self.early_merge:
            # Early merge: embed features before compression
            x_enc_embedded = self.input_embedding(x_enc, x_mark_enc)
            x_enc_compressed, skip_features_enc = self.embedded_temporal_encoder(x_enc_embedded.transpose(1, 2))
        else:
            # Late merge: compress first, then embed
            x_enc_compressed, skip_features_enc = self.enc_temporal_encoder(x_enc.transpose(1, 2))
        # Output: [B, latent_dim, compressed_T], skip_features
        
        # ========== Step 2: Embed and process in Transformer Encoder ==========
        # Convert to [B, compressed_T, latent_dim] for embedding
        latent_enc = x_enc_compressed.transpose(1, 2)
        
        # Calculate compressed time steps for time features
        compressed_T = latent_enc.shape[1]
        
        # Handle embedding based on merge setting
        if self.early_merge:
            # Features already embedded, project to d_model dimension
            enc_out = self.enc_embedding(latent_enc, None)
        else:
            # Downsample time features to match compressed sequence length
            x_mark_enc_compressed = F.interpolate(
                x_mark_enc.transpose(1, 2), 
                size=compressed_T, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            # Embed and encode
            enc_out = self.enc_embedding(latent_enc, x_mark_enc_compressed)
        
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # Output: [B, compressed_T, d_model]
        
        # ========== Step 3: Compress decoder input to latent space ==========
        # Input: [B, label_len+pred_len, dec_in]
        # Use causal temporal encoder for decoder input to maintain causality
        x_dec_compressed, skip_features_dec = self.dec_temporal_encoder(x_dec.transpose(1, 2))
        latent_dec = x_dec_compressed.transpose(1, 2)
        # Output: [B, compressed_T', latent_dim]
        
        # Downsample decoder time features
        compressed_T_dec = latent_dec.shape[1]
        x_mark_dec_compressed = F.interpolate(
            x_mark_dec.transpose(1, 2), 
            size=compressed_T_dec, 
            mode='linear', 
            align_corners=False
        ).transpose(1, 2)
        
        # ========== Step 4: Decode in latent space ==========
        dec_out = self.dec_embedding(latent_dec, x_mark_dec_compressed)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        # Output: [B, compressed_T', latent_dim]
        
        # ========== Step 5: Decompress to original time space ==========
        # Convert back to [B, latent_dim, compressed_T']
        dec_out = dec_out.transpose(1, 2)
        
        # Decode using skip connections from decoder path
        output = self.temporal_decoder(dec_out, skip_features_dec)
        # Output: [B, c_out, T']
        
        # Convert back to [B, T', c_out]
        output = output.transpose(1, 2)
        
        return output
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass
        
        Args:
            x_enc: [B, seq_len, enc_in] - Encoder input
            x_mark_enc: [B, seq_len, time_features] - Encoder time features
            x_dec: [B, label_len+pred_len, dec_in] - Decoder input
            x_mark_dec: [B, label_len+pred_len, time_features] - Decoder time features
            
        Returns:
            output: [B, pred_len, c_out] - Predictions
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # Return only the prediction part
            return dec_out[:, -self.pred_len:, :]
        
        # For other tasks, implement as needed
        return None

