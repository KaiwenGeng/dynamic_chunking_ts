import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Vanilla Transformer for Time Series Forecasting
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    
    Data Flow:
    1. Input: x_enc [B, seq_len, enc_in], x_dec [B, label_len+pred_len, dec_in]
    2. Encoder: Process historical data (seq_len=96)
    3. Decoder: Generate predictions (pred_len=96/192/336/720)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        
        # ========== 1. ENCODER EMBEDDING ==========
        # Convert input features to d_model dimension
        # Input: [B, seq_len, enc_in] -> Output: [B, seq_len, d_model]
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        
        # ========== 2. ENCODER STACK ==========
        # Multiple encoder layers with self-attention
        # Each layer: Self-Attention + Feed Forward + Residual Connection
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)  # e_layers=2 encoder layers
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # ========== 3. DECODER (for forecasting tasks) ==========
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Decoder embedding: Convert decoder input to d_model dimension
            # Input: [B, label_len+pred_len, dec_in] -> Output: [B, label_len+pred_len, d_model]
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            
            # Decoder stack: Multiple decoder layers
            # Each layer: Self-Attention + Cross-Attention + Feed Forward + Residual Connection
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        # Self-attention: Attend to decoder input sequence
                        AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        # Cross-attention: Attend to encoder output
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)  # d_layers=1 decoder layer
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                # Final projection: Convert d_model to output dimension
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Main forecasting function - shows complete data flow
        
        Args:
            x_enc: [B, seq_len, enc_in] - Historical input data (96 time steps)
            x_mark_enc: [B, seq_len, 4] - Time features for encoder input
            x_dec: [B, label_len+pred_len, dec_in] - Decoder input (48+96=144 time steps)
            x_mark_dec: [B, label_len+pred_len, 4] - Time features for decoder input
            
        Returns:
            dec_out: [B, label_len+pred_len, c_out] - Predicted values
        """
        
        # ========== STEP 1: ENCODER PROCESSING ==========
        # 1.1 Encoder Embedding: Convert raw features to d_model dimension
        # Input: [B, 96, 7] -> Output: [B, 96, d_model]
        # Combines: Value embedding + Positional embedding + Temporal embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # 1.2 Encoder Stack: Process historical data with self-attention
        # Input: [B, 96, d_model] -> Output: [B, 96, d_model]
        # Each encoder layer: Self-Attention + Feed Forward + Residual + LayerNorm
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # ========== STEP 2: DECODER PROCESSING ==========
        # 2.1 Decoder Embedding: Convert decoder input to d_model dimension
        # Input: [B, 144, 7] -> Output: [B, 144, d_model]
        # Note: 144 = label_len(48) + pred_len(96)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        
        # 2.2 Decoder Stack: Generate predictions using encoder context
        # Input: [B, 144, d_model] -> Output: [B, 144, c_out]
        # Each decoder layer: Self-Attention + Cross-Attention + Feed Forward + Residual + LayerNorm
        # Self-attention: Attend to decoder sequence
        # Cross-attention: Attend to encoder output (key insight!)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Main forward function - entry point for all tasks
        
        For long_term_forecast task:
        - Input: Historical data (96 steps) + Decoder input (48+96=144 steps)
        - Output: Only the predicted part (last 96 steps)
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Get full decoder output: [B, 144, c_out]
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # Return only predictions: [B, pred_len, c_out]
            # This extracts the last pred_len steps (the actual predictions)
            return dec_out[:, -self.pred_len:, :]  # [B, 96, 7] for pred_len=96
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
