import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PositionalEmbedding
import numpy as np



class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.position_embedding = PositionalEmbedding(configs.d_model)
        
        

        self.embedding = nn.Linear(configs.dec_in, configs.d_model)
        self.decoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)  # e_layers=2 encoder layers
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        x_dec[:, :self.seq_len, :] = x_dec[:, :self.seq_len, :] - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc
        x_dec[:, :self.seq_len, :] = x_dec[:, :self.seq_len, :] / std_enc
        decoder_input = self.embedding(x_dec) + self.position_embedding(x_dec)


        dec_out, _= self.decoder(decoder_input)

        dec_out = self.out_layer(dec_out)
        dec_out = dec_out * std_enc + mean_enc
        dec_out = dec_out.to(torch.float32)
        
        return dec_out



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':

            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

            return dec_out[:, -self.pred_len:, :]  

