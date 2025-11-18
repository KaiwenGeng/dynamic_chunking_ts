import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from einops import rearrange
from layers.Embed import PositionalEmbedding
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.hnet.hnet.modules.dc import (
    RoutingModule,
    ChunkLayer,
    DeChunkLayer,
)

def rearrange_boundary_predictions(boundary_predictions, c_in):
    for i in range(len(boundary_predictions)):
        # print(boundary_predictions[i].boundary_mask.shape)
        # print(boundary_predictions[i].boundary_prob.shape)
        # print(boundary_predictions[i].selected_probs.shape)
        boundary_predictions[i].boundary_mask = rearrange(boundary_predictions[i].boundary_mask, '(b c) l -> b c l', c=c_in)
        boundary_predictions[i].boundary_prob = rearrange(boundary_predictions[i].boundary_prob, '(b c) l d -> b c l d', c=c_in, d=2)
        boundary_predictions[i].selected_probs = rearrange(boundary_predictions[i].selected_probs, '(b c) l d -> b c l d', c=c_in, d=1)
    return boundary_predictions


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class ByteEmbedding(nn.Module):
    def __init__(self, d_model):
        super(ByteEmbedding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(256, d_model //4)
        self.flatten = nn.Flatten(start_dim=-2)
    def forward(self, x):
        # make sure x is float32
        assert x.dtype == torch.float32, "x must be float32"
        # print("before embedding, the shape of x is", x.shape)
        x = x.contiguous().view(torch.uint8).reshape(x.shape + (4,))
        x = x.long()
        result = self.embed(x)
        result = self.flatten(result)
        return result



class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.seq_len = configs.seq_len
        self.c_in = configs.enc_in
        self.d_model = configs.d_model
        self.bottleneck =  configs.patch_len


        self.value_embedding = ByteEmbedding(self.d_model)
        self.global_embedding = nn.Linear(self.seq_len, self.d_model * self.bottleneck)
        self.routing = RoutingModule(d_model=self.d_model)
        self.chunk_layer = ChunkLayer()
        self.decoder = Decoder(
                [
                    DecoderLayer(
                        # Self-attention: Attend to decoder input sequence
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
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
            )
        self.position_embedding1 = PositionalEmbedding(self.d_model)
        self.position_embedding2 = PositionalEmbedding(self.d_model)

        self.output_head = nn.Sequential(nn.Flatten(start_dim=-2), nn.Linear(self.bottleneck * self.d_model, self.pred_len))


    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        assert self.label_len == self.seq_len, "Label len must be the same as the seq len"
        assert torch.equal(x_enc, x_dec[:, :self.label_len, :]), "x_enc must be the same as the first part of x_dec"
        assert torch.equal(x_mark_enc, x_mark_dec[:,:self.label_len,:]) , "x_mark_enc must be the same as the first part of x_mark_dec"

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / stdev

        mask = torch.zeros(x_enc.shape[0], self.seq_len, dtype=torch.bool, device=x_dec.device)
        mask[:, :] = True  
        mask = mask.repeat(self.c_in, 1)

        x_enc = rearrange(x_enc, 'b l c -> (b c) l')

        x_global = rearrange(self.global_embedding(x_enc), '(b c) (l d) -> (b c) l d', c=self.c_in, l=self.bottleneck, d=self.d_model)

        hidden_states = self.value_embedding(x_enc) 

        bpred_output = self.routing(
            hidden_states,
            mask=mask,
        )

        hidden_states, _ , _, _ = self.chunk_layer(
            hidden_states, bpred_output.boundary_mask, None, mask=mask
        )
        # here, hidden_states is [(b c), num_chunks, d_model]
        # print("after chunk layer, the shape of hidden_states is", hidden_states.shape)
        # here, x_global is [(b c), bottleneck, d_model]
        # print("after chunk layer, the shape of x_global is", x_global.shape)
        decoder_output = self.decoder(x_global + self.position_embedding1(x_global), hidden_states + self.position_embedding2(hidden_states))
        # print("after decoder, the shape of decoder_output is", decoder_output.shape)
        decoder_output = self.output_head(decoder_output)
        # print("after output head, the shape of decoder_output is", decoder_output.shape)
        # breakpoint()
        decoder_output = rearrange(decoder_output, '(b c) l -> b l c', c=self.c_in)

    

        decoder_output = decoder_output * stdev + means

        boundary_predictions = rearrange_boundary_predictions([bpred_output], self.c_in)

        return decoder_output, boundary_predictions

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, boundary_predictions = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], boundary_predictions