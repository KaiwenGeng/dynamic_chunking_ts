import torch
from torch import nn
from layers.DC_Embed import TS_Dynamic_Chunking
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.hnet.hnet.modules.dc import DeChunkLayer, RoutingModuleOutput

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.ts_dynamic_chunking = TS_Dynamic_Chunking(configs.enc_in, configs.d_model, configs.dropout)
        self.dechunk = DeChunkLayer(d_model=configs.d_model * configs.enc_in, headdim=configs.n_heads, block_size=configs.hnet_ssm_chunk_size)
        self.c_in = configs.enc_in
        self.d_model = configs.d_model
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
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        assert self.label_len == self.seq_len, "Label len must be the same as the seq len"
        assert torch.equal(x_enc, x_dec[:, :self.label_len, :]), "x_enc must be the same as the first part of x_dec"
        assert torch.equal(x_mark_enc, x_mark_dec[:,:self.label_len,:]) , "x_mark_enc must be the same as the first part of x_mark_dec"
        # print("===original x_enc and x_dec shape===")
        # print(f"x_enc shape: {x_enc.shape}")
        # print(f"x_dec shape: {x_dec.shape}")
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        x_dec[:, :self.label_len, :] = x_dec[:, :self.label_len, :] - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / stdev
        x_dec[:, :self.label_len, :] = x_dec[:, :self.label_len, :] / stdev

        mask = torch.zeros(x_dec.shape[0], self.seq_len + self.pred_len, dtype=torch.bool, device=x_dec.device)
        mask[:, :self.seq_len ] = True  # True for lookback, False for padded forecast  # [batch, seq_len + pred_len]


        x_dec = x_dec.permute(0, 2, 1) # [batch, c_in, seq_len + pred_len]

        # print("===x_dec shape after permute===")
        
        # print(f"x_dec shape: {x_dec.shape}")
        # print(f"mask shape: {mask.shape}") 
        # print("===ts_dynamic_chunking===")
        chunked_hidden_states, boundary_mask, boundary_prob, chunked_mask= self.ts_dynamic_chunking(x_dec, mask) # [batch*c_in, number of chunks, d_model], [batch, seq_len + pred_len], [batch, seq_len + pred_len, 2], [batch, number of chunks]

        # Wrap routing outputs for load balancing loss compatibility
        selected_probs = boundary_prob.max(dim=-1).values.unsqueeze(-1)
        router_output = RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
        )
        # print("===out of ts_dynamic_chunking===")
        # print(f"chunked_hidden_states shape: {chunked_hidden_states.shape}")
        # print(f"boundary_mask shape: {boundary_mask.shape}")
        # print(f"boundary_prob shape: {boundary_prob.shape}")
        '''
        TBD
        '''
        mask_bc = chunked_mask.repeat_interleave(self.c_in, dim=0)  # [batch*c_in, number of chunks]
        chunked_hidden_states = chunked_hidden_states.masked_fill(~mask_bc.unsqueeze(-1), 0.0)
        '''
        TBD
        '''
        
        enc_out, _ = self.encoder(chunked_hidden_states) # [batch*c_in, number of chunks, d_model]
        # print(f"enc_out after seq2seq shape: {enc_out.shape}")
        number_of_chunks = enc_out.shape[1]
        enc_out = enc_out.reshape(-1, number_of_chunks, self.c_in * self.d_model) # [batch, number of chunks, c_in * d_model]
        # print(f"enc_out after reshape back for dechunk layer: {enc_out.shape}") 
        # apply the dechunk layer
        dec_out = self.dechunk(enc_out, boundary_mask, boundary_prob, mask=None) # [batch, seq_len + pred_len, c_in * d_model]
        dec_out = dec_out.reshape(-1, self.seq_len + self.pred_len, self.c_in, self.d_model) # [batch, seq_len + pred_len, c_in, d_model]
        dec_out = dec_out.mean(dim=-1) # [batch, seq_len + pred_len, c_in]



        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len + self.pred_len, 1))
        # print(f"dec_out after mean on d_model dimension: {dec_out.shape}") 
        # breakpoint()

        return dec_out, router_output
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, router_output = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], [router_output]  # [B, L, D], RoutingModuleOutput