import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from hnet.models.hnet import HNet
# from layers.DC_Embed import TS_Dynamic_Chunking
from hnet.models.config_hnet import HNetConfig, SSMConfig, AttnConfig
from einops import rearrange
from layers.Embed import PositionalEmbedding

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
        self.d_model = configs.hnet_d_model[0]
        self.num_patches = configs.seq_len // configs.hnet_num_experts

        arch_layout = json.loads(configs.hnet_arch_layout)
        ssm_cfg = SSMConfig(
            d_conv=configs.hnet_ssm_d_conv,
            expand=configs.hnet_ssm_expand,
            d_state=configs.hnet_ssm_d_state,
            chunk_size=configs.hnet_ssm_chunk_size,
        )
        attn_cfg = AttnConfig(
            num_heads=configs.hnet_attn_num_heads,
            rotary_emb_dim=configs.hnet_attn_rotary_emb_dim,
            window_size=configs.hnet_attn_window_size,
        )
        hnet_cfg = HNetConfig(
            arch_layout=arch_layout,
            d_model=configs.hnet_d_model,
            d_intermediate=configs.hnet_d_intermediate,
            ssm_cfg=ssm_cfg,
            attn_cfg=attn_cfg,
        )

        self.value_embedding = ByteEmbedding(self.d_model)
        self.position_embedding = PositionalEmbedding(self.d_model)

        self.output_head = self.output_head = nn.Sequential(nn.Flatten(start_dim=-2), nn.Linear(self.seq_len * self.d_model, self.pred_len))
        self.hnet = HNet(config=hnet_cfg, stage_idx=0, dtype=torch.bfloat16)

        self.dropout = nn.Dropout(configs.dropout)
    
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

        x_enc = self.value_embedding(x_enc) 

        # print("after value embedding, the shape of x_enc is", x_enc.shape)
        # breakpoint()

        x_enc = x_enc + self.position_embedding(x_enc)

        x_enc = self.dropout(x_enc)

        x_enc = x_enc.to(torch.bfloat16)

        self.hnet = self.hnet.to(torch.bfloat16)
        hnet_output, main_network_output, boundary_predictions = self.hnet(
            hidden_states=x_enc,
            mask=mask,
            inference_params=None,
        )
        boundary_predictions = rearrange_boundary_predictions(boundary_predictions, self.c_in)
        # outer_prob = boundary_predictions[0].boundary_prob[:,:,:,1]
        # outer_prob = outer_prob.to(torch.float32)
        hnet_output = hnet_output.to(torch.float32)
        hnet_output = rearrange(hnet_output, '(b c) l d -> b c l d', c=self.c_in, d=self.d_model)

        '''
        outer_prob: boundary probability shape (B, C, seq_len), might help?
        hnet_output: shape (B, C, seq_len, d_model)
        want: (B, C, pred_len)
        '''

        hnet_output = self.output_head(hnet_output)
        # print("after output head, the shape of hnet_output is", hnet_output.shape)
        # breakpoint()
        hnet_output = hnet_output.permute(0, 2, 1)
    

        hnet_output = hnet_output * stdev + means

        return hnet_output, boundary_predictions

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, boundary_predictions = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], boundary_predictions