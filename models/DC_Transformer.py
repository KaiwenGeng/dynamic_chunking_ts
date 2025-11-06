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



class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.seq_len = configs.seq_len
        # self.dc_embedding = TS_Dynamic_Chunking(configs.enc_in, configs.d_model, configs.dropout)
        self.c_in = configs.enc_in
        self.d_model = configs.d_model

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

        self.value_embedding = nn.Linear(self.seq_len + self.pred_len, (self.seq_len + self.pred_len) * self.d_model)
        self.position_embedding = PositionalEmbedding(configs.d_model)
        
        self.hnet = HNet(config=hnet_cfg, stage_idx=0, dtype=torch.bfloat16)
        self.output_head = nn.Linear(configs.d_model, 1)
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        assert self.label_len == self.seq_len, "Label len must be the same as the seq len"
        assert torch.equal(x_enc, x_dec[:, :self.label_len, :]), "x_enc must be the same as the first part of x_dec"
        assert torch.equal(x_mark_enc, x_mark_dec[:,:self.label_len,:]) , "x_mark_enc must be the same as the first part of x_mark_dec"

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        x_dec[:, :self.label_len, :] = x_dec[:, :self.label_len, :] - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / stdev
        x_dec[:, :self.label_len, :] = x_dec[:, :self.label_len, :] / stdev

        mask = torch.zeros(x_dec.shape[0], self.seq_len + self.pred_len, dtype=torch.bool, device=x_dec.device)
        mask[:, :self.label_len] = True  # True for lookback, False for padded forecast  # [batch, seq_len + pred_len]
        mask = mask.repeat(self.c_in, 1)

        x_dec = rearrange(x_dec, 'b l c -> b c l')

        x_dec = self.value_embedding(x_dec)

        x_dec = rearrange(x_dec, 'b c (l d) -> (b c) l d', d=self.d_model)

        x_dec = x_dec + self.position_embedding(x_dec)
        x_dec = x_dec.to(torch.bfloat16)
        self.hnet = self.hnet.to(torch.bfloat16)
        hnet_output, main_network_input, boundary_predictions = self.hnet(
            hidden_states=x_dec,
            mask=mask,
            inference_params=None,
        )
        hnet_output = hnet_output.to(torch.float32)
        hnet_output = self.output_head(hnet_output)
        hnet_output = rearrange(hnet_output, '(b c) l 1 -> b l c', c=self.c_in)

        hnet_output = hnet_output * stdev + means
        
        return hnet_output, boundary_predictions

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, router_output = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]