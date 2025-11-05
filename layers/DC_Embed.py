from models.hnet.hnet.modules.dc import (
    RoutingModule,
    ChunkLayer,
    DeChunkLayer,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class TS_Dynamic_Chunking(nn.Module):
    def __init__(self,c_in, d_model,dropout):
        super(TS_Dynamic_Chunking, self).__init__()
        # expected input shape: [batch, c_in, seq_len + pred_len]
        # Embed first: lift each channel independently to d_model dimensions
        self.value_lift = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in * d_model,
            kernel_size=1,
            groups=c_in,          # keeps channels independent
            bias=False
        )
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.c_in = c_in
        self.routing = RoutingModule(d_model=d_model * c_in)
        self.chunk = ChunkLayer()
        

    def forward(self, x, mask):
        # x: [batch, c_in, seq_len + pred_len]
        # mask: [batch, seq_len + pred_len]
        # print(f"x shape: {x.shape}")
        total_length = x.shape[2]
        valid_m = mask.sum(dim=1)
        unmasked_length = int(valid_m.max().item())
        
        # print(f"total_length: {total_length}")
        # print(f"unmasked_length: {unmasked_length}")
        # make sure valid length values are all equal to unmasked_length
        assert (valid_m == unmasked_length).all(), "valid length values must be all equal to unmasked_length"
        
        x = self.value_lift(x[:, :, :unmasked_length])
        # print(f"x shape after lift: {x.shape}") # should be [batch, c_in * d_model, seq_len]
        # padd the masked length with 0 
        tempx = x.clone()
        x = F.pad(x, (0, total_length - unmasked_length), mode="constant", value=0)
        # print(f"does the x and tempx are the same up to unmasked_length: {torch.equal(x[:, :, :unmasked_length], tempx[:, :, :unmasked_length])}")
        # print(f"is x all zero after the unmasked_length: {torch.all(x[:, :, unmasked_length:] == 0)}")
        # print(f"x shape after pad: {x.shape}") # should be [batch, c_in * d_model, seq_len + pred_len(total_length)]
        assert x.shape[1] == self.d_model * self.c_in, "x shape after pad must be [batch, c_in * d_model, seq_len + pred_len(total_length)]"
        assert x.shape[2] == total_length, "x shape after pad must be [batch, c_in * d_model, seq_len + pred_len(total_length)]"
        assert mask.dtype == torch.bool, "mask must be bool type"

        dc_hidden = x.transpose(1, 2).contiguous() # should be [batch, seq_len + pred_len(total_length), c_in * d_model]
        routing_out = self.routing(dc_hidden, mask=mask)
        boundary_mask = routing_out.boundary_mask
        boundary_prob = routing_out.boundary_prob
        chunked_hidden_states,  _, _, chunked_mask = self.chunk(dc_hidden, boundary_mask, mask=mask) # both next_cu_seqlens and next_max_seqlen are None since we provide mask
        # chunked_hidden_states should be [batch, number of chunks, c_in * d_model]
        # print(f"chunked_hidden_states shape: {chunked_hidden_states.shape}")
        # reshape to preserve channel independence for the upcomming transformer
        chunked_hidden_states = rearrange(chunked_hidden_states, 'b n (c d) -> (b c) n d', c=self.c_in, d=self.d_model) # should be [batch*c_in, number of chunks, d_model]
        chunked_hidden_states = self.dropout(chunked_hidden_states)


        return chunked_hidden_states, boundary_mask, boundary_prob, chunked_mask



