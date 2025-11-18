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
from layers.Embed import PositionalEmbedding


class TS_Dynamic_Chunking(nn.Module):
    def __init__(self,c_in, d_model,dropout):
        super(TS_Dynamic_Chunking, self).__init__()
        # expected input shape: [(batch cin), seq_len, d_model]
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.c_in = c_in
        self.routing = RoutingModule(d_model=d_model)
        self.chunk = ChunkLayer()

    def forward(self, x, mask):
        # x: [(batch cin), seq_len, d_model]
        # mask: [(batch cin), seq_len]
        routing_out = self.routing(x, mask=mask)
        boundary_mask = routing_out.boundary_mask
        boundary_prob = routing_out.boundary_prob
        chunked_hidden_states,  _, _, _ = self.chunk(x, boundary_mask, mask=mask) # [(batch cin), num_boundary_tokens, d_model]
        chunked_hidden_states += 