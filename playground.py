import sys
from typing import Dict

import torch

# Allow importing local modules when running this file directly
sys.path.append("/home/kgeng/usr/dynamic_chunking_ts")
from models.hnet.hnet.modules.dc import (
    RoutingModule,
    ChunkLayer,
    DeChunkLayer,
)


def demo_dynamic_chunking(seq: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Demo: Given seq [B, D, L] and mask [B, L],
    1) detect boundaries, 2) chunk to boundary-only tokens, 3) dechunk back to [B, D, L].

    No feature-wise mixing is performed (headdim == D), so feature dimension is preserved.
    """
    assert seq.dim() == 3, "seq must be [B, D, L]"
    assert mask.dim() == 2, "mask must be [B, L]"
    B, D, L = seq.shape
    assert mask.shape == (B, L), "mask must match batch and length"

    # Convert to [B, L, D] for the modules
    hidden_states = seq.transpose(1, 2).contiguous()
    mask_bool = mask.to(dtype=torch.bool)

    # 1) Routing: boundary detection via cosine dissimilarity
    routing = RoutingModule(d_model=D)
    routing_out = routing(hidden_states, mask=mask_bool)
    boundary_mask = routing_out.boundary_mask  # [B, L] booleans
    boundary_prob = routing_out.boundary_prob  # [B, L, 2]

    # 2) Chunk: keep only boundary tokens per sequence
    chunk = ChunkLayer()
    chunked_hidden_states, _, _, chunked_mask = chunk(
        hidden_states=hidden_states,
        boundary_mask=boundary_mask,
        mask=mask_bool,
    )  # chunked_hidden_states: [B, M, D], chunked_mask: [B, M]

    # (Optional) Apply heavier processing on chunked_hidden_states here.
    # For this demo, we keep them as-is (identity).

    # 3) Dechunk: expand back to full length via EMA; set headdim=D to avoid feature mixing
    dechunk = DeChunkLayer(d_model=D, headdim=D)
    reconstructed_bld = dechunk(
        hidden_states=chunked_hidden_states,
        boundary_mask=boundary_mask,
        boundary_prob=boundary_prob,
        mask=None,
    )  # [B, L, D]

    # Convert back to [B, D, L]
    reconstructed = reconstructed_bld.transpose(1, 2).contiguous()

    return {
        "boundary_mask": boundary_mask,
        "selected_probs": routing_out.selected_probs,  # [B, L, 1]
        "chunked_hidden_states": chunked_hidden_states,
        "chunked_mask": chunked_mask,
        "reconstructed": reconstructed,
    }


