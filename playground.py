# import torch
# from mamba_ssm import Mamba

# batch, length, dim = 2, 64, 16
# x = torch.randn(batch, length, dim).to("cuda")
# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=dim, # Model dimension d_model
#     d_state=16,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")
# y = model(x)
# print(y.shape)
# assert y.shape == x.shape

# import flash_attn
# print("Hello, World!")


from hnet.models.hnet import HNet
from hnet.models.config_hnet import HNetConfig, SSMConfig, AttnConfig
import torch

# 2-stage hierarchical configuration
config = HNetConfig(
    arch_layout=["m1", ["m1", ["T1"], "m1"], "m1"],  # Your config
    d_model=[64, 64, 96],
    d_intermediate=[0, 128, 192],
    vocab_size=256,
    ssm_cfg=SSMConfig(
        chunk_size=64,
        d_conv=4,
        d_state=64,
        expand=2
    ),
    attn_cfg=AttnConfig(
        num_heads=[8, 8, 12],
        rotary_emb_dim=[8, 8, 8],
        window_size=[-1, -1, -1]
    )
)

# Initialize the model with bfloat16 dtype
model = HNet(config=config, stage_idx=0, dtype=torch.bfloat16).to("cuda")

# Prepare your input with bfloat16 dtype
batch, length, dim = 2, 64, 64
x = torch.randn(batch, length, dim, dtype=torch.bfloat16).to("cuda")

# Create a mask (required)
mask = torch.ones(batch, length, dtype=torch.bool, device="cuda")

# Forward pass
output, main_network_input, boundary_predictions = model(
    hidden_states=x,
    mask=mask,
    inference_params=None
)

print(f"Input shape: {x.shape}, dtype: {x.dtype}")
print(f"Output shape: {output.shape}, dtype: {output.dtype}")
print(f"Main network input shape: {main_network_input.shape}, dtype: {main_network_input.dtype}")
print(f"Shapes match: {output.shape == x.shape}")
print(f"Main network input shapes match: {output.shape == main_network_input.shape}")
print(f"Number of hierarchical stages: {len(boundary_predictions)}")