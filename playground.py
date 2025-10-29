# # import torch
# # from mamba_ssm import Mamba

# # batch, length, dim = 2, 64, 16
# # x = torch.randn(batch, length, dim).to("cuda")
# # model = Mamba(
# #     # This module uses roughly 3 * expand * d_model^2 parameters
# #     d_model=dim, # Model dimension d_model
# #     d_state=16,  # SSM state expansion factor
# #     d_conv=4,    # Local convolution width
# #     expand=2,    # Block expansion factor
# # ).to("cuda")
# # y = model(x)
# # print(y.shape)
# # assert y.shape == x.shape

# # import flash_attn
# # print("Hello, World!")









# from hnet.models.hnet import HNet
# from hnet.models.config_hnet import HNetConfig, SSMConfig, AttnConfig
# import torch

# # 2-stage hierarchical configuration
# config = HNetConfig(
#     arch_layout=["m1", ["m1", ["T1"], "m1"], "m1"],  
#     # T/t: Attention with/without SwiGLU MLP, M/m: Mamba2 with/without SwiGLU MLP
#     # number means number of layers in the block, e.g., m1 means 1 Mamba2 layer, T1 means 1 Attention layer with SwiGLU MLP
#     d_model=[32, 64, 128],
#     #Channel width per stage. 
#     d_intermediate=[0, 128, 192],  # used by uppercase blocks only, e.g., T1 / M1
#     ssm_cfg=SSMConfig(
#         chunk_size=64, # performance tuning parameter, no effect on accuracy
#         d_conv=4,
#         d_state=32,
#         expand=2,
#     ),
#     attn_cfg=AttnConfig(
#         num_heads=[2, 4, 8],
#         rotary_emb_dim=[8, 8, 8],
#         window_size=[-1, -1, -1]
#     )
#     # 1 = full causal attention, >0 = sliding window length.
# )

# # Initialize the model with bfloat16 dtype
# model = HNet(config=config, stage_idx=0, dtype=torch.bfloat16).to("cuda")
# # Prepare your input with bfloat16 dtype
# batch, length, dim = 2, 96, 32
# x = torch.randn(batch, length, dim, dtype=torch.bfloat16).to("cuda")

# # Create a mask (required)
# mask = torch.ones(batch, length, dtype=torch.bool, device="cuda") # True for valid tokens, False for masked tokens

# # Forward pass
# output, main_network_input, boundary_predictions = model(
#     hidden_states=x,
#     mask=mask,
#     inference_params=None
# )

# print(f"Input shape: {x.shape}, dtype: {x.dtype}")
# print(f"Output shape: {output.shape}, dtype: {output.dtype}")
# print(f"Main network input shape: {main_network_input.shape}, dtype: {main_network_input.dtype}")
# print(f"Shapes match: {output.shape == x.shape}")
# print(f"Main network input shapes match: {output.shape == main_network_input.shape}")
# print(f"Number of hierarchical stages: {len(boundary_predictions)}")










# # import torch
# # from mamba_ssm import Mamba

# # batch, length, dim = 2, 64, 16
# # x = torch.randn(batch, length, dim).to("cuda")
# # model = Mamba(
# #     # This module uses roughly 3 * expand * d_model^2 parameters
# #     d_model=dim, # Model dimension d_model
# #     d_state=16,  # SSM state expansion factor
# #     d_conv=4,    # Local convolution width
# #     expand=2,    # Block expansion factor
# # ).to("cuda")
# # y = model(x)
# # print(y.shape)
# # assert y.shape == x.shape

# # import flash_attn
# # print("Hello, World!")





# Quick test - you can run this in Python to check:
import mamba_ssm
print(mamba_ssm.__file__)
print(mamba_ssm.__version__)
