import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding_Reconstruct

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, seq_len):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, min(latent_dim*2, seq_len // 2)),
            nn.ReLU(),
            nn.Linear(min(latent_dim*2, seq_len // 2), seq_len)
        )

    def forward(self, z):  # (B, C, latent_dim)
        return self.dec(z) 

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # vae parameters
        self.latent_dim = int(configs.compress_ratio * configs.seq_len)
        print(configs.compress_ratio, configs.seq_len, self.latent_dim)
        self.reconstruction_mode = configs.reconstruction_mode
        self.vae_decoder = VAEDecoder(self.latent_dim, configs.seq_len)

        # patching and embedding
        self.patch_embedding = PatchEmbedding_Reconstruct(
            configs.d_model, patch_len, stride, padding, configs.dropout, configs.enc_in, self.latent_dim, self.reconstruction_mode)

        
        

        # Encoder
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

        

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        original_input = x_enc.clone()
        # print(f"original_input shape: {original_input.shape}")
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model] / [bs x nvars x latent_dim]
        enc_out, sampled_latent, n_vars, mu, logvar = self.patch_embedding(x_enc)
       
        kl_loss =  -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        reconstructed_input = self.vae_decoder(sampled_latent)
        reconstructed_input = reconstructed_input.permute(0, 2, 1)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))


        reconstructed_input = reconstructed_input * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        reconstructed_input = reconstructed_input + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))

        reconstruction_loss = nn.MSELoss()(reconstructed_input, original_input)

        return dec_out, reconstructed_input,reconstruction_loss, kl_loss



    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, reconstructed_input,reconstruction_loss, kl_loss= self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], reconstructed_input,reconstruction_loss, kl_loss  # [B, L, D]

