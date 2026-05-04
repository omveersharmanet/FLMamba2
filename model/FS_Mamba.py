import torch
import torch.nn as nn
from layers.FMamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted
from mamba_ssm import Mamba

from layers.SelfAttention_Family import Mahalanobis_mask_freq,FullAttention, AttentionLayer
from einops import rearrange
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.mamba_embedding=nn.Linear(configs.d_model,configs.d_model)
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        self.class_strategy = configs.class_strategy

        self.encoder = Encoder(
            [
                EncoderLayer(
                        Mamba(
                                d_model=configs.d_model,  # Model dimension d_model
                                d_state=configs.d_state,  # SSM state expansion factors
                                d_conv=2,  # Local convolution width
                                expand=1,  # Block expansion factor)
                                ),    
                        Mamba(
                                d_model=configs.d_model,  # Model dimension d_model
                                d_state=configs.d_state,  # SSM state expansion factors
                                d_conv=2,  # Local convolution width
                                expand=1,  # Block expansion factor)
                                ),    
                        AttentionLayer(
                            FullAttention(
                                True,
                                configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=configs.output_attention,
                            ),
                            configs.d_model,
                            configs.n_heads,
                        ),                    
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation, enc_in=configs.enc_in,#mamba2_cfg.hidden_size,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )        
               # --- Output projection ---
        self.conv1 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=configs.d_ff, out_channels=configs.pred_len, kernel_size=1)
        self.dropout = nn.Dropout(configs.dropout)
        self.activation = F.relu if configs.activation == "relu" else F.gelu
        self.projector = nn.Linear(configs.pred_len, configs.pred_len, bias=True)
        self.mask_generator = Mahalanobis_mask_freq(configs.seq_len)
        print(torch.cuda.is_available())

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        _, _, N = x_enc.shape  # B L N

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # B L N -> B N E
        if x_mark_enc is not None:
            changed_input = rearrange(torch.cat((x_enc, x_mark_enc),dim=-1), 'b l n -> b n l')
        else:
            changed_input = rearrange(x_enc, 'b l n -> b n l')
        channel_mask = self.mask_generator(changed_input)
        #print('channel mask shape',channel_mask.shape)
        enc_out, attns = self.encoder(enc_out,attn_mask=channel_mask)
        
        if hasattr(enc_out, "last_hidden_state"):
            enc_out = enc_out.last_hidden_state
        
        enc_out = enc_out.permute(0, 2, 1)[:, :, :N].permute(0, 2, 1) 

        enc_out = self.dropout(self.activation(self.conv1(enc_out.transpose(-1, 1))))
        enc_out = self.dropout(self.conv2(enc_out).transpose(-1, 1))

        dec_out = self.projector(enc_out).permute(0, 2, 1)#[:, :, :N]  # B N E -> B N S -> B S N
        
        if self.use_norm:
            # De-Normalization
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
