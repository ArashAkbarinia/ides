"""
ATM Network https://github.com/dongyangli-del/EEG_Image_decode/tree/main
"""

import numpy as np
import math

import torch
import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2 + 1])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + pe
        return x


class EEGAttention(nn.Module):
    def __init__(self, channel, d_model, nhead):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.permute(1, 2, 0)  # Change shape back to [batch_size, channel, time_length]


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)
        # print("x", x.shape)
        x = self.tsconv(x)
        # print("tsconv", x.shape)
        x = self.projection(x)
        # print("projection", x.shape)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class ATM_S_reconstruction_scale(nn.Module):
    def __init__(self, channels_size=63, temporal_size=250, num_subjects=1, num_features=64,
                 num_latents=1024, num_blocks=1, representation_size=1024, **kwargs):
        super(ATM_S_reconstruction_scale, self).__init__()
        self.channels_size = channels_size
        self.temporal_size = temporal_size
        self.representation_size = representation_size
        self.attention_model = EEGAttention(channels_size, channels_size, nhead=1)
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(temporal_size, temporal_size) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        if temporal_size == 250:
            embedding_dim = 1440
        elif temporal_size == 701:
            embedding_dim = 5040
        else:
            embedding_dim = 840
        self.proj_eeg = Proj_eeg(embedding_dim=embedding_dim, proj_dim=representation_size)

    def save_params(self):
        return {
            "channels_size": self.channels_size,
            "temporal_size": self.temporal_size,
            # "num_layers": self.num_layers,
            # "num_heads": self.num_heads,
            # "hidden_dim": self.hidden_dim,
            "representation_size": self.representation_size,
            # "mlp_dim": self.mlp_dim,
            # "dropout": self.dropout
        }

    def forward(self, x):
        #print(x.shape)
        x = self.attention_model(x)
        #print(f'After attention shape: {x.shape}')
        #print(x.shape)
        x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)
        # print(eeg_embedding.shape)

        out = self.proj_eeg(eeg_embedding)
        return out, out
