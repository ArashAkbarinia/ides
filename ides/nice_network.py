"""
The network architecture of the NICE paper
https://github.com/eeyhsong/NICE-EEG
"""

import torch
import torch.nn as nn
import torch.nn.init as init

from einops.layers.torch import Rearrange


def _weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b, _, _, _ = x.shape
        x = self.tsconv(x)
        x = self.projection(x)
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
    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class EGGEncoder(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )


class EEGProjector(nn.Sequential):
    def __init__(self, encoder_emb=40, projector_emb=1440, representation_size=768, drop_proj=0.5,
                 channels_size=63, temporal_size=250):
        super().__init__(
            EGGEncoder(encoder_emb),
            nn.Linear(projector_emb, representation_size),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(representation_size, representation_size),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(representation_size),
        )
        self.channels_size = channels_size
        self.temporal_size = temporal_size
        self.representation_size = representation_size

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
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=1)
        x = super().forward(x)
        return x, x
