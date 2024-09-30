"""
EEG Transformer.
"""

import math
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
from torchvision.ops.misc import MLP


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None,
                         dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i + 1}.{type}"
                    new_key = f"{prefix}{3 * i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads,
                                                    dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3,
                      f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
            self,
            seq_length: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3,
                      f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class EEGTransformer(nn.Module):
    """EEG Transformer."""

    def __init__(
            self,
            channels_size: int,
            temporal_size: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            num_classes: int = 1654,
            representation_size: Optional[int] = None,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.channels_size = channels_size
        self.temporal_size = temporal_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        self.conv_proj = nn.Conv1d(
            in_channels=self.channels_size, out_channels=hidden_dim, kernel_size=1
        )

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length = self.temporal_size + 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )

        if representation_size is None:
            self.head = nn.Linear(hidden_dim, num_classes)
        else:
            self.representation = nn.Sequential(
                nn.Linear(hidden_dim, representation_size),
                nn.LayerNorm(representation_size)
            )
            if num_classes > 0:
                self.head = nn.Linear(representation_size, num_classes)

        if isinstance(self.conv_proj, nn.Conv1d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)

        if representation_size is not None and isinstance(self.representation[0], nn.Linear):
            fan_in = self.representation[0].in_features
            nn.init.trunc_normal_(self.representation[0].weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.representation[0].bias)

        if num_classes > 0 and isinstance(self.head, nn.Linear):
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, t = x.shape
        torch._assert(c == self.channels_size,
                      f"Wrong number of channels! Expected {self.channels_size} but got {c}!")
        torch._assert(t == self.temporal_size,
                      f"Wrong temporal size! Expected {self.temporal_size} but got {t}!")

        # (n, c, t) -> (n, hidden_dim, t)
        x = self.conv_proj(x)

        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        # (n, hidden_dim, t) -> (n, t, hidden_dim)
        x = x.permute(0, 2, 1)

        return x

    def save_params(self):
        return {
            "channels_size": self.channels_size,
            "temporal_size": self.temporal_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "representation_size": self.representation_size,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout
        }

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        cls_token = x[:, 0]

        if self.representation_size is not None:
            cls_token = self.representation(cls_token)

        x = self.head(cls_token) if self.num_classes > 0 else cls_token

        return x, cls_token
