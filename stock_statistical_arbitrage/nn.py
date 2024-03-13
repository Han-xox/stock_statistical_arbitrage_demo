import math
from typing import Any, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

MAX_EMBEDDINGS = 4096


def _ts_zscore(x: Tensor):
    """normalize [batch_size, time_step, channel_dim] shape data across time dimension"""

    mu = x.mean(1, keepdim=True).detach()
    x = x - mu
    sigma = torch.sqrt(
        torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
    ).detach()
    x = x / sigma

    return x, mu, sigma


class EmbeddingMachine(nn.Module):
    def __init__(self, embedding_sizes: List[int], embedding_dims: List[int]) -> None:
        super().__init__()

        self.n = len(embedding_sizes)
        self.embedding_layers = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=size, embedding_dim=dim)
                for size, dim in zip(embedding_sizes, embedding_dims)
            ]
        )

    def forward(self, x: torch.LongTensor):
        # x shape -> [B, L]
        # x[:, i] -> [B,]
        # embedding(x[:, i]) -> [B, d]
        x = [self.embedding_layers[i](x[:, i]) for i in range(0, self.n)]
        x = torch.concat(x, dim=1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len: int = MAX_EMBEDDINGS):
        super().__init__()

        # 1. create position encoding (pe) memory
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # 2. compute position encoding
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # 3. save position encoding
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Parameters
        ----------
        x: shape [batch_size, time_step, model_dim]
        """
        x = x + self.pe[:, : x.size(1)]

        return x


class MLP(nn.Module):
    def __init__(
        self,
        layer_dims: List[int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers = []
        num_layers = len(layer_dims) - 1
        for i in range(0, num_layers):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i != (num_layers - 1):
                layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])
        x = self.layers(x)
        x = x.reshape(input_shape[:-1] + (x.shape[-1],))
        return x


class TransformerEncoder(nn.Module):
    """input shape is [B, T, C]"""

    def __init__(
        self,
        channel_dim: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Embedding layer
        self.embedding = nn.Linear(channel_dim, model_dim)

        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=model_dim)

        # Transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True,
        )

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)

        return x


class RNN(nn.Module):
    """input shape is [B, T, C]"""

    RNN_CLS = {"LSTM": nn.LSTM, "GRU": nn.GRU}

    def __init__(
        self,
        cell_type: str,
        channel_dim: int,
        model_dim: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert cell_type in ["LSTM", "GRU"]

        self.channel_dim = channel_dim
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        rnn_cls = self.RNN_CLS[cell_type]

        self.rnn = rnn_cls(
            input_size=self.channel_dim,
            hidden_size=self.model_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
        )

    def forward(self, x):
        """RNN outputs two parts:
        1) hidden from all cells [B, T, model_dim * bidirectional]
        2) states, it's different betweens cell types
        """

        # [B, T, C] -> [B, T, model_dim * bidirectional]
        x, _ = self.rnn(x)

        return x
