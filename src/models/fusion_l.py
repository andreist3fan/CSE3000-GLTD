# import transformer
import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb
from src.models.mpnn import GnnHelper
from torch_geometric.nn import Linear
from src.util import unpack_dict_ns


class Fusion_Layer(torch.nn.Module):
    def __init__(self, input_dim_a=20, input_dim_b=20, n_hidden=100, config=None):
        super().__init__()
        # print(config)
        self.config = config
        self.n_hidden = n_hidden

        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b

        if config.model == "fmlp":
            activation = None
            if config.activation == "relu":
                activation = nn.ReLU()
            elif config.activation == "gelu":
                activation = nn.GELU()

            self.fusion = nn.Sequential(
                Linear(input_dim_a + input_dim_b, config.n_hidden),
                activation,
                nn.Dropout(config.dropout),
                Linear(config.n_hidden, input_dim_a + input_dim_b),
            )
        elif config.model == "gmu":
            self.fusion = GmuHelper(
                input_dim_a=input_dim_a,
                input_dim_b=input_dim_b,
                hidden_dim=n_hidden,
                config=config,
            )
            # raise NotImplementedError("GMU fusion not implemented")

    def forward(self, data):
        # Initial Embedding Layers
        # print("self.config.model", self.config.model)
        if self.config.model == "fmlp":
            out = self.fusion(data)

            out1 = out[:, : self.input_dim_a]
            out2 = out[:, self.input_dim_a :]

            return out1, out2
        elif self.config.model == "gmu":
            data_a = data[:, : self.input_dim_a]
            data_b = data[:, self.input_dim_a :]

            out = self.fusion(data_a, data_b)

            return out, out

        raise NotImplementedError("Fusion layer not implemented")


class GmuHelper(torch.nn.Module):
    """
    An implementation of the GMU fusion layer.
    The GMU fusion layer is a neural network layer that fuses two input tensors
    using a gating mechanism.

    This implementation is based on the two-input GMU fusion layer described in the paper
    "Gated Multimodal Units for Information Fusion" by Arevalo et al. (2017).
    """

    def __init__(self, input_dim_a=20, input_dim_b=20, hidden_dim=20, config=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b
        self.config = config

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim_a, hidden_dim), nn.Tanh(), nn.Dropout(config.dropout)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(input_dim_b, hidden_dim), nn.Tanh(), nn.Dropout(config.dropout)
        )

        self.z_gate = nn.Linear(input_dim_a + input_dim_b, self.hidden_dim)

    def forward(self, edge_attr1, edge_attr2):
        """
        edge_attr1: Tensor of shape [num_edges, input_dim_a]
        edge_attr2: Tensor of shape [num_edges, input_dim_b]
        """
        h1 = self.fc1(edge_attr1)
        h2 = self.fc2(edge_attr2)

        z_input = torch.cat([edge_attr1, edge_attr2], dim=-1)
        z = torch.sigmoid(self.z_gate(z_input))

        fused = z * h1 + (1 - z) * h2
        return fused
