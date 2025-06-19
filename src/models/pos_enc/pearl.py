"""
This file contains an adaptation of the PEARLPositionalEncoder described in <https://arxiv.org/abs/2502.01122>.

The following repositories serve as references for the implementation:
https://github.com/ehejin/Pearl-PE/blob/main/PEARL/
https://github.com/Graph-COM/SPE/blob/master/src/stable_expressive_pe.py
https://github.com/ivam-he/BernNet
"""


import math
import torch
import torch.nn as nn
from torch_geometric.utils import subgraph, to_scipy_sparse_matrix

from typing import Any, Callable, List, Optional
from torch_geometric.utils import add_self_loops, get_laplacian, remove_self_loops
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import get_laplacian, to_dense_adj
import torch

from src.models.pos_enc.mlp import MLP, create_mlp
from src.models.pos_enc.gin import GetSampleAggregator



class SwiGLU(nn.Module):
    def __init__(self, input_dim):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        swish_part = self.fc1(x) * torch.sigmoid(self.fc1(x))
        gate = torch.sigmoid(self.fc2(x))  # Sigmoid
        return swish_part * gate





def filter(S, W, k):
    """
    Basic graph filtering used in all experiments of our paper.
    """
    # S is laplacian and W is NxN e or NxM x_m
    out = W
    w_list = []
    w_list.append(out.unsqueeze(-1))
    for i in range(k - 1):
        out = S @ out  # NxN or NxM
        w_list.append(out.unsqueeze(-1))
    return torch.cat(w_list, dim=-1)  # NxMxK




def bern_filter(S, W, k):
    
    out = W
    w_list = []
    w_list.append(out.unsqueeze(-1))
    for i in range(1, k):
        L = (
            (1 / (2**k))
            * math.comb(k, i)
            * torch.linalg.matrix_power(
                (2 * (torch.eye(S.shape[0]).to(S.device)) - S), k
            )
            @ S
        )
        out = L @ W  # NxN or NxM
        w_list.append(out.unsqueeze(-1))
    return torch.cat(w_list, dim=-1)


class PEARLPositionalEncoder(nn.Module):
    def __init__(
        self,
        sample_aggr: nn.Module,
        basis,
        k=16,
        mlp_nlayers=1,
        mlp_hid=16,
        pearl_act="relu",
        mlp_out=16,
    ) -> None:
        super().__init__()
        self.mlp_nlayers = mlp_nlayers
        if mlp_nlayers > 0:
            if mlp_nlayers == 1:
                assert mlp_hid == mlp_out
            print(
                f"PEARLPositionalEncoder: Using MLP with {mlp_nlayers} layers, hidden size {mlp_hid}, output size {mlp_out}, activation {pearl_act}")
            self.mlp_nlayers = mlp_nlayers
            self.layers = nn.ModuleList(
                [
                    nn.Linear(
                        k if i == 0 else mlp_hid,
                        mlp_hid if i < mlp_nlayers - 1 else mlp_out,
                        bias=True,
                    )
                    for i in range(mlp_nlayers)
                ]
            )
            self.norms = nn.ModuleList(
                [
                    nn.BatchNorm1d(
                        mlp_hid if i < mlp_nlayers - 1 else mlp_out,
                        track_running_stats=True,
                    )
                    for i in range(mlp_nlayers)
                ]
            )
        if pearl_act == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif pearl_act == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = SwiGLU(
                mlp_hid
            )  ## edit if you want more than 1 mlp layers!!
        self.sample_aggr = sample_aggr
        self.k = k
        self.basis = basis

        print()

    def forward(
        self, Lap, W, edge_index: torch.Tensor, batch: torch.Tensor, final=True
    ) -> torch.Tensor:
        """
        :param Lap: Laplacian
        :param W: B*[NxM] or BxNxN
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param batch: Batch index vector. [N_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        W_list = []
        device = edge_index.device
        # for loop N times for each Nx1 e
        if isinstance(W[0], int):
            # split into N*B*[Nx1]
            j = 0
            for lap, w in zip(Lap, W):
                for i in range(w):
                    e_i = torch.zeros(w).to(device)
                    e_i[i] = 1
                    output = filter(
                        lap, e_i, self.k
                    )  # can also use bern_filter(lap, e_i, self.k)
                    W_list.append(output)  # [NxMxK]*B
                if j == 0:
                    out = self.sample_aggr(W_list, edge_index, self.basis)
                else:
                    out += self.sample_aggr(W_list, edge_index, self.basis)
                j += 1
            return out
        else:
            for lap, w in zip(Lap, W):
                output = filter(lap, w, self.k)  # output [NxMxK]
                if self.mlp_nlayers > 0:
                    for layer, bn in zip(self.layers, self.norms):
                        output = output.transpose(0, 1)
                        output = layer(output)
                        output = bn(output.transpose(1, 2)).transpose(1, 2)
                        output = self.activation(output)
                        output = output.transpose(0, 1)
                W_list.append(output)  # [NxMxK]*B
            return self.sample_aggr(
                W_list, edge_index, self.basis, final=final
            )  # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.sample_aggr.out_dims



def get_PEARL_wrapper(config):
    sample_aggr = GetSampleAggregator(config, create_mlp, config, config.device)
    pearl_encoder = PEARLPositionalEncoder(
        sample_aggr=sample_aggr,
        k=config.pearl_k,
        pearl_act=config.pearl_act,
        basis=config.use_identity_basis,
    )
    pearl_wrapper = PEARLWrapper(encoder=pearl_encoder, out_dim=config.n_hidden)

    return pearl_wrapper


class PEARLWrapper(nn.Module):
    def __init__(self, encoder, k=2, use_identity_basis=False, out_dim=None):
        super().__init__()
        self.encoder = encoder
        self.k = k
        self.use_identity_basis = use_identity_basis
        self.out_dim = out_dim
        self.mlp = None

    def forward(self, data):
        def _get_lap(data):
            # Method addapted from the PEARL repository
            n = data.num_nodes
            L_edge_index, L_values = get_laplacian(
                data.edge_index, normalization="sym", num_nodes=n
            )  # [2, X], [X]
            L = to_dense_adj(L_edge_index, edge_attr=L_values, max_num_nodes=n).squeeze(
                dim=0
            )
            return L

        x, edge_index = data.x, data.edge_index
        num_nodes = data.num_nodes
        device = x.device

        # _get_lap, can also be done as a transform and used within LinkNeighborLoader (as in PEARL examples - get_lap).
        # Tried locally, this way works faster.
        L = _get_lap(data)

        # Build basis W
        if self.use_identity_basis:
            W = torch.eye(num_nodes, device=device)
        else:
            W = torch.randn(num_nodes, self.k, device=device)

        # Compute positional encodings
        # only one graph => batch_vec only zeroes, meaning they all belong to the same graph
        batch_vec = torch.zeros(num_nodes, dtype=torch.long, device=device)
        pos_enc = self.encoder([L], [W], edge_index, batch_vec)

        combined = torch.cat([x, pos_enc.to(device)], dim=-1)

        if self.out_dim is not None and self.mlp is None:
            in_dim = combined.shape[-1]
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, self.out_dim),
                nn.ReLU(),
            ).to(device)

        if self.mlp is not None:
            data.x = self.mlp(combined)
        else:
            data.x = combined
        return data
