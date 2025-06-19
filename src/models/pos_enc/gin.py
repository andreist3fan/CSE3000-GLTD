import math
import torch
import torch.nn as nn
from torch_geometric.utils import subgraph, to_scipy_sparse_matrix

from typing import Any, Callable, List, Optional
from torch_geometric.utils import add_self_loops, get_laplacian, remove_self_loops
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import get_laplacian, to_dense_adj, degree
from src.models.pos_enc.mlp import MLP, create_mlp

# This code is from:
# https://github.com/Graph-COM/SPE/blob/master/src/gin.py


class GIN(nn.Module):
    layers: nn.ModuleList

    def __init__(
        self,
        n_layers: int,
        in_dims: int,
        hidden_dims: int,
        out_dims: int,
        create_mlp: Callable[[int, int], MLP],
        config: Any,
        bn: bool = False,
        residual: bool = False,
        laplacian=None,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.bn = bn
        self.residual = residual
        if bn:
            self.batch_norms = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = GINLayer(create_mlp(in_dims, hidden_dims, config))
            self.layers.append(layer)
            in_dims = hidden_dims
            if bn:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims))
        layer = GINLayer(create_mlp(hidden_dims, out_dims, config))
        self.layers.append(layer)
        self.laplacian = laplacian

    def forward(
        self, X: torch.Tensor, edge_index: torch.Tensor, mask=None
    ) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """
        for i, layer in enumerate(self.layers):
            X0 = X
            X = layer(
                X, edge_index, laplacian=self.laplacian, mask=mask
            )  # [N_sum, ***, D_hid] or [N_sum, ***, D_out]
            if mask is not None:
                X[~mask] = 0
            # batch normalization
            if self.bn and i < len(self.layers) - 1:
                if mask is None:
                    if X.ndim == 3:
                        X = self.batch_norms[i](X.transpose(2, 1)).transpose(2, 1)
                    else:
                        X = self.batch_norms[i](X)
                else:
                    X[mask] = self.batch_norms[i](X[mask])
            if self.residual:
                X = X + X0
        return X  # [N_sum, ***, D_out]

    @property
    def out_dims(self) -> int:
        return self.layers[-1].out_dims


class GINLayer(MessagePassing):
    eps: nn.Parameter
    mlp: MLP

    def __init__(self, mlp: MLP) -> None:
        # Use node_dim=0 because message() output has shape [E_sum, ***, D_in] - https://stackoverflow.com/a/68931962
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)

        self.eps = torch.nn.Parameter(
            data=torch.randn(1), requires_grad=True
        )  # torch.empty(1), requires_grad=True)
        self.mlp = mlp

    def forward(
        self, X: torch.Tensor, edge_index: torch.Tensor, laplacian=False, mask=None
    ) -> torch.Tensor:
        """
        :param X: Node feature matrix. [N_sum, ***, D_in]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Output node feature matrix. [N_sum, ***, D_out]
        """
        # Contains sum(j in N(i)) {message(j -> i)} for each node i.
        if laplacian == "L":
            edge_index, edge_weight = get_laplacian(
                edge_index, normalization="sym", num_nodes=X.size(0)
            )
            laplacian = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze(
                0
            )  # [N_sum, N_sum]
            S = torch.einsum("ij,jkd->ikd", laplacian, X)  # [N_sum, ***, D_in]

            Z = (1 + self.eps) * X  # [N_sum, ***, D_in]
            Z = Z + S  # [N_sum, ***, D_in]
            return self.mlp(Z, mask)
        elif laplacian == "RW":
            adj = to_dense_adj(edge_index).squeeze(0)  # [N_sum, N_sum]
            deg = degree(
                edge_index[0], num_nodes=X.size(0), dtype=torch.float
            )  # [N_sum]
            deg_inv = 1.0 / deg  # Inverse of the degree
            deg_inv[deg_inv == float("inf")] = (
                0  # Handle division by zero for isolated nodes
            )
            deg_inv_diag = torch.diag(deg_inv)  # [N_sum, N_sum]
            random_walk = torch.matmul(adj, deg_inv_diag)  # [N_sum, N_sum]
            S = torch.einsum("ij,jkd->ikd", random_walk, X)  # [N_sum, *, D_in]
            Z = (1 + self.eps) * X  # [N_sum, ***, D_in]
            Z = Z + S  # [N_sum, ***, D_in]
            return self.mlp(Z, mask)  # [N_sum, ***, D_out]

        S = self.propagate(edge_index, X=X)  # [N_sum, *** D_in]

        Z = (1 + self.eps) * X  # [N_sum, ***, D_in]
        Z = Z + S  # [N_sum, ***, D_in]
        return self.mlp(Z, mask)  # [N_sum, ***, D_out]

    def message(self, X_j: torch.Tensor) -> torch.Tensor:
        """
        :param X_j: Features of the edge sources. [E_sum, ***, D_in]
        :return: The messages X_j for each edge (j -> i). [E_sum, ***, D_in]
        """
        return X_j  # [E_sum, ***, D_in]

    @property
    def out_dims(self) -> int:
        return self.mlp.out_dims


class GINSampleAggregator(nn.Module):
    gin: GIN

    def __init__(
        self,
        n_layers: int,
        in_dims: int,
        hidden_dims: int,
        out_dims: int,
        create_mlp: Callable[[int, int], MLP],
        bn: bool,
        config,
    ) -> None:
        super().__init__()
        self.gin = GIN(
            n_layers,
            in_dims,
            hidden_dims,
            out_dims,
            create_mlp,
            config,
            bn,
            laplacian=None,
        )
        self.mlp = create_mlp(out_dims, out_dims, config, use_bias=True)
        self.running_sum = 0

    def forward(
        self,
        W_list: List[torch.Tensor],
        edge_index: torch.Tensor,
        basis,
        mean=False,
        final=True,
    ) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        if not basis:
            W = torch.cat(W_list, dim=0)
            PE = self.gin(W, edge_index)
            if mean:
                PE = (PE).mean(dim=1)  # sum or mean along M dimension
            else:
                PE = (PE).sum(dim=1)
                self.running_sum += PE
            if final:
                PE = self.running_sum
                self.running_sum = 0
            return PE
        else:
            n_max = max(W.size(0) for W in W_list)
            W_pad_list = []  # [N_i, N_max, M] * B
            mask = []  # node masking, [N_i, N_max] * B
            for W in W_list:
                zeros = torch.zeros(
                    W.size(0), n_max - W.size(1), W.size(2), device=W.device
                )
                W_pad = torch.cat([W, zeros], dim=1)  # [N_i, N_max, M]
                W_pad_list.append(W_pad)
                mask.append(
                    (torch.arange(n_max, device=W.device) < W.size(0)).tile(
                        (W.size(0), 1)
                    )
                )  # [N_i, N_max]
            W = torch.cat(W_pad_list, dim=0)  # [N_sum, N_max, M]
            mask = torch.cat(mask, dim=0)  # [N_sum, N_max]
            PE = self.gin(W, edge_index, mask=mask)  # [N_sum, N_max, D_pe]
            PE = (PE * mask.unsqueeze(-1)).sum(dim=1)
            return PE

    @property
    def out_dims(self) -> int:
        return self.gin.out_dims


def GetSampleAggregator(
    cfg: Any, create_mlp: Callable[[int, int], MLP], config, device
):
    return GINSampleAggregator(
        cfg.n_sample_aggr_layers,
        cfg.pearl_mlp_out,
        cfg.sample_aggr_hidden_dims,
        cfg.pe_dims,
        create_mlp,
        cfg.batch_norm,
        config,
    )
