import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, PNAConv
import torch.nn.functional as F
import torch
import wandb
from src.util import get_pearl_config
from src.models.pos_enc.pearl import get_PEARL_wrapper


class MPNN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_gnn_layers,
        n_classes=2,
        n_hidden=100,
        edge_updates=False,
        edge_dim=None,
        final_dropout=0.5,
        deg=None,
        config=None,
    ):
        super().__init__()
        self.config = config
        self.n_hidden = n_hidden
        self.final_dropout = final_dropout

        self.posenc = None
        if config.use_pe:
            pecpy = get_pearl_config(config, config.model)
            self.posenc = get_PEARL_wrapper(pecpy)

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.gnn = GnnHelper(
            num_gnn_layers=num_gnn_layers,
            n_hidden=n_hidden,
            edge_updates=edge_updates,
            final_dropout=final_dropout,
            deg=deg,
            config=config,
        )

        self.mlp = nn.Sequential(
            Linear(n_hidden * 3, 50),
            nn.ReLU(),
            nn.Dropout(self.final_dropout),
            Linear(50, 25),
            nn.ReLU(),
            nn.Dropout(self.final_dropout),
            Linear(25, n_classes),
        )

    def forward(self, data, emb=False):
        # Initial Embedding Layers
        x = self.node_emb(data.x)
        edge_attr = self.edge_emb(data.edge_attr)

        # Positional Encoding
        if self.config.use_pe:
            data.x = x
            data.edge_attr = edge_attr
            data = self.posenc(data)

            x = data.x
            edge_attr = data.edge_attr

        # Message Passing Layers
        x, edge_attr = self.gnn(x, data.edge_index, edge_attr)

        if emb:
            return x, edge_attr
        # Prediction Head
        x = x[data.edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        out = self.mlp(x)

        return out


class GnnHelper(torch.nn.Module):
    def __init__(
        self,
        num_gnn_layers,
        n_hidden=100,
        edge_updates=False,
        final_dropout=0.5,
        deg=None,
        config=None,
    ):
        super().__init__()

        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.final_dropout = final_dropout
        self.config = config
        self.edge_updates = edge_updates
        # print("config mode", config.model)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            if config.model == "gin":
                conv = GINEConv(
                    nn.Sequential(
                        nn.Linear(self.n_hidden, self.n_hidden),
                        nn.ReLU(),
                        nn.Linear(self.n_hidden, self.n_hidden),
                    ),
                    edge_dim=self.n_hidden,
                )
            elif config.model == "pna":
                aggregators = ["mean", "min", "max", "std"]
                scalers = ["identity", "amplification", "attenuation"]
                conv = PNAConv(
                    in_channels=n_hidden,
                    out_channels=n_hidden,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg,
                    edge_dim=n_hidden,
                    towers=5,
                    pre_layers=1,
                    post_layers=1,
                    divide_input=False,
                )

            if self.edge_updates:
                self.emlps.append(
                    nn.Sequential(
                        nn.Linear(3 * self.n_hidden, self.n_hidden),
                        nn.ReLU(),
                        nn.Linear(self.n_hidden, self.n_hidden),
                    )
                )

            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        for i in range(self.num_gnn_layers):
            n_new = self.convs[i](x, edge_index, edge_attr)
            x = (x + F.relu(self.batch_norms[i](n_new))) / 2
            if self.edge_updates:
                edge_attr = (
                    edge_attr
                    + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
                )

        return x, edge_attr
