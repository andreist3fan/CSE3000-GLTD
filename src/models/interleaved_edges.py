# import transformer
import torch.nn as nn
import torch.nn.functional as F
import torch
import wandb
from src.models.mpnn import GnnHelper
from src.models.megagnn import MEGAGnnHelper
from torch_geometric.nn import Linear
from src.util import unpack_dict_ns
from src.models.pos_enc.pearl import get_PEARL_wrapper


class Interleaved_Edges(torch.nn.Module):
    def __init__(
        self,
        num_features,
        n_classes=2,
        n_hidden=20,
        n_hidden_edge=20,
        edge_dim=None,
        final_dropout=0.5,
        deg=None,
        config=None,
        index_=None,
    ):
        """
        Instantiates the Interleaved model.
        
        The interleaved model is a combination of two GNNs and two transformers.
        The flow of the model is as follows:
        Node & Edge Embedding -> GNN1  -> | -> Transformer1   | ->     GNN2     -> MLP

        Args:
        - num_features (int): Number of features in the input graph.
        - n_classes (int): Number of classes in the output graph.
        - n_hidden (int): Number of hidden units in the model.
        - n_hidden_edge (int): Number of hidden units for edge features.
        - edge_dim (int): Number of features in the edge embedding.
        - final_dropout (float): Dropout rate for the final layer.
        - deg (int): Degree of the graph.
        - config (dict): Configuration dictionary containing the model parameters.
        
        """
        super().__init__()
        # print(config)
        self.config = config
        self.n_hidden = n_hidden
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden_edge)

        if config.use_pe:
            pecpy = unpack_dict_ns(config, -1)
            self.posenc = get_PEARL_wrapper(pecpy)

        fcpy = unpack_dict_ns(config, 0)
        # print(fcpy)
        self.gnn1 = None
        self.mega = False

        if fcpy.model.startswith("mega"):
            self.mega = True
            self.gnn1 = MEGAGnnHelper(
                num_gnn_layers=fcpy.n_gnn_layers,
                n_hidden=fcpy.n_hidden,
                edge_updates=config.emlps,
                final_dropout=fcpy.final_dropout,
                deg=deg,
                index_=index_,
                args=fcpy,
            )
        else:
            self.gnn1 = GnnHelper(
                num_gnn_layers=fcpy.n_gnn_layers,
                n_hidden=fcpy.n_hidden,
                edge_updates=config.emlps,
                final_dropout=fcpy.final_dropout,
                deg=deg,
                config=fcpy,
            )

        scpy = unpack_dict_ns(config, 1)

        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=scpy.n_hidden,
                nhead=scpy.no_heads,
                dim_feedforward=2 * scpy.n_hidden,
                dropout=scpy.dropout,
                activation="relu",
                batch_first=True,
            ),
            num_layers=scpy.n_layers,
        )

        tcpy = unpack_dict_ns(config, 2)
        self.gnn2 = None
        if tcpy.model.startswith("mega"):
            self.gnn2 = MEGAGnnHelper(
                num_gnn_layers=tcpy.n_gnn_layers,
                n_hidden=tcpy.n_hidden,
                edge_updates=config.emlps,
                final_dropout=tcpy.final_dropout,
                deg=deg,
                args=tcpy,
                index_=index_,
            )
        else:
            self.gnn2 = GnnHelper(
                num_gnn_layers=tcpy.n_gnn_layers,
                n_hidden=tcpy.n_hidden,
                edge_updates=config.emlps,
                final_dropout=tcpy.final_dropout,
                deg=deg,
                config=tcpy,
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

    def forward(self, data):
        # Initial Embedding Layers
        
        x = self.node_emb(data.x)
        edge_attr = self.edge_emb(data.edge_attr)

        simp_edge_batch = None
        if self.mega:
            simp_edge_batch = data.simp_edge_batch
        
        # Positional Encoding
        if self.config.use_pe:
            data.x = x
            data.edge_attr = edge_attr
            data = self.posenc(data)

            x = data.x
            edge_attr = data.edge_attr

        
        # First GNN Layer
        if self.mega:
            x, edge_attr = self.gnn1(
                x, data.edge_index, edge_attr, simp_edge_batch=simp_edge_batch
            )
        else:
            x, edge_attr = self.gnn1(x, data.edge_index, edge_attr)

        # Transformer Layer
        edge_attr = edge_attr.unsqueeze(0)
        edge_attr = self.transformer(edge_attr)
        edge_attr = edge_attr.squeeze(0)

        # Second GNN Layer
        if self.mega:
            x, edge_attr = self.gnn2(
                x, data.edge_index, edge_attr, simp_edge_batch=simp_edge_batch
            )
        else:
            x, edge_attr = self.gnn2(x, data.edge_index, edge_attr)

        # Prediction Head
        x = x[data.edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        out = self.mlp(x)

        return out
