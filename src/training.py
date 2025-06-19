import torch
import tqdm
from typing import Any, Tuple

from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import summary
from torch_geometric.utils import degree
import logging
import numpy as np
import sklearn.metrics

import copy
from torch.optim import Optimizer
from torch.nn import Module

from .util import add_arange_ids, save_model
from .models.megagnn import MultiMPNN
from .models.mpnn import MPNN
from .models.interleaved_edges import Interleaved_Edges
from .models.fullfusion import Full_Fusion

from .optimizers.cosine_warmup import cosine_with_warmup_scheduler


from tqdm import tqdm
import wandb
from types import SimpleNamespace


def get_loaders(
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, config
):

    tr_loader = LinkNeighborLoader(
        tr_data,
        num_neighbors=config.num_neighs,
        batch_size=config.batch_size,
        shuffle=True,
        transform=transform,
    )
    val_loader = LinkNeighborLoader(
        val_data,
        num_neighbors=config.num_neighs,
        edge_label_index=val_data.edge_index[:, val_inds],
        edge_label=val_data.y[val_inds],
        batch_size=config.batch_size,
        shuffle=False,
        transform=transform,
    )
    te_loader = LinkNeighborLoader(
        te_data,
        num_neighbors=config.num_neighs,
        edge_label_index=te_data.edge_index[:, te_inds],
        edge_label=te_data.y[te_inds],
        batch_size=config.batch_size,
        shuffle=False,
        transform=transform,
    )

    return tr_loader, val_loader, te_loader


def compute_binary_metrics(preds: np.array, labels: np.array):
    """
    Computes metrics based on raw/ normalized model predictions
    :param preds: Raw (or normalized) predictions (can vary threshold here if raw scores are provided)
    :param labels: Binary target labels
    :return: Accuracy, illicit precision/ recall/ F1, and ROC AUC scores
    """

    probs = preds[:, 1]
    preds = preds.argmax(axis=-1)

    precisions, recalls, _ = sklearn.metrics.precision_recall_curve(
        labels, probs
    )  # probs: probabilities for the positive class
    f1 = sklearn.metrics.f1_score(labels, preds, zero_division=0)
    auc = sklearn.metrics.auc(recalls, precisions)

    precision = sklearn.metrics.precision_score(labels, preds, zero_division=0)
    recall = sklearn.metrics.recall_score(labels, preds, zero_division=0)

    return f1, auc, precision, recall


def train_epoch(
    loader: Any,
    model: Module,
    optimizer: Optimizer,
    loss_fn: Module,
    tr_inds: torch.Tensor,
    config: SimpleNamespace,
    scheduler: Any = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Trains the model for one epoch.

    Args:
        loader: Data loader for training
        model: Model to train
        optimizer: Optimizer for training
        loss_fn: Loss function
        tr_inds: Training indices
        config: Configuration object

    Returns:
        Tuple containing:
            - Average loss for the epoch
            - Model predictions
            - Ground truth labels
    """
    model.train()
    total_loss = total_examples = 0
    preds = []
    ground_truths = []

    device = config.device

    batch_accum = config.batch_accum
    batch_count = 0

    optimizer.zero_grad()

    for batch in tqdm(loader, total=len(loader), desc="Training"):

        # Select the seed edges from which the batch was created
        inds = tr_inds.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        # Remove the unique edge id from the edge features, as it's no longer needed
        batch.edge_attr = batch.edge_attr[:, 1:]

        batch.to(device)

        out = model(batch)

        pred = out[mask]
        ground_truth = batch.y[mask]
        loss = loss_fn(pred, ground_truth)

        loss.backward()
        batch_count += 1

        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()

        preds.append(pred.detach().cpu())
        ground_truths.append(ground_truth.detach().cpu())

        if batch_accum == 0 or batch_count == batch_accum:
            # print("Batch accum")
            if config.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()
            batch_count = 0

    if batch_count > 0:
        optimizer.step()
        optimizer.zero_grad()

    if scheduler is not None:
        scheduler.step()

    pred = torch.cat(preds, dim=0).numpy()
    ground_truth = torch.cat(ground_truths, dim=0).numpy()

    return total_loss / total_examples, pred, ground_truth


@torch.no_grad()
def eval_epoch(
    loader: Any, inds: torch.Tensor, model: Module, device: torch.device
) -> Tuple[float, float, float, float]:
    """Evaluates the model on the given loader.

    Args:
        loader: Data loader for evaluation
        inds: Evaluation indices
        model: Model to evaluate
        device: Device to evaluate on

    Returns:
        Tuple containing evaluation metrics:
            - F1 score
            - AUC score
            - Precision
            - Recall
    """
    model.eval()
    assert not model.training, "Test error: Model is not in evaluation mode"

    preds = []
    ground_truths = []
    for batch in loader:
        # select the seed edges from which the batch was created
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        # remove the unique edge id from the edge features, as it's no longer needed
        batch.edge_attr = batch.edge_attr[:, 1:]

        with torch.no_grad():
            batch.to(device)

            out = model(batch)
            pred = out[mask]
            preds.append(pred.detach().cpu())
            ground_truths.append(batch.y[mask].detach().cpu())

    pred = torch.cat(preds, dim=0).numpy()
    ground_truth = torch.cat(ground_truths, dim=0).numpy()

    # Compute Metrics
    f1, auc, precision, recall = compute_binary_metrics(pred, ground_truth)
    return f1, auc, precision, recall


def train(
    tr_loader: Any,
    val_loader: Any,
    te_loader: Any,
    tr_inds: torch.Tensor,
    val_inds: torch.Tensor,
    te_inds: torch.Tensor,
    model: Module,
    optimizer: Optimizer,
    loss_fn: Module,
    config: Any,
    scheduler: Any = None,
) -> Module:
    """Main training loop with validation and model selection.

    Args:
        tr_loader: Training data loader
        val_loader: Validation data loader
        te_loader: Test data loader
        tr_inds: Training indices
        val_inds: Validation indices
        te_inds: Test indices
        model: Model to train
        optimizer: Optimizer for training
        loss_fn: Loss function
        config: Training configuration

    Returns:
        Module: Best model
    """
    best_val_f1 = 0
    best_state_dict = None

    for epoch in range(config.epochs):
        logging.info(f"****** EPOCH {epoch} ******")

        # Training phase
        total_loss, pred, ground_truth = train_epoch(
            tr_loader, model, optimizer, loss_fn, tr_inds, config, scheduler
        )

        # Compute training metrics
        f1, auc, precision, recall = compute_binary_metrics(pred, ground_truth)

        # Log training metrics
        logging.info(
            {
                "Train": {
                    "F1": f"{f1:.4f}",
                    "Precision": f"{precision:.4f}",
                    "Recall": f"{recall:.4f}",
                    "PR-AUC": f"{auc:.4f}",
                }
            }
        )

        # Evaluation phase
        val_f1, val_auc, val_precision, val_recall = eval_epoch(
            val_loader, val_inds, model, config.device
        )
        te_f1, te_auc, te_precision, te_recall = eval_epoch(
            te_loader, te_inds, model, config.device
        )

        # Log validation metrics
        logging.info(
            {
                "Val": {
                    "F1": f"{val_f1:.4f}",
                    "Precision": f"{val_precision:.4f}",
                    "Recall": f"{val_recall:.4f}",
                    "PR-AUC": f"{val_auc:.4f}",
                }
            }
        )

        # Log test metrics
        logging.info(
            {
                "Test": {
                    "F1": f"{te_f1:.4f}",
                    "Precision": f"{te_precision:.4f}",
                    "Recall": f"{te_recall:.4f}",
                    "PR-AUC": f"{te_auc:.4f}",
                }
            }
        )

        # Log loss
        logging.info(f"Loss: {total_loss}")
        logging.info(f"LR: {scheduler.get_last_lr()[0] if scheduler else -1.0}")

        #

        if not config.testing:
            wandb.log(
                {
                    "Train": {
                        "F1": round(f1, 4),
                        "Precision": round(precision, 4),
                        "Recall": round(recall, 4),
                        "PR-AUC": round(auc, 4),
                    }
                },
                step=epoch,
            )
            wandb.log(
                {
                    "Val": {
                        "F1": round(val_f1, 4),
                        "Precision": round(val_precision, 4),
                        "Recall": round(val_recall, 4),
                        "PR-AUC": round(val_auc, 4),
                    }
                },
                step=epoch,
            )
            wandb.log(
                {
                    "Test": {
                        "F1": round(te_f1, 4),
                        "Precision": round(te_precision, 4),
                        "Recall": round(te_recall, 4),
                        "PR-AUC": round(te_auc, 4),
                    }
                },
                step=epoch,
            )
            wandb.log({"Loss": total_loss}, step=epoch)
            wandb.log(
                {"LR": scheduler.get_last_lr()[0] if scheduler else -1.0}, step=epoch
            )

        # Model selection based on validation F1 score
        if epoch == 0:
            logging.info({"best_test_f1": f"{te_f1:.4f}"})
            best_state_dict = copy.deepcopy(model.state_dict())
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = copy.deepcopy(model.state_dict())
            logging.info({"best_test_f1": f"{te_f1:.4f}"})
            if not config.testing:
                wandb.log({"best_test_f1": round(te_f1, 4)}, step=epoch)

            # Save best model
            if config.save_model:
                save_model(model, optimizer, epoch, config)

    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if not config.testing:
        wandb.finish()
    return model


def get_model(sample_batch, config):
    """Creates a model instance based on the provided configuration.

    Args:
        sample_batch: Sample batch for model initialization
        config: Model configuration

    Returns:
        Module: Initialized model
    """
    n_feats = sample_batch.x.shape[1]
    e_dim = sample_batch.edge_attr.shape[1]
    
    index_ = sample_batch.simp_edge_batch 

    d = degree(
        sample_batch.edge_index[1], num_nodes=sample_batch.num_nodes, dtype=torch.long
    )
    deg = torch.bincount(d, minlength=1)
    model = None
    if config.model == "interleaved":

        model = Interleaved_Edges(
            num_features=n_feats,
            n_classes=2,
            n_hidden=round(config.n_hidden),
            edge_dim=e_dim,
            final_dropout=config.final_dropout,
            deg=deg,
            config=config,
            index_=index_,
        )
    elif config.model == "fusion":
        model = Full_Fusion(
            num_features=n_feats,
            n_classes=2,
            n_hidden=round(config.n_hidden),
            edge_dim=e_dim,
            final_dropout=config.final_dropout,
            deg=deg,
            config=config,
            index_=index_,
        )

    elif config.model.startswith("mega"):
        
        model = MultiMPNN(num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2, 
                      n_hidden=round(config.n_hidden), edge_updates=config.emlps, edge_dim=e_dim, 
                      final_dropout=config.final_dropout, index_=index_, deg=deg, args=config)

    else:
        model = MPNN(
            num_features=n_feats,
            num_gnn_layers=config.n_gnn_layers,
            n_classes=2,
            n_hidden=round(config.n_hidden),
            edge_updates=config.emlps,
            edge_dim=e_dim,
            final_dropout=config.final_dropout,
            deg=deg,
            config=config,
        )

    return model


def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, config):
    """Main entry point for training the GNN model.

    Args:
        tr_data: Training data
        val_data: Validation data
        te_data: Test data
        tr_inds: Training indices
        val_inds: Validation indices
        te_inds: Test indices

    Returns:
        Module: Trained model
    """
    device = config.device
    # Add unique IDs to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])

    # Get data loaders
    tr_loader, val_loader, te_loader = get_loaders(
        tr_data,
        val_data,
        te_data,
        tr_inds,
        val_inds,
        te_inds,
        transform=None,
        config=config,
    )

    # Get a sample batch and initialize the model
    sample_batch = next(iter(tr_loader))
    sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
    model = get_model(sample_batch, config)
    model.to(device)
    # Move sample batch to device and log model summary
    sample_batch.to(device)
    logging.info(summary(model, sample_batch))

    # Define loss function and Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # print(config)
    if config.optimizer is not None:
        if config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.lr, weight_decay=1e-5
            )

    scheduler = None
    if config.scheduler == "cosine":
        scheduler = cosine_with_warmup_scheduler(
            optimizer=optimizer,
            num_warmup_epochs=config.warmup,
            max_epoch=int(config.epochs * config.false_epoch_mult),
        )

    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device)
    )

    # Train the model
    model = train(
        tr_loader,
        val_loader,
        te_loader,
        tr_inds,
        val_inds,
        te_inds,
        model,
        optimizer,
        loss_fn,
        config,
        scheduler=scheduler,
    )

    return model
