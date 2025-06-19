import argparse
import numpy as np
import torch
import random
import logging
import os
import sys
import json
import copy
import wandb
from types import SimpleNamespace


def extract_param(parameter_name: str, config, model: str = None) -> float:
    """
    Extract the value of the specified parameter for the given model.

    Args:
    - parameter_name (str): Name of the parameter (e.g., "lr").
    - config: Arguments given to this specific run.

    Returns:
    - float: Value of the specified parameter.
    """

    # Get the architecture name from the config
    # as it is used to retrieve the right model settings

    modL = ""
    if config.model is not None:
        modL = config.model

    file_path = f"./model_settings_{modL}.json"

    # print("-------------------")
    # print(f"Loading model settings from {file_path}")

    with open(file_path, "r") as file:
        data = json.load(file)

    md = None
    if model is None:
        md = config.model
    else:
        md = model

    return data.get(md, {}).get("params", {}).get(parameter_name, None)


def add_arange_ids(data_list):
    """
    Add the index as an id to the edge features to find seed edges in training, validation and testing.

    Args:
    - data_list (str): List of tr_data, val_data and te_data.
    """
    for data in data_list:
        data.edge_attr = torch.cat(
            [torch.arange(data.edge_attr.shape[0]).view(-1, 1), data.edge_attr], dim=1
        )


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")


def save_model(model, optimizer, epoch, config):
    # Save the model in a dictionary
    path = os.path.join(config.checkpoint_dir, f"epoch_{epoch+1}.tar")

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )

    if not config.testing:
        wandb.save(path)


def unpack_dict_ns(config, arch_id):
    """
    Unpack the dictionary and convert it to a SimpleNamespace object.

    Args:
    - config (dict): Configuration namespace.
    - arch_id (int): Architecture ID.

    Returns:
    - SimpleNamespace: Unpacked configuration.
    """

    # Unpack the architecture parameters
    arch_params = config.arch[arch_id]
    print(arch_params)
    configpy = copy.deepcopy(config)
    delattr(configpy, "arch")

    # print(arch_params)
    for key, value in arch_params.items():
        # print(key, value)
        setattr(configpy, key, value)

    # Convert the architecture parameters to SimpleNamespace

    return configpy


def find_parallel_edges(edge_index):
    simplified_edge_mapping = {}
    simplified_edge_batch = []
    i = 0
    for edge in edge_index.T:
        tuple_edge = tuple(edge.tolist())
        if tuple_edge not in simplified_edge_mapping:
            simplified_edge_mapping[tuple_edge] = i
            i += 1
        simplified_edge_batch.append(simplified_edge_mapping[tuple_edge])
    simplified_edge_batch = torch.LongTensor(simplified_edge_batch)

    return simplified_edge_batch


def get_pearl_config(config, model=None):
    file_path = f"./model_settings_{model}.json"

    # print("-------------------")
    # print(f"Loading model settings from {file_path}")
    with open(file_path, "r") as file:
        data = json.load(file)

    cfg = copy.deepcopy(config)

    param_dict = data.get("rpearl", {}).get("params", {})
    for key, value in param_dict.items():
        setattr(cfg, key, value)

    return cfg
