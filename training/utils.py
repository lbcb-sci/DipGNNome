import os
import yaml
import torch
import random
import numpy as np


def set_seed(seed=42):
    """Set random seed to enable reproducibility.

    Parameters
    ----------
    seed : int, optional
        A number used to set the random seed

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    #dgl.seed(seed)


def get_num_nodes_per_cluster(g, config):
    num_nodes_per_cluster_min = int(config['num_nodes_per_cluster'] * config['npc_lower_bound'])
    num_nodes_per_cluster_max = int(config['num_nodes_per_cluster'] * config['npc_upper_bound']) + 1
    num_nodes_for_g = torch.LongTensor(1).random_(num_nodes_per_cluster_min,
                                                  num_nodes_per_cluster_max).item()
    num_clusters = g.num_nodes() // num_nodes_for_g + 1
    return num_clusters


def create_dataset_dicts(data_config='dataset.yml'):
    with open(data_config) as file:
        config_file = yaml.safe_load(file)
    train_dataset = config_file['training']
    val_dataset = config_file['validation']

    # Initialize the full_dataset dictionary
    full_dataset = {}
    # Add all keys and values from train_dataset to full_dataset
    for key, value in train_dataset.items():
        full_dataset[key] = value
    # Add keys from val_dataset to full_dataset, summing values if key already exists
    if val_dataset is not None:
        for key, value in val_dataset.items():
            if key in full_dataset:
                full_dataset[key] += value
            else:
                full_dataset[key] = value

    return full_dataset, val_dataset, train_dataset


def get_numbered_graphs(graph_dict, starting_counts=None):
    if graph_dict is None:
        return []
    numbered_graph_list = []
    if starting_counts is None:
        starting_counts = {}
    for chr, amount in graph_dict.items():
        # Determine the starting index for each chromosome based on starting_counts
        if chr in starting_counts.keys():
            start_index = starting_counts[chr]
        else:
            start_index = 0
        for i in range(start_index, start_index + amount):
            chr_id = f'{chr.replace(".", "_")}_{i}'
            numbered_graph_list.append(chr_id)

    return numbered_graph_list
