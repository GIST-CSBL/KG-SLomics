import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import precision_recall_fscore_support
import pickle
import os
from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset

from torch_geometric.transforms import RemoveDuplicatedEdges
from torch_geometric.utils import negative_sampling, k_hop_subgraph
from torch_geometric.data import Data

MAINPATH = '../../../data/'

with open(MAINPATH + 'KG/entity_dict.pkl', 'rb') as fr:
    entity_dict = pickle.load(fr)
    
with open(MAINPATH + 'KG/relation_dict.pkl', 'rb') as fr:
    relation_dict = pickle.load(fr)

    
def map_sl_gene2id(df, entity_dict):
    df['Gene1'] = df.apply(lambda L: entity_dict[L.iloc[1]], axis=1)
    df['Gene2'] = df.apply(lambda L: entity_dict[L.iloc[2]], axis=1)

    return df


def map_sl_id2gene(df, entity_dict):
    df['Gene1'] = df.apply(lambda L: entity_dict[L.iloc[0]], axis=1)
    df['Gene2'] = df.apply(lambda L: entity_dict[L.iloc[1]], axis=1)

    return df


def kg_genes_omics_idconvert(KG_genelist, ccle_dataset):
    KGgenesOmics = pd.merge(KG_genelist, ccle_dataset, on='Gene', how='left')
    KGgenesOmics = KGgenesOmics.fillna(0)
    KGgenesOmics['Gene'] = KGgenesOmics['Gene'].map(entity_dict)
    KGgenesOmics.rename(columns={"Gene": "NewID"}, inplace=True)
    KGgenesOmics.set_index('NewID', inplace=True)

    return KGgenesOmics


def remove_duplicate_pairs(edges):
    transform = RemoveDuplicatedEdges()
    
    if type(edges) == list:
        allEdges = Data(edge_index=torch.cat([edges[i].edge_index for i in range(len(edges))], dim=1))
        allEdges.num_nodes = len(torch.unique(allEdges.edge_index))
        nodupEdges = transform(allEdges)
    else:
        allEdges = Data(edge_index=edges)
        allEdges.num_nodes = len(torch.unique(allEdges.edge_index))
        nodupEdges = transform(allEdges)

    return nodupEdges


# Convert SL edge_index to the PyG Dataset object
class nodupSLData(Dataset):
    def __init__(self, edge_list):
        self.edge_list = edge_list
        self.edge_index = self.edge_list.edge_index
        self.edge_label = torch.ones(self.edge_index.size(1))
        
    def __getitem__(self, idx):
        edge_index = self.edge_index[:, idx]
        edge_label = self.edge_label[idx]
        
        return edge_index, edge_label
    
    def __len__(self):
        return edge_index.size(1)
    

def negative_pool_generation(pos_edges):
    bidirectional_pos = torch.cat([pos_edges, pos_edges[[1,0]]], axis=1) # Prevent the consideration of reverse direction of true pos gene pairs as neg
    u_pos_g = torch.unique(bidirectional_pos) #extract unique genes
    u_pos_g_list = u_pos_g.tolist()
    index_dict = {value: index for index, value in enumerate(u_pos_g_list)} #dictionary for reindexing

    mapped_pos_init = bidirectional_pos.clone()
    mapped_pos = mapped_pos_init.apply_(index_dict.get)
    neg_samples = negative_sampling(mapped_pos, num_neg_samples = mapped_pos.size(1)*10)

    inv_index_dict = {v: k for k, v in index_dict.items()}
    reindexed_neg_edges = neg_samples.apply_(inv_index_dict.get)

    return reindexed_neg_edges


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=1234)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset.edge_index[0])), dataset.edge_label):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset.edge_index[0]), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, val_indices, test_indices


def find_optimal_threshold(pred_probs, true_labels, thresholds=np.arange(0.05, 0.95, 0.05)):
    best_threshold = 0.5
    best_f1 = 0.0
    
    for thresh in thresholds:
        preds = (pred_probs >= thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            
    return best_threshold, best_f1


def subgraph_sampling_per_cellline(batch_data):
    edge_mask, edge_type_mask = {}, {}
    batch_data.edge_class = batch_data.edge_class[batch_data.input_id]

    seed_class = batch_data.edge_class
    src, dst = batch_data.edge_label_index

    E = batch_data.edge_index.size(1)
    batch_edge_index = batch_data.edge_index
    num_nodes = batch_data.num_nodes

    # dict: class_id -> edge_mask (torch.bool, shape=[E])
    edge_mask, edge_type_mask = {}, {}
    num_hops = 2

    # Pre-compute unique classes to avoid repeated calls
    unique_classes = torch.unique(seed_class)
    
    for c in unique_classes:
        c_id = int(c.item())

        seed_mask = (seed_class == c)
        if seed_mask.sum() == 0:
            # Use device-specific tensors for better performance
            device = batch_data.edge_index.device
            edge_mask[c_id] = torch.zeros(E, dtype=torch.bool, device=device)
            edge_type_mask[c_id] = torch.zeros(E, dtype=torch.bool, device=device)
            continue

        seed_src = src[seed_mask]
        seed_dst = dst[seed_mask]
        seed_nodes = torch.unique(torch.cat([seed_src, seed_dst], dim=0))

        _, edge_mask_c, _, mask_idx = k_hop_subgraph(
            seed_nodes,
            num_hops=num_hops,
            edge_index=batch_edge_index,
            relabel_nodes=False,
            num_nodes=num_nodes,
            directed=True
        )

        edge_mask[c_id] = edge_mask_c
        edge_type_mask[c_id] = batch_data.edge_type[mask_idx]

    batch_data.edge_mask = edge_mask
    batch_data.edge_type = edge_type_mask
    
    return batch_data


def compute_class_weights(edge_class):
    sample_counts = {}
    for i in torch.unique(edge_class):
        sample_counts[int(i.item())] = (edge_class == i).sum().item()
    
    raw_weights = {}
    for cl, count in sample_counts.items():
        raw_weights[cl] = 1.0 / (count ** 0.5)
    
    total_weight = sum(raw_weights.values())
    num_cell_lines = len(raw_weights)
    normalized_weights = {cl: (w * num_cell_lines / total_weight) for cl, w in raw_weights.items()}
    
    return normalized_weights


def compute_sl_rarity_weights(cellline_GraphList):
    """
    Compute rarity weights for SL pairs based on their frequency across cell lines.
    Rare SL pairs (appearing in fewer cell lines) get higher weights.
    """
    sl_pair_frequency = {}
    total_cell_lines = len(cellline_GraphList)
    
    # Count how often each SL pair appears across all cell lines
    for graph in cellline_GraphList:
        edges = graph.edge_index.t().tolist()
        for edge in edges:
            # Make pairs undirected by sorting
            pair = tuple(sorted(edge))
            sl_pair_frequency[pair] = sl_pair_frequency.get(pair, 0) + 1
    
    # Convert frequency to rarity weights
    sl_rarity_weights = {}
    for pair, frequency in sl_pair_frequency.items():
        # Higher weight for pairs appearing in fewer cell lines
        rarity_weight = total_cell_lines / frequency
        sl_rarity_weights[pair] = rarity_weight
    
    print(f"SL rarity weights computed for {len(sl_rarity_weights)} unique SL pairs")
    print(f"Weight range: {min(sl_rarity_weights.values()):.2f} - {max(sl_rarity_weights.values()):.2f}")
    
    return sl_rarity_weights


def apply_sl_rarity_weights(edge_label_index, edge_label, sl_weights, device):
    """
    Apply rarity weights to positive SL pairs in the current batch.
    """
    weights = torch.ones_like(edge_label, dtype=torch.float, device=device)
    
    for i, (src, dst, label) in enumerate(zip(edge_label_index[0], edge_label_index[1], edge_label)):
        if label == 1:  # Only weight positive SL pairs
            pair = tuple(sorted([src.item(), dst.item()]))
            if pair in sl_weights:
                weights[i] = sl_weights[pair]
    
    return weights

                
def print_gpu_usage(note=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"[GPU] {note} | Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
    else:
        print("CUDA is not available.")