import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pickle

import torch
from torch import nn, optim
from torch.utils.data import Dataset

from torch_geometric.transforms import RemoveDuplicatedEdges
from torch_geometric.data import Data

MAINPATH = '../data/'

with open(MAINPATH + '/KG/entity_dict.pkl', 'rb') as fr:
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
