import numpy as np
import pandas as pd
import random
import os
import argparse
import pickle

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.nn import RotatE, avg_pool_neighbor_x
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import negative_sampling
from torch_geometric import seed_everything

from utils import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device('cuda:0')
if device=="cuda": torch.cuda.empty_cache()

np.random.seed(seed=1234)
torch.manual_seed(1234)
seed_everything(1234)

# Import the datasets (single file) and initialize the cell-line list
cellline_list = ['A375', 'A549', 'HeLa', 'MDAMB468', 'Jurkat', 'K562', 'IPC298', 'MELJUSO', 'MEWO', 'PC3'] #, 'GI1'
BaseKG = torch.load(os.path.join(MAINPATH, 'KG', 'myKG_PyG.pt'))


# Import the datasets (KG, CCLE)
def KGdata(file_path):
    print('Loading the knowledge graph data...')

    opt_rotate = RotatE(
        num_nodes = BaseKG.num_nodes,
        num_relations = BaseKG.num_edge_types,
        hidden_channels = 128,
        margin= 10.0)
    
    #Call the pre-trained RotatE parameters
    opt_rotate.load_state_dict(torch.load(os.path.join(file_path, 'KG', 'myKG_RotatE_batch2048_lr1e-3_margin12.pt'), map_location=device)) 
    
    return opt_rotate


def ccle_preparation(cellline_list):
    KGgenes = pd.read_csv(MAINPATH + '/KG/KG_genes.txt', sep='\t')
    cellline_OmicsList = []
    
    for om in range(len(cellline_list)):
        cellline_OmicsList.append(kg_genes_omics_idconvert(KGgenes, pd.read_csv(MAINPATH + cellline_list[om] + '_ccle_excnamutprot_final_24Q2_innerjoin_log2cna.csv')))
    
    nonzero_index = cellline_OmicsList[0].index.values
    
    return cellline_OmicsList, nonzero_index


#Need to check
def multiomics_augmentation(cellline_Omics, KG):
    _, KGCCLE_cID = ccle_preparation(cellline_list)

    tmp = torch.zeros((BaseKG.num_nodes, 4), dtype=torch.float)
    nonzero_index = torch.tensor(KGCCLE_cID)
    tmp.index_add_(0, nonzero_index, torch.tensor(cellline_Omics.values, dtype=torch.float))
    
    BaseKG_undir = ToUndirected()(KG)
    BaseKG_undir.x = tmp
    
    zero_mask = (BaseKG_undir.x == 0).all(dim=1)
    if zero_mask.any():
        BaseKG_undir = avg_pool_neighbor_x(BaseKG_undir.cpu())
        BaseKG_undir.x[nonzero_index] = tmp[nonzero_index]
    
    return BaseKG_undir.x


def generate_SL_network(SL_set, omics):
    SL_pos_set_id = map_sl_gene2id(SL_set, entity_dict)
            
    pos_edge_index_init = torch.zeros((2, len(SL_pos_set_id)), dtype=torch.long)

    for i in range(len(SL_pos_set_id)):
        pos_edge_index_init[0,i] = SL_pos_set_id.iloc[i, 1]
        pos_edge_index_init[1,i] = SL_pos_set_id.iloc[i, 2]
    
    sl_ccle_graph = Data(edge_index=pos_edge_index_init, edge_label=torch.ones(pos_edge_index_init.size(1), dtype=torch.long))
    sl_ccle_graph.x = multiomics_augmentation(omics, BaseKG)
    sl_ccle_graph.edge_class = torch.zeros(len(sl_ccle_graph.edge_label))

    return sl_ccle_graph


def combine_SL_labels(cellline_list):
    cellline_OmicsList, _ = ccle_preparation(cellline_list)
    cellline_GraphList = []

    #Generate completed PyG graph object per cancer cell line
    for cl in range(len(cellline_list)):
        SL_pos_set = pd.read_csv(MAINPATH + 'SLKB_Final/' + cellline_list[cl] + '_SL_filtered.csv')
        
        cellline_GraphList.append(generate_SL_network(SL_pos_set, cellline_OmicsList[cl]))
        cellline_GraphList[cl].edge_class = torch.full([len(cellline_GraphList[cl].edge_label)], cl)

    return cellline_GraphList


def remove_duplicate_pairs(edges):
    transform = RemoveDuplicatedEdges()
    allEdges = Data(edge_index=torch.cat([edges[i].edge_index for i in range(len(edges))], dim=1))
    nodupEdges = transform(allEdges)

    return nodupEdges


def kfold_pair_split_dataset(SLs):
    unique_tr_folds, unique_val_folds, unique_ts_folds = [], [], []
    recov_sl_tr_folds, recov_sl_val_folds, recov_sl_ts_folds = [], [], []
    
    # Remove duplicated SL pairs across the cancer cell lines
    nodupEdges = remove_duplicate_pairs(SLs)
    uniqueSLDataset = nodupSLData(nodupEdges) #converting edge_index to PyG graph object

    # Split the unique SL pairs into k-fold cross-validation format
    for fold, (train_idx, val_idx, test_idx) in enumerate(zip(*k_fold(uniqueSLDataset, 5))):
        unique_train = Data(edge_index = uniqueSLDataset.edge_index[:, train_idx], edge_label=uniqueSLDataset.edge_label[train_idx])
        unique_val = Data(edge_index = uniqueSLDataset.edge_index[:, val_idx], edge_label=uniqueSLDataset.edge_label[val_idx])
        unique_test = Data(edge_index = uniqueSLDataset.edge_index[:, test_idx], edge_label=uniqueSLDataset.edge_label[test_idx])
        
        unique_tr_folds.append(unique_train.edge_index)
        unique_val_folds.append(unique_val.edge_index)
        unique_ts_folds.append(unique_test.edge_index)

    # Recover the original SL pairs (containing duplicates) from the splitted unique SL pairs
    # training sls
    for i in range(5):
        recov_tr_edge_index, recov_val_edge_index, recov_ts_edge_index = [], [], []
        
        unique_tr_expand = unique_tr_folds[i].unsqueeze(2)
        unique_val_expand = unique_val_folds[i].unsqueeze(2)
        unique_ts_expand = unique_ts_folds[i].unsqueeze(2)
        
        for j in range(len(SLs)):
            origin_sls = SLs[j].edge_index
            origin_sls_expand = origin_sls.unsqueeze(1)
            
            tr_comparison = (unique_tr_expand == origin_sls_expand).all(dim=0)
            _, tr_original_indices = tr_comparison.nonzero(as_tuple=True)
            recov_tr_edge_index.append(origin_sls[:, tr_original_indices])
            
            val_comparison = (unique_val_expand == origin_sls_expand).all(dim=0)
            _, val_original_indices = val_comparison.nonzero(as_tuple=True)
            recov_val_edge_index.append(origin_sls[:, val_original_indices])
            
            ts_comparison = (unique_ts_expand == origin_sls_expand).all(dim=0)
            _, ts_original_indices = ts_comparison.nonzero(as_tuple=True)
            recov_ts_edge_index.append(origin_sls[:, ts_original_indices])
        
        recov_sl_tr_folds.append(recov_tr_edge_index)
        recov_sl_val_folds.append(recov_val_edge_index)
        recov_sl_ts_folds.append(recov_ts_edge_index)
        
    return recov_sl_tr_folds, recov_sl_val_folds, recov_sl_ts_folds, nodupEdges # 5 x i(cell-lines) x 3(tr/val/ts) edge index


def kfold_gene_split_dataset(SLs):
    unique_tr_folds, unique_val_folds, unique_ts_folds = [], [], []
    recov_sl_tr_folds, recov_sl_val_folds, recov_sl_ts_folds = [], [], []
    
    # Remove duplicated Sl pairs across the cancer cell lines
    nodupEdges = remove_duplicate_pairs(SLs)
    u_pos_g = torch.unique(nodupEdges.edge_index) #extract unique genes
    
    # Split the unique SL pairs into k-fold cross-validation format
    kf = KFold(n_splits = 5, shuffle=True, random_state=1234)
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(u_pos_g)):
        train_val_genes = u_pos_g[train_val_idx]
        
        val_split_idx = int(0.2 * len(train_val_genes))
        train_genes = train_val_genes[:-val_split_idx]
        val_genes = train_val_genes[-val_split_idx:]
        test_genes = u_pos_g[test_idx]

        
        unique_tr_folds.append(nodupEdges.edge_index[:, torch.isin(nodupEdges.edge_index[0],train_genes)&torch.isin(nodupEdges.edge_index[1],train_genes)])
        unique_val_folds.append(nodupEdges.edge_index[:, torch.isin(nodupEdges.edge_index[0],val_genes)&torch.isin(nodupEdges.edge_index[1],val_genes)])
        unique_ts_folds.append(nodupEdges.edge_index[:, torch.isin(nodupEdges.edge_index[0],test_genes)&torch.isin(nodupEdges.edge_index[1],test_genes)])
    
    # Recover the original SL pairs (containing duplicates) from the splitted unique SL pairs
    # training sls
    for i in range(5):
        recov_tr_edge_index, recov_val_edge_index, recov_ts_edge_index = [], [], []
        
        unique_tr_expand = unique_tr_folds[i].unsqueeze(2)
        unique_val_expand = unique_val_folds[i].unsqueeze(2)
        unique_ts_expand = unique_ts_folds[i].unsqueeze(2)
        
        for j in range(len(SLs)):
            origin_sls = SLs[j]
            origin_sls_expand = origin_sls.unsqueeze(1)
            
            tr_comparison = (unique_tr_expand == origin_sls_expand).all(dim=0)
            _, tr_original_indices = tr_comparison.nonzero(as_tuple=True)
            recov_tr_edge_index.append(origin_sls[:, tr_original_indices])
            
            val_comparison = (unique_val_expand == origin_sls_expand).all(dim=0)
            _, val_original_indices = val_comparison.nonzero(as_tuple=True)
            recov_val_edge_index.append(origin_sls[:, val_original_indices])
            
            ts_comparison = (unique_ts_expand == origin_sls_expand).all(dim=0)
            _, ts_original_indices = ts_comparison.nonzero(as_tuple=True)
            recov_ts_edge_index.append(origin_sls[:, ts_original_indices])
            
        recov_sl_tr_folds.append(recov_tr_edge_index)
        recov_sl_val_folds.append(recov_val_edge_index)
        recov_sl_ts_folds.append(recov_ts_edge_index)
        
    return recov_sl_tr_folds, recov_sl_val_folds, recov_sl_ts_folds, nodupEdges


def neg_sampling(unique_edges):
    u_pos_g = torch.unique(unique_edges) #extract unique genes
    u_pos_g_list = u_pos_g.tolist()
    index_dict = {value: index for index, value in enumerate(u_pos_g_list)} #dictionary for reindexing

    mapped_pos_init = unique_edges.clone()
    mapped_pos = mapped_pos_init.apply_(index_dict.get)
    neg_samples = negative_sampling(mapped_pos, num_neg_samples = mapped_pos.size(1)*10)

    inv_index_dict = {v: k for k, v in index_dict.items()}
    reindexed_neg_edges = neg_samples.apply_(inv_index_dict.get)

    return reindexed_neg_edges


def generate_complete_SLDataset(tr, val, ts, u_edges):
    tr_edge_info, val_edge_info, ts_edge_info = [], [], []
    sampled_neg_edges_pool = neg_sampling(u_edges.edge_index)
    cellline_GraphList = combine_SL_labels(cellline_list)
    opt_rotate = KGdata(MAINPATH)
    
    for j in range(len(cellline_GraphList)):
        if os.path.exists(MAINPATH + 'SLKB_Final/' + cellline_list[j] + '_nonSL_filtered.csv'):
            #print('The true negatives exist')
            SL_neg_set = pd.read_csv(MAINPATH + 'SLKB_Final/' + cellline_list[j] + '_SL_filtered.csv')
            SL_neg_set_id = map_sl_gene2id(SL_neg_set, entity_dict)
            neg_edge_index_init = torch.zeros((2, len(SL_neg_set_id)), dtype=torch.long)

            for i in range(len(SL_neg_set_id)):
                neg_edge_index_init[0,i] = SL_neg_set_id.iloc[i, 1]
                neg_edge_index_init[1,i] = SL_neg_set_id.iloc[i, 2]
                
            folds_neg = torch.randperm(neg_edge_index_init.size(1))
                
        else:
            #print('The true negative absent')
            folds_neg = torch.randperm(sampled_neg_edges_pool.size(1))

        tr_neg_edges = sampled_neg_edges_pool[:, folds_neg[:(tr[j].size(1))]]
        tr_edge_info.append(Data(edge_label = torch.cat([torch.ones(tr[j].size(1)), torch.zeros(tr[j].size(1))]),
                                 edge_label_index = torch.cat([tr[j], tr_neg_edges], dim=1),
                                 edge_class = torch.full([tr[j].size(1)*2], j)))

        val_neg_edges = sampled_neg_edges_pool[:, (folds_neg[(tr[j].size(1)):(tr[j].size(1)+val[j].size(1))])]
        val_edge_info.append(Data(edge_label = torch.cat([torch.ones(val[j].size(1)), torch.zeros(val[j].size(1))]),
                                 edge_label_index = torch.cat([val[j], val_neg_edges], dim=1),
                                 edge_class = torch.full([val[j].size(1)*2], j)))

        ts_neg_edges = sampled_neg_edges_pool[:, (folds_neg[(tr[j].size(1)+val[j].size(1)):(tr[j].size(1)+val[j].size(1)+ts[j].size(1))])]
        ts_edge_info.append(Data(edge_label = torch.cat([torch.ones(ts[j].size(1)), torch.zeros(ts[j].size(1))]),
                                 edge_label_index = torch.cat([ts[j], ts_neg_edges], dim=1),
                                 edge_class = torch.full([ts[j].size(1)*2], j)))

    kf_train = Data(edge_index = BaseKG.edge_index,
                    edge_type=BaseKG.edge_type,
                    edge_label = torch.cat([tr_edge_info[k].edge_label for k in range(len(tr_edge_info))]),
                    edge_label_index = torch.cat([tr_edge_info[k].edge_label_index for k in range(len(tr_edge_info))], dim=1),
                    edge_class = torch.cat([tr_edge_info[k].edge_class for k in range(len(tr_edge_info))]),
                    x=[opt_rotate.node_emb.weight, [cellline_Graph.x for cellline_Graph in cellline_GraphList]],
                    num_nodes=opt_rotate.num_nodes)

    kf_val = Data(edge_index = BaseKG.edge_index,
                    edge_type=BaseKG.edge_type,
                    edge_label = torch.cat([val_edge_info[k].edge_label for k in range(len(val_edge_info))]),
                    edge_label_index = torch.cat([val_edge_info[k].edge_label_index for k in range(len(val_edge_info))], dim=1),
                    edge_class = torch.cat([val_edge_info[k].edge_class for k in range(len(val_edge_info))]),
                    x=[opt_rotate.node_emb.weight, [cellline_Graph.x for cellline_Graph in cellline_GraphList]],
                    num_nodes=opt_rotate.num_nodes)

    kf_ts = Data(edge_index = BaseKG.edge_index,
                    edge_type=BaseKG.edge_type,
                    edge_label = torch.cat([ts_edge_info[k].edge_label for k in range(len(ts_edge_info))]),
                    edge_label_index = torch.cat([ts_edge_info[k].edge_label_index for k in range(len(ts_edge_info))], dim=1),
                    edge_class = torch.cat([ts_edge_info[k].edge_class for k in range(len(ts_edge_info))]),
                    x=[opt_rotate.node_emb.weight, [cellline_Graph.x for cellline_Graph in cellline_GraphList]],
                    num_nodes=opt_rotate.num_nodes)

    return kf_train, kf_val, kf_ts

'''
if __name__ == "__main__":
    kf_f1, kf_auc, kf_aupr, kf_sens, kf_spec = {}, {}, {}, {}, {}
    cellline_GraphList = combine_SL_labels(cellline_list)
    opt_rotate = KGdata(MAINPATH)
    #opt_rotate.node_emb.weight.requires_grad=False

    print('=========================================-')
    if args.splitbypair == True:
        print(f'Split by gene pairs')
        print('-------------------------------------------')
        tr_folds, val_folds, ts_folds, unique_edges = kfold_pair_split_dataset(cellline_GraphList) #variable[fold][cellline]
    
    else:
        pass
    
    for fold in range(args.kfold):
        print('-------------------------------------------')
        print(f'FOLD {fold}')
        kf_train_data, kf_val_data, kf_test_data = generate_complete_SLDataset(tr_folds[fold], val_folds[fold], ts_folds[fold], unique_edges)
        
        print(kf_train_data)
        print(kf_val_data)
        print(kf_test_data)
        
        #break
'''
