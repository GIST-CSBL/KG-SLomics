import numpy as np
import pandas as pd
import random
import os
import argparse
import pickle
import sys

import torch

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
cellline_list = ['A375', 'A549', 'HeLa', 'MDAMB468', 'Jurkat', 'K562', 'IPC298', 'MELJUSO', 'MEWO', 'PC3'] 
BaseKG = torch.load(os.path.join(MAINPATH, 'KG', 'myKG_PyG.pt'), weights_only=False)


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
        SL_pos_set = pd.read_csv(MAINPATH + 'CombinedSLs/' + cellline_list[cl] + '_SL.csv')
        
        cellline_GraphList.append(generate_SL_network(SL_pos_set, cellline_OmicsList[cl]))
        cellline_GraphList[cl].edge_class = torch.full([len(cellline_GraphList[cl].edge_label)], cl)

    return cellline_GraphList


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

    # Recover the original SL pairs (containing duplicated genes) from the splitted unique SL pairs
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
        
    return recov_sl_tr_folds, recov_sl_val_folds, recov_sl_ts_folds, nodupEdges # [0]: tr; [1]: val; [2]: ts; [3]: nodupEdges; [0][i]: ith fold; [0][i][j]: jth cell line


def kfold_gene_split_dataset(SLs): 
    unique_tr_folds, unique_val_folds, unique_ts_folds = [], [], []
    recov_sl_tr_folds, recov_sl_val_folds, recov_sl_ts_folds = [], [], []
    
    nodupEdges = remove_duplicate_pairs(SLs)
    u_pos_g = torch.unique(nodupEdges.edge_index) #extract unique genes
    
    kf = KFold(n_splits = 5, shuffle=True, random_state=1234)
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(u_pos_g)):
        recov_tr_edge_index, recov_val_edge_index, recov_ts_edge_index = [], [], []
        
        train_val_genes = u_pos_g[train_val_idx]
        val_split_idx = int(0.2 * len(train_val_genes))
        
        train_genes = train_val_genes[:-val_split_idx]
        val_genes = train_val_genes[-val_split_idx:]
        test_genes = u_pos_g[test_idx]
        
        unique_tr_folds = nodupEdges.edge_index[:, torch.isin(nodupEdges.edge_index[0],train_genes)&torch.isin(nodupEdges.edge_index[1],train_genes)]
        unique_val_folds = nodupEdges.edge_index[:, torch.isin(nodupEdges.edge_index[0],val_genes)&torch.isin(nodupEdges.edge_index[1],val_genes)]
        unique_ts_folds = (nodupEdges.edge_index[:, torch.isin(nodupEdges.edge_index[0],test_genes)&torch.isin(nodupEdges.edge_index[1],test_genes)])
        
        unique_tr_expand = unique_tr_folds.unsqueeze(2)
        unique_val_expand = unique_val_folds.unsqueeze(2)
        unique_ts_expand = unique_ts_folds.unsqueeze(2)
            
        # Recover the original SL pairs (without duplicated genes) from the splitted unique SL pairs
        # Already allocated as the original gene ID
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
        
    return recov_sl_tr_folds, recov_sl_val_folds, recov_sl_ts_folds, nodupEdges


def neg_pair_sampling(cellline_id):
    #for j in range(len(cellline_GraphList)):
    if os.path.exists(MAINPATH + 'CombinedSLs/' + cellline_list[cellline_id] + '_nonSL.csv'):
        #print('The true negatives exist')
        SL_neg_set = pd.read_csv(MAINPATH + 'CombinedSLs/' + cellline_list[cellline_id] + '_nonSL.csv')
        SL_neg_set_id = map_sl_gene2id(SL_neg_set, entity_dict)
        neg_edge_index_init = torch.zeros((2, len(SL_neg_set_id)), dtype=torch.long)

        for i in range(len(SL_neg_set_id)):
            neg_edge_index_init[0,i] = SL_neg_set_id.iloc[i, 1]
            neg_edge_index_init[1,i] = SL_neg_set_id.iloc[i, 2]
        
        #print('small size of true negatives')
        if neg_edge_index_init.size(1) < cellline_GraphList[cellline_id].edge_index.size(1)*args.negratio:
            sampled_neg_edges_pool = negative_pool_generation(cellline_GraphList[cellline_id].edge_index)
            sampling_idx = torch.randperm(sampled_neg_edges_pool.size(1))
            n_supple = cellline_GraphList[cellline_id].edge_index.size(1)*args.negratio - neg_edge_index_init.size(1)
            sampled_neg_edges_idx = sampling_idx[:n_supple]
            
            final_neg_edge_index = torch.concat([neg_edge_index_init, sampled_neg_edges_pool[:, sampled_neg_edges_idx]], axis=1)
            
        #print('sufficient size of true negatives')
        else:
            sampling_idx = torch.randperm(neg_edge_index_init.size(1))
            n_supple = cellline_GraphList[cellline_id].edge_index.size(1)*args.negratio
            
            final_neg_edge_index = neg_edge_index_init[:, sampling_idx[:n_supple]]
                
    else:
        #print('The true negative absent')
        unique_pairs = remove_duplicate_pairs(cellline_GraphList)
        bidirectional_uniques = torch.cat([unique_pairs.edge_index, unique_pairs.edge_index[[1,0]]], axis=1)
        
        sampled_neg_edges_pool = negative_pool_generation(bidirectional_uniques)
        sampling_idx = torch.randperm(sampled_neg_edges_pool.size(1))
        n_supple = cellline_GraphList[cellline_id].edge_index.size(1)*args.negratio
        sampled_neg_edges_idx = sampling_idx[:n_supple]
        
        final_neg_edge_index = sampled_neg_edges_pool[:, sampled_neg_edges_idx]
            
    return final_neg_edge_index


def conditional_neg_sampling(true_neg, split_data):
    if true_neg.size(1) < split_data.size(1)*args.negratio:
        sampled_neg_edges_pool = negative_pool_generation(torch.concat([split_data, true_neg], axis=1))
        rand_idx = torch.randperm(sampled_neg_edges_pool.size(1))
        final_neg_edge_index = torch.concat([true_neg,
                                             sampled_neg_edges_pool[:, rand_idx[:((split_data.size(1)*args.negratio)-true_neg.size(1))]]], axis=1)
    else:
        sampling_idx = torch.randperm(true_neg.size(1))
        final_neg_edge_index = true_neg[:, sampling_idx[:split_data.size(1)*args.negratio]]
        
    return final_neg_edge_index


def generate_complete_SLpDataset(tr, val, ts):
    tr_edge_info, val_edge_info, ts_edge_info = [], [], []
    
    for j in range(len(cellline_GraphList)):
        sampled_neg_edges_pool = neg_pair_sampling(j) # negative sampling for positive samples in jth cell line
        folds_neg = torch.randperm(sampled_neg_edges_pool.size(1))
        
        tr_neg_edges = sampled_neg_edges_pool[:, folds_neg[:(tr[j].size(1))*args.negratio]]
        tr_edge_info.append(Data(edge_label = torch.cat([torch.ones(tr[j].size(1)), torch.zeros((tr[j].size(1))*args.negratio)]),
                                 edge_label_index = torch.cat([tr[j], tr_neg_edges], dim=1),
                                 edge_class = torch.full([tr[j].size(1)*(1+args.negratio)], j)))
        
        val_neg_edges = sampled_neg_edges_pool[:, (folds_neg[(tr[j].size(1)*args.negratio):((tr[j].size(1)+val[j].size(1))*args.negratio)])]
        val_edge_info.append(Data(edge_label = torch.cat([torch.ones(val[j].size(1)), torch.zeros(val[j].size(1)*args.negratio)]),
                                 edge_label_index = torch.cat([val[j], val_neg_edges], dim=1),
                                 edge_class = torch.full([val[j].size(1)*(1+args.negratio)], j)))
        
        ts_neg_edges = sampled_neg_edges_pool[:, (folds_neg[(tr[j].size(1)+val[j].size(1))*args.negratio:((tr[j].size(1)+val[j].size(1))*args.negratio+ts[j].size(1)*args.negratio)])]
        ts_edge_info.append(Data(edge_label = torch.cat([torch.ones(ts[j].size(1)), torch.zeros(ts[j].size(1)*args.negratio)]),
                                 edge_label_index = torch.cat([ts[j], ts_neg_edges], dim=1),
                                 edge_class = torch.full([ts[j].size(1)*(1+args.negratio)], j)))
    
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


def generate_complete_SLgDataset(tr, val, ts):
    tr_edge_info, val_edge_info, ts_edge_info = [], [], []
    
    for j in range(len(cellline_GraphList)):
        if os.path.exists(MAINPATH + 'CombinedSLs/' + cellline_list[j] + '_nonSL.csv'):
            #print('The true negatives exist')
            SL_neg_set = pd.read_csv(MAINPATH + 'CombinedSLs/' + cellline_list[j] + '_nonSL.csv')
            SL_neg_set_id = map_sl_gene2id(SL_neg_set, entity_dict)
            neg_edge_index_init = torch.zeros((2, len(SL_neg_set_id)), dtype=torch.long)
        
            for i in range(len(SL_neg_set_id)):
                neg_edge_index_init[0,i] = SL_neg_set_id.iloc[i, 1]
                neg_edge_index_init[1,i] = SL_neg_set_id.iloc[i, 2]
            
            tr_mask = (neg_edge_index_init.unsqueeze(0) == torch.unique(tr[j]).view(-1, 1, 1)).any(dim=0).any(dim=0)
            filtered_tr_neg = neg_edge_index_init[:, tr_mask]
            tr_neg_edges = conditional_neg_sampling(filtered_tr_neg, tr[j])
            
            val_mask = (neg_edge_index_init.unsqueeze(0) == torch.unique(val[j]).view(-1, 1, 1)).any(dim=0).any(dim=0)
            filtered_val_neg = neg_edge_index_init[:, val_mask]
            val_neg_edges = conditional_neg_sampling(filtered_val_neg, val[j])
            
            ts_mask = (neg_edge_index_init.unsqueeze(0) == torch.unique(ts[j]).view(-1, 1, 1)).any(dim=0).any(dim=0)
            filtered_ts_neg = neg_edge_index_init[:, ts_mask]
            ts_neg_edges = conditional_neg_sampling(filtered_ts_neg, ts[j])
            
        else:
            #print('The true negative absent')
            transform = RemoveDuplicatedEdges()
            
            all_tr_pos_data = Data(edge_index=torch.cat(tr, dim=1))
            all_tr_pos_data.num_nodes = len(torch.unique(all_tr_pos_data.edge_index))
            tr_nodupEdges = transform(all_tr_pos_data)
            tr_neg_pool = negative_pool_generation(tr_nodupEdges.edge_index)
            tr_perm_idx = torch.randperm(tr_neg_pool.size(1))
            tr_neg_edges = tr_neg_pool[:, tr_perm_idx[:tr[j].size(1)*args.negratio]]
            
            all_val_pos_data = Data(edge_index=torch.cat(val, dim=1))
            all_val_pos_data.num_nodes = len(torch.unique(all_val_pos_data.edge_index))
            val_nodupEdges = transform(all_val_pos_data)
            val_neg_pool = negative_pool_generation(val_nodupEdges.edge_index)
            val_perm_idx = torch.randperm(val_neg_pool.size(1))
            val_neg_edges = val_neg_pool[:, val_perm_idx[:val[j].size(1)*args.negratio]]
            
            all_ts_pos_data = Data(edge_index=torch.cat(ts, dim=1))
            all_ts_pos_data.num_nodes = len(torch.unique(all_ts_pos_data.edge_index))
            ts_nodupEdges = transform(all_ts_pos_data)
            ts_neg_pool = negative_pool_generation(ts_nodupEdges.edge_index)
            ts_perm_idx = torch.randperm(ts_neg_pool.size(1))
            ts_neg_edges = ts_neg_pool[:, ts_perm_idx[:ts[j].size(1)*args.negratio]]
            
            
        tr_edge_info.append(Data(edge_label = torch.cat([torch.ones(tr[j].size(1)), torch.zeros((tr[j].size(1))*args.negratio)]),
                             edge_label_index = torch.cat([tr[j], tr_neg_edges], dim=1),
                             edge_class = torch.full([tr[j].size(1)*(1+args.negratio)], j)))
        
        val_edge_info.append(Data(edge_label = torch.cat([torch.ones(val[j].size(1)), torch.zeros(val[j].size(1)*args.negratio)]),
                             edge_label_index = torch.cat([val[j], val_neg_edges], dim=1),
                             edge_class = torch.full([val[j].size(1)*(1+args.negratio)], j)))
        
        ts_edge_info.append(Data(edge_label = torch.cat([torch.ones(ts[j].size(1)), torch.zeros(ts[j].size(1)*args.negratio)]),
                             edge_label_index = torch.cat([ts[j], ts_neg_edges], dim=1),
                             edge_class = torch.full([ts[j].size(1)*(1+args.negratio)], j)))
        
        
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--num_epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--heads', type=int, default=4, help='The number of heads in GAT')
    parser.add_argument('--kfold', type=int, default=5, help='The number of k for CV')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--es', type=int, default=10, help='Early Stopping patience')
    parser.add_argument('--splitby', type=str, default='pair', help='Split the dataset by pair or gene (input: pair | gene)')
    parser.add_argument('--negratio', type=int, default=10, help='Ratio of negative gene pairs')
    args = parser.parse_args()

    cellline_GraphList = combine_SL_labels(cellline_list)
    opt_rotate = KGdata(MAINPATH)
    #opt_rotate.node_emb.weight.requires_grad=False

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    starter.record()
    
    print('=========================================-')
    if args.splitby == 'pair':
        print(f'Split by gene pairs')
        print('-------------------------------------------')
        tr_folds, val_folds, ts_folds, unique_edges = kfold_pair_split_dataset(cellline_GraphList) #variable[fold][cellline]
    
    elif args.splitby == 'gene':
        print(f'Split by genes')
        print('-------------------------------------------')
        tr_folds, val_folds, ts_folds, unique_edges = kfold_gene_split_dataset(cellline_GraphList) #variable[fold][cellline]
        
    else:
        print('Please insert the valid parameter; pair or gene')
        sys.exit(1)

    
    for fold in range(args.kfold):
        print('-------------------------------------------')
        print(f'FOLD {fold}')
        
        if args.splitby == 'pair':
            kf_train_data, kf_val_data, kf_test_data = generate_complete_SLpDataset(tr_folds[fold], val_folds[fold], ts_folds[fold])
        elif args.splitby == 'gene':
            kf_train_data, kf_val_data, kf_test_data = generate_complete_SLgDataset(tr_folds[fold], val_folds[fold], ts_folds[fold])
        else:
            print('Please insert the valid parameter; pair or gene')
            sys.exit(1)
        
        print(kf_train_data)
        print(kf_val_data)
        print(kf_test_data)
        
        #break
    
    print('-----------------------------------------')
    print('Time and Memory information')
    print('-----------------------------------------')
    ender.record()
    torch.cuda.synchronize()
    
    infer_time = starter.elapsed_time(ender)
    print("Elapsed time: {} s".format(infer_time * 1e-3))
        
    # Print model GPU
    print('Memory Usage:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Max Alloc:', round(torch.cuda.max_memory_allocated(0)/1024**3, 1), 'GB')