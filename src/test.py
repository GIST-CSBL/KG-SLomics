import numpy as np
import os
from itertools import combinations
import pickle
from tqdm import tqdm
import argparse

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset

from torch_geometric import seed_everything
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import RGATConv, RotatE, avg_pool_neighbor_x
from torch_geometric.loader import LinkNeighborLoader
from torch_scatter import scatter_add

from utils import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    
np.random.seed(seed=1234)
torch.manual_seed(1234)
seed_everything(1234)


########## Data preprocessing
BaseKG = torch.load(os.path.join(MAINPATH, 'KG', 'myKG_PyG.pt'), weights_only=False, map_location="cpu")

def KGdata(file_path):
    opt_rotate = RotatE(
        num_nodes = BaseKG.num_nodes,
        num_relations = BaseKG.num_edge_types,
        hidden_channels = 128,
        margin= 10.0)
    
    #Call the pre-trained RotatE parameters
    opt_rotate.load_state_dict(torch.load(os.path.join(file_path, 'KG', 'myKG_RotatE_batch2048_lr1e-3_margin12.pt'), map_location="cpu")) 
    
    return opt_rotate


def ccle_preparation(cellline_name):
    KGgenes = pd.read_csv(MAINPATH + '/KG/KG_genes.txt', sep='\t')

    cellline_OmicsList = kg_genes_omics_idconvert(KGgenes, pd.read_csv(MAINPATH + cellline_name + '_ccle_excnamutprot_final_24Q2_innerjoin_log2cna.csv'))
    nonzero_index = cellline_OmicsList.index.values
    
    return cellline_OmicsList, nonzero_index


def scatter_multiomics_augmentation(cellline_Omics, cID, KG):
    max_iters=10
    idx = torch.as_tensor(cID, dtype=torch.long)
    omics = torch.as_tensor(cellline_Omics.values, dtype=torch.float32)

    N = KG.num_nodes
    x = torch.zeros((N, 4), dtype=torch.float32)
    x[idx] = omics

    data = ToUndirected()(KG)
    row, col = data.edge_index

    deg = torch.bincount(col, minlength=N).to(torch.float32).unsqueeze(1)
    denom = deg + 1.0

    known = torch.zeros(N, dtype=torch.bool)
    known[idx] = True
    frontier = known.clone()
    total_neigh_sum = torch.zeros_like(x)

    if frontier.any():
        mask = frontier[row]
        total_neigh_sum += scatter_add(x[row[mask]], col[mask], dim=0, dim_size=N)

    for _ in range(max_iters):
        cand = (~known) & (total_neigh_sum.abs().sum(dim=1) > 0)
        if not cand.any():
            break

        x[cand] = total_neigh_sum[cand] / denom[cand]
        frontier = cand
        known[cand] = True

        mask = frontier[row]
        if not mask.any():
            break
        total_neigh_sum += scatter_add(x[row[mask]], col[mask], dim=0, dim_size=N)

    x[idx] = omics
    
    return x


########## Model definition
class OmicsLayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)

        self.c_act = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.c_act(x)
        x = self.fc2(x)

        return x


class KGSLomics(torch.nn.Module):
    def __init__(self, args, in_channels, hidden_channels, out_channels, num_relations):
        super(KGSLomics, self).__init__()
        
        #MLP for omics data amplification
        self.ccle_layer = OmicsLayer(in_channels = 4, hidden_channels = 32, out_channels = 128)
                
        # RGAT
        self.conv1 = RGATConv(in_channels, hidden_channels//args.heads, num_relations, heads=args.heads) # in_channel, hidden_channel
        self.conv2 = RGATConv(hidden_channels, out_channels//args.heads, num_relations, heads=args.heads, concat=True) # hidden_channel*heads, out_channel
        
        self.skip_connection = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels), 
            torch.nn.LeakyReLU(), 
            torch.nn.Linear(hidden_channels, out_channels))
        
        self.sl_predictor = torch.nn.Sequential(
            torch.nn.Linear(out_channels*2, out_channels//4),
            torch.nn.LeakyReLU(), 
            torch.nn.Linear(out_channels//4, 1)
        )
        
        self.act=nn.LeakyReLU() 
        self.dropout=nn.Dropout(args.dropout)
        

    def forward(self, kg_emb, ccle, node_id, edge_index, edge_type, track_attn: bool = False):

        x_dict, attn1_dict, attn2_dict = {}, {}, {} # = {} #

        # trial2: Concat
        ccle_out={}
        ccle_out = self.ccle_layer(ccle) #, edge_index
        x_in = torch.cat((kg_emb[node_id], ccle_out[node_id]), dim=1)
        
        if track_attn:
            x, (edge_index_1, a_1) = self.conv1(x_in, edge_index, edge_type, return_attention_weights=True)
            x = self.act(x)
            x = self.dropout(x)
            x, (edge_index_2, a_2) = self.conv2(x, edge_index, edge_type, return_attention_weights=True)
            x = x + self.skip_connection(x_in)
            x = self.act(x)
    
            attn1_dict = (edge_index_1, a_1)
            attn2_dict = (edge_index_2, a_2)
            
        # Ignore attention scores for the GPU memories
        else:
            x, _ = self.conv1(x_in, edge_index, edge_type, return_attention_weights=False)
            x = self.act(x)
            x = self.dropout(x)
            x, _ = self.conv2(x, edge_index, edge_type, return_attention_weights=False)
            x = x + self.skip_connection(x_in)
            x = self.act(x)
            
        return x, attn1_dict, attn2_dict
    
    #Revised
    def decode(self, z, edge_label_index):
        
        u = z[edge_label_index[0]]
        v = z[edge_label_index[1]]
        h = u * v
        d = torch.abs(u - v)
        x = torch.cat([h, d], dim = 1)
        
        x = self.sl_predictor(x)
            
        return x.squeeze(dim=1)


########## Inference
def map_sl_gene2id_inf(df, entity_dict):
    df['Hugo Symbol'] = df.apply(lambda L: entity_dict[L.iloc[0]], axis=1)
    
    return df


def generate_inf_edge_list(cellline):
    cancergenes = pd.read_csv('./cancerGeneList_250826_filtered.txt', sep='\t')
    
    KGID_cancergenes = map_sl_gene2id_inf(cancergenes, entity_dict)
    cancer_edge_index_init = torch.tensor(KGID_cancergenes['Hugo Symbol'].values)
    
    OmicsList, KGCCLE_cID = ccle_preparation(cellline)
    src = cancer_edge_index_init.repeat_interleave(len(KGCCLE_cID))
    dst = torch.tensor(np.tile(KGCCLE_cID, len(cancer_edge_index_init)))
    
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    
    cancer_edge_index = torch.stack([src, dst], dim=0)
    
    return cancer_edge_index


@torch.no_grad()
def inference(cellline, args):

    # Data preprocessing
    inv_entity_dict = {v: k for k, v in entity_dict.items()}
    opt_rotate = KGdata(MAINPATH)
    
    OmicsList, KGCCLE_cID = ccle_preparation(cellline)    
    augmentedOmicsList = scatter_multiomics_augmentation(OmicsList, KGCCLE_cID, BaseKG)
    inf_edge_index = generate_inf_edge_list(cellline)
    
    test_data = Data(edge_index = BaseKG.edge_index, 
                     edge_type=BaseKG.edge_type,
                     edge_label_index = inf_edge_index,
                     x=[opt_rotate.node_emb.weight, augmentedOmicsList],
                     num_nodes=opt_rotate.num_nodes)
    
    test_loader = LinkNeighborLoader(test_data, edge_label_index = test_data.edge_label_index, batch_size=args.batch, 
                                     shuffle=False, neg_sampling=None, 
                                     neg_sampling_ratio=0.0, num_neighbors=[args.n_neighbor,args.n_neighbor//2])
    
    
    # Model call
    model_opt = KGSLomics(args, in_channels=256, hidden_channels=128, out_channels=64, num_relations=opt_rotate.num_relations).to(device)
    model_opt.load_state_dict(torch.load('checkpoint_0918.pt'))
    model_opt.eval()
    
    
    # Inference step
    pairs, y_pred, y_pred_prob = [], [], []
    batch_y_pred, batch_y_pred_prob = [], []
    
    for data in tqdm(test_loader):
        data = data.to(device)
        pairs.append(data.n_id[data.edge_label_index])
        
        z, a1_dict, a2_dict = model_opt(data.x[0], data.x[1], data.n_id, data.edge_index, data.edge_type, track_attn=True)
        ts_out = model_opt.decode(z, data.edge_label_index)
        ts_out_sig = ts_out.sigmoid()
        
        batch_y_pred.append(ts_out_sig>0.5)
        batch_y_pred_prob.append(ts_out_sig)
        
        del(data)
    
    all_pairs = torch.cat(pairs, dim=1)
    y_pred = torch.cat(batch_y_pred)
    y_pred_prob = torch.cat(batch_y_pred_prob)
    
    results_df = pd.DataFrame(torch.transpose(torch.vstack((all_pairs, y_pred, y_pred_prob)), 0, 1).cpu().detach().numpy(), 
                              columns=['Gene1', 'Gene2', 'y_pred', 'y_pred_prob'])
    results_df_name = map_sl_id2gene(results_df, inv_entity_dict)
    results_df_name.sort_values(by=['y_pred_prob'], ascending=False, inplace=True)

    a1_df = pd.DataFrame(torch.hstack((torch.transpose(a1_dict[0], 0, 1), a1_dict[1])).cpu().detach().numpy(),
                         columns=['Gene1', 'Gene2', 'head0', 'head1', 'head2', 'head3'])
    a1_df_name = map_sl_id2gene(a1_df, inv_entity_dict)
    a1_df_name.rename(columns={'Gene1':'Entity1', 'Gene2':'Entity2'}, inplace=True)

    a2_df = pd.DataFrame(torch.hstack((torch.transpose(a2_dict[0], 0, 1), a2_dict[1])).cpu().detach().numpy(),
                         columns=['Gene1', 'Gene2', 'head0', 'head1', 'head2', 'head3'])
    a2_df_name = map_sl_id2gene(a2_df, inv_entity_dict)
    a2_df_name.rename(columns={'Gene1':'Entity1', 'Gene2':'Entity2'}, inplace=True)
    
    return results_df_name, a1_df_name, a2_df_name


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--num_epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--heads', type=int, default=4, help='The number of heads in GAT')
    parser.add_argument('--kfold', type=int, default=5, help='The number of k for CV')
    parser.add_argument('--wd', type=float, default=1e-3, help='Weight decay')
    parser.add_argument('--es', type=int, default=10, help='Early Stopping patience')
    parser.add_argument('--splitby', type=str, default='pair', help='Split the dataset by pair or gene (input: pair | gene)')
    parser.add_argument('--negratio', type=int, default=1, help='Ratio of negative gene pairs')
    parser.add_argument('--n_neighbor', type=int, default=100, help='Number of sampled neighbors from the seed nodes')
    parser.add_argument('--cellline', type=str, help='Cell line for the inference')
    args = parser.parse_args()


    test_cellline = args.cellline
    inference_result, attention1, attention2 = inference(test_cellline, args)

    inference_result.to_csv('./SL_prediction/'+ test_cellline +'_SL_list.csv', index=False)
    attention1.to_csv('./SL_prediction/'+ test_cellline +'_att1.csv', index=False)
    attention2.to_csv('./SL_prediction/'+ test_cellline +'_att2.csv', index=False)
