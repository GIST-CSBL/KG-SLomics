import numpy as np
import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGATConv
from torch_geometric import seed_everything

from data_preparation import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
if use_cuda:
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

np.random.seed(seed=1234)
torch.manual_seed(1234)
seed_everything(1234)


class GCNLayer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.c_conv1 = GCNConv(in_channels, hidden_channels)
        self.c_conv2 = GCNConv(hidden_channels, out_channels)

        self.c_act = nn.LeakyReLU()

    def forward(self, x, edge_index):
        x = self.c_conv1(x, edge_index)
        x = self.c_act(x)
        x = self.c_conv2(x, edge_index)

        return x


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
        #self.ccle_layer = GCNLayer(in_channels = 4, hidden_channels = 32, out_channels = 128)
        self.ccle_layer = OmicsLayer(in_channels = 4, hidden_channels = 32, out_channels = 128)
        
        #MLPs for SL prediction module
        self.fc1 = torch.nn.Linear(out_channels*2, out_channels)
        self.fc2 = torch.nn.Linear(out_channels, 1)
        
        self.act=nn.LeakyReLU() 
        self.dropout=nn.Dropout(args.dropout)
                
        # RGAT
        self.conv1 = RGATConv(in_channels, hidden_channels//args.heads, num_relations, heads=args.heads) # in_channel, hidden_channel
        self.conv2 = RGATConv(hidden_channels, out_channels//args.heads, num_relations, heads=args.heads, concat=True) # hidden_channel*heads, out_channel
        self.skip_connection = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels), 
            torch.nn.LeakyReLU(), 
            torch.nn.Linear(hidden_channels, out_channels))


    def forward(self, kg_emb, ccle, node_id, neighbor_cl_ids, edge_index, edge_type, track_attn: bool = False):

        x_dict, attn1_dict, attn2_dict = {}, {}, {} # = {} #

        # trial2: Concat
        for e_i_class in torch.unique(neighbor_cl_ids):
            ccle_out={}
            #ccle_out[int(e_i_class.item())] = self.ccle_layer(ccle[int(e_i_class.item())], edge_index)
            ccle_out = self.ccle_layer(ccle[int(e_i_class.item())])
            x_in = torch.cat((kg_emb[node_id], ccle_out[node_id]), dim=1)
            
            # Get attention scores for the analysis
            if track_attn:
                x, (edge_index_1, a_1) = self.conv1(x_in, edge_index[int(e_i_class.item())], edge_type[int(e_i_class.item())], return_attention_weights=True)
                x = self.act(x)
                x = self.dropout(x)
                x, (edge_index_2, a_2) = self.conv2(x, edge_index[int(e_i_class.item())], edge_type[int(e_i_class.item())], return_attention_weights=True)
                x = x + self.skip_connection(x_in)
                x = self.act(x)
    
                x_dict[int(e_i_class.item())] = x
                attn1_dict[int(e_i_class.item())] = (edge_index_1, a_1)
                attn2_dict[int(e_i_class.item())] = (edge_index_2, a_2)
            
            # Ignore attention scores for the GPU memories
            else:
                x, _ = self.conv1(x_in, edge_index[int(e_i_class.item())], edge_type[int(e_i_class.item())], return_attention_weights=False)
                x = self.act(x)
                x = self.dropout(x)
                x, _ = self.conv2(x, edge_index[int(e_i_class.item())], edge_type[int(e_i_class.item())], return_attention_weights=False)
                x = x + self.skip_connection(x_in)
                x = self.act(x)
    
                x_dict[int(e_i_class.item())] = x
            
        return x_dict, attn1_dict, attn2_dict
    
    #Revised
    def decode(self, z, edge_label_index):
        
        u = z[edge_label_index[0]]
        v = z[edge_label_index[1]]
        h = u * v
        d = torch.abs(u - v)
        x = torch.cat([h, d], dim = 1)
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
            
        return x.squeeze(dim=1)

