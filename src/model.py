import numpy as np
import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, RGATConv
from torch_geometric import seed_everything
from data_preparation import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device('cuda:0')
if device=="cuda": torch.cuda.empty_cache()

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


#Model size: 0.353 MiB
class MyModel(torch.nn.Module):
    def __init__(self, args, in_channels, hidden_channels, out_channels, num_relations):
        super(MyModel, self).__init__()
        
        #MLP for omics data amplification
        self.ccle_layer = GCNLayer(in_channels = 4, hidden_channels = 32, out_channels = 128)
        
        #MLPs for SL prediction module
        self.fc1 = torch.nn.Linear(out_channels*2, out_channels)
        self.fc2 = torch.nn.Linear(out_channels, int(out_channels/2))
        self.fc3 = torch.nn.Linear(int(out_channels/2), 1)
        
        self.act=nn.LeakyReLU()
        self.dropout=nn.Dropout(args.dropout)
        
        # GAT
        #self.conv1 = GATConv(in_channels, out_channels = (hidden_channels // args.heads), heads=args.heads) # in_channel, hidden_channel, heads
        #self.conv2 = GATConv(hidden_channels, out_channels//4, heads=4, concat=True) # hidden_channel * heads, out_channel, heads
        
        self.conv1 = RGATConv(in_channels, hidden_channels//args.heads, num_relations, heads=args.heads) # in_channel, hidden_channel, heads
        self.conv2 = RGATConv(hidden_channels, out_channels//args.heads, num_relations, heads=args.heads, concat=True) # hidden_channel * heads, out_channel, heads
        self.lin = torch.nn.Linear(in_channels, out_channels)


    def forward(self, kg_emb, ccle, node_id, neighbor_cl_ids, edge_index, edge_type): #모델이 입력으로 받는 부분

        x_dict, attn1_dict, attn2_dict = {}, {}, {} # = {} #

        # trial2: Concat
        for e_i_class in torch.unique(neighbor_cl_ids):
            ccle[int(e_i_class.item())] = self.ccle_layer(ccle[int(e_i_class.item())], edge_index)
            
            x = torch.cat((kg_emb[node_id], ccle[int(e_i_class.item())][node_id]), dim=1) #(107940, 256) -> 병합
            x, (edge_index_1, a_1) = self.conv1(x, edge_index, edge_type, return_attention_weights=True) #
            x = self.act(x) #+self.norm1(x_1)
            x = self.dropout(x)
            x, (edge_index_2, a_2) = self.conv2(x, edge_index, edge_type, return_attention_weights=True) #
            x = x + self.lin(torch.cat((kg_emb[node_id], ccle[int(e_i_class.item())][node_id]), dim=1))

            x_dict[int(e_i_class.item())] = x
            attn1_dict[int(e_i_class.item())] = (edge_index_1, a_1)
            attn2_dict[int(e_i_class.item())] = (edge_index_2, a_2)
            
        return x_dict, attn1_dict, attn2_dict
    
    
    def decode(self, z, edge_label_index):
        x_probs = {}
            
        x = torch.cat((z[edge_label_index[0]], z[edge_label_index[1]]), dim=1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
            
        return x.squeeze(dim=1)

