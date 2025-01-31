import numpy as np
import os

import torch
from torch import nn, optim
import torch.nn.functional as F

from torch_geometric.nn import GATConv, BatchNorm, LayerNorm
from torch_geometric import seed_everything

from data_preparation import *
from utils import *
from model import *

device = "cuda:7" if torch.cuda.is_available() else "cpu"
np.random.seed(seed=1234)
torch.manual_seed(1234)
seed_everything(1234)


class MyModel(torch.nn.Module):
    def __init__(self, args, in_channels, hidden_channels, out_channels):
        super(MyModel, self).__init__()
        
        #MLP
        self.ccle_layer = nn.Sequential(
            nn.Linear(4, 32, bias=True),
            nn.LeakyReLU(),
            nn.Linear(32, 128, bias=True)
        )
        
        self.act=nn.LeakyReLU()
        self.dropout=nn.Dropout(0.3)
        
        # GAT
        self.conv1 = GATConv(in_channels, out_channels = (hidden_channels // args.heads), heads=args.heads) # in_channel, hidden_channel, heads
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1, concat=False) # hidden_channel * heads, out_channel, heads
        self.lin = torch.nn.Linear(hidden_channels, out_channels)


    def forward(self, kg_emb, ccle, neighbor_cl_ids, edge_index, args):

        x_dict, attn1_dict, attn2_dict = {}, {}, {} = {} #

        # trial2: Concat
        for e_i_class in torch.unique(neighbor_cl_ids):
            ccle[int(e_i_class.item())] = self.ccle_layer(ccle[int(e_i_class.item())])
            
            x = torch.cat((kg_emb, ccle[int(e_i_class.item())]), dim=1) #(107940, 512) -> 병합
            x_1, a_1 = self.conv1(x, edge_index, return_attention_weights=True) #
            x = self.act(x_1) #+self.norm1(x_1)
            x = self.dropout(x)
            x_2, a_2 = self.conv2(x, edge_index, return_attention_weights=True) #
            x = x_2 + self.lin(x)

            x_dict[int(e_i_class.item())] = x
            attn1_dict[int(e_i_class.item())] = a_1
            attn2_dict[int(e_i_class.item())] = a_2
            
        return x_dict, attn1_dict, attn2_dict

@torch.no_grad()
def inference(threshold, args):

    test_cellline_list = ['A549']
    _, KGCCLE_cID = ccle_preparation(cellline_list)
    test_cellline_GraphList = combine_posneg_SL_labels(test_cellline_list)
    SLlist = SLData(cellline_GraphList)
    opt_rotate = KGdata(MAINPATH)

    test_data = Data(edge_index=BaseKG.edge_index, 
                        edge_label=SLlist.edge_label, 
                        edge_label_index=SLlist.edge_index, 
                        edge_class = SLlist.edge_class, 
                        x=[opt_rotate.node_emb.weight, SLlist.x],
                        num_nodes=opt_rotate.num_nodes)

    test_loader = LinkNeighborLoader(test_data, edge_label_index=test_data.edge_label_index, edge_label=test_data.edge_label,
                                     batch_size=args.batch, shuffle=False, neg_sampling_ratio=0.0, num_neighbors=[-1,-1])

    model_opt = MyModel(args, in_channels=256, hidden_channels=128, out_channels=64).to(device)
    model_opt.load_state_dict(torch.load('checkpoint.pt'))
    model_opt.eval()

    test.specificity = 0
    test.sensitivity = 0
    test.test_f1 = 0
    test.test_auc = 0
    test.test_aupr = 0
    
    y_pred, y_pred_prob, y_true = [], [], []
    for data in tqdm(test_loader):
        data = data.to(device)
        data.edge_class = data.edge_class[data.input_id]
        data.edge_index_class = torch.zeros(len(data.edge_index[0])).to(device)
        
        y_true.append(data.edge_label)
        z, a1, a2 = model_opt(data.x[0], data.x[1], data.edge_class, data.edge_index, args)

        for i in torch.unique(data.edge_class):
            i = int(i.item())
            try:
                ts_out = ((z[i][data.edge_label_index[0][data.edge_class==i]] * z[i][data.edge_label_index[1][data.edge_class==i]]).mean(dim=-1)).view(-1)
                ts_out_sig = ((z[i][data.edge_label_index[0][data.edge_class==i]] * z[i][data.edge_label_index[1][data.edge_class==i]]).mean(dim=-1)).view(-1).sigmoid()
                
            except KeyError:
                pass
        
            y_pred.append((ts_out_sig>threshold).float().cpu())
            y_pred_prob.append((ts_out_sig).float().cpu())

        
    y, pred, pred_prob = torch.cat(y_true, dim=0).cpu().numpy(), torch.cat(y_pred, dim=0).cpu().numpy(), torch.cat(y_pred_prob, dim=0).cpu().numpy()
    for i in torch.unique(test_loader.data.edge_class):
        i = int(i.item())

        tn, fp, fn, tp = confusion_matrix(y[test_loader.data.edge_class==i], pred[test_loader.data.edge_class==i]).ravel()
        test.specificity = tn/(tn+fp)
        test.sensitivity = tp/(tp+fn)
        test.test_f1 = f1_score(y[test_loader.data.edge_class==i], pred[test_loader.data.edge_class==i])
        test.test_auc = roc_auc_score(y[test_loader.data.edge_class==i], pred_prob[test_loader.data.edge_class==i])
        test.test_aupr = average_precision_score(y[test_loader.data.edge_class==i], pred_prob[test_loader.data.edge_class==i])
        print(f"Performance of {cellline_list[i]} --> Test AUC: {test.test_auc:.4f}, Test AUPR: {test.test_aupr:.4f}, Test F1-score: {test.test_f1:.4f}")

    
    # Inference
    inv_entity_dict = {v: k for k, v in entity_dict.items()}
    for i in range(len(test_cellline_list)):
        raw_SL_probs = z[i][KGCCLE_cID] @ z[i][KGCCLE_cID].t()
        SL_probs= pd.DataFrame((raw_SL_probs.sigmoid()>0.5).nonzero(as_tuple=False).cpu().numpy(), columns=['Gene1', 'Gene2'])
        SL_probs['SL_prob'] = raw_SL_probs[(raw_SL_probs.sigmoid()>0.7)].sigmoid().cpu().detach().numpy()
        SL_probs_name = map_sl_id2gene(SL_probs, inv_entity_dict)
        SL_probs_name.sort_values(by=['SL_prob'], ascending=False, inplace=True)

        SL_probs_name.to_csv(MAINPATH + '/SL_prediction'+ test_cellline_list[i] +'/SL_list.csv', index=False) #Need to change for the multiple cell-lines

    return SL_probs_name.head(10)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--heads', type=int, default=4, help='The number of heads in GAT')
    parser.add_argument('--kfold', type=int, default=5, help='The number of k for CV')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--es', type=int, default=10, help='Early Stopping patience')
    parser.add_argument('--n_neighbors', type=int, default=3, help='The number of sampled neighbors per iterations')
    
    args = parser.parse_args()


    inference(0.5, args)