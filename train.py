import numpy as np
import pandas as pd
import random
import os
import argparse
import sklearn
from sklearn import preprocessing
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
import scipy
import pickle
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
from pytorchtools import EarlyStopping

from torch_geometric import seed_everything
from torch_geometric.loader import LinkNeighborLoader, ImbalancedSampler

from model import *
from utils import *
from data_preparation import *


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device('cuda:0')
if device=="cuda": torch.cuda.empty_cache()

np.random.seed(seed=1234)
torch.manual_seed(1234)
seed_everything(1234)
#torch.use_deterministic_algorithms(True)

# Experimental settings
parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
parser.add_argument('--num_epochs', type=int, default=200, help='Maximum number of epochs')
parser.add_argument('--batch', type=int, default=8, help='Batch size')
parser.add_argument('--heads', type=int, default=4, help='The number of heads in GAT')
parser.add_argument('--kfold', type=int, default=5, help='The number of k for CV')
parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--es', type=int, default=10, help='Early Stopping patience')
parser.add_argument('--splitbypair', type=bool, default=True, help='Split dataset by pair (If False, split by gene)')
args = parser.parse_args()



def train(args):
    early_stopping = EarlyStopping(patience=args.es, verbose=True)

    for epoch in range(1, args.num_epochs+1):
        
        tr_losses = 0
        val_losses = 0
        tr_loss_sum = 0
        val_loss_sum = 0
        
        model.train()
        for data in tqdm(train_loader):
            tr_loss = []
            data = data.to(device)
            data.edge_class = data.edge_class[data.input_id]

            optimizer.zero_grad()
            z, _, _ = model(data.x[0], data.x[1], data.n_id, data.edge_class, data.edge_index, data.edge_type) #전체 node feature를 모델에 넣음
            
            for i in torch.unique(data.edge_class):
                tr_loss_mean = 0
                i = int(i.item())
                
                try:
                    #tr_out = ((z[i][data.edge_label_index[0][data.edge_class==i]] * z[i][data.edge_label_index[1][data.edge_class==i]]).mean(dim=-1)).view(-1)
                    tr_out = model.decode(z[i], data.edge_label_index[:, data.edge_class==i])
                    tr_loss.append(criterion(tr_out, data.edge_label[data.edge_class==i].float()))
                except KeyError:
                    pass
                
            tr_loss_mean = sum(tr_loss)/len(tr_loss)
            tr_loss_mean.backward(retain_graph=True) #
            optimizer.step()

            tr_losses += tr_loss_mean.item()
        avg_tr_loss = tr_losses/len(train_loader.dataset)
        

        model.eval()
        with torch.no_grad():
            y_val_pred, y_val_pred_prob, y_val_true = [], [], []
            for data in tqdm(val_loader):
                val_loss = []
                data = data.to(device)
                data.edge_class = data.edge_class[data.input_id]
                
                y_val_true.append(data.edge_label)
                z, _, _ = model(data.x[0], data.x[1], data.n_id, data.edge_class, data.edge_index, data.edge_type)

                for i in torch.unique(data.edge_class):
                    val_loss_mean = 0
                    i = int(i.item())
                    
                    try:
                        #val_out = ((z[i][data.edge_label_index[0][data.edge_class==i]] * z[i][data.edge_label_index[1][data.edge_class==i]]).mean(dim=-1)).view(-1)\
                        val_out = model.decode(z[i], data.edge_label_index[:, data.edge_class==i])
                        val_out_sig = val_out.sigmoid()
                        val_loss.append(criterion(val_out, data.edge_label[data.edge_class==i].float()))
                    except KeyError:
                        pass
                    
                    y_val_pred.append((val_out_sig>0.5).float().cpu())
                    y_val_pred_prob.append((val_out_sig).float().cpu())

                val_loss_mean = sum(val_loss) / len(val_loss)
                val_losses += val_loss_mean.item()
                
        avg_val_loss = val_losses/len(val_loader.dataset)
        print(f'Epoch: {epoch:03d}, Training Loss: {avg_tr_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        y, pred, pred_prob = torch.cat(y_val_true, dim=0).cpu().numpy(), torch.cat(y_val_pred, dim=0).cpu().numpy(), torch.cat(y_val_pred_prob, dim=0).cpu().numpy()
        for i in torch.unique(val_loader.data.edge_class):
            i = int(i.item())
    
            val_f1 = f1_score(y[val_loader.data.edge_class==i], pred[val_loader.data.edge_class==i]) #average='micro'
            val_auc = roc_auc_score(y[val_loader.data.edge_class==i], pred_prob[val_loader.data.edge_class==i])
            val_aupr = average_precision_score(y[val_loader.data.edge_class==i], pred_prob[val_loader.data.edge_class==i])
            print(f'Performance of {cellline_list[i]} --> Validation AUC: {val_auc:.4f}, Validation AUPR: {val_aupr:.4f}, Validation F1-score: {val_f1:.4f}')

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


@torch.no_grad()
def test(loader, threshold, args):
    model_opt = MyModel(args, in_channels=256, hidden_channels=128, out_channels=64, num_relations=opt_rotate.num_relations).to(device)
    model_opt.load_state_dict(torch.load('checkpoint.pt'))
    model_opt.eval()

    test.specificity = []
    test.sensitivity = []
    test.test_f1 = []
    test.test_auc = []
    test.test_aupr = []
    
    y_pred, y_pred_prob, y_true = [], [], []
    for data in tqdm(test_loader):
        data = data.to(device)
        data.edge_class = data.edge_class[data.input_id]
        data.edge_index_class = torch.zeros(len(data.edge_index[0])).to(device)
        
        y_true.append(data.edge_label)
        z, a1_dict, a2_dict = model_opt(data.x[0], data.x[1], data.n_id, data.edge_class, data.edge_index, data.edge_type)

        for i in torch.unique(data.edge_class):
            i = int(i.item())
            try:
                #ts_out = ((z[i][data.edge_label_index[0][data.edge_class==i]] * z[i][data.edge_label_index[1][data.edge_class==i]]).mean(dim=-1)).view(-1)
                ts_out = model.decode(z[i], data.edge_label_index[:, data.edge_class==i])
                ts_out_sig = ts_out.sigmoid()
                
            except KeyError:
                pass
        
            y_pred.append((ts_out_sig>threshold).float().cpu())
            y_pred_prob.append((ts_out_sig).float().cpu())

        
    y, pred, pred_prob = torch.cat(y_true, dim=0).cpu().numpy(), torch.cat(y_pred, dim=0).cpu().numpy(), torch.cat(y_pred_prob, dim=0).cpu().numpy()
    for i in torch.unique(test_loader.data.edge_class):
        i = int(i.item())
        
        tn, fp, fn, tp = confusion_matrix(y[test_loader.data.edge_class==i], pred[test_loader.data.edge_class==i]).ravel()
        test.specificity.append(tn/(tn+fp))
        test.sensitivity.append(tp/(tp+fn))

        test.test_f1.append(f1_score(y[test_loader.data.edge_class==i], pred[test_loader.data.edge_class==i], average='weighted')) #average='micro'
        test.test_auc.append(roc_auc_score(y[test_loader.data.edge_class==i], pred_prob[test_loader.data.edge_class==i]))
        test.test_aupr.append(average_precision_score(y[test_loader.data.edge_class==i], pred_prob[test_loader.data.edge_class==i]))
        print(f'Performance of {cellline_list[i]} --> Test AUC: {test.test_auc[i]:.4f}, Test AUPR: {test.test_aupr[i]:.4f}, Test F1-score: {test.test_f1[i]:.4f}')
        
    return y, pred, pred_prob, a1_dict, a2_dict


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

        tr_df = pd.DataFrame(torch.transpose(torch.cat((kf_train_data.edge_label_index, 
                                                        kf_train_data.edge_class.unsqueeze(0), 
                                                        kf_train_data.edge_label.unsqueeze(0)), dim=0), 0, 1).numpy(), 
                             columns=['Gene1', 'Gene2', 'CellLine', 'Label'])
        val_df = pd.DataFrame(torch.transpose(torch.cat((kf_val_data.edge_label_index, 
                                                         kf_val_data.edge_class.unsqueeze(0), 
                                                         kf_val_data.edge_label.unsqueeze(0)), dim=0), 0, 1).numpy(), 
                              columns=['Gene1', 'Gene2', 'CellLine', 'Label'])
        ts_df = pd.DataFrame(torch.transpose(torch.cat((kf_test_data.edge_label_index, 
                                                        kf_test_data.edge_class.unsqueeze(0), 
                                                        kf_test_data.edge_label.unsqueeze(0)), dim=0), 0, 1).numpy(), 
                             columns=['Gene1', 'Gene2', 'CellLine', 'Label'])
    
        tr_df.to_csv('./CV/Fold'+str(fold)+'_tr.csv', index=False)
        val_df.to_csv('./CV/Fold'+str(fold)+'_val.csv', index=False)
        ts_df.to_csv('./CV/Fold'+str(fold)+'_ts.csv', index=False)

        #tr_sampler=ImbalancedSampler(dataset = kf_train_data.edge_label)
    
        train_loader = LinkNeighborLoader(kf_train_data, edge_label_index=kf_train_data.edge_label_index, edge_label=kf_train_data.edge_label,
                                          batch_size=args.batch, shuffle=True, neg_sampling_ratio=0.0, num_neighbors=[32,32]) #args.n_neighbors, -1
        val_loader = LinkNeighborLoader(kf_val_data, edge_label_index=kf_val_data.edge_label_index, edge_label=kf_val_data.edge_label,
                                        batch_size=args.batch, shuffle=False, neg_sampling_ratio=0.0, num_neighbors=[32,32])
        test_loader = LinkNeighborLoader(kf_test_data, edge_label_index=kf_test_data.edge_label_index, edge_label=kf_test_data.edge_label,
                                         batch_size=args.batch, shuffle=False, neg_sampling_ratio=0.0, num_neighbors=[32,32])
    
        ### Model declaration ###
        model = MyModel(args, in_channels=256, hidden_channels=128, out_channels=64, num_relations=opt_rotate.num_relations).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd) #
        criterion = torch.nn.BCEWithLogitsLoss()

        ### Training and pre-Test ###
        print("Starting to train and test model")
        train(args)
        y, pred, pred_prob, a1s_dict, a2s_dict = test(test_loader, 0.5, args)

        kf_f1[fold] = test.test_f1
        kf_auc[fold] = test.test_auc
        kf_aupr[fold] = test.test_aupr
        kf_sens[fold] = test.sensitivity
        kf_spec[fold] = test.specificity
    

    print(f'5-fold cross validation result')
    print('-------------------------------------------')
    for i in range(len(cellline_list)):
        print(f'===Performance of ', cellline_list[i], '===')
        print(f'Average AUC=', np.array([elem[i] for elem in kf_auc.values()]).mean(), ', std=', np.array([elem[i] for elem in kf_auc.values()]).std())
        print(f'Average AUPR=', np.array([elem[i] for elem in kf_aupr.values()]).mean(), ', std=', np.array([elem[i] for elem in kf_aupr.values()]).std())
        print(f'Average F1=', np.array([elem[i] for elem in kf_f1.values()]).mean(), ', std=', np.array([elem[i] for elem in kf_f1.values()]).std())
        print(f'Average Sensitivity=', np.array([elem[i] for elem in kf_sens.values()]).mean(), ', std=', np.array([elem[i] for elem in kf_sens.values()]).std())
        print(f'Average Specificity=', np.array([elem[i] for elem in kf_spec.values()]).mean(), ', std=', np.array([elem[i] for elem in kf_spec.values()]).std())
        
    
    with open('attention_1.pickle', 'wb') as fw:
        pickle.dump(a1s_dict, fw)
        
    with open('attention_2.pickle', 'wb') as fw:
        pickle.dump(a2s_dict, fw)

