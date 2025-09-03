import scipy
import pickle
from tqdm import tqdm
import argparse
import math

import sklearn
from sklearn import preprocessing
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, auc, precision_recall_fscore_support

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
from pytorchtools import EarlyStopping

from torch_geometric import seed_everything
from torch_geometric.loader import LinkNeighborLoader

from model_subgraph import *
from utils import *
from data_preparation import *


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
if use_cuda:
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

np.random.seed(seed=1234)
torch.manual_seed(1234)
seed_everything(1234)
#torch.use_deterministic_algorithms(True)

# Experimental settings
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
parser.add_argument('--num_epochs', type=int, default=200, help='Maximum number of epochs')
parser.add_argument('--batch', type=int, default=4, help='Batch size')
parser.add_argument('--heads', type=int, default=4, help='The number of heads in GAT')
parser.add_argument('--kfold', type=int, default=5, help='The number of k for CV')
parser.add_argument('--wd', type=float, default=1e-3, help='Weight decay')
parser.add_argument('--es', type=int, default=10, help='Early Stopping patience')
parser.add_argument('--splitby', type=str, default='pair', help='Split the dataset by pair or gene (input: pair | gene)')
parser.add_argument('--negratio', type=int, default=1, help='Ratio of negative gene pairs')
parser.add_argument('--n_neighbor', type=int, default=128, help='Ratio of negative gene pairs')
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
            data = subgraph_sampling_per_cellline(data)
            
            optimizer.zero_grad()
            z, _, _ = model(data.x[0], data.x[1], data.n_id, data.edge_class, data.edge_mask, data.edge_type, track_attn=False) #data.edge_index|data.edge_mask
            
            # Pre-compute weights once per batch instead of recalculating
            normalized_weights = compute_class_weights(data.edge_class)
            
            # Optimized loss calculation
            unique_classes = torch.unique(data.edge_class)
            for i in unique_classes:
                i = int(i.item())
                class_mask = (data.edge_class == i)
                
                if class_mask.sum() > 0:
                    tr_out = model.decode(z[i], data.edge_label_index[:, class_mask])
                    loss = criterion(tr_out, data.edge_label[class_mask].float())
                    weighted_loss = loss * normalized_weights[i]
                    tr_loss.append(weighted_loss)
            
            if tr_loss:
                tr_loss_mean = sum(tr_loss) / len(tr_loss)
                tr_loss_mean.backward()
                optimizer.step()

                tr_losses += tr_loss_mean.item()
        avg_tr_loss = tr_losses/len(train_loader)
        

        model.eval()
        with torch.no_grad():
            y_val_pred, y_val_pred_prob, y_val_true, edge_class_val = [], [], [], []
            val_loss = []
            for data in tqdm(val_loader):
                data = data.to(device)
                data = subgraph_sampling_per_cellline(data)
                
                y_val_true.append(data.edge_label)
                edge_class_val.append(data.edge_class)  # Collect edge_class from each batch

                #data.edge_index or data.edge_mask
                z, _, _ = model(data.x[0], data.x[1], data.n_id, data.edge_class, data.edge_mask, data.edge_type, track_attn=False) 
        
                # Pre-compute weights once per batch instead of recalculating
                normalized_weights = compute_class_weights(data.edge_class)

                batch_val_loss = []
                batch_y_val_pred = []
                batch_y_val_pred_prob = []
                
                unique_classes = torch.unique(data.edge_class)
                for i in unique_classes:
                    i = int(i.item())
                    class_mask = (data.edge_class == i)
                    
                    if class_mask.sum() > 0:
                        val_out = model.decode(z[i], data.edge_label_index[:, class_mask])
                        val_out_sig = val_out.sigmoid()
                        loss = criterion(val_out, data.edge_label[class_mask].float())
                        
                        weighted_loss = loss * normalized_weights[i]
                        batch_val_loss.append(weighted_loss)
                        
                        batch_y_val_pred.append((val_out_sig > 0.5).float().cpu())
                        batch_y_val_pred_prob.append(val_out_sig.float().cpu())
                
                if batch_val_loss:
                    batch_val_loss_mean = sum(batch_val_loss) / len(batch_val_loss)
                    val_losses += batch_val_loss_mean.item()
                    
                    y_val_pred.extend(batch_y_val_pred)
                    y_val_pred_prob.extend(batch_y_val_pred_prob)
                
            avg_val_loss = val_losses/len(val_loader)
            print(f'Epoch: {epoch:03d}, Training Loss: {avg_tr_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        cell_line_thresholds = {}
        if y_val_pred:  # Only if we have predictions
            y = torch.cat(y_val_true, dim=0).cpu().numpy()
            pred_prob = torch.cat(y_val_pred_prob, dim=0).cpu().numpy()
            edge_class_all = torch.cat(edge_class_val, dim=0).cpu().numpy()
        
            for i in np.unique(edge_class_all):
                i = int(i)
                mask = edge_class_all == i
                
                if np.sum(mask) > 0:
                    optimal_threshold, optimal_f1 = find_optimal_threshold(pred_prob[mask], y[mask])
                    cell_line_thresholds[i] = optimal_threshold
                    
                    # Calculate metrics with optimal threshold
                    pred_optimal = (pred_prob[mask] >= optimal_threshold).astype(int)
                    val_auc = roc_auc_score(y[mask], pred_prob[mask])
                    val_aupr = average_precision_score(y[mask], pred_prob[mask])
                    print(f'Performance of {cellline_list[i]} --> Validation AUC: {val_auc:.4f}, Validation AUPR: {val_aupr:.4f}, Validation F1-score: {optimal_f1:.4f}')
                    
        if not hasattr(train, 'optimal_thresholds'):
            train.optimal_thresholds = {}
        train.optimal_thresholds = cell_line_thresholds

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


@torch.no_grad()
def test(loader, optimal_thresholds, args):
    model_opt = KGSLomics(args, in_channels=256, hidden_channels=128, out_channels=64, num_relations=opt_rotate.num_relations).to(device)
    model_opt.load_state_dict(torch.load('checkpoint.pt'))
    model_opt.eval()

    test.test_f1 = []
    test.test_auc = []
    test.test_aupr = []
    
    y_pred, y_pred_prob, y_true, edge_class_test = [], [], [], []
    for data in tqdm(test_loader):
        data = subgraph_sampling_per_cellline(data)
        data = data.to(device)
            
        y_true.append(data.edge_label)
        edge_class_test.append(data.edge_class)  # Collect edge_class from each batch
        #z, a1_dict, a2_dict = model_opt(data.x[0], data.x[1], data.n_id, data.edge_class, data.edge_index, data.edge_type, track_attn=True)
        z, a1_dict, a2_dict = model_opt(data.x[0], data.x[1], data.n_id, data.edge_class, data.edge_mask, data.edge_type, track_attn=True)

        batch_y_pred = []
        batch_y_pred_prob = []
        
        for i in torch.unique(data.edge_class):
            i = int(i.item())
            try:
                ts_out = model_opt.decode(z[i], data.edge_label_index[:, data.edge_class==i])
                ts_out_sig = ts_out.sigmoid()
                
                cell_threshold = optimal_thresholds.get(i, 0.5)
                batch_y_pred.append((ts_out_sig>cell_threshold).float().cpu())
                batch_y_pred_prob.append((ts_out_sig).float().cpu())
            except KeyError:
                pass
        
        y_pred.extend(batch_y_pred)
        y_pred_prob.extend(batch_y_pred_prob)

    y = torch.cat(y_true, dim=0).cpu().numpy()
    pred_prob = torch.cat(y_pred_prob, dim=0).cpu().numpy()
    edge_class_all = torch.cat(edge_class_test, dim=0).cpu().numpy()
    
    for i in np.unique(edge_class_all):
        i = int(i)
        mask = edge_class_all == i
        
        if np.sum(mask) > 0:  
            cell_threshold = optimal_thresholds.get(i, 0.5)
            pred_optimal = (pred_prob[mask] >= cell_threshold).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y[mask], pred_optimal).ravel()

            test.test_f1.append(f1_score(y[mask], pred_optimal))
            test.test_auc.append(roc_auc_score(y[mask], pred_prob[mask]))
            test.test_aupr.append(average_precision_score(y[mask], pred_prob[mask]))
            idx = len(test.test_f1) - 1
            print(f'Performance of {cellline_list[i]} Test AUC: {test.test_auc[idx]:.4f}, Test AUPR: {test.test_aupr[idx]:.4f}, Test F1-score: {test.test_f1[idx]:.4f}')
        else:
            test.test_f1.append(0.0)
            test.test_auc.append(0.0)
            test.test_aupr.append(0.0)
            print(f'Performance of {cellline_list[i]} --> No samples for evaluation')
        
    return y, pred, pred_prob, a1_dict, a2_dict


if __name__ == "__main__":
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    kf_f1, kf_auc, kf_aupr = {}, {}, {}
    cellline_GraphList = combine_SL_labels(cellline_list)
    opt_rotate = KGdata(MAINPATH)
    
    starter.record()
    
    print('=========================================')
    if args.splitby == 'pair':
        print(f'Split by gene pairs')
        print('-----------------------------------------')
        tr_folds, val_folds, ts_folds, unique_edges = kfold_pair_split_dataset(cellline_GraphList) #variable[fold][cellline]
    
    elif args.splitby == 'gene':
        print(f'Split by genes')
        print('-----------------------------------------')
        tr_folds, val_folds, ts_folds, unique_edges = kfold_gene_split_dataset(cellline_GraphList) #variable[fold][cellline]
        
    else:
        print('Please insert the valid parameter; pair or gene')
        sys.exit(1)
        
    
    for fold in range(args.kfold):
        print('-----------------------------------------')
        print(f'FOLD {fold}')
        
        if args.splitby == 'pair':
            kf_train_data, kf_val_data, kf_test_data = generate_complete_SLpDataset(tr_folds[fold], val_folds[fold], ts_folds[fold], args)
            
        elif args.splitby == 'gene':
            kf_train_data, kf_val_data, kf_test_data = generate_complete_SLgDataset(tr_folds[fold], val_folds[fold], ts_folds[fold], args)
            
        else:
            print('Please insert the valid parameter; pair or gene')
            sys.exit(1)

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

        train_loader = LinkNeighborLoader(kf_train_data, edge_label_index=kf_train_data.edge_label_index, edge_label=kf_train_data.edge_label,
                                          batch_size=args.batch, shuffle=True, neg_sampling_ratio=0.0, num_neighbors=[args.n_neighbor,args.n_neighbor//2]) #args.n_neighbors, -1
        val_loader = LinkNeighborLoader(kf_val_data, edge_label_index=kf_val_data.edge_label_index, edge_label=kf_val_data.edge_label,
                                        batch_size=args.batch, shuffle=False, neg_sampling_ratio=0.0, num_neighbors=[args.n_neighbor,args.n_neighbor//2])
        test_loader = LinkNeighborLoader(kf_test_data, edge_label_index=kf_test_data.edge_label_index, edge_label=kf_test_data.edge_label,
                                         batch_size=args.batch, shuffle=False, neg_sampling_ratio=0.0, num_neighbors=[args.n_neighbor,args.n_neighbor//2])
    
        model = KGSLomics(args, in_channels=256, hidden_channels=128, out_channels=64, num_relations=opt_rotate.num_relations).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd) #
        criterion = torch.nn.BCEWithLogitsLoss()

        print("Starting to train and test model")
        train(args)      
        optimal_thresholds = getattr(train, 'optimal_thresholds', {})
        y, pred, pred_prob, a1s_dict, a2s_dict = test(test_loader, optimal_thresholds, args)

        kf_f1[fold] = test.test_f1
        kf_auc[fold] = test.test_auc
        kf_aupr[fold] = test.test_aupr
    
    
    print('-----------------------------------------')
    print(f'5-fold cross validation result')
    print('-----------------------------------------')
    for i in range(len(cellline_list)):
        print(f'=== Performance of ', cellline_list[i], ' ===')
        print(f'Average AUC=', np.array([elem[i] for elem in kf_auc.values()]).mean(), ', std=', np.array([elem[i] for elem in kf_auc.values()]).std())
        print(f'Average AUPR=', np.array([elem[i] for elem in kf_aupr.values()]).mean(), ', std=', np.array([elem[i] for elem in kf_aupr.values()]).std())
        print(f'Average F1=', np.array([elem[i] for elem in kf_f1.values()]).mean(), ', std=', np.array([elem[i] for elem in kf_f1.values()]).std())
    
    
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
    
    
    with open('attention_1.pickle', 'wb') as fw:
        pickle.dump(a1s_dict, fw)
        
    with open('attention_2.pickle', 'wb') as fw:
        pickle.dump(a2s_dict, fw)

