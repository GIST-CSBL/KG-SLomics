import scipy
import pickle
from tqdm import tqdm
import argparse
import math

import sklearn
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_recall_curve

import torch
from torch import nn, optim
from torch.utils.data import Dataset
import torch.nn as nn
from torch.amp import autocast, GradScaler
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

# Experimental settings
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
args = parser.parse_args()


def train(args, sl_rarity_weights):
    early_stopping = EarlyStopping(patience=args.es, verbose=True)
    scaler = GradScaler('cuda')

    for epoch in range(1, args.num_epochs+1):
        
        tr_losses = 0
        val_losses = 0
        tr_loss_sum = 0
        val_loss_sum = 0
        
        model.train()
        for data in tqdm(train_loader):
            tr_loss = []
            data = data.to(device)
            
            optimizer.zero_grad()
            # Mixed precision forward pass
            with autocast('cuda'):
                z, _, _ = model(data.x[0], data.x[1], data.n_id, data.edge_class, data.edge_mask, data.edge_type, track_attn=False) 
                
                # Pre-compute weights once per batch instead of recalculating
                normalized_weights = compute_class_weights(data.edge_class)
                
                # Optimized loss calculation
                unique_classes = torch.unique(data.edge_class)
                for i in unique_classes:
                    i = int(i.item())
                    class_mask = (data.edge_class == i)
                    
                    if class_mask.sum() > 0:
                        tr_out = model.decode(z[i], data.edge_label_index[:, class_mask])
                        
                        # Apply SL rarity weights
                        rarity_weights = apply_sl_rarity_weights(
                            data.edge_label_index[:, class_mask], 
                            data.edge_label[class_mask], 
                            sl_rarity_weights, 
                            device
                        )
                        
                        # Compute loss with rarity weighting
                        base_loss = criterion(tr_out, data.edge_label[class_mask].float())
                        if base_loss.dim() == 0:  # scalar loss
                            rarity_weighted_loss = base_loss * rarity_weights.mean()
                        else:  # element-wise loss
                            rarity_weighted_loss = (base_loss * rarity_weights).mean()
                        
                        # Apply cell line weights
                        final_weighted_loss = rarity_weighted_loss * normalized_weights[i]
                        tr_loss.append(final_weighted_loss)
                
                if tr_loss:
                    tr_loss_mean = sum(tr_loss) / len(tr_loss)
            
            # Mixed precision backward pass
            if tr_loss:
                scaler.scale(tr_loss_mean).backward()
                scaler.step(optimizer)
                scaler.update()

                tr_losses += tr_loss_mean.item()
        avg_tr_loss = tr_losses/len(train_loader)
        

        model.eval()
        with torch.no_grad():
            y_val_pred, y_val_pred_prob, y_val_true, edge_class_val = [], [], [], []
            val_loss = []
            for data in tqdm(val_loader):
                data = data.to(device)
                
                y_val_true.append(data.edge_label)
                edge_class_val.append(data.edge_class)  # Collect edge_class from each batch

                # Mixed precision forward pass for validation
                with autocast('cuda'):
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
                            
                            # Apply SL rarity weights to validation loss
                            rarity_weights = apply_sl_rarity_weights(
                                data.edge_label_index[:, class_mask], 
                                data.edge_label[class_mask], 
                                sl_rarity_weights, 
                                device
                            )
                            
                            # Compute loss with rarity weighting
                            base_loss = criterion(val_out, data.edge_label[class_mask].float())
                            if base_loss.dim() == 0:  # scalar loss
                                rarity_weighted_loss = base_loss * rarity_weights.mean()
                            else:  # element-wise loss
                                rarity_weighted_loss = (base_loss * rarity_weights).mean()
                            
                            # Apply cell line weights
                            final_weighted_loss = rarity_weighted_loss * normalized_weights[i]
                            batch_val_loss.append(final_weighted_loss)
                            
                            batch_y_val_pred.append((val_out_sig > 0.5).float().cpu())
                            batch_y_val_pred_prob.append(val_out_sig.float().cpu())
                
                if batch_val_loss:
                    batch_val_loss_mean = sum(batch_val_loss) / len(batch_val_loss)
                    val_losses += batch_val_loss_mean.item()
                    
                    y_val_pred.extend(batch_y_val_pred)
                    y_val_pred_prob.extend(batch_y_val_pred_prob)
                
            avg_val_loss = val_losses/len(val_loader)
            print(f'Epoch: {epoch:03d}, Training Loss: {avg_tr_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
            torch.cuda.empty_cache()

        # Check if this is the best model so far (before early stopping updates it)
        is_best_model = early_stopping.best_score is None or (-avg_val_loss) > early_stopping.best_score
        
        early_stopping(avg_val_loss, model)
        
        # Calculate optimal thresholds only when model improves
        if is_best_model and y_val_pred:
            print("Model improved - calculating optimal thresholds...")
            y = torch.cat(y_val_true, dim=0).cpu().numpy()
            pred_prob = torch.cat(y_val_pred_prob, dim=0).cpu().numpy()
            edge_class_all = torch.cat(edge_class_val, dim=0).cpu().numpy()
            
            cell_line_thresholds = {}
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
            
            # Store the best thresholds
            if not hasattr(train, 'optimal_thresholds'):
                train.optimal_thresholds = {}
            train.optimal_thresholds = cell_line_thresholds
            
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
    
    pairs, y_pred, y_pred_prob, y_true, edge_class_test = [], [], [], [], []
    for data in tqdm(loader):
        data = data.to(device)
        pairs.append(data.n_id[data.edge_label_index])
        y_true.append(data.edge_label)
        edge_class_test.append(data.edge_class)  # Collect edge_class from each batch
                
        # Mixed precision forward pass for test
        with autocast('cuda'):
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

    all_pairs = torch.cat(pairs, dim=1).cpu().numpy()
    y = torch.cat(y_true, dim=0).cpu().numpy()
    y_vals = torch.cat(y_pred, dim=0).cpu().numpy()
    pred_prob = torch.cat(y_pred_prob, dim=0).cpu().numpy()
    edge_class_all = torch.cat(edge_class_test, dim=0).cpu().numpy()
    
    for i in np.unique(edge_class_all):
        i = int(i)
        mask = edge_class_all == i
        
        if np.sum(mask) > 0:  
            cell_threshold = optimal_thresholds.get(i, 0.5)
            pred_optimal = (pred_prob[mask] >= cell_threshold).astype(int)
            
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
        
    return all_pairs, y, y_vals, pred_prob, edge_class_all, a1_dict, a2_dict


if __name__ == "__main__":
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    # Import the datasets (single file) and initialize the cell-line list
    cellline_list = ['A375', 'A549', 'HeLa', 'MDAMB468', 'Jurkat', 'K562', 'IPC298', 'MELJUSO', 'MEWO', 'PC3'] 
    BaseKG = torch.load(os.path.join(MAINPATH, 'KG', 'myKG_PyG.pt'), weights_only=False, map_location="cpu")
    
    kf_f1, kf_auc, kf_aupr = {}, {}, {}
    cellline_GraphList = combine_SL_labels(cellline_list)
    opt_rotate = KGdata(MAINPATH)
    inv_entity_dict = {v: k for k, v in entity_dict.items()}
    
    # Compute SL rarity weights based on frequency across cell lines
    print('Computing SL rarity weights...')
    sl_rarity_weights = compute_sl_rarity_weights(cellline_GraphList)
    
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
                                          batch_size=args.batch, shuffle=True, neg_sampling_ratio=0.0, num_neighbors=[args.n_neighbor,args.n_neighbor//2], 
                                          transform = subgraph_sampling_per_cellline)
        val_loader = LinkNeighborLoader(kf_val_data, edge_label_index=kf_val_data.edge_label_index, edge_label=kf_val_data.edge_label,
                                        batch_size=args.batch, shuffle=False, neg_sampling_ratio=0.0, num_neighbors=[args.n_neighbor,args.n_neighbor//2], 
                                        transform = subgraph_sampling_per_cellline)
        test_loader = LinkNeighborLoader(kf_test_data, edge_label_index=kf_test_data.edge_label_index, edge_label=kf_test_data.edge_label,
                                         batch_size=args.batch, shuffle=False, neg_sampling_ratio=0.0, num_neighbors=[args.n_neighbor,args.n_neighbor//2], 
                                         transform = subgraph_sampling_per_cellline)
    
        model = KGSLomics(args, in_channels=256, hidden_channels=128, out_channels=64, num_relations=opt_rotate.num_relations).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd) #
        criterion = torch.nn.BCEWithLogitsLoss()

        print("Starting to train and test model")
        train(args, sl_rarity_weights)      
        optimal_thresholds = getattr(train, 'optimal_thresholds', {})
        
        # Fallback: if no optimal thresholds were calculated, use default 0.5
        if not optimal_thresholds:
            print("Warning: No optimal thresholds found, using default 0.5 for all cell lines")
            optimal_thresholds = {i: 0.5 for i in range(len(cellline_list))}
        all_pairs, y, y_vals, pred_prob, edge_class_all, a1s_dict, a2s_dict = test(test_loader, optimal_thresholds, args)

        kf_f1[fold] = test.test_f1
        kf_auc[fold] = test.test_auc
        kf_aupr[fold] = test.test_aupr
        
        torch.save(model.state_dict(), 'intermediate_checkpoint.pt')
        
        test_fold_result = pd.DataFrame(np.vstack((all_pairs, y, y_vals, pred_prob, edge_class_all)).T, 
                               columns=['Gene1', 'Gene2', 'y', 'y_vals', 'pred_prob', 'cell line ID'])
        test_fold_result_name = map_sl_id2gene(test_fold_result, inv_entity_dict)
        test_fold_result_name.to_csv('./test_result_genename_fold'+str(fold)+'.csv', index=False)
    
    
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

