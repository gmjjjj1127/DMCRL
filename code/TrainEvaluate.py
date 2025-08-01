# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 14:01:34 2025

@author: FHt
"""

from copy import deepcopy

import numpy as np
import os

import pandas as pd

import sys

from tqdm import tqdm

import torch
from torch import nn

from torch_geometric import data as DATA
from torch_geometric.loader import DataLoader

from Data import GNNDataset

from Models import DrugProteinGNN

from utils import get_cindex
from utils import get_rm2

from utils import get_mse
from utils import get_metrics



class TrainValidate(object):
    def __init__(self, args, dataDict):
        self.args = args
        
        self.prepareData(dataDict)
        self.PregraphData(dataDict)
        
        return
    def prepareData(self, dataDict):
        args = self.args
        
        batch_size = args['batch_size']
        
        root = None
        
        dataset_tra = GNNDataset(root, dataDict, split='trai')
        loader_tra = DataLoader(dataset_tra, 
                                batch_size=batch_size, 
                                shuffle=True)

        dataset_val = GNNDataset(root, dataDict, split='vali')
        loader_val = DataLoader(dataset_val, 
                                batch_size=batch_size, 
                                shuffle=False)

        dataset_tes = GNNDataset(root, dataDict, split='test')
        loader_tes = DataLoader(dataset_tes, 
                                batch_size=batch_size, 
                                shuffle=False)
        self.loader_tra = loader_tra
        self.loader_val = loader_val
        self.loader_tes = loader_tes
        return

    def PregraphData(self, dataDict):
        args = self.args
        num_drug = args['num_drug']
        num_prot = args['num_prot']
        weighted = args['weighted']
        
        adj_norm = dataDict['adj_norm']
        
        adj_1 = adj_norm
        adj_2 = adj_norm.T
        
        adj = np.concatenate((
            np.concatenate((np.zeros([num_drug, num_drug]), adj_1), 1), 
            np.concatenate((adj_2, np.zeros([num_prot, num_prot])), 1)
            ), 0)
        
        train_raw_ids, train_col_ids = np.where(adj != 0)
        edge_indexs = np.concatenate((
            np.expand_dims(train_raw_ids, 0),
            np.expand_dims(train_col_ids, 0)
        ), 0)
        edge_weights = adj[train_raw_ids, train_col_ids]
        
        node_type_features = np.concatenate((
            np.tile(np.array([1, 0]), (num_drug, 1)), 
            np.tile(np.array([0, 1]), (num_prot, 1))
        ), 0)
        
        adj_features = np.zeros_like(adj)
        adj_features[adj != 0] = 1
        
        features = np.concatenate((node_type_features, adj_features), 1)
        
        x = torch.FloatTensor(features)
        if weighted:
            adj = torch.FloatTensor(adj)
        else:
            adj = torch.FloatTensor(adj_features)
        edge_index = torch.LongTensor(edge_indexs)
        affinity_graph = DATA.Data(x=x, 
                                   adj=adj, 
                                   edge_index=edge_index)
        
        affinity_graph.__setitem__('edge_weight', 
                                   torch.FloatTensor(edge_weights))
        
        args['graph_in_dim'] = x.shape[1]
        self.affinity_graph = affinity_graph
        return
    
    def runTrVaTe(self):
        args = self.args
        out_path = args['out_path']
        save_best = args['save_best']
        
        self.modelInit()
        output_traval = self.train()
        
        model_best = output_traval['model_best']
        metrics_df = output_traval['metrics_df']
        metr_val = output_traval['metr_val']
        

        if save_best:
            model_path = f'{out_path}_model_best.pt'
            torch.save(model_best, model_path)
            print(f'best model has been saved to {model_path}')
        
        metrics_path = f'{out_path}_metrics.csv'
        metrics_df.to_csv(metrics_path,
                          index=True,
                          index_label='epoch',
                          float_format='%.4f',
                          header=True)
        return metr_val

    def modelInit(self):
        args = self.args
        task = args['task']
        lr = args['lr']
        device = args['device']
        
        model = DrugProteinGNN(args)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        if False:
            sys.exit(1)
        elif task == 'Classification':
            loss_fn = nn.BCEWithLogitsLoss()
        elif task == 'Regression':
            loss_fn = nn.MSELoss()
        else:
            sys.exit(2)
        
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        return 
    def evaluate(self, loader):
        model = self.model
        loss_fn = self.loss_fn
        
        affinity_graph = self.affinity_graph
        
        args = self.args
        aux_weight = args['aux_weight']
        task = args['task']
        device = args['device']
        
        model.eval()
    
        y_pred_ls = []
        y_true_ls = []
        loss_all = 0
        with torch.no_grad():
            for batch_data in loader:
    
                x = batch_data.x.to(device)
                edge_index = batch_data.edge_index.to(device)
                batch = batch_data.batch.to(device)
                protein = batch_data.protein.to(device)
                drug_ids = batch_data.drug_id.to(device)
                prot_ids = batch_data.prot_id.to(device)
                affinity_graph = affinity_graph.to(device)
        
                output = model(x, edge_index, batch, 
                               protein, 
                               drug_ids, prot_ids,
                               affinity_graph)
                y_pred = output['y_pred']
                
                y_true = batch_data.y.view(-1).to(device)
                
                loss_d = output['loss_d']
                loss_p = output['loss_p']
                
                loss = loss_fn(y_pred, y_true)
                loss = loss + aux_weight * (loss_d + loss_p)
                
                if False:
                    sys.exit(7)
                elif task == 'Classification':
                    y_pred_ls += torch.sigmoid(
                                    y_pred).detach().cpu().numpy().tolist()
                elif task == 'Regression':
                    y_pred_ls += y_pred.detach().cpu().numpy().tolist()
                else:
                    sys.exit(8)
                    
                y_true_ls += y_true.detach().cpu().numpy().tolist()
                
                loss_all += loss.item() * len(y_true)
        loss_ave = loss_all / len(y_true_ls)
        mse_ave = get_mse(y_true_ls, y_pred_ls)
        # metrics = get_metrics(task, y_true_ls, y_pred_ls)
        return mse_ave #, metrics

    def train(self):
        args = self.args
        model = self.model
        optimizer = self.optimizer
        loss_fn = self.loss_fn
        task = args['task']
        patience = args['patience']
        device = args['device']
        
        loader_tra = self.loader_tra
        loader_val = self.loader_val
        loader_tes = self.loader_tes
        
        affinity_graph = self.affinity_graph
        
        epochs = args['epochs']
        aux_weight = args['aux_weight']

        metrics_all = []
        loss_best = 1000
        model_best = deepcopy(model.state_dict())
        counter = 0

        for epoch in range(epochs):
            print(f'[Epoch: {epoch+1:04d}/{epochs:04d}]')
        
            model.train()

            y_pred_ls = []
            y_true_ls = []
            loss_all = 0
            metrics_tra = dict()

            for batch_data in tqdm(loader_tra, total=len(loader_tra)):
                
                optimizer.zero_grad()
                x = batch_data.x.to(device)
                edge_index = batch_data.edge_index.to(device)
                batch = batch_data.batch.to(device)
                protein = batch_data.protein.to(device)
                drug_ids = batch_data.drug_id.to(device)
                prot_ids = batch_data.prot_id.to(device)
                affinity_graph = affinity_graph.to(device)
            
                output = model(x, edge_index, batch, 
                               protein, 
                               drug_ids, prot_ids,
                               affinity_graph)
                y_pred = output['y_pred']
                
                y_true = batch_data.y.view(-1).to(device)
                
                loss_d = output['loss_d']
                loss_p = output['loss_p']
                
                loss_tra = loss_fn(y_pred, y_true)
                loss_tra = loss_tra + aux_weight * (loss_d + loss_p)
                loss_tra.backward()
                optimizer.step()
                
                if False:
                    sys.exit()
                elif task == 'Classification':
                    y_pred_tmp = torch.sigmoid(
                        y_pred).detach().cpu().numpy().tolist()
                elif task == 'Regression':
                    y_pred_tmp = y_pred.detach().cpu().numpy().tolist()
                    
                else:
                    sys.exit(3)
                y_pred_ls += y_pred_tmp
                
                y_true_tmp = y_true.detach().cpu().numpy().tolist()
                y_true_ls += y_true_tmp
                
                loss_all += loss_tra.item() * len(y_true)
                
                metrics_tra_batch = get_metrics(task, y_true_tmp, y_pred_tmp)
                for key, value in metrics_tra_batch.items():
                    metrics_tra[key] = metrics_tra.get(key,
                                                       0) + value * len(y_true)
            
            n_sample = len(y_true_ls)
            loss_tra = loss_all / n_sample
            
            for key, value in metrics_tra.items():
                metrics_tra[key] = value / n_sample
            
            loss_val = self.evaluate(loader_val)
            
            # loss_tes, metrics_tes = self.evaluate(loader_tes)
            
            print(f'Train - Loss: {loss_tra:.4f}')
            print(f'Valid - Loss: {loss_val:.4f}')
            # print(f'Test  - Loss: {loss_tes:.4f}')
            
            metric_all = [loss_tra, loss_val] + [metrics_tra[key] 
                                                 for key in metrics_tra.keys()]
            for key in metrics_tra.keys():
                print(f'Train - {key}: {metrics_tra[key]:.4f}')
                
                
            metrics_all.append(metric_all)
            
            if loss_val < loss_best:
                loss_best = loss_val
                model_best = deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                
            if counter > patience:
                print(f'Early stopping at epoch {epoch}')
                break
                
        model.load_state_dict(model_best)
            
        columns = ['loss_tra', 'metr_val']
        for key in metrics_tra.keys():
            columns += [key+'_'+temp for temp in ['tra']]
        # 初始化空的 DataFrame
        metrics_df = pd.DataFrame(metrics_all, 
                                  columns=columns, 
                                  index=range(len(metrics_all)))
        
        output_traval = {'model_best': model_best,
                         'metrics_df': metrics_df,
                         'metr_val': loss_best}
        
        return output_traval
    
    
    
class Test(object):
    def __init__(self, args, affinity_graph, loader_tes):
        self.args = args
        self.affinity_graph = affinity_graph
        self.loader_tes = loader_tes
        return
    def runTest(self):
        args = self.args
        out_path = args['out_path']
        loader_tes = self.loader_tes
        
        self.modelInit()
        model_best_name = f'{out_path}_model_best.pt'
        self.model.load_state_dict(torch.load(model_best_name,
                                              weights_only=True))
        metrics_df = self.test(loader_tes)
        
        
        metrics_path = f'{out_path}_metrics_test.csv'
        metrics_df.to_csv(metrics_path,
                          float_format='%.4f',
                          header=True)
        return
    def modelInit(self):
        args = self.args
        task = args['task']
        device = args['device']
        
        model = DrugProteinGNN(args)
        model = model.to(device)
        
        if False:
            sys.exit(1)
        elif task == 'Classification':
            loss_fn = nn.BCEWithLogitsLoss()
        elif task == 'Regression':
            loss_fn = nn.MSELoss()
        else:
            sys.exit(2)
        
        
        self.model = model
        self.loss_fn = loss_fn
        return 
    def test(self, loader):
        model = self.model
        loss_fn = self.loss_fn
        
        affinity_graph = self.affinity_graph
        
        args = self.args
        aux_weight = args['aux_weight']
        task = args['task']
        device = args['device']
        
        model.eval()
    
        y_pred_ls = []
        y_true_ls = []
        loss_all = 0
        with torch.no_grad():
            for batch_data in loader:
    
                x = batch_data.x.to(device)
                edge_index = batch_data.edge_index.to(device)
                batch = batch_data.batch.to(device)
                protein = batch_data.protein.to(device)
                drug_ids = batch_data.drug_id.to(device)
                prot_ids = batch_data.prot_id.to(device)
                affinity_graph = affinity_graph.to(device)
        
                output = model(x, edge_index, batch, 
                               protein, 
                               drug_ids, prot_ids,
                               affinity_graph)
                y_pred = output['y_pred']
                
                y_true = batch_data.y.view(-1).to(device)
                
                loss_d = output['loss_d']
                loss_p = output['loss_p']
                
                loss = loss_fn(y_pred, y_true)
                loss = loss + aux_weight * (loss_d + loss_p)
                
                if False:
                    sys.exit(7)
                elif task == 'Classification':
                    y_pred_ls += torch.sigmoid(
                                    y_pred).detach().cpu().numpy().tolist()
                elif task == 'Regression':
                    y_pred_ls += y_pred.detach().cpu().numpy().tolist()
                else:
                    sys.exit(8)
                    
                y_true_ls += y_true.detach().cpu().numpy().tolist()
                
                loss_all += loss.item() * len(y_true)
        loss_ave = loss_all / len(y_true_ls)
        metrics = get_metrics(task, y_true_ls, y_pred_ls)
        metrics['loss'] = loss_ave
        
        metrics_all = [[metrics[key] for key in metrics.keys()]]
        
        columns = []
        for key in metrics.keys():
            columns += [key+'_'+temp for temp in ['tes']]
        # 初始化空的 DataFrame
        metrics_df = pd.DataFrame(metrics_all, 
                                  columns=columns)
        return metrics_df