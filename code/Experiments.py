# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 18:05:39 2025

@author: FHt
"""

import os

import random

import sys

from datetime import datetime

from Args import get_args

from Data import getData

import optuna

#from TrainEvaluate import TraEvaTes
from TrainEvaluate import TrainValidate
from TrainEvaluate import Test


class Objective(object):
    def __init__(self, args):
        self.args = args
        return
    def __call__(self, trial):
        args = self.args
        args['random_state'] = random.randint(0, 100)
        dataDict = getData(args)
        
        weighted = trial.suggest_categorical('weighted', [0, 1])
        aux_weight = trial.suggest_categorical('aux_weight', [0.001, 0.05, 0.01, 0.1, 0.2, 0.5, 1, 2])
        lr = trial.suggest_categorical('lr', [0.0005, 0.001, 0.005, 0.01, 0.05, 0.02, 0.1])
        
        emb_dim = trial.suggest_categorical('emb_dim', [16, 32, 64, 128, 256])
        num_layer_drug = trial.suggest_categorical('num_layer_drug', [1, 2, 3, 4])
        num_layer_protein = trial.suggest_categorical('num_layer_protein', [1, 2, 3, 4])
        
        dropout_rate = trial.suggest_categorical('dropout_rate', [0, 0.1, 0.2, 0.3,
                                                                  0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        nhead = trial.suggest_categorical('nhead', [2, 4])
        temperature = trial.suggest_float('temperature', 0.1, 1.5, step=0.05)
        
        args['weighted'] = weighted
        args['aux_weight'] = aux_weight
        args['lr'] = lr
        
        args['emb_dim'] = emb_dim
        args['num_layer_drug'] = num_layer_drug
        args['num_layer_protein'] = num_layer_protein
        
        args['dropout_rate'] = dropout_rate
        args['nhead'] = nhead
        args['temperature'] = temperature

        TET_obj = TrainValidate(args, dataDict)
        metr_val = TET_obj.runTrVaTe()
        
        return metr_val

        
    
class runExp(object):
    def __init__(self):
        args = get_args()
        self.args = args
        
        out_path = args['out_path']
        dataName =args['dataName']
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        expName = args['expName']
        
        print(f'data name is: {dataName}')
        
        dataDict = getData(args)
        self.dataDict = dataDict
        
        
        if expName == 'runModel':
            self.runModel()
        elif expName == 'optuna':
            self.run_optuna()
        else:
            sys.exit(f'wrong experiment name: {expName}')
        return
    def runModel(self):
        args = self.args
        dataDict = self.dataDict
        out_path = args['out_path']
        dataName = args['dataName']
        
        
        timestamp = args['time_str']
        args['out_path'] = f'{out_path}/{dataName}_{timestamp}'
        
        print('Training')
        
        TET_obj = TrainValidate(args, dataDict)
        TET_obj.runTrVaTe()
        
        print('Testing')
        
        affinity_graph = TET_obj.affinity_graph
        loader_tes = TET_obj.loader_tes
        test_obj = Test(args, affinity_graph, loader_tes)
        test_obj.runTest()

        return
    
    def run_optuna(self):
        args = self.args
        dataName = args['dataName']
        expName = args['expName']
        n_trials = args['n_trials']
        outKey = args['outKey']
                
        objective = Objective(args)
        
        out_path = args['out_path']
        db_path = f'{out_path}/{dataName}_{expName}_{outKey}.db'
        
        storage_name = f'sqlite:///{db_path}'
        study = optuna.create_study(
                                direction="minimize",
                                storage=storage_name,
                                study_name="dta_optimization",
                                load_if_exists=True,
                            )
        
        study.optimize(objective, n_trials=n_trials)
        
        print("最佳超参数:", study.best_params)
        print("最佳得分:", study.best_value)
        
        # 获取所有 trials 的数据
        trials_df = study.trials_dataframe()
        
        # 清理列名（去掉前缀 'params_' 和 'value'）
        trials_df.columns = [col.replace('params_', '') for col in trials_df.columns]
        trials_df.rename(columns={'value': 'objective_value'}, inplace=True)
        
        csv_path = f'{db_path}.csv'
        trials_df.to_csv(csv_path, index=True)
        
        print(f"已保存至: {csv_path}")
        return





if __name__ == '__main__':
    
    runExp()
    
    



