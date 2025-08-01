import argparse

import sys

import time
from datetime import datetime

import torch


def get_args():
    
    parser = argparse.ArgumentParser(
        description='project for drug-target prediction',
        epilog='示例: python Args.py --dataName davis --expName runModel'
        )
    
    parser.add_argument('--dataName', default='davis',
                        choices=['davis', 'kiba',
                        'human', 'celegans',
                        'toxcast','metz'], 
                        help='data name')
    parser.add_argument('--expName', default='runModel',
                        choices=['runModel', 
                                 'optuna'])
    parser.add_argument('--split_type', default='warm',
                        choices=['warm', 'cold-drug', 'cold-prot','cold-drug-prot'])
    parser.add_argument('--outKey', default=' ', type=str)
    parser.add_argument('--patience', default=30, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--cuda', default=0, choices=[0, 1], type=int)
    
    parser.add_argument('--n_trials', default=10, type=int)
    parser.add_argument('--save_best', default=0, choices=[0, 1], type=int)
    parser_args = parser.parse_args()
    parser_dict = vars(parser_args)
    
    
    
    args = dict()
    
    args['expName'] = parser_dict['expName']
    
    args['dataName'] = parser_dict['dataName'] 
                               
    args['split_type'] = parser_dict['split_type'] 
    args['outKey'] = parser_dict['outKey']
    args['patience'] = parser_dict['patience']
    
    args['epochs'] = parser_dict['epochs']
    
    cuda = parser_dict['cuda']
    args['device'] = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() 
                                  else 'cpu')
    
    args['n_trials'] = parser_dict['n_trials']
    args['save_best'] = parser_dict['save_best']
    
    args['weighted'] = 1 
    
    args['aux_weight'] = 1 
    
    args['lr'] = 0.001 
    
    args['emb_dim'] = 128 
    args['num_layer_drug'] = 3
    args['num_layer_protein'] = 3
    args['dropout_rate'] = 0.2 
    args['nhead'] = 2  
    
    args['batch_size'] = 32
    
    
    args['target_len'] = 1200 
    args['node_feature_dim'] = 22
    
    
    args['time_str'] =  datetime.now().strftime('%Y%m%d_%H%M%S')

    
    args['random_state'] = 42
    args['out_path'] = '../output_cold-drug'
    args['atom_feat_dim'] = 22
    args['temperature'] = 1.0 
    
    
    
    dataName = args['dataName']
    if False:
        sys.exit()
    elif dataName == 'human':
        args['task'] = 'Classification'
        
        args['weighted'] = 1
        args['aux_weight'] = 0.001
        args['lr'] = 0.0005
        args['emb_dim'] = 16
        args['num_layer_drug'] = 4
        args['num_layer_protein'] = 2
        args['dropout_rate'] = 0.8
        args['nhead'] = 4
        args['temperature'] = 0.9
    elif dataName == 'celegans':
        args['task'] = 'Classification'
        
        args['weighted'] = 0
        args['aux_weight'] = 2
        args['lr'] = 0.001
        args['emb_dim'] = 128
        args['num_layer_drug'] = 4
        args['num_layer_protein'] = 4
        args['dropout_rate'] = 0.6
        args['nhead'] = 2
        args['temperature'] = 1.1
    elif dataName == 'davis':
        args['task'] = 'Regression'
        
        args['weighted'] = 1
        args['aux_weight'] = 0.01
        args['lr'] = 0.0005
        args['emb_dim'] = 256
        args['num_layer_drug'] = 1
        args['num_layer_protein'] = 3
        args['dropout_rate'] = 0.2
        args['nhead'] = 2
        args['temperature'] = 1.25
    elif dataName == 'kiba':
        args['task'] = 'Regression'

        args['weighted'] = 1
        args['aux_weight'] = 0.2
        args['lr'] = 0.0005
        args['emb_dim'] = 256
        args['num_layer_drug'] = 2
        args['num_layer_protein'] = 1
        args['dropout_rate'] = 0
        args['nhead'] = 2
        args['temperature'] = 0.15
    elif dataName == 'toxcast':
        args['task'] = 'Regression'
        
        args['weighted'] = 1
        args['aux_weight'] = 0.05
        args['lr'] = 0.001
        args['emb_dim'] = 128
        args['num_layer_drug'] = 4
        args['num_layer_protein'] = 2
        args['dropout_rate'] = 0.2
        args['nhead'] = 4
        args['temperature'] = 0.85
    elif dataName == 'metz':
        args['task'] = 'Regression'
        
        args['weighted'] = 0
        args['aux_weight'] = 0.2
        args['lr'] = 0.0005
        args['emb_dim'] = 256
        args['num_layer_drug'] = 2
        args['num_layer_protein'] = 2
        args['dropout_rate'] = 0.3
        args['nhead'] = 2
        args['temperature'] = 1.2
    else:
        sys.exit(f'Wrong dataName {dataName} for task')
        
    print(args)
    return args

if __name__ == '__main__':
    args = get_args()
