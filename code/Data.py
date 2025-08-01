# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 18:37:20 2025

@author: FHt
"""


import os

import os.path as osp

import numpy as np
import pandas as pd

import sys

from tqdm import tqdm



from torch_geometric.sampler.neighbor_sampler import neg_sample


from sklearn.model_selection import train_test_split

import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

import torch
from torch import nn

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


def get_nodes(g):
    feat = []
    for n, d in g.nodes(data=True):
        h_t = []
        h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I', ]]
        h_t.append(d['a_num'])
        h_t.append(d['acceptor'])
        h_t.append(d['donor'])
        h_t.append(int(d['aromatic']))
        h_t += [int(d['hybridization'] == x) \
                for x in (Chem.rdchem.HybridizationType.SP, \
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3)]
        h_t.append(d['num_h'])
        # 5 more
        h_t.append(d['ExplicitValence'])
        h_t.append(d['FormalCharge'])
        h_t.append(d['ImplicitValence'])
        h_t.append(d['NumExplicitHs'])
        h_t.append(d['NumRadicalElectrons'])
        feat.append((n, h_t))
    feat.sort(key=lambda item: item[0])
    node_attr = torch.FloatTensor([item[1] for item in feat])

    return node_attr



def get_edges(g):
    e = {}
    for n1, n2, d in g.edges(data=True):
        e_t = [int(d['b_type'] == x)
               for x in (Chem.rdchem.BondType.SINGLE, \
                         Chem.rdchem.BondType.DOUBLE, \
                         Chem.rdchem.BondType.TRIPLE, \
                         Chem.rdchem.BondType.AROMATIC)]

        e_t.append(int(d['IsConjugated'] == False))
        e_t.append(int(d['IsConjugated'] == True))
        e[(n1, n2)] = e_t

    if len(e) == 0:
        return torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

    edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
    edge_attr = torch.FloatTensor(list(e.values()))
    return edge_index, edge_attr



def mol2graph(mol):
    if mol is None:
        return None
    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    
    feats = chem_feature_factory.GetFeaturesForMol(mol)
    g = nx.DiGraph()

    # Create nodes
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        g.add_node(i,
                   a_type=atom_i.GetSymbol(),
                   a_num=atom_i.GetAtomicNum(),
                   acceptor=0,
                   donor=0,
                   aromatic=atom_i.GetIsAromatic(),
                   hybridization=atom_i.GetHybridization(),
                   num_h=atom_i.GetTotalNumHs(),

                   # 5 more node features
                   ExplicitValence=atom_i.GetExplicitValence(),
                   FormalCharge=atom_i.GetFormalCharge(),
                   ImplicitValence=atom_i.GetImplicitValence(),
                   NumExplicitHs=atom_i.GetNumExplicitHs(),
                   NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
                   )

    for i in range(len(feats)):
        if feats[i].GetFamily() == 'Donor':
            node_list = feats[i].GetAtomIds()
            for n in node_list:
                g.nodes[n]['donor'] = 1
        elif feats[i].GetFamily() == 'Acceptor':
            node_list = feats[i].GetAtomIds()
            for n in node_list:
                g.nodes[n]['acceptor'] = 1


    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j,
                           b_type=e_ij.GetBondType(),
                           
                           IsConjugated=int(e_ij.GetIsConjugated()),
                           )

    node_attr = get_nodes(g)
    edge_index, edge_attr = get_edges(g)

    return node_attr, edge_index, edge_attr




def seqs2int(target):
    VOCAB_PROTEIN = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
    				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
    				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
    				"U": 19, "T": 20, "W": 21, 
    				"V": 22, "Y": 23, "X": 24, 
    				"Z": 25 }
    result = [VOCAB_PROTEIN[s] for s in target]
    return result




def minMaxNormalize(Y, Y_min=None, Y_max=None):
    if Y_min is None:
        Y_min = np.min(Y)
    if Y_max is None:
        Y_max = np.max(Y)
    normalize_Y = (Y - Y_min) / (Y_max - Y_min)
    return normalize_Y


def denseAffinityRefine(adj, k):
    refine_adj = np.zeros_like(adj)
    indexs1 = np.tile(np.expand_dims(np.arange(adj.shape[0]), 0), (k, 1)).transpose()
    indexs2 = np.argpartition(adj, -k, 1)[:, -k:]
    refine_adj[indexs1, indexs2] = adj[indexs1, indexs2]
    return refine_adj


def preAffinityData(args, dataDict):
    
    dataName = args['dataName']
    
    num_drug = len(dataDict['drugDict_i2s'])
    num_prot = len(dataDict['protDict_i2s'])
    
    adj_ori = np.zeros((num_drug, num_prot))
    
    num_tra = len(dataDict['index_trai'])
    for sample_i in range(num_tra):
        tra_index = dataDict['index_trai'][sample_i]
        drug_id = dataDict['drug_id_ls'][tra_index]
        prot_id = dataDict['prot_id_ls'][tra_index]
        label = dataDict['label'][tra_index]
        adj_ori[drug_id, prot_id] = label
    
    if False:
        sys.exit()
    elif dataName == 'human':
        adj_norm = adj_ori  
    elif dataName == 'celegans':
        adj_norm = adj_ori  
    elif dataName == 'davis':
        adj_ori[adj_ori != 0] -= 5
        adj_norm = minMaxNormalize(adj_ori, 0)
    elif dataName == 'kiba':
        adj_norm = minMaxNormalize(adj_ori)
    elif dataName == 'toxcast':
        adj_ori[adj_ori != 0] -= 1
        adj_norm = minMaxNormalize(adj_ori, 0)
    elif dataName == 'metz':
        adj_ori[adj_ori != 0] -= 4
        adj_norm = minMaxNormalize(adj_ori, 0)
    else:
        sys.exit(f'wrong dataName: {dataName}')
        
    dataDict['adj_norm'] = adj_norm
    
    args['num_drug'] = num_drug
    args['num_prot'] = num_prot
    return


def getData(args):
    dataName = args['dataName']
    dataPath = f'../data/{dataName}/raw/data.csv'
    data = pd.read_csv(dataPath)
    
    random_state = args['random_state']
    target_len = args['target_len']
    split_type = args['split_type']
    
    drugLs = data['compound_iso_smiles'].values.tolist()
    protLs = data['target_sequence'].values.tolist()
    
    drugNum = len(drugLs)
    protNum = len(protLs)
    
    drugDict_s2i = dict()
    i = 0
    for d in drugLs:
        if d not in drugDict_s2i.keys():
            drugDict_s2i[d] = i
            i += 1
    drugDict_i2s = {value: key for key, value in drugDict_s2i.items()}
    drug_id_ls = [drugDict_s2i[temp] for temp in drugLs]
    
    drug_graph = dict()
    for smi_id, smile in drugDict_i2s.items():
        mol = Chem.MolFromSmiles(smile)
        g = mol2graph(mol)
        drug_graph[smi_id] = g
    
    
    protDict_s2i = dict()
    i = 0
    for p in protLs:
        if p not in protDict_s2i.keys():
            protDict_s2i[p] = i
            i += 1
    protDict_i2s = {value: key for key, value in protDict_s2i.items()}
    prot_id_ls = [protDict_s2i[temp] for temp in protLs]
    
    prot_int_dict = dict()
    for prot_id, sequence in protDict_i2s.items():
        target = seqs2int(sequence)
        
        if len(target) < target_len:
            target = np.pad(target, (0, target_len- len(target)))
        else:
            target = target[:target_len]
            
        prot_int_dict[prot_id] = list(target)
        
    label = data['affinity'].values.tolist()
    
    if False:
        sys.exit()
    elif split_type == 'warm':
        index_all = range(data.shape[0])
        index_trva, index_test = train_test_split(index_all, 
                                                  test_size=1./6, 
                                                  random_state=random_state)
        index_trai, index_vali = train_test_split(index_trva, 
                                                  test_size=1./5, 
                                                  random_state=random_state)
    elif split_type == 'cold-drug':
        index_all = range(drugNum)
        index_trva_d, index_test_d = train_test_split(index_all, 
                                                      test_size=1./6, 
                                                      random_state=random_state)
        index_trai_d, index_vali_d = train_test_split(index_trva_d, 
                                                      test_size=1./5, 
                                                      random_state=random_state)
        index_trai = []
        index_vali = []
        index_test = []
        for i in range(data.shape[0]):
            drug_i = data.loc[i, 'compound_iso_smiles']

            drug_ind = drugDict_s2i[drug_i]
            if False:
                sys.exit()
            elif drug_ind in index_trai_d:
                index_trai.append(i)
            elif drug_ind in index_vali_d:
                index_vali.append(i)
            elif drug_ind in index_test_d:
                index_test.append(i)
            else:
                sys.exit()
    elif split_type == 'cold-prot':
        index_all = range(protNum)
        index_trva_p, index_test_p = train_test_split(index_all, 
                                                      test_size=1./6, 
                                                      random_state=random_state)
        index_trai_p, index_vali_p = train_test_split(index_trva_p, 
                                                      test_size=1./5, 
                                                      random_state=random_state)
        index_trai = []
        index_vali = []
        index_test = []
        for i in range(data.shape[0]):
            prot_i = data.loc[i, 'target_sequence']
            
            prot_ind = protDict_s2i[prot_i]
            if False:
                sys.exit()
            elif prot_ind in index_trai_p:
                index_trai.append(i)
            elif prot_ind in index_vali_p:
                index_vali.append(i)
            elif prot_ind in index_test_p:
                index_test.append(i)
            else:
                sys.exit()
    elif split_type == 'cold-drug-prot':
        index_all_d = range(drugNum)
        index_trva_d, index_test_d = train_test_split(index_all_d, 
                                                      test_size=1./6, 
                                                      random_state=random_state)
        index_trai_d, index_vali_d = train_test_split(index_trva_d, 
                                                      test_size=1./5, 
                                                      random_state=random_state)
        
        index_all_p = range(protNum)
        index_trva_p, index_test_p = train_test_split(index_all_p, 
                                                      test_size=1./6, 
                                                      random_state=random_state)
        index_trai_p, index_vali_p = train_test_split(index_trva_p, 
                                                      test_size=1./5, 
                                                      random_state=random_state)
        index_trai = []
        index_vali = []
        index_test = []
        for i in range(data.shape[0]):
            drug_i = data.loc[i, 'compound_iso_smiles']
            prot_i = data.loc[i, 'target_sequence']
            
            drug_ind = drugDict_s2i[drug_i]
            prot_ind = protDict_s2i[prot_i]
            if False:
                sys.exit()
            elif drug_ind in index_trai_d and prot_ind in index_trai_p:
                index_trai.append(i)
            elif drug_ind in index_vali_d and prot_ind in index_vali_p:
                index_vali.append(i)
            elif drug_ind in index_test_d and prot_ind in index_test_p:
                index_test.append(i)
            else:
                pass
    else:
        sys.exit()
    
    dataDict = dict()
    
    dataDict['index_trai'] = index_trai
    dataDict['index_vali'] = index_vali
    dataDict['index_test'] = index_test
    
    dataDict['drugDict_s2i'] = drugDict_s2i
    dataDict['drugDict_i2s'] = drugDict_i2s
    dataDict['drug_id_ls'] = drug_id_ls
    dataDict['drug_graph'] = drug_graph
    
    dataDict['protDict_s2i'] = protDict_s2i
    dataDict['protDict_i2s'] = protDict_i2s
    dataDict['prot_id_ls'] = prot_id_ls
    dataDict['prot_int_dict'] = prot_int_dict
    
    dataDict['label'] = label
    
    preAffinityData(args, dataDict)
    
    return dataDict






class GNNDataset(InMemoryDataset):
    def __init__(self, root, dataDict, split='trai', 
                 transform=None, pre_transform=None):
        
        super().__init__(root, transform, pre_transform)
        
        self.dataDict = dataDict
        self.split = split
        
        data_list = self.process_data()
        self.data, self.slices = self.collate(data_list)
        return

    def process_data(self):
        dataDict = self.dataDict
        
        key = f'index_{self.split}'
        index = dataDict[key]

        data_list = []
        for ind in tqdm(index):
            
            drug_idx = dataDict['drug_id_ls'][ind]
            prot_idx = dataDict['prot_id_ls'][ind]

            graph_data = dataDict['drug_graph'][drug_idx]
            x, edge_index, edge_attr = graph_data
            
            protein = torch.tensor(dataDict['prot_int_dict'][prot_idx], 
                                   dtype=torch.long)

            y = torch.FloatTensor([dataDict['label'][ind]])

            data = Data(x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        protein=protein,
                        y=y,
                        drug_id = torch.LongTensor([drug_idx]),
                        prot_id = torch.LongTensor([prot_idx]))

            data_list.append(data)
        return data_list

    def _download(self):
        pass

    def _process(self):
        pass



    
    
    
    