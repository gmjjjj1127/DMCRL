import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool



class DrugEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        node_in = args['atom_feat_dim']
        hidden_dim = args['emb_dim']
        
        dropout_rate = args['dropout_rate']
        
        self.gcn1 = GCNConv(node_in, hidden_dim)
        self.gcn1_act = nn.LeakyReLU(negative_slope=0.01)
        self.gcn1_dropout = nn.Dropout(dropout_rate)
        
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2_act = nn.LeakyReLU(negative_slope=0.01)
        self.gcn2_dropout = nn.Dropout(dropout_rate)
        return
    def forward(self, drug_x, edge_index, batch):
        drug_x = self.gcn1(drug_x, edge_index)
        drug_x = self.gcn1_act(drug_x)
        drug_x = self.gcn1_dropout(drug_x)

        drug_x = self.gcn2(drug_x, edge_index)
        drug_x = self.gcn2_act(drug_x)
        drug_x = self.gcn2_dropout(drug_x)
        
        drug_x = global_mean_pool(drug_x, batch)  
        return drug_x


class ResidualConvBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        if self.use_residual:
            out = out + x
        return out

class ProtEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        hidden_dim = args['emb_dim']
        dropout_rate = args['dropout_rate']
        use_residual = args.get('use_residual', True)

        
        self.embedding = nn.Embedding(25 + 1, hidden_dim, padding_idx=0)

        
        self.block1 = ResidualConvBlock(hidden_dim, 
                                        dropout=dropout_rate, 
                                        use_residual=use_residual)
        self.block2 = ResidualConvBlock(hidden_dim, 
                                        dropout=dropout_rate, 
                                        use_residual=use_residual)

    
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, protein_seq):
       
        x = self.embedding(protein_seq)       
        x = x.permute(0, 2, 1)               
        x = self.block1(x)                    
        x = self.block2(x)                   

        x = self.pool(x).squeeze(-1)        
        return x


class ProtEncoder_old(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        dropout_rate = args['dropout_rate']
        hidden_dim = args['emb_dim']

        
        self.protein_emb = nn.Embedding(25+1, hidden_dim, padding_idx=0)
        self.protein_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.prot_dropout = nn.Dropout(dropout_rate)
        return
    def forward(self, protein_seq):
    
        prot_x = self.protein_emb(protein_seq)  
        prot_x = prot_x.permute(0, 2, 1) 
        prot_x = self.protein_conv(prot_x).squeeze(-1)  
        prot_x = self.prot_dropout(prot_x)
        return prot_x


class AffinityGraphEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        in_dim = args['graph_in_dim']
        hidden_dim = args['emb_dim']
        dropout_rate = args['dropout_rate']
        
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.act2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        return
    
    def forward(self, x, edge_index, edge_weight):
        x = self.gcn1(x, edge_index, edge_weight)
        x = self.act1(x)
        x = self.dropout1(x)
        
        x = self.gcn2(x, edge_index, edge_weight)
        x = self.act2(x)
        x = self.dropout2(x)
        return x  



class SimCLRRegularizer(nn.Module):
    def __init__(self, temperature=0.5):
        super(SimCLRRegularizer, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
      
        batch_size = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        representations = torch.cat([z1, z2], dim=0)  # [2B, D]
        similarity_matrix = torch.matmul(representations, representations.T)  # [2B, 2B]
        similarity_matrix = similarity_matrix / self.temperature

        labels = torch.cat([torch.arange(batch_size) + batch_size,
                            torch.arange(batch_size)], dim=0).to(z1.device)  # 正样本索引
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
    
    

class Predictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        hidden_dim = args['emb_dim']
        
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1))
        return
    def forward(self, drug_rep, prot_rep):
        drug_prot = torch.cat([drug_rep, prot_rep], dim=1)
        
        y_pred = self.classifier(drug_prot)  

        y_pred = y_pred.squeeze(-1)
        return y_pred


class DrugProteinGNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        temperature = args['temperature']
        
        self.drugEncoder = DrugEncoder(args)

        self.protEncoder = ProtEncoder(args)
        
        self.graphEncoder = AffinityGraphEncoder(args)
        
        self.Regularizer_d = SimCLRRegularizer(temperature)
        self.Regularizer_p = SimCLRRegularizer(temperature)
        
        self.predictor = Predictor(args)
        return

    def forward(self, drug_x, edge_index, batch, 
                protein_seq,
                drug_ids,
                prot_ids,
                affinity_graph):
        args = self.args
        num_drug = args['num_drug']
        
        
        drug_x = self.drugEncoder(drug_x, edge_index, batch)

        target_len = args['target_len']
        protein_seq = protein_seq.view(-1, target_len)
        prot_x = self.protEncoder(protein_seq)

        
        graph_x = affinity_graph.x
        edge_index = affinity_graph.edge_index
        
        edge_weight = affinity_graph.adj[affinity_graph.edge_index[0], 
                                         affinity_graph.edge_index[1]]
        drug_prot_graph = self.graphEncoder(graph_x, edge_index, edge_weight)
        
        drug_g = drug_prot_graph[drug_ids]
        prot_g = drug_prot_graph[prot_ids + num_drug]
        
        loss_d = self.Regularizer_d(drug_x, drug_g)
        loss_p = self.Regularizer_p(prot_x, prot_g)
        
        drug_rep = torch.cat([drug_x, drug_g], dim=1)
        prot_rep = torch.cat([prot_x, prot_g], dim=1)
        
        
        y_pred = self.predictor(drug_rep, prot_rep)
        
        output = dict()
        output['y_pred'] = y_pred
        
        output['loss_d'] = loss_d
        output['loss_p'] = loss_p
        
        return output


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    