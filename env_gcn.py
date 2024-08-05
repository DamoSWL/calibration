import os.path as osp
import numpy as np
import random
import os
from sys import argv
import torch
from scipy.sparse import csr_matrix
from collections import defaultdict
import scipy.sparse as sp
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging
from time import time
import scipy.sparse as sp
from utils.util import data_loader, data_loader_OOD, sparse_mx_to_torch_sparse_tensor,manual_seed,accuracy
from utils.normalization import fetch_normalization
import math
import torch.optim as optim
from collections import defaultdict
from sklearn.metrics import roc_auc_score,average_precision_score
import pickle
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
from datetime import datetime
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math
import time
from copy import deepcopy

from models import GAT, SpGAT


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = torch.exp(logits)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece +=  torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def update_bin_cof(self, logits, labels):
        bin_cof = []
        epsilon = 0.015
        softmaxes = torch.exp(logits)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):

            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())

            prop_in_bin = in_bin.float().mean()
            cof = 0
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                if accuracy_in_bin > avg_confidence_in_bin + epsilon:
                    cof = -1
                elif accuracy_in_bin < avg_confidence_in_bin - epsilon:
                    cof = 1
                else:
                    cof = 0
            bin_cof.append(cof)


        bin_cof = np.array(bin_cof)
        return bin_cof


# class GraphConvolution(nn.Module):


#     def __init__(self, input_dim, output_dim,
#                  dropout=0.,
#                  use_bias=True,
#                  activation = F.relu):
#         super(GraphConvolution, self).__init__()


#         self.dropout = dropout
#         self.use_bias = use_bias
#         self.activation = activation
#         self.input_dim = input_dim
#         self.output_dim = output_dim

#         self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
#         self.bias = None
#         if self.use_bias:
#             self.bias = nn.Parameter(torch.zeros(output_dim))
#         self.init_model()

#     def init_model(self):
#         # torch.nn.init.kaiming_uniform_(self.weight)
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.use_bias:
#             self.bias.data.uniform_(-stdv, stdv)
#             # torch.nn.init.zeros_(self.bias)

#     def forward(self,x,adj):

#         support = torch.mm(x, self.weight)
#         output = torch.sparse.mm(adj, support)
#         if self.use_bias:
#             output += self.bias
#         return output


# class GCN(nn.Module):


#     def __init__(self, input_dim, output_dim, hidden_dim, dropout):
#         super(GCN, self).__init__()

#         self.input_dim = input_dim # 1433
#         self.output_dim = output_dim


#         self.gcn1 = GraphConvolution(self.input_dim, hidden_dim,
#                                                      activation=F.relu,
#                                                      dropout=dropout)

#         self.gcn2 = GraphConvolution(hidden_dim, output_dim,
#                                                      activation=F.relu,
#                                                      dropout=dropout)

#     def init_model(self):
        
#         self.gcn1.init_model()
#         self.gcn2.init_model()

#     def forward(self, x, adj):
#         h = F.relu(self.gcn1(x, adj))
#         x = F.dropout(x, 0.5, training=self.training)
#         logits = self.gcn2(h,adj)
#         x = F.log_softmax(logits, dim=1)

#         return x


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
  
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
  

    def init_model(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()


    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)



class gcn_env_GCN(object):
    def __init__(self,dataset='cora',datapath='data',task_type='semi',args=None):
        
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')       
        self.dataset = dataset
        self.model_path = args.model_path

        self.adj, _, self.features,_,self.labels,self.idx_train,self.idx_val, \
        self.idx_test,self.degree,self.learning_type,self.idx_test_ood, \
        self.idx_test_id,self.test_mask_id,self.test_mask_ood = data_loader_OOD(dataset, datapath, "NoNorm", False, task_type)

        # weighted_adj= self.weighted_adj(1.0)
        # self.adj = sp.coo_matrix(weighted_adj)
        self.seed = args.seed
          
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor(self.labels)
        self.idx_train = torch.LongTensor(self.idx_train)
        self.idx_val = torch.LongTensor(self.idx_val)
        self.idx_test = torch.LongTensor(self.idx_test)
        self.idx_test_ood = torch.LongTensor(self.idx_test_ood) 
        self.idx_test_id = torch.LongTensor(self.idx_test_id) 
        self.test_mask_id = torch.BoolTensor(self.test_mask_id) 
        self.test_mask_ood = torch.BoolTensor(self.test_mask_ood)

        self.features = self.features.to(self.device)
        self.labels = self.labels.to(self.device)
        self.idx_train = self.idx_train.to(self.device)
        self.idx_val= self.idx_val.to(self.device)
        self.idx_test = self.idx_test.to(self.device)
        self.idx_test_ood = self.idx_test_ood.to(self.device)
        self.idx_test_id = self.idx_test_id.to(self.device)
        self.test_mask_id =self.test_mask_id.to(self.device)
        self.test_mask_ood = self.test_mask_ood.to(self.device)

     
        self.rng = np.random.default_rng(self.seed)
        self.nfeat = self.features.shape[1]
        self.ndata = self.features.shape[0]
        self.nclass = int(self.labels.max().item() + 1)

 
        self.adj = sp.csr_matrix(self.adj) + sp.identity(self.adj.shape[0])
        self.adj = self.adj.astype(np.float64)

        self.candidate_adj = defaultdict(list)
        self.candidate_adj_s = defaultdict(list)
        self.candidate_node_num = 10
        self.candidate_adj_num = 8
        self.candidate_node = None
        self.init_candidate_node()
        # self.init_candidate_adj()

        self.agent_memory_size = self.candidate_node_num * self.candidate_adj_num * 10

        self.bin_cof = np.zeros(10)

        self.alpha = 0.90
        self.policy = None
        self.args = args

        self.ECEFunc = ECELoss()


        self.model = GCN(nfeat=self.nfeat, nclass=self.nclass, nhid=self.args.hidden,dropout=0.5)
      
        self.model = self.model.to(self.device)
        self.gcn_optimizer = optim.Adam(self.model.parameters(), lr=1e-2,weight_decay=5e-4)
       
        self.test_times = 10


    def init_model(self):
        self.model.init_model()

        self.gcn_optimizer.zero_grad()

    def init_candidate_node(self):
        target_nodes = self.idx_val

        self.candidate_node = self.rng.permutation(target_nodes.cpu().numpy())[:self.candidate_node_num]
        self.candidate_node = torch.from_numpy(self.candidate_node).to(self.device)

    def init_candidate_adj(self):
        self.candidate_adj = defaultdict(list)
        self.candidate_adj_s = defaultdict(list)
        self.get_candidate_adj(self.candidate_node,self.candidate_adj,self.candidate_adj_num-1)
        

    def init_init_candidate_adj_s(self):
        self.candidate_adj_s = deepcopy(self.candidate_adj)

    def load_adj(self):
        with np.load(f'/localtmp/weili/Project/GraphOOD/DDPG/adj/{self.dataset }_adj.npz') as data:
            self.adj = sp.coo_matrix((data['value'],(data['row'],data['col'])),shape=data['shape'])
    

    def normalize_adj(self,adj):
        """Symmetrically normalize adjacency matrix."""
        # adj = sp.coo_matrix(adj)
        # rowsum = np.array(adj.sum(1)) # D
        # d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
        # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^-0.5

        # adj = sp.csr_matrix(adj) + sp.eye(adj.shape[0])
        adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()


    def get_candidate_adj(self,idx,candidate_adj,count):  
        self.adj = sp.coo_matrix(self.adj)
        adj = np.vstack((self.adj.row,self.adj.col)) 

        for i in idx:
            tmp_adj = []

            i = i.item()        
            mask = adj[0] == i
            one_hop_nodes = adj[1,mask]
            one_hop_nodes = one_hop_nodes[one_hop_nodes != i]

            one_hop_nodes = self.rng.permutation(one_hop_nodes)[:(count+1)//4]
            candidate_adj[i] += [(i,t) for t in one_hop_nodes]

            tmp_len = len(candidate_adj[i])
            
            for j in one_hop_nodes:
                mask = adj[0] == j
                two_hop_nodes = adj[1,mask]
                two_hop_nodes = two_hop_nodes[two_hop_nodes != j]
                
                tmp_adj.extend([(j,k) for k in two_hop_nodes if k != i and (k,j) not in tmp_adj])
 
            if len(tmp_adj) >= count-tmp_len:
                candidate_adj[i] += self.rng.permutation(tmp_adj)[:count-tmp_len].tolist()
            else:
                candidate_adj[i] += self.rng.choice(tmp_adj,count-tmp_len).tolist()
    
            candidate_adj[i].insert(0,(i,i))



    def weighted_adj(self, adj_weight):
        forbidden_idx = set(self.idx_test_ood)
        adj_size = self.adj.shape[0]
        dense_adj = self.adj.todense().A.astype('float64')
        for row in range(adj_size):
            for col in range(adj_size):
                if (row in forbidden_idx or col in forbidden_idx) and dense_adj[row][col] > 0:
                    dense_adj[row][col] = adj_weight
        weighted_adj = np.asmatrix(dense_adj)
        return weighted_adj


    def get_adj_num(self,index):
        node = self.candidate_node[index].item()
        return len(self.candidate_adj[node])


    def reset(self,node_index):

        state = []
        node = self.candidate_node[node_index]
        for i in node:
            i = i.item()
            node_feature_1 = self.features[self.candidate_adj[i][0][0]]
            node_feature_2 = self.features[self.candidate_adj[i][0][1]]
            edge_feature = (node_feature_1 + node_feature_2)/2.0  
        
            state.append(edge_feature)
        
        # state = self.features[node]

        state = torch.vstack(state)
        state = state.to(self.device)

        return state

    def get_val_nodes(self,node,hop=1):
        self.adj = sp.coo_matrix(self.adj)
        adj = np.vstack((self.adj.row,self.adj.col)) 

        target_nodes = torch.cat([self.idx_train,self.idx_val])
        target_nodes = target_nodes.cpu().numpy().tolist()

        val_nodes = []
        if hop >= 0:
            if node in target_nodes:
                val_nodes.append(node)

            mask = adj[0] == node
            neighbor_nodes = adj[1,mask]
            neighbor_nodes = neighbor_nodes[neighbor_nodes != node]
            for adj_node in neighbor_nodes:
                if adj_node in target_nodes:
                    val_nodes.append(adj_node)


        return val_nodes


    def obtain_reward(self,adj,node,node_1,node_2):
        self.model.eval()
        adj = self.preprocess_adj(adj)  
 
        val_nodes = []

        # for t in node_1:
        #     val_nodes.extend(self.get_val_nodes(t))
        # for t in node_2:
        #     val_nodes.extend(self.get_val_nodes(t))   

        # val_nodes = list(set(val_nodes))
        val_nodes = node

        with torch.no_grad():
       
            output = self.model(self.features,adj)
            output_val = output[val_nodes]


        prob_val = torch.exp(output_val)
        pred_val = prob_val.argmax(dim=-1)
        labels_val = self.labels[val_nodes]
        conf_val = torch.gather(prob_val, dim=-1, index=pred_val.unsqueeze(-1)).squeeze()

        # print(prob_val)
        # print(conf_val)


        acc = [1 if i > 0 else 0 for i in (pred_val==labels_val).to(torch.uint8)]
        acc = np.array(acc)

        # print(acc)

        entropy = -1.0 * prob_val * (torch.log(prob_val.squeeze()) / torch.log(torch.tensor([self.nclass],dtype=float,device=self.device)))
        entropy = entropy.sum(axis=-1)
        entropy = entropy.cpu().numpy()

        # print(entropy)

        sign = np.ones((len(val_nodes)))
        bins = np.linspace(0.1, 1, 10)
        index = np.digitize(conf_val.cpu().numpy()-1e-5, bins)
 
        sign = self.bin_cof[index]

        # print(sign)

        alpha = np.ones((len(val_nodes)))
        beta = np.ones((len(val_nodes)))

        # print(f'acc {acc}')
        # print(f'sign {sign}')

        # alpha[(acc == 1) & (sign == 1)] = -0.01
        # alpha[(acc == 1) & (sign == -1)] = -1.0
        # alpha[(acc == 1) & (sign == 0)] = 0
        # alpha[acc == 0] = 1.0

        # beta[(acc == 1) & (sign == 1)] = -1.0
        # beta[(acc == 1) & (sign == -1)] = 1.0
        # beta[(acc == 1) & (sign == 0)] = 0
        # beta[acc == 0] = 1.0

        alpha[(acc == 1) & (sign == 1)] = 1.0
        alpha[(acc == 1) & (sign == -1)] = -1.0
        alpha[(acc == 1) & (sign == 0)] = 0
        alpha[acc == 0] = 1.0

        beta[(acc == 1) & (sign == 1)] = 1.0
        beta[(acc == 1) & (sign == -1)] = 1.0
        beta[(acc == 1) & (sign == 0)] = 0
        beta[acc == 0] = 1.0


        # print(f'alpha {alpha}')
        # print(f'beta {beta}')
 
        reward = acc +  alpha * np.power(entropy,beta)

        # reward = acc
        # reward = alpha * np.power(entropy,beta)
        # print(reward)
        # exit()
       

        reward = torch.as_tensor(reward).float()

        return reward


    def update_adj(self):  
        self.adj = sp.coo_matrix(self.adj)

        edge_index =  np.vstack((self.adj.row, self.adj.col)).astype(np.int32)

        self.adj = sp.csr_matrix(self.adj)
  
        batch = 2000
        index = 0

        labels = self.labels.cpu().numpy()

        actions_plot = []
        actions_test = []

        previous_state = 0

        s_time = time.time()

        while index < edge_index.shape[1]:
            node_feature_1 = self.features[edge_index[0,index:index+batch]]
            node_feature_2 = self.features[edge_index[1,index:index+batch]]
            edge_feature = (node_feature_1 + node_feature_2)/2.0  
            # if hasattr(previous_state,'shape'):
            #     row = min(edge_feature.shape[0],previous_state.shape[0])
            #     state = self.alpha * edge_feature[:row] + (1-self.alpha) * previous_state[:row]
            # else:
            state = edge_feature
            actions = self.policy.select_action(state,False)
            actions = actions.cpu().numpy() 
            

            # actions_plot.extend(actions.copy().tolist())
            actions_test.extend(actions.copy().tolist())

            # node_1 = edge_index[0,index:index+batch]
            # node_2 = edge_index[1,index:index+batch]

            # mask_1 = [1 if labels[i] >= 0 else 0 for i in node_1]
            # mask_2 = [1 if labels[i] >= 0 else 0 for i in node_2]
            # mask_1 = np.array(mask_1)
            # mask_2 = np.array(mask_2)
            # mask = mask_1 * mask_2
            # actions = [1 if i > 0 else 0.5 for i in mask]
            # actions_test.extend(actions)


            # mask_1 = [1 if labels[i] >= 0 else -1 for i in node_1]
            # mask_2 = [1 if labels[i] >= 0 else -1 for i in node_2]
            # mask_1 = np.array(mask_1)
            # mask_2 = np.array(mask_2)
            # mask = mask_1 + mask_2
            # actions = [1 if np.abs(i) > 0 else 0.5 for i in mask]
            # actions_test.extend(actions)
            
            index += batch
            previous_state = state

        actions_test = np.array(actions_test) 

        actions_test = actions_test / actions_test.max()
        # actions_test = actions_test.clip(0.2,1.0)

        self.adj[edge_index[0,:],edge_index[1:]] = actions_test
        self.adj = sp.coo_matrix(self.adj)

        print(time.time()-s_time)

        np.savez(f'adj/{self.dataset}_adj',value = self.adj.data, row=self.adj.row,col=self.adj.col,shape=self.adj.shape)

        logging.info(f'edge {self.adj}')
   
        logging.info(f'edge max {actions_test.max()} edge min {actions_test.min()} edge mean {actions_test.mean()} edge std {actions_test.std()}')



    # def step(self, state, action, index):

    #     row = self.candidate_node[index[0]].tolist()
    #     col = index[1] 

    #     node_1 = []
    #     node_2 = []
    #     for i in row:
    #         node_1.append(self.candidate_adj[i][col][0])
    #         node_2.append(self.candidate_adj[i][col][1])
 
    #     self.adj = sp.csr_matrix(self.adj)
    #     self.adj[node_1,node_2] = action.cpu().numpy()
    #     reward = self.obtain_reward(self.adj, row, node_1,node_2)
    
  
    #     col += 1
    #     done = 0
    #     if col >= self.candidate_adj_num:
    #         col = self.candidate_adj_num - 1
    #         done = 1
        
    #     node_1, node_2 = self.candidate_adj[row][col]
    #     edge_feature = (self.features[node_1] + self.features[node_2])/2
    #     next_state = self.alpha * edge_feature + (1-self.alpha) * state
    #     # next_state = F.normalize(next_state,p=1,dim=-1)
    #     done = torch.as_tensor([done]*len(row),dtype=torch.bool)
       
    #     return next_state, reward, done



    def step(self, state, action, index):

        row = self.candidate_node[index[0]].tolist()
        col = index[1] 

        node_1 = []
        node_2 = []
        for i in row:
            f_node = self.candidate_adj_s[i].pop()
            node_1.append(f_node[0])
            node_2.append(f_node[1])

 
        self.adj = sp.csr_matrix(self.adj)
        self.adj[node_1,node_2] = action.cpu().numpy()
        reward = self.obtain_reward(self.adj, row, node_1,node_2)
    
        col += 1
        done = 0

        if col >= self.candidate_adj_num:
            done = 1
        else:
            node_1 = []
            node_2 = []
            self.sort_candidate_adj()
            for i in row:
                f_node = self.candidate_adj_s[i][0]
                node_1.append(f_node[0])
                node_2.append(f_node[1])
            
        edge_feature = (self.features[node_1] + self.features[node_2])/2
        next_state = self.alpha * edge_feature + (1-self.alpha) * state
        # next_state = F.normalize(next_state,p=1,dim=-1)
        done = torch.as_tensor([done]*len(row),dtype=torch.bool)
       
        return next_state, reward, done


    def sort_candidate_adj(self):
        self.model.eval()

        adj = self.preprocess_adj(self.adj)
    
        with torch.no_grad():
            output = self.model(self.features, adj)

        for t_node in self.candidate_node:
            t_node = t_node.item()
            if len(self.candidate_adj_s[t_node]) > 0:
                score = [0.5*(F.kl_div(output[j[0]],torch.exp(output[t_node])).item() + F.kl_div(output[j[1]],torch.exp(output[t_node])).item()) for j in self.candidate_adj_s[t_node]]
                score = np.array(score)
                index = np.argsort(score)

                self.candidate_adj_s[t_node] = np.array(self.candidate_adj_s[t_node])
                self.candidate_adj_s[t_node] = self.candidate_adj_s[t_node][index]
                self.candidate_adj_s[t_node] = self.candidate_adj_s[t_node].tolist()
        


    def preprocess_adj(self, adj):
        r_adj = self.normalize_adj(adj)
        r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()  
        r_adj = r_adj.to(self.device)
        return r_adj


    def train(self,adj,total_epoch,save=False,last=True):

        adj = self.preprocess_adj(adj)

        self.model.train()

        result = 0
        best_result = 100

        bad_counter = 0
        patience = 100

        for epoch in range(total_epoch):

            output = self.model(self.features,adj)
            loss = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
            loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
         
            self.gcn_optimizer.zero_grad()
            loss.backward()
            self.gcn_optimizer.step()

            train_acc = accuracy(output[self.idx_train].detach(), self.labels[self.idx_train].detach()) 
            val_acc = accuracy(output[self.idx_val].detach(), self.labels[self.idx_val].detach())
            # logging.info(f'loss {loss} train acc {train_acc} val acc {val_acc}')

            if not last:
                if epoch % 1 == 0:
                    result = loss_val
                    if result < best_result:
                        best_result = result
                        bad_counter = 0
                        if save:
                            path = Path(self.model_path)
                            if not path.exists():
                                path.mkdir()
                            torch.save({'model_state_dict': self.model.state_dict()},str(path / 'GCN_best_model.pth'))
                    else:
                        bad_counter += 1

                    if bad_counter >= patience:
                        break


            if last:
                if epoch % 1 == 0:
                    if save:
                        path =Path(self.model_path)
                        if not path.exists():
                            path.mkdir()

                        # logging.info(f'save the GCN model')
                        torch.save({'model_state_dict': self.model.state_dict()},str(path / 'GCN_best_model.pth'))

      
            # if epoch % 1 == 0:  
            #     result = val_acc
            #     if result > best_result:
            #         best_result = result
            #         if save:
            #             path = Path(self.model_path)
            #             if not path.exists():
            #                 path.mkdir()

            #             # logging.info(f'save the GCN model')
            #             torch.save({'model_state_dict': self.model.state_dict()},str(path / 'GCN_best_model.pth'))



    def validate(self,adj):
        self.model.eval()

        path =Path(self.model_path)
        checkpoint = torch.load(str(path / 'GCN_best_model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])

        adj = self.preprocess_adj(adj)
    
        with torch.no_grad():
     
            output = self.model(self.features, adj)
            loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
        
        acc = accuracy(output[self.idx_val], self.labels[self.idx_val])
        ECE = self.ECEFunc(output[self.idx_val], self.labels[self.idx_val])


        return acc.item(), ECE.item(), loss_val.item()


    def update_bin_cof(self,adj):
        self.model.eval()
        adj = self.preprocess_adj(adj)

        with torch.no_grad():
   
            output = self.model(self.features, adj)

        self.bin_cof = self.ECEFunc.update_bin_cof(output[self.idx_val],self.labels[self.idx_val])

        logging.info(f'bin {self.bin_cof}')
 



    def test(self,adj):

        average_acc = 0
        average_ece = 0

        adj = self.preprocess_adj(adj)

        for i in range(self.test_times): 
            test_idx_i = np.random.choice(self.idx_test.cpu().numpy(), size=1000, replace=False)
            test_idx_filtered = [ele for ele in test_idx_i if ele in self.idx_test_id.cpu().numpy()]
            test_idx_filtered = torch.tensor(test_idx_filtered)
            test_idx_filtered = test_idx_filtered.to(self.device)

            acc, ece = self.test_single(adj,test_idx_filtered,i)
            average_acc += acc.item()
            average_ece += ece.item()


        return average_acc/self.test_times, average_ece/self.test_times


    def test_single(self,adj,idx,i):
        self.model.eval()

        path = Path(self.model_path)
        checkpoint = torch.load(str(path / 'GCN_best_model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])


        with torch.no_grad():
            output = self.model(self.features, adj)
        
        loss = F.nll_loss(output[idx], self.labels[idx])
        acc = accuracy(output[idx], self.labels[idx])

        ECE = self.ECEFunc(output[idx], self.labels[idx])

        # logging.info(f'test results: loss {loss} acc {acc} ece {ECE}')

        # if i == 0:
        #     self.plot_acc_calibration(idx,output,self.labels)

    

        return acc, ECE
   
    
    def plot_acc_calibration(self, idx_test, output, labels, n_bins=10, title=None):
        output = torch.exp(output)
        pred_label = torch.max(output[idx_test], 1)[1]
        p_value = torch.max(output[idx_test], 1)[0]
        ground_truth = labels[idx_test]
        confidence_all, confidence_acc = np.zeros(n_bins), np.zeros(n_bins)
        for index, value in enumerate(p_value):
            #value -= suboptimal_prob[index]
            interval = int(value / (1 / n_bins) -0.0001)
            confidence_all[interval] += 1
            if pred_label[index] == ground_truth[index]:
                confidence_acc[interval] += 1
        for index, value in enumerate(confidence_acc):
            if confidence_all[index] == 0:
                confidence_acc[index] = 0
            else:
                confidence_acc[index] /= confidence_all[index]

        start = np.around(1/n_bins/2, 3)
        step = np.around(1/n_bins, 3)
        plt.figure(figsize=(5, 4))
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams["font.weight"] = "bold"

        plt.bar(np.around(np.arange(start, 1.0, step), 3),
                np.around(np.arange(start, 1.0, step), 3), alpha=0.6, width=0.09, color='lightcoral', label='Expected')
        plt.bar(np.around(np.arange(start, 1.0, step), 3), confidence_acc,
                alpha=0.6, width=0.09, color='dodgerblue', label='Outputs')       
        plt.plot([0,1], [0,1], ls='--',c='k')
        plt.xlabel('Confidence', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.tick_params(labelsize=13)
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        #title = 'Uncal. - Cora - 20 - GCN'
        plt.title(title, fontsize=16, fontweight="bold")
        plt.legend(fontsize=16)
        plt.savefig('images/' + str(self.seed)+ '_' + 'uncal_gcn' + '_' + self.dataset + '.png' , format='png', dpi=300,
                    pad_inches=0, bbox_inches = 'tight')

    