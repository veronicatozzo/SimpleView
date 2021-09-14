import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from all_utils import DATASET_NUM_CLASS

from .deepsets_utils import reshape_x_and_lengths, MySequential, mask_matrix, BatchSetNorm, connect, BatchFeatureNorm, get_norm, InformationPreservingNorm, ReLUNoStats, TanhNoStats


class MLP(nn.Module):
    def __init__(self, in_features, out_features, block_connect="none", block_norm="none", activation=nn.ReLU, sample_size=None,
                normalization_after=True, pipeline='ours'):
        super(MLP, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=(block_norm=="none"))
        if activation is not None and block_norm !='ipn':
            self.activation = activation()
        elif activation is not None and block_norm =='ipn':
            self.activation = ReLUNoStats()
        
        if pipeline == 'he':
            self.norm = get_norm(block_norm, sample_size=sample_size, dim_V=in_features)
        else:
            self.norm = get_norm(block_norm, sample_size=sample_size, dim_V=out_features)
        self.block_connect = block_connect
        self.normalization_after = normalization_after
        self.pipeline = pipeline
        if self.pipeline == 'resnet' or self.pipeline == 'he': 
            self.fc2 = nn.Linear(in_features=out_features + 2 if block_norm == 'ipn' else out_features, out_features=out_features, bias=(block_norm=="none"))
           
            self.norm2 = get_norm(block_norm, sample_size=sample_size, dim_V=out_features)
        
    
    def forward(self, X):
        if self.pipeline == 'resnet':
            O = self.fc(X)
            O = O if getattr(self, 'norm', None) is None else self.norm(O)
            O = O if getattr(self, 'activation', None) is None else self.activation(O)
            O = self.fc2(O)
            O = O if getattr(self, 'norm2', None) is None else self.norm2(O)  
            O = connect(X, O, self.block_connect)
            O = O if getattr(self, 'activation', None) is None else self.activation(O)
        elif self.pipeline == 'he':
            O = X if getattr(self, 'norm', None) is None else self.norm(X)
            O = O if getattr(self, 'activation', None) is None else self.activation(O)
            O = self.fc(O)
            O = O if getattr(self, 'norm2', None) is None else self.norm2(O)
            O = O if getattr(self, 'activation', None) is None else self.activation(O)
            O = self.fc2(O)  
            O = connect(X, O, self.block_connect)
        else:
            O = self.fc(X)
            O = O if getattr(self, 'norm', None) is None else self.norm(O)
            O = O if getattr(self, 'activation', None) is None else self.activation(O)
            O = connect(X, O, self.block_connect)
            
        return O



    
class DeepSetsOurs(nn.Module):
    def __init__(self,  task, dataset, n_inputs=3,n_enc_layers=50,n_dec_layers=3,
                dim_hidden=128,  block_connect="resid", block_norm="ipn", pipeline='resnet'):
        super(DeepSets, self).__init__()
        self.num_outputs = DATASET_NUM_CLASS[dataset]
        self.aggregations = []
        self.block_norm = block_norm
        layers = []
        for j in range(n_enc_layers):
            if j == 0:
                layers.append(MLP(in_features=n_inputs, out_features=dim_hidden, 
                                  activation=nn.ReLU, block_norm=block_norm, 
                                  block_connect=block_connect, 
                                  sample_size=1024, pipeline=pipeline))  
            elif j == n_enc_layers - 1:
                layers.append(MLP(in_features=dim_hidden+2 if block_norm=="ipn" else dim_hidden, 
                                  out_features=dim_hidden, activation=None, 
                                  block_norm=block_norm, block_connect=block_connect,
                                  sample_size=1024,pipeline=pipeline))
            else:
                layers.append(MLP(in_features=dim_hidden+2 if block_norm=="ipn" else dim_hidden, 
                                  out_features=dim_hidden, activation=nn.ReLU, 
                                  block_norm=block_norm, block_connect=block_connect, 
                                  sample_size=1024, pipeline=pipeline))

        layers.append(nn.Linear(dim_hidden+2 if block_norm=="ipn" else dim_hidden, dim_hidden))
        self.enc = nn.Sequential(*layers)
            
        layers = []
        for j in range(n_dec_layers):
            layers.append(MLP(in_features=dim_hidden, out_features=dim_hidden, 
                                  activation=nn.ReLU, block_norm="none", 
                                  block_connect="none", sample_size=1))
        self.dec = nn.Sequential(*layers)
        self.fc = nn.Linear(dim_hidden, self.num_outputs)
    
    def forward(self, pc): # added for compatibility
        pc = pc.cuda()
        out = self.enc(pc)
        out = torch.mean(out, axis=1, keepdim=True) 
        out = self.dec(out)
        out = self.fc(out)
        out = out.squeeze()
        out = {'logit': out}
        
        return out



class MLPTanh(nn.Module):
    def __init__(self, in_features, out_features, sample_size=None, block_norm='ipn', block_connect="resid"):
        super(MLP, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=(block_norm=="none"))
        self.activation = nn.Tanh() #TanhNoStats()
        self.norm = get_norm(block_norm, sample_size=sample_size, dim_V=in_features)
        self.fc2 = nn.Linear(in_features=out_features + 2 if block_norm == 'ipn' else out_features, out_features=out_features, bias=(block_norm=="none"))
        self.norm2 = get_norm(block_norm, sample_size=sample_size, dim_V=out_features)
        self.block_connect = block_connect
    
    def forward(self, X):
        O = X if getattr(self, 'norm', None) is None else self.norm(X)
        O = self.activation(O)
        O = self.fc(O)
        O = O if getattr(self, 'norm2', None) is None else self.norm2(O)
        O = self.activation(O)
        O = self.fc2(O)  
        O = connect(X, O, self.block_connect)      
        return O


class DeepSetsTanh(nn.Module):
    def __init__(self,  task, dataset, n_inputs=3,n_enc_layers=50,n_dec_layers=3,
                dim_hidden=128,  block_connect="resid", block_norm="ipn", pipeline='resnet'):
        super(DeepSetsTanh, self).__init__()
        self.num_outputs = DATASET_NUM_CLASS[dataset]
        self.aggregations = []
        self.block_norm = block_norm
        layers = []
        for j in range(n_enc_layers):
            if j == 0:
                layers.append(MLPTanh(in_features=n_inputs, out_features=dim_hidden, 
                                  activation=nn.ReLU, block_norm=block_norm, 
                                  block_connect=block_connect, 
                                  sample_size=1024))  
            elif j == n_enc_layers - 1:
                layers.append(MLP(in_features=dim_hidden+2 if block_norm=="ipn" else dim_hidden, 
                                  out_features=dim_hidden, activation=None, 
                                  block_norm=block_norm, block_connect=block_connect,
                                  sample_size=1024))
            else:
                layers.append(MLP(in_features=dim_hidden+2 if block_norm=="ipn" else dim_hidden, 
                                  out_features=dim_hidden, activation=nn.ReLU, 
                                  block_norm=block_norm, block_connect=block_connect, 
                                  sample_size=1024))

        layers.append(nn.Linear(dim_hidden+2 if block_norm=="ipn" else dim_hidden, dim_hidden))
        self.enc = nn.Sequential(*layers)
            
        self.dec = nn.Sequential(
           nn.Dropout(p=0.5),
           nn.Linear(dim_hidden,dim_hidden),
           nn.Tanh(),
           nn.Dropout(p=0.5),
           nn.Linear(dim_hidden, 40),
        )
        
    def forward(self, pc): # added for compatibility
        pc = pc.cuda()
        out = self.enc(pc)
        out = torch.mean(out, axis=1, keepdim=True) 
        out = self.dec(out)
        out = self.fc(out)
        out = out.squeeze()
        out = {'logit': out}
        
        return out
        




        


