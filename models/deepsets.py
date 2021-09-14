import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

class PermEqui1_max(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PermEqui1_max, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm, _ = x.max(1, keepdim=True)
        x = self.Gamma(x-xm)
        return x

    
class DeepSets(nn.Module):
    def __init__(self, d_dim=256, x_dim=3):
        super(DeepSets, self).__init__()
        self.d_dim = d_dim
        self.x_dim = x_dim

        self.phi = nn.Sequential(
          PermEqui1_max(self.x_dim, self.d_dim),
          nn.Tanh(),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.Tanh(),
        )
       
        self.ro = nn.Sequential(
           nn.Dropout(p=0.5),
           nn.Linear(self.d_dim, self.d_dim),
           nn.Tanh(),
           nn.Dropout(p=0.5),
           nn.Linear(self.d_dim, 40),
        )
        print(self)

    def forward(self, pc):
        pc = pc.cuda()
        phi_output = self.phi(pc)
        sum_output, _ = phi_output.max(1)
        ro_output = self.ro(sum_output)
        out = {'logit': ro_output}
        return out