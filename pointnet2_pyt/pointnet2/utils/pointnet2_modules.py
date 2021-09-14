from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import etw_pytorch_utils as pt_utils
from etw_pytorch_utils import * 

from pointnet2.utils import pointnet2_utils

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *

    
    

        




        
class SharedMLP(nn.Sequential):

    def __init__(self,
                 args,
                 bn = False,
                 activation=nn.ReLU(inplace=True),
                 preact = False,
                 first = False,
                 name = ""):
        # type: (SharedMLP, List[int], bool, Any, bool, bool, AnyStr) -> None
        super(SharedMLP, self).__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation if (not first or not preact or
                                              (i != 0)) else None,
                    preact=preact))

            
class MLP(nn.Module):
    def __init__(self, in_features, out_features, block_connect="none", block_norm="none", activation=nn.ReLU(inplace=True), sample_size=None, preact=True,  kernel_size = (1, 1),stride = (1, 1),  padding = (0, 0),  dilation = (1, 1),bias = True):
        super(MLP, self).__init__()
        bias = bias and (block_norm=="none")
        self.fc = torch.nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias)
      
        nn.init.kaiming_normal_(self.fc.weight)
        if bias:
            nn.init.constant_(self.fc.bias, 0)
        if activation is not None:
            self.activation = activation
        self.norm = pointnet2_utils.get_norm(block_norm, sample_size=sample_size, dim_V=out_features)
  
        self.block_connect = block_connect
        self.preact = preact


    def forward(self, X):
        ## input [B, C, npoints, nsamples] 
        B, C, npoints, nsamples = X.shape
        O = self.fc(X)
        O = O.permute(0, 2, 3, 1)
        O = torch.flatten(O, start_dim=1, end_dim=2)
        C = O.shape[-1]
        O = O if getattr(self, 'norm', None) is None else self.norm(O)
        O = self.activation(O)
        O = pointnet2_utils.connect(X, O, self.block_connect)
        O = torch.transpose(O, 2, 1).contiguous()
        O = O.reshape(B, C, npoints, nsamples)
        return O
    
    
    
class SeqMLP(nn.Sequential):
    # we get an input which is [B, C, npoints, nsamples] 
    # we want an input which is [B, npoints, nsamples, C]
    def __init__(self, sample_size, n_dims, block_connect="none", block_norm="none",  
                 activation=nn.ReLU(inplace=True), preact=True,
                 first=False, name = ""):
        super(SeqMLP, self).__init__()
        for i in range(len(n_dims)-1):
            self.add_module(
                name + 'layer{}'.format(i),
                MLP(in_features=n_dims[i], out_features=n_dims[i+1], block_connect=block_connect,
                    block_norm=block_norm, 
                    activation=activation, 
                    sample_size=sample_size, preact=preact) # hard coded number of samples
                            )
                  
        
    
    
class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz, features=None):
        # type: (_PointnetSAModuleBase, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []
        B = xyz.shape[0]
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else torch.zeros((B, 1, 3)).to(xyz.device)
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)
            
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)
        
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """
    
    
#     def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):  #as it was
    def __init__(self, npoint, radii, nsamples, mlps, bn="fn", block_connect="none",use_xyz=True):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
#             self.mlps.append(SharedMLP(mlp_spec, bn=bn)) # AS IT WAS
            self.mlps.append(SeqMLP(sample_size=nsample*npoint if npoint is not None else nsample, n_dims=mlp_spec, block_norm=bn, block_connect=block_connect))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """
  
#     def __init__( #AS IT WAS
#         self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
#     ):
    def __init__(self, mlp, npoint=None, radius=None, nsample=None, bn="fn", block_connect="none",use_xyz=True):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
            block_connect=block_connect
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    from torch.autograd import Variable

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = Variable(torch.randn(2, 9, 3).cuda(), requires_grad=True)
    xyz_feats = Variable(torch.randn(2, 9, 6).cuda(), requires_grad=True)

    test_module = PointnetSAModuleMSG(
        npoint=2, radii=[5.0, 10.0], nsamples=[6, 3], mlps=[[9, 3], [9, 6]]
    )
    test_module.cuda()
    print(test_module(xyz, xyz_feats))

    #  test_module = PointnetFPModule(mlp=[6, 6])
    #  test_module.cuda()
    #  from torch.autograd import gradcheck
    #  inputs = (xyz, xyz, None, xyz_feats)
    #  test = gradcheck(test_module, inputs, eps=1e-6, atol=1e-4)
    #  print(test)

    for _ in range(1):
        _, new_features = test_module(xyz, xyz_feats)
        new_features.backward(torch.cuda.FloatTensor(*new_features.size()).fill_(1))
        print(new_features)
        print(xyz.grad)


