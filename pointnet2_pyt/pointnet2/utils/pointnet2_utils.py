from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
from torch.autograd import Function
import torch.nn as nn
import etw_pytorch_utils as pt_utils
import sys

try:
    import builtins
except:
    import __builtin__ as builtins

try:
    import pointnet2._ext as _ext
except ImportError:
    if not getattr(builtins, "__POINTNET2_SETUP__", False):
        raise ImportError(
            "Could not import _ext module.\n"
            "Please see the setup instructions in the README: "
            "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
        )

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *

    import torch
import torch.nn as nn

def aggregation(x, lengths, input_shape, device, type='mean'):
    """
    x: [batch, n_dist * sample_size, hidden_units]
    (due to the concatenation of the individual encoder outputs)
    lengths: [batch, n_dist]

    """
    batch, n_dists, n_samples, _ = input_shape
    x = x.reshape(batch, n_dists, n_samples, -1)
    # [batch, n_dists, n_samples]    
    length_mask = torch.arange(n_samples).expand(lengths.shape[0], lengths.shape[1], n_samples).to(device) < lengths.unsqueeze(-1)
    if type == 'sum':
        out = (x * length_mask.unsqueeze(-1)).sum(dim=-2)
    elif type == 'mean':
        # numerator is [batch, n_dists, hidden_units]
        # denominator is [batch, n_dists, 1]
        out = (x * length_mask.unsqueeze(-1)).sum(dim=-2) / length_mask.sum(dim=-1).unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported type aggregation: {type}")

    out = out.reshape(batch, -1)
    assert len(out.shape) == 2

    if torch.all(torch.eq(lengths, n_samples)):
        # print('assertion')
        if type == 'mean':
            assert torch.allclose(out, x.mean(dim=2).reshape(batch, -1), rtol=1e-05, atol=1e-05), f"aggregation is off: {out} vs. {x.mean(dim=2).reshape(batch, -1)}"
        elif type == 'sum':
            assert torch.allclose(out, x.sum(dim=2).reshape(batch, -1), rtol=1e-05, atol=1e-05), f"aggregation is off: {out} vs. {x.sum(dim=2).reshape(batch, -1)}"
    return out


def reshape_x_and_lengths(x, lengths, device):
    if len(x.shape) == 3:
        x = x.unsqueeze(1)
    if type(lengths) == type(None):
        # print('creating lengths')
        lengths = x.shape[2] * torch.ones((x.shape[0], x.shape[1])).to(device)
    else:
        lenghts = lengths.reshape(x.shape[0], x.shape[1])
    assert lengths.shape == x.shape[:2], f"lengths should be shaped [batch, n_dist]: {lengths.shape} vs. {x.shape[:2]}"
    return x, lengths


class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                print('len(inputs)', len(inputs))
                inputs = module(inputs)
        return inputs

def mask_matrix(matrix, lengths):
    """
    lengths is [batch, 1]

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert lengths.shape == (matrix.shape[0], 1), f"{lengths.shape} vs. {(matrix.shape[0], 1)}"
    batch, n_samples, n_feats = matrix.shape
    # [batch, n_samples]
    length_mask = torch.arange(n_samples).expand(batch, n_samples).to(device) < lengths
    return matrix * length_mask.unsqueeze(-1)

import subprocess

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


class BatchSetNorm(nn.BatchNorm1d):
    def _check_input_dim(self, input):
        if input.dim() != self.expected_input_during_norm:
            raise ValueError(
                f"expected {self.expected_input_during_norm}D input (got {input.dim()}D input)"
            )

    def forward(self, x):
        # [batch, features] after aggregation, [batch, samples, features] otherwise
        assert x.dim() == 2 or x.dim() == 3, f"Initial input should be 2D or 3D, got {x.dim()}D"
        self.expected_input_during_norm = x.dim() + 1
        out = super().forward(x.unsqueeze(1))
        assert out.dim() == x.dim() + 1, f"Input before squeeze should be {x.dim() + 1}D, got {out.dim()}D"
        return out.squeeze(1)


def connect(input_tensor, output_tensor, block_connect="none"):
    if block_connect == "resid":
        if input_tensor.shape[-1] != output_tensor.shape[-1]:
            # zero pad feature dimension
            padding = (0, output_tensor.shape[-1] - input_tensor.shape[-1])
            input_tensor = nn.functional.pad(input_tensor, padding)
        output_tensor = output_tensor + input_tensor
    elif block_connect == "dense":
        feats = output_tensor.shape[-1]
        output_tensor = torch.cat([output_tensor, input_tensor], dim=-1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        fc = nn.Linear(output_tensor.shape[-1], feats).to(device)
        output_tensor = fc(output_tensor)
    return output_tensor


# class BatchFeatureNorm(pt_utils.BatchNorm2d):
            
#     def forward(self, x):
#         # [batch, features] after aggregation, [batch, samples, features] otherwise
#         orig_dim = x.dim()
#         orig_shape = x.shape
#         assert orig_dim == 3 or orig_dim == 4, f"Initial input should be 2D or 3D, got {orig_dim}D"
# #         if orig_dim == 2:
# #             x = x.unsqueeze(-1)
# #         x = torch.transpose(x, 1, 2)  # [batch, features, npoints, samples]
#         out = super().forward(x)
# #         if orig_dim == 2:
# #             out = out.squeeze(1)
# #         else:
# #             out = torch.transpose(out, 1, 2)  # [batch, npoints, samples, features]
#         assert out.shape == orig_shape, f"Input should be {orig_shape}, got {out.shape}"
#         return out

    
    
class BatchFeatureNorm(pt_utils.BatchNorm1d):
            
    def forward(self, x):
        # [batch, features] after aggregation, [batch, samples, features] otherwise
        orig_dim = x.dim()
        orig_shape = x.shape
        assert orig_dim == 2 or orig_dim == 3, f"Initial input should be 2D or 3D, got {orig_dim}D"
        if orig_dim == 2:
            x = x.unsqueeze(-1)
        x = torch.transpose(x, 1, 2)  # [batch, features, npoints, samples]
        out = super().forward(x)
        if orig_dim == 2:
            out = out.squeeze(1)
        else:
            out = torch.transpose(out, 1, 2)  # [batch, npoints, samples, features]
        assert out.shape == orig_shape, f"Input should be {orig_shape}, got {out.shape}"
        return out



class BatchFeatureNorm1(nn.BatchNorm1d):
     def forward(self, x):
        # [batch, features] after aggregation, [batch, samples, features] otherwise
        orig_dim = x.dim()
        orig_shape = x.shape
        assert orig_dim == 2 or orig_dim == 3, f"Initial input should be 2D or 3D, got {orig_dim}D"
        if orig_dim == 2:
            x = x.unsqueeze(1)
        x = torch.transpose(x, 1, 2)  # [batch, features, samples]
        out = super().forward(x)
        if orig_dim == 2:
            out = out.squeeze(-1)
        else:
            out = torch.transpose(out, 1, 2)  # [batch, samples, features]
        assert out.shape == orig_shape, f"Input should be {orig_shape}, got {out.shape}"
        return out

class PerSetPerFeatureNorm(nn.LayerNorm):
    def forward(self, x):
        # [batch, features] after aggregation, [batch, samples, features] otherwise
        orig_dim = x.dim()
        orig_shape = x.shape
        assert orig_dim == 2 or orig_dim == 3, f"Initial input should be 2D or 3D, got {orig_dim}D"
        if orig_dim == 2:
            x = x.unsqueeze(1)
        x = torch.transpose(x, 1, 2)  # [batch, features, samples]
        out = super().forward(x)
        if orig_dim == 2:
            out = out.squeeze(-1)
        else:
            out = torch.transpose(out, 1, 2)  # [batch, samples, features]
        assert out.shape == orig_shape, f"Input should be {orig_shape}, got {out.shape}"
        return out

def get_norm(block_norm, sample_size, dim_V):
    if block_norm == "bsn":
        return BatchSetNorm(1)
    # per set, per sample over the features
    elif block_norm == "ln":
        return nn.LayerNorm(dim_V)
    elif block_norm == "fn":
        return BatchFeatureNorm(dim_V)
    elif block_norm == "fn1":
        return BatchFeatureNorm1(dim_V)
    elif block_norm == "bn":
        return pt_utils.BatchNorm1d(sample_size)
    # TODO: only works for fixed sets
    # implementing per set layer norm over samples and features is tricky since it needs to take variable length sets into account 
    elif block_norm == "ln1":
        return nn.LayerNorm([sample_size, dim_V])
    elif block_norm == "ln1_noparams":
        return nn.LayerNorm([sample_size, dim_V], elementwise_affine=False)
    elif block_norm == "ln_noparams":
        return nn.LayerNorm(dim_V, elementwise_affine=False)
    elif block_norm == "fn1_noparams":
        return BatchFeatureNorm1(dim_V, affine=False)
    elif block_norm == "bn_noparams":
        return nn.BatchNorm1d(sample_size, affine=False)
    elif block_norm == "sfn_noparams":
        return PerSetPerFeatureNorm(sample_size, elementwise_affine=False)
    return None

class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        return _ext.furthest_point_sampling(xyz, npoint)

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """

        _, C, N = features.size()

        ctx.for_backwards = (idx, C, N)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = _ext.three_nn(unknown, known)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, m = ctx.three_interpolate_for_backward

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        return _ext.ball_query(new_xyz, xyz, radius, nsample)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
