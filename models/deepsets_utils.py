import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ReLUNoStats(nn.ReLU):
    def __init__(self):
        super(ReLUNoStats, self).__init__()
   
    def forward(self, x):
        stats = x[:, :, -2:]
        x = super().forward(x[:, :, :-2])
        x = torch.cat([x, stats], axis=2)
        return x 
    
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


class BatchFeatureNorm(nn.BatchNorm1d):
    def forward(self, x):
        # [batch, features] after aggregation, [batch, samples, features] otherwise
        orig_dim = x.dim()
        orig_shape = x.shape
        assert orig_dim == 2 or orig_dim == 3, f"Initial input should be 2D or 3D, got {orig_dim}D"
        if orig_dim == 2:
            x = x.unsqueeze(-1)
        x = torch.transpose(x, 1, 2)  # [batch, features, samples]
        out = super().forward(x)
        if orig_dim == 2:
            out = out.squeeze(1)
        else:
            out = torch.transpose(out, 1, 2)  # [batch, samples, features]
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


class PerSetFeatureParams(nn.LayerNorm):
    def __init__(self, *args, feature_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.empty(feature_dim))
        self.biases = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)

    def forward(self, x):
        # standardization
        out = super().forward(x)
        # transform params
        out = F.linear(out, torch.diag_embed(self.weights), self.biases)
        return out

class PerSetPerFeatureNormFeatureParams(PerSetPerFeatureNorm):
    def __init__(self, *args, feature_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.empty(feature_dim))
        self.biases = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)
        
    def forward(self, x):
        # standardization
        out = super().forward(x)
        # transform params
        out = F.linear(out, torch.diag_embed(self.weights), self.biases)
        return out

class BatchSetNormFeatureParams(BatchSetNorm):
    def __init__(self, *args, feature_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.empty(feature_dim))
        self.biases = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)
        
    def forward(self, x):
        # standardization
        out = super().forward(x)
        # transform params
        out = F.linear(out, torch.diag_embed(self.weights), self.biases)
        return out


class InformationPreservingNorm(nn.Module):
    def __init__(self, feature_dim, eps=1e-5):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(feature_dim + 2))
        self.biases = nn.Parameter(torch.empty(feature_dim + 2))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)
        self.eps = eps
    
    def forward(self, x):
        assert x.dim() == 3
        mean = torch.mean(x, axis=[1, 2], keepdim=True)  # per set mean
        std = torch.std(x, axis=[1, 2], unbiased=False, keepdim=True)  # per set std
        out = (x - mean)/(std + self.eps) 
#         print(mean, std)
        full_out = torch.cat([out, mean.repeat(1, out.shape[1], 1), std.repeat(1, out.shape[1], 1)], axis=2)
        return F.linear(full_out, torch.diag_embed(self.weights), self.biases)



def get_norm(block_norm, sample_size, dim_V):
    if block_norm == "bsn":
        return BatchSetNorm(1)
    # per set, per sample over the features
    elif block_norm == "ln":
        return nn.LayerNorm(dim_V, elementwise_affine=False)
    elif block_norm == "fn":
        return BatchFeatureNorm(dim_V)
    elif block_norm == "fn1":
        return BatchFeatureNorm1(dim_V)
    elif block_norm == "bn":
        return nn.BatchNorm1d(sample_size)
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
    elif block_norm == "ln1_fp":
        return PerSetFeatureParams([sample_size, dim_V], elementwise_affine=False, feature_dim=dim_V)
    elif block_norm == "sfn_fp":
        return PerSetPerFeatureNormFeatureParams(sample_size, elementwise_affine=False, feature_dim=dim_V)
    elif block_norm == "bsn_fp":
        return BatchSetNormFeatureParams(1, affine=False, feature_dim=dim_V)
    elif block_norm == "ipn":
        return InformationPreservingNorm(feature_dim=dim_V)
    return None


def get_2d_norm(block_norm, sample_size, dim_V):
    """ normalizations that work on 2D feature dims by flattening the features """
    if block_norm == "bsn":
        return BatchSetNorm2D(1)
    # per set, per sample over the features
    elif block_norm == "ln":
        return LayerNorm2D(dim_V, elementwise_affine=False)
    elif block_norm == "fn":
        return BatchFeatureNorm2D(dim_V)
    elif block_norm == "fn1":
        return BatchFeatureNorm2D(dim_V)
    elif block_norm == "bn":
        return BatchNorm2D(sample_size)
    # TODO: only works for fixed sets
    # implementing per set layer norm over samples and features is tricky since it needs to take variable length sets into account 
    elif block_norm == "ln1":
        return LayerNorm2D([sample_size, dim_V])
    elif block_norm == "ln1_noparams":
        return LayerNorm2D([sample_size, dim_V], elementwise_affine=False)
    elif block_norm == "ln_noparams":
        return LayerNorm2D(dim_V, elementwise_affine=False)
    elif block_norm == "fn1_noparams":
        return BatchFeatureNorm2D(dim_V, affine=False)
    elif block_norm == "bn_noparams":
        return BatchNorm2D(sample_size, affine=False)
    elif block_norm == "sfn_noparams":
        return PerSetPerFeatureNorm2D(sample_size, elementwise_affine=False)
    elif block_norm == "ln1_fp":
        return PerSetFeatureParams2D([sample_size, dim_V], elementwise_affine=False, feature_dim=dim_V)
    elif block_norm == "sfn_fp":
        return PerSetPerFeatureNormFeatureParams2D(sample_size, elementwise_affine=False, feature_dim=dim_V)
    elif block_norm == "bsn_fp":
        return BatchSetNormFeatureParams2D(1, affine=False, feature_dim=dim_V)
    elif block_norm == "ipn":
        return InformationPreservingNorm2D(feature_dim=dim_V)
    return None


class Norm2DMixin():
    def forward(self, x):
        assert x.dim() == 4, f"Initial input should be 4D, got {x.dim()}"  # [batch * sample_size, n_channels, height, width]
        flattened_x = x.reshape(x.shape[0]//10, 10, -1)
        out = super().forward(flattened_x)
        return out.reshape(x.shape)

class InformationPreservingNorm2D(nn.Module):
    def __init__(self, feature_dim, eps=1e-5):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(feature_dim + 2))
        self.biases = nn.Parameter(torch.empty(feature_dim + 2))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)
        self.eps = eps
    
    def forward(self, x):
        assert x.dim() == 4, f"Initial input should be 4D, got {x.dim()}"  # [batch * sample_size, n_channels, height, width]
        _, n_channels, height, width = x.shape
        sample_size = 10
        batch = x.shape[0]//sample_size
        new_x = x.reshape(batch, sample_size, n_channels, height, width)

        mean = torch.mean(new_x, axis=[1, 2, 3, 4], keepdim=True)  # per set mean
        std = torch.std(new_x, axis=[1, 2, 3, 4], unbiased=False, keepdim=True)  # per set std
        out = (new_x - mean)/(std + self.eps)

        # mean and std are each and additional output filter 
        out = torch.cat([out, mean.repeat(1, sample_size, 1, height, width), std.repeat(1, sample_size, 1, height, width)], axis=2)
        out = out.reshape(batch*sample_size, n_channels + 2, height, width)
        # apply scale and bias to each channel
        out = F.linear(torch.transpose(out, 1, -1), torch.diag_embed(self.weights), self.biases)
        out = torch.transpose(out, 1, -1)
        return out.reshape(batch*sample_size, n_channels + 2, height, width)

class BatchSetNorm2D(Norm2DMixin, BatchSetNorm): pass
class LayerNorm2D(Norm2DMixin, nn.LayerNorm): pass
class BatchFeatureNorm2D(Norm2DMixin, BatchFeatureNorm1): pass
class BatchNorm2D(Norm2DMixin, nn.BatchNorm1d): pass
class PerSetPerFeatureNorm2D(Norm2DMixin, PerSetPerFeatureNorm): pass
class PerSetFeatureParams2D(Norm2DMixin, PerSetFeatureParams): pass
class PerSetPerFeatureNormFeatureParams2D(Norm2DMixin, PerSetPerFeatureNormFeatureParams): pass
class BatchSetNormFeatureParams2D(Norm2DMixin, BatchSetNormFeatureParams): pass

