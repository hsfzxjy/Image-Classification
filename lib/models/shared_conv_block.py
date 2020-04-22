import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
# from lib.models.tools.module_helper import ModuleHelper

__all__ = ['ParameterManager', 'shared_conv2d', 'shared_conv_class', 'Bottleneck', 'BasicBlock', 'SharedConvBuilder', 'shared_model_replicated_hook']

class ModuleHelper:

    @staticmethod
    def BatchNorm2d(*args, **kw):
        return nn.BatchNorm2d

class ParameterManager(object):

    def __init__(self, host):
        self.host = host

    def _encode(self, key: tuple) -> str:
        return '_'.join(map(str, key))

    def ensure(self, key: tuple, init_fn: callable):
        key = self._encode(key)
        if hasattr(self.host, key):
            return
        value = init_fn()
        setattr(self.host, key, nn.Parameter(value))

    def __getitem__(self, key: tuple):
        key = self._encode(key)
        return getattr(self.host, key)

class BasicSharedConv2d(nn.Conv2d):

    def __init__(self, mgr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parameter_manager = mgr

        del self.weight
        mgr.ensure(self._weight_key, self._init_weight)

        self.use_bias = self.bias is not None
        if self.use_bias:
            del self.bias
            self.bias = 1  # We need self.bias to be a non-None value for printing
            mgr.ensure(self._bias_key, self._init_bias)

    def replace_mgr(self, new_mgr):
        self._parameter_manager = new_mgr

    @property
    def stdv(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        return stdv

    @property
    def _weight_key(self):
        return ('conv_weight', self.out_channels, self.in_channels // self.groups, *self.kernel_size)

    def _init_weight(self):
        weight = torch.Tensor(
            self.out_channels, self.in_channels // self.groups, *self.kernel_size
        )
        stdv = self.stdv
        weight.data.uniform_(-stdv, stdv)

        return weight

    @property
    def _bias_key(self):
        return 'conv_bias', self.out_channels

    def _init_bias(self):
        bias = torch.Tensor(self.out_channels)
        stdv = self.stdv
        bias.data.uniform_(-stdv, stdv)        
        return bias

    def forward(self, input):
        weight = self._parameter_manager[self._weight_key]
        bias = self._parameter_manager[self._bias_key] if self.use_bias else None
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

_conv_backend_mapping = {
    'basic': BasicSharedConv2d,
}

def shared_conv2d(backend, *args, **kwargs):
    return _conv_backend_mapping[backend](*args, **kwargs)

def shared_conv_class(backend):
    return _conv_backend_mapping[backend]

def SharedConvBuilder(*args, **kwargs):
    return partial(shared_conv2d, *args, **kwargs)

def shared_model_replicated_hook(self, *args, **kwargs):
    """
    See lib/extensions/parallel/data_parallel.py:L195. This hook is crucial.

    After the model `net` is replicated into `net1`, `net2`, etc., the `_parameter_manager` of
    convs in replicas still point to `net`, which is undesired.

    Here we should calibrated all `_parameter_manager` in current replica.

    NOTE that ParameterManager is very light-weighted (which contains only a reference),
    so this part would not be a performance bottleneck.
    """
    new_mgr = ParameterManager(self)
    for m in self.modules():
        if hasattr(m, 'replace_mgr'):
            m.replace_mgr(new_mgr)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, conv_builder,
        inplanes, planes, stride=1, downsample=None, bn_type=None, 
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_builder(
            inplanes, planes,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv2 = conv_builder(
            planes, planes,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_in(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, conv_builder,
        inplanes, planes, stride=1, downsample=None, bn_type=None
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_builder(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.conv2 = conv_builder(
            planes, planes,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes)
        self.conv3 = conv_builder(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_in(out)

        return out