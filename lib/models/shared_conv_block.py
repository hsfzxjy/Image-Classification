import os
import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from contextlib import contextmanager
from collections import OrderedDict
# from lib.utils.tools.logger import Logger as Log
from .pos_embedding import PositionalEmbedding

__all__ = ['ParameterManager', 'shared_conv2d', 'shared_conv_class', 'Bottleneck', 'BasicBlock', 'SharedConvBuilder', 'shared_model_replicated_hook']

class Log:

    info = info_once = staticmethod(print)

class ModuleHelper:

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return nn.BatchNorm2d

class _Context(object):
    
    def __init__(self):
        self._finalizers = OrderedDict([])

    def __getattr__(self, name):
        """
        The function is called when an attribute `name` is not found by normal mechanism.
        We fallback to return `None`.

        e.g.,
        ```
        c = _Context()
        c.obj  # None
        ```
        """
        return None

    def register_finalizer(self, name, fn):
        if name in self._finalizers:
            return

        self._finalizers[name] = fn

    def finalize(self, *args, **kwargs):
        for fn in self._finalizers.values():
            fn(*args, **kwargs)

class ParameterManager(object):
    """
    maintain all the convolution shared parameters.
    we introduce a counter to record the repeated call times, and we apply a layer-wise positional embedding
    for each separate calling. then we apply a transform on the layer-wise positional embedding to predict 
    adaptive convolution kernel rescaling factors, e.g., the kernel rescaling along the channels.
    """

    def __init__(self, host):
        self.host = host
        self._counter = 0
        self.ctx = None

    def reset_counter(self):
        self._counter = 0

    def get_counter(self) -> int:
        return self._counter

    def incr_counter(self) -> int:
        self._counter += 1
        return self._counter

    def _encode(self, key: tuple or str) -> str:
        if isinstance(key, str):
            return key
        return '_'.join(map(str, key))

    def ensure(self, key: tuple, init_fn: callable):
        """
        `init_fn` could return either a Tensor (as Parameter), or a Module (as extra layers).
        """
        key = self._encode(key)
        if hasattr(self.host, key):
            return
        value = init_fn()
        # NOTE that if `value` is `nn.Parameter` with `requires_grad = False`,
        # `nn.Parameter(value)` will still have `requires_grad = True`.
        # To avoid this, if `value` is already an `nn.Parameter`,
        # we pass it through without doing anything.
        if isinstance(value, torch.Tensor) and not isinstance(value, nn.Parameter):
            value = nn.Parameter(value)
        setattr(self.host, key, value)

    def __getitem__(self, key: tuple):
        key = self._encode(key)
        return getattr(self.host, key)

    @contextmanager
    def new_context(self):
        """
        Consider the situations as following:

        1) We want to construct an embedding of size #conv x D. But during initializing convs, we 
        don't know the number of #conv;
        2) At the start of forward, we want to apply 1D conv to embedding first, then cache the result
        and use it at every layer.

        The above two require to
        1) execute some code at some "global stage", e.g., in backbone;
        2) temporarily share some information (cached embedding) across layers (the Tensor's lifetime
        should be longer than `SharedConv.forward()`, but shorten than `backbone.forward()`)
        Ordinarily, we can not implement the above two without messing up the code of backbone.

        This function `new_context()` is designed to decouple the "globally executed" code with
        higher-level logic.

        By executing `with mgr.new_context()`, 
        
        1) `mgr` will have an attribute `mgr.ctx`, where you can store some temporary information;
        2) you can call `mgr.ctx.register_finalizer(name, fn)` to defer a function call to the end of
        the context. Functions with same `name` will be registered only once.

        Currently, two places use this technique:

        1) `__init__` of backbone;
        2) `forward` of backbone.

        NOTE that the this method is not re-entrant, i.e., any form of cascaded calling to this method
        is disallowed. This makes sure we will not mess up the contexts.

        NOTE that this method is not thread-safe.
        """
        if self.ctx is not None:
            raise RuntimeError("{}.new_context() is not reentrant.".format(self.__class__.__name__))

        self.ctx = _Context()
        yield
        self.ctx.finalize()
        del self.ctx
        self.ctx = None

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
        Log.info("Add Conv Tensor of Shape {}x{}x{}x{}".format(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
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

    def _get_weight(self):
        return self._parameter_manager[self._weight_key]

    def forward(self, input):
        weight = self._get_weight()
        bias = self._parameter_manager[self._bias_key] if self.use_bias else None
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

class DynamicSharedConv2d(BasicSharedConv2d):

    def __init__(self, mgr, *args, **kwargs):
        super().__init__(mgr, *args, **kwargs)

        self.global_conv_id = mgr.get_counter()
        mgr.incr_counter()
        self.embedding_in_size = int(os.environ.get('embedding_in_size', 128))
        self.embedding_out_size = int(os.environ.get('embedding_out_size', 8))
        self.embedding_applying_mode = os.environ.get('embedding_applying_mode', 'out').lower()
        Log.info_once('[{}] in_size: {}, out_size: {}, mode: {}'.format(
            self.__class__.__name__,
            self.embedding_in_size, self.embedding_out_size, self.embedding_applying_mode
        ))
        assert self.embedding_applying_mode in ('in', 'out')

        mgr.ensure(self._embedding_transform_key, self._init_embedding_transform)
        mgr.ctx.register_finalizer('init_embedding', self._init_embedding)

    @property
    def _embedding_key(self):
        return 'pos_embedding'

    def _init_embedding(self):
        """
        Return shape: 1 x in_size x 1
        """
        conv_count = self._parameter_manager.get_counter()
        learned = os.environ.get('learned') is not None
        embedding = PositionalEmbedding(conv_count, self.embedding_in_size, learned=learned).permute(1, 0).unsqueeze(0)
        Log.info("Create Positional Embedding Tensor of Shape {}, learned: {}".format(embedding.shape, learned))
        embedding = nn.Parameter(embedding, requires_grad=learned)
        self._parameter_manager.ensure(self._embedding_key, lambda: embedding)

    def _init_embedding_transform(self):
        Log.info("Create Layer-Wise Attention Convolution Tensor of Shape {}x{}x{}x{}".format(self.embedding_in_size, self.embedding_out_size, 1, 1))
        structure = os.environ.get('embedding_transform_structure', 'conv-sigmoid')
        bias = os.environ.get('embedding_transform_bias') is not None
        if structure == 'conv-sigmoid':
            return nn.Sequential(
                nn.Conv1d(self.embedding_in_size, self.embedding_out_size, kernel_size=1, padding=0, bias=bias),
                nn.Sigmoid()
            )
        elif structure == 'conv-tanh':
            return nn.Sequential(
                nn.Conv1d(self.embedding_in_size, self.embedding_out_size, kernel_size=1, padding=0, bias=bias),
                nn.Tanh()
            )
        elif structure == 'conv-relu-conv-sigmoid':
            return nn.Sequential(
                nn.Conv1d(self.embedding_in_size, self.embedding_in_size, kernel_size=1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.embedding_in_size, self.embedding_out_size, kernel_size=1, padding=0, bias=bias),
                nn.Sigmoid()
            )        
        elif structure == 'conv-relu-conv-tanh':
            return nn.Sequential(
                nn.Conv1d(self.embedding_in_size, self.embedding_in_size, kernel_size=1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.embedding_in_size, self.embedding_out_size, kernel_size=1, padding=0, bias=bias),
                nn.Tanh()
            )        

    @property
    def _embedding_transform_key(self):
        return 'embedding_transform_', self.embedding_in_size, self.embedding_out_size, self.embedding_applying_mode

    def _get_transformed_embedding(self):
        if self._parameter_manager.ctx.transformed_embedding is not None:
            embedding = self._parameter_manager.ctx.transformed_embedding
        else:
            embedding = self._parameter_manager[self._embedding_key]
            embedding = self._parameter_manager[self._embedding_transform_key](embedding).squeeze(0)
            self._parameter_manager.ctx.transformed_embedding = embedding

        return embedding[:, self.global_conv_id]

    def _get_weight(self):

        embedding = self._get_transformed_embedding()
        weight = self._parameter_manager[self._weight_key]

        if (self.embedding_applying_mode == 'in' and weight.size(1) % self.embedding_out_size != 0) or \
           (self.embedding_applying_mode == 'out' and weight.size(0) % self.embedding_out_size != 0):
            """
            e.g., HRNet-W48 has out-channels 64 on stem, but multipliers of 18 on stages.
            We simply return the weight.
            """
            return weight

        if self.embedding_applying_mode == 'in':
            embedding = embedding.repeat(weight.size(1) // embedding.size(0)).view(1, -1, 1, 1)
        elif self.embedding_applying_mode == 'out':
            embedding = embedding.repeat(weight.size(0) // embedding.size(0)).view(-1, 1, 1, 1)

        return weight * embedding.expand_as(weight)

_conv_backend_mapping = {
    'basic': BasicSharedConv2d,
    'dynamic': DynamicSharedConv2d,
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
    self.mgr = new_mgr
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