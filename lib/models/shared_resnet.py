##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Microsoft Research
## Author: hsfzxjy, RainbowSecret
## Copyright (c) 2020
## hsfzxjy@gmail.com, yuyua@microsoft.com
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import math
from copy import deepcopy
from collections import OrderedDict
import functools
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

# from lib.models.tools.module_helper import ModuleHelper
# from lib.utils.tools.logger import Logger as Log
from .shared_conv_block import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ModuleHelper:

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return nn.BatchNorm2d

class SharedResNet(nn.Module):

    __data_parallel_replicate__ = shared_model_replicated_hook

    def __init__(self, backend, block, layers, num_classes=1000, deep_base=False, bn_type=None):
        super().__init__()
        self.inplanes = 128 if deep_base else 64
        self.deep_base = deep_base
        self.backend = backend
        self.bn_type = bn_type

        self.mgr = mgr = ParameterManager(self)
        conv_builder = SharedConvBuilder(backend, mgr)

        with mgr.new_context():
            if self.deep_base:
                self.resinit = nn.Sequential(OrderedDict([
                    ('conv1', conv_builder(
                        3, 64, 
                        kernel_size=3, stride=2, padding=1, bias=False,
                    )),
                    ('bn1', ModuleHelper.BatchNorm2d(bn_type=bn_type)(64)),
                    ('relu1', nn.ReLU(inplace=False)),
                    ('conv2', conv_builder(
                        64, 64, 
                        kernel_size=3, stride=1, padding=1, bias=False,
                    )),
                    ('bn2', ModuleHelper.BatchNorm2d(bn_type=bn_type)(64)),
                    ('relu2', nn.ReLU(inplace=False)),
                    ('conv3', conv_builder(
                        64, 128, 
                        kernel_size=3, stride=1, padding=1, bias=False
                    )),
                    ('bn3', ModuleHelper.BatchNorm2d(bn_type=bn_type)(self.inplanes)),
                    ('relu3', nn.ReLU(inplace=False))]
                ))
            else:
                # we do not handle the original stem case.
                self.resinit = nn.Sequential(OrderedDict([
                    ('conv1', conv_builder(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
                    ('bn1', ModuleHelper.BatchNorm2d(bn_type=bn_type)(self.inplanes)),
                    ('relu1', nn.ReLU(inplace=False))]
                ))

            self.maxpool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, ceil_mode=False
            )

            self.layer1 = self._make_layer(conv_builder, block, 64, layers[0], bn_type=bn_type)
            self.layer2 = self._make_layer(conv_builder, block, 128, layers[1], stride=2, bn_type=bn_type)
            self.layer3 = self._make_layer(conv_builder, block, 256, layers[2], stride=2, bn_type=bn_type)
            self.layer4 = self._make_layer(conv_builder, block, 512, layers[3], stride=2, bn_type=bn_type)
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for name, param in self.named_parameters():
            if 'conv_weight' in name:
                n = param.shape[0] * param.shape[2] * param.shape[3]
                param.data.normal_(0, math.sqrt(2. / n))

        for m in self.modules():
            if isinstance(m, ModuleHelper.BatchNorm2d(bn_type=bn_type, ret_cls=True)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, conv_builder, block, planes, blocks, stride=1, bn_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_builder(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                ModuleHelper.BatchNorm2d(bn_type=bn_type)(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            conv_builder, self.inplanes, planes, stride, downsample, bn_type=bn_type,
        ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(
                conv_builder, self.inplanes, planes, bn_type=bn_type,
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        with self.mgr.new_context():
            x = self.resinit(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x

class SharedResNetModels(object):

    _layer_config = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
    }

    def deepbase_shared_resnet(self, backend, depth, **kwargs):
        model = SharedResNet(backend, Bottleneck, self._layer_config[depth], deep_base=True,
                             bn_type=0, **kwargs)
        return model

class cls_basic_shared_resnet101:

    @staticmethod
    def get_cls_net(config, **kw):
        return SharedResNetModels().deepbase_shared_resnet('basic', 101)

class cls_dynamic_shared_resnet101:

    @staticmethod
    def get_cls_net(config, **kw):
        return SharedResNetModels().deepbase_shared_resnet('dynamic', 101)