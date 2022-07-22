"""
@File  :DCIRL.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/21 5:36 PM
@Desc  :
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy


class MLP(nn.Module):
    def __init__(self, num_i, num_h, num_o, layer_h=1):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(num_i, num_h)
        self.layer2 = self._make_layer(num_h, layer_h)
        self.layer3 = nn.Linear(num_h, num_o)

    @staticmethod
    def _make_layer(num_h, layer_h):
        layers = nn.Sequential()
        layers.add_module("hidden_relu_0", nn.ReLU())

        for i in range(layer_h):
            layers.add_module("hidden_linear_%d" % (i + 1), nn.Linear(num_h, num_h))
            layers.add_module("hidden_relu_%d" % (i + 1), nn.ReLU())
        return layers

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

image_size = 224 * 224 * 3
basic_module_hidden_size = 112 * 56
basic_module_hidden_number = 3
middle_module_hidden_size = 28 * 112
middle_module_hidden_number = 2
class DCModule(nn.Module):
    def __init__(self):
        super(DCModule, self).__init__()
        self.basic_module = MLP(base_module_)
        self.middle_module = MLP(128 * 64, 64 * 64, 32 * 64,2)
