"""
@File  :DCIRL.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/21 5:36 PM
@Desc  :
"""
from torch import nn
import torch.nn.functional as F
import numpy


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, hidden_layers=1):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(in_features, hidden_features)
        self.layer2 = self._make_layer(hidden_features, hidden_layers)
        self.layer3 = nn.Linear(hidden_features, out_features)

    @staticmethod
    def _make_layer(hidden_features, hidden_layers):
        layers = nn.Sequential()
        layers.add_module("hidden_relu_0", nn.ReLU())

        for i in range(hidden_layers):
            layers.add_module("hidden_linear_%d" % (i + 1), nn.Linear(hidden_features, hidden_features))
            layers.add_module("hidden_relu_%d" % (i + 1), nn.ReLU())
        return layers

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class DCModule(nn.Module):
    def __init__(self, model_config):
        super(DCModule, self).__init__()
        self.basic_module = MLP(in_features=model_config['image_size'],
                                hidden_features=model_config['basic_module']['hidden_features'],
                                out_features=model_config['basic_module']['output_features'],
                                hidden_layers=model_config['basic_module']['hidden_layers'],
                                )
        self.middle_module = MLP(in_features=model_config['basic_module']['output_features'],
                                 hidden_features=model_config['middle_module']['hidden_features'],
                                 out_features=model_config['middle_module']['output_features'],
                                 hidden_layers=model_config['middle_module']['hidden_layers'],
                                 )
        self.senior_module = nn.Linear(in_features=model_config['middle_module']['output_features'],
                                       out_features=model_config['senior_module']['output_features']
                                       )

    def forward(self, input_feature):
        basic_feature = self.basic_module(input_feature)
        middle_feature = self.middle_module(basic_feature)
        senior_feature = self.senior_module(middle_feature)
        return basic_feature, middle_feature, senior_feature
