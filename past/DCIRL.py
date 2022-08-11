"""
@File  :DCIRL.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/21 5:36 PM
@Desc  :
"""

from torch import nn

from model.model_factory import get_backbone


def tuple_multiplication(input_tuple):
    if type(input_tuple) != tuple:
        return input_tuple

    result = 1
    for i in input_tuple:
        result = result * i
    return result


class DCModule(nn.Module):
    def __init__(self,
                 backbone_pretrained,
                 basic_module_name,
                 basic_module_out_dim,
                 middle_module_features_list,
                 senior_module_features_list,
                 class_num
                 ):
        super(DCModule, self).__init__()

        self.basic_module = get_backbone(basic_module_name, backbone_pretrained)

        self.middle_module = self._make_layers(
            list(map(lambda x: tuple_multiplication(x), [basic_module_out_dim] + middle_module_features_list)),
            bn=False, no_linear=True
        )
        self.middle_module_out_dim = middle_module_features_list[-1]

        self.senior_module = self._make_layers(
            list(map(lambda x: tuple_multiplication(x), [self.middle_module_out_dim] + senior_module_features_list)),
            bn=True, no_linear=False
        )

        self.fc = nn.Linear(senior_module_features_list[-1], class_num)

    def forward(self, input_feature):
        # input_feature = input_feature.reshape(input_feature.size(0), -1)
        basic_feature = self.basic_module(input_feature)
        middle_feature = self.middle_module(basic_feature)
        senior_feature = self.senior_module(middle_feature)
        classifier = self.fc(senior_feature)
        return middle_feature.reshape(-1, *self.middle_module_out_dim), senior_feature, classifier

    @staticmethod
    def _make_layers(layer_feature_list, bn=True, no_linear=True):
        layers = []
        for i in range(len(layer_feature_list) - 1):
            layers.append(nn.Linear(layer_feature_list[i], layer_feature_list[i + 1], bias=False))
            if bn:
                layers.append(nn.BatchNorm1d(layer_feature_list[i + 1]))
            if no_linear:
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
