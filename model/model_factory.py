"""
@File  :model_factory.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/8/1 10:28
@Desc  :build the model
"""

from model import resnet

backbone_map = {
    'resnet18': resnet.resnet18,
    'resnet50': resnet.resnet50,
    'convnet': resnet.ConvNet
}


def get_encoder(name):
    if name not in backbone_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return backbone_map[name](**kwargs)

    return get_network_fn


def get_backbone(backbone_name, pretrained):
    return get_encoder(backbone_name)(pretrained=pretrained)
