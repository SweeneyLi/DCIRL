# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:09:23 2020

@author: syh
"""
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from typing import Any


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),            #1
            nn.ReLU(inplace=True),                                            #2
            nn.MaxPool2d(kernel_size=3, stride=2),                            #3
            nn.Conv2d(64, 192, kernel_size=5, padding=2),                     #4
            nn.ReLU(inplace=True),                                            #5
            nn.MaxPool2d(kernel_size=3, stride=2),                            #6
            nn.Conv2d(192, 384, kernel_size=3, padding=1),                    #7
            nn.ReLU(inplace=True),                                            #8
            # nn.Conv2d(384, 256, kernel_size=3, padding=1),                    #9
            # nn.ReLU(inplace=True),                                            #10
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),                    #11
            # nn.ReLU(inplace=True),                                            #12
            # nn.MaxPool2d(kernel_size=3, stride=2),                            #13            
        )
###########################################################################################        
        self.features2 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),                    #9
            nn.ReLU(inplace=True),                                            #10
            nn.Conv2d(256, 256, kernel_size=3, padding=1),                    #11
            nn.ReLU(inplace=True),                                            #12
            nn.MaxPool2d(kernel_size=3, stride=2),                            #13            
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),                    #9
            nn.ReLU(inplace=True),                                            #10
            nn.Conv2d(256, 256, kernel_size=3, padding=1),                    #11
            nn.ReLU(inplace=True),                                            #12
            nn.MaxPool2d(kernel_size=3, stride=2),                            #13            
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),                    #9
            nn.ReLU(inplace=True),                                            #10
            nn.Conv2d(256, 256, kernel_size=3, padding=1),                    #11
            nn.ReLU(inplace=True),                                            #12
            nn.MaxPool2d(kernel_size=3, stride=2),                            #13            
        )
###########################################################################################        
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))                           #14 
        # self.classifier = nn.Sequential( 
        #     nn.Dropout(),                                                     #15
        #     nn.Linear(256 * 6 * 6, 4096),                                     #16
        #     nn.ReLU(inplace=True),                                            #17
        #     nn.Dropout(),                                                     #18
        #     nn.Linear(4096, 4096),                                            #19
        #     nn.ReLU(inplace=True),                                            #20
        #     nn.Linear(4096, num_classes),                                     #21
        # )
        
        self.avgpool1 = nn.AdaptiveAvgPool2d((6, 6))                           #14 
        self.avgpool2 = nn.AdaptiveAvgPool2d((6, 6))
        self.avgpool3 = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier_expression = nn.Sequential( 
            nn.Dropout(),                                                     #15
            nn.Linear(256 * 6 * 6, 512),                                     #16
            nn.ReLU(inplace=True),                                            #17
            nn.Dropout(),                                                     #18
            nn.Linear(512, 7),                                            #19
        )
        self.classifier_age = nn.Sequential( 
            nn.Dropout(),                                                     #15
            nn.Linear(256 * 6 * 6, 512),                                     #16
            nn.ReLU(inplace=True),                                            #17
            nn.Dropout(),                                                     #18
            nn.Linear(512, 1),                                            #19
        )
        self.classifier_gender = nn.Sequential( 
            nn.Dropout(),                                                     #15
            nn.Linear(256 * 6 * 6, 512),                                     #16
            nn.ReLU(inplace=True),                                            #17
            nn.Dropout(),                                                     #18
            nn.Linear(512, 2),                                               #19
        )
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        # return x
        x = self.features(x)
        
        x1 = self.features2(x)
        x1 = self.avgpool1(x1)
        x1 = torch.flatten(x1, 1)
        expression = self.classifier_expression(x1)
        
        x2 = self.features3(x)
        x2 = self.avgpool2(x2)
        x2 = torch.flatten(x2, 1)
        age = self.classifier_age(x2)
        
        x3 = self.features4(x)
        x3 = self.avgpool3(x3)
        x3 = torch.flatten(x3, 1)
        gender = self.classifier_gender(x3)
        return x1,x3,expression,age,gender
        


def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        # model.load_state_dict(state_dict)
        
        # resNet50 = models.resnet50(pretrained=True)
        # ResNet50 = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=2)
        
        # 读取参数
        pretrained_dict = state_dict
        model_dict = model.state_dict()
        
        # 将pretained_dict里不属于model_dict的键剔除掉
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        # 更新现有的model_dict
        model_dict.update(pretrained_dict)
        
        # 加载真正需要的state_dict
    
        model.load_state_dict(model_dict)
    return model
