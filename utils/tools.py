"""
@File  :tools.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/23 9:27 AM
@Desc  :common tools
"""
import yaml
import os
import datetime
import torch
import numpy as np


def read_config(yaml_config_path):
    config_list = os.listdir(yaml_config_path)
    config = {}
    for config_name in config_list:
        with open(os.path.join(yaml_config_path, config_name), 'r') as f:
            config.update(yaml.load(f.read(), Loader=yaml.Loader))
    return config


def float_2_scientific(float_number):
    return '{:2e}'.format(float_number)


def time_format(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return '%02d:%02d:%02d' % (h, m, s)


def str_to_tuple(input_string):
    if type(input_string) == str and ',' in input_string:
        return tuple([int(i) for i in input_string.lstrip('(').rstrip(')').split(',')])

    if type(input_string) == list:
        return list(map(lambda x: str_to_tuple(x), input_string))

    return input_string


def get_current_time_string(hours=8, time_format='%m%d_%H%M'):
    return (datetime.datetime.now() + datetime.timedelta(hours=hours)).strftime(time_format)


def combine_dict(input_dict):
    temp_dict = {}
    for k, v in input_dict.items():
        if type(v) == dict:
            for kk, vv in v.items():
                temp_dict[k + '_' + kk] = vv

    return {**input_dict, **temp_dict}


def calculate_correct(scores, labels):
    assert scores.size(0) == labels.size(0)
    _, pred = scores.max(dim=1)
    correct = torch.sum(pred.eq(labels))
    return correct.item()


def calculate_class_correct(scores, labels):
    _, pred = scores.max(dim=1)
    correct_labels_add_1 = labels.add(1) * pred.eq(labels)
    class_correct = torch.bincount(correct_labels_add_1.cpu()).numpy()
    class_correct = np.delete(class_correct, 0)
    class_number = torch.bincount(labels.cpu()).numpy()
    return class_correct, class_number
