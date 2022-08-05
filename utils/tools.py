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


def read_config(yaml_config_path):
    assert os.path.isfile(yaml_config_path) is True
    with open(yaml_config_path, 'r') as f:
        yaml_config = yaml.load(f.read(), Loader=yaml.Loader)
    return yaml_config


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


def calculate_correct(scores, labels):
    assert scores.size(0) == labels.size(0)
    _, pred = scores.max(dim=1)
    correct = torch.sum(pred.add(1).eq(labels))
    return correct


def calculate_class_correct(scores, labels):
    _, pred = scores.max(dim=1)
    correct_labels = labels * pred.add(1).eq(labels)
    class_correct = torch.bincount(correct_labels.cpu()).numpy()
    class_correct[0] = 0
    class_number = torch.bincount(labels.cpu()).numpy()
    return class_correct, class_number
