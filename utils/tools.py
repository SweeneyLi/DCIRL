"""
@File  :tools.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/23 9:27 AM
@Desc  :common tools
"""
import yaml
import os


def read_config(yaml_config_path):
    assert os.path.isfile(yaml_config_path) is True
    with open(yaml_config_path, 'r') as f:
        yaml_config = yaml.load(f.read(), Loader=yaml.Loader)
    return yaml_config
