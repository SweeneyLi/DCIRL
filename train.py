"""
@File  :train.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/23 9:21 AM
@Desc  :
"""
import torch.cuda
from model.DCIRL import DCModule
from model.optimizer_factory import get_optim_and_scheduler

from utils.tools import read_config

config_path = 'config.yaml'
config = read_config(config_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self):
        self.dc_module = DCModule(config['model_config'])

        self.optimizer, self.scheduler = get_optim_and_scheduler(
            self.dc_module,
            config['optimizer_config'])


if __name__ == '__main__':
    trainer = Trainer()
    trainer.do_training()
