"""
@File  :finetune.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/21 11:11 AM
@Desc  :
"""
import os

import torch.cuda
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

from pretrain import BaseTrainer
from data.dataloader import get_data_loader_dict
from model.Resnet import resnet18
from model.loss_function import DCLoss
from model.optimizer_factory import get_optim_and_scheduler
from utils.tools import read_config, str_to_tuple, calculate_correct, calculate_class_correct
from utils.logger import Logger
from utils.visualizaiton import Visualization


class Finetuner(BaseTrainer):
    phase = 'finetune'

    def __init__(self):
        super(Finetuner, self).__init__()

        # output
        self.current_epoch = 1
        self.global_iter = 1
        self.best_val_acc = 0
        self.best_val_epoch = 0
        self.best_val_acc_info = None
        self.results = {'val': [{} for _ in range(self.epochs + 1)], 'test': [{} for _ in range(self.epochs + 1)]}

        self.output_root_dir = self.config['output']['root_path']
        self.logger = Logger(
            config=self.config,
            max_epochs=self.epochs,
            min_save_epoch=self.config['output']['min_save_epoch'],
            dataset_name=self.dataset_name,
            output_root_dir=self.output_root_dir,
            log_layer=self.config['log'][
                'layer'],
            update_frequency=
            self.config['log']['update_frequency']
        )

        # network
        self.model = resnet18(num_classes=self.config['dataset']['few_shot']['n_ways']).cuda()
        self.model = torch.nn.DataParallel(self.model)


        # load state dict
        load_state_dict_config = self.config['model'].get('load_state_dict', None)
        if load_state_dict_config:
            self.train_stage = load_state_dict_config.get('start_stage', self.train_stage)
            self.lr = load_state_dict_config.get('start_lr', self.lr)

            load_dict = torch.load(load_state_dict_config['path'])
            self.model.load_state_dict(load_dict['model_state_dict'])

            del load_dict['model_state_dict']
            self.logger.logging.critical(
                'state_dict:\n%s' % ("\n".join(["%32s: %s" % (str(k), str(v)) for k, v in load_dict.items()]))
            )