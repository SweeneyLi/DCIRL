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

        # network
        self.model = resnet18(num_classes=self.config['dataset']['few_shot']['n_ways']).cuda()
        self.model = torch.nn.DataParallel(self.model)

        # loss
        self.train_stage = 1
        self.middle_loss_length = 4
        self.stage_epoch_threshold = self.config['loss']['stage_epoch_threshold']
        self.criterion = CrossEntropyLoss()
        self.loss_function = DCLoss(
            same_coefficient=self.config['loss']['coefficient']['same'],
            different_coefficient=self.config['loss']['coefficient']['different']
        )

        # finetune
        self.epochs = self.config['finetune']['epoch']
        self.lr = self.config['finetune']['lr']

        # output
        self.current_epoch = 1
        self.best_val_acc = 0
        self.best_val_epoch = 0
        self.best_val_acc_info = None
        self.results = {'val': [{} for _ in range(self.epochs + 1)], 'test': [{} for _ in range(self.epochs + 1)]}

        self.output_root_dir = self.config['output']['root_path']
        self.logger = Logger(
            config=self.config,
            max_epochs=self.epochs,
            min_save_epoch=self.config['output']['min_save_epoch'],
            dataset_name=self.config['dataset']['name'],
            output_root_dir=self.output_root_dir,
            log_layer=self.config['log']['layer'],
            update_frequency=self.config['log']['update_frequency']
        )

        # visual
        self.visual_image_number = self.config['visdom']['image_win_basic']['number']
        self.visualize = Visualization(self.config['visdom'])

        # load state dict
        load_state_dict_config = self.config['finetune'].get('load_state_dict', None)
        if load_state_dict_config:
            self.train_stage = load_state_dict_config.get('start_stage', self.train_stage)
            self.lr = load_state_dict_config.get('start_lr', self.lr)

            load_dict = torch.load(load_state_dict_config['path'])
            self.model.load_state_dict(load_dict['model_state_dict'])

            del load_dict['model_state_dict']
            self.logger.logging.critical(
                'state_dict:\n%s' % ("\n".join(["%32s: %s" % (str(k), str(v)) for k, v in load_dict.items()]))
            )

        self.logger.logging.info('model info: ' + str(self.model))
        self.logger.logging.info('label dict: ' + str(self.data_loader_dict['train'].dataset.label_to_index_dict))
        self.logger.logging.info(
            'parameters: %.2f M' % (sum(param.numel() for param in self.model.parameters()) / (1024 * 1024)))

        # optimizer
        self.optimizer, self.scheduler = get_optim_and_scheduler(
            model=self.model,
            lr=self.lr,
            optimizer_config=self.config['finetune']['optimizer']
        )

    def do_finetune(self):
