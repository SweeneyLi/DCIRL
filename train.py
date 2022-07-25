"""
@File  :train.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/23 9:21 AM
@Desc  :train the model
"""
import os

import torch.cuda
from torch.autograd import Variable

from data.dataloader import get_data_loader_dict
from model.DCIRL import DCModule
from model.loss_function import DCLoss
from model.optimizer_factory import get_optim_and_scheduler
from utils.tools import read_config
from utils.logger import Logger
from utils.visualizaiton import Visualization

# load config
config_path = 'config.yaml'
config = read_config(config_path)

# set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self):
        # dataset
        self.dataset_name = config['dataset']['name']
        self.data_loader_dict = get_data_loader_dict(
            dataset_name=config['dataset']['name'],
            dataset_path=config['dataset']['path'],
            image_size=config['model']['image_size'],
            batch_size=config['train']['batch_size'],
            shuffle=True,
            num_workers=4)
        self.class_number = self.data_loader_dict['train'].dataset.class_number

        # network
        self.model = DCModule(
            basic_module_in_features=config['model']['basic_module']['input_features'],
            basic_module_hidden_features=config['model']['basic_module']['hidden_features'],
            basic_module_out_features=config['model']['basic_module']['out_features'],
            basic_module_hidden_layers=config['model']['middle_module']['hidden_layers'],
            middle_module_hidden_features=config['model']['middle_module']['hidden_features'],
            middle_module_out_features=config['model']['middle_module']['out_features'],
            middle_module_hidden_layers=config['model']['middle_module']['hidden_layers'],
            senior_module_out_features=config['model']['senior_module']['out_features']
        ).cuda()
        self.model = torch.nn.DataParallel(self.model)

        # loss
        self.loss_function = DCLoss(
            batch_size=config['train']['batch_size'],
            class_num=self.class_number,
            l=config['loss']['l'],
            a=config['loss']['a'],
            b=config['loss']['b'],
            g=config['loss']['g'],
            accuracy_same_threshold=config['log']['accuracy_same_threshold'],
            accuracy_different_threshold=config['log']['accuracy_different_threshold']
        )

        # train
        self.batch_size = config['train']['batch_size']
        self.epochs = config['train']['epoch']
        self.lr = config['train']['lr']

        # optimizer
        self.optimizer, self.scheduler = get_optim_and_scheduler(
            model=self.model,
            optimizer_config=config['optimizer']
        )

        # output
        self.output_root_dir = config['output']['root_path']
        self.logger = Logger(
            config=config,
            max_epochs=self.epochs,
            dataset_name=self.dataset_name,
            output_root_dir=self.output_root_dir,
            tf_logger=config['log']['tf_logger'],
            update_frequency=config['log']['update_frequency']
        )
        self.current_epoch = 1
        self.global_iter = 1
        self.best_val_acc = 0
        self.best_val_epoch = 0
        self.results = {}

        # visual
        self.visualize = Visualization(
            visual_env=config['log']['env'],
            visual_nrow=config['log']['nrow'],
            visual_scale=config['log']['scale'],
            batch_size=config['train']['batch_size'],
            height=config['log']['height'],
            width=config['log']['width']
        )

    def do_training(self):
        self.logger.save_config()

        # self.results = {"val": torch.zeros(self.epochs), "test": torch.zeros(self.epochs)}
        # self.best_val_acc = 0
        # self.best_val_epoch = 0

        for self.current_epoch in range(1, self.epochs + 1):
            self.scheduler.step()

            self.logger.new_epoch([group["lr"] for group in self.optimizer.param_groups])
            self._do_train_epoch()
            self.logger.finish_epoch()

        # val_res = self.results['val']
        # test_res = self.results['test']
        # self.logger.save_best_acc(val_res, test_res, self.best_val_acc, self.best_val_epoch - 1)

    def _do_train_epoch(self):
        self.model.train()
        train_data_loader = self.data_loader_dict['train']
        max_iter = len(train_data_loader)

        for iter_idx, (samples, labels) in enumerate(train_data_loader):
            # preprocessing
            contrast_samples, contrast_labels = train_data_loader.dataset.get_contrast_batch()
            samples.extend(contrast_samples)
            labels.extend(contrast_labels)

            samples = Variable(samples.cuda())
            labels = Variable(labels.cuda())

            # zero grad
            self.optimizer.zero_grad()

            # forward
            loss_dict = {}
            correct_dict = {}

            # calculate features and losses
            basic_feature, middle_feature, senior_feature = self.model(samples)

            whole_loss, (correct_number_same, correct_number_different,
                         batch_number) = self.loss_function.get_whole_loss_and_correct_number(
                senior_feature, labels)
            loss_dict['whole_loss'] = whole_loss
            correct_dict['same'] = correct_number_same
            correct_dict['different'] = correct_number_different

            contrast_loss = self.loss_function.get_contrast_loss(middle_feature, labels)
            loss_dict['contrast_loss'] = contrast_loss

            independent_loss = self.loss_function.get_independent_loss(basic_feature)
            loss_dict['independent_loss'] = independent_loss

            total_loss = whole_loss + contrast_loss + independent_loss

            # loss backward
            total_loss.backward()

            # step
            self.optimizer.step()

            # visualize
            self.visualize.run(
                samples.cpu().detach(),
                whole_loss.item(),
                contrast_loss.item(),
                independent_loss.item(),
                total_loss.item(),
                correct_dict['same'] / batch_number,
                correct_dict['different'] / batch_number,
                self.global_iter
            )
            self.global_iter += 1

            # log
            self.logger.log(
                iter_idx=iter_idx,
                max_iter=max_iter,
                losses_dict=loss_dict,
                correct_dict=correct_dict,
                batch_number=batch_number
            )

            # evaluation
            with torch.no_grad():
                for phase in ['val', 'test']:
                    data_loader = self.data_loader_dict[phase]
                    total_number = len(data_loader.dataset)
                    class_correct = self.do_eval(data_loader)
                    for k, v in class_correct.items():
                        class_correct[k] = float(class_correct[k]) / total_number
                    self.logger.log_test(phase, class_correct)
                    self.results[phase][self.current_epoch] = class_correct

                current_val_dict = self.results['val'][self.current_epoch]
                current_val_acc = sum(current_val_dict.values()) / len(current_val_dict.keys())
                if current_val_acc >= self.best_val_acc:
                    self.best_val_acc = current_val_acc
                    self.best_val_epoch = self.current_epoch
                    self.logger.save_model(self.model, self.best_val_acc, 'best')

    def do_eval(self, data_loader):
        correct_dict = {}
        for iter_idx, (samples, labels) in enumerate(data_loader):
            # preprocessing
            contrast_samples, contrast_labels = data_loader.dataset.get_contrast_batch()
            samples.extend(contrast_samples)
            labels.extend(contrast_labels)

            samples = Variable(samples.cuda())
            labels = Variable(labels.cuda())

            basic_feature, middle_feature, senior_feature = self.model(samples)

            whole_loss, (correct_number_same, correct_number_different,
                         batch_number) = self.loss_function.get_whole_loss_and_correct_number(
                senior_feature, labels)
            correct_dict['same'] = correct_number_same
            correct_dict['different'] = correct_number_different
        return correct_dict


if __name__ == '__main__':
    trainer = Trainer()
    trainer.do_training()
