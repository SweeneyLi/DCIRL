"""
@File  :train.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/23 9:21 AM
@Desc  :train the model
"""
import os
import numpy as np
from collections import deque

import torch.cuda
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

from data.dataloader import get_data_loader_dict
from model.DCIRL import DCModule
from model.loss_function import DCLoss
from model.optimizer_factory import get_optim_and_scheduler
from utils.tools import read_config, str_to_tuple, calculate_correct, calculate_class_correct
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
            dataset_file=config['dataset']['file'],
            dataset_name=config['dataset']['name'],
            dataset_path=config['dataset']['path'],
            image_size=str_to_tuple(config['dataset']['image_size']),
            batch_size=config['train']['batch_size'],
            shuffle=True,
            num_workers=4)
        self.class_number = self.data_loader_dict['train'].dataset.class_number
        self.index_to_label_dict = self.data_loader_dict['train'].dataset.index_to_label_dict

        # network
        self.model = DCModule(
            backbone_pretrained=config['model']['basic_module']['pretrained'],
            basic_module_name=config['model']['basic_module']['name'],
            basic_module_out_dim=config['model']['basic_module']['out_dim'],
            middle_module_features_list=str_to_tuple(config['model']['middle_module']['features_list']),
            senior_module_features_list=str_to_tuple(config['model']['senior_module']['features_list']),
            class_num=self.class_number
        ).cuda()
        self.model = torch.nn.DataParallel(self.model)

        # loss
        self.criterion = CrossEntropyLoss()
        self.loss_function = DCLoss(
            batch_size=config['train']['batch_size'],
            class_num=self.class_number,
            off_diag_coefficient=config['loss']['coefficient']['off_diag_coefficient'],
            common_coefficient=config['loss']['coefficient']['common_coefficient'],
            different_coefficient=config['loss']['coefficient']['different_coefficient'],
            whole_different_coefficient=config['loss']['coefficient']['whole_different_coefficient']
        )
        self.train_stage = 1

        # optimizer
        self.optimizer, self.scheduler = get_optim_and_scheduler(
            model=self.model,
            lr=config['train']['lr'],
            optimizer_config=config['optimizer']
        )

        # train
        self.batch_size = config['train']['batch_size']
        self.epochs = config['train']['epoch']
        self.lr = config['train']['lr']

        self.current_epoch = 1
        self.global_iter = 1
        self.best_val_acc = 0
        self.best_val_epoch = 0
        self.best_val_acc_info = None
        self.results = {'val': [{} for _ in range(self.epochs + 1)], 'test': [{} for _ in range(self.epochs + 1)]}

        # output
        self.output_root_dir = config['output']['root_path']
        self.logger = Logger(
            config=config,
            max_epochs=self.epochs,
            min_save_epoch=config['output']['min_save_epoch'],
            dataset_name=self.dataset_name,
            output_root_dir=self.output_root_dir,
            update_frequency=config['log']['update_frequency']
        )

        # visual
        self.visualize = Visualization(
            visual_env=config['log']['env'],
            visual_nrow=config['log']['nrow'],
            visual_scale=config['log']['scale'],
            batch_size=config['train']['batch_size'],
            height=config['log']['height'],
            width=config['log']['width']
        )

        # load state dict
        load_state_dict_config = config['model'].get('load_state_dict', None)
        if load_state_dict_config:
            self.train_stage = load_state_dict_config['start_stage']

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

    def do_training(self):
        self.logger.save_config()

        self.best_val_acc = 0
        self.best_val_epoch = 0

        for self.current_epoch in range(1, self.epochs + 1):
            self.logger.new_epoch(self.current_epoch, [group["lr"] for group in self.optimizer.param_groups])
            self._do_train_epoch()
            self.logger.finish_epoch()

        val_res = self.results['val']
        test_res = self.results['test']
        # remove sentry
        val_res.pop(0)
        test_res.pop(0)
        self.best_val_epoch = self.best_val_epoch - 1

        self.logger.save_result(val_res, test_res, self.best_val_acc, self.best_val_epoch)

    def _do_train_epoch(self):
        self.model.train()

        train_data_loader = self.data_loader_dict['train']
        max_iter = len(train_data_loader)

        for iter_idx, (origin_samples, origin_labels) in enumerate(train_data_loader, start=1):
            small_batch_size = len(origin_labels)
            batch_size = small_batch_size * 3

            # preprocessing
            same_samples, different_samples, different_labels = train_data_loader.dataset.get_contrast_batch(
                origin_labels)
            samples = torch.cat((origin_samples, same_samples, different_samples))
            labels = torch.cat((origin_labels, origin_labels, different_labels))

            samples = Variable(samples.cuda())
            labels = Variable(labels.cuda())

            # zero grad
            self.optimizer.zero_grad()

            # forward
            total_loss = torch.tensor(0.0, requires_grad=True).cuda()
            loss_dict = {}

            # calculate features and losses
            middle_feature, senior_feature, classifier = self.model(samples)

            # correct number
            correct_number = calculate_correct(classifier, labels)

            # classifier loss
            classifier_loss = self.criterion(classifier, labels.add(-1))
            loss_dict['classifier_loss'] = classifier_loss.item()
            total_loss += classifier_loss

            # whole loss
            if self.train_stage > 1:
                senior_feature_origin, senior_feature_same, senior_feature_different = torch.split(
                    senior_feature, [small_batch_size, small_batch_size, small_batch_size]
                )
                whole_loss, whole_loss_same, whole_loss_different = self.loss_function.get_whole_loss(
                    senior_feature_origin, senior_feature_same, senior_feature_different
                )

                total_loss += whole_loss
                loss_dict['whole_loss'] = whole_loss.item()
                loss_dict['whole_loss_same'] = whole_loss_same.item()
                loss_dict['whole_loss_different'] = whole_loss_different.item()

            # contrast loss
            if self.train_stage > 2:
                middle_feature_origin, middle_feature_same, middle_feature_different = torch.split(
                    middle_feature, [small_batch_size, small_batch_size, small_batch_size]
                )
                contrast_loss, contrast_common_loss, contrast_different_loss = self.loss_function.get_contrast_loss(
                    middle_feature_origin, middle_feature_same, middle_feature_different)

                total_loss += contrast_loss
                loss_dict['contrast_loss'] = contrast_loss.item()
                loss_dict['contrast_same_loss'] = contrast_common_loss.item()
                loss_dict['contrast_different_loss'] = contrast_different_loss.item()

            loss_dict['total_loss'] = total_loss.item()
            # loss backward
            total_loss.backward()

            # step
            self.optimizer.step()

            # visualize
            self.visualize.visual_train(
                origin_samples.cpu().detach(),
                same_samples.cpu().detach(),
                different_samples.cpu().detach(),
                correct_number.item() / batch_size,
                total_loss.item(),
                classifier_loss.item(),
                loss_dict.get('whole_loss', 0),
                loss_dict.get('whole_loss_same', 0),
                loss_dict.get('whole_loss_different', 0),
                loss_dict.get('contrast_loss', 0),
                loss_dict.get('contrast_loss_same', 0),
                loss_dict.get('contrast_loss_different', 0),
                self.global_iter
            )

            # log
            grad_dict = {
                'last fc': self.model.module.fc.weight.grad[0][:3],
                'senior module linear': self.model.module.senior_module[0].weight.grad[0][:3],
                'middle module linear': self.model.module.middle_module[0].weight.grad[0][:3],
                'basic module last conv2': self.model.module.basic_module.layer4[1].conv2.weight.grad[0][0][0][:3],
                'basic module first conv1': self.model.module.basic_module.conv1.weight.grad[0][0][0][:3]
            }
            self.logger.log(
                iter_idx=iter_idx,
                max_iter=max_iter,
                global_iter=self.global_iter,
                losses_dict=loss_dict,
                correct_number=correct_number,
                batch_size=batch_size,
                grad_dict=grad_dict
            )
            self.global_iter += 1

        self.scheduler.step()

        # evaluation
        self.model.eval()
        with torch.no_grad():
            for phase in ['val', 'test']:
                data_loader = self.data_loader_dict[phase]
                class_accuracy_dict, accuracy_info_dict = self.do_eval(data_loader)

                self.logger.log_test(phase, class_accuracy_dict, accuracy_info_dict)
                self.results[phase][self.current_epoch] = {
                    'class_accuracy': class_accuracy_dict,
                    'accuracy_info': accuracy_info_dict
                }

        current_val_acc = self.results['val'][self.current_epoch]['accuracy_info']['average_accuracy']
        if current_val_acc >= self.best_val_acc:
            self.best_val_acc = current_val_acc
            self.best_val_epoch = self.current_epoch
            self.best_val_acc_info = self.results['val'][self.current_epoch]['accuracy_info']
            self.logger.save_model(self.model, self.best_val_epoch, self.best_val_acc_info,
                                   self.optimizer.param_groups[0]['lr'], 'best')

        self.logger.save_model(self.model, self.current_epoch,
                               self.results['val'][self.current_epoch]['accuracy_info'],
                               self.optimizer.param_groups[0]['lr'])

        # change train stage
        if self.train_stage < 3 and self.current_epoch - self.best_val_epoch > 5:
            self.train_stage += 1
            self.logger.logging.critical('change to %d stage' % self.train_stage)

        val_accuracy = self.results['val'][self.current_epoch]['accuracy_info']['average_accuracy']
        test_accuracy = self.results['val'][self.current_epoch]['accuracy_info']['average_accuracy']
        self.visualize.visual_eval(val_accuracy, test_accuracy, self.global_iter)

    def do_eval(self, data_loader):
        class_correct_list = [0 for _ in range(self.class_number + 1)]
        class_number_list = [0 for _ in range(self.class_number + 1)]
        for iter_idx, (samples, labels) in enumerate(data_loader):
            samples = Variable(samples.cuda())
            labels = Variable(labels.cuda())

            _, _, classifier = self.model(samples)
            batch_class_correct, batch_class_number = calculate_class_correct(classifier, labels)

            for i, number in enumerate(batch_class_number):
                class_number_list[i] += number
            for i, number in enumerate(batch_class_correct):
                class_correct_list[i] += number

        min_accuracy, max_accuracy = 1, 0
        min_accuracy_class, max_accuracy_class = None, None
        class_accuracy_dict = {}
        for i in range(1, self.class_number + 1):
            class_name = self.index_to_label_dict[i]
            the_accuracy = class_correct_list[i] / class_number_list[i]
            class_accuracy_dict[class_name] = the_accuracy

            if the_accuracy < min_accuracy:
                min_accuracy = the_accuracy
                min_accuracy_class = class_name
            elif the_accuracy > max_accuracy:
                max_accuracy = the_accuracy
                max_accuracy_class = class_name

        accuracy_info_dict = {
            'average_accuracy': float(sum(class_correct_list)) / sum(class_number_list),
            'min_accuracy': {
                'class_name': min_accuracy_class,
                'accuracy': min_accuracy
            }, 'max_accuracy': {
                'class_name': max_accuracy_class,
                'accuracy': max_accuracy
            }
        }
        return class_accuracy_dict, accuracy_info_dict


if __name__ == '__main__':
    print('start')
    trainer = Trainer()
    trainer.do_training()
