"""
@File  :logger.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/23 7:23 PM
@Desc  :log tools
"""
import os
from time import time
import datetime
import json
import yaml
import logging

import torch

from utils.tools import time_format
from utils.tools import get_current_time_string


def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


class Logger:
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

    def __init__(self, config, max_epochs, min_save_epoch, dataset_name, output_root_dir, log_layer=[],update_frequency=60):
        self.config = config

        self.current_iter = 0
        self.current_epoch = 0
        self.max_epochs = max_epochs
        self.min_save_epoch = min_save_epoch

        self.start_time = time()
        self.last_update = None

        self.output_path = self.get_output_path(dataset_name, output_root_dir)

        self.epoch_loss = {}
        self.total_correct_number = 0
        self.total_number = 0

        self.update_frequency = update_frequency
        self.log_layer = log_layer

        # DEBUG, INFO, WARNING, ERROR, CRITICAL, NOTSET
        self.logging = logging
        self.logging.Formatter.converter = beijing
        self.logging.basicConfig(filename=os.path.join(self.output_path, '%s.log' % dataset_name),
                                 filemode='w',
                                 level=logging.DEBUG,
                                 format=self.LOG_FORMAT,
                                 datefmt=self.DATE_FORMAT
                                 )
        self.logging.debug('output_path: %s' % self.output_path)

    def save_config(self):
        with open(os.path.join(self.output_path, 'config.yaml'), 'w') as file:
            yaml.dump(self.config, file)

    def _clean_epoch_stats(self):
        self.epoch_stats = {}
        self.epoch_loss = {}
        self.total_number = 0

    def new_epoch(self, learning_rates):
        self.current_epoch += 1
        self.last_update = time()
        epoch_string = self.get_epoch_string()

        self.logging.info(epoch_string + '>' * 120)
        self.logging.info(epoch_string + "lr: %s " % str(learning_rates))
        self._clean_epoch_stats()

    def finish_epoch(self):
        epoch_string = self.get_epoch_string()

        self.logging.info('-' * 60)
        self.logging.info(epoch_string + "current cost: %s " % time_format(time() - self.last_update))
        self.logging.info(epoch_string + 'total   cost: %s' % time_format(time() - self.start_time))
        self.logging.info(epoch_string + 'predict time: %s' % time_format(
            (self.max_epochs - self.current_epoch) * (time() - self.last_update)
        ))
        self.logging.info(epoch_string + '<' * 120)

    def get_epoch_string(self):
        return "<%d/%d epoch> - " % (self.current_epoch, self.max_epochs)

    def update_epoch_loss(self, losses_dict, default_value=0.0):
        for k, v in losses_dict.items():
            if type(v) == dict:
                for kk, vv in v.items():
                    if self.epoch_loss.get(k, None) is None:
                        self.epoch_loss[k] = {}
                    self.epoch_loss[k][kk] = self.epoch_loss[k].get(kk, default_value) + vv
            else:
                self.epoch_loss[k] = self.epoch_loss.get(k, default_value) + v

    def get_average_loss(self, epoch_number):
        result = self.epoch_loss.copy()
        for k, v in result.items():
            if type(v) == dict:
                if result.get(k, None) is None:
                    result[k] = {}
                for kk, vv in v.items():
                    result[k][kk] = result[k][kk] / epoch_number
            else:
                result[k] = result[k] / epoch_number
        return result

    def log(self, iter_idx, max_iter, global_iter, losses_dict, correct_number, batch_size, model_parameters):
        self.current_iter += 1
        epoch_string = self.get_epoch_string()

        # update epoch loss
        self.update_epoch_loss(losses_dict)

        # update epoch accuracy
        self.total_correct_number += correct_number
        self.total_number += batch_size

        # log
        if iter_idx % self.update_frequency == 0:
            self.logging.debug(epoch_string + "-" * 60)
            self.logging.debug(
                epoch_string + "<%d/%d iteration, %d/%d global iteration>" % (
                    iter_idx, max_iter, global_iter, max_iter * self.max_epochs)
            )
            acc_string = "%.3f%%" % (100 * float(correct_number) / batch_size)
            self.logging.debug(epoch_string + "<acc>: %s" % acc_string)

            self.logging.debug(epoch_string + "<loss>:")
            for string in ["%24s: %s" % (str(k), str(v)) for k, v in losses_dict.items()]:
                self.logging.debug(epoch_string + string)

            self.logging.debug(epoch_string + "<gradient info>:")
            weight_grad_dict = self.get_weight_grad_dict(model_parameters)
            for string in ["%24s: %s" % (str(k), str(v)) for k, v in weight_grad_dict.items()]:
                self.logging.debug(epoch_string + string)

        if iter_idx % max_iter == 0:
            epoch_average_loss = self.get_average_loss(max_iter)
            self.logging.info(epoch_string + '-' * 60)

            self.logging.info(epoch_string + "<Losses on train>:")
            for string in ["%24s: %s" % (str(k), str(v)) for k, v in epoch_average_loss.items()]:
                self.logging.info(epoch_string + string)

            epoch_acc_string = "%.3f%%" % (100 * float(self.total_correct_number) / self.total_number)
            self.logging.info(epoch_string + "<Accuracies on train>: " + epoch_acc_string)

    def get_weight_grad_dict(self, model_parameters):
        weight_grad_dict = {}
        for name, params in model_parameters:
            if name not in self.log_layer:
                continue

            weight_grad_dict[name] = {
                'grad': None,
                'weight': None
            }
            if params.grad:
                weight_grad_dict[name]['grad'] = [
                    params.flatten().min().item(),
                    params.flatten().max().item(),
                ]

            if params:
                weight_grad_dict[name]['weight'] = [
                    params.flatten().min().item(),
                    params.flatten().max().item(),
                ]
        return weight_grad_dict

    def log_test(self, phase, class_accuracy_dict, accuracy_info_dict):
        epoch_string = self.get_epoch_string()

        self.logging.critical(epoch_string + "=" * 60)
        self.logging.critical(epoch_string + "<Accuracies on %s> " % phase + ", ".join(
            ["%s : %.2f%%" % (k, v * 100) for k, v in class_accuracy_dict.items()]))
        self.logging.critical(epoch_string + "<Summary Accuracies on %s>" % phase)
        for string in ["%24s: %s" % (str(k), str(v)) for k, v in accuracy_info_dict.items()]:
            self.logging.critical(epoch_string + string)

    def save_model(self, model, epoch, val_acc_info, lr, name=None):
        epoch_string = self.get_epoch_string()
        if epoch < self.min_save_epoch:
            return
        if name is None:
            name = '%depoch_%.2f%%acc' % (epoch, 100 * val_acc_info['average_accuracy'])
        torch.save({
            'epoch': epoch,
            'lr': lr,
            'val_acc_info': val_acc_info,
            'model_state_dict': model.state_dict(),
        }, os.path.join(self.output_path, f'{name}.pth'))
        self.logging.critical(epoch_string + 'save model to %s.ptg' % name)

    def save_result(self, val_res=None, test_res=None, best_val_acc=None, best_val_epoch=None, name='result.json'):

        self.logging.info('*' * 120)
        self.logging.info("Best val on %d epoch, accuracy: %.2f, best_val_result: %s" % (
            best_val_epoch, best_val_acc, val_res[best_val_epoch]))
        best_acc_dict = {
            'best_val_epoch': best_val_epoch,
            'best_val_acc': best_val_acc,
            'best_val_result': val_res[best_val_epoch],
            'val_res': val_res,
            'test_res': test_res
        }

        with open(os.path.join(self.output_path, name), 'w', encoding='utf-8') as file:
            json.dump(best_acc_dict, file, indent=4)

    @staticmethod
    def get_output_path(dataset_name, output_root_dir):
        output_dir = os.path.join(
            output_root_dir,
            dataset_name,
            'temp',
            get_current_time_string()
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return output_dir
