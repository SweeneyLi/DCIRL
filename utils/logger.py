"""
@File  :logger.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/23 7:23 PM
@Desc  :log tools
"""
import os
from time import time
import json
import yaml
import logging

import torch

from utils.tools import time_format
from utils.tools import get_current_time_string


class Logger:
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

    def __init__(self, config, max_epochs, min_save_epoch, dataset_name, output_root_dir, update_frequency=30):
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

        # DEBUG, INFO, WARNING, ERROR, CRITICAL, NOTSET
        self.logging = logging
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

    def new_epoch(self, current_epoch, learning_rates):
        self.current_epoch += 1
        self.last_update = time()

        self.logging.info('>' * 60)
        self.logging.info("%d epoch start" % (current_epoch))
        self.logging.info("lr: %s " % str(learning_rates))
        self._clean_epoch_stats()

    def finish_epoch(self):
        self.logging.info('-' * 30)
        self.logging.info("%d epoch end" % self.current_epoch)
        self.logging.info("current cost: %s " % time_format(time() - self.last_update))
        self.logging.info('total   cost: %s' % time_format(time() - self.start_time))
        self.logging.info('predict time: %s' % time_format(
            (self.max_epochs - self.current_epoch) * (time() - self.last_update)
        ))
        self.logging.info('<' * 60)

    def log(self, iter_idx, max_iter, global_iter, losses_dict, correct_number, batch_size, grad_dict):
        self.current_iter += 1
        epoch_string = "<%d/%d epoch> - " % (self.current_epoch, self.max_epochs)
        # loss
        loss_string = ", ".join(["%s : %.3f" % (k, v) for k, v in losses_dict.items()])
        for k, v in losses_dict.items():
            self.epoch_loss[k] = self.epoch_loss.get(k, 0.0) + v

        # accuracy
        acc_string = "%.3f%%" % (100 * float(correct_number) / batch_size)
        self.total_correct_number += correct_number
        self.total_number += batch_size

        # log
        if iter_idx % self.update_frequency == 0:
            self.logging.debug(
                "[%d/%d iteration of epoch %d/%d, %d/%d global iteration]" % (
                    iter_idx, max_iter, self.current_epoch, self.max_epochs, global_iter, max_iter * self.max_epochs)
            )
            self.logging.debug(epoch_string + "<loss>:%s" % loss_string)
            self.logging.debug(epoch_string + "<acc>:%s" % acc_string)
            self.logging.debug(epoch_string + "<gradient info>:")
            for string in ["%32s: %s" % (str(k), str(v)) for k, v in grad_dict.items()]:
                self.logging.debug(string)

        if iter_idx % max_iter == 0:
            epoch_loss_string = ", ".join(["%s : %.3f" % (k, v / max_iter) for k, v in self.epoch_loss.items()])
            epoch_acc_string = "%.3f%%" % (100 * float(self.total_correct_number) / self.total_number)
            self.logging.info(epoch_string + '-' * 30)
            self.logging.info(epoch_string + "<Losses on train> " + epoch_loss_string)
            self.logging.info(epoch_string + "<Accuracies on train> " + epoch_acc_string)

    def log_test(self, phase, class_accuracy_dict, accuracy_info_dict):
        self.logging.critical("-" * 30)
        self.logging.critical(
            "<Accuracies on %s> " % phase + ", ".join(
                ["%s : %.2f%%" % (k, v * 100) for k, v in class_accuracy_dict.items()]))
        self.logging.critical("<Summary Accuracies on %s>" % phase)
        for string in ["%32s: %s" % (str(k), str(v)) for k, v in accuracy_info_dict.items()]:
            self.logging.critical(string)

    def save_model(self, model, epoch, val_acc_info, lr, name=None):
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

    def save_result(self, val_res=None, test_res=None, best_val_acc=None, best_val_epoch=None, name='result.json'):
        self.logging.info('*' * 30)
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
            get_current_time_string()
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return output_dir
