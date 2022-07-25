"""
@File  :logger.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/23 7:23 PM
@Desc  :log tools
"""
import os
from time import time, localtime, strftime
import json
import yaml

import torch

from tf_logger import TFLogger
from tools import time_format


class Logger:
    def __init__(self, config, max_epochs, dataset_name, output_root_dir, tf_logger=False, update_frequency=30):
        self.config = config

        self.current_iter = 0
        self.current_epoch = 0
        self.max_epochs = max_epochs

        self.start_time = time()
        self.last_update = time()

        self.output_path = self.get_output_path(dataset_name, output_root_dir)

        self.epoch_stats = {}
        self.epoch_loss = {}
        self.total_number = 0

        self.tf_logger = TFLogger(self.output_path) if tf_logger else None
        self.update_frequency = update_frequency

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

        print('>' * 30)
        print("New epoch -lr: %s " % ", ".join([str(lr) for lr in learning_rates]))
        self._clean_epoch_stats()

        if self.tf_logger:
            for n, v in enumerate(learning_rates):
                self.tf_logger.scalar_summary("aux/lr%d" % n, v, self.current_epoch)

    def finish_epoch(self):
        print('-' * 30)
        print("Total epoch time: %.2f" % (time() - self.last_update))
        print('<' * 30)

    def log(self, iter_idx, max_iter, losses_dict, correct_dict, batch_number):
        self.current_iter += 1

        for k, v in losses_dict.items():
            self.epoch_loss[k] = self.epoch_loss.get(k, 0.0) + v
        loss_string = ", ".join(["%s : %.3f" % (k, v) for k, v in losses_dict.items()])

        for k, v in correct_dict.items():
            self.epoch_stats[k] = self.epoch_stats.get(k, 0.0) + v
            self.total_number = self.total_number + batch_number
        acc_string = ", ".join(["%s : %.2f" % (k, 100 * (v / batch_number)) for k, v in correct_dict.items()])

        if iter_idx % self.update_frequency == 0:
            print("[%d/%d iteration of epoch %d/%d] \n {loss} %s \n {acc} %s" %
                  (iter_idx + 1, max_iter, self.current_epoch, self.max_epochs, loss_string, acc_string))
            # update tf log
            if self.tf_logger:
                for k, v in correct_dict.items():
                    self.tf_logger.scalar_summary("train_step/loss_%s" % k, v, self.current_iter)

        if (iter_idx + 1) % max_iter == 0:
            epoch_loss_string = ", ".join(["%s : %.3f" % (k, v / max_iter) for k, v in self.epoch_loss.items()])
            epoch_acc_string = ", ".join(
                ["%s : %.2f" % (k, 100 * (v / self.total_number)) for k, v in self.epoch_stats.items()])

            print('-' * 30)
            print("<Losses on train> " + epoch_loss_string)
            print("<Accuracies on train> " + epoch_acc_string)
            print('Train epoch time: %s' % time_format(time() - self.last_update))
            print('Duration time: %s' % time_format(self.start_time - time()))
            print('Predict time: %s' % time_format(
                (self.max_epochs - self.current_epoch) * (time() - self.last_update)
            ))

            if self.tf_logger:
                for k, v in self.epoch_loss.items():
                    self.tf_logger.scalar_summary("train_epoch/loss_%s" % k, v / max_iter, self.current_epoch)
                for k, v in self.epoch_stats.items():
                    self.tf_logger.scalar_summary("train_epoch/acc_%s" % k, v / self.total_number, self.current_epoch)

    def log_test(self, phase, class_accuracy):
        print("-" * 30)
        print(
            "<Accuracies on %s> " % phase + ", ".join(["%s : %.2f" % (k, v * 100) for k, v in class_accuracy.items()]))
        if self.tf_logger:
            for k, v in class_accuracy.items():
                self.tf_logger.scalar_summary("%s/acc_%s" % (phase, k), v, self.current_epoch)

    def save_model(self, model, val_acc, name='best_model'):
        torch.save({
            'epoch': self.current_epoch,
            'val_acc': val_acc,
            'model_state_dict': model.state_dict(),
        }, os.path.join(self.output_path, f'{name}.pth'))

    @staticmethod
    def get_output_path(dataset_name, output_root_dir):
        output_dir = os.path.join(
            output_root_dir,
            dataset_name,
            strftime("%Y-%m-%d-%H-%M", localtime())
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return output_dir
