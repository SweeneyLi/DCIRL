"""
@File  :pretrain.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/23 9:21 AM
@Desc  :
"""
import os

import torch.cuda
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

from data.dataloader import get_data_loader_dict
from model.Resnet import resnet18
from model.loss_function import DCLoss
from model.optimizer_factory import get_optim_and_scheduler
from utils.tools import read_config, str_to_tuple, calculate_correct, calculate_class_correct
from utils.logger import Logger
from utils.visualizaiton import Visualization


class BaseTrainer:
    def __init__(self, config_dir_path='config', gpu_device='0,1'):
        super(BaseTrainer, self).__init__()
        self.config = self.get_config(config_dir_path)
        self.device = self.get_device(gpu_device)

    @staticmethod
    def get_config(config_dir_path):
        return read_config(config_dir_path)

    @staticmethod
    def get_device(gpu_device):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Pretrainer(BaseTrainer):
    phase = 'pretrain'

    def __init__(self):
        super(Pretrainer, self).__init__()
        # dataset
        self.data_loader_dict = get_data_loader_dict(
            phase=self.phase,
            root_path=self.config['dataset']['root_path'],
            dataset_name=self.config['dataset']['name'],
            image_size=str_to_tuple(self.config['dataset']['image_size']),
            n_ways=self.config['dataset']['few_shot']['n_ways'],
            k_shots=self.config['dataset']['few_shot']['k_shots'],
            query_shots=self.config['dataset']['few_shot']['query_shots'],
            batch_size=self.config['pretrain']['batch_size'],
            shuffle=self.config['dataset']['data_loader']['shuffle'],
            num_workers=self.config['dataset']['data_loader']['num_workers'],
        )
        self.dataset_name = self.config['dataset']['name']
        self.class_number = self.data_loader_dict['train'].dataset.class_number
        self.index_to_label_dict = self.data_loader_dict['train'].dataset.index_to_label_dict

        # network
        self.model = resnet18(num_classes=self.class_number, pretrained=False).cuda()
        self.model = torch.nn.DataParallel(self.model)

        # loss
        self.train_stage = 0
        self.middle_loss_length = 4
        self.stage_epoch_threshold = self.config['loss']['stage_epoch_threshold']
        self.criterion = CrossEntropyLoss()
        self.loss_function = DCLoss(
            same_coefficient=self.config['loss']['coefficient']['same'],
            different_coefficient=self.config['loss']['coefficient']['different']
        )

        # pretrain
        self.epochs = self.config['pretrain']['epoch']
        self.lr = self.config['pretrain']['lr']

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
            log_layer=self.config['log']['layer'],
            update_frequency=self.config['log']['update_frequency']
        )

        # visual
        self.visual_image_number = self.config['visdom']['image_win_basic']['number']
        self.visualize = Visualization(self.config['visdom'])

        # load state dict
        load_state_dict_config = self.config['pretrain'].get('load_state_dict', None)
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
            optimizer_config=self.config['pretrain']['optimizer']
        )

    def do_training(self):
        self.logger.save_config()

        self.best_val_acc = 0
        self.best_val_epoch = 0

        for self.current_epoch in range(1, self.epochs + 1):
            self.logger.new_epoch([group["lr"] for group in self.optimizer.param_groups])
            self._do_train_epoch()
            self.logger.finish_epoch()

        val_res = self.results['val']
        test_res = self.results['test']

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
            middle_feature_dict, final_feature = self.model(samples)

            # correct number
            correct_number = calculate_correct(final_feature, labels)

            # classifier loss
            classifier_loss = self.criterion(final_feature, labels)
            loss_dict['classifier_loss'] = classifier_loss.item()
            total_loss += classifier_loss

            # whole loss
            for i in range(1, self.train_stage + 1):
                middle_layer_name = list(middle_feature_dict.keys())[self.middle_loss_length - i]
                middle_feature = middle_feature_dict[middle_layer_name]
                middle_origin_feature, middle_same_feature, middle_different_feature = torch.split(
                    middle_feature, [small_batch_size, small_batch_size, small_batch_size]
                )
                middle_loss, middle_same_loss, middle_different_loss = self.loss_function.get_same_different_loss(
                    middle_origin_feature, middle_same_feature, middle_different_feature
                )

                total_loss += middle_loss
                loss_dict[middle_layer_name] = {}
                loss_dict[middle_layer_name]['loss'] = middle_loss.item()
                loss_dict[middle_layer_name]['same_loss'] = middle_same_loss.item()
                loss_dict[middle_layer_name]['different_loss'] = middle_different_loss.item()

            loss_dict['total_loss'] = total_loss.item()
            # loss backward
            total_loss.backward()

            # step
            self.optimizer.step()

            # visualize
            self.visualize.visual_text({
                'train_curve': {
                    'x': self.global_iter,
                    'accuracy': correct_number / batch_size,
                    **loss_dict,
                }
            })
            self.visualize.visual_image({
                'origin_samples_domain': origin_samples[:self.visual_image_number].cpu().detach(),
                'same_samples_domain': same_samples[:self.visual_image_number].cpu().detach(),
                'different_samples_domain': different_samples[:self.visual_image_number].cpu().detach(),
            })

            self.logger.log(
                iter_idx=iter_idx,
                max_iter=max_iter,
                global_iter=self.global_iter,
                losses_dict=loss_dict,
                correct_number=correct_number,
                batch_size=batch_size,
                model_parameters=self.model.named_parameters()
            )
            self.global_iter += 1

        self.scheduler.step()

        # evaluation
        self.model.eval()
        with torch.no_grad():
            for sub_phase in ['val', 'test']:
                data_loader = self.data_loader_dict[sub_phase]
                class_accuracy_dict, accuracy_info_dict = self.do_eval(data_loader)

                self.logger.log_test(sub_phase, class_accuracy_dict, accuracy_info_dict)
                self.results[sub_phase][self.current_epoch] = {
                    'class_accuracy': class_accuracy_dict,
                    'accuracy_info': accuracy_info_dict
                }

        current_val_acc = self.results['val'][self.current_epoch]['accuracy_info']['average_accuracy']
        if current_val_acc >= self.best_val_acc:
            self.best_val_acc = current_val_acc
            self.best_val_epoch = self.current_epoch
            self.best_val_acc_info = self.results['val'][self.current_epoch]['accuracy_info']
            self.logger.save_model(self.model, self.best_val_epoch, self.best_val_acc_info,
                                   self.optimizer.param_groups[-1]['lr'], 'best')

        self.logger.save_model(self.model, self.current_epoch,
                               self.results['val'][self.current_epoch]['accuracy_info'],
                               self.optimizer.param_groups[-1]['lr'])

        # change train stage
        if self.train_stage < self.middle_loss_length and self.current_epoch - self.best_val_epoch > self.stage_epoch_threshold:
            self.train_stage += 1
            self.logger.logging.critical(self.logger.get_epoch_string() + 'change to %d stage' % self.train_stage)

        # visual eval curve
        val_accuracy = self.results['val'][self.current_epoch]['accuracy_info']['average_accuracy']
        test_accuracy = self.results['val'][self.current_epoch]['accuracy_info']['average_accuracy']

        self.visualize.visual_text({
            'accuracy_curve': {
                'x': self.current_epoch,
                'val_accuracy': val_accuracy,
                'test_accuracy': test_accuracy,
            }
        })

    def do_eval(self, data_loader):
        class_correct_list = [0 for _ in range(self.class_number + 1)]
        class_number_list = [0 for _ in range(self.class_number + 1)]
        for iter_idx, (samples, labels) in enumerate(data_loader):
            samples = Variable(samples.cuda())
            labels = Variable(labels.cuda())

            _, classifier = self.model(samples)
            batch_class_correct, batch_class_number = calculate_class_correct(classifier, labels)

            for i, number in enumerate(batch_class_number):
                class_number_list[i] += number
            for i, number in enumerate(batch_class_correct):
                class_correct_list[i] += number

        min_accuracy, max_accuracy = 1, 0
        min_accuracy_class, max_accuracy_class = None, None
        class_accuracy_dict = {}
        for i in range(self.class_number):
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
    pretrainer = Pretrainer()
    pretrainer.do_training()
    print('end')
