import numpy as np
from visdom import Visdom
import torch.nn as nn


class Visualization(object):

    def __init__(self, visdom_config):
        self.viz = Visdom(
            env=visdom_config['env'],
            port=visdom_config['port'],
            server=visdom_config['server'],
            base_url=visdom_config['base_url'],
            use_incoming_socket=True,
        )
        assert self.viz.check_connection(timeout_seconds=3), 'Please Open The Visdom Server'

        self.image_win_basic_config = visdom_config['image_win_basic']
        self.text_win_basic_config = visdom_config['text_win_basic']

        self.text_win = self.get_text_win(visdom_config['text_win'])
        self.image_win = self.get_image_win(visdom_config['image_win'])

    def get_image_win(self, image_win_config):
        for win_name in image_win_config.keys():
            if image_win_config[win_name] is None:
                image_win_config[win_name] = {}
            for k, v in self.image_win_basic_config.items():
                image_win_config[win_name][k] = image_win_config[win_name].get(k, self.image_win_basic_config[k])
            image_win_tensor = np.zeros((
                image_win_config[win_name]['number'],
                3,
                int(image_win_config[win_name]['width'] / image_win_config[win_name]['scale']),
                int(image_win_config[win_name]['height'] / image_win_config[win_name]['scale'])
            ))
            image_win_config[win_name]['win'] = self.viz.images(
                tensor=image_win_tensor,
                nrow=image_win_config[win_name]['n_row'],
                opts={'title': win_name}
            )
            image_win_config[win_name]['down_sample'] = nn.Upsample(
                scale_factor=1. / image_win_config[win_name]['scale'], mode='bilinear', align_corners=True)
        return image_win_config

    def get_text_win(self, text_win_config):
        for win_name in text_win_config.keys():
            for k, v in self.text_win_basic_config.items():
                text_win_config[win_name][k] = text_win_config[win_name].get(k, self.text_win_basic_config[k])

            text_win_config['win_name']['title'] = win_name
            legend_length = len(text_win_config['win_name']['legend'])
            text_win_config['win_name']['win'] = self.viz.line(
                X=np.zeros(legend_length, dtype=int),
                Y=np.zeros(legend_length, dtype=float),
                opts=text_win_config['win_name'])
        return text_win_config

    def visual_train(self, origin_samples, same_samples, different_samples, accuracy, total_loss, classifier_loss,
                     whole_loss,
                     whole_loss_same, whole_loss_different,
                     contrast_loss, contrast_loss_same, contrast_loss_different, iter):
        X = np.column_stack(([iter], [iter], [iter], [iter], [iter], [iter], [iter], [iter], [iter]))
        Y = np.column_stack(
            ([accuracy], [total_loss], [classifier_loss],
             [whole_loss], [whole_loss_same], [whole_loss_different],
             [contrast_loss], [contrast_loss_same], [contrast_loss_different]))
        self.viz.line(X=X, Y=Y, win=self.train_loss_win, update='append')

        self.viz.images(self.post_processing(origin_samples).numpy(), nrow=self.nrow,
                        win=self.origin_sample_win)
        self.viz.images(self.post_processing(same_samples).numpy(), nrow=self.nrow,
                        win=self.same_sample_win)
        self.viz.images(self.post_processing(different_samples).numpy(), nrow=self.nrow,
                        win=self.different_sample_win)

    def visual_eval(self, val_accuracy, test_accuracy, iter):
        X = np.column_stack(([iter], [iter]))
        Y = np.column_stack(([val_accuracy], [test_accuracy]))
        self.viz.line(X=X, Y=Y, win=self.eval_win, update='append')

    def post_processing(self, image):
        channel = image.shape[1]
        if channel == 1:
            image.repeat(1, 3, 1, 1)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for i in range(image.shape[0]):
            image[i][0] = image[i][0] * std[0] + mean[0]
            image[i][1] = image[i][1] * std[1] + mean[1]
            image[i][2] = image[i][2] * std[2] + mean[2]
        image = image * 255
        image = self.downsample(image)
        return image
