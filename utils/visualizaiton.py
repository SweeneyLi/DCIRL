import numpy as np
from visdom import Visdom
import torch.nn as nn

from utils.tools import combine_dict


class Visualization(object):

    def __init__(self, visdom_config):
        self.viz = Visdom(
            env=visdom_config['env'],
            port=visdom_config['network']['port'],
            server=visdom_config['network']['server'],
            base_url=visdom_config['network']['base_url'],
            use_incoming_socket=True,
        )
        assert self.viz.check_connection(timeout_seconds=3), 'Please Open The Visdom Server'

        self.image_win_basic_config = visdom_config['image_win_basic']
        self.line_win_basic_config = visdom_config['line_win_basic']

        self.line_win = self.get_line_win(visdom_config['line_win'])
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
                nrow=image_win_config[win_name]['nrow'],
                opts={'title': win_name}
            )
            image_win_config[win_name]['down_sample'] = nn.Upsample(
                scale_factor=1. / image_win_config[win_name]['scale'], mode='bilinear', align_corners=True)
        return image_win_config

    def get_line_win(self, line_win_config):
        for win_name in line_win_config.keys():
            for k, v in self.line_win_basic_config.items():
                line_win_config[win_name][k] = line_win_config[win_name].get(k, self.line_win_basic_config[k])

            # title
            line_win_config[win_name]['title'] = win_name
            # legend
            legend_list = []
            for legends in line_win_config[win_name]['legend']:
                legend_list.extend(legends.replace(' ', '').split(','))
            line_win_config[win_name]['legend'] = legend_list

            legend_length = len(legend_list)
            line_win_config[win_name]['win'] = self.viz.line(
                X=np.column_stack([0 for _ in range(legend_length)]),
                Y=np.column_stack([0.0 for _ in range(legend_length)]),
                opts=line_win_config[win_name])
            line_win_config[win_name]['legend_length'] = legend_length

        return line_win_config

    def visual_text(self, text_dict):
        """
        visual text in visdom, the key in text_dict should correspond in config.yaml
        :param text_dict: {'layer1': {'loss': 0.9}, 'accuracy': 0.9, 'x': 1}
        :return: None
        """
        for win_name, texts in text_dict.items():
            texts = combine_dict(texts)

            legend_length = self.line_win[win_name]['legend_length']
            X = np.column_stack([texts['x'] for _ in range(legend_length)])
            Y = np.column_stack([0.0 for _ in range(legend_length)])

            for i, legend in enumerate(self.line_win[win_name]['legend']):
                Y[0][i] = texts.get(legend, 0)

            self.viz.line(X=X, Y=Y, win=self.line_win[win_name]['win'], update='append')

    def visual_image(self, image_dict):
        for win_name, image_sample in image_dict.items():
            self.viz.images(
                post_processing(image_sample, self.image_win[win_name]['down_sample']).numpy(),
                nrow=self.image_win[win_name]['nrow'],
                win=self.image_win[win_name]['win']
            )


def post_processing(image, down_sample):
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
    image = down_sample(image)
    return image
