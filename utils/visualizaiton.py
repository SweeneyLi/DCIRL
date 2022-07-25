import numpy as np
from visdom import Visdom
import torch.nn as nn


class Visualization(object):
    def __init__(self, visual_env, visual_nrow, visual_scale, batch_size, height, width):
        self.viz = Visdom(port=7557, server='http://localhost', base_url='/', username='', password='',
                          use_incoming_socket=True, env=visual_env)
        assert self.viz.check_connection(timeout_seconds=3), 'Please Open The Visdom Server'

        self.nrow = visual_nrow
        self.scale = visual_scale
        self.batch_size = batch_size
        self.height = height
        self.width = width

        self.sample_win = self.viz.images(
            np.zeros((self.batch_size, 3, int(self.height / self.scale), int(self.width / self.scale))), nrow=self.nrow,
            opts={'title': 'Samples Domain'})
        self.train_loss_win = self.viz.line(
            X=np.column_stack(([0], [0], [0], [0], [0], [0])),
            Y=np.column_stack(([0], [0], [0], [0], [0], [0])),
            opts={'title': 'train curve',
                  'xlabel': 'iteration',
                  'ylabel': 'loss',
                  'legend': ['whole_loss', 'contrast_loss', 'independent_loss', 'total_loss',
                             'same_accuracy', 'different_accuracy'],
                  'width': 1024,
                  'height': 256,
                  'showlegend': True})
        self.downsample = nn.Upsample(scale_factor=1. / self.scale, mode='bilinear', align_corners=True)

    def run(self, samples, whole_loss, contrast_loss, independent_loss, total_loss,
            same_accuracy, different_accuracy, iter):

        self.viz.line(X=np.column_stack(([iter], [iter], [iter], [iter], [iter], [iter])),
                      Y=np.column_stack((whole_loss, contrast_loss, independent_loss,
                                         total_loss, same_accuracy, different_accuracy)),
                      win=self.train_loss_win,
                      update='append')
        post_samples = self.post_processing(samples)
        self.viz.images(post_samples.numpy(),
                        nrow=self.nrow,
                        win=self.sample_win,
                        opts={'title': 'Samples Domain'})

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
