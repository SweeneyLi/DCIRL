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

        win_tensor = np.zeros((self.batch_size, 3, int(self.height / self.scale), int(self.width / self.scale)))

        self.eval_win = self.viz.line(
            X=np.column_stack(([0], [0])),
            Y=np.column_stack(([0], [0])),
            opts={'title': 'accuracy curve',
                  'xlabel': 'iteration',
                  'ylabel': 'accuracy',
                  'legend': ['val_accuracy', 'test_accuracy'],
                  'width': 1200,
                  'height': 200,
                  'showlegend': True})
        self.downsample = nn.Upsample(scale_factor=1. / self.scale, mode='bilinear', align_corners=True)

        self.train_loss_win = self.viz.line(
            X=np.column_stack(([0], [0], [0], [0], [0], [0], [0], [0], [0])),
            Y=np.column_stack(([0], [0], [0], [0], [0], [0], [0], [0], [0])),
            opts={'title': 'train curve',
                  'xlabel': 'iteration',
                  'ylabel': 'loss',
                  'legend': ['accuracy', 'total_loss', 'classifier_loss',
                             'whole_loss', 'whole_loss_same', 'whole_loss_different',
                             'contrast_loss', 'contrast_loss_same', 'contrast_loss_different'],
                  'width': 1200,
                  'height': 500,
                  'showlegend': True})
        self.origin_sample_win = self.viz.images(
            tensor=win_tensor, nrow=self.nrow, opts={'title': 'Origin Samples Domain'}
        )
        self.same_sample_win = self.viz.images(
            tensor=win_tensor, nrow=self.nrow, opts={'title': 'Same Samples Domain'}
        )
        self.different_sample_win = self.viz.images(
            tensor=win_tensor, nrow=self.nrow, opts={'title': 'Different Samples Domain'}
        )

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
