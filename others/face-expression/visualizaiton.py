import os
import numpy as np
import cv2
from visdom import Visdom
import pdb
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw

class Visualization(object):
  def __init__(self, opt):
    self.viz = Visdom(port=7557, server='http://localhost', base_url='/', username='',password='', use_incoming_socket=True, env=opt.visual_env)
    assert self.viz.check_connection(timeout_seconds=3), 'Please Open The Visdom Server'

    self.nrow = opt.visual_nrow
    self.scale = opt.visual_scale
    self.batch_size = opt.batch_size
    self.height = opt.height
    self.width = opt.width

    self.sample_win = self.viz.images(np.zeros((self.batch_size, 3, int(self.height/self.scale), int(self.width/self.scale))),nrow = self.nrow, opts={'title':'Samples Domain'})
    self.train_loss_win = self.viz.line(
            X=np.column_stack(([0],[0],[0],[0],[0],[0],[0],[0],[0])),
            Y=np.column_stack(([0],[0],[0],[0],[0],[0],[0],[0],[0])),
            opts={'title':'train curve',
                  'xlabel':'iteration',
                  'ylabel':'loss',
                  'legend':['triplet_loss_exp','triplet_loss_gender','triplet_loss_age','cls_loss_expression','cls_loss_gender','regression_loss_age','accuracy_expression','accuracy_gender','accuracy_age'],
                  'width':1024,
                  'height':256,
                  'showlegend':True})
    self.downsample = nn.Upsample(scale_factor=1./self.scale, mode='bilinear', align_corners=True)

  def run(self,samples,triplet_loss_expression,triplet_loss_gender,triplet_loss_age,cls_loss_expression,cls_loss_gender,regression_loss_age,accuracy_expression,accuracy_gender,accuracy_age,cnt):

    self.viz.line(X=np.column_stack(([cnt],[cnt], [cnt], [cnt], [cnt], [cnt], [cnt], [cnt], [cnt])),
                  Y=np.column_stack((triplet_loss_expression,triplet_loss_gender,triplet_loss_age,cls_loss_expression,cls_loss_gender,regression_loss_age,accuracy_expression,accuracy_gender,accuracy_age)),
                  win=self.train_loss_win,
                  update='append')
    post_samples = self.post_processing(samples)
    self.viz.images(post_samples.numpy(),
                    nrow=self.nrow,
                    win=self.sample_win,
                    opts={'title':'Samples Domain'})
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
    image = image*255
    image = self.downsample(image)
    return image
