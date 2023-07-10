import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import random
import numpy as np
import itertools
from network import Face_Expression, Triplet_Semi_Hard_Loss
from dataloader import Expression_Dataset
from visualizaiton import Visualization
import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

def train(opt):
  visualize = Visualization(opt)
  model = Face_Expression(opt)
  model = model.cuda()
  model=torch.nn.DataParallel(model)
  if opt.model:
    model.load_state_dict(torch.load(opt.model))
  optimizer = optim.SGD(model.parameters(),lr=opt.lr, momentum=0.9, weight_decay=0.001)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)
  criterion_softmax = torch.nn.CrossEntropyLoss()
  criterion_triplet = Triplet_Semi_Hard_Loss()
  lr = opt.lr
  model.train()
  cnt = 0
  for epoch in range(opt.epochs):
    if (epoch + 1)%opt.step_size==0:
      lr *= opt.gamma
    train_dataset = Expression_Dataset(opt)
    train_dataloader = DataLoader(
            dataset = train_dataset,
            batch_size = opt.batch_size,
            shuffle=True,
            num_workers = 4)
    model.train()

    for i, (samples, labels) in enumerate(train_dataloader):
      samples = Variable(samples.cuda())
      labels = Variable(labels.cuda())
      optimizer.zero_grad()
      features, cls_prob = model(samples)
      cls_loss = criterion_softmax(cls_prob, labels)
      triplet_loss = criterion_triplet(features, labels)
      loss = cls_loss + triplet_loss
      loss.backward()
      _, predicted = torch.max(cls_prob, 1)
      accuracy = (predicted==labels).sum().item()*1.0 / opt.batch_size

      if (i+1)%opt.print_freq == 0:
        print('Train: epoch:%d, min_batch:%3d, lr=%f, loss=%.4f, cls_loss=%.4f, triplet_loss=%.4f, accuracy=%.4f'%(epoch+1, i, lr, loss, cls_loss, triplet_loss, accuracy))
        visualize.run(samples.cpu().detach(),
                      loss.item(),
                      cls_loss.item(),
                      triplet_loss.item(),
                      accuracy,
                      predicted.cpu(),
                      labels.cpu(),
                      cnt)
        cnt = cnt + 1
        optimizer.step()
    if(epoch + 1)%opt.save_freq == 0:
      torch.save(model.state_dict(), 'checkout/model_' + str(epoch+1) + '.pth')
    scheduler.step()
  torch.save(model.state_dict(), 'checkout/model_final.pth')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--phase', type=str, default='train')
  parser.add_argument('--cuda', action='store_true', default=True)
  parser.add_argument('--batch_size', type=int, default=400)
  parser.add_argument('--channel', type=int, default=3)
  parser.add_argument('--height', type=int, default=256)
  parser.add_argument('--width', type=int, default=256)
  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument('--epochs', type=int, default=150)
  parser.add_argument('--model', type=str, default='')
  parser.add_argument('--num_class', type=int, default=7)
  parser.add_argument('--step_size', type=int, default=50)
  parser.add_argument('--gamma', type=float, default=0.1)
  parser.add_argument('--print_freq', type=int, default=5)
  parser.add_argument('--save_freq', type=int, default=2)
  #parser.add_argument('--train_dataset_path', type=str, default='dataset/train_dataset_61k/train_dataset_61k_aligned')
  #parser.add_argument('--train_image_list', type=str, default='dataset/train_dataset_61k/train_dataset_61k_easy_data.csv')
  
  parser.add_argument('--train_dataset_path', type=str, default='dataset/train_dataset_154k/train_dataset_154k_aligned')
  parser.add_argument('--train_image_list', type=str, default='dataset/train_dataset_154k/train_dataset_154k_easy_data.csv')

  parser.add_argument('--visual_nrow', type=int, default=20)
  parser.add_argument('--visual_scale', type=int, default=4)
  parser.add_argument('--data_root', type=str, default='/home/shiyh/face-expression')
  parser.add_argument('--visual_env', type=str, default='train-env-py3')
  opt = parser.parse_args()
  train(opt)
  print('train finished')

