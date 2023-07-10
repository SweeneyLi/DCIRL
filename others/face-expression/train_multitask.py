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
from dataloader_bak import Expression_Dataset
from visualizaiton import Visualization
import pdb
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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
  criterion_regression = torch.nn.MSELoss()
  lr = opt.lr
  
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
    total = 0
    correct = 0
    for i, (samples, labels_expression, labels_gender, labels_age) in enumerate(train_dataloader):
      samples = Variable(samples.cuda())
      labels_expression = Variable(labels_expression.cuda())
      labels_gender = Variable(labels_gender.cuda())
      labels_age = Variable(labels_age.cuda())
      optimizer.zero_grad()
      start_t = time.time()
      cls_expression_prob, cls_age_prob, cls_gender_prob,feat_expression,feat_gender,feat_age = model(samples)
      end_t = time.time()
      cls_loss_expression = criterion_softmax(cls_expression_prob, labels_expression)
      cls_loss_gender = criterion_softmax(cls_gender_prob, labels_gender)
#      cls_loss_age = criterion_softmax(cls_age_prob, labels_age)
      triplet_loss_gender = criterion_triplet(feat_gender, labels_gender)
      triplet_loss_expression = criterion_triplet(feat_expression, labels_expression)
      triplet_loss_age = criterion_triplet(feat_age, labels_age)
      regression_loss_age = 0.01 * criterion_regression(torch.squeeze(cls_age_prob,1), labels_age.clone().float())

      loss = cls_loss_gender + cls_loss_expression + regression_loss_age + triplet_loss_gender + triplet_loss_expression + triplet_loss_age
      loss.backward()
      _, predicted_expression = torch.max(cls_expression_prob, 1)
      _, predicted_gender = torch.max(cls_gender_prob, 1)
      # _, predicted_age = torch.max(cls_age_prob, 1)
      total += labels_age.size(0)
      res=abs(torch.squeeze(cls_age_prob,1).detach().cpu().numpy().astype(np.int)-labels_age.detach().cpu().numpy())
      correct += np.sum(np.where(res<6,1,0) == 1)

      accuracy_expression = (predicted_expression==labels_expression).sum().item()*1.0 / samples.shape[0]
      accuracy_gender = (predicted_gender==labels_gender).sum().item()*1.0 / samples.shape[0]
     # accuracy_age = (predicted_age==labels_age).sum().item()*1.0 / samples.shape[0]

      accuracy_age = correct/total    

      if (i+1)%opt.print_freq == 0:
        print('--------------------------------'+str(epoch+1)+'--------------------------------')
        print('consume time: {%f}'%(end_t - start_t))
        print('Train: epoch:%d,\tmin_batch:%3d,\tlr=%f\t'%(epoch+1, i, lr))
        print('cls_loss_expression=%.4f,\tcls_loss_gender=%.4f\t'%(cls_loss_expression,cls_loss_gender))
        print('age_loss=%.4f,\ttriplet_loss_exp:%.4f,\ttriplet_loss_gender:%.4f\t'%(regression_loss_age,triplet_loss_expression,triplet_loss_gender))
        print('accuracy_expression=%.4f,\taccuracy_gender=%.4f,\taccuracy_age=%.4f\t'%(accuracy_expression,accuracy_gender,accuracy_age))
        print('--------------------------------------------------------------------------------')

        visualize.run(samples.cpu().detach(),
                      triplet_loss_expression.item(),
       	      triplet_loss_gender.item(),
                      triplet_loss_age.item(),
                      cls_loss_expression.item(),
                      cls_loss_gender.item(),
                      regression_loss_age.item(),
                      accuracy_expression,
                      accuracy_gender,
                      accuracy_age,
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
  parser.add_argument('--batch_size', type=int, default=350)
  parser.add_argument('--channel', type=int, default=3)
  parser.add_argument('--height', type=int, default=224)
  parser.add_argument('--width', type=int, default=224)
  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument('--epochs', type=int, default=150)
  parser.add_argument('--model', type=str, default='')
  parser.add_argument('--num_class', type=int, default=7)
  parser.add_argument('--step_size', type=int, default=50)
  parser.add_argument('--gamma', type=float, default=0.1)
  parser.add_argument('--print_freq', type=int, default=5)
  parser.add_argument('--save_freq', type=int, default=2)
  parser.add_argument('--train_dataset_path', type=str, default='dataset/emotion_340w_cut_aligned')
  parser.add_argument('--train_image_list', type=str, default='dataset/emotion_340w_cut_easy_data_train_nolandmark.csv')


  parser.add_argument('--visual_nrow', type=int, default=20)
  parser.add_argument('--visual_scale', type=int, default=2)
  parser.add_argument('--data_root', type=str, default='/home/lixy/ml_example/face-expression')
  parser.add_argument('--visual_env', type=str, default='train-env-py3')
  opt = parser.parse_args()
  train(opt)
  print('train finished')

