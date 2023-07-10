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
from PIL import Image
from torchvision import transforms
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
label_dict = {'0':'angry','1':'disgust', '2':'fear', '3':'happy', '4':'sad', '5':'surprise', '6':'neutral'}
def find_newest_model():
    models_path = '/home/shiyh/face-expression/checkout'
    models = os.listdir(models_path)
    models.sort(key=lambda fn:os.path.getmtime(models_path + "/" + fn))
    model_new = os.path.join(models_path,models[-1])   
    print(model_new)
    return model_new

def test(opt):
    #pdb.set_trace()
    model = Face_Expression(opt)
    model = model.cuda()
    model=torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(find_newest_model()))
    

    test_dataset = Expression_Dataset(opt)
    test_dataload = DataLoader(
                                dataset = test_dataset,
                                batch_size = opt.batch_size,
                                shuffle=True,
                                num_workers = 4)
    
    model = model.eval()
    correct_age = 0

    criterion_softmax = nn.CrossEntropyLoss()
    #pdb.set_trace()
    correct_expression=0
    correct_gender=0
    sample_num=0

    for i, (samples, labels_expression, labels_gender, labels_age) in enumerate(test_dataload):
      sample_num += samples.shape[0]

      samples = Variable(samples.cuda())
      labels_expression = Variable(labels_expression.cuda())
      labels_gender = Variable(labels_gender.cuda())
      labels_age = Variable(labels_age.cuda())

      cls_expression_prob,cls_age_prob,cls_gender_prob,x,x2,x3 = model(samples)
      
      _, predicted_expression = torch.max(cls_expression_prob, 1)
      _, predicted_gender = torch.max(cls_gender_prob, 1)

      #total += labels_age.size(0)
      res=abs(torch.squeeze(cls_age_prob,1).detach().cpu().numpy().astype(np.int)-labels_age.detach().cpu().numpy())
      correct_age += np.sum(np.where(res<6,1,0) == 1)
      
      correct_expression += (predicted_expression==labels_expression).sum().item()*1.0
      correct_gender += (predicted_gender==labels_gender).sum().item()*1.0
      #accuracy_expression = (predicted_expression==labels_expression).sum().item()*1.0 / samples.shape[0]
      #accuracy_gender = (predicted_gender==labels_gender).sum().item()*1.0 / samples.shape[0]
      #accuracy_age = correct/total 

    
    print("expression accuracy: %.4f"%(correct_expression/sample_num))
    print("igender accuracy: %.4f"%(correct_gender/sample_num))
    print("age accuracy: %.4f"%(correct_age/sample_num))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--phase', type=str, default='test')
  parser.add_argument('--cuda', action='store_true', default=True)
  parser.add_argument('--batch_size', type=int, default=300)
  parser.add_argument('--channel', type=int, default=3)
  parser.add_argument('--height', type=int, default=224)
  parser.add_argument('--width', type=int, default=224)
  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument('--epochs', type=int, default=100)
  parser.add_argument('--model', type=str, default='checkout/model_final_1-29.pth')
  parser.add_argument('--num_class', type=int, default=7)
  parser.add_argument('--step_size', type=int, default=20)
  parser.add_argument('--gamma', type=float, default=0.1)
  
  
  parser.add_argument('--test_image_list', type=str, default='dataset/emotion_340w_cut_easy_data_test_nolandmark.csv')
  parser.add_argument('--test_dataset_path', type=str, default='dataset/emotion_340w_cut_aligned')


  parser.add_argument('--print_freq', type=int, default=5)
  parser.add_argument('--save_freq', type=int, default=2)
  parser.add_argument('--visual_nrow', type=int, default=20)
  parser.add_argument('--visual_scale', type=int, default=4)
  parser.add_argument('--data_root', type=str, default='/home/shiyh/face-expression')
  parser.add_argument('--visual_env', type=str, default='train-env-py3')
  opt = parser.parse_args()
  test(opt)
  print('test finished')

