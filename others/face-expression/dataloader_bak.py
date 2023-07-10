import os
import random
import torch
import torch.utils.data as data
from torchvision import transforms
import pdb
import pandas as pd
from PIL import Image
import numpy as np

class Expression_Dataset(data.Dataset):
  def __init__(self, opt):
    super(Expression_Dataset, self).__init__()

    if opt.phase == 'train':
        file_fid = pd.read_csv(os.path.join(opt.data_root, opt.train_image_list))
        self.dataset_path = opt.train_dataset_path
    elif opt.phase == 'test':
        file_fid = pd.read_csv(os.path.join(opt.data_root, opt.test_image_list))
        self.dataset_path = opt.test_dataset_path
    else:
        raise RuntimeError("please select train or test")
 
    self.image_list = file_fid['img_name']
    self.expression_list = file_fid['emotion_type']
    self.gender_list = file_fid['gender_type']
    self.age_list = file_fid['age']
    self.label_dict = {'angry':0,'disgust':1, 'fear':2, 'happy':3, 'sad':4, 'surprise':5, 'neutral':6, '0':0, '1':1}
    self.opt = opt
    #self.crop = transforms.CenterCrop((224,224))
    self.resize = transforms.Resize([224,224])
    self.totensor = transforms.ToTensor()
    self.normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

  def __len__(self):
    return len(self.image_list)
  def __getitem__(self, idx):
    image_path = self.image_list[idx]
    
    image_expression_label = self.label_dict[self.expression_list[idx]]
    image_gender_label = self.label_dict[str(self.gender_list[idx])]
    image_age_label = self.age_list[idx]
    
    sample = Image.open(os.path.join(self.opt.data_root, self.dataset_path, image_path))
    #sample = self.crop(sample)
    sample = self.resize(sample)
    sample = self.totensor(sample)
    sample = self.normalize(sample)
    image_expression_label = torch.from_numpy(np.array(image_expression_label).astype(np.long))
    image_gender_label = torch.from_numpy(np.array(image_gender_label).astype(np.long))
    image_age_label = torch.from_numpy(np.array(image_age_label).astype(np.long))
    return sample, image_expression_label, image_gender_label, image_age_label
