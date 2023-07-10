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
    file_fid = pd.read_csv(os.path.join(opt.data_root, opt.train_image_list))
    self.image_list = file_fid['img_name']
    self.expression_list = file_fid['emotion_type']
    self.label_dict = {'angry':0,'disgust':1, 'fear':2, 'happy':3, 'sad':4, 'surprise':5, 'neutral':6}
    self.opt = opt
    self.totensor = transforms.ToTensor()
  def __len__(self):
    return len(self.image_list)
  def __getitem__(self, idx):
    image_path = self.image_list[idx]
    image_label = self.label_dict[self.expression_list[idx]]
    sample = Image.open(os.path.join(self.opt.data_root,'dataset/expression_dataset_120k/Emotion_Dataset_120k_aligned', image_path))
    sample = self.totensor(sample)
    image_label = torch.from_numpy(np.array(image_label).astype(np.long))
    return sample, image_label
