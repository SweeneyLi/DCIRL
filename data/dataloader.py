"""
@File  :dataloader.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/23 3:00 PM
@Desc  :load different datasets
"""
import os
import random

from torch.utils.data import DataLoader
import torch.utils.data as data
from torchvision import transforms
from PIL import Image


def get_data_loader_dict(dataset_name, dataset_path, image_size, phase, batch_size, shuffle=True, num_workers=4):
    data_loader_dict = {}
    for phase in ['train', 'val', 'test']:
        if dataset_name == 'cat_dog':
            dataset = CatDogDataset(dataset_path, image_size, phase)
        else:
            dataset = None
            
        assert batch_size - dataset.class_number > 0
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size - dataset.class_number,
            shuffle=shuffle,
            num_workers=num_workers
        )
        data_loader_dict[phase] = data_loader
    return data_loader_dict


class CatDogDataset(data.Dataset):

    def __init__(self, dataset_path, image_size=224, phase='train'):
        super(CatDogDataset, self).__init__()

        # image
        self.dataset_path = dataset_path
        self.image_name_list = os.listdir(dataset_path)
        assert len(self.image_name_list) > 1200

        self.class_name = ['cat', 'dog']
        self.class_number = len(self.class_name)
        self.image_dict = {}
        self.label_dict = {}

        if phase == 'train':
            start_index, end_index = 0, 1000
        elif phase == 'val':
            start_index, end_index = 1000, 1200
        elif phase == 'test':
            start_index, end_index = 1200, 1400
        else:
            raise Exception('wrong phase!')

        temp = []
        for i, name in enumerate(self.class_name):
            self.image_dict[name] = list(filter(
                lambda x: x.startswith(name), self.image_name_list))[start_index: end_index]
            temp.extend(self.image_dict[name])
            self.label_dict[name] = i
        self.image_name_list = temp  # for test
        print('The label dict:\n', self.label_dict)

        # data process
        self.resize = transforms.Resize([image_size, image_size])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, sample_idx):
        image_path = self.image_name_list[sample_idx]
        label = self.label_dict.get(image_path.split('.')[0], '-1')

        sample = Image.open(os.path.join(self.dataset_path, image_path))
        sample = self.resize(sample)
        sample = self.to_tensor(sample)
        sample = self.normalize(sample)

        return sample, label

    def get_contrast_batch(self, idx_list=None):
        if idx_list is None:
            idx_list = []
            for a_class in self.class_name:
                idx_list.append(random.randint(0, len(self.image_dict[a_class])))

        contrast_samples = []
        contrast_labels = []
        for idx in idx_list:
            sample, label = self.__getitem__(idx)
            contrast_samples.append(sample)
            contrast_labels.append(label)
        return contrast_samples, contrast_labels
