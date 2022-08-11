"""
@File  :dataloader.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/23 3:00 PM
@Desc  :load different datasets
"""
import os
import random
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
from torchvision import transforms

random.seed(1)


def get_data_loader_dict(dataset_file, dataset_name, dataset_path, image_size, batch_size, shuffle=True, num_workers=4):
    data_loader_dict = {}
    for phase in ['train', 'val', 'test']:
        if dataset_name == 'miniImageNet':
            dataset = MiniImageNetDataset(dataset_path, image_size, dataset_file, phase)
        else:
            dataset = None

        assert batch_size - dataset.class_number > 0
        if phase != 'train':
            batch_size = batch_size * 3
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        data_loader_dict[phase] = data_loader

    return data_loader_dict


def get_split_data(data, phase):
    data_len = len(data)
    assert data_len >= 3
    if data_len < 5:
        train_number, val_number, test_number = data_len - 2, 1, 1
    else:
        train_number = int(0.6 * data_len)
        val_number = int(0.2 * data_len)
        test_number = data_len - train_number - val_number

    if phase == 'train':
        start_index, number = 0, train_number
    elif phase == 'val':
        start_index, number = train_number, val_number
    elif phase == 'test':
        start_index, number = train_number + val_number, test_number
    else:
        raise ValueError('Name of phase unknown %s' % phase)
    return data[start_index: start_index + number]


class MiniImageNetDataset(data.Dataset):

    def __init__(self, dataset_path, image_size=224, dataset_file='train', phase='train'):
        super(MiniImageNetDataset, self).__init__()

        self.image_size = image_size

        self.dataset_info = pd.read_csv(os.path.join(dataset_path, '%s.csv' % dataset_file))
        self.images_path = os.path.join(dataset_path, 'images')

        # label_dict
        # label index start from 1
        self.label_to_index_dict = {}
        self.index_to_label_dict = {}
        class_list = list(self.dataset_info['label'].unique())
        for i, class_name in enumerate(class_list, start=1):
            self.label_to_index_dict[class_name] = i
            self.index_to_label_dict[i] = class_name
        self.class_number = len(class_list)

        # image_dict: {label: filename_list}
        temp = self.dataset_info.groupby('label').agg({'filename': list}).reset_index()
        self.image_dict = dict(zip(temp['label'], temp['filename']))
        self.image_list = []
        for label in self.image_dict.keys():
            label_data = get_split_data(self.image_dict[label], phase)
            self.image_dict[label] = label_data
            self.image_list.extend(label_data)

        # data process
        self.resize = transforms.Resize([image_size[0], image_size[1]])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, sample_idx):
        image_name = self.image_list[sample_idx]
        return self.get_item(image_name)

    def get_item(self, image_name):
        label = self.label_to_index_dict.get(image_name.split('.')[0][:9])

        sample = Image.open(os.path.join(self.images_path, image_name))
        sample = self.resize(sample)
        sample = self.to_tensor(sample)
        sample = self.normalize(sample)

        return sample, torch.tensor(label)

    @staticmethod
    def randint(low, high, discard):
        random_idx = random.randint(low, high - 1)
        if random_idx >= discard:
            random_idx += 1
        return random_idx

    def get_contrast_batch(self, labels):
        small_batch_size = len(labels)
        same_samples = torch.zeros((small_batch_size, 3, self.image_size[0], self.image_size[1]))

        different_samples = torch.zeros((small_batch_size, 3, self.image_size[0], self.image_size[1]))
        different_labels = torch.zeros(small_batch_size, dtype=int)

        for i, label_idx in enumerate(labels):
            label_name = self.index_to_label_dict[label_idx.item()]

            # same
            the_image_list = self.image_dict[label_name]
            same_sample_filename = the_image_list[random.randint(0, len(the_image_list) - 1)]
            same_samples[i], _ = self.get_item(same_sample_filename)

            # different
            different_idx = self.randint(1, self.class_number, label_idx.item())
            different_label_name = self.index_to_label_dict[different_idx]
            the_image_list = self.image_dict[different_label_name]
            different_sample_file_name = the_image_list[random.randint(0, len(the_image_list) - 1)]

            different_samples[i], different_labels[i] = self.get_item(different_sample_file_name)

        return same_samples, different_samples, different_labels
