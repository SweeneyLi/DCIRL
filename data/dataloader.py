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

# random.seed(1)

get_class_fun_dict = {
    'tiered_imagenet': lambda x: x.split('.')[0][:9],
    'miniImageNet': lambda x: x.split('.')[0][:9]
}


def get_data_loader_dict(phase, root_path, dataset_name, image_size,
                         n_ways, k_shots, query_shots,
                         batch_size, shuffle=True, num_workers=4):
    data_loader_dict = {}

    if phase == 'pretrain':
        sub_phase_list = ['train', 'val', 'test']
    else:
        sub_phase_list = ['train', 'test']

    for sub_phase in sub_phase_list:
        dataset = ImageClassificationDataset(
            root_path=root_path, dataset_name=dataset_name,
            get_class_fun=get_class_fun_dict[dataset_name],
            image_size=image_size,
            phase=phase, sub_phase=sub_phase,
            n_ways=n_ways, k_shots=k_shots, query_shots=query_shots)

        if phase == 'pretrain' or (sub_phase == 'train' and k_shots > 1):
            batch_size = batch_size / 3
            num_workers = num_workers
        else:
            num_workers = 1

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        data_loader_dict[sub_phase] = data_loader

    return data_loader_dict


class ImageClassificationDataset(data.Dataset):
    """
    dataset format:
    Dataset
    - train
        - 001
            - 001001.jpg
    - test
    - val
    """

    def __init__(self, root_path, dataset_name,
                 get_class_fun,
                 image_size=224,
                 phase='pretrain', sub_phase='train',
                 n_ways=None, k_shots=None, query_shots=None):
        super(ImageClassificationDataset, self).__init__()

        self.images_path = os.path.join(root_path, dataset_name, phase.split('-')[1])
        self.get_class_name_fun = get_class_fun
        self.image_size = image_size

        # label_dict
        # label index start from 0
        class_list = os.listdir(self.images_path)
        if phase != 'pretrain':
            class_list = random.sample(class_list, k=n_ways)

        self.class_number = len(class_list)
        self.label_to_index_dict = {}
        self.index_to_label_dict = {}
        for i, class_name in enumerate(class_list, start=0):
            self.label_to_index_dict[class_name] = i
            self.index_to_label_dict[i] = class_name

        # image_dict: {label: filename_list}
        self.image_dict = {}
        self.image_list = []
        for label in class_list:
            label_data = os.listdir(os.path.join(self.images_path, label))
            label_data = self.split_dataset(label_data, phase, sub_phase, k_shots, query_shots)
            self.image_dict[label] = label_data
            self.image_list.extend(label_data)

        # data process
        self.resize = transforms.Resize([image_size[0], image_size[1]])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def split_dataset(self, data, phase, sub_phase, k_shots, query_shots):
        random.shuffle(data)
        data_len = len(data)
        if phase == 'pretrain':
            train_number = int(0.6 * data_len)
            val_number = int(0.2 * data_len)
            test_number = data_len - train_number - val_number
        else:
            train_number = k_shots
            val_number = query_shots
            test_number = 0

        start_index, number = 0, 0
        if sub_phase == 'train':
            start_index, number = 0, train_number
        elif sub_phase == 'val':
            start_index, number = train_number, val_number
        elif sub_phase == 'test':
            start_index, number = train_number + val_number, test_number

        return data[start_index: start_index + number]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, sample_idx):
        image_name = self.image_list[sample_idx]
        return self.get_item(image_name)

    def get_item(self, image_name):
        class_name = self.get_class_name_fun(image_name)
        label_index = self.label_to_index_dict.get(class_name)

        sample = Image.open(os.path.join(self.images_path, class_name, image_name))
        sample = self.resize(sample)
        sample = self.to_tensor(sample)
        sample = self.normalize(sample)

        return sample, torch.tensor(label_index)

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
            different_idx = self.randint(0, self.class_number - 1, label_idx.item())
            different_label_name = self.index_to_label_dict[different_idx]
            the_image_list = self.image_dict[different_label_name]
            different_sample_file_name = the_image_list[random.randint(0, len(the_image_list) - 1)]

            different_samples[i], different_labels[i] = self.get_item(different_sample_file_name)

        return same_samples, different_samples, different_labels
