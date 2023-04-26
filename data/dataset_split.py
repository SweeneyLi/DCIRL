"""
@File  :dataset_split.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/8/17 20:17
@Desc  :
"""

import pandas as pd
import shutil
import os


def split_by_file(dataset_path, data_path, file_ext='csv'):
    csv_list = list(filter(lambda x: x.endswith(file_ext), os.listdir(dataset_path)))
    print('csv list: {}'.format(csv_list))
    for csv_file in csv_list:
        phase = csv_file.split('.')[0]
        print('phase: {}'.format(phase))

        phase_path = os.path.join(dataset_path, phase)
        os.mkdir(phase_path)

        csv_info = pd.read_csv(os.path.join(dataset_path, csv_file))

        label_list = list(csv_info['label'].unique())
        for label in label_list:
            os.mkdir(os.path.join(phase_path, label))

        for i, row in csv_info.iterrows():
            filename, label = row['filename'], row['label']

            ori_path = os.path.join(data_path, filename)
            new_path = os.path.join(phase_path, label, filename)
            shutil.copyfile(ori_path, new_path)
            print('copy {} to {}'.format(ori_path, new_path))


if __name__ == '__main__':
    dataset_path = '/home/share/image_classification/mini-imagenet'
    data_path = os.path.join(dataset_path, 'images')

    split_by_file(dataset_path, data_path)
