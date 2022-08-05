"""
@File  :loss_function.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/23 8:39 PM
@Desc  :different loss function
"""
import torch
from random import randint


class DCLoss:
    def __init__(self, batch_size, class_num,
                 whole_different_coefficient,
                 off_diag_coefficient,
                 common_coefficient,
                 different_coefficient
                 ):
        self.whole_different_coefficient = whole_different_coefficient
        self.off_diag_coefficient = off_diag_coefficient
        self.common_coefficient = common_coefficient
        self.different_coefficient = different_coefficient
        self.batch_size = batch_size
        self.class_num = class_num

    def get_different_labels(self, label):
        return (label + randint(1, self.class_num - 1)) % self.class_num

    def get_origin_same_and_different_features(self, batch_features, batch_labels):
        origin_features_num = batch_features.size(0) - self.class_num
        contrast_features_num = self.class_num
        origin_features, contrast_features = torch.split(batch_features, [origin_features_num, contrast_features_num])
        origin_labels, _ = torch.split(batch_labels, [origin_features_num, contrast_features_num])

        same_features, different_features = torch.zeros(origin_features.shape), torch.zeros(origin_features.shape)

        for i in range(origin_features.size(0)):
            label_index = int(origin_labels[i].item())
            same_features[i] = contrast_features[label_index]
            different_features[i] = contrast_features[self.get_different_labels(label_index)]
        return origin_features, same_features, different_features

    def get_whole_loss(self, feature_origin, feature_same, feature_different):
        # calculate similarity
        feature_origin_norm = feature_origin / torch.norm(feature_origin, dim=1).unsqueeze(1)
        feature_same_norm = feature_same / torch.norm(feature_same, dim=1).unsqueeze(1)
        feature_different_norm = feature_different / torch.norm(feature_different, dim=1).unsqueeze(1)

        cosine_similarity_same = torch.matmul(
            feature_origin_norm.unsqueeze(1), feature_same_norm.unsqueeze(1).transpose(1, 2).cuda())
        cosine_similarity_different = torch.matmul(
            feature_origin_norm.unsqueeze(1), feature_different_norm.unsqueeze(1).transpose(1, 2).cuda())

        # get whole loss
        whole_loss_same = 1 - cosine_similarity_same.mean()
        whole_loss_different = cosine_similarity_different.mean()
        whole_loss = self.whole_different_coefficient * whole_loss_different + whole_loss_same

        return whole_loss, whole_loss_same, whole_loss_different

    def get_contrast_loss(self, feature_origin, feature_same, feature_different):
        contrast_common_loss = self.get_common_loss(feature_origin, feature_same)
        contrast_different_loss = self.get_different_loss(feature_origin, feature_different)
        return self.common_coefficient * contrast_common_loss + self.different_coefficient * contrast_different_loss, \
               contrast_common_loss, contrast_different_loss

    def get_common_loss(self, feature_origin, feature_same):
        on_diag_loss, off_diag_loss = self.get_cross_correlation_matrix_loss(feature_origin, feature_same)
        return on_diag_loss + self.off_diag_coefficient * off_diag_loss

    def get_different_loss(self, feature_origin, feature_different):
        on_diag_loss, off_diag_loss = self.get_cross_correlation_matrix_loss(feature_origin, feature_different)
        return on_diag_loss + self.off_diag_coefficient * off_diag_loss

    def get_cross_correlation_matrix_loss(self, feature_origin, feature_contrast):
        cross_correlation_matrix_different = self.get_cross_correlation_matrix(feature_origin, feature_contrast)

        temp = torch.diagonal(cross_correlation_matrix_different, dim1=1, dim2=2).pow_(2)
        off_diag_number = self.get_off_diag_matrix_number(cross_correlation_matrix_different.size(), temp.size())
        off_diag_loss = (cross_correlation_matrix_different.pow_(2).sum() - temp.sum()) / off_diag_number

        on_diag_loss = torch.diagonal(cross_correlation_matrix_different, dim1=1, dim2=2).pow_(2).mean()
        return on_diag_loss, off_diag_loss

    @staticmethod
    def get_off_diag_matrix_number(matrix_size, on_diag_size):
        batch_size, x, y = matrix_size
        _, m = on_diag_size
        return batch_size * (x * y - m)

    @staticmethod
    def get_cross_correlation_matrix(feature_a, feature_b):
        # dimension: batch * (x * y)
        feature_a = feature_a / torch.norm(feature_a, dim=2).unsqueeze(2)
        feature_b = feature_b / torch.norm(feature_b, dim=2).unsqueeze(2)
        return torch.matmul(feature_a, feature_b.transpose(1, 2).cuda())

    # @staticmethod
    # def get_off_diagonal(matrix):
    #     batch_size, x, y = matrix.shape
    #     matrix.view(batch_size, x * y).view(batch_size, x - 1, x + 1)[:, :, 1:].view(batch_size, -1)
