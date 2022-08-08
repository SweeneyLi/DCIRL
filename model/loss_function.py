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

    def get_whole_loss(self, feature_origin, feature_same, feature_different):
        # normalization
        feature_origin_norm = self.get_z_score_matrix(feature_origin, dimension=2)
        feature_same_norm = self.get_z_score_matrix(feature_same, dimension=2)
        feature_different_norm = self.get_z_score_matrix(feature_different, dimension=2)

        # calculate similarity
        feature_size = feature_origin.size(1)
        cosine_similarity_same = torch.matmul(
            feature_origin_norm.unsqueeze(1), feature_same_norm.unsqueeze(1).transpose(1, 2).cuda()) / feature_size
        cosine_similarity_different = torch.matmul(
            feature_origin_norm.unsqueeze(1), feature_different_norm.unsqueeze(1).transpose(1, 2).cuda()) / feature_size

        # get whole loss
        whole_loss_same = 1 - cosine_similarity_same.mean()
        whole_loss_different = cosine_similarity_different.mean()
        whole_loss = self.whole_different_coefficient * whole_loss_different + whole_loss_same

        return whole_loss, whole_loss_same, whole_loss_different

    def get_contrast_loss(self, feature_origin, feature_same, feature_different):
        feature_origin_norm = self.get_z_score_matrix(feature_origin, dimension=3)
        feature_same_norm = self.get_z_score_matrix(feature_same, dimension=3)
        feature_different_norm = self.get_z_score_matrix(feature_different, dimension=3)

        contrast_common_loss = self.get_common_loss(feature_origin_norm, feature_same_norm)
        contrast_different_loss = self.get_different_loss(feature_origin_norm, feature_different_norm)
        return self.common_coefficient * contrast_common_loss + self.different_coefficient * contrast_different_loss, \
               contrast_common_loss, contrast_different_loss

    def get_common_loss(self, feature_origin_norm, feature_same_norm):
        on_diag_loss, off_diag_loss = self.get_cross_correlation_matrix_loss(
            feature_origin_norm, feature_same_norm
        )
        return on_diag_loss + self.off_diag_coefficient * off_diag_loss

    def get_different_loss(self, feature_origin_norm, feature_different_norm):
        on_diag_loss, off_diag_loss = self.get_cross_correlation_matrix_loss(
            feature_origin_norm, feature_different_norm
        )
        return on_diag_loss + self.off_diag_coefficient * off_diag_loss

    def get_cross_correlation_matrix_loss(self, feature_origin_norm, feature_contrast_norm):
        batch_size, feature_size, _ = feature_origin_norm.size()
        cross_correlation_matrix = torch.matmul(
            feature_origin_norm, feature_contrast_norm.transpose(1, 2).cuda()
        ) / feature_size
        on_diag_loss = torch.diagonal(cross_correlation_matrix, dim1=1, dim2=2).pow_(2).sum() / batch_size
        off_diag_loss = self.get_off_diagonal(cross_correlation_matrix).pow_(2).sum() / batch_size

        return on_diag_loss, off_diag_loss

    @staticmethod
    def get_z_score_matrix(matrix, dimension=3):
        calculate_dimension = dimension - 1
        return (matrix - matrix.mean(calculate_dimension).unsqueeze(calculate_dimension)) / (
                matrix.std(calculate_dimension).unsqueeze(calculate_dimension) + 1e-6)

    @staticmethod
    def get_off_diagonal(matrix):
        batch_size, x, y = matrix.shape
        return matrix.view(batch_size, x * y)[:, :-1].view(batch_size, x - 1, x + 1)[:, :, 1:].flatten()
