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
    def __init__(self, same_coefficient, different_coefficient):
        self.same_coefficient = same_coefficient
        self.different_coefficient = different_coefficient

    def get_same_different_loss(self, feature_origin, feature_same, feature_different):
        # normalization
        feature_origin_norm = self.get_z_score_matrix(feature_origin, dimension=2)
        feature_same_norm = self.get_z_score_matrix(feature_same, dimension=2)
        feature_different_norm = self.get_z_score_matrix(feature_different, dimension=2)

        # calculate similarity
        feature_size = feature_origin.size(1)
        cosine_similarity_same = torch.matmul(feature_origin_norm, feature_same_norm.T) / feature_size
        cosine_similarity_different = torch.matmul(feature_origin_norm, feature_different_norm.T) / feature_size

        # get whole loss
        same_loss = 1 - cosine_similarity_same.mean()
        different_loss = 1 + cosine_similarity_different.mean()

        return same_loss, different_loss, self.same_coefficient * same_loss + self.different_coefficient * different_loss

    @staticmethod
    def get_z_score_matrix(matrix, dimension=3):
        calculate_dimension = dimension - 1
        return (matrix - matrix.mean(calculate_dimension).unsqueeze(calculate_dimension)) / (
                matrix.std(calculate_dimension).unsqueeze(calculate_dimension) + 1e-6)
