"""
@File  :loss_function.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/23 8:39 PM
@Desc  :different loss function
"""
import torch
import torch.nn.functional as F
from random import randint


class DCLoss:
    def __init__(self, same_coefficient, different_coefficient):
        self.same_coefficient = same_coefficient
        self.different_coefficient = different_coefficient

    def get_same_different_loss(self, feature_origin, feature_same, feature_different):
        # normalization
        feature_origin_norm = F.normalize(feature_origin, p=2, dim=1)
        feature_same_norm = F.normalize(feature_same, p=2, dim=1)
        feature_different_norm = F.normalize(feature_different, p=2, dim=1)

        # calculate similarity
        cosine_similarity_same = torch.matmul(feature_origin_norm.unsqueeze(1), feature_same_norm.unsqueeze(2))
        cosine_similarity_different = torch.matmul(feature_origin_norm.unsqueeze(1), feature_different_norm.unsqueeze(2))

        # get whole loss
        same_loss = 1 - cosine_similarity_same.mean()
        different_loss = cosine_similarity_different.mean()

        return self.same_coefficient * same_loss + self.different_coefficient * different_loss, same_loss, different_loss

    @staticmethod
    def get_z_score_matrix(matrix, dimension=3):
        calculate_dimension = dimension - 1
        return (matrix - matrix.mean(calculate_dimension).unsqueeze(calculate_dimension)) / (
                matrix.std(calculate_dimension).unsqueeze(calculate_dimension) + 1e-6)
