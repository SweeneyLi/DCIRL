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
    def __init__(self, abstraction_coefficient, contrast_coefficient):
        self.abstraction_coefficient = abstraction_coefficient
        self.contrast_coefficient = contrast_coefficient

    def get_contrast_abstraction_loss(self, feature_origin, feature_same, feature_different):
        # normalization
        feature_origin_norm = self.get_z_score_matrix(feature_origin, dimension=2)
        feature_same_norm = self.get_z_score_matrix(feature_same, dimension=2)
        feature_different_norm = self.get_z_score_matrix(feature_different, dimension=2)

        # calculate similarity
        feature_size = feature_origin.size(1)
        cosine_similarity_same = torch.matmul(feature_origin_norm, feature_same_norm.T) / feature_size
        cosine_similarity_different = torch.matmul(feature_origin_norm, feature_different_norm.T) / feature_size

        # get whole loss
        abstraction_loss = 1 - cosine_similarity_same.mean()
        contrast_loss = 1 + cosine_similarity_different.mean()

        return abstraction_loss, contrast_loss, self.abstraction_coefficient * abstraction_loss + self.contrast_coefficient * contrast_loss

    @staticmethod
    def get_z_score_matrix(matrix, dimension=3):
        calculate_dimension = dimension - 1
        return (matrix - matrix.mean(calculate_dimension).unsqueeze(calculate_dimension)) / (
                matrix.std(calculate_dimension).unsqueeze(calculate_dimension) + 1e-6)
