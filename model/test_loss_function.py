"""
@File  :test_loss_function.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/24 3:48 PM
@Desc  :
"""
import os
import torch

from model.loss_function import DCLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dc_loss = DCLoss(same_coefficient=1,
                 different_coefficient=1
                 )


def prepare_data():
    # 5 * (2 * 3)
    batch_split_features = torch.Tensor([
        [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]],
        [[0.2, 0.3, 0.4], [0.8, 0.7, 0.6]],
        [[0.3, 0.4, 0.5], [0.7, 0.6, 0.5]],
        [[0.4, 0.5, 0.6], [0.6, 0.5, 0.4]],
        [[0.5, 0.6, 0.7], [0.5, 0.4, 0.3]],
    ]).float()
    label = torch.Tensor(
        [0, 1, 0, 0, 1]
    )
    # 3 * (2 * 3)
    feature_origin = torch.Tensor([
        [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]],
        [[0.2, 0.3, 0.4], [0.8, 0.7, 0.6]],
        [[0.3, 0.4, 0.5], [0.7, 0.6, 0.5]]
    ]).float()
    feature_same = torch.Tensor([
        [[0.4, 0.5, 0.6], [0.6, 0.5, 0.4]],
        [[0.5, 0.6, 0.7], [0.5, 0.4, 0.3]],
        [[0.4, 0.5, 0.6], [0.6, 0.5, 0.4]],
    ]).float()
    feature_different = torch.Tensor([
        [[0.5, 0.6, 0.7], [0.5, 0.4, 0.3]],
        [[0.4, 0.5, 0.6], [0.6, 0.5, 0.4]],
        [[0.5, 0.6, 0.7], [0.5, 0.4, 0.3]],
    ]).float()
    return batch_split_features.cuda(), label.cuda(), feature_origin.cuda(), feature_same.cuda(), feature_different.cuda()


class TestDCLoss:

    def setup_class(self):
        self.batch_split_features, self.label, self.feature_origin, self.feature_same, self.feature_different = prepare_data()
    #
    # def test_get_off_diagonal(self):
    #     matrix = torch.arange(2 * 3 * 3).reshape(2, 3, 3)
    #     expected_value = torch.tensor([1, 2, 3, 5, 6, 7, 10, 11, 12, 14, 15, 16])
    #
    #     result = dc_loss.get_off_diagonal(matrix)
    #
    #     assert result.equal(expected_value)
    #
    # def test_get_cross_correlation_matrix(self):
    #     # f: 3 * (2 * 3)
    #     c_m = dc_loss.get_cross_correlation_matrix(self.feature_origin, self.feature_same)
    #
    #     expect_value = torch.zeros((3, 2, 2), dtype=float).cuda()
    #     for i in range(3):  # batch
    #         feature_a = self.feature_origin[i]  # 2 * 3
    #         feature_b = self.feature_same[i]  # 2 * 3
    #
    #         feature_a = feature_a / torch.norm(feature_a, dim=1).unsqueeze(1)
    #         feature_b = feature_b / torch.norm(feature_b, dim=1).unsqueeze(1)
    #
    #         expect_value[i] = torch.mm(feature_a, feature_b.T)
    #
    #     assert c_m.float().equal(expect_value.float())

    # def test_off_diagonal_loss(self):
    #     x = dc_loss.off_diagonal_loss(
    #         torch.Tensor([
    #             [0.1, 0.2, 0.3],
    #             [0.4, 0.5, 0.6],
    #             [0.7, 0.8, 0.9]
    #         ]))
    #     expect_value = torch.tensor([
    #         0.2, 0.3, 0.4, 0.6, 0.7, 0.8
    #     ]).pow_(2).mean()
    #     assert x.equal(expect_value)
