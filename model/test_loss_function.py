"""
@File  :test_loss_function.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/24 3:48 PM
@Desc  :
"""
import torch

from model.loss_function import DCLoss

dc_loss = DCLoss(batch_size=5, class_num=2, off_diag_coefficient=0.005, independent_coefficient=1, common_coefficient=1,
                 different_coefficient=1, whole_different_coefficient=1, accuracy_same_threshold=0.8,
                 accuracy_different_threshold=0.2)


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
    return batch_split_features, label, feature_origin, feature_same, feature_different


class TestDCLoss:

    def setup_class(self):
        self.batch_split_features, self.label, self.feature_origin, self.feature_same, self.feature_different = prepare_data()

    def test_get_different_labels(self):
        for i in range(1, 100):
            assert i != dc_loss.get_different_labels(i)

    def test_get_origin_same_and_different_features(self):
        f_o, f_s, f_d = dc_loss.get_origin_same_and_different_features(self.batch_split_features, self.label)
        assert f_o.equal(self.feature_origin)
        assert f_s.equal(self.feature_same)
        assert f_d.equal(self.feature_different)

    # def test_get_whole_loss_and_correct_number(self):
    #     assert False
    #
    # def test_get_contrast_loss(self):
    #     assert False
    #
    # def test_get_common_loss(self):
    #     assert False
    #
    # def test_get_different_loss(self):
    #     assert False
    #
    # def test_get_independent_loss(self):
    #     assert False

    def test_get_cross_correlation_matrix(self):
        # f: 3 * (2 * 3)
        c_m = dc_loss.get_cross_correlation_matrix(self.feature_origin, self.feature_same)

        expect_value = torch.zeros((3, 2, 2), dtype=float)
        for i in range(3):  # batch
            feature_a = self.feature_origin[i]  # 2 * 3
            feature_b = self.feature_same[i]  # 2 * 3

            feature_a = feature_a / torch.norm(feature_a, dim=1).unsqueeze(1)
            feature_b = feature_b / torch.norm(feature_b, dim=1).unsqueeze(1)

            expect_value[i] = torch.mm(feature_a, feature_b.T)

        assert c_m.float().equal(expect_value.float())

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
