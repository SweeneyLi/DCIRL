"""
@File  :test_tools.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/8/3 16:35
@Desc  :
"""

from tools import *
import torch
import numpy


class TestTools:
    def test_calculate_class_correct(self):
        scores = torch.tensor([
            [0.9, 0.1, 0.2],
            [0.8, 0.9, 0.2],
            [0.3, 0.1, 0.9],
            [0.9, 0.1, 0.9],
        ])
        labels = torch.tensor([1, 1, 3, 1])
        a, b = numpy.array([0, 2, 0, 1]), numpy.array([0, 3, 0, 1])

        r_a, r_b = calculate_class_correct(scores, labels)

        assert (a == r_a).all()
        assert (b == r_b).all()

    def test_calculate_correct(self):
        scores = torch.tensor([
            [0.9, 0.1, 0.2],
            [0.8, 0.9, 0.2],
            [0.3, 0.1, 0.9],
            [0.9, 0.1, 0.9],
        ])
        labels = torch.tensor([1, 1, 3, 1])
        e = torch.tensor(3)

        r = calculate_correct(scores, labels)

        assert r.equal(e)
