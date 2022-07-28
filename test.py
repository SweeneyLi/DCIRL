"""
@File  :test.py
@Author:SweeneyLi
@Email :sweeneylee.gm@gmail.com
@Date  :2022/7/21 11:11 AM
@Desc  :test the model
"""
import numpy
import os
import pytest
import random
import logging

logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                    filename='new.log',
                    filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志, a是追加模式，默认如果不写的话，就是追加模式
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'  # 日志格式
                    )
logging.debug('debug')


# random.seed(1)

# print(os.getcwd())
# a = [1,2,3,4,5,6]
# random.shuffle(a)
# print(a)
