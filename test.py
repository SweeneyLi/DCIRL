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
import time
from visdom import Visdom
import numpy as np

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

logging.basicConfig(filename='test.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format=LOG_FORMAT,
                    datefmt=DATE_FORMAT
                    )
a = {'last fc grad': ([0.0009, 0.0012, -0.0003]), 'senior module linear': ([4.4867e-05, 6.8537e-05, -1.0721e-03]),
     'middle module linear': ([-6.2061e-05, 1.1691e-04, -1.2964e-04]),
     'basic module conv2': ([-0.0006, -0.0007, -0.0005]), 'basic module conv1': ([-0.0135, -0.0093, -0.0106])}
logging.debug('{gradient info}\n%s' % "\n".join(["%32s: %s" % (k, v) for k, v in a.items()]))
