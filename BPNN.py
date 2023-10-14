import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from palmerpenguins import load_penguins

np.random.seed(22)

# 0.加载并处理palmerpenguins数据集
penguins = load_penguins()
penguins = penguins[['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
penguins.loc[penguins['species'] == 'Adelie', 'species'] = 0
penguins.loc[penguins['species'] == 'Gentoo', 'species'] = 1
penguins.loc[penguins['species'] == 'Chinstrap', 'species'] = 2
penguins.dropna(inplace=True)


def initializer(n_x, n_h, n_y):
    """
    1.初始化参数
    :param n_x: 输入层神经元个数
    :param n_h: 隐藏层神经元个数
    :param n_y: 输出层神经元个数
    :return: 初始化的参数
    """
    # 权重和偏置矩阵
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    # 通过字典保存参数
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    return parameters


def forward_propagation(X, parameters):
    """
    2.前向传播
    :param X: 输入
    :param parameters: 参数
    :return: a2, cache
    """
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    # 通过前向传播计算a2
    z1 = np.dot(w1, X) + b1
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))  # 隐层和输出层神经元都使用Sigmoid函数
    # 通过字典保存参数
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache


def compute_cost(a2, Y):
    """
    3.计算代价函数
    :param a2: 网络输出
    :param Y: 真实值
    :return: cost
    """
    m = 3  # 三类企鹅
    # 采用交叉熵（cross-entropy）作为代价函数
    log_probs = np.multiply(np.log(a2), Y) + np.multiply((1 - Y), np.log(1 - a2))
    cost = - np.sum(log_probs) / m
    return cost

