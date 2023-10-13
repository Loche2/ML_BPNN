import copy

import numpy as np
import pandas as pd


class BPNN(object):
    def __init__(self, layer_dims, learning_rate=0.1, seed=22, initializer='he', optimizer='gd'):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.seed = seed
        self.initializer = initializer
        self.optimizer = optimizer

        self.parameters = None
        self.costs = None

    def fit(self, x, y, num_epoch=100):
        m, n = x.shape
        layer_dims = copy.deepcopy(self.layer_dims)
        layer_dims.insert(0, n)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.parameters = xavier_initializer(layer_dims, self.seed)
        self.parameters, self.costs = self.gd_optimizer(x, y, self.parameters, num_epoch, self.learning_rate)
        return self

    def predict(self, x):
        if self.parameters is None:
            raise Exception('you must fit before predict')

        a_last, _ = self.forward_L_layer(x, self.parameters)
        if a_last.shape[1] == 1:
            predict = np.zeros(a_last.shape)
            predict[a_last >= 0.5] = 1
        else:
            predict = np.argmax(a_last, axis=1)
        return predict

    def compute_cost(self, y_hat, y):
        if y.ndim == 1:
            y = y.reshape(-1, -1)
        if y.shape[1] == 1:
            cost = cross_entropy_sigmoid(y_hat, y)
        return cost

    def backward_one_layer(self, da, cache, activation):
        (a_pre, w, b, z) = cache
        m = da.shape[0]

        dz = sigmoid_backward(da, z)
        dw = np.dot(dz.T, a_pre) / m
        db = np.sum(dz, axis=0, keepdims=True) / m
        da_pre = np.dot(dz, w)
        assert dw.shape == w.shape
        assert db.shape == b.shape
        assert da_pre.shape == a_pre.shape
        return da_pre, dw, db


def xavier_initializer(layer_dims, seed=22):
    np.random.seed(seed)
    parameters = {}
    num_L = len(layer_dims)
    for L in range(num_L - 1):
        temp_w = np.random.randn(layer_dims[L + 1], layer_dims[L]) * np.sqrt(1 / layer_dims[L])
        temp_b = np.zeros((1, layer_dims[L + 1]))
        parameters['W' + str(L + 1)] = temp_w
        parameters['b' + str(L + 1)] = temp_b
    return parameters


def cross_entropy_sigmoid(y_hat, y):
    """
    计算二分类的交叉熵
    :param y_hat: 模型输出值
    :param y: 样本真实值
    :return:
    """
    loss = -(np.dot(y.T, np.log(y_hat)) + np.dot(1 - y.T, np.log(1 - y_hat))) / y.shape[0]
    return np.squeeze(loss)


def sigmoid_backward(da, cache_z):
    a = 1 / (1 + np.exp(-cache_z))
    dz = da * a * (1 - a)
    assert dz.shape == cache_z.shape
    return dz



