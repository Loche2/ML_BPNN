import copy

import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split

import BPNNutil


class BpNN(object):
    def __init__(self, layer_dims_, learning_rate=0.01, seed=22):
        self.layer_dims_ = layer_dims_
        self.learning_rate = learning_rate
        self.seed = seed
        self.costs = None
        self.parameters_ = None

    def fit(self, X_, y_, num_epochs=100):
        m, n = X_.shape
        layer_dims_ = copy.deepcopy(self.layer_dims_)
        layer_dims_.insert(0, n)
        if y_.ndim == 1:
            y_ = y_.reshape(-1, 1)

        self.parameters_ = BPNNutil.xavier_initializer(layer_dims_, self.seed)
        self.parameters_, self.costs = self.optimizer_gd(X_, y_, self.parameters_, num_epochs, self.learning_rate)
        return self

    def predict(self, X_):
        if not hasattr(self, "parameters_"):
            raise Exception('you must to fit before predict.')

        a_last, _ = self.forward_L_layer(X_, self.parameters_)
        if a_last.shape[1] == 1:
            predict_ = np.zeros(a_last.shape)
            predict_[a_last >= 0.5] = 1
        else:
            predict_ = np.argmax(a_last, axis=1)
        return predict_

    def compute_cost(self, y_hat_, y_):
        if y_.ndim == 1:
            y_ = y_.reshape(-1, 1)
        if y_.shape[1] == 1:
            cost = BPNNutil.cross_entry_sigmoid(y_hat_, y_)
        else:
            cost = BPNNutil.cross_entry_softmax(y_hat_, y_)
        return cost

    def backward_one_layer(self, da_, cache_, activation_):
        (a_pre_, w_, b_, z_) = cache_
        m = da_.shape[0]

        assert activation_ in ('sigmoid', 'softmax')
        if activation_ == 'sigmoid':
            dz_ = BPNNutil.sigmoid_backward(da_, z_)
        elif activation_ == 'softmax':
            dz_ = BPNNutil.softmax_backward(da_, z_)

        dw = np.dot(dz_.T, a_pre_) / m
        db = np.sum(dz_, axis=0, keepdims=True) / m
        da_pre = np.dot(dz_, w_)

        assert dw.shape == w_.shape
        assert db.shape == b_.shape
        assert da_pre.shape == a_pre_.shape

        return da_pre, dw, db

    def backward_L_layer(self, a_last, y_, caches):
        grads = {}
        L = len(caches)

        if y_.ndim == 1:
            y_ = y_.reshape(-1, 1)

        if y_.shape[1] == 1:  # 目标值只有一列表示为二分类
            da_last = -(y_ / a_last - (1 - y_) / (1 - a_last))
            da_pre_L_1, dwL_, dbL_ = self.backward_one_layer(da_last, caches[L - 1], 'sigmoid')

        else:  # 经过one hot，表示为多分类

            # 在计算softmax的梯度时，可以直接用 dz = a - y可计算出交叉熵损失函数对z的偏导， 所以这里第一个参数输入直接为y_
            da_pre_L_1, dwL_, dbL_ = self.backward_one_layer(y_, caches[L - 1], 'softmax')

        grads['da' + str(L)] = da_pre_L_1
        grads['dW' + str(L)] = dwL_
        grads['db' + str(L)] = dbL_

        for i in range(L - 1, 0, -1):
            da_pre_, dw, db = self.backward_one_layer(grads['da' + str(i + 1)], caches[i - 1], 'sigmoid')

            grads['da' + str(i)] = da_pre_
            grads['dW' + str(i)] = dw
            grads['db' + str(i)] = db

        return grads

    def forward_one_layer(self, a_pre_, w_, b_, activation_):
        z_ = np.dot(a_pre_, w_.T) + b_
        assert activation_ in ('sigmoid', 'softmax')

        if activation_ == 'sigmoid':
            a_ = BPNNutil.sigmoid(z_)
        elif activation_ == 'softmax':
            a_ = BPNNutil.softmax(z_)

        cache_ = (a_pre_, w_, b_, z_)  # 将向前传播过程中产生的数据保存下来，在向后传播过程计算梯度的时候要用上的。
        return a_, cache_

    def forward_L_layer(self, X_, parameters_):
        L_ = int(len(parameters_) / 2)
        caches = []
        a_ = X_
        for i in range(1, L_):
            w_ = parameters_['W' + str(i)]
            b_ = parameters_['b' + str(i)]
            a_pre_ = a_
            a_, cache_ = self.forward_one_layer(a_pre_, w_, b_, 'sigmoid')
            caches.append(cache_)

        w_last = parameters_['W' + str(L_)]
        b_last = parameters_['b' + str(L_)]

        if w_last.shape[0] == 1:
            a_last, cache_ = self.forward_one_layer(a_, w_last, b_last, 'sigmoid')
        else:
            a_last, cache_ = self.forward_one_layer(a_, w_last, b_last, 'softmax')

        caches.append(cache_)
        return a_last, caches

    def optimizer_gd(self, X_, y_, parameters_, num_epochs, learning_rate):
        costs = []
        for i in range(num_epochs):
            a_last, caches = self.forward_L_layer(X_, parameters_)
            grads = self.backward_L_layer(a_last, y_, caches)

            parameters_ = BPNNutil.update_parameters_with_gd(parameters_, grads, learning_rate)
            cost = self.compute_cost(a_last, y_)

            costs.append(cost)

        return parameters_, costs

    def test_accuracy(self, X_, y_):
        if not hasattr(self, "parameters_"):
            raise Exception('you must to fit before testing accuracy.')

        m = X_.shape[0]
        predict_ = self.predict(X_)

        if y_.ndim == 2:
            # 多分类问题
            y_true = np.argmax(y_, axis=1)
        else:
            # 二分类问题
            y_true = y_.flatten()

        accuracy = np.sum(predict_ == y_true) / m

        return accuracy


if __name__ == '__main__':
    penguins = load_penguins()
    penguins = penguins[['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
    penguins.loc[penguins['species'] == 'Adelie', 'species'] = 0
    penguins.loc[penguins['species'] == 'Gentoo', 'species'] = 1
    penguins.loc[penguins['species'] == 'Chinstrap', 'species'] = 2
    penguins.dropna(inplace=True)

    X = penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
    X = (X - np.mean(X, axis=0)) / np.var(X, axis=0)

    y = penguins[['species']]
    y = pd.get_dummies(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22, shuffle=True)

    bp = BpNN([10, 3], learning_rate=0.5)
    bp.fit(X_train.values, y_train.values, num_epochs=500)

    # 测试准确度
    accuracy = bp.test_accuracy(X_test.values, y_test.values)
    print(f"Accuracy: {accuracy}")

    BPNNutil.plot_costs([bp.costs], ['cost'])
