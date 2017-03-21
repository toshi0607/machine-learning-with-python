# -*- coding: utf-8 -*-
import scipy.io
import scipy.special
import numpy as np

digits_data_path = './data/ex3data1.mat'
weights_data_path = './data/ex3weights.mat'

def loadmat(file_path, *names):
    mat = scipy.io.loadmat(file_path)
    # type(mat)
    # -> <class 'dict'>
    return [mat[name] for name in names]

def sigmoid(z):
    return scipy.special.expit(z)

def softmax(x):
    max = np.max(x)
    return np.exp(x - max) / np.sum(np.exp(x - max))

def _softmax(x):
    return np.exp(x) / np.sum(np.exp(x))



X, y = loadmat(digits_data_path, 'X', 'y')
# X.shape
# -> (5000, 400)
# y.shape
# -> (5000, 1)

# Xの行数を後で調整するためにとっておく
m, _n = X.shape

# Xにbias項が含まれてない
# theta1はbias項分も考慮して25*401になっているので1を各行にマージする
X = np.concatenate((np.ones((m, 1)), X), axis=1)
# X.shape
# -> (5000, 401)

theta1, theta2 = loadmat(weights_data_path, 'Theta1', 'Theta2')
# theta1.shape
# -> (25, 401)
# theta2.shape
# -> (10, 26)

# 中間層
a = sigmoid(np.dot(X, theta1.T))
# 内積の場合転置不要 a = sigmoid(np.inner(X, theta1))
# a.shape
# -> (5000, 25)

# unitの数が25でbiasも考慮すると入力は26になる
# theta2はbias項分も考慮してるので、aの各行に1をマージ
aa = np.concatenate((np.ones((m, 1)), a), axis=1)
# -> (5000, 26)

# 出力層
a2 = softmax(np.dot(aa, theta2.T))
# 内積の場合転置不要 a2 = softmax(np.inner(aa, theta2))
# a2.shape
# -> (5000, 10)

# 答え合わせ
correct = 0
for i in range(0, m):
    # 10 labels, from 1 to 10 とのことでindexとラベルがずれてるので1足して調整する
    prediction = np.argmax(a2[i]) + 1
    correct += prediction == y[i]

print('Accuracy: %.2f%%' % (correct * 100.0 / m))
# -> Accuracy: 97.52%
